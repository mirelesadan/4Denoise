"""Peak matching and numerical strain fitting, independent of data objects."""
from dataclasses import dataclass
from numbers import Real
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture


def _validate_matching_parameters(radius, ambiguity_ratio):
    """Validate options once, even when every requested scan point is masked."""
    for label, value in (('match_radius', radius), ('ambiguity_ratio', ambiguity_ratio)):
        if value is None:
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f'{label} must be a finite real scalar or None.')
        if value <= 0 or (label == 'ambiguity_ratio' and value >= 1):
            raise ValueError(f'{label} must be positive' + (' and less than 1.' if label == 'ambiguity_ratio' else '.'))
    return radius, ambiguity_ratio


def _initial_transform(rotation, transform, peak_units):
    """Map a Cartesian CCW angle to the supplied (y, x) coordinate convention."""
    if rotation is not None and transform is not None:
        raise ValueError('Specify initial_rotation or initial_transform, not both.')
    if transform is not None:
        matrix = np.asarray(transform, dtype=float)
        if matrix.shape != (2, 2) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
            raise ValueError('initial_transform must be a finite nonsingular (2, 2) matrix.')
        return matrix
    if rotation is None:
        return np.eye(2)
    angle = float(rotation)
    if not np.isfinite(angle):
        raise ValueError('initial_rotation must be finite.')
    theta = np.deg2rad(angle)
    c, s = np.cos(theta), np.sin(theta)
    sign = 1 if peak_units == 'pixels' else -1
    return np.array([[c, -sign * s], [sign * s, c]])


def _automatic_match_radius(reference, measured):
    """Use less than half the typical nearest-peak separation as the gate."""
    for peaks in (reference, measured):
        if len(peaks) > 1:
            distances = cdist(peaks, peaks)
            np.fill_diagonal(distances, np.inf)
            distances[distances <= np.finfo(float).eps] = np.inf
            nearest = distances.min(axis=1)
            finite = nearest[np.isfinite(nearest)]
            if len(finite):
                return 0.45 * float(np.median(finite))
    return 1.0


def _gated_assignment(distances, radius):
    """Maximize accepted cardinality, then distance, with private unmatched slots."""
    if not distances.size:
        return np.array([], dtype=int), np.array([], dtype=int)
    if not np.isfinite(radius):
        return linear_sum_assignment(distances)
    nref, nmeas = distances.shape
    penalty = (nref + 1) * radius
    cost = np.full((nref, nmeas + nref), penalty)
    cost[:, :nmeas] = np.where(distances <= radius, distances, 2 * penalty)
    rows, columns = linear_sum_assignment(cost)
    valid = columns < nmeas
    return rows[valid], columns[valid]


def _match_peak_candidates(reference, measured, center, *, mode='auto',
                           radius=None, rotation=None, transform=None,
                           translation=None, fit_translation=False,
                           peak_units='pixels', ambiguity_ratio=None):
    """One-to-one local matching without centroids distorted by missing peaks.

    Initial transforms are used only to predict candidates; strain fitting
    still estimates the complete reference-to-measured transform. Distances
    and the optional gate use the supplied peak-coordinate units. A robust
    translation consensus is sought only when translation fitting is enabled.
    Explicit Hungarian mode has no gate unless radius is supplied. Ordered
    mode retains caller-provided pairing. Ambiguity confidence is a geometric
    distance margin, not a calibrated probability.
    """
    if mode not in ('auto', 'nearest', 'hungarian', 'ordered'):
        raise ValueError("match_peaks must be 'auto', 'nearest', 'hungarian', or 'ordered'.")
    radius, ambiguity_ratio = _validate_matching_parameters(radius, ambiguity_ratio)
    matrix = _initial_transform(rotation, transform, peak_units)
    predicted = (reference - center) @ matrix.T + center
    base_radius = _automatic_match_radius(predicted, measured)
    gate = float(radius) if radius is not None else (np.inf if mode == 'hungarian' else base_radius)
    shift = np.zeros(2) if translation is None else np.asarray(translation, dtype=float)
    if shift.shape != (2,) or not np.isfinite(shift).all():
        raise ValueError('initial_translation must contain finite (y, x) coordinates.')
    if translation is not None and not fit_translation:
        raise ValueError('initial_translation requires fit_translation=True.')
    if fit_translation and translation is None and len(reference) and len(measured):
        # Histogram voting proposes shifts without assuming equally populated
        # peak sets. Local assignments score each proposal, including zero.
        deltas = (measured[None, :, :] - predicted[:, None, :]).reshape(-1, 2)
        bins = np.rint(deltas / (base_radius * 0.5))
        unique, inverse, counts = np.unique(bins, axis=0, return_inverse=True, return_counts=True)
        proposals = [np.zeros(2)]
        for index in np.argsort(-counts, kind='stable')[:16]:
            proposals.append(np.median(deltas[inverse == index], axis=0))
        best = None
        for candidate in proposals:
            distances = cdist(predicted + candidate, measured)
            rows, columns = _gated_assignment(distances, float(radius) if radius is not None else base_radius)
            score = (len(rows), -float(np.sum(distances[rows, columns])), -float(np.linalg.norm(candidate)))
            if best is None or score > best:
                best, shift = score, candidate
    distances = cdist(predicted + shift, measured)
    if mode == 'ordered':
        rows = columns = np.arange(min(len(reference), len(measured)))
    else:
        rows, columns = _gated_assignment(distances, gate)
    candidates = np.sum(distances <= gate, axis=1)
    paired_distances = distances[rows, columns]
    confidence = np.ones(len(rows))
    for i, (r, c, distance) in enumerate(zip(rows, columns, paired_distances)):
        alternatives = np.r_[np.delete(distances[r], c), np.delete(distances[:, c], r)]
        second = float(alternatives.min()) if len(alternatives) else np.inf
        confidence[i] = 1.0 if not np.isfinite(second) else max(0.0, 1 - distance / max(second, np.finfo(float).eps))
        if second == distance == 0:
            confidence[i] = 0.0
    keep = np.ones(len(rows), dtype=bool) if ambiguity_ratio is None else confidence >= 1 - ambiguity_ratio
    info = {
        'match_radius': gate, 'initial_transform': matrix.copy(), 'initial_translation': shift.copy(),
        'candidate_counts': candidates, 'pair_distances': paired_distances[keep],
        'pair_confidence': confidence[keep], 'ambiguous_pair_count': int(np.count_nonzero(~keep)),
        'unmatched_reference_indices': np.setdiff1d(np.arange(len(reference)), rows[keep]),
        'unmatched_measured_indices': np.setdiff1d(np.arange(len(measured)), columns[keep]),
        'matched_reference_indices': rows[keep], 'matched_measured_indices': columns[keep],
        'status': 'matched' if np.any(keep) else ('ambiguous_matches' if len(rows) else 'no_candidates'),
    }
    return rows[keep], columns[keep], info

@dataclass(slots=True)
class _StrainFitEngine:
    """Validated fitting settings and robust residual/outlier calculations."""
    fit_translation: bool = False
    max_condition_number: float = 1e8
    min_peak_pairs: int = 2
    reject_peak_outliers: bool = False
    outlier_threshold: float = 3.5
    outlier_min_peak_pairs: int = 4
    outlier_min_peak_fraction: float | None = None
    outlier_max_iterations: int = 2
    outlier_space: str = 'radial_angular'
    outlier_method: str = 'auto_mixture'
    outlier_bic_delta: float = 0.0

    def _fit_peak_transform(self, reference, measured, strain_center, weights=None):
        reference = reference - strain_center
        measured = measured - strain_center
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            weights = np.where(np.isfinite(weights), weights, 0)
            weights = np.clip(weights, 0, None)
            if np.sum(weights) <= 0:
                weights = None
        if self.fit_translation:
            if weights is None:
                ref_origin = np.mean(reference, axis=0)
                measured_origin = np.mean(measured, axis=0)
            else:
                ref_origin = np.average(reference, axis=0, weights=weights)
                measured_origin = np.average(measured, axis=0, weights=weights)
            X = reference - ref_origin
            Y = measured - measured_origin
        else:
            ref_origin = np.zeros(2)
            measured_origin = np.zeros(2)
            X = reference
            Y = measured
        if weights is None:
            X_fit = X
            Y_fit = Y
        else:
            sqrt_weights = np.sqrt(weights)[:, np.newaxis]
            X_fit = X * sqrt_weights
            Y_fit = Y * sqrt_weights
        coeffs, _, rank, singular_values = np.linalg.lstsq(X_fit, Y_fit, rcond=None)
        if rank < 2 or singular_values[-1] <= 0 or singular_values[0] / singular_values[-1] > self.max_condition_number:
            raise np.linalg.LinAlgError('Matched reference peaks are collinear or too ill-conditioned for a reliable 2D strain fit.')
        transform_matrix = coeffs.T
        if not np.isfinite(transform_matrix).all() or abs(np.linalg.det(transform_matrix)) <= np.finfo(float).eps:
            raise np.linalg.LinAlgError('The fitted peak transform is singular.')
        translation = measured_origin - ref_origin @ coeffs
        return (transform_matrix, translation)

    def _robust_zscore(self, values):
        values = np.asarray(values, dtype=float)
        z = np.full(values.shape, np.inf, dtype=float)
        finite = np.isfinite(values)
        if not np.any(finite):
            return z
        finite_values = values[finite]
        median = np.median(finite_values)
        deviation = np.abs(finite_values - median)
        scale = 1.4826 * np.median(deviation)
        if scale <= np.finfo(float).eps:
            z[finite] = 0.0
            z[np.flatnonzero(finite)[deviation > np.finfo(float).eps]] = np.inf
            return z
        z[finite] = np.abs(values[finite] - median) / scale
        return z

    def _peak_residual_vectors(self, reference, measured, transform_matrix, translation, strain_center):
        reference_centered = reference - strain_center
        measured_centered = measured - strain_center
        predicted = reference_centered @ transform_matrix.T + translation
        return measured_centered - predicted

    def _peak_residual_distances(self, reference, measured, transform_matrix, translation, strain_center):
        residual = self._peak_residual_vectors(reference, measured, transform_matrix, translation, strain_center)
        return np.linalg.norm(residual, axis=1)

    def _fit_residual_quality(self, reference, measured, transform_matrix, translation, strain_center, weights):
        residual = self._peak_residual_vectors(reference, measured, transform_matrix, translation, strain_center)
        residual_sq = np.einsum('ij,ij->i', residual, residual)
        reference_centered = reference - strain_center
        reference_radius_sq = np.einsum('ij,ij->i', reference_centered, reference_centered)
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            weights = np.where(np.isfinite(weights), weights, 0.0)
            weights = np.clip(weights, 0.0, None)
            if np.sum(weights) <= 0:
                weights = None
        if weights is None:
            mean_residual_sq = np.mean(residual_sq)
            mean_reference_radius_sq = np.mean(reference_radius_sq)
        else:
            mean_residual_sq = np.average(residual_sq, weights=weights)
            mean_reference_radius_sq = np.average(reference_radius_sq, weights=weights)
        rmse = float(np.sqrt(mean_residual_sq))
        reference_rms_radius = float(np.sqrt(mean_reference_radius_sq))
        relative_rmse = rmse / reference_rms_radius if reference_rms_radius > np.finfo(float).eps else np.nan
        return (rmse, relative_rmse)

    def _peak_outlier_scores(self, reference, measured, transform_matrix, translation, strain_center):
        reference_centered = reference - strain_center
        residual = self._peak_residual_vectors(reference, measured, transform_matrix, translation, strain_center)
        if self.outlier_space == 'radial_angular':
            radii = np.linalg.norm(reference_centered, axis=1)
            safe_radii = np.where(radii > np.finfo(float).eps, radii, 1.0)
            radial_unit = np.divide(reference_centered, safe_radii[:, np.newaxis], out=np.zeros_like(reference_centered, dtype=float), where=safe_radii[:, np.newaxis] > 0)
            tangential_unit = np.column_stack((-radial_unit[:, 1], radial_unit[:, 0]))
            radial_residual = np.sum(residual * radial_unit, axis=1) / safe_radii
            angular_residual = np.sum(residual * tangential_unit, axis=1) / safe_radii
            radial_z = self._robust_zscore(radial_residual)
            angular_z = self._robust_zscore(angular_residual)
            score = np.hypot(radial_z, angular_z)
        elif self.outlier_space == 'cartesian':
            y_z = self._robust_zscore(residual[:, 0])
            x_z = self._robust_zscore(residual[:, 1])
            score = np.hypot(y_z, x_z)
        elif self.outlier_space == 'distance':
            score = self._robust_zscore(np.linalg.norm(residual, axis=1))
        else:
            raise ValueError("outlier_space must be 'radial_angular', 'cartesian', or 'distance'.")
        return score

    def _auto_mixture_outlier_mask(self, reference, measured, transform_matrix, translation, strain_center):
        residual_distances = self._peak_residual_distances(reference, measured, transform_matrix, translation, strain_center)
        probabilities = np.full(residual_distances.shape, np.nan, dtype=float)
        model_info = {'method': 'auto_mixture', 'accepted_two_component_model': False, 'bic_one_component': np.nan, 'bic_two_component': np.nan, 'low_residual_log_mean': np.nan, 'high_residual_log_mean': np.nan}
        reject = np.zeros(residual_distances.shape, dtype=bool)
        finite = np.isfinite(residual_distances)
        if np.count_nonzero(finite) < 2:
            return (reject, residual_distances, probabilities, model_info)
        finite_distances = residual_distances[finite]
        if np.ptp(finite_distances) <= np.finfo(float).eps:
            return (reject, residual_distances, probabilities, model_info)
        eps = max(np.finfo(float).eps, np.nanmedian(finite_distances) * 1e-12)
        log_distances = np.log10(np.maximum(finite_distances, eps))
        if np.unique(log_distances).size < 2:
            return (reject, residual_distances, probabilities, model_info)
        values = log_distances.reshape(-1, 1)
        low_start, high_start = np.percentile(log_distances, [25, 90])
        if high_start <= low_start:
            low_start, high_start = (np.min(log_distances), np.max(log_distances))
        one_component = GaussianMixture(n_components=1, covariance_type='full', reg_covar=1e-09, random_state=0, init_params='random', n_init=1)
        two_component = GaussianMixture(n_components=2, covariance_type='full', reg_covar=1e-09, random_state=0, init_params='random', means_init=np.array([[low_start], [high_start]]), weights_init=np.array([0.8, 0.2]), n_init=1)
        try:
            one_component.fit(values)
            two_component.fit(values)
        except ValueError:
            return (reject, residual_distances, probabilities, model_info)
        bic_one = one_component.bic(values)
        bic_two = two_component.bic(values)
        model_info['bic_one_component'] = float(bic_one)
        model_info['bic_two_component'] = float(bic_two)
        if not np.isfinite(bic_one) or not np.isfinite(bic_two):
            return (reject, residual_distances, probabilities, model_info)
        if bic_two + self.outlier_bic_delta >= bic_one:
            return (reject, residual_distances, probabilities, model_info)
        means = two_component.means_.reshape(-1)
        high_component = int(np.argmax(means))
        low_component = int(np.argmin(means))
        model_info['low_residual_log_mean'] = float(means[low_component])
        model_info['high_residual_log_mean'] = float(means[high_component])
        labels = two_component.predict(values)
        high_probabilities = two_component.predict_proba(values)[:, high_component]
        finite_indices = np.flatnonzero(finite)
        probabilities[finite_indices] = high_probabilities
        reject[finite_indices] = labels == high_component
        model_info['accepted_two_component_model'] = bool(np.any(reject))
        return (reject, residual_distances, probabilities, model_info)

    def _resolve_outlier_min_peak_pairs(self, reference_peak_count):
        if self.outlier_min_peak_fraction is None:
            return self.outlier_min_peak_pairs
        return max(2, int(np.ceil(float(reference_peak_count) * self.outlier_min_peak_fraction)))

    def _reject_peak_outliers(self, reference, measured, weights, strain_center, local_outlier_min_peak_pairs):
        keep = np.ones(reference.shape[0], dtype=bool)
        residual_distances = np.full(reference.shape[0], np.nan, dtype=float)
        probabilities = np.full(reference.shape[0], np.nan, dtype=float)
        model_info = {'method': self.outlier_method, 'accepted_two_component_model': False, 'bic_one_component': np.nan, 'bic_two_component': np.nan}
        if not self.reject_peak_outliers or reference.shape[0] < local_outlier_min_peak_pairs or self.outlier_max_iterations <= 0:
            return (keep, residual_distances, probabilities, model_info)
        min_remaining = max(self.min_peak_pairs, local_outlier_min_peak_pairs)
        for _ in range(self.outlier_max_iterations):
            if np.count_nonzero(keep) < min_remaining:
                break
            current_weights = weights[keep] if weights is not None else None
            transform_matrix, translation = self._fit_peak_transform(reference[keep], measured[keep], strain_center, weights=current_weights)
            kept_indices = np.flatnonzero(keep)
            if self.outlier_method == 'auto_mixture':
                local_reject, local_residual_distances, local_probabilities, local_model_info = self._auto_mixture_outlier_mask(reference[keep], measured[keep], transform_matrix, translation, strain_center)
                residual_distances[kept_indices] = local_residual_distances
                probabilities[kept_indices] = local_probabilities
                model_info = local_model_info
            else:
                scores = self._peak_outlier_scores(reference[keep], measured[keep], transform_matrix, translation, strain_center)
                local_residual_distances = self._peak_residual_distances(reference[keep], measured[keep], transform_matrix, translation, strain_center)
                residual_distances[kept_indices] = local_residual_distances
                local_reject = ~np.isfinite(scores) | (scores > self.outlier_threshold)
            if not np.any(local_reject):
                break
            available_rejections = np.count_nonzero(keep) - min_remaining
            if available_rejections <= 0:
                break
            reject_candidates = np.flatnonzero(local_reject)
            if reject_candidates.size > available_rejections:
                if self.outlier_method == 'auto_mixture':
                    candidate_scores = local_residual_distances[reject_candidates]
                else:
                    candidate_scores = scores[reject_candidates]
                candidate_scores = np.where(np.isfinite(candidate_scores), candidate_scores, np.inf)
                reject_candidates = reject_candidates[np.argsort(candidate_scores)[-available_rejections:]]
            candidate_keep = keep.copy()
            candidate_keep[np.flatnonzero(keep)[reject_candidates]] = False
            if np.array_equal(candidate_keep, keep):
                break
            keep = candidate_keep
        return (keep, residual_distances, probabilities, model_info)
