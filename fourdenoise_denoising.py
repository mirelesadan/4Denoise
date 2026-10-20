"""Private numerical denoising kernels, contracts, and array routing.

HyperData owns calibration, unfolding orchestration, and result metadata.
An injected object factory keeps this module independent of that facade.
"""

import numpy as np
from dataclasses import dataclass
from math import prod
from numbers import Integral
from scipy.ndimage import median_filter
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.special import erfc
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_tv_chambolle
import cv2
import inspect
import tensorly as tl
from tensorly.tt_matrix import tt_matrix_to_tensor
from tensorly.tt_tensor import tt_to_tensor
from tensorly.tenalg import multi_mode_dot
from tensorly.decomposition import constrained_parafac
from tensorly.decomposition import parafac2 as par2
from tensorly.decomposition import tensor_ring_als as tr_als
from tensorly.decomposition import tensor_ring_als_sampled as tr_als_sampled
from tensorly.decomposition import tensor_train_matrix as tt_mat
from tensorly.decomposition import tensor_train as tt
from tensorly.decomposition import non_negative_tucker_hals as nnth
from tensorly.decomposition import non_negative_tucker as nnt
from tensorly.decomposition import partial_tucker as partial_tuck
from tensorly.decomposition import tucker as tuck
from tensorly.decomposition import randomised_parafac as rand_parafac
from tensorly.decomposition import non_negative_parafac_hals as nn_parafac_hals
from tensorly.decomposition import non_negative_parafac as nn_parafac
from tensorly.decomposition import parafac as par
from tensorly.decomposition import robust_pca as robust_tensor_pca
from tensorly.decomposition import parafac_power_iteration as parafac_power_iter
from tensorly.decomposition import symmetric_parafac_power_iteration as sym_parafac_power_iter

class _DenoisingMethods:
    """Private collection of numerical denoising algorithms."""

    def __init__(self, hyperdata_type=None):
        self._hyperdata_type = hyperdata_type

    def _as_hyperdata(self, array):
        """Wrap decomposition payloads or route explicit unfolding."""
        if self._hyperdata_type is None:
            raise RuntimeError(
                "This operation requires a HyperData factory; use HyperData.denoise()."
            )
        return self._hyperdata_type(array)

    def _unfold_with_hyperdata(self, array, unfold_domain=None, unfold_method='row_major'):
        """
        Unfold an array through the HyperData API and return array + metadata.

        Decomposition methods operate on arrays, but unfolding belongs to
        HyperData. This helper keeps that ownership clear while letting the
        numerical routines work with plain ndarrays.
        """
        if unfold_domain is None:
            return array, None

        unfolded = self._as_hyperdata(array).unfold(
            domain=unfold_domain,
            method=unfold_method,
        )
        return unfolded.array, unfolded.unfold_metadata

    def _refold_with_hyperdata(self, array, unfold_metadata):
        """Restore an unfolded array through HyperData.unfold(undo=True)."""
        if unfold_metadata is None:
            return array

        return self._as_hyperdata(array).unfold(
            undo=True,
            metadata=unfold_metadata,
        ).array

    @staticmethod
    def _decomposition_preset(method_name, performance_preset):
        """
        Return practical decomposition defaults for denoising workflows.

        TensorLy's native defaults are careful general-purpose optimization
        settings. For denoising, a lower-iteration approximate factorization is
        often more useful than a slow, tightly converged decomposition.
        """
        preset_name = str(performance_preset).lower()
        presets = {
            'parafac': {
                'fast': {
                    'n_iter_max': 15,
                    'init': 'random',
                    'tol': 1e-4,
                },
                'balanced': {
                    'n_iter_max': 35,
                    'init': 'random',
                    'tol': 1e-5,
                },
                'accurate': {
                    'n_iter_max': 100,
                    'init': 'svd',
                    'tol': 1e-8,
                },
                'tensorly': {
                    'n_iter_max': 100,
                    'init': 'svd',
                    'tol': 1e-8,
                },
            },
            'parafac2': {
                'fast': {
                    'n_iter_max': 10,
                    'n_iter_parafac': 1,
                    'init': 'random',
                    'tol': 1e-4,
                    'linesearch': False,
                },
                'balanced': {
                    'n_iter_max': 25,
                    'n_iter_parafac': 2,
                    'init': 'random',
                    'tol': 1e-5,
                    'linesearch': True,
                },
                'accurate': {
                    'n_iter_max': 2000,
                    'n_iter_parafac': 5,
                    'init': 'random',
                    'tol': 1e-8,
                    'linesearch': True,
                },
                'tensorly': {
                    'n_iter_max': 2000,
                    'n_iter_parafac': 5,
                    'init': 'random',
                    'tol': 1e-8,
                    'linesearch': True,
                },
            },
        }

        if method_name not in presets:
            raise ValueError(f"No performance presets defined for {method_name!r}.")
        if preset_name not in presets[method_name]:
            valid = ', '.join(presets[method_name])
            raise ValueError(
                f"performance_preset must be one of: {valid}, or None."
            )
        return presets[method_name][preset_name]

    @staticmethod
    def _split_cp_result(result, return_errors=False):
        """Return ``(cp_tensor, errors)`` from TensorLy CP-style outputs."""
        if return_errors:
            cp_tensor, errors = result
        else:
            cp_tensor = result
            errors = None
        return cp_tensor, errors

    @staticmethod
    def _looks_like_error_sequence(values):
        """Return True when ``values`` looks like TensorLy reconstruction errors."""
        try:
            values = list(values)
        except TypeError:
            return False

        return all(np.isscalar(value) or np.asarray(value).ndim == 0 for value in values)

    @classmethod
    def _split_tucker_result(cls, result, return_errors=False):
        """Return ``(tucker_tensor, errors)`` from TensorLy Tucker-style outputs."""
        if (
            return_errors
            and isinstance(result, tuple)
            and len(result) == 2
            and cls._looks_like_error_sequence(result[1])
        ):
            return result[0], result[1]

        return result, None

    # =============================================================================
    # Spatial Filters
    # =============================================================================

    @staticmethod
    def _validate_filter_array(target_data, method, allowed_ndim):
        """Require finite, real-valued image data for spatial filters."""
        image = np.asarray(target_data)
        if image.ndim not in allowed_ndim:
            dimensions = ', '.join(str(ndim) for ndim in allowed_ndim)
            raise ValueError(f"{method} requires a {dimensions}D array.")
        if (
            image.size == 0
            or not np.issubdtype(image.dtype, np.number)
            or np.iscomplexobj(image)
        ):
            raise ValueError(f"{method} requires a nonempty, real-valued numeric array.")
        if not np.all(np.isfinite(image)):
            raise ValueError(f"{method} requires finite input values.")
        return image

    def gaussian(self, target_data, kernel_size=3, sigma=1):
        """Smooth a 2D image with a Gaussian kernel, preserving count scale.

        Parameters
        ----------
        target_data : ndarray
            Real-valued 2D image. Integer input is promoted to float so
            interpolated intensities are not rounded back to integers.
        kernel_size : odd int or (odd y_size, odd x_size), optional
            Spatial extent of the filter; ``0`` is not accepted.
        sigma : float, optional
            Nonnegative Gaussian standard deviation in pixels. Zero asks
            OpenCV to infer it from the kernel size.

        Returns
        -------
        ndarray
            Filtered image in the same intensity units as the input.
        """
        image = self._validate_filter_array(target_data, 'gaussian', (2,))
        if (
            isinstance(kernel_size, (int, np.integer))
            and not isinstance(kernel_size, (bool, np.bool_))
        ):
            kernel_size = (kernel_size, kernel_size)
        if not isinstance(kernel_size, (tuple, list)) or len(kernel_size) != 2 or any(
            isinstance(size, (bool, np.bool_))
            or not isinstance(size, (int, np.integer))
            or size <= 0 or size % 2 == 0
            for size in kernel_size
        ):
            raise ValueError(
                "kernel_size must be an odd positive integer or a pair of them."
            )
        try:
            sigma = float(sigma)
        except (TypeError, ValueError) as exc:
            raise ValueError("sigma must be a nonnegative finite number.") from exc
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError("sigma must be a nonnegative finite number.")

        dtype = (
            np.float64 if np.issubdtype(image.dtype, np.integer)
            else np.result_type(image.dtype, np.float32)
        )
        working = np.ascontiguousarray(image, dtype=dtype)
        ky, kx = kernel_size
        return cv2.GaussianBlur(working, (int(kx), int(ky)), sigmaX=sigma, sigmaY=sigma)


    # Tested successfully
    def median(self, target_data, window_size=5, mode='reflect', cval=0.0,
               origin=0, axes=None):
        """Apply an axis-aware median filter to an image or volume.

        The median filter replaces each pixel value with the median value of its neighborhood,
        effectively removing salt-and-pepper noise while preserving edges.

        Parameters
        ----------
        target_data : ndarray
            Image or volume to be denoised.
        window_size : int or tuple of int, optional
            Window width for every filtered axis, or one width per axis.
        mode : str, optional
            Boundary-extension mode passed to ``scipy.ndimage.median_filter``.
        cval : scalar, optional
            Constant fill value used when ``mode='constant'``.
        origin : int or sequence, optional
            Placement of the filter relative to each filtered pixel.
        axes : tuple of int or None, optional
            Axes over which to apply the median. ``None`` filters every axis,
            including the stack axis of unfolded 3D data. For 4D data,
            ``HyperData.denoise`` supplies the selected real- or
            reciprocal-space axes unless ``domain=None`` is requested.

        Returns
        -------
        ndarray
            Filtered data with the input shape and dtype.

        Notes
        -----
        The median filter is particularly effective for removing salt-and-pepper noise.
        It may be less effective for Gaussian noise.
        """
        return median_filter(
            target_data,
            size=window_size,
            mode=mode,
            cval=cval,
            origin=origin,
            axes=axes,
        )


    # Tested successfully on (4/30/2024) for real- and reciprocal-space denoising
    def bilateral(self, target_data, d=9, sigma_color=75, sigma_space=75):
        """Smooth a 2D image while retaining intensity-dependent edges.

        Parameters
        ----------
        target_data : ndarray
            Real-valued 2D image.
        d : int, optional
            Pixel-neighborhood diameter. Zero lets OpenCV infer it from
            ``sigma_space``.
        sigma_color : float, optional
            Positive intensity-distance scale in the input's units.
        sigma_space : float, optional
            Positive spatial-distance scale in pixels.

        Returns
        -------
        ndarray
            Float image in the original intensity units. OpenCV filters in
            float32; subtracting the minimum first preserves small differences
            on top of a large baseline.
        """
        image = self._validate_filter_array(target_data, 'bilateral', (2,))
        if (
            isinstance(d, (bool, np.bool_))
            or not isinstance(d, (int, np.integer))
            or d < 0
        ):
            raise ValueError("d must be a nonnegative integer.")
        try:
            sigma_color = float(sigma_color)
            sigma_space = float(sigma_space)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "sigma_color and sigma_space must be positive finite numbers."
            ) from exc
        if (
            not np.isfinite(sigma_color) or sigma_color <= 0
            or not np.isfinite(sigma_space) or sigma_space <= 0
        ):
            raise ValueError("sigma_color and sigma_space must be positive finite numbers.")

        working = np.asarray(image, dtype=np.float64)
        baseline = float(working.min())
        shifted = np.ascontiguousarray(working - baseline, dtype=np.float32)
        if not np.all(np.isfinite(shifted)):
            raise ValueError("bilateral input range exceeds float32 capacity.")
        filtered = cv2.bilateralFilter(shifted, int(d), sigma_color, sigma_space)
        dtype = (
            np.float64 if np.issubdtype(image.dtype, np.integer)
            else np.result_type(image.dtype, np.float32)
        )
        return filtered.astype(dtype) + np.asarray(baseline, dtype=dtype)


    # Tested successfully on (4/30/2024) for real-space denoising
    def non_local_means(self, target_data, h=1.15, patch_size=5, patch_distance=6, fast_mode=True):
        """Apply non-local means to a 2D image or 3D volume.

        ``h`` multiplies the estimated noise standard deviation; it is not
        an absolute intensity threshold. Intensities stay in input units,
        including for integer-valued diffraction data.

        Parameters
        ----------
        target_data : ndarray
            Finite, real-valued 2D image or 3D volume.
        h : float, optional
            Nonnegative multiplier for the estimated noise level. Zero
            returns an independent floating-point copy.
        patch_size : int, optional
            Positive patch width in pixels.
        patch_distance : int, optional
            Nonnegative search distance in pixels.
        fast_mode : bool, optional
            Use scikit-image's faster algorithm (at higher memory cost).
        """
        image = self._validate_filter_array(target_data, 'non_local_means', (2, 3))
        try:
            h = float(h)
        except (TypeError, ValueError) as exc:
            raise ValueError("h must be a nonnegative finite number.") from exc
        if not np.isfinite(h) or h < 0:
            raise ValueError("h must be a nonnegative finite number.")
        for name, value, minimum in (
            ('patch_size', patch_size, 1),
            ('patch_distance', patch_distance, 0),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}.")

        dtype = (
            np.float64 if np.issubdtype(image.dtype, np.integer)
            else np.result_type(image.dtype, np.float32)
        )
        working = np.asarray(image, dtype=dtype)
        if h == 0:
            return working.copy()
        sigma_est = float(np.mean(estimate_sigma(working, channel_axis=None)))
        if not np.isfinite(sigma_est) or sigma_est < 0:
            raise ValueError("non_local_means could not estimate a finite noise level.")
        if sigma_est == 0:
            return working.copy()
        return denoise_nl_means(
            working, h=h * sigma_est, patch_size=int(patch_size),
            patch_distance=int(patch_distance), fast_mode=fast_mode,
            preserve_range=True, channel_axis=None,
        )


    @staticmethod
    def _estimate_noise_sigma(image):
        """Estimate scalar Gaussian noise level for a 2D image or 3D volume."""
        try:
            sigma_est = estimate_sigma(image, channel_axis=None)
        except TypeError:
            sigma_est = estimate_sigma(image, multichannel=False)
        return float(np.mean(sigma_est))

    @staticmethod
    def _resolve_bm_stage(module, stage, enum_name):
        """Resolve user-friendly BM3D/BM4D stage names to package enums."""
        if stage is None:
            stage = 'all'
        if not isinstance(stage, str):
            return stage

        normalized = stage.strip().lower().replace('-', '_')
        stage_enum = getattr(module, enum_name, None)
        if stage_enum is None:
            return stage

        aliases = {
            'all': 'ALL_STAGES',
            'both': 'ALL_STAGES',
            'all_stages': 'ALL_STAGES',
            'full': 'ALL_STAGES',
            'hard': 'HARD_THRESHOLDING',
            'ht': 'HARD_THRESHOLDING',
            'hard_thresholding': 'HARD_THRESHOLDING',
            'wiener': 'WIENER_FILTERING',
            'wf': 'WIENER_FILTERING',
            'wiener_filtering': 'WIENER_FILTERING',
        }
        if normalized not in aliases:
            valid = ', '.join(sorted(aliases))
            raise ValueError(f"stage must be one of: {valid}, or a {enum_name} value.")
        return getattr(stage_enum, aliases[normalized])

    def bm3d(self, target_data, sigma_psd=None, stage='all', profile='np',
             clip_output=False, **kwargs):
        """
        Denoise 2D images with BM3D block matching and 3D filtering.

        BM3D normally takes a 2D image as input. The "3D" in BM3D refers to
        the internal grouping/filtering of similar 2D patches. For a 3D stack
        with shape ``(N, Y, X)``, this wrapper applies BM3D independently to
        each 2D image along the first axis.

        Parameters
        ----------
        target_data : ndarray
            Two-dimensional image or three-dimensional image stack to denoise.
        sigma_psd : float, ndarray, or sequence, optional
            Noise standard deviation or noise power spectral density passed to
            ``bm3d.bm3d``. If omitted, a scalar Gaussian noise level is
            estimated from each image. For 3D stacks, a 1D sequence with
            length ``N`` is interpreted as one scalar noise level per image.
        stage : {'all', 'hard', 'wiener'} or BM3DStages value, optional
            BM3D stage to run.
        profile : str or BM3DProfile, optional
            BM3D profile passed through to the external package.
        clip_output : bool, optional
            If True, clip each denoised image to the min/max range of the
            corresponding input image.
        **kwargs
            Additional keyword arguments passed to ``bm3d.bm3d``.

        Returns
        -------
        ndarray
            Denoised image or image stack with the same shape as the input.
        """
        try:
            import bm3d as bm3d_module
        except ImportError as exc:
            raise ImportError(
                "The BM3D denoising method requires the optional 'bm3d' "
                "package. Install it with `pip install bm3d` in the active "
                "environment, then rerun denoise(method='bm3d', ...)."
            ) from exc

        data = np.asarray(target_data)
        if data.ndim not in (2, 3):
            raise ValueError(
                "bm3d expects a 2D image or a 3D stack with shape "
                f"(N, Y, X); got shape {data.shape}."
            )
        if np.iscomplexobj(data):
            raise TypeError("bm3d requires real-valued data.")
        if not np.all(np.isfinite(data)):
            raise ValueError("bm3d requires finite input values.")
        if not isinstance(clip_output, (bool, np.bool_)):
            raise ValueError("clip_output must be a boolean.")

        stage_arg = self._resolve_bm_stage(
            bm3d_module,
            stage,
            'BM3DStages',
        )

        def _denoise_image(image, image_sigma):
            image = np.asarray(image)
            if image_sigma is None:
                image_sigma = self._estimate_noise_sigma(image)
            result = bm3d_module.bm3d(
                image.astype(np.result_type(image.dtype, np.float32), copy=False),
                sigma_psd=image_sigma,
                profile=profile,
                stage_arg=stage_arg,
                **kwargs,
            )
            if isinstance(result, tuple):
                result = result[0]
            result = np.asarray(result)
            if result.shape != image.shape:
                raise ValueError(
                    "bm3d changed image shape from "
                    f"{image.shape} to {result.shape}; 4Denoise expects "
                    "shape-preserving denoising methods."
                )
            if clip_output:
                result = np.clip(result, np.min(image), np.max(image))
            return result

        if data.ndim == 2:
            return _denoise_image(data, sigma_psd)

        per_image_sigma = None
        shared_sigma = sigma_psd
        if sigma_psd is not None:
            sigma_array = np.asarray(sigma_psd)
            if sigma_array.ndim == 1 and sigma_array.size == data.shape[0]:
                per_image_sigma = sigma_array
                shared_sigma = None

        first_result = _denoise_image(
            data[0],
            per_image_sigma[0] if per_image_sigma is not None else shared_sigma,
        )
        denoised = np.empty(
            (data.shape[0],) + first_result.shape,
            dtype=first_result.dtype,
        )
        denoised[0] = first_result
        for idx in range(1, data.shape[0]):
            image_sigma = (
                per_image_sigma[idx]
                if per_image_sigma is not None
                else shared_sigma
            )
            denoised[idx] = _denoise_image(data[idx], image_sigma)

        return denoised

    def bm4d(self, target_data, sigma_psd=None, stage='all', profile='np',
             clip_output=False, **kwargs):
        """
        Denoise a 3D volume with BM4D block matching and 4D filtering.

        BM4D is the volumetric counterpart to BM3D and expects a single 3D
        volume. To apply BM4D to 4D-STEM data, unfold the 4D tensor to a 3D
        representation with ``HyperData.denoise(..., unfold_domain=...)``.

        Parameters
        ----------
        target_data : ndarray
            Three-dimensional volume to denoise.
        sigma_psd : float or ndarray, optional
            Noise standard deviation or noise power spectral density passed to
            ``bm4d.bm4d``. If omitted, a scalar Gaussian noise level is
            estimated from the volume.
        stage : {'all', 'hard', 'wiener'} or BM4DStages value, optional
            BM4D stage to run.
        profile : str or BM4DProfile, optional
            BM4D profile passed through to the external package.
        clip_output : bool, optional
            If True, clip the denoised volume to the min/max range of the
            input volume.
        **kwargs
            Additional keyword arguments passed to ``bm4d.bm4d``.

        Returns
        -------
        ndarray
            Denoised volume with the same shape as the input.
        """
        try:
            import bm4d as bm4d_module
        except ImportError as exc:
            raise ImportError(
                "The BM4D denoising method requires the optional 'bm4d' "
                "package. Install it with `pip install bm4d` in the active "
                "environment, then rerun denoise(method='bm4d', ...)."
            ) from exc

        data = np.asarray(target_data)
        if data.ndim != 3:
            raise ValueError(
                "bm4d expects a single 3D volume. For 4D-STEM data, use "
                "unfold_domain to convert the tensor to 3D before denoising. "
                f"Got shape {data.shape}."
            )
        if np.iscomplexobj(data):
            raise TypeError("bm4d requires real-valued data.")
        if not np.all(np.isfinite(data)):
            raise ValueError("bm4d requires finite input values.")
        if not isinstance(clip_output, (bool, np.bool_)):
            raise ValueError("clip_output must be a boolean.")

        stage_arg = self._resolve_bm_stage(
            bm4d_module,
            stage,
            'BM4DStages',
        )
        volume_sigma = (
            self._estimate_noise_sigma(data)
            if sigma_psd is None
            else sigma_psd
        )

        result = bm4d_module.bm4d(
            data.astype(np.result_type(data.dtype, np.float32), copy=False),
            volume_sigma,
            profile=profile,
            stage_arg=stage_arg,
            **kwargs,
        )
        if isinstance(result, tuple):
            result = result[0]
        result = np.asarray(result)
        if result.shape != data.shape:
            raise ValueError(
                "bm4d changed volume shape from "
                f"{data.shape} to {result.shape}; 4Denoise expects "
                "shape-preserving denoising methods."
            )
        if clip_output:
            result = np.clip(result, np.min(data), np.max(data))

        return result


    def anisotropic_diffusion(self, target_data, niter=10, kappa=30, gamma=0.2, option=2):
        """Apply edge-preserving Perona-Malik diffusion to a real N-D array.

        Each neighboring pixel pair exchanges equal and opposite flux, so
        constant images and total intensity are preserved with no-flux edges.
        The explicit update is split into stable substeps when ``gamma`` is
        larger than ``1 / (2 * target_data.ndim)``.

        Parameters
        ----------
        target_data : ndarray
            Real-valued image or volume with at least two dimensions.
        niter : int, optional
            Number of diffusion iterations; zero returns an independent copy.
        kappa : float, optional
            Positive contrast scale for edge inhibition.
        gamma : float, optional
            Nonnegative total step size per iteration.
        option : {1, 2}, optional
            Exponential or reciprocal conduction function, respectively.

        Returns
        -------
        ndarray
            Diffused data. Integer inputs are promoted to floating point;
            values are not clipped to an arbitrary minimum.
        """
        data = np.asarray(target_data)
        if data.ndim < 2:
            raise ValueError("anisotropic_diffusion expects at least 2D data.")
        if not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data):
            raise TypeError("anisotropic_diffusion requires real numeric data.")
        if not np.all(np.isfinite(data)):
            raise ValueError("anisotropic_diffusion requires finite input values.")
        if isinstance(niter, (bool, np.bool_)) or not isinstance(niter, Integral) or niter < 0:
            raise ValueError("niter must be a nonnegative integer.")
        if isinstance(kappa, (bool, np.bool_)) or not np.isscalar(kappa):
            raise ValueError("kappa must be positive and finite.")
        if isinstance(gamma, (bool, np.bool_)) or not np.isscalar(gamma):
            raise ValueError("gamma must be nonnegative and finite.")
        try:
            kappa = float(kappa)
            gamma = float(gamma)
        except (TypeError, ValueError) as exc:
            raise ValueError("kappa and gamma must be finite numbers.") from exc
        if not np.isfinite(kappa) or kappa <= 0:
            raise ValueError("kappa must be positive and finite.")
        if not np.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be nonnegative and finite.")
        if isinstance(option, (bool, np.bool_)) or option not in (1, 2):
            raise ValueError("option must be 1 or 2.")
        if niter == 0 or gamma == 0:
            return np.array(data, copy=True)

        result = data.astype(np.result_type(data.dtype, np.float32), copy=True)
        substeps = max(1, int(np.ceil(gamma * 2 * data.ndim)))
        step = gamma / substeps

        for _ in range(niter):
            for _ in range(substeps):
                update = np.zeros_like(result)
                for axis in range(data.ndim):
                    difference = np.diff(result, axis=axis)
                    normalized = difference / kappa
                    if option == 1:
                        conduction = np.exp(-(normalized ** 2))
                    else:
                        conduction = 1 / (1 + normalized ** 2)
                    flux = conduction * difference
                    lower = (slice(None),) * axis + (slice(None, -1),)
                    upper = (slice(None),) * axis + (slice(1, None),)
                    trailing = (slice(None),) * (data.ndim - axis - 1)
                    update[lower + trailing] += flux
                    update[upper + trailing] -= flux
                result += step * update

        return result


    # Tested successfully on (4/30/2024) for real-space denoising
    def total_variation(self, target_data, weight=30, eps=0.0001, max_num_iter=100):
        """Apply Chambolle total-variation denoising in native intensity units.

        A 3D unfolded stack is treated as a volume: its first axis is filtered
        together with the two image axes. Integer data are converted to float
        without scikit-image's automatic 0-to-1 intensity rescaling.

        Parameters
        ----------
        target_data : ndarray
            Finite real-valued 2D, 3D, or 4D data.
        weight : float, optional
            Nonnegative regularization strength in the input's intensity units.
            Zero returns an independent floating-point copy.
        eps : float, optional
            Positive relative convergence tolerance.
        max_num_iter : int, optional
            Positive maximum number of iterations.

        Returns
        -------
        ndarray
            Denoised floating-point array in the input's intensity units.
        """
        data = np.asarray(target_data)
        if data.ndim not in (2, 3, 4):
            raise ValueError("total_variation expects a 2D, 3D, or 4D array.")
        if data.size == 0 or not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data):
            raise TypeError("total_variation requires real numeric data.")
        if not np.all(np.isfinite(data)):
            raise ValueError("total_variation requires finite input values.")
        if isinstance(weight, (bool, np.bool_)) or isinstance(eps, (bool, np.bool_)):
            raise ValueError("weight and eps must be finite numbers.")
        try:
            weight = float(weight)
            eps = float(eps)
        except (TypeError, ValueError) as exc:
            raise ValueError("weight and eps must be finite numbers.") from exc
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("weight must be nonnegative and finite.")
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be positive and finite.")
        if (
            isinstance(max_num_iter, (bool, np.bool_))
            or not isinstance(max_num_iter, (Integral, np.integer))
            or max_num_iter < 1
        ):
            raise ValueError("max_num_iter must be a positive integer.")

        working = data.astype(np.result_type(data.dtype, np.float32), copy=False)
        if weight == 0:
            return working.copy()
        return denoise_tv_chambolle(
            working, weight=weight, eps=eps,
            max_num_iter=max_num_iter, channel_axis=None,
        )

    def adaptive_median_filter(self, target_data, s=3, sMax=7):
        """
        Apply an adaptive median filter to reduce noise while preserving edges.

        Parameters
        ----------
        target_data : numpy.ndarray
            The single-channel (grayscale) image to denoise.
        s : int, optional
            Odd positive initial window size for the median filter.
        sMax : int, optional
            Odd positive maximum window size, at least ``s``.

        Returns
        -------
        filtered_target_data : numpy.ndarray
            The denoised image.

        Notes
        -----
        This filter grows a reflected-boundary window until the median is
        distinguishable from its extrema. If this never happens, it returns
        the largest window's median rather than retaining an impulse.
        """

        image = self._validate_filter_array(target_data, 'adaptive_median_filter', (2,))
        for name, size in (('s', s), ('sMax', sMax)):
            if (
                isinstance(size, (bool, np.bool_))
                or not isinstance(size, (int, np.integer))
                or size < 1 or size % 2 == 0
            ):
                raise ValueError(f"{name} must be an odd positive integer.")
        if sMax < s:
            raise ValueError("sMax must be at least s.")

        padded_target_data = np.pad(image, sMax // 2, mode='reflect')
        H, W = image.shape
        filtered_target_data = np.empty_like(image)

        for i in range(H):
            for j in range(W):
                value = self._process_pixel(padded_target_data, i + sMax//2, j + sMax//2, s, sMax)
                filtered_target_data[i, j] = value

        return filtered_target_data

    # Private method (helper function for adaptive median filter)
    def _process_pixel(self, padded_target_data, y, x, s, sMax):
        """
        Private method (helper function for adaptive median filter): process
        each pixel by adapting the window size and applying levels A and B checks.

        Parameters
        ----------
        padded_target_data : numpy.ndarray
            The padded input image.
        y : int
            y-coordinate in the padded image.
        x : int
            x-coordinate in the padded image.
        s : int
            Current window size.
        sMax : int
            Maximum window size.

        Returns
        -------
        value : float
            The new pixel value after filtering.

        Notes
        -----
        This method increases the window size until the conditions of Levels A
        or B are met or the maximum window size is reached.
        """

        while True:
            window = padded_target_data[y-s//2:y+s//2+1, x-s//2:x+s//2+1]
            Z_min, Z_med, Z_max = np.min(window), np.median(window), np.max(window)

            if Z_min < Z_med < Z_max:
                return self._level_b(window, Z_min, Z_med, Z_max)

            s += 2
            if s > sMax:
                return Z_med

    # Private method (helper function for adaptive median filter)
    def _level_b(self, window, Z_min, Z_med, Z_max):
        """
        Level B processing to determine the output pixel value based on window statistics.

        Parameters
        ----------
        window : numpy.ndarray
            The current window of pixel values.
        Z_min : float
            Minimum value in the window.
        Z_med : float
            Median value in the window.
        Z_max : float
            Maximum value in the window.

        Returns
        -------
        value : float
            Either the original or the median pixel value, based on conditions.

        Notes
        -----
        This method checks if the central pixel in the window is not an extreme
        value. If true, it returns the original central pixel value; otherwise,
        it returns the median value.
        """

        Z_xy = window[window.shape[0]//2, window.shape[1]//2]

        if Z_min < Z_xy < Z_max:
            return Z_xy
        else:
            return Z_med

    # #
    # def wiener(self, other):



    #     return clip_values(filtered_data)

    # =============================================================================
    # Unsupervised Learning
    # =============================================================================

    # #
    # def PCA(self, other):



    #     return clip_values(filtered_data)

    # #
    # def kernelPCA(self, other):



    #     return clip_values(filtered_data)

    # #
    # def fastICA(self, other):



    #     return clip_values(filtered_data)

    def nmf(self, X, n_components=None, init='random', update_H=True, solver='cd',
            beta_loss='frobenius', tol=0.0001, max_iter=200, alpha_W=0.0, alpha_H='same',
            l1_ratio=0.0, random_state=None, verbose=0, shuffle=False, return_decomposition=False,
            plot_eigenvalues=False, unfold_domain=None, unfold_method='row_major'):
        """
        Non-negative Matrix Factorization (NMF) using scikit-learn's `non_negative_factorization`.

        Computes a decomposition of matrix X into two non-negative matrices W and H such that:
        X ≈ W @ H

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data matrix to decompose.

        n_components : int, optional, default=None
            Number of components to use for the factorization. If not set, all features are kept.

        init : {'random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom'}, optional, default='random'
            Initialization method to use:
            - 'random': non-negative random matrices
            - 'nndsvd': Nonnegative Double Singular Value Decomposition
            - 'nndsvda': NNDSVD with zeros filled with the average of X
            - 'nndsvdar': NNDSVD with zeros filled with small random values
            - 'custom': Custom matrices W and H must be provided if update_H is True

        update_H : bool, default=True
            Whether to update the matrix H in the factorization. If False, H is fixed.

        solver : {'cd', 'mu'}, default='cd'
            The numerical solver to use:
            - 'cd': Coordinate Descent
            - 'mu': Multiplicative Update

        beta_loss : float or {'frobenius', 'kullback-leibler', 'itakura-saito'}, default='frobenius'
            Beta divergence to be minimized, measuring the distance between X and the dot product WH.

        tol : float, default=0.0001
            Tolerance of the stopping condition.

        max_iter : int, default=200
            Maximum number of iterations to run the algorithm.

        alpha_W : float, default=0.0
            Regularization parameter for W. Set to 0 for no regularization.

        alpha_H : float or "same", default="same"
            Regularization parameter for H. Set to 0 for no regularization. If "same", it takes the same value as alpha_W.

        l1_ratio : float, default=0.0
            The regularization mixing parameter, with 0 <= l1_ratio <= 1. Determines the balance between L1 and L2 penalties.

        random_state : int, RandomState instance or None, default=None
            Used for NMF initialization and in the Coordinate Descent solver. Pass an int for reproducible results.

        verbose : int, default=0
            The verbosity level.

        shuffle : bool, default=False
            If True, randomize the order of coordinates in the Coordinate Descent solver.

        return_decomposition : bool, default=False
            If True, return the decomposition W and H.

        plot_scree : bool, default=False
            If True, plot a scree plot showing the variance explained by each component.

        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            The resulting matrix of the factorization.

        H : ndarray of shape (n_components, n_features)
            The resulting matrix of the factorization.

        n_iter : int
            The number of iterations run.

        reconstruction : ndarray of shape (n_samples, n_features)
            The reconstructed matrix X from W and H, returned if return_reconstruction is True.
        """
        from sklearn.decomposition import non_negative_factorization

        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        if unfold_domain is not None:
            X, unfold_metadata = self._unfold_with_hyperdata(
                X, unfold_domain, unfold_method
            )

        # Perform NMF
        W, H, n_iter = non_negative_factorization(X, n_components=n_components, init=init,
                                                  update_H=update_H, solver=solver,
                                                  beta_loss=beta_loss, tol=tol, max_iter=max_iter,
                                                  alpha_W=alpha_W, alpha_H=alpha_H, l1_ratio=l1_ratio,
                                                  random_state=random_state, verbose=verbose, shuffle=shuffle)

        # Calculate the sum of squared elements in each component (row of H)
        component_significance = np.sum(H ** 2, axis=1)

        # Sort components by significance
        sorted_indices = np.argsort(-component_significance)
        H = H[sorted_indices]
        W = W[:, sorted_indices]
        component_significance = component_significance[sorted_indices]

        # Plotting option for log10 of eigenvalues vs. number of components
        if plot_eigenvalues:
            plt.figure(figsize=(8, 5))
            plt.plot(np.arange(1, len(component_significance) + 1), np.log10(component_significance), 'o-')
            plt.title('Log10 of Eigenvalues vs. Number of Components')
            plt.xlabel('Component Number')
            plt.ylabel('Log10(Eigenvalue)')
            plt.grid()
            plt.show()

        # Reconstruct decomposition
        reconstruction = W @ H

        # Re-fold the tensor if it was unfolded
        if unfold_domain is not None:
            reconstruction = self._refold_with_hyperdata(
                reconstruction, unfold_metadata
            )

        if return_decomposition:
            return [reconstruction, [W, H, n_iter]]

        else:
            return reconstruction

    ##########################
    # TensorLy Decomposition #
    ##########################


    # Good results with 'reciprocal' unfolding and rank=50
    def parafac(self, tensor, rank, n_iter_max=100, init='svd', svd='truncated_svd',
                normalize_factors=False, orthogonalise=False, tol=1e-08, random_state=None,
                verbose=0, return_errors=False, sparsity=None, l2_reg=0, mask=None,
                cvg_criterion='abs_rec_error', fixed_modes=None, svd_mask_repeats=5,
                linesearch=False, callback=None, implementation='tensorly',
                return_decomposition=False, unfold_domain=None,
                unfold_method='row_major', performance_preset=None,
                working_dtype=None):
        """CANDECOMP/PARAFAC decomposition via alternating least squares (ALS)

        Computes a rank-rank decomposition of tensor such that:
        tensor = [|weights; factors[0], ..., factors[-1] |]

        Parameters
        ----------
        tensor : ndarray
            The input tensor to decompose.

        rank : int
            Number of components.

        n_iter_max : int, optional, default is 100
            Maximum number of iterations.

        init : {'svd', 'random', CPTensor}, optional, default is 'svd'
            Type of factor matrix initialization. If a CPTensor is passed, this
            is directly used for initialization.

        svd : str, optional, default is 'truncated_svd'
            Function to use to compute the SVD. Acceptable values are in tensorly.SVD_FUNS.

        normalize_factors : bool, optional, default is False
            If True, aggregate the weights of each factor in a 1D-tensor of shape
            (rank, ), which will contain the norms of the factors.

        orthogonalise : bool, optional, default is False
            If True, enforce orthogonality on the factors.

        tol : float, optional, default is 1e-08
            Relative reconstruction error tolerance. The algorithm is considered
            to have found the global minimum when the reconstruction error is
            less than tol.

        random_state : {None, int, np.random.RandomState}, optional
            Random seed or state to initialize the random number generator.

        verbose : int, optional, default is 0
            Level of verbosity.

        return_errors : bool, optional, default is False
            Activate return of iteration errors.

        sparsity : float or int, optional, default is None
            If sparsity is not None, we approximate tensor as a sum of low_rank_component
            and sparse_component, where low_rank_component = cp_to_tensor((weights, factors)).
            sparsity denotes desired fraction or number of non-zero elements in
            the sparse_component of the tensor.

        l2_reg : float, optional, default is 0
            L2 regularization parameter.

        mask : ndarray, optional
            Array of booleans with the same shape as tensor. Should be 0 where
            the values are missing and 1 everywhere else.

        cvg_criterion : {'abs_rec_error', 'rec_error'}, optional, default is 'abs_rec_error'
            Stopping criterion for ALS, works if tol is not None. If 'rec_error',
            ALS stops at current iteration if (previous rec_error - current rec_error) < tol.
            If 'abs_rec_error', ALS terminates when |previous rec_error - current rec_error| < tol.

        fixed_modes : list, optional, default is None
            A list of modes for which the initial value is not modified.
            The last mode cannot be fixed due to error computation.

        svd_mask_repeats : int, optional, default is 5
            If using a tensor with masked values, this initializes using SVD
            multiple times to remove the effect of these missing values on the initialization.

        linesearch : bool, optional, default is False
            Whether to perform line search as proposed by Bro.

        callback : callable, optional
            Function to call at the end of each iteration.

        implementation : str, optional, default is 'tensorly'
            Implementation to use for the decomposition.

        return_decomposition : bool, optional
            Whether to return the decomposition along with the reconstruction.

        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor
                                        and computation time).
        performance_preset : {'fast', 'balanced', 'accurate', 'tensorly'} or None
            Optional 4Denoise speed/accuracy preset. If None, use TensorLy's
            documented defaults exactly. If provided, this overrides
            ``n_iter_max``, ``init``, and ``tol``.
        working_dtype : dtype or None
            Optional dtype used during decomposition, e.g. ``np.float32`` to
            reduce memory pressure. If None, keep the input dtype.

        Returns
        -------
        CPTensor(weight, factors) : tuple
            weights : 1D array of shape (rank, )
                All ones if normalize_factors is False (default).
                Weights of the (normalized) factors otherwise.
            factors : list of ndarray
                List of factors of the CP decomposition. Element i is of shape
                (tensor.shape[i], rank).
            sparse_component : nD array of shape tensor.shape
                Returns only if sparsity is not None.

        errors : list
            A list of reconstruction errors at each iteration of the algorithms
            (if return_errors is True).

        Notes
        -----
        CANDECOMP/PARAFAC decomposition via alternating least squares (ALS).
        """
        if performance_preset is not None:
            preset = self._decomposition_preset('parafac', performance_preset)
            n_iter_max = preset['n_iter_max']
            init = preset['init']
            tol = preset['tol']

        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        if unfold_domain is not None:
            tensor, unfold_metadata = self._unfold_with_hyperdata(
                tensor, unfold_domain, unfold_method
            )
        else:
            unfold_metadata = None

        if working_dtype is not None:
            tensor = np.asarray(tensor, dtype=working_dtype)

        # Conditional implementation
        if implementation == 'tensorly':
            # CANDECOMP/PARAFAC decomposition via ALS
            result = par(tensor, rank=rank, n_iter_max=n_iter_max, init=init, svd=svd, normalize_factors=normalize_factors,
                                                    orthogonalise=orthogonalise, tol=tol, random_state=random_state, verbose=verbose, return_errors=return_errors,
                                                    sparsity=sparsity, l2_reg=l2_reg, mask=mask, cvg_criterion=cvg_criterion, fixed_modes=fixed_modes,
                                                    svd_mask_repeats=svd_mask_repeats, linesearch=linesearch, callback=callback)
            if return_errors:
                cp_result, errors = result
            else:
                cp_result = result
                errors = None
        else:
            raise ValueError(f"Unknown implementation: {implementation}")

        if sparsity is not None:
            cp_tensor, sparse_component = cp_result
        else:
            cp_tensor = cp_result
            sparse_component = None

        # Reconstruct the tensor from CP factors
        reconstruction = tl.cp_to_tensor(cp_tensor)

        # Re-fold the tensor if it was unfolded
        if unfold_domain is not None:
            reconstruction = self._refold_with_hyperdata(
                reconstruction, unfold_metadata
            )

        if return_decomposition:
            weights, factors = cp_tensor
            results = [weights, factors, reconstruction]
            if return_errors:
                results.append(errors)
            if sparsity is not None:
                results.append(sparse_component)
            return results
        if return_errors:
            return reconstruction, errors
        return reconstruction

    # Dataset must be 3D
    def parafac2(self, tensor_slices, rank, n_iter_max=2000, init='random',
                 svd='truncated_svd', normalize_factors=False, tol=1e-08,
                 nn_modes=None, random_state=None, verbose=False,
                 return_errors=False, n_iter_parafac=5, linesearch=True,
                 implementation='tensorly', return_decomposition=False,
                 unfold_domain=None, unfold_method='row_major',
                 performance_preset=None, working_dtype=None):
        """
        Apply PARAFAC2 decomposition to a 3D input tensor.

        The TensorLy-facing parameters mirror
        ``tensorly.decomposition.parafac2``. ``performance_preset`` and
        ``working_dtype`` are 4Denoise conveniences and are optional.
        """
        if implementation != 'tensorly':
            raise ValueError(
                f"Unknown parafac2 implementation {implementation!r}; "
                "supported implementation: 'tensorly'."
            )
        if performance_preset is not None:
            preset = self._decomposition_preset('parafac2', performance_preset)
            n_iter_max = preset['n_iter_max']
            n_iter_parafac = preset['n_iter_parafac']
            init = preset['init']
            tol = preset['tol']
            linesearch = preset['linesearch']

        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        if unfold_domain is not None:
            tensor_slices, unfold_metadata = self._unfold_with_hyperdata(
                tensor_slices, unfold_domain, unfold_method
            )
        else:
            unfold_metadata = None

        if working_dtype is not None:
            tensor_slices = np.asarray(tensor_slices, dtype=working_dtype)

        if np.ndim(tensor_slices) != 3:
            raise ValueError(
                "parafac2 requires a 3D array after optional unfolding. "
                "Use unfold_domain='real' or unfold_domain='reciprocal' for "
                "4D data; unfold_domain='both' produces a 2D matrix and is not "
                "valid for parafac2."
            )

        decomposition_result = par2(
            tensor_slices,
            rank=rank,
            n_iter_max=n_iter_max,
            init=init,
            svd=svd,
            normalize_factors=normalize_factors,
            tol=tol,
            nn_modes=nn_modes,
            random_state=random_state,
            verbose=verbose,
            return_errors=return_errors,
            n_iter_parafac=n_iter_parafac,
            linesearch=linesearch,
        )

        if return_errors:
            decomposition, errors = decomposition_result
        else:
            decomposition = decomposition_result
            errors = None

        reconstruction = tl.parafac2_tensor.parafac2_to_tensor(decomposition)
        if unfold_domain is not None:
            reconstruction = self._refold_with_hyperdata(
                reconstruction, unfold_metadata
            )

        if return_decomposition:
            results = [decomposition, self._as_hyperdata(reconstruction)]
            if return_errors:
                results.append(errors)
            return results
        if return_errors:
            return reconstruction, errors
        return reconstruction

    # Testing...
    def randomised_parafac(self, tensor, rank, n_samples, n_iter_max=100, init='random',
                           svd='truncated_svd', tol=1e-08, max_stagnation=20,
                           return_errors=False, random_state=None, verbose=0,
                           callback=None, implementation='tensorly',
                           return_decomposition=False, unfold_domain=None,
                           unfold_method='row_major', working_dtype=None):
        """Randomised CP decomposition via sampled ALS

        Parameters
        ----------
        tensor : ndarray
            The input tensor to decompose.

        rank : int
            Number of components.

        n_samples : int
            Number of samples per ALS step.

        n_iter_max : int, optional, default is 100
            Maximum number of iterations.

        init : {'svd', 'random'}, optional, default is 'random'
            Method to initialize the decomposition.

        svd : str, optional, default is 'truncated_svd'
            Function to use to compute the SVD. Acceptable values are in tensorly.SVD_FUNS.

        tol : float, optional, default is 1e-08
            Tolerance: the algorithm stops when the variation in the reconstruction error is less than the tolerance.

        max_stagnation : int, optional, default is 20
            Maximum allowed number of iterations with no decrease in fit.

        random_state : {None, int, np.random.RandomState}, optional, default is None
            Random seed or state to initialize the random number generator.

        return_errors : bool, optional, default is False
            If True, return the per-iteration relative reconstruction errors
            reported by TensorLy's callback. TensorLy 0.10 may estimate these
            from sampled tensor entries; they are not necessarily exact errors.

        verbose : int, optional, default is 0
            Level of verbosity.

        callback : callable, optional
            Called with the initial decomposition alone, then with the
            decomposition and relative error after each iteration. Returning
            True from an iteration callback stops the fit. Error collection
            preserves these calls and the callback's stopping decision.

        implementation : str, optional, default is 'tensorly'
            Implementation to use for the decomposition.

        return_decomposition : bool, optional
            Whether to return the decomposition along with the reconstruction.

        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor and computation time).

        working_dtype : dtype or None
            Optional dtype used during decomposition, e.g. ``np.float32`` to
            reduce memory pressure. If None, keep the input dtype.

        Returns
        -------
        ndarray or tuple or list
            Dense reconstruction by default, or ``(reconstruction, errors)``
            when ``return_errors=True``. With ``return_decomposition=True``,
            return ``[weights, factors, reconstruction]``, appending the error
            history when requested. The initialization callback contributes
            no entry to that history because it supplies no error.

        Notes
        -----
        Randomised CP decomposition via sampled ALS.
        """
        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        if unfold_domain is not None:
            tensor, unfold_metadata = self._unfold_with_hyperdata(
                tensor, unfold_domain, unfold_method
            )
        else:
            unfold_metadata = None

        if working_dtype is not None:
            tensor = np.asarray(tensor, dtype=working_dtype)

        # Conditional implementation
        if implementation == 'tensorly':
            errors = [] if return_errors else None
            if return_errors:
                def collect_error(decomposition, relative_error=None):
                    # Initialization supplies only the decomposition in both
                    # TensorLy 0.9 and 0.10.
                    if relative_error is not None:
                        errors.append(float(relative_error))
                    if callback is not None:
                        if relative_error is None:
                            return callback(decomposition)
                        return callback(decomposition, relative_error)
                    return None
            else:
                collect_error = callback

            cp_tensor = rand_parafac(
                tensor,
                rank=rank,
                n_samples=n_samples,
                n_iter_max=n_iter_max,
                init=init,
                svd=svd,
                tol=tol,
                max_stagnation=max_stagnation,
                random_state=random_state,
                verbose=verbose,
                callback=collect_error,
            )
        else:
            raise ValueError(f"Unknown implementation: {implementation}")

        # Reconstruct the tensor from CP factors
        reconstruction = tl.cp_to_tensor(cp_tensor)

        # Re-fold the tensor if it was unfolded
        if unfold_domain is not None:
            reconstruction = self._refold_with_hyperdata(
                reconstruction, unfold_metadata
            )

        if return_decomposition:
            weights, factors = cp_tensor
            results = [weights, factors, reconstruction]
            if return_errors:
                results.append(errors)
            return results
        if return_errors:
            return reconstruction, errors
        return reconstruction


    def parafac_power_iteration(self, tensor, rank, n_repeat=10, n_iteration=10,
                                verbose=0, implementation='tensorly',
                                return_decomposition=False, unfold_domain=None, unfold_method='row_major'):
        """CP Decomposition via Robust Tensor Power Iteration

        Parameters
        ----------
        tensor : tl.tensor
            Input tensor to decompose.

        rank : int
            Rank of the decomposition (number of rank-1 components).

        n_repeat : int, optional, default is 10
            Number of initializations to be tried.

        n_iteration : int, optional, default is 10
            Number of power iterations.

        verbose : bool, optional, default is 0
            Level of verbosity.

        implementation : str, optional, default is 'tensorly'
            Implementation to use for the decomposition.

        return_decomposition : bool, optional
            Whether to return the decomposition along with the reconstruction.

        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor
                                        and computation time).

        Returns
        -------
        weights : 1D tl.tensor of length rank
            Contains the eigenvalue of each eigenvector.

        factors : list of 2-D tl.tensor of shape (size, rank)
            Each column of each factor corresponds to one eigenvector.

        Notes
        -----
        CP Decomposition via Robust Tensor Power Iteration.
        """
        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        if unfold_domain is not None:
            tensor, unfold_metadata = self._unfold_with_hyperdata(
                tensor, unfold_domain, unfold_method
            )

        # Conditional implementation
        if implementation == 'tensorly':
            # CP Decomposition via Robust Tensor Power Iteration
            weights, factors = parafac_power_iter(
                tensor,
                rank=rank,
                n_repeat=n_repeat,
                n_iteration=n_iteration,
                verbose=verbose,
            )

        else:
            raise ValueError(f"Unknown implementation: {implementation}")

        # Reconstruct the tensor from CP factors
        reconstruction = tl.cp_to_tensor((weights, factors))

        # Re-fold the tensor if it was unfolded
        if unfold_domain is not None:
            reconstruction = self._refold_with_hyperdata(
                reconstruction, unfold_metadata
            )

        if return_decomposition:
            return weights, factors, reconstruction
        else:
            return reconstruction


    def symmetric_parafac_power_iteration(self, tensor, rank, n_repeat=10,
                                          n_iteration=10, verbose=False,
                                          implementation='tensorly',
                                          return_decomposition=False, unfold_domain=None, unfold_method='row_major'):
        """Symmetric CP Decomposition via Robust Symmetric Tensor Power Iteration

        Parameters
        ----------
        tensor : tl.tensor
            Input tensor to decompose, must be symmetric of shape (size, )*order.

        rank : int
            Rank of the decomposition (number of rank-1 components).

        n_repeat : int, optional, default is 10
            Number of initializations to be tried.

        n_iteration : int, optional, default is 10
            Number of power iterations.

        verbose : bool, optional, default is False
            Level of verbosity.

        implementation : str, optional, default is 'tensorly'
            Implementation to use for the decomposition.

        return_decomposition : bool, optional
            Whether to return the decomposition along with the reconstruction.

        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor and computation time).

        Returns
        -------
        weights : 1D tl.tensor of length rank
            Contains the eigenvalue of each eigenvector.

        factor : 2-D tl.tensor of shape (size, rank)
            Each column corresponds to one eigenvector.

        Notes
        -----
        Symmetric CP Decomposition via Robust Symmetric Tensor Power Iteration.
        """
        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        if unfold_domain is not None:
            tensor, unfold_metadata = self._unfold_with_hyperdata(
                tensor, unfold_domain, unfold_method
            )

        # Conditional implementation
        if implementation == 'tensorly':
            # Symmetric CP Decomposition via Robust Symmetric Tensor Power Iteration
            weights, factor = sym_parafac_power_iter(
                tensor,
                rank=rank,
                n_repeat=n_repeat,
                n_iteration=n_iteration,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Unknown implementation: {implementation}")

        # A symmetric CP decomposition uses the same factor for every mode.
        factors = [factor] * np.ndim(tensor)
        reconstruction = tl.cp_to_tensor((weights, factors))

        # Re-fold the tensor if it was unfolded
        if unfold_domain is not None:
            reconstruction = self._refold_with_hyperdata(
                reconstruction, unfold_metadata
            )

        if return_decomposition:
            return weights, factor, reconstruction
        else:
            return reconstruction


    def robust_pca(self, tensor, mask=None, tol=1e-06, reg_E=1.0,
                   reg_J=1.0, mu_init=0.0001, mu_max=1e10,
                   learning_rate=1.1, n_iter_max=100, return_errors=False,
                   verbose=1, implementation='tensorly',
                   return_decomposition=False, unfold_domain=None,
                   unfold_method='row_major', working_dtype=None):
        """
        Denoise a tensor by separating low-rank signal and sparse corruption.

        This wraps :func:`tensorly.decomposition.robust_pca`. The low-rank
        component is returned as the denoised reconstruction by default. Use
        ``return_decomposition=True`` to also obtain the sparse component and,
        when requested, the per-iteration reconstruction errors.

        Parameters
        ----------
        tensor : ndarray
            Input tensor.
        mask : ndarray, optional
            Boolean array with the same shape as ``tensor``. It should be zero
            where values are missing and one elsewhere.
        tol : float, optional
            Convergence tolerance.
        reg_E : float, optional
            Regularization strength for the sparse component.
        reg_J : float, optional
            Regularization strength for the low-rank component.
        mu_init : float, optional
            Initial augmented-Lagrangian penalty.
        mu_max : float, optional
            Maximum augmented-Lagrangian penalty.
        learning_rate : float, optional
            Multiplicative increase applied to the penalty each iteration.
        n_iter_max : int, optional
            Maximum number of iterations.
        return_errors : bool, optional
            If True, return ``(low_rank, errors)`` by default, or append
            errors to the decomposition payload when requested.
        verbose : int, optional
            TensorLy verbosity level.
        implementation : {'tensorly'}, optional
            Numerical implementation to use.
        return_decomposition : bool, optional
            If True, return ``[low_rank, sparse_component]`` and append
            errors when ``return_errors=True``.
        unfold_domain : {'real', 'reciprocal', 'both'} or None, optional
            Unfold a 4D tensor before decomposition and refold both components
            afterward.
        unfold_method : str, optional
            Traversal method used when unfolding.
        working_dtype : dtype or None, optional
            Optional dtype used during decomposition.

        Returns
        -------
        ndarray or list
            Low-rank denoised tensor, or decomposition details when
            ``return_decomposition=True``.
        """
        if implementation != 'tensorly':
            raise ValueError(f"Unknown implementation: {implementation}")

        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != np.shape(tensor):
                raise ValueError(
                    "mask must have the same shape as the input tensor; "
                    f"got {mask.shape} and {np.shape(tensor)}."
                )

        if unfold_domain is not None:
            tensor, unfold_metadata = self._unfold_with_hyperdata(
                tensor, unfold_domain, unfold_method
            )
            if mask is not None:
                mask, _ = self._unfold_with_hyperdata(
                    mask, unfold_domain, unfold_method
                )
        else:
            unfold_metadata = None

        if working_dtype is not None:
            tensor = np.asarray(tensor, dtype=working_dtype)

        result = robust_tensor_pca(
            tensor,
            mask=mask,
            tol=tol,
            reg_E=reg_E,
            reg_J=reg_J,
            mu_init=mu_init,
            mu_max=mu_max,
            learning_rate=learning_rate,
            n_iter_max=n_iter_max,
            return_errors=return_errors,
            verbose=verbose,
        )

        if return_errors:
            low_rank, sparse_component, errors = result
        else:
            low_rank, sparse_component = result
            errors = None

        if unfold_metadata is not None:
            low_rank = self._refold_with_hyperdata(
                low_rank, unfold_metadata
            )
            sparse_component = self._refold_with_hyperdata(
                sparse_component, unfold_metadata
            )

        if return_decomposition:
            results = [low_rank, sparse_component]
            if return_errors:
                results.append(errors)
            return results
        if return_errors:
            return low_rank, errors
        return low_rank



    def non_negative_parafac_hals(self, tensor, rank, n_iter_max=100, init='svd',
                    svd='truncated_svd', tol=1e-07, random_state=None,
                    sparsity_coefficients=None, fixed_modes=None,
                    nn_modes='all', exact=False, normalize_factors=False,
                    verbose=False, return_errors=False, cvg_criterion='abs_rec_error',
                    implementation='tensorly', return_decomposition=False,
                    unfold_domain=None, unfold_method='row_major',
                    working_dtype=None):
        """Non-negative CP decomposition via HALS

        Uses Hierarchical ALS (Alternating Least Squares) which updates each factor
        column-wise (one column at a time while keeping all other columns fixed).

        Parameters
        ----------
        tensor : ndarray
            The input tensor to decompose.

        rank : int
            Number of components.

        n_iter_max : int, optional, default is 100
            Maximum number of iterations.

        init : {'svd', 'random'}, optional, default is 'svd'
            Method to initialize the decomposition.

        svd : str, optional, default is 'truncated_svd'
            Function to use to compute the SVD. Acceptable values are in tensorly.SVD_FUNS.

        tol : float, optional, default is 1e-07
            Tolerance: the algorithm stops when the variation in the reconstruction
            error is less than the tolerance.

        random_state : {None, int, np.random.RandomState}, optional, default is None
            Random seed or state to initialize the random number generator.

        sparsity_coefficients : array of float, optional, default is None
            The sparsity coefficients on each factor. If None, the algorithm is
            computed without sparsity.

        fixed_modes : array of integers, optional, default is None
            Indices of modes that should not be updated.

        nn_modes : None, 'all' or array of integers, optional, default is 'all'
            Specify which modes to impose non-negativity constraints on. If 'all',
            then non-negativity is imposed on all modes.

        exact : bool, optional, default is False
            If True, the algorithm gives results with high precision but it needs
            high computational cost. If False, the algorithm gives an approximate solution.

        normalize_factors : bool, optional, default is False
            If True, aggregate the weights of each factor in a 1D-tensor of shape
            (rank, ), which will contain the norms of the factors.

        verbose : bool, optional, default is False
            Indicates whether the algorithm prints the successive reconstruction
            errors or not.

        return_errors : bool, optional, default is False
            Indicates whether the algorithm should return all reconstruction
            errors and computation time of each iteration.

        cvg_criterion : {'abs_rec_error', 'rec_error'}, optional, default is 'abs_rec_error'
            Stopping criterion for ALS, works if tol is not None. If 'rec_error',
            ALS stops at current iteration if (previous rec_error - current rec_error) < tol.
            If 'abs_rec_error', ALS terminates when |previous rec_error - current rec_error| < tol.

        return_decomposition : bool, optional
            Whether to return the decomposition along with the reconstruction.

        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor
                                        and computation time).

        working_dtype : dtype or None
            Optional dtype used during decomposition, e.g. ``np.float32`` to
            reduce memory pressure. If None, keep the input dtype.

        Returns
        -------
        factors : ndarray list
            List of positive factors of the CP decomposition. Element i is of
            shape (tensor.shape[i], rank).

        errors : list
            A list of reconstruction errors at each iteration of the algorithm
            (if return_errors is True).

        Notes
        -----
        Non-negative CP decomposition via HALS.
        """
        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        if unfold_domain is not None:
            tensor, unfold_metadata = self._unfold_with_hyperdata(
                tensor, unfold_domain, unfold_method
            )
        else:
            unfold_metadata = None

        if working_dtype is not None:
            tensor = np.asarray(tensor, dtype=working_dtype)

        # Conditional implementation
        if implementation == 'tensorly':
            # Non-negative CP decomposition via HALS
            result = nn_parafac_hals(
                tensor,
                rank=rank,
                n_iter_max=n_iter_max,
                init=init,
                svd=svd,
                tol=tol,
                random_state=random_state,
                sparsity_coefficients=sparsity_coefficients,
                fixed_modes=fixed_modes,
                nn_modes=nn_modes,
                exact=exact,
                normalize_factors=normalize_factors,
                verbose=verbose,
                return_errors=return_errors,
                cvg_criterion=cvg_criterion,
            )
            cp_tensor, errors = self._split_cp_result(result, return_errors)
        else:
            raise ValueError(f"Unknown implementation: {implementation}")

        # Reconstruct the tensor from CP factors
        reconstruction = tl.cp_to_tensor(cp_tensor)

        # Re-fold the tensor if it was unfolded
        if unfold_domain is not None:
            reconstruction = self._refold_with_hyperdata(
                reconstruction, unfold_metadata
            )

        if return_decomposition:
            weights, factors = cp_tensor
            results = [weights, factors, reconstruction]
            if return_errors:
                results.append(errors)
            return results
        if return_errors:
            return reconstruction, errors
        return reconstruction

    def non_negative_parafac(self, tensor, rank, n_iter_max=100, init='svd',
                             svd='truncated_svd', tol=1e-06, random_state=None,
                             verbose=0, normalize_factors=False, return_errors=False,
                             mask=None, cvg_criterion='abs_rec_error', fixed_modes=None,
                             implementation='tensorly', return_decomposition=False,
                             unfold_domain=None, unfold_method='row_major',
                             working_dtype=None):
        """Non-negative CP decomposition using multiplicative updates

        Parameters
        ----------
        tensor : ndarray
            The input tensor to decompose.

        rank : int
            Number of components.

        n_iter_max : int, optional, default is 100
            Maximum number of iterations.

        init : {'svd', 'random'}, optional, default is 'svd'
            Method to initialize the decomposition.

        svd : str, optional, default is 'truncated_svd'
            Function to use to compute the SVD. Acceptable values are in tensorly.SVD_FUNS.

        tol : float, optional, default is 1e-06
            Tolerance: the algorithm stops when the variation in the reconstruction error is less than the tolerance.

        random_state : {None, int, np.random.RandomState}, optional
            Random seed or state to initialize the random number generator.

        verbose : int, optional, default is 0
            Level of verbosity.

        normalize_factors : bool, optional, default is False
            If True, aggregate the weights of each factor in a 1D-tensor of shape (rank, ), which will contain the norms of the factors.

        return_errors : bool, optional, default is False
            Indicates whether to return all reconstruction errors and computation time of each iteration.

        mask : ndarray, optional
            Array of booleans with the same shape as tensor. Should be 0 where the values are missing and 1 everywhere else.

        cvg_criterion : {'abs_rec_error', 'rec_error'}, optional, default is 'abs_rec_error'
            Stopping criterion for ALS, works if tol is not None. If 'rec_error', ALS stops at current iteration if (previous rec_error - current rec_error) < tol. If 'abs_rec_error', ALS terminates when |previous rec_error - current rec_error| < tol.

        fixed_modes : list, optional, default is None
            A list of modes for which the initial value is not modified. The last mode cannot be fixed due to error computation.

        implementation : str, optional, default is 'tensorly'
            Implementation to use for the decomposition.

        return_decomposition : bool, optional
            Whether to return the decomposition along with the reconstruction.

        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor and computation time).

        working_dtype : dtype or None
            Optional dtype used during decomposition, e.g. ``np.float32`` to
            reduce memory pressure. If None, keep the input dtype.

        Returns
        -------
        factors : ndarray list
            List of positive factors of the CP decomposition. Element i is of shape (tensor.shape[i], rank).

        errors : list
            A list of reconstruction errors at each iteration of the algorithm (if return_errors is True).

        Notes
        -----
        Non-negative CP decomposition using multiplicative updates.
        """
        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        if unfold_domain is not None:
            tensor, unfold_metadata = self._unfold_with_hyperdata(
                tensor, unfold_domain, unfold_method
            )
        else:
            unfold_metadata = None

        if working_dtype is not None:
            tensor = np.asarray(tensor, dtype=working_dtype)

        # Conditional implementation
        if implementation == 'tensorly':
            # Non-negative CP decomposition using multiplicative updates
            result = nn_parafac(
                tensor,
                rank=rank,
                n_iter_max=n_iter_max,
                init=init,
                svd=svd,
                tol=tol,
                random_state=random_state,
                verbose=verbose,
                normalize_factors=normalize_factors,
                return_errors=return_errors,
                mask=mask,
                cvg_criterion=cvg_criterion,
                fixed_modes=fixed_modes,
            )
            cp_tensor, errors = self._split_cp_result(result, return_errors)
        else:
            raise ValueError(f"Unknown implementation: {implementation}")

        # Reconstruct the tensor from CP factors
        reconstruction = tl.cp_to_tensor(cp_tensor)

        # Re-fold the tensor if it was unfolded
        if unfold_domain is not None:
            reconstruction = self._refold_with_hyperdata(
                reconstruction, unfold_metadata
            )

        if return_decomposition:
            weights, factors = cp_tensor
            results = [weights, factors, reconstruction]
            if return_errors:
                results.append(errors)
            return results
        if return_errors:
            return reconstruction, errors
        return reconstruction


    def cp_constrained(self, tensor, rank, n_iter_max=100, n_iter_max_inner=10,
                       init='svd', svd='truncated_svd', tol_outer=1e-08,
                       tol_inner=1e-06, random_state=None, verbose=0,
                       return_errors=False, cvg_criterion='abs_rec_error',
                       fixed_modes=None, non_negative=None, l1_reg=None,
                       l2_reg=None, l2_square_reg=None, unimodality=None,
                       normalize=None, simplex=None, normalized_sparsity=None,
                       soft_sparsity=None, smoothness=None, monotonicity=None,
                       hard_sparsity=None, implementation='tensorly',
                       return_decomposition=False, unfold_domain=None,
                       unfold_method='row_major', working_dtype=None):
        """
        Apply constrained PARAFAC decomposition to an input tensor.

        Parameters
        ----------
        tensor : ndarray
            Input tensor to decompose.
        rank : int
            Number of components.
        n_iter_max : int
            Maximum number of outer iterations.
        n_iter_max_inner : int
            Maximum number of inner ADMM iterations.
        init : {'svd', 'random', CPTensor}, optional
            TensorLy initialization method.
        svd : str, optional
            SVD function used by TensorLy.
        tol_outer : float, optional
            Relative reconstruction-error tolerance for the outer loop.
        tol_inner : float, optional
            Absolute reconstruction-error tolerance for inner ADMM updates.
        return_decomposition : bool, optional
            If True, return ``[weights, factors, reconstruction]`` and append
            errors when ``return_errors=True``.
        unfold_domain : any, optional
            Apply unfolding before decomposition.
        working_dtype : dtype or None
            Optional dtype used during decomposition, e.g. ``np.float32``.

        Returns
        -------
        ndarray or list
            Reconstructed tensor, or decomposition details when
            ``return_decomposition=True``.

        Examples
        --------
        >>> my4Dobject = HyperData(data)
        >>> reconstructed_data = my4Dobject.denoise(method='cp_constrained',
                                                    unfold_domain='real',
                                                    rank=3, n_iter_max=50)
        """

        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        if unfold_domain is not None:
            tensor, unfold_metadata = self._unfold_with_hyperdata(
                tensor, unfold_domain, unfold_method
            )
        else:
            unfold_metadata = None

        if working_dtype is not None:
            tensor = np.asarray(tensor, dtype=working_dtype)

        if implementation == 'tensorly':
            result = constrained_parafac(
                tensor,
                rank=rank,
                n_iter_max=n_iter_max,
                n_iter_max_inner=n_iter_max_inner,
                init=init,
                svd=svd,
                tol_outer=tol_outer,
                tol_inner=tol_inner,
                random_state=random_state,
                verbose=verbose,
                return_errors=return_errors,
                cvg_criterion=cvg_criterion,
                fixed_modes=fixed_modes,
                non_negative=non_negative,
                l1_reg=l1_reg,
                l2_reg=l2_reg,
                l2_square_reg=l2_square_reg,
                unimodality=unimodality,
                normalize=normalize,
                simplex=simplex,
                normalized_sparsity=normalized_sparsity,
                soft_sparsity=soft_sparsity,
                smoothness=smoothness,
                monotonicity=monotonicity,
                hard_sparsity=hard_sparsity,
            )
            cp_tensor, errors = self._split_cp_result(result, return_errors)
        else:
            raise ValueError(f"Unknown implementation: {implementation}")

        reconstruction = tl.cp_to_tensor(cp_tensor)

        if unfold_domain is not None:
            reconstruction = self._refold_with_hyperdata(
                reconstruction, unfold_metadata
            )

        if return_decomposition:
            weights, factors = cp_tensor
            results = [weights, factors, reconstruction]
            if return_errors:
                results.append(errors)
            return results
        if return_errors:
            return reconstruction, errors
        return reconstruction


    def tensor_ring_als(self, tensor, rank, ls_solve='lstsq',
                        n_iter_max=100, tol=1e-06, random_state=None,
                        verbose=False, callback=None, return_errors=False,
                        implementation='tensorly', return_decomposition=False,
                        unfold_domain=None, unfold_method='row_major',
                        working_dtype=None):
        """
        Denoise a tensor with Tensor-Ring alternating least squares.

        This wraps :func:`tensorly.decomposition.tensor_ring_als` and returns
        the dense Tensor-Ring reconstruction by default.

        Parameters
        ----------
        tensor : ndarray
            Input tensor to decompose.
        rank : int or sequence of int
            Tensor-Ring rank. An integer applies the same rank to every core.
        ls_solve : {'lstsq', 'normal_eq'}, optional
            Least-squares solver. ``'lstsq'`` is more numerically stable;
            ``'normal_eq'`` can be faster but less accurate.
        n_iter_max : int, optional
            Maximum number of ALS iterations.
        tol : float, optional
            Stop when the relative reconstruction-error change is below this
            value.
        random_state : None, int, or numpy.random.RandomState, optional
            Random state used to initialize the Tensor-Ring cores.
        verbose : bool, optional
            If True, print TensorLy iteration information.
        callback : callable or None, optional
            TensorLy callback receiving the current ``TRTensor`` and relative
            reconstruction error after each iteration.
        return_errors : bool, optional
            Return the relative error sequence collected through TensorLy's
            iteration callback.
        implementation : {'tensorly'}, optional
            Numerical implementation to use.
        return_decomposition : bool, optional
            If True, return ``(tr_decomposition, reconstruction)`` and append
            errors when ``return_errors=True``.
        unfold_domain : {'real', 'reciprocal', 'both'} or None, optional
            Unfold a 4D tensor before decomposition and refold the dense
            reconstruction afterward.
        unfold_method : str, optional
            Traversal method used when unfolding.
        working_dtype : dtype or None, optional
            Optional dtype used during decomposition.

        Returns
        -------
        ndarray or tuple
            Dense Tensor-Ring reconstruction, or the ``TRTensor`` and dense
            reconstruction when ``return_decomposition=True``.
        """
        if implementation != 'tensorly':
            raise ValueError(f"Unknown implementation: {implementation}")

        if unfold_domain is not None:
            tensor, unfold_metadata = self._unfold_with_hyperdata(
                tensor, unfold_domain, unfold_method
            )
        else:
            unfold_metadata = None

        if working_dtype is not None:
            tensor = np.asarray(tensor, dtype=working_dtype)

        errors = [] if return_errors else None
        if return_errors:
            def collect_error(decomposition, relative_error):
                errors.append(float(relative_error))
                if callback is not None:
                    return callback(decomposition, relative_error)
                return None
        else:
            collect_error = callback

        tr_decomposition = tr_als(
            tensor,
            rank=rank,
            ls_solve=ls_solve,
            n_iter_max=n_iter_max,
            tol=tol,
            random_state=random_state,
            verbose=verbose,
            callback=collect_error,
        )
        reconstruction = tl.tr_to_tensor(tr_decomposition)

        if unfold_metadata is not None:
            reconstruction = self._refold_with_hyperdata(
                reconstruction, unfold_metadata
            )

        if return_decomposition:
            if return_errors:
                return tr_decomposition, reconstruction, errors
            return tr_decomposition, reconstruction
        if return_errors:
            return reconstruction, errors
        return reconstruction


    def tensor_ring_als_sampled(
            self, tensor, rank, n_samples, n_iter_max=100, tol=1e-06,
            uniform_sampling=False, randomized_error=False,
            random_state=None, verbose=False, callback=None,
            return_errors=False,
            implementation='tensorly', return_decomposition=False,
            unfold_domain=None, unfold_method='row_major',
            working_dtype=None):
        """
        Denoise a tensor with sampled Tensor-Ring alternating least squares.

        Sampling reduces the least-squares problem size and can make Tensor-Ring
        decomposition faster at the cost of approximation accuracy.

        Parameters
        ----------
        tensor : ndarray
            Input tensor to decompose.
        rank : int or sequence of int
            Tensor-Ring rank. An integer applies the same rank to every core.
        n_samples : int or sequence of int
            Rows sampled while updating each core. An integer applies the same
            sample count to every mode.
        n_iter_max : int, optional
            Maximum number of sampled ALS iterations.
        tol : float, optional
            Stop when the relative reconstruction-error change is below this
            value.
        uniform_sampling : bool, optional
            Use uniform sampling instead of leverage-score sampling.
        randomized_error : bool, optional
            Estimate the residual using random sampling instead of computing
            the exact residual after each iteration.
        random_state : None, int, or numpy.random.RandomState, optional
            Random state used for initialization and sampling.
        verbose : bool, optional
            If True, print TensorLy iteration information.
        callback : callable or None, optional
            TensorLy callback receiving the current ``TRTensor`` and relative
            reconstruction error after each iteration.
        return_errors : bool, optional
            Return the relative error sequence collected through TensorLy's
            iteration callback.
        implementation : {'tensorly'}, optional
            Numerical implementation to use.
        return_decomposition : bool, optional
            If True, return ``(tr_decomposition, reconstruction)`` and append
            errors when ``return_errors=True``.
        unfold_domain : {'real', 'reciprocal', 'both'} or None, optional
            Unfold a 4D tensor before decomposition and refold the dense
            reconstruction afterward.
        unfold_method : str, optional
            Traversal method used when unfolding.
        working_dtype : dtype or None, optional
            Optional dtype used during decomposition.

        Returns
        -------
        ndarray or tuple
            Dense Tensor-Ring reconstruction, or the ``TRTensor`` and dense
            reconstruction when ``return_decomposition=True``.
        """
        if implementation != 'tensorly':
            raise ValueError(f"Unknown implementation: {implementation}")

        if unfold_domain is not None:
            tensor, unfold_metadata = self._unfold_with_hyperdata(
                tensor, unfold_domain, unfold_method
            )
        else:
            unfold_metadata = None

        if working_dtype is not None:
            tensor = np.asarray(tensor, dtype=working_dtype)

        errors = [] if return_errors else None
        if return_errors:
            def collect_error(decomposition, relative_error):
                errors.append(float(relative_error))
                if callback is not None:
                    return callback(decomposition, relative_error)
                return None
        else:
            collect_error = callback

        tr_decomposition = tr_als_sampled(
            tensor,
            rank=rank,
            n_samples=n_samples,
            n_iter_max=n_iter_max,
            tol=tol,
            uniform_sampling=uniform_sampling,
            randomized_error=randomized_error,
            random_state=random_state,
            verbose=verbose,
            callback=collect_error,
        )
        reconstruction = tl.tr_to_tensor(tr_decomposition)

        if unfold_metadata is not None:
            reconstruction = self._refold_with_hyperdata(
                reconstruction, unfold_metadata
            )

        if return_decomposition:
            if return_errors:
                return tr_decomposition, reconstruction, errors
            return tr_decomposition, reconstruction
        if return_errors:
            return reconstruction, errors
        return reconstruction

    def tensor_train_matrix(self, tensor, rank, svd='truncated_svd', verbose=False,
                            implementation='tensorly', return_decomposition=False,
                            unfold_domain=None, unfold_method='row_major',
                            working_dtype=None):
        """Decompose a tensor into a matrix in tt-format

        Decomposes the input tensor into a matrix in Tensor Train (TT) format.

        Parameters
        ----------
        tensor : tensorized matrix

        rank : 'same', float or int tuple
            If 'same', creates a decomposition with the same number of parameters as tensor.
            If float, creates a decomposition with rank x the number of parameters of tensor.
            Otherwise, the actual rank to be used, e.g., (1, rank_2, ..., 1) of size tensor.ndim//2.
            Note that boundary conditions dictate that the first rank = last rank = 1.

        svd : str, optional, default is 'truncated_svd'
            Function to use to compute the SVD. Acceptable values are in tensorly.SVD_FUNS.

        verbose : boolean, optional
            Level of verbosity.

        return_decomposition : boolean, optional
            Whether to return the decomposition along with the reconstruction.

        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor and computation time).

        working_dtype : dtype or None
            Optional dtype used during decomposition, e.g. ``np.float32``.

        Returns
        -------
        reconstruction :
        list containing 'reconstruction' and 'tt_matrix'

        Notes
        -----
        Tensor Train (TT) decomposition decomposes the input tensor into a sequence of matrices
        in TT-format by recursively applying SVD.
        """

        if implementation != 'tensorly':
            raise ValueError(f"Unknown implementation: {implementation}")

        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        if unfold_domain is not None:
            tensor, unfold_metadata = self._unfold_with_hyperdata(
                tensor, unfold_domain, unfold_method
            )
        else:
            unfold_metadata = None

        if working_dtype is not None:
            tensor = np.asarray(tensor, dtype=working_dtype)

        # Tensor Train matrix decomposition
        tt_matrix = tt_mat(tensor, rank=rank, svd=svd, verbose=verbose)

        # Reconstruct the tensorized matrix from TT-Matrix factors.
        reconstruction = tt_matrix_to_tensor(tt_matrix)

        # Re-fold the tensor if it was unfolded
        if unfold_domain is not None:
            reconstruction = self._refold_with_hyperdata(
                reconstruction, unfold_metadata
            )

        if return_decomposition:
            return tt_matrix, self._as_hyperdata(reconstruction)
        else:
            return reconstruction


    def tensor_train(self, input_tensor, rank, svd='truncated_svd', verbose=False,
                     return_decomposition=False, unfold_domain=None, unfold_method='row_major',
                     implementation='tensorly', working_dtype=None):
        """TT decomposition via recursive SVD

        Decomposes input_tensor into a sequence of order-3 tensors (factors) – also known as Tensor-Train decomposition.

        Parameters
        ----------
        input_tensor : tensorly.tensor
            The input tensor to decompose.

        rank : {int, int list}
            Maximum allowable TT rank of the factors. If int, then this is the same for all the factors.
            If int list, then rank[k] is the rank of the kth factor.

        svd : str, optional, default is 'truncated_svd'
            Function to use to compute the SVD. Acceptable values are in tensorly.SVD_FUNS.

        verbose : boolean, optional
            Level of verbosity.

        return_decomposition : boolean, optional
            Whether to return the decomposition along with the reconstruction.

        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor and computation time).

        working_dtype : dtype or None
            Optional dtype used during decomposition, e.g. ``np.float32``.

        Returns
        -------
        factors : TT factors
            Order-3 tensors of the TT decomposition.

        Notes
        -----
        Tensor-Train (TT) decomposition decomposes the input tensor into a sequence of order-3 tensors
        (factors) by recursively applying SVD.
        """

        if implementation != 'tensorly':
            raise ValueError(f"Unknown implementation: {implementation}")

        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        if unfold_domain is not None:
            input_tensor, unfold_metadata = self._unfold_with_hyperdata(
                input_tensor, unfold_domain, unfold_method
            )
        else:
            unfold_metadata = None

        if working_dtype is not None:
            input_tensor = np.asarray(input_tensor, dtype=working_dtype)

        # Tensor-Train decomposition
        factors = tt(input_tensor, rank=rank, svd=svd, verbose=verbose)

        # Reconstruct the tensor from TT factors.
        reconstruction = tt_to_tensor(factors)

        # Re-fold the tensor if it was unfolded
        if unfold_domain is not None:
            reconstruction = self._refold_with_hyperdata(
                reconstruction, unfold_metadata
            )

        if return_decomposition:
            return factors, self._as_hyperdata(reconstruction)
        else:
            return reconstruction

    def non_negative_tucker_hals(self, tensor, rank, n_iter_max=100, init='svd',
                                 svd='truncated_svd', tol=1e-08, sparsity_coefficients=None,
                                 core_sparsity_coefficient=None, fixed_modes=None,
                                 random_state=None, verbose=False, normalize_factors=False,
                                 return_errors=False, exact=False, algorithm='fista',
                                 return_decomposition=False, unfold_domain=None, unfold_method='row_major',
                                 implementation='tensorly', working_dtype=None):
        """Non-negative Tucker decomposition with HALS

        Uses HALS to update each factor column-wise and uses FISTA or active set algorithm to update the core.

        Parameters
        ----------
        tensor : ndarray
            The input tensor to decompose.

        rank : None, int or int list
            Core size, with one rank per tensor axis. An integer applies the
            same rank to every axis.

        n_iter_max : int, optional, default is 100
            Maximum number of iterations.

        init : {'svd', 'random'}, optional, default is 'svd'
            Method to initialize the decomposition.

        svd : str, optional, default is 'truncated_svd'
            Function to use to compute the SVD. Acceptable values are in tensorly.SVD_FUNS.

        tol : float, optional, default is 1e-08
            Tolerance: the algorithm stops when the variation in the reconstruction error is less than the tolerance.

        sparsity_coefficients : array of float, optional, default is None
            The sparsity coefficients for each factor. If None, the algorithm is computed without sparsity.

        core_sparsity_coefficient : array of float, optional, default is None
            Coefficient imposing sparsity on the core when updated with FISTA.

        fixed_modes : array of integers, optional, default is None
            Indices of modes that should not be updated.

        random_state : any, optional
            Random seed or state to initialize the random number generator.

        verbose : bool, optional, default is False
            Level of verbosity.

        normalize_factors : bool, optional, default is False
            If True, aggregates the norms of the factors in the core.

        return_errors : bool, optional, default is False
            Indicates whether to return all reconstruction errors and computation time of each iteration.

        exact : bool, optional, default is False
            If True, the HALS NNLS subroutines give results with high precision but with a higher computational cost.
            If False, the algorithm gives an approximate solution.

        algorithm : {'fista', 'active_set'}, optional, default is 'fista'
            Non-negative least square solution to update the core.

        return_decomposition : bool, optional
            Whether to return the decomposition along with the reconstruction.

        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor and computation time).

        working_dtype : dtype or None
            Optional dtype used during decomposition, e.g. ``np.float32``.

        Returns
        -------
        factors : ndarray list
            List of positive factors of the Tucker decomposition.

        errors : list
            List of reconstruction errors at each iteration of the algorithm (if return_errors is True).

        Notes
        -----
        Non-negative Tucker decomposition decomposes the input tensor into a core tensor and factor matrices,
        ensuring all elements are non-negative.
        """

        if implementation != 'tensorly':
            raise ValueError(f"Unknown implementation: {implementation}")

        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        if unfold_domain is not None:
            tensor, unfold_metadata = self._unfold_with_hyperdata(
                tensor, unfold_domain, unfold_method
            )
        else:
            unfold_metadata = None

        if working_dtype is not None:
            tensor = np.asarray(tensor, dtype=working_dtype)

        # Non-negative Tucker decomposition with HALS
        result = nnth(
            tensor,
            rank=rank,
            n_iter_max=n_iter_max,
            init=init,
            svd=svd,
            tol=tol,
            sparsity_coefficients=sparsity_coefficients,
            core_sparsity_coefficient=core_sparsity_coefficient,
            fixed_modes=fixed_modes,
            random_state=random_state,
            verbose=verbose,
            normalize_factors=normalize_factors,
            return_errors=return_errors,
            exact=exact,
            algorithm=algorithm,
        )
        tucker_tensor, errors = self._split_tucker_result(result, return_errors)
        core, factors = tucker_tensor

        # Reconstruct the tensor from Tucker factors and core
        reconstruction = tl.tucker_tensor.tucker_to_tensor((core, factors))

        # Re-fold the tensor if it was unfolded
        if unfold_domain is not None:
            reconstruction = self._refold_with_hyperdata(
                reconstruction, unfold_metadata
            )

        if return_decomposition:
            results = [core, factors, reconstruction]
            if return_errors:
                results.append(errors)
            return results
        if return_errors:
            return reconstruction, errors
        return reconstruction


    def non_negative_tucker(self, tensor, rank, n_iter_max=10, init='svd', tol=0.0001,
                            random_state=None, verbose=False, return_errors=False,
                            normalize_factors=False, return_decomposition=False, unfold_domain=None, unfold_method='row_major',
                            implementation='tensorly', working_dtype=None):
        """Non-negative Tucker decomposition

        Iterative multiplicative update.

        Parameters
        ----------
        tensor : ndarray
            The input tensor to decompose.

        rank : None, int or int list
            Size of the core tensor, (len(ranks) == tensor.ndim) if int, the same rank is used for all modes.

        n_iter_max : int, optional, default is 10
            Maximum number of iterations.

        init : {'svd', 'random'}, optional, default is 'svd'
            Method to initialize the decomposition.

        tol : float, optional, default is 0.0001
            Tolerance: the algorithm stops when the variation in the reconstruction error is less than the tolerance.

        random_state : {None, int, np.random.RandomState}, optional
            Random seed or state to initialize the random number generator.

        verbose : int, optional
            Level of verbosity.

        return_errors : bool, optional, default is False
            Indicates whether to return all reconstruction errors and computation time of each iteration.

        normalize_factors : bool, optional, default is False
            If True, aggregates the norms of the factors in the core.

        return_decomposition : bool, optional
            Whether to return the decomposition along with the reconstruction.

        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor and computation time).

        working_dtype : dtype or None
            Optional dtype used during decomposition, e.g. ``np.float32``.

        Returns
        -------
        core : ndarray
            Positive core of the Tucker decomposition, has shape ranks.

        factors : ndarray list
            List of factors of the Tucker decomposition, element i is of shape (tensor.shape[i], rank).

        Notes
        -----
        Non-negative Tucker decomposition decomposes the input tensor into a core tensor and factor matrices,
        ensuring all elements are non-negative.
        """

        if implementation != 'tensorly':
            raise ValueError(f"Unknown implementation: {implementation}")

        # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
        if unfold_domain is not None:
            tensor, unfold_metadata = self._unfold_with_hyperdata(
                tensor, unfold_domain, unfold_method
            )
        else:
            unfold_metadata = None

        if working_dtype is not None:
            tensor = np.asarray(tensor, dtype=working_dtype)

        # Non-negative Tucker decomposition
        result = nnt(
            tensor,
            rank=rank,
            n_iter_max=n_iter_max,
            init=init,
            tol=tol,
            random_state=random_state,
            verbose=verbose,
            return_errors=return_errors,
            normalize_factors=normalize_factors,
        )
        tucker_tensor, errors = self._split_tucker_result(result, return_errors)
        core, factors = tucker_tensor

        # Reconstruct the tensor from Tucker factors and core
        reconstruction = tl.tucker_tensor.tucker_to_tensor((core, factors))

        # Re-fold the tensor if it was unfolded
        if unfold_domain is not None:
            reconstruction = self._refold_with_hyperdata(
                reconstruction, unfold_metadata
            )

        if return_decomposition:
            results = [core, factors, reconstruction]
            if return_errors:
                results.append(errors)
            return results
        if return_errors:
            return reconstruction, errors
        return reconstruction

    def partial_tucker(self, tensor, rank, modes=None, n_iter_max=100,
                       init='svd', tol=0.0001, svd='truncated_svd', random_state=None,
                       verbose=False, mask=None, svd_mask_repeats=5,
                       return_errors=False, return_decomposition=False,
                       unfold_domain=None,
                       unfold_method='row_major', implementation='tensorly',
                       working_dtype=None):
        """Partial Tucker decomposition via Higher Order Orthogonal Iteration (HOI)

        Decomposes tensor into a Tucker decomposition exclusively along the provided modes.

        Parameters
        ----------
        tensor : ndarray
            The input tensor to decompose.

        rank : None, int or int list
            Rank for the selected modes. A sequence must have the same length
            as ``modes``; an integer applies to every selected mode.

        modes : None, int list, optional
            List of the modes on which to perform the decomposition.

        n_iter_max : int, optional, default is 100
            Maximum number of iterations.

        init : {'svd', 'random'}, or TuckerTensor, optional, default is 'svd'
            Method to initialize the decomposition. If a TuckerTensor is provided, this is used for initialization.

        svd : str, optional, default is 'truncated_svd'
            Function to use to compute the SVD. Acceptable values are in tensorly.tenalg.svd.SVD_FUNS.

        tol : float, optional, default is 0.0001
            Tolerance: the algorithm stops when the variation in the reconstruction error is less than the tolerance.

        random_state : {None, int, np.random.RandomState}, optional
            Random seed or state to initialize the random number generator.

        verbose : int, optional
            Level of verbosity.

        mask : ndarray, optional
            Array of booleans with the same shape as tensor. Should be 0 where the values are missing and 1 everywhere else.
            Note: if tensor is sparse, then mask should also be sparse with a fill value of 1 (or True).

        svd_mask_repeats : int, optional, default is 5
            Number of repetitions for the SVD in case of masking.

        return_decomposition : bool, optional
            Whether to return the decomposition along with the reconstruction.
        return_errors : bool, optional
            Return TensorLy's reconstruction errors for each iteration.

        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor and computation time).

        working_dtype : dtype or None
            Optional dtype used during decomposition, e.g. ``np.float32``.

        Returns
        -------
        core : ndarray
            Core tensor of the Tucker decomposition.

        factors : ndarray list
            One factor for each selected mode, in ``modes`` order.

        Notes
        -----
        Partial Tucker decomposition decomposes the input tensor into a core tensor and factor matrices
        along the specified modes.
        """

        if implementation == 'tensorly':

            # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
            if unfold_domain is not None:
                tensor, unfold_metadata = self._unfold_with_hyperdata(
                    tensor, unfold_domain, unfold_method
                )
            else:
                unfold_metadata = None

            if working_dtype is not None:
                tensor = np.asarray(tensor, dtype=working_dtype)

            # Partial Tucker decomposition
            result = partial_tuck(
                tensor,
                rank=rank,
                modes=modes,
                n_iter_max=n_iter_max,
                init=init,
                tol=tol,
                svd=svd,
                random_state=random_state,
                verbose=verbose,
                mask=mask,
                svd_mask_repeats=svd_mask_repeats,
            )
            tucker_tensor, errors = self._split_tucker_result(
                result,
                return_errors=True,
            )
            core, factors = tucker_tensor

            # Partial factors correspond to the selected modes, not axes 0..N.
            reconstruction_modes = (
                list(range(np.ndim(tensor))) if modes is None else list(modes)
            )
            reconstruction = multi_mode_dot(
                core, factors, modes=reconstruction_modes,
            )

            # Re-fold the tensor if it was unfolded
            if unfold_domain is not None:
                reconstruction = self._refold_with_hyperdata(
                    reconstruction, unfold_metadata
                )

            if return_decomposition:
                if return_errors:
                    return core, factors, reconstruction, errors
                return core, factors, reconstruction
            if return_errors:
                return reconstruction, errors
            return reconstruction
        else:
            raise ValueError(f"Unknown implementation: {implementation}")

    def tucker(self, tensor, rank, fixed_factors=None, n_iter_max=100, init='svd',
               return_errors=False, svd='truncated_svd', tol=0.0001, random_state=None,
               mask=None, verbose=False, return_decomposition=False,
               unfold_domain=None, unfold_method='row_major',
               implementation='tensorly', working_dtype=None):

        """Tucker decomposition via Higher Order Orthogonal Iteration (HOI)

        Decomposes tensor into a Tucker decomposition: tensor = [| core; factors[0], ...factors[-1] |]

        Parameters
        ----------
        tensor : ndarray
            The input tensor to decompose.

        rank : None, int or int list
            Size of the core tensor, (len(ranks) == tensor.ndim) if int, the same rank is used for all modes.

        fixed_factors : int list or None, optional, default is None
            If not None, list of modes for which to keep the factors fixed. Only valid if a Tucker tensor is provided as init.

        n_iter_max : int, optional, default is 100
            Maximum number of iterations.

        init : {'svd', 'random'}, optional, default is 'svd'
            Method to initialize the decomposition.

        return_errors : bool, optional, default is False
            Indicates whether to return all reconstruction errors and computation time of each iteration.

        svd : str, optional, default is 'truncated_svd'
            Function to use to compute the SVD. Acceptable values are in tensorly.SVD_FUNS.

        tol : float, optional, default is 0.0001
            Tolerance: the algorithm stops when the variation in the reconstruction error is less than the tolerance.

        random_state : {None, int, np.random.RandomState}, optional
            Random seed or state to initialize the random number generator.

        mask : ndarray, optional
            Array of booleans with the same shape as tensor. Should be 0 where the values are missing and 1 everywhere else.
            Note: if tensor is sparse, then mask should also be sparse with a fill value of 1 (or True).

        verbose : int, optional
            Level of verbosity.

        return_decomposition : bool, optional
            Whether to return the decomposition along with the reconstruction.

        unfold_domain : any, optional
            Apply unfolding if desired (reduces dimensionality of input tensor and computation time).

        working_dtype : dtype or None
            Optional dtype used during decomposition, e.g. ``np.float32``.

        implementation : string, optional
            'tensorly' to use Tensorly package implementation or
            'zhang' to use that in Zhang et al. (2020), see 'Notes' below.

        Returns
        -------
        core : ndarray
            Core tensor of the Tucker decomposition.

        factors : ndarray list
            List of factors of the Tucker decomposition. Its i-th element is of shape (tensor.shape[i], ranks[i]).

        Notes
        -----
        Tucker decomposition decomposes the input tensor into a core tensor and factor matrices.

        The "zhang" implementation is based on that in Zhang et al. (2020):
            Zhang, C., Han, R., Zhang, A. R., & Voyles, P. M. (2020).
            "Denoising atomic resolution 4D scanning transmission electron
            microscopy data with tensor singular value decomposition."
            Ultramicroscopy, 219, 113123.
        """

        if return_errors and fixed_factors is not None:
            raise NotImplementedError(
                "TensorLy 0.9 does not return convergence errors when "
                "tucker uses fixed_factors."
            )

        if implementation == 'tensorly':

            # Apply unfolding if desired (this reduces dimensionality of input tensor and computation time)
            if unfold_domain is not None:
                tensor, unfold_metadata = self._unfold_with_hyperdata(
                    tensor, unfold_domain, unfold_method
                )
            else:
                unfold_metadata = None

            if working_dtype is not None:
                tensor = np.asarray(tensor, dtype=working_dtype)

            # Tucker decomposition
            result = tuck(
                tensor,
                rank=rank,
                fixed_factors=fixed_factors,
                n_iter_max=n_iter_max,
                init=init,
                return_errors=return_errors,
                svd=svd,
                tol=tol,
                random_state=random_state,
                mask=mask,
                verbose=verbose,
            )
            tucker_tensor, errors = self._split_tucker_result(
                result,
                return_errors=return_errors,
            )
            core, factors = tucker_tensor

            # Reconstruct the tensor from Tucker factors and core
            reconstruction = tl.tucker_tensor.tucker_to_tensor((core, factors))

            # Re-fold the tensor if it was unfolded
            if unfold_domain is not None:
                reconstruction = self._refold_with_hyperdata(
                    reconstruction, unfold_metadata
                )

            if return_decomposition:
                results = [core, factors, reconstruction]
                if return_errors:
                    results.append(errors)
                return results
            if return_errors:
                return reconstruction, errors
            return reconstruction
        else:
            raise ValueError(f"Unknown implementation: {implementation}")


    # # =============================================================================
    # # Transform Domain Filtering
    # # =============================================================================

    def fourier_filter(self, target_data, mode='pass', r_inner=0, r_outer=None, sigma=10):
        """Filter a 2D image by a circular or annular Fourier-frequency mask.

        Frequency radii are measured in Fourier pixels from the DC component.
        With ``r_outer=None``, ``r_inner`` is a disk cutoff. The default
        ``r_inner=0`` uses a Gaussian low-pass with width ``sigma`` instead of
        a one-pixel disk. With ``r_outer`` specified, the pass region is the
        annulus from ``r_inner`` to ``r_outer``. ``mode='cut'`` uses the exact
        complement of the pass mask. ``sigma=0`` gives hard boundaries;
        positive ``sigma`` gives Gaussian-CDF edge transitions.

        The inverse transform restores the original Fourier axis order and
        retains signed values (or complex values for complex input).

        Parameters
        ----------
        target_data : ndarray
            Two-dimensional real or complex image.
        mode : {'pass', 'cut'}, optional
            Keep or reject the selected frequency region.
        r_inner : float, optional
            Nonnegative disk cutoff or annulus inner radius.
        r_outer : float or None, optional
            Annulus outer radius, greater than ``r_inner``.
        sigma : float, optional
            Nonnegative frequency-space edge width.

        Returns
        -------
        ndarray
            Filtered image with the same shape as the input.
        """
        data = np.asarray(target_data)
        if data.ndim != 2:
            raise ValueError("fourier_filter requires a 2D image.")
        if not np.issubdtype(data.dtype, np.number):
            raise TypeError("fourier_filter requires numeric image data.")
        if not isinstance(mode, str) or mode.lower() not in ('pass', 'cut'):
            raise ValueError("mode must be 'pass' or 'cut'.")
        mode = mode.lower()

        def finite_nonnegative(value, label):
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
                raise ValueError(f"{label} must be a nonnegative finite number.")
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{label} must be a nonnegative finite number."
                ) from exc
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{label} must be a nonnegative finite number.")
            return value

        r_inner = finite_nonnegative(r_inner, 'r_inner')
        sigma = finite_nonnegative(sigma, 'sigma')
        if r_outer is not None:
            r_outer = finite_nonnegative(r_outer, 'r_outer')
            if r_outer <= r_inner:
                raise ValueError("r_outer must be greater than r_inner.")

        rows, cols = data.shape
        y = np.arange(rows) - rows // 2
        x = np.arange(cols) - cols // 2
        radius = np.hypot(y[:, None], x[None, :])

        if r_outer is None and r_inner == 0:
            pass_mask = (
                np.exp(-0.5 * (radius / sigma) ** 2)
                if sigma > 0 else (radius == 0).astype(float)
            )
        elif r_outer is None:
            pass_mask = (
                0.5 * erfc((radius - r_inner) / (np.sqrt(2) * sigma))
                if sigma > 0 else (radius <= r_inner).astype(float)
            )
        elif sigma == 0:
            pass_mask = ((radius >= r_inner) & (radius <= r_outer)).astype(float)
        else:
            inner_gate = (
                1.0 if r_inner == 0 else
                0.5 * erfc((r_inner - radius) / (np.sqrt(2) * sigma))
            )
            outer_gate = 0.5 * erfc((radius - r_outer) / (np.sqrt(2) * sigma))
            pass_mask = inner_gate * outer_gate

        mask = pass_mask if mode == 'pass' else 1.0 - pass_mask
        mask_dtype = (
            np.float32 if data.dtype in (np.float32, np.complex64)
            else np.float64
        )
        spectrum = fftshift(fft2(data))
        filtered = ifft2(ifftshift(spectrum * mask.astype(mask_dtype)))
        return filtered if np.iscomplexobj(data) else filtered.real

    # #
    # def wavelet_thresholding(self, other):
    #     """
    #     Fourier filtering.

    #     discussed in [reference].
    #     """


    #     return clip_values(filtered_data)

    # #
    # def curvelet(self, other):
    #     """

    #     """

    #     return clip_values(filtered_data)

#%%

@dataclass(frozen=True)
class _DenoisingMethodContract:
    """Input accepted by one numerical method, before HyperData routing."""

    family: str
    input_ndim: tuple[int, ...]
    input_layout: str
    constraints: str = ''


# This describes the numerical helpers as they exist today. HyperData's 4D
# slicing/unfolding choices are reported separately by _DenoiseEngine.
_DENOISING_METHOD_CONTRACTS = {
    'adaptive_median_filter': _DenoisingMethodContract(
        'image filter', (2,), 'single 2D image',
        's and sMax must be odd positive window sizes.'),
    'anisotropic_diffusion': _DenoisingMethodContract(
        'tensor filter', (2, 3, 4), 'N-dimensional array',
        'Real-valued input; integer data are promoted to floating point.'),
    'bilateral': _DenoisingMethodContract(
        'image filter', (2,), 'single 2D image',
        'sigma_color is in input intensity units; OpenCV filters a '
        'baseline-shifted float32 image.'),
    'bm3d': _DenoisingMethodContract(
        'image filter', (2, 3), '2D image or (N, Y, X) stack',
        'A 3D stack is filtered one 2D image at a time; requires bm3d.'),
    'bm4d': _DenoisingMethodContract(
        'volume filter', (3,), 'single 3D volume',
        'Requires bm4d; 4D data must first be unfolded to 3D.'),
    'cp_constrained': _DenoisingMethodContract(
        'tensor decomposition', (2, 3, 4), 'N-dimensional tensor'),
    'fourier_filter': _DenoisingMethodContract(
        'frequency filter', (2,), 'single 2D image',
        'Radii and sigma are measured in Fourier pixels.'),
    'gaussian': _DenoisingMethodContract(
        'image filter', (2,), 'single 2D image',
        'Integer input is promoted to float; kernel_size is odd and positive.'),
    'median': _DenoisingMethodContract(
        'image filter', (2, 3, 4), 'image or tensor with selected axes',
        'axes=None filters every axis, including a 3D stack axis. '
        'HyperData selects real or reciprocal axis pairs for 4D domain routing.'),
    'nmf': _DenoisingMethodContract(
        'matrix factorization', (2,), 'nonnegative 2D matrix',
        'Negative input values are not accepted by NMF.'),
    'non_local_means': _DenoisingMethodContract(
        'image filter', (2, 3), '2D image or 3D volume',
        'h multiplies estimated noise; intensity range is preserved.'),
    'non_negative_parafac': _DenoisingMethodContract(
        'tensor decomposition', (2, 3, 4), 'nonnegative tensor'),
    'non_negative_parafac_hals': _DenoisingMethodContract(
        'tensor decomposition', (2, 3, 4), 'nonnegative tensor'),
    'non_negative_tucker': _DenoisingMethodContract(
        'tensor decomposition', (2, 3, 4), 'nonnegative tensor'),
    'non_negative_tucker_hals': _DenoisingMethodContract(
        'tensor decomposition', (2, 3, 4), 'nonnegative tensor'),
    'parafac': _DenoisingMethodContract(
        'tensor decomposition', (2, 3, 4), 'N-dimensional tensor'),
    'parafac2': _DenoisingMethodContract(
        'tensor decomposition', (3,), '3D stack of matrix slices',
        '4D data must first be unfolded to 3D.'),
    'parafac_power_iteration': _DenoisingMethodContract(
        'tensor decomposition', (2, 3, 4), 'N-dimensional tensor'),
    'partial_tucker': _DenoisingMethodContract(
        'tensor decomposition', (2, 3, 4), 'N-dimensional tensor'),
    'randomised_parafac': _DenoisingMethodContract(
        'tensor decomposition', (2, 3, 4), 'N-dimensional tensor'),
    'robust_pca': _DenoisingMethodContract(
        'tensor decomposition', (2, 3, 4), 'N-dimensional tensor',
        'The default reconstruction is the low-rank component.'),
    'symmetric_parafac_power_iteration': _DenoisingMethodContract(
        'tensor decomposition', (2, 3, 4), 'symmetric N-dimensional tensor',
        'All axis lengths must match and the tensor should be symmetric.'),
    'tensor_ring_als': _DenoisingMethodContract(
        'tensor decomposition', (2, 3, 4), 'N-dimensional tensor'),
    'tensor_ring_als_sampled': _DenoisingMethodContract(
        'tensor decomposition', (2, 3, 4), 'N-dimensional tensor'),
    'tensor_train': _DenoisingMethodContract(
        'tensor decomposition', (2, 3, 4), 'N-dimensional tensor'),
    'tensor_train_matrix': _DenoisingMethodContract(
        'tensor decomposition', (2, 4), 'even-order tensorized matrix',
        'The input order must be even; odd-order 3D unfolding is invalid.'),
    'total_variation': _DenoisingMethodContract(
        'tensor filter', (2, 3, 4), 'N-dimensional image/volume',
        'Integer data are promoted to float without rescaling intensities.'),
    'tucker': _DenoisingMethodContract(
        'tensor decomposition', (2, 3, 4), 'N-dimensional tensor'),
}


class _DenoiseEngine:
    """
    Private dispatcher for applying denoising methods to 2D, 3D, and 4D data.

    ``HyperData.denoise(...)`` is the public API. This helper owns method
    lookup, argument validation, and dimensional routing, including the
    whole-array axis-aware median path, while ``_DenoisingMethods`` stores the
    numerical algorithms.

    Attributes
    ----------
    target_data : ndarray
        The data to be denoised, which can be 2D, 3D, or 4D.

    methods : _DenoisingMethods
        The private numerical-method collection.

    Methods
    -------
    denoise(method_name, target_data=None, **kwargs)
        Applies the specified denoising method to the target data using the provided parameters.
    apply(method, domain='reciprocal', **kwargs)
        Applies a method directly to 2D/3D data, over a selected coordinate
        domain for 4D data, or directly to the whole array when ``domain=None``.

    Notes
    -----
    - The `denoise` method fetches the appropriate method from the _DenoisingMethods instance and applies it to
      the data. It raises a ValueError if the specified method is not found.
    - The `apply` method supports direct 2D/3D denoising and 4D routing through
      real or reciprocal coordinates. Axis-aware median filtering uses one
      whole-array call; other image methods retain slice-wise routing.
    """

    def __init__(self, target_data, *, hyperdata_type=None):
        self.array = np.asarray(target_data)
        self.ndim = self.array.ndim

        self.methods = _DenoisingMethods(hyperdata_type=hyperdata_type)

        self.available_methods = sorted(
            name for name, _ in inspect.getmembers(
                self.methods, predicate=inspect.ismethod,
            ) if not name.startswith('_')
        )

    @staticmethod
    def _parameter_display(param):
        """Return a compact display string for a method parameter."""
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            return f"*{param.name}"
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return f"**{param.name}"
        if param.default is inspect.Parameter.empty:
            return f"{param.name}=..."
        return f"{param.name}={repr(param.default)}"

    @staticmethod
    def _parameter_info(param):
        """Return structured information for one user-facing parameter."""
        default = None
        has_default = param.default is not inspect.Parameter.empty
        if has_default:
            default = param.default

        annotation = None
        if param.annotation is not inspect.Parameter.empty:
            annotation = (
                param.annotation.__name__
                if hasattr(param.annotation, '__name__')
                else str(param.annotation)
            )

        return {
            'name': param.name,
            'kind': param.kind.description,
            'required': (
                not has_default
                and param.kind not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            ),
            'default': default,
            'has_default': has_default,
            'annotation': annotation,
        }

    @staticmethod
    def _doc_summary(docstring):
        """Return the first paragraph of a docstring."""
        if not docstring:
            return ''
        paragraphs = docstring.strip().split('\n\n')
        return ' '.join(paragraph.strip() for paragraph in paragraphs[0].splitlines())

    @staticmethod
    def _four_dimensional_routes(method_name, contract):
        """Describe existing 4D routes without changing the dispatcher."""
        if method_name == 'median':
            return (
                "domain='real' or 'reciprocal': filter the selected axis pair",
                "unfold_domain='both': filter the resulting 2D matrix",
            )

        routes = []
        if 2 in contract.input_ndim:
            routes.append(
                "domain='real' or 'reciprocal': filter each 2D image independently"
            )
            routes.append("unfold_domain='both': denoise a 2D matrix")
        if 3 in contract.input_ndim:
            routes.append(
                "unfold_domain='real' or 'reciprocal': denoise the 3D result"
            )
        if 4 in contract.input_ndim:
            routes.append("domain=None: denoise the full 4D tensor")
        return tuple(routes)

    def _contract_info(self, method_name, method):
        """Combine a stable input contract with current routing and flags."""
        contract = _DENOISING_METHOD_CONTRACTS.get(method_name)
        if contract is None:
            raise RuntimeError(
                f"Denoising method {method_name!r} has no registered input contract."
            )

        signature = inspect.signature(method)
        result_flags = tuple(
            name for name in ('return_decomposition', 'return_errors')
            if name in signature.parameters
        )
        return {
            'family': contract.family,
            'supported_input_ndim': contract.input_ndim,
            'input_layout': contract.input_layout,
            'constraints': contract.constraints,
            'four_dimensional_routes': self._four_dimensional_routes(
                method_name, contract
            ),
            'default_output': (
                'shape-preserving HyperData reconstruction; ndarray with '
                'return_array=True'
            ),
            'optional_result_flags': result_flags,
            'output_caveat': (
                'return_errors=True yields (reconstruction, errors); '
                'return_decomposition=True retains a method-specific payload.'
                if 'return_errors' in result_flags
                else (
                    'return_decomposition=True retains a method-specific payload.'
                    if result_flags else ''
                )
            ),
        }

    def method_info(self, method_name=None, include_doc=True, print_info=False):
        """
        Return signature, input contract, and routing information.

        The first numerical-method argument is the data array supplied by
        ``HyperData.denoise``. It is intentionally excluded from
        ``method_parameters`` because users should not pass it manually.
        The contracts describe the underlying numerical methods; they do not
        alter routing or promise that all algorithms have passed numerical
        validation. Optional result flags retain their existing return types.
        """
        if method_name is None:
            info = {
                'available_methods': tuple(self.available_methods),
                'method_contracts': {
                    name: self._contract_info(name, getattr(self.methods, name))
                    for name in self.available_methods
                },
            }
            if print_info:
                print("Available denoising methods:")
                print(', '.join(self.available_methods))
            return info

        if not isinstance(method_name, str) or not method_name:
            raise ValueError("method_name must be a non-empty string or None.")

        method = getattr(self.methods, method_name, None)
        if method_name not in self.available_methods or method is None:
            raise ValueError(
                f"No such method '{method_name}'. Available methods are: "
                f"{', '.join(self.available_methods)}"
            )

        signature = inspect.signature(method)
        parameters = list(signature.parameters.values())

        injected_data_parameter = None
        method_parameters = parameters
        if parameters and parameters[0].kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            injected_data_parameter = parameters[0].name
            method_parameters = parameters[1:]

        parameter_info = [
            self._parameter_info(param)
            for param in method_parameters
        ]
        required = [
            param['name']
            for param in parameter_info
            if param['required']
        ]
        optional = [
            param
            for param in parameter_info
            if not param['required']
        ]
        method_kwargs_display = ', '.join(
            self._parameter_display(param)
            for param in method_parameters
        )
        example_call = f"my_dataset.denoise(method='{method_name}'"
        if method_kwargs_display:
            example_call += f", {method_kwargs_display}"
        example_call += ")"

        docstring = inspect.getdoc(method) or ''
        info = {
            'method': method_name,
            'full_signature': f"{method_name}{signature}",
            'method_kwargs_signature': f"{method_name}({method_kwargs_display})",
            'example_call': example_call,
            'injected_data_parameter': injected_data_parameter,
            'required_parameters': required,
            'optional_parameters': optional,
            'parameters': parameter_info,
            'doc_summary': self._doc_summary(docstring),
        }
        info.update(self._contract_info(method_name, method))
        if include_doc:
            info['docstring'] = docstring

        if print_info:
            print(f"Method: {method_name}")
            if info['doc_summary']:
                print(info['doc_summary'])
            print()
            print("Use:")
            print(f"  {example_call}")
            print()
            print(
                "Data input supplied automatically by HyperData.denoise: "
                f"{injected_data_parameter}"
            )
            print(
                f"Input: {info['input_layout']} "
                f"(ndim {', '.join(map(str, info['supported_input_ndim']))})"
            )
            print(f"Default output: {info['default_output']}")
            if info['constraints']:
                print(f"Constraints: {info['constraints']}")
            print("For 4D data:")
            for route in info['four_dimensional_routes']:
                print(f"  - {route}")
            if info['optional_result_flags']:
                print(
                    "Optional result flags: "
                    + ', '.join(info['optional_result_flags'])
                )
                print(f"  {info['output_caveat']}")
            if required:
                print("Required method arguments:")
                for name in required:
                    print(f"  - {name}")
            else:
                print("Required method arguments: none")

            if optional:
                print("Optional method arguments:")
                for param in optional:
                    if param['kind'] == 'variadic keyword':
                        print(f"  - **{param['name']}")
                    elif param['kind'] == 'variadic positional':
                        print(f"  - *{param['name']}")
                    elif param['has_default']:
                        print(f"  - {param['name']}={repr(param['default'])}")
                    else:
                        print(f"  - {param['name']}")

        return info

    def _resolve_method(self, method_name):
        """Return only public numerical methods, never private helpers."""
        if not isinstance(method_name, str) or not method_name:
            raise ValueError("method must be a non-empty string.")
        if method_name not in self.available_methods:
            raise ValueError(
                f"No such method '{method_name}'. Available methods are: "
                f"{', '.join(self.available_methods)}"
            )
        return getattr(self.methods, method_name)

    def _validate_call(self, method_name, method, target_data, kwargs):
        """Check the method contract and arguments before numerical work."""
        contract = _DENOISING_METHOD_CONTRACTS[method_name]
        if target_data.ndim not in contract.input_ndim:
            supported = ', '.join(f'{ndim}D' for ndim in contract.input_ndim)
            raise ValueError(
                f"Method '{method_name}' expects {contract.input_layout} "
                f"({supported}); got shape {target_data.shape}. "
                "See denoising_method_info for supported 4D routing."
            )
        if (
            method_name == 'symmetric_parafac_power_iteration'
            and len(set(target_data.shape)) != 1
        ):
            raise ValueError(
                "symmetric_parafac_power_iteration requires equal axis "
                f"lengths; got shape {target_data.shape}."
            )
        try:
            inspect.signature(method).bind(target_data, **kwargs)
        except TypeError as exc:
            raise TypeError(
                f"Invalid arguments for denoising method '{method_name}': {exc}"
            ) from exc

    @staticmethod
    def require_reconstruction(result, expected_shape, *, method,
                               context='denoising', hyperdata_type=()):
        """Unwrap and validate a reconstruction before it enters a dataset."""
        if isinstance(result, hyperdata_type):
            result = result.array
        if not isinstance(result, np.ndarray):
            raise TypeError(
                f"Method '{method}' returned {type(result).__name__} during "
                f"{context}; expected an ndarray or HyperData reconstruction. "
                "Optional decomposition results cannot be routed as images."
            )
        expected_shape = tuple(expected_shape)
        if result.shape != expected_shape:
            raise ValueError(
                f"Method '{method}' changed shape during {context}: "
                f"expected {expected_shape}, got {result.shape}."
            )
        return result

    @staticmethod
    def split_reconstruction_errors(result, method):
        """Separate a reconstruction from its convergence history."""
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError(
                f"Method '{method}' with return_errors=True must return "
                "(reconstruction, errors)."
            )
        reconstruction, errors = result
        if errors is None:
            raise ValueError(
                f"Method '{method}' did not provide convergence errors."
            )
        return reconstruction, errors

    def denoise(self, method_name, target_data=None, **kwargs):
        """Invoke one method on a whole 2D, 3D, or 4D array."""
        method = self._resolve_method(method_name)
        target_data = self.array if target_data is None else np.asarray(target_data)
        self._validate_call(method_name, method, target_data, kwargs)
        return method(target_data, **kwargs)

    def _apply_to_slices(self, method_name, domain, kwargs):
        """Apply a 2D method to 4D slices without truncating its output dtype."""
        if kwargs.get('return_decomposition', False):
            raise ValueError(
                "Slice-wise 4D denoising requires one image per slice; use "
                "return_decomposition=False, or domain=None for a whole-tensor "
                "decomposition."
            )
        if kwargs.get('return_errors', False):
            raise ValueError(
                "Slice-wise 4D denoising has no single convergence history; "
                "use unfold_domain or domain=None with return_errors=True."
            )

        method = self._resolve_method(method_name)
        if domain == 'real':
            index_shape = self.array.shape[2:4]
            expected_shape = self.array.shape[:2]
            description = 'Filtering real-space images'

            def get_slice(index):
                return self.array[:, :, index[0], index[1]]

            def put_slice(output, index, values):
                output[:, :, index[0], index[1]] = values
        else:
            index_shape = self.array.shape[:2]
            expected_shape = self.array.shape[2:4]
            description = 'Filtering diffraction patterns'

            def get_slice(index):
                return self.array[index[0], index[1], :, :]

            def put_slice(output, index, values):
                output[index[0], index[1], :, :] = values

        if 0 in index_shape or 0 in expected_shape:
            raise ValueError("Cannot denoise 4D data with an empty axis.")
        first_index = (0, 0)
        first_slice = get_slice(first_index)
        self._validate_call(method_name, method, first_slice, kwargs)
        first = self.require_reconstruction(
            method(first_slice, **kwargs), expected_shape,
            method=method_name, context='slice-wise 4D denoising',
            hyperdata_type=self.methods._hyperdata_type or (),
        )
        output = np.empty(self.array.shape, dtype=first.dtype)
        put_slice(output, first_index, first)

        indices = np.ndindex(index_shape)
        next(indices)
        for index in tqdm(indices, total=prod(index_shape) - 1, desc=description):
            values = self.require_reconstruction(
                method(get_slice(index), **kwargs), expected_shape,
                method=method_name, context='slice-wise 4D denoising',
                hyperdata_type=self.methods._hyperdata_type or (),
            )
            if not np.can_cast(values.dtype, output.dtype, casting='safe'):
                raise TypeError(
                    f"Method '{method_name}' returned inconsistent slice dtypes "
                    f"({output.dtype} then {values.dtype}); refusing a lossy cast."
                )
            put_slice(output, index, values)
        return output

    def apply(self, method, domain='reciprocal', **kwargs):
        """Route 4D domains; apply 2D/3D arrays directly as before."""
        self._resolve_method(method)
        if domain is None:
            return self.denoise(method, **kwargs)

        if not isinstance(domain, str):
            raise ValueError("domain must be a string or None.")
        domain = domain.lower()
        if domain in ('real_space', 'r'):
            domain = 'real'
        elif domain in ('reciprocal_space', 'k', 'k_space'):
            domain = 'reciprocal'
        if domain not in ('real', 'reciprocal'):
            raise ValueError("domain must be 'real', 'reciprocal', or None.")

        if self.ndim in (2, 3):
            return self.denoise(method, **kwargs)
        if self.ndim != 4:
            raise ValueError("Unsupported image dimensionality; expected 2D, 3D, or 4D.")

        if method == 'median':
            if 'axes' in kwargs:
                raise TypeError(
                    "For 4D median denoising, axes are selected by domain; "
                    "do not pass axes explicitly."
                )
            axes = (0, 1) if domain == 'real' else (2, 3)
            return self.denoise(method, axes=axes, **kwargs)

        return self._apply_to_slices(method, domain, kwargs)
