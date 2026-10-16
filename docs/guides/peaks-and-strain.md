# Peaks and Strain Results

The [DEMO notebook](https://github.com/mirelesadan/4Denoise/blob/main/DEMO_exp_4dstem_ripple_processing.ipynb) shows
one verified path from a reference diffraction pattern through peak
detection, center refinement, intensity integration, and a relative strain
map. Peak-selection thresholds and reference regions are data-dependent;
the [experimental case study](../case-studies/3d-strain-mapping.md) uses
different choices.

For a shared reference, detect spots once on an averaged diffraction pattern
and pass those `(y, x)` pixel positions to `get_centers`. Alternatively,
`HyperData.get_peaks(...)` detects each pattern independently and returns
ragged peak lists when counts vary. Keep the peak coordinate order and units
consistent when supplying `ref_centers` or `intensity_array` to a strain fit.

```python
reference_dp = data.get_dp(operation="mean")
reference_peaks = reference_dp.get_peaks(
    radius=3, min_distance=7, threshold_rel=0.1, r_range=(10, 61),
)
measured = data.get_centers(r=5, ref_coords=reference_peaks)
```

These thresholds illustrate the included mini dataset, not general detector
defaults. Inspect an overlay before fitting strain and adjust the peak set
for your material and calibration.

`HyperData.get_strains(centers=measured, ref_centers=reference)` returns a
`StrainResult` with named maps and fit diagnostics:

```python
result = data.get_strains(centers=measured, ref_centers=reference)
result.as_real_space("exx").show()
result.as_real_space("relative_fit_rmse").show()
valid = result.valid_mask
```

`exx`, `eyy`, and `exy` are dimensionless. `erot` is in radians.

**Tensor directions are separate from map positions.** Peak coordinates have
order `(ky, kx)`. The default `output_frame="reciprocal"`, at `ang=0`, assigns
`exx` to detector kx and `eyy` to detector ky. A `(Ry, Rx)` map's location in
the scan does not make those tensor directions scan-aligned.

To report strain along real-space directions, establish the real-to-detector
orientation and store it on the dataset. You can use `compare_rq(...).apply()`
after adjusting the interactive comparison, or set a known calibration:

```python
data.set_rq_calibration(rotation_deg=verified_angle, mirror_axis=None)
result = data.get_strains(
    centers=measured,
    ref_centers=reference,
    output_frame="real",
)
result.as_real_space("exx").show()
print(result.output_frame)
print(result.rq_calibration)
```

`verified_angle` must describe the counterclockwise real-to-detector rotation
in Cartesian image axes (x right, y up). Set `mirror_axis="x"` or `"y"` when
the real frame is reflected about that axis before rotation. Recorded
acquisition scan rotation alone does not establish this relationship.

`output_frame="real"` reads the stored calibration and uses its full rotation
and reflection matrix. `exx` and `eyy` then follow real-space x and y, and
`exy` is the Cartesian off-diagonal component of `I - U`, where `U` is the
polar stretch. `erot` is positive counterclockwise. Pixel peaks have downward
positive rows, while calibrated peaks have upward positive ky; both are
converted to the same Cartesian reporting frame. Reflections can reverse the
rotation sign relative to detector-frame output. Map locations, match counts,
and fit errors are retained. Real output requires stored `rq_calibration`.

For a manual detector-basis correction, use
`get_strains(..., ang=angle_degrees, output_frame="reciprocal")`. For example,
`ang=90` exchanges the normal components and reverses the shear sign without
moving map pixels. A nonzero `ang` cannot be combined with `output_frame="real"`.
With pixel peaks and a rotation-only calibration, real output is equivalent
to the corresponding manual angle; using metadata also handles reflections
and the different handedness of calibrated peak coordinates.

`result.output_basis_matrix` maps centered input `(ky, kx)` vectors to the
reporting coordinates. `result.rq_calibration` records the calibration used
for real output; `result.basis_angle_deg` records only the manual `ang` and
is zero for real output. Fitted transforms and translations in `diagnostics`
remain in the supplied peak-coordinate frame. `as_real_space()` displays
the already-selected components without applying another basis conversion.

`fit_rmse` measures residual peak-position error in the supplied peak
coordinate units; `relative_fit_rmse` divides that error by the RMS
reference-peak radius. Neither number is a confidence probability. Check
`match_counts`, `outlier_counts`, and `valid_mask` as well, especially
when only a few peaks are fitted. A 2D map inherits scan calibration from
the source 4D data when its shape matches that scan.

Existing four-value unpacking and numeric indexing still work. With
`return_transform=True`, transform and peak-rejection diagnostics are
available in `result.diagnostics` and as `result[4]`.

## Local Peak Pairing and Failure Diagnostics

The default matcher uses one-to-one candidates within a neighborhood of each
reference peak, not centroids of differently populated peak sets. Extra,
missing, or distant peaks can remain unmatched. `match_radius` uses the same
units as the supplied peaks; None selects 45% of the median nearest-reference
peak spacing. For a large misorientation, supply an initial guess:

```python
result = data.get_strains(
    centers=measured, ref_centers=reference, center=beam_center,
    initial_rotation=rotation_angles, match_radius=0.15,
    peak_units="calibrated", return_transform=True,
)
```

Angles are Cartesian counterclockwise degrees, either scalar or scan-shaped.
`initial_transform` instead accepts global/scan-shaped (2, 2) matrices in
the input (ky, kx) order. Guesses guide matching only: the full fitted transform
and its rotation are retained, not just a residual rotation. Stored current
beam centers take precedence over historical pre-alignment mean-fit centers.

`fit_translation=True` enables translation consensus that does not assume
equal peak counts. An explicit `initial_translation` may be supplied in peak
units. `ambiguity_ratio` optionally rejects close competing alternatives;
smaller values are stricter. The reported pair-confidence margin is geometric,
not a calibrated probability. Intensity weights remain attached to original
measured-peak indices through cleaning, filtering and assignment.

Inspect `result.fit_status` for masked locations, insufficient peaks/matches,
no candidates, ambiguity rejection, ill-conditioned fits, or success.
`condition_numbers` reports fitted design conditioning when available.
With `return_transform=True`, `diagnostics['matching_information'][index]`
contains original matched/unmatched indices, candidate counts, pair distances,
confidence margins and the initial transform/translation actually used.
Failed fits remain NaN; they are not silently converted to zero strain.

Use `match_peaks="ordered"` only for known correspondences. Explicit
`"hungarian"` allows unrestricted assignment when no gate is specified; it
can force unrelated peaks into pairs and is not the recommended default.
