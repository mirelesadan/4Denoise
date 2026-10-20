# Visualize a 4D Scan

For a runnable hexagonal teaching dataset and quantitative detector checks,
open [Tutorial 2: Visualization and virtual detectors](../tutorials/visualization-and-virtual-detectors.md).
It needs no external acquisition or output from another notebook.

`HyperData` holds the full scan. `get_dp(...)` returns a
`ReciprocalSpace` view of one or more diffraction patterns, while
`virtual_image(...)` returns a `RealSpace` image integrated through a
virtual detector. The final two array axes are always the pattern axes.
For native or optional RosettaSciIO disk-backed inputs, use
`HyperData.open(...)` and its `get_dp`/`virtual_image` methods instead.
They return independent in-memory patterns/images with the same display
interfaces; they do not load the complete diffraction tensor.

## Diffraction Patterns and Peak Overlays

```python
from fourdenoise import HyperData

data = HyperData("mini_dataset_binned.npy")
dp = data.get_dp(y=7, x=6)
dp.show(
    axis_units="pixels", grid=True, grid_ticks=(5, 5),
    scale_bar=20, scale_bar_color="white",
)
```

To show measured peak positions, pass `coords` in `(y, x)` pixel order:

```python
dp.show(coords=reference_peaks, c="white", s=25)
```

`mode="polar"` displays a circular radius/angle projection without changing
the stored data. For an actual transformed array, use
`HyperData.to_polar(...)`; the returned object's metadata records the polar
geometry, and `to_cartesian(...)` remaps it back by interpolation.

## Aggregate Diffraction Patterns Without a Large Selected Copy

Mean, population standard deviation (`ddof=0`), minimum and maximum are
calculated over bounded scan blocks. A Boolean `mask` of shape `(Ry, Rx)`
selects which patterns contribute, without gathering the complete selected
stack first. The returned `ReciprocalSpace` object retains detector calibration
and polar/cepstral geometry. For example, if `clusters` is a scan-shaped label map:

```python
mean_dp = data.get_dp(mask=clusters == 0, operation="mean", working_memory_mb=64)
mean_dp.show()
std_dp = data.get_dp(operation="std", working_memory_mb=64)
```

The same reductions support half-open scan regions with `y=(start, stop)`
and `x=(start, stop)`, or the entire scan when no selection is supplied.
Use `selection_units="calibrated"` for physical scan limits. A mask takes
precedence over y/x, and empty or non-Boolean masks raise errors.

Both source and in-memory HyperData selections default to pixel indices.
Set `selection_units="calibrated"` for physical limits or `"auto"` to use
calibration when present; `axis_units` controls display independently.
Sources support mean/std/min/max, but reject exact median rather than
gathering the full selected stack. See [disk-backed sources](io-and-calibration.md#disk-backed-sources-and-large-files).

`working_memory_mb=None` automatically uses 256 MiB (one MiB is `1024**2`
bytes). This budget accounts conservatively for explicit processing arrays,
including dtype conversion and statistic buffers; it does not cap total RAM
or include the resident source, caller's mask, allocator overhead or file
caches. A budget too small for one pattern and its buffers raises an error
showing the minimum required size. Integer means/stds return floating-point
arrays; floating input retains its output dtype, with accumulation using at
least float64 precision. No NaN filtering or intensity clipping is implicit.

`operation="median"` remains exact and may allocate a large selected array.
It is not bounded-memory; explicit `working_memory_mb` is rejected for median
and random selection. Two scalar scan indices return one pattern directly
without reduction buffers. The heuristic flat-field operation and its
`percentile`/`show_mask` options have been removed; select a scientifically
justified scan mask explicitly instead.

## Virtual Detectors

Choose exactly one detector specification: an annulus, one or more disks,
or a pixel mask. The detector's numeric inputs use `detector_units`.

```python
virtual = data.virtual_image(
    annulus=(37, 50), detector_units="pixels", show=False,
)
virtual.show(axes=False, cmap="magma")
```

Use `plot_mask=True` to preview the detector over a mean diffraction
pattern. `ring_color`, `ring_alpha`, `mask_cmap`, `mask_vmin`, and
`mask_vmax` control that preview. For an arbitrary detector, supply a
Boolean or weighted `mask` with shape `(Ky, Kx)`.

The same call works inside a source context:

```python
with HyperData.open("mini_dataset_binned.npy") as source:
    virtual = source.virtual_image(
        annulus=(37, 50), detector_units="pixels", show=False,
        working_memory_mb=64,
    )

virtual.show(axis_units="auto", axes=False, cmap="magma")
```

Only the complete 2D scan image is retained; the 4D diffraction data stays
on disk. A detector preview or `return_detector=True` also accumulates the
mean pattern in the same pass. The latter returns `(image, masked_detector)`;
the preview displays an unmasked mean with the detector overlay. Without
either option, no mean is computed. Weights are summed without normalization,
overlapping disks count each pixel once, and complex amplitudes must first be
converted to real-valued intensities by an appropriate physical procedure.

Polar virtual detectors use saved radial/angular geometry, including wrapped
`theta_range`; Cartesian disks are not valid on polar data. Source virtual
images require a 4D scan, because a bare 3D stack has no 2D scan layout.
See the [streaming tutorial](../getting-started/streaming.md) for an executable
example and the [I/O guide](io-and-calibration.md#streamed-virtual-images)
for budgets and return types.

## Units and Scale Bars

When `real_units` and `real_conv_factor` are defined, `RealSpace.show`
accepts `axis_units="calibrated"` and `scale_bar` in those units. Likewise,
`ReciprocalSpace.show` uses reciprocal calibration for its axes and scale
bar. `axis_units="auto"` selects calibration when available and otherwise
uses pixels. Use `axis_units="pixels"` to request pixel labels explicitly.
Detector geometry and displayed scan axes are independent:
`virtual_image(detector_units=..., axis_units=...)` controls them separately.

See the [data model](../concepts/data-model.md) for the axis convention.

## Compare Real and Diffraction Orientations

Use `compare_rq` with a real-space image and a diffraction-space shadow image
showing the same specimen features. The real image stays fixed while the
diffraction image can rotate, translate, scale, or reflect. These are manual
preview controls; matching an ordinary Bragg pattern to a STEM image by
appearance does not establish an orientation calibration.

In Jupyter, install the optional backend with `%pip install ipympl` (or install
the package with `python -m pip install -e ".[interactive]"`). Restart the
kernel if this is the first installation, then run this before making the
viewer:

```python
%matplotlib widget
```

In Spyder, select an interactive Qt graphics backend. An inline notebook
image is static; the viewer warns if it detects that backend. Retain the
returned viewer to adjust it programmatically or export it later.

```python
# real_image is a RealSpace image or a 2D array.
# shadow_image is a Cartesian ReciprocalSpace image or a 2D array.
viewer = data.compare_rq(
    real_image=real_image,
    reciprocal_image=shadow_image,
    layout="overlay",              # also "side_by_side"
    real_alpha=0.7,
    reciprocal_alpha=0.7,
    real_show_kwargs={"cmap": "gray", "axis_units": "auto"},
    reciprocal_show_kwargs={"cmap": "magma", "logScale": False},
)
```

Each image has its own opacity slider. The diffraction rotation also has a
numeric entry box. Translation is `(dy, dx)` in reference-image pixels,
positive down/right. The preview scale is real horizontal pixels per
diffraction pixel; it does not overwrite the physical pixel calibration.
Known unequal scan pixel spacings are respected during rotation.

The pivot uses the diffraction image's `center_beam_metadata`: `center_px`,
then `mean_fit_center_px`, or a calibrated center converted to pixels. A raw
array matching the dataset's pattern shape inherits its beam metadata. Without
a stored center, the pivot is the exact fractional midpoint
`((Ky - 1) / 2, (Kx - 1) / 2)`. Stale metadata is rejected. The pivot marker
and its source are visible in the viewer. Initially it is placed at the
real-image midpoint; translation moves that anchor, and subsequent rotations
leave it fixed.

The displayed **diffraction correction** is positive counterclockwise. The
stored `RQCalibration` maps real directions to detector directions, so it
uses the inverse transform. Without reflection these two angles have opposite
signs. With reflection, the full matrix is inverted; the GUI mirrors the
diffraction image about its x or y axis before rotating it.

```python
viewer.set_parameters(rotation_deg=12.5, translation=(0.5, -2.0))
calibration = viewer.apply()        # same as Apply calibration in the GUI
print(calibration.rotation_deg, calibration.mirror_axis)
print(calibration.matrix)          # real -> reciprocal Cartesian (x, y)
viewer.export("rq_comparison.png") # image panels without GUI controls
data.save("calibrated.4denoise")
```

Apply stores only the orientation. The arrays are unchanged, and preview
scale, position and opacity do not become physical calibrations. Save/load,
copy, and value-only processing preserve the orientation; flips and
`rotate_dps` update its coordinate frame. If an angle is already known, use
`data.set_rq_calibration(rotation_deg=..., mirror_axis=...)` before opening
the viewer. This setter uses the real-to-detector convention.

The viewer accepts the intensity-related `.show()` options listed in the
[API reference](../reference/hyperdata.rst). Its axes and optional scale bar
use the reference image's units, including in side-by-side mode, where the
second panel shows the transformed diffraction image in that same frame.
Reset returns to the opening preview settings; it does not undo an already
applied calibration. Close the viewer with `viewer.close()` when finished.
