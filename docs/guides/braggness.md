# Concentrated Sharp Diffraction Contrast

`HyperData.get_braggness()` measures how much sharp positive contrast is
concentrated in the brightest residual pixels of a detector annulus. It does
not require peak coordinates and does not compute an FFT or identify a phase.

## Definition

For each complete Cartesian diffraction pattern `D`:

1. Smooth `D` with a Gaussian (`reflect` boundaries, truncation at four sigma).
2. Calculate `R = maximum(D - Gaussian(D), 0)`.
3. Select pixels with `inner <= radius < outer` around `(ky, kx)`.
4. Sum the largest `top_pixels` selected residuals.
5. Divide by `max(sum(D inside the annulus), denominator_floor)`.

Explicit center, radius, and sigma settings are in **detector pixels**,
independent of reciprocal calibration. The Gaussian operates on the full
pattern before annulus selection and never mixes neighboring scan positions.
The outer bound is exclusive; these conventions must not be confused with
`virtual_image`.

Omitting `center` uses the dataset's current beam metadata, in this order:
`center_px`, `mean_fit_center_px`, then `center_calibrated` converted to pixels.
Alignment records this metadata on the **returned aligned object**. Without a
stored center, the fallback is the subpixel image center. Explicit `center`
always takes priority; stale or malformed metadata raises an error. The
resolved pixel pair and its source are recorded in `result.metadata`.

## Example

```python
import numpy as np
from fourdenoise import HyperData

# A small synthetic 4D scan for exploring the API, not a phase standard.
patterns = np.full((3, 4, 9, 13), 10, dtype=np.float32)
patterns[:, :, 4, 8] += 100
data = HyperData(patterns, real_units="nm", real_conv_factor=0.5,
                 real_origin=(10, 20))
result = data.get_braggness(
    annulus=(1, 5), center=(4, 6), sigma=2, top_pixels=10,
    denominator_floor=1, nonfinite="raise",
)

B = result.score                  # float32, shape (3, 4)
valid = result.valid_mask          # boolean, shape (3, 4)
print(result.metadata)            # JSON-serializable settings and counts
result.as_real_space().show(cmap="viridis", vmin=0)
```

A 2D input produces a scalar array of shape `()`; a 3D pattern stack produces
`(N,)`. Only a 4D scan has a 2D score map that can be represented by
`as_real_space()`. That view retains real-space units, signed per-axis spacing,
and origin without transposing the scan.

To measure aligned data using the alignment center, omit `center`:

```python
aligned = data.alignment(r_center=3)
result = aligned.get_braggness(annulus=(1, 5), top_pixels=10)
print(result.metadata["center"], result.metadata["center_source"])
```

This evaluates the **aligned intensities**. Do not transfer aligned-object
center metadata to raw data whose detector geometry/orientation is different.

## Raw 4D-TOMO Reproduction

The reference workflow uses **stored, unflipped, unaligned intensities**, not
its main detector-y-flipped and COM-aligned object. Wrap an independent
read-only view to reproduce it without loading or copying the whole acquisition:

```python
raw = np.memmap("scan_x256_y256.raw", dtype="<f4", mode="r",
                shape=(256, 256, 130, 128))
data = HyperData(raw[:, :, :128, :])  # Trim metadata rows, retain orientation.
scan_mask = np.ones(data.scan_shape, dtype=bool)
# Only for the alpha=-20 acquisition, whose source pattern is known all-NaN:
scan_mask[120, 50] = False

result = data.get_braggness(
    annulus=(18, 50), center=(64, 64), sigma=2, top_pixels=50,
    denominator_floor=1, scan_mask=scan_mask, nonfinite="raise",
    progress=True,
)
```

The explicit center `(64, 64)` differs from the 128x128 image center
`(63.5, 63.5)`. Flipping or aligning a pattern changes the input measurement;
do not assume exact flip invariance with the same explicit off-center detector.
Float32 input retains float32 Gaussian/subtraction/residual calculations.
Both sums accumulate in float64, and scores are float32, matching the notebook.
Float16 is promoted to float32; integer and other float inputs use float64
temporaries. Extended precision is reduced, and very large integers can lose
precision during conversion. See the API docstring for numerical-failure policy.

Memory use beyond the source consists of score/validity scan maps, a
detector-sized mask, and per-pattern temporaries. Trimmed and flipped views
are indexed directly, without flattening/copying the acquisition. The source,
its metadata, and its backing file remain unchanged.

## Invalid Positions and Interpretation

`scan_mask=True` means evaluate. False positions stay NaN and invalid without
inspection or repair. `nonfinite="raise"` reports the scan index of an evaluated
pattern containing any NaN/infinity, even outside the annulus. `"omit"`
invalidates the entire pattern instead. Numerical overflow follows the same
policy. Skipped, nonfinite, and numerically failed counts are in `metadata`.

Keep invalid positions excluded during **all** subsequent mask operations:

```python
candidate = result.valid_mask & (result.score > 0.012)
# If morphology or a hull creates refined_mask, exclude invalid positions again:
refined_mask = candidate.copy()  # Replace with your own refinement.
refined_mask &= result.valid_mask
```

Higher values indicate more concentrated sharp contrast for the chosen
configuration. Noise, exposure/intensity, denominator flooring, detector
geometry, sigma, annulus, and retained-pixel count affect this index. Compare
data using consistent settings and preprocessing. It is dimensionless but
not necessarily bounded by one, and is **not** a crystalline volume fraction,
phase identity, or orientation confidence. Those require additional evidence.

See the [HyperData API](../reference/hyperdata.rst) and
[result API](../reference/views-and-results.rst) for full parameter contracts.
