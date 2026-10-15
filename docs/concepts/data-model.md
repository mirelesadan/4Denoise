# Data Model

`HyperData` treats the final two axes as an image or diffraction pattern.
For a 4D-STEM scan, the convention is `(Ry, Rx, Ky, Kx)`: two real-space
scan coordinates followed by two reciprocal-space detector coordinates.

| Input | `scan_shape` | `pattern_shape` | `real_shape` |
| --- | --- | --- | --- |
| 2D image | `()` | `(Ky, Kx)` | `None` |
| 3D stack | `(N,)` | `(Ky, Kx)` | `None` |
| 4D scan | `(Ry, Rx)` | `(Ky, Kx)` | `(Ry, Rx)` |

A 3D stack does not retain a 2D scan layout unless you provide it when
reshaping or undoing a row-major unfold. `real_units` and
`real_conv_factor` describe the scan axes; `reciprocal_units` and
`reciprocal_conv_factor` describe the detector axes. Conversion factors
are units per pixel. `real_conv_factor` can also be `(y, x)` for unequal
scan-axis steps, and `real_origin` locates scan pixel `(0, 0)`.

Scan axes and detector axes are not necessarily physically aligned. A strain
map's `(Ry, Rx)` layout locates the measurement, while `exx` and `eyy` describe
directions in the fitted tensor basis. The default
`get_strains(output_frame="reciprocal", ang=0)` reports detector kx and ky.
Use `output_frame="real"` to report Cartesian real-space components using
stored `rq_calibration`, including its rotation and optional reflection.
The result records its output frame and the calibration used. Converting
the result to `RealSpace` displays those components at the original scan
positions without applying another tensor rotation. A recorded acquisition
scan rotation does not automatically establish R/Q calibration.
See [peaks and strain](../guides/peaks-and-strain.md) for this distinction.

Value-only methods generally return a new object rather than altering the
input. Geometric operations can change calibration or invalidate metadata
that no longer describes the result. Inspect the method's return value and
metadata before chaining a shape-changing operation.
