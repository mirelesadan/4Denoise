# Tutorial 5: Peak Analysis and Braggness

**[Read the complete tutorial online, with code and figures](notebooks/05_peak_analysis_and_braggness.ipynb).**

```{toctree}
:hidden:

notebooks/05_peak_analysis_and_braggness
```

{download}`Download the notebook <../../tutorials/05_peak_analysis_and_braggness.ipynb>`.
In a checkout, open `tutorials/05_peak_analysis_and_braggness.ipynb` in Jupyter.
Core dependencies and an IPython kernel suffice; no microscope file is needed.

## What You Will Learn

- Detect peaks independently per pattern without a shared reference or
  imposing a common peak count or rotational symmetry.
- Keep ragged detections, correlation scores, refined centers, and integrated
  intensities aligned within each pattern.
- Refine subpixel centers, audit a raw circular-window sum, and explicitly
  request a local-background correction.
- Compare thresholds on weak-spot and diffuse patterns, including false
  detections and missed peaks rather than only ideal overlays.
- Check localization against registered reference coordinates when available.
- Audit the Braggness formula and interpret its validity mask and score map
  separately from detected peaks or intensity-window sums.
- Reuse an alignment center from current beam metadata on a separate copy,
  without silently changing the original known-position analysis.
- Export small ragged tables and settings as numeric NPZ data without pickle.

## Data and Return Types

The default `(10, 15, 64, 64)` float32 fixture contains assigned strong six-spot,
weak four-spot, and diffuse-ring regions, a central Gaussian, and reproducible
Poisson noise. Intensity scales and nm/mrad calibration are invented. These
are teaching image features, not simulated scattering or verified phases.

`HyperData.get_peaks(return_details=True)` returns nested `[Ry][Rx]` lists of
`PeakDetectionResult`. Their integer `(n_peaks, 2)` coordinates are absolute
`(y, x)` pixels; counts can vary. Refinement and integration return corresponding
ragged lists. Correlation scores are not integrated intensities or confidence
probabilities. Symmetry repair is disabled, so no assumed peaks are generated.

The source API's `DiffractionSource.get_peaks()` instead returns a
`PeakScanResult` or writes its own numeric peak file; these are distinct return
contracts. Use the source workflow for bounded input processing or large
collections instead of materializing every tutorial list.

## Independent Measurements

The peak-count map shows how many template maxima passed the chosen rule.
The peak-window sum includes local background and can double-count overlapping
integration windows. Background rings require a suitable local model; corrected
values are not silently clipped.

Braggness uses the original selected intensities, not the detected coordinates:
subtract a full-pattern Gaussian, keep positive residuals, sum the brightest
selected residual pixels, and divide by the floored original annular sum.
An explicit one-pattern calculation checks this formula. Its annulus is
`inner <= radius < outer`. All detector settings remain pixels regardless of
reciprocal calibration. The result is dimensionless, not necessarily bounded
by one, and is not an FFT peak-count metric.

Masked positions stay invalid/NaN rather than being interpreted as zero peaks
or non-crystalline. A valid pattern with zero detections can still have a
nonzero sharp-contrast score. Scores, counts, and intensity maps depend on
noise, exposure, orientation, geometry, and settings; none is a crystalline
volume fraction, phase identity, or orientation confidence.

## Replace the Input

The marked cell accepts a file, numerical array, or existing `HyperData` before
the common analysis. Existing metadata is retained; synthetic units, reference
coordinates, and beam parameters are not transferred. An external input needs
an explicitly inspected `analysis_annulus_px`. Optional reference coordinates
must already match its detector geometry. Without them, localization-accuracy
checks are skipped. Review all template, refinement, integration, background,
and Braggness settings for that acquisition.

The lesson uses finite nonnegative intensities at evaluated positions.
The Boolean scan mask can exclude unusable patterns. The default synthetic
mask deliberately skips `(0, 0)`; external inputs do not inherit this exclusion.
Native loads are eager, in-memory inputs are copied, and the 64-MiB size guard
does not cap process RAM or an earlier eager file load. See
[using your own data](using-your-own-data.md).

## Alignment and Reversibility

Omitting the center uses valid current beam metadata before the image midpoint.
The separate alignment demonstration is enabled for the synthetic fixture only
by default. External data require opt-in and a chosen central radius. It scores
an unblocked aligned copy with `center` omitted, verifies the metadata source,
and illustrates direct-beam blocking from the same metadata.

COM's stored radius is the chosen central window, not an independently fitted
physical disk edge. Interpolation changes intensities, so aligned scores are a
different measurement. The original peak analysis and known-position checks
are not silently rerun with untransformed references on that copy.

## Run It

From the repository root with the `docs-test` extra installed:

```bash
python scripts/execute_tutorials.py --notebook 05_peak_analysis_and_braggness.ipynb
```

The clean source is not overwritten; executed reports go to
`docs/_build/tutorials/`. Temporary output is removed in the final cell.
Synthetic-only tolerances check this fixture's localization, not an original
microscope acquisition. No package algorithms or research notebooks are edited.

Continue from [Tutorial 4](denoising-and-unfolding.md), or consult
[peaks and strain](../guides/peaks-and-strain.md) and
[the Braggness definition](../guides/braggness.md) for exact contracts.
