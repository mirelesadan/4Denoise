"""Run the documented native-source mini-data workflow without research files.

From the repository root, run ``python scripts/streaming_example.py``.
Outputs are temporary unless ``--output DIRECTORY`` is supplied. Existing
destinations are never overwritten. RosettaSciIO is not required.
"""

import argparse
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def run_example(data_path, output_dir):
    """Return compact views, peaks, and paths after closing every source.

    Input is a nonempty 4D NumPy/HDF5/HyperData source. The documented dataset
    is the included (15, 13, 128, 128) scan. Detector settings are mini-data
    choices in pixels, not general scientific defaults. Interpolated polar
    and Cartesian outputs are not an exact round trip of the original array.
    """
    from fourdenoise import HyperData, PeakScanResult

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with HyperData.open(data_path) as source:
        if source.ndim != 4:
            raise ValueError("This tutorial requires a 4D (Ry, Rx, Ky, Kx) scan.")
        original_shape = source.shape
        print(f"Source: {source.shape}, {source.dtype}, backend={source.backend}")
        mean_dp = source.get_dp(operation="mean", working_memory_mb=64)
        image = source.virtual_image(
            annulus=(37, 50), detector_units="pixels",
            working_memory_mb=64, show=False,
        )
        peak_path = source.get_peaks(
            radius=3, min_distance=7, trench_width=2,
            threshold_abs=None, threshold_rel=0.1, r_range=(10, 61),
            intensity_radius=3, coords_units="pixels",
            destination=output_dir / "peaks.h5",
            working_memory_mb=64, progress=False,
        )
        polar_path = source.to_polar(
            output_shape=(64, 180), destination=output_dir / "polar.4denoise",
            working_memory_mb=64, progress=False,
        )

    with HyperData.open(polar_path) as polar_source:
        assert polar_source.shape == (*original_shape[:2], 64, 180)
        cartesian_path = polar_source.to_cartesian(
            output_shape="original", destination=output_dir / "cartesian.4denoise",
            working_memory_mb=64, progress=False,
        )

    # Explicitly load the compact peak collection, not the diffraction tensor.
    peaks = PeakScanResult.load(peak_path, max_memory_mb=64)
    assert peaks.scan_shape == original_shape[:2]
    assert peaks.processed_mask.all()
    with HyperData.open(cartesian_path) as restored_source:
        assert restored_source.shape == original_shape

    print(f"Observed peaks: {int(peaks.observed_peak_counts.sum())}")
    print(f"Virtual image: {image.shape}; peak scan: {peaks.scan_shape}")
    return dict(image=image, mean_dp=mean_dp, peaks=peaks,
                peak_path=peak_path, polar_path=polar_path,
                cartesian_path=cartesian_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "mini_dataset_binned.npy",
                        help="Input 4D file (default: included mini dataset).")
    parser.add_argument("--output", type=Path,
                        help="Keep output files in this directory; use fresh filenames.")
    args = parser.parse_args(argv)
    if not args.data.is_file():
        parser.error(f"Input file does not exist: {args.data}")
    if args.output is None:
        with tempfile.TemporaryDirectory(prefix="4denoise-streaming-") as directory:
            run_example(args.data, directory)
        print("Passed; temporary output files removed. Input data is unchanged.")
    else:
        result = run_example(args.data, args.output)
        for name in ("peak_path", "polar_path", "cartesian_path"):
            print(f"{name}: {result[name]}")
        print("Passed; output files retained. Input data is unchanged.")


if __name__ == "__main__":
    main()
