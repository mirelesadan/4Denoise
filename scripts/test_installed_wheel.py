"""Run the unchanged regression suite against a locally built, installed wheel.

Build with ``python -m pip wheel . --no-deps --wheel-dir wheelhouse``, then run
``python scripts/test_installed_wheel.py wheelhouse``. Runtime dependencies must
already be installed in this interpreter; this runner never downloads them.
"""

import argparse
import importlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


# Check optional simulation modules without importing their optional dependencies.
# Keep this list in sync with tool.setuptools.py-modules in pyproject.toml.
MODULE_NAMES = (
    "fourdenoise",
    "fourdenoise_geometry",
    "fourdenoise_metadata",
    "fourdenoise_rq",
    "fourdenoise_reductions",
    "fourdenoise_resampling",
    "fourdenoise_strain",
    "fourdenoise_streaming",
    "fourdenoise_traversals",
    "fourdenoise_plotting",
    "fourdenoise_io",
    "fourdenoise_sources",
    "fourdenoise_detectors",
    "fourdenoise_rosetta",
    "abTEM_helpers",
    "rippleModel",
    "kin_diff_sim",
)
DISTRIBUTION = "fourdenoise-main"
ROOT = Path(__file__).resolve().parents[1]


def resolve_wheel(path):
    """Accept one wheel or a directory containing exactly one wheel, on any OS."""
    path = Path(path).resolve()
    wheels = sorted(path.glob("*.whl")) if path.is_dir() else [path]
    if len(wheels) != 1 or not wheels[0].is_file() or wheels[0].suffix != ".whl":
        raise ValueError(f"Expected exactly one .whl file at {path}; found {len(wheels)} candidates")
    return wheels[0]


def copy_support(source_root, target):
    """Copy only tests and their support, never checkout package modules."""
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "_build",
                                    ".pytest_cache", ".ipynb_checkpoints")
    for directory in ("tests", "scripts", "docs"):
        shutil.copytree(source_root / directory, target / directory, ignore=ignore)


def assert_installed_origins(target):
    """Fail instead of falling back to an editable install or checkout module."""
    target = Path(target).resolve()
    distribution = importlib.metadata.distribution(DISTRIBUTION)
    if Path(distribution.locate_file("")).resolve() != target:
        raise RuntimeError(f"{DISTRIBUTION} metadata is not from the wheel target: {target}")
    for name in MODULE_NAMES:
        expected = target / f"{name}.py"
        spec = importlib.util.find_spec(name)
        if not expected.is_file() or spec is None or spec.origin is None:
            raise RuntimeError(f"Wheel is missing packaged module {name}: {expected}")
        if Path(spec.origin).resolve() != expected:
            raise RuntimeError(f"{name} resolved outside the installed wheel: {spec.origin}")
        loaded = sys.modules.get(name)
        if loaded is not None and Path(loaded.__file__).resolve() != expected:
            raise RuntimeError(f"{name} was imported outside the installed wheel: {loaded.__file__}")


def run_installed_suite(source_root, *, pattern="test*.py", verbosity=1):
    """This branch runs only in a fresh, isolated interpreter in the target."""
    if not sys.flags.isolated:
        raise RuntimeError("Installed-wheel tests require Python's -I isolated mode")
    target = ROOT
    source_root = Path(source_root).resolve()
    # -I excludes cwd/PYTHONPATH, but an editable .pth can still expose the checkout.
    sys.path[:] = [entry for entry in sys.path
                   if entry and not Path(entry).resolve().is_relative_to(source_root)]
    sys.path.insert(0, str(target))
    importlib.invalidate_caches()
    assert_installed_origins(target)
    module = importlib.import_module("fourdenoise")
    assert_installed_origins(target)
    print(f"Installed-wheel module: {module.__file__}", flush=True)
    suite = unittest.defaultTestLoader.discover(str(target / "tests"), pattern=pattern)
    if suite.countTestCases() == 0:
        raise RuntimeError("Installed-wheel discovery found no tests")
    print(f"Running {suite.countTestCases()} installed-wheel tests", flush=True)
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    assert_installed_origins(target)
    return 0 if result.wasSuccessful() else 1


def test_wheel(wheel, source_root, *, pattern="test*.py", verbosity=1):
    """Install offline into a disposable target, then launch its copied runner."""
    source_root = Path(source_root).resolve()
    wheel = resolve_wheel(wheel)
    # A short OS-temp root also works when these tests run inside an installed
    # wheel: nesting targets under a long checkout exceeds Windows path limits.
    with tempfile.TemporaryDirectory(prefix="4dw-") as directory:
        work = Path(directory)
        target = work / "site-packages"
        scratch = work / "scratch"
        scratch.mkdir()
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env.update({"TMP": str(scratch), "TEMP": str(scratch), "TMPDIR": str(scratch),
                    "MPLCONFIGDIR": str(scratch / "matplotlib"),
                    "NUMBA_CACHE_DIR": str(scratch / "numba")})
        env.setdefault("MPLBACKEND", "Agg")
        subprocess.run(
            [sys.executable, "-I", "-B", "-m", "pip", "--isolated", "install",
             "--no-index", "--no-deps", "--no-compile", "--no-cache-dir",
             "--target", str(target), str(wheel)],
            cwd=work, env=env, check=True,
        )
        copy_support(source_root, target)
        return subprocess.run(
            [sys.executable, "-I", "-B", str(target / "scripts" / Path(__file__).name),
             "--_run-installed", "--source-root", str(source_root),
             "--pattern", pattern, "--verbosity", str(verbosity)],
            cwd=target, env=env, check=False,
        ).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", nargs="?", type=Path, help="Wheel file or directory containing one wheel")
    parser.add_argument("--source-root", type=Path, default=ROOT, help="Checkout containing tests/support")
    parser.add_argument("--pattern", default="test*.py", help="unittest discovery pattern (default: full suite)")
    parser.add_argument("--verbosity", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--_run-installed", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._run_installed:
        return run_installed_suite(args.source_root, pattern=args.pattern, verbosity=args.verbosity)
    if args.wheel is None:
        parser.error("a wheel file or wheel directory is required")
    try:
        return test_wheel(args.wheel, args.source_root, pattern=args.pattern, verbosity=args.verbosity)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Installed-wheel setup failed: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
