"""Offline contracts for the isolated installed-wheel regression runner."""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "test_installed_wheel.py"
SPEC = importlib.util.spec_from_file_location("wheel_runner", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class WheelRunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="4dwt-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def make_fixture(self, *, missing=None, fail=False):
        repo = self.root / "checkout"
        for directory in ("tests", "scripts", "docs", "tutorials", ".github/workflows"):
            (repo / directory).mkdir(parents=True)
        (repo / ".gitignore").write_text("/docs/_build/\n", encoding="utf-8")
        (repo / ".github/workflows/docs.yml").write_text("name: docs\n", encoding="utf-8")
        shutil.copy2(SCRIPT, repo / "scripts" / SCRIPT.name)
        (repo / "scripts" / "support.py").write_text("VALUE = 'support'\n", encoding="utf-8")
        (repo / "docs" / "conf.py").write_text("VALUE = 'docs'\n", encoding="utf-8")
        (repo / "tutorials" / "example.ipynb").write_text('{"cells": []}', encoding="utf-8")
        for name in runner.MODULE_NAMES:
            (repo / f"{name}.py").write_text("raise RuntimeError('checkout imported')\n", encoding="utf-8")
        assertion = "self.fail('deliberate wheel-suite failure')" if fail else "self.assertEqual(fd.VALUE, 'wheel')"
        (repo / "tests" / "test_fixture.py").write_text(
            "import importlib.util\nimport sys\nimport unittest\nfrom pathlib import Path\n"
            "ROOT = Path(__file__).resolve().parents[1]\n"
            "sys.path.insert(0, str(ROOT))\nimport fourdenoise as fd\n"
            "if Path(fd.__file__).resolve() != ROOT / 'fourdenoise.py':\n"
            "    raise ImportError('wrong fourdenoise origin')\n"
            "class FixtureTests(unittest.TestCase):\n"
            "    def test_installed_module(self):\n"
            f"        {assertion}\n"
            "    def test_copied_support(self):\n"
            "        for directory, filename, expected in (('scripts', 'support.py', 'support'), "
            "('docs', 'conf.py', 'docs')):\n"
            "            spec = importlib.util.spec_from_file_location('fixture_support', ROOT / directory / filename)\n"
            "            module = importlib.util.module_from_spec(spec)\n"
            "            spec.loader.exec_module(module)\n"
            "            self.assertEqual(module.VALUE, expected)\n"
            "        self.assertTrue((ROOT / 'tutorials' / 'example.ipynb').is_file())\n"
            "        self.assertEqual((ROOT / '.gitignore').read_text(), '/docs/_build/\\n')\n"
            "        self.assertEqual((ROOT / '.github/workflows/docs.yml').read_text(), 'name: docs\\n')\n",
            encoding="utf-8",
        )
        wheel = self.root / "fourdenoise_main-1.0.0-py3-none-any.whl"
        metadata = "fourdenoise_main-1.0.0.dist-info"
        with zipfile.ZipFile(wheel, "w") as archive:
            for name in runner.MODULE_NAMES:
                if name != missing:
                    archive.writestr(f"{name}.py", "VALUE = 'wheel'\n")
            archive.writestr(f"{metadata}/METADATA", "Metadata-Version: 2.1\nName: fourdenoise-main\nVersion: 1.0.0\n")
            archive.writestr(f"{metadata}/WHEEL", "Wheel-Version: 1.0\nGenerator: runner-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
            archive.writestr(f"{metadata}/RECORD", "")
        return repo, wheel

    def invoke(self, repo, wheel):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo)
        return subprocess.run(
            [sys.executable, "-I", "-B", str(SCRIPT), str(wheel), "--source-root", str(repo)],
            cwd=repo, env=env, capture_output=True, text=True, timeout=90,
        )

    def test_complete_suite_uses_installed_wheel_despite_checkout_and_pythonpath(self):
        repo, wheel = self.make_fixture()
        result = self.invoke(repo, wheel)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Running 2 installed-wheel tests", result.stdout)
        self.assertIn("site-packages", result.stdout)
        self.assertIn("OK", result.stderr)
        self.assertFalse(list(repo.glob(".installed-wheel-*")))

    def test_missing_packaged_helper_cannot_fall_back_to_checkout(self):
        repo, wheel = self.make_fixture(missing="fourdenoise_geometry")
        result = self.invoke(repo, wheel)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Wheel is missing packaged module fourdenoise_geometry", result.stderr)
        self.assertNotIn("Running 2 installed-wheel tests", result.stdout)

    def test_suite_failure_propagates_nonzero_exit_and_cleans_target(self):
        repo, wheel = self.make_fixture(fail=True)
        result = self.invoke(repo, wheel)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("deliberate wheel-suite failure", result.stderr)
        self.assertFalse(list(repo.glob(".installed-wheel-*")))

    def test_support_copy_does_not_copy_modules_or_stale_caches(self):
        repo, _ = self.make_fixture()
        (repo / "tests" / "__pycache__").mkdir()
        (repo / "tests" / "__pycache__" / "test_stale.pyc").write_bytes(b"stale")
        (repo / "docs" / "_build").mkdir()
        (repo / "tutorials" / ".ipynb_checkpoints").mkdir()
        (repo / "tutorials" / ".ipynb_checkpoints" / "stale.ipynb").write_text("{}", encoding="utf-8")
        target = self.root / "installed"
        target.mkdir()
        runner.copy_support(repo, target)
        self.assertTrue((target / "tests" / "test_fixture.py").is_file())
        self.assertTrue((target / "scripts" / "support.py").is_file())
        self.assertTrue((target / "docs" / "conf.py").is_file())
        self.assertTrue((target / "tutorials" / "example.ipynb").is_file())
        self.assertFalse((target / "fourdenoise.py").exists())
        self.assertFalse((target / "tests" / "__pycache__").exists())
        self.assertFalse((target / "docs" / "_build").exists())
        self.assertFalse((target / "tutorials" / ".ipynb_checkpoints").exists())

    def test_wheel_directory_requires_exactly_one_wheel(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            runner.resolve_wheel(self.root)
        _, wheel = self.make_fixture()
        # Windows temporary paths may use an 8.3 alias for the user directory.
        self.assertEqual(runner.resolve_wheel(self.root), wheel.resolve())
        self.assertEqual(runner.resolve_wheel(wheel), wheel.resolve())
        shutil.copy2(wheel, self.root / "another.whl")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            runner.resolve_wheel(self.root)

    def test_child_requires_isolated_python(self):
        with patch.object(runner.sys, "flags") as flags:
            flags.isolated = 0
            with self.assertRaisesRegex(RuntimeError, "isolated mode"):
                runner.run_installed_suite(ROOT)


if __name__ == "__main__":
    unittest.main()
