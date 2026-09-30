from __future__ import annotations

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class CleanBuildPy(_build_py):
    """Prevent deleted SWAAG modules from surviving in setuptools build/lib."""

    def run(self) -> None:
        package_root = Path(self.build_lib) / "swaag"
        shutil.rmtree(package_root, ignore_errors=True)
        super().run()
        # Broad package-data globs can otherwise pick up interpreter cache files
        # from benchmark fixture directories. The built wheel must contain only
        # source/package-data artifacts, never transient bytecode caches.
        for cache_dir in list(package_root.rglob("__pycache__")):
            shutil.rmtree(cache_dir, ignore_errors=True)
        for pattern in ("*.pyc", "*.pyo"):
            for artifact in package_root.rglob(pattern):
                artifact.unlink(missing_ok=True)


setup(cmdclass={"build_py": CleanBuildPy})
