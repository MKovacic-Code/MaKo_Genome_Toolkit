from pathlib import Path

from setuptools import setup

try:
    from Cython.Build import cythonize
except ImportError as exc:
    raise SystemExit(
        "Cython is required to build the cython_helpers extension. "
        "Install it with 'python -m pip install cython' before running setup.py."
    ) from exc

ROOT = Path(__file__).parent.resolve()
SCRIPTS_DIR = ROOT / "scripts"

setup(
    name="nucleotide_cython_helpers",
    ext_modules=cythonize(
        [str(SCRIPTS_DIR / "cython_helpers.pyx"), str(SCRIPTS_DIR / "triplex_scan.pyx")],
        compiler_directives={"language_level": "3"},
    ),
)
