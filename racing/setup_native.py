"""Build the optional exact fused MuJoCo rollout extension in-place.

Usage from the directory containing ``racing/``::

    python racing/setup_native.py build_ext --inplace

or from inside the package directory::

    python setup_native.py build_ext --inplace
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

from setuptools import Extension, setup

try:
    import mujoco
    import numpy as np
    import pybind11
except ImportError as exc:
    raise SystemExit(
        "Native build requires mujoco, numpy and pybind11. "
        "Run: python -m pip install 'pybind11>=2.12'"
    ) from exc

package_dir = Path(__file__).resolve().parent
project_dir = package_dir.parent
os.chdir(project_dir)

mujoco_dir = Path(mujoco.__file__).resolve().parent
include_dir = mujoco_dir / "include"
if not include_dir.exists():
    raise SystemExit(f"MuJoCo headers not found at {include_dir}")

if sys.platform.startswith("linux"):
    libraries = sorted(mujoco_dir.glob("libmujoco.so.*"))
    compile_args = ["-O3", "-std=c++17", "-march=native", "-DNDEBUG", "-pthread"]
    link_args = ["-pthread", f"-Wl,-rpath,{mujoco_dir}"]
elif sys.platform == "darwin":
    libraries = sorted(mujoco_dir.glob("libmujoco.*.dylib"))
    compile_args = ["-O3", "-std=c++17", "-DNDEBUG"]
    link_args = [f"-Wl,-rpath,{mujoco_dir}"]
else:
    raise SystemExit("The fused evaluator build script currently supports Linux and macOS.")

if not libraries:
    raise SystemExit(f"Could not find the MuJoCo shared library in {mujoco_dir}")

ext = Extension(
    "racing._fused_mujoco",
    sources=[str(package_dir / "native" / "fused_mujoco.cpp")],
    include_dirs=[pybind11.get_include(), np.get_include(), str(include_dir)],
    extra_objects=[str(libraries[-1])],
    language="c++",
    extra_compile_args=compile_args,
    extra_link_args=link_args,
)

setup(
    name="racing-fused-mujoco",
    version="0.1.0",
    ext_modules=[ext],
)
