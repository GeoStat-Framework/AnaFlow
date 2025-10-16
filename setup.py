from __future__ import annotations

from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

SRC_DIR = Path("src")
MODULE_PATH = SRC_DIR / "anaflow" / "flow"
MODULE_NAME = "anaflow.flow._laplace_accel"

extensions = [
    Extension(
        MODULE_NAME,
        [str(MODULE_PATH / "_laplace_accel.pyx")],
        include_dirs=[np.get_include()],
    )
]

extensions = cythonize(
    extensions,
    language_level=3,
    compiler_directives={
        "boundscheck": False,
        "wraparound": False,
        "initializedcheck": False,
    },
)

setup(ext_modules=extensions, package_dir={"": "src"})
