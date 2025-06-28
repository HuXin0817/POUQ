import os

import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

num_cores = os.cpu_count() or 1

ext_modules = [
    Pybind11Extension(
        "pouq",
        ["binding.cpp"],
        include_dirs=[pybind11.get_include(), "."],
        language="c++",
        cxx_std=17,
        extra_compile_args=["-fopenmp", "-O3", "-DWITH_THREAD"],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    name="pouq",
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
