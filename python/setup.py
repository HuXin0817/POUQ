import pybind11
from setuptools import Extension, setup

ext_modules = [
    Extension(
        "pouq",
        ["binding.cpp"],
        include_dirs=[pybind11.get_include(), "."],
        language="c++",
        extra_compile_args=["-std=c++17", "-fopenmp", "-O3", "-fno-finite-math-only"],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    name="pouq",
    version="0.1",
    ext_modules=ext_modules,
)
