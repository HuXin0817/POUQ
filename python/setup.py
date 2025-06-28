import platform

import pybind11
from setuptools import Extension, setup

# 基础编译参数
base_compile_args = ["-std=c++17", "-fopenmp", "-O3"]


# 根据平台和架构添加 SIMD 指令集支持
def get_compile_args():
    compile_args = base_compile_args.copy()

    # 检查是否为 x86_64 架构
    machine = platform.machine().lower()
    if machine in ["x86_64", "amd64"]:
        # 在 x86_64 架构上添加 AVX2 和 FMA 支持
        compile_args.extend(["-mavx2", "-mfma"])

    return compile_args


ext_modules = [
    Extension(
        "pouq",
        ["binding.cpp"],
        include_dirs=[pybind11.get_include(), "."],
        language="c++",
        extra_compile_args=get_compile_args(),
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    name="pouq",
    version="0.1",
    ext_modules=ext_modules,
)
