#!/usr/bin/env python3
"""
Setup script for POUQ Python bindings
"""

from setuptools import setup
from setuptools.command.build_py import build_py
import os
import sys
import subprocess

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBPOUQ_DIR = os.path.join(PROJECT_ROOT, "libpouq")


class CustomBuildPy(build_py):
    """Custom build_py to compile C library as shared library"""
    
    def run(self):
        # Build the shared library before building Python packages
        self.build_shared_lib()
        super().run()
    
    def build_shared_lib(self):
        """Build libpouq as a shared library"""
        import platform
        
        # Determine shared library extension
        if platform.system() == "Darwin":
            lib_ext = ".dylib"
        else:
            lib_ext = ".so"
        
        lib_name = f"libpouq{lib_ext}"
        # Use build_lib to ensure library is in the right place
        build_dir = os.path.join(self.build_lib, "pouq")
        os.makedirs(build_dir, exist_ok=True)
        lib_path = os.path.join(build_dir, lib_name)
        
        # Get all C source files
        c_files = []
        for root, dirs, files in os.walk(LIBPOUQ_DIR):
            for file in files:
                if file.endswith(".c"):
                    c_files.append(os.path.join(root, file))
        
        # Compiler flags
        cc = os.environ.get("CC", "gcc")
        cflags = [
            "-std=c2x",
            "-Wall",
            "-Wextra",
            "-O3",
            "-mavx2",
            "-mfma",
            "-fopenmp",
            "-fPIC",
        ]
        
        # Link flags
        ldflags = ["-lm", "-fopenmp", "-lpthread", "-shared"]
        
        # Include directories
        include_dirs = [LIBPOUQ_DIR]
        
        # Build command - compile objects first, then link
        # Use a temporary directory for object files (in build_lib)
        obj_dir = os.path.join(self.build_lib, "pouq_obj")
        os.makedirs(obj_dir, exist_ok=True)
        
        obj_files = []
        for c_file in c_files:
            obj_file = os.path.join(obj_dir, os.path.basename(c_file).replace(".c", ".o"))
            obj_files.append(obj_file)
            compile_cmd = [cc] + cflags + ["-c", c_file, "-o", obj_file]
            for inc_dir in include_dirs:
                compile_cmd.extend(["-I", inc_dir])
            print(f"Compiling {os.path.basename(c_file)}...")
            subprocess.check_call(compile_cmd)
        
        # Link command
        cmd = [cc] + obj_files + ["-o", lib_path] + ldflags
        
        print(f"Building shared library: {lib_name}")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            subprocess.check_call(cmd)
            print(f"Successfully built {lib_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error building shared library: {e}")
            sys.exit(1)


setup(
    name="pouq",
    version="0.1.0",
    description="Python bindings for POUQ library",
    author="POUQ Contributors",
    package_dir={"": "."},
    packages=["pouq"],
    cmdclass={"build_py": CustomBuildPy},
    python_requires=">=3.6",
    install_requires=["numpy>=1.16.0"],
)

