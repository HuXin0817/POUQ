"""
POUQ Python bindings using ctypes

This module provides a Python interface to the POUQ C library.
"""

import ctypes
import os
import platform
from typing import Optional

import numpy as np


def _find_library():
    """Find the POUQ shared library"""
    # Get the directory where this module is located
    module_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(module_dir)

    # Determine library extension
    if platform.system() == "Darwin":
        lib_ext = ".dylib"
    elif platform.system() == "Windows":
        lib_ext = ".dll"
    else:
        lib_ext = ".so"

    # Possible library locations
    possible_paths = [
        os.path.join(module_dir, f"libpouq{lib_ext}"),
        os.path.join(package_dir, f"libpouq{lib_ext}"),
        os.path.join(package_dir, "build", "lib", "pouq", f"libpouq{lib_ext}"),
        f"libpouq{lib_ext}",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    # Try to load from system
    try:
        if platform.system() == "Windows":
            return ctypes.CDLL("libpouq.dll")
        else:
            return ctypes.CDLL(f"libpouq{lib_ext}")
    except OSError:
        pass

    raise RuntimeError(
        f"Could not find libpouq{lib_ext}. Please build the library first using setup.py"
    )


# Load the library
try:
    _lib = ctypes.CDLL(_find_library())
except Exception as e:
    _lib = None
    print(f"Warning: Could not load POUQ library: {e}")


# Define C structures
class CodeUnit(ctypes.Structure):
    """C CodeUnit structure"""

    _fields_ = [
        ("x1", ctypes.c_uint8),
        ("x2", ctypes.c_uint8),
        ("code", ctypes.c_uint16),
    ]


class RecPara(ctypes.Structure):
    """C RecPara structure (contains __m128 which we'll handle as 4 floats)"""

    _fields_ = [
        ("lower", ctypes.c_float * 4),  # __m128 as 4 floats
        ("step_size", ctypes.c_float * 4),  # __m128 as 4 floats
    ]


class Parameter(ctypes.Structure):
    """C Parameter structure"""

    _fields_ = [
        ("max_iter", ctypes.c_int),
        ("particle_count", ctypes.c_int),
        ("scale_factor", ctypes.c_float),
        ("init_inertia", ctypes.c_float),
        ("final_inertia", ctypes.c_float),
        ("c1", ctypes.c_float),
        ("c2", ctypes.c_float),
    ]


class Result(ctypes.Structure):
    """C Result structure"""

    _fields_ = [
        ("code", ctypes.POINTER(CodeUnit)),
        ("rec_para", ctypes.POINTER(RecPara)),
    ]


# Set up function signatures
if _lib is not None:
    # train function
    _lib.train.argtypes = [
        ctypes.c_int,  # dim
        ctypes.POINTER(ctypes.c_float),  # data
        ctypes.c_int,  # size
        Parameter,  # parameter
    ]
    _lib.train.restype = Result

    # Add free function for memory management (if available)
    try:
        _lib.free.argtypes = [ctypes.c_void_p]
        _lib.free.restype = None
        _has_free = True
    except AttributeError:
        _has_free = False
        # Use Python's ctypes free
        _ctypes_free = ctypes.CDLL(None).free
        _ctypes_free.argtypes = [ctypes.c_void_p]
        _ctypes_free.restype = None

    # distance function
    _lib.distance.argtypes = [
        ctypes.c_int,  # dim
        ctypes.POINTER(CodeUnit),  # code
        ctypes.POINTER(RecPara),  # rec_para
        ctypes.POINTER(ctypes.c_float),  # data
        ctypes.c_int,  # offset
    ]
    _lib.distance.restype = ctypes.c_float

    # decode function
    _lib.decode.argtypes = [
        ctypes.c_int,  # dim
        ctypes.POINTER(CodeUnit),  # code
        ctypes.POINTER(RecPara),  # rec_para
        ctypes.POINTER(ctypes.c_float),  # dist
        ctypes.c_int,  # offset
    ]
    _lib.decode.restype = None


class Quantizer:
    def __init__(self, dim: int):
        self.dim = dim
        self.data: Optional[Result] = None
    
    def __del__(self):
        if self.data is not None:
            if _has_free:
                _lib.free(self.data.code)
                _lib.free(self.data.rec_para)
            else:
                _ctypes_free(self.data.code)
                _ctypes_free(self.data.rec_para)

    def train(
        self,
        data: np.ndarray,
        parameter: Optional[Parameter] = None,
    ):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        else:
            data = data.astype(np.float32)

        if data.ndim != 2 or data.shape[1] != self.dim:
            raise ValueError(
                f"Data must be 2D array with shape [n_samples, {self.dim}]"
            )

        size = data.shape[0] * self.dim  # Total number of floats
        data_flat = data.flatten()
        data_ptr = data_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if parameter is None:
            # Default parameters
            parameter = Parameter(
                max_iter=100,
                particle_count=30,
                scale_factor=0.1,
                init_inertia=0.9,
                final_inertia=0.4,
                c1=2.0,
                c2=2.0,
            )

        self.data = _lib.train(self.dim, data_ptr, size, parameter)

    def distance(
        self,
        data: np.ndarray,
        i: int = 0,
    ) -> float:
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        else:
            data = data.astype(np.float32)

        return _lib.distance(
            self.dim,
            self.data.code,
            self.data.rec_para,
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            i * self.dim,
        )

    def decode(
        self,
        i: int = 0,
    ) -> np.ndarray[np.float32]:
        dist = np.zeros(self.dim, dtype=np.float32)

        _lib.decode(
            self.dim,
            self.data.code,
            self.data.rec_para,
            dist.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            i * self.dim,
        )

        return dist
