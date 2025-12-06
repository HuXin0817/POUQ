"""
POUQ Python bindings using ctypes

This module provides a Python interface to the POUQ C library.
"""

import ctypes
import os
import platform
import sys
import numpy as np
from typing import Tuple, Optional

# Try to locate the shared library
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


class Bound(ctypes.Structure):
    """C Bound structure"""
    _fields_ = [
        ("lower", ctypes.c_float),
        ("upper", ctypes.c_float),
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
    
    # train_impl function
    _lib.train_impl.argtypes = [
        ctypes.c_int,  # dim
        ctypes.POINTER(CodeUnit),  # code
        ctypes.POINTER(RecPara),  # rec_para
        ctypes.POINTER(ctypes.c_float),  # data
        ctypes.c_int,  # size
        Parameter,  # parameter
    ]
    _lib.train_impl.restype = None
    
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
    
    # segment function
    _lib.segment.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # data_map
        ctypes.POINTER(ctypes.c_int),  # freq_map
        ctypes.c_int,  # size
        ctypes.c_bool,  # do_count_freq
        ctypes.POINTER(ctypes.c_float),  # lowers
        ctypes.POINTER(ctypes.c_float),  # uppers
    ]
    _lib.segment.restype = ctypes.c_int
    
    # optimize function
    _lib.optimize.argtypes = [
        ctypes.c_float,  # init_lower
        ctypes.c_float,  # init_upper
        ctypes.POINTER(ctypes.c_float),  # data_map
        ctypes.POINTER(ctypes.c_int),  # freq_map
        ctypes.c_int,  # size
        Parameter,  # parameter
        ctypes.c_bool,  # do_count_freq
    ]
    _lib.optimize.restype = Bound
    
    # get_sorted_data function
    _lib.get_sorted_data.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # data
        ctypes.c_int,  # size
        ctypes.c_int,  # d
        ctypes.c_int,  # dim
        ctypes.POINTER(ctypes.c_float),  # sorted_data
    ]
    _lib.get_sorted_data.restype = ctypes.c_int
    
    # count_freq function
    _lib.count_freq.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # sorted_data
        ctypes.c_int,  # sorted_data_size
        ctypes.POINTER(ctypes.c_float),  # data_map
        ctypes.POINTER(ctypes.c_int),  # freq_map
    ]
    _lib.count_freq.restype = ctypes.c_int


# Python-friendly wrapper functions
def train(
    dim: int,
    data: np.ndarray,
    parameter: Optional[Parameter] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train POUQ model
    
    Args:
        dim: Dimension of the data
        data: Input data array (shape: [n_samples, dim])
        parameter: Training parameters (optional)
    
    Returns:
        Tuple of (code, rec_para) where:
        - code: CodeUnit array with shape [n_samples // 8, 3] (x1, x2, code)
        - rec_para: RecPara array with shape [dim * 64, 8] (lower[4], step_size[4])
    """
    if _lib is None:
        raise RuntimeError("POUQ library not loaded")
    
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float32)
    else:
        data = data.astype(np.float32)
    
    if data.ndim != 2 or data.shape[1] != dim:
        raise ValueError(f"Data must be 2D array with shape [n_samples, {dim}]")
    
    size = data.shape[0] * dim  # Total number of floats
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
    
    result = _lib.train(dim, data_ptr, size, parameter)
    
    # Extract arrays from result
    # code size: size / 8
    code_size = size // 8
    code_array = np.zeros((code_size, 3), dtype=np.uint16)
    for i in range(code_size):
        code_array[i, 0] = result.code[i].x1
        code_array[i, 1] = result.code[i].x2
        code_array[i, 2] = result.code[i].code
    
    # rec_para size: dim * 64
    rec_para_size = dim * 64
    rec_para_array = np.zeros((rec_para_size, 8), dtype=np.float32)
    for i in range(rec_para_size):
        # Extract lower (4 floats)
        for j in range(4):
            rec_para_array[i, j] = result.rec_para[i].lower[j]
        # Extract step_size (4 floats)
        for j in range(4):
            rec_para_array[i, j + 4] = result.rec_para[i].step_size[j]
    
    # Free allocated memory
    if _has_free:
        _lib.free(result.code)
        _lib.free(result.rec_para)
    else:
        _ctypes_free(result.code)
        _ctypes_free(result.rec_para)
    
    return code_array, rec_para_array


def distance(
    dim: int,
    code: np.ndarray,
    rec_para: np.ndarray,
    data: np.ndarray,
    offset: int = 0,
) -> float:
    """
    Calculate distance using POUQ code
    
    Args:
        dim: Dimension of the data
        code: CodeUnit array with shape [n_code_units, 3] (x1, x2, code)
        rec_para: RecPara array with shape [dim * 64, 8] (lower[4], step_size[4])
        data: Input data array (flattened)
        offset: Offset in the data array
    
    Returns:
        Distance value
    """
    if _lib is None:
        raise RuntimeError("POUQ library not loaded")
    
    # Convert inputs to appropriate C types
    code_size = len(code)
    code_array = (CodeUnit * code_size)()
    for i, c in enumerate(code):
        code_array[i] = CodeUnit(
            ctypes.c_uint8(c[0]),
            ctypes.c_uint8(c[1]),
            ctypes.c_uint16(c[2]),
        )
    
    rec_para_size = len(rec_para)
    rec_para_array = (RecPara * rec_para_size)()
    for i, rp in enumerate(rec_para):
        rec_para_array[i] = RecPara(
            (ctypes.c_float * 4)(*rp[:4]),
            (ctypes.c_float * 4)(*rp[4:8]),
        )
    
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float32)
    else:
        data = data.astype(np.float32)
    
    data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    return _lib.distance(
        dim,
        code_array,
        rec_para_array,
        data_ptr,
        offset,
    )


def decode(
    dim: int,
    code: np.ndarray,
    rec_para: np.ndarray,
    dist: np.ndarray,
    offset: int = 0,
) -> None:
    """
    Decode using POUQ code
    
    Args:
        dim: Dimension of the data
        code: CodeUnit array with shape [n_code_units, 3] (x1, x2, code)
        rec_para: RecPara array with shape [dim * 64, 8] (lower[4], step_size[4])
        dist: Output distance array (modified in place)
        offset: Offset in the distance array
    """
    if _lib is None:
        raise RuntimeError("POUQ library not loaded")
    
    # Similar conversion as distance function
    code_size = len(code)
    code_array = (CodeUnit * code_size)()
    for i, c in enumerate(code):
        code_array[i] = CodeUnit(
            ctypes.c_uint8(c[0]),
            ctypes.c_uint8(c[1]),
            ctypes.c_uint16(c[2]),
        )
    
    rec_para_size = len(rec_para)
    rec_para_array = (RecPara * rec_para_size)()
    for i, rp in enumerate(rec_para):
        rec_para_array[i] = RecPara(
            (ctypes.c_float * 4)(*rp[:4]),
            (ctypes.c_float * 4)(*rp[4:8]),
        )
    
    if not isinstance(dist, np.ndarray):
        dist = np.array(dist, dtype=np.float32)
    else:
        dist = dist.astype(np.float32)
    
    dist_ptr = dist.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    _lib.decode(dim, code_array, rec_para_array, dist_ptr, offset)


def segment(
    data_map: np.ndarray,
    freq_map: np.ndarray,
    do_count_freq: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment data
    
    Args:
        data_map: Data map array
        freq_map: Frequency map array
        do_count_freq: Whether to count frequency
    
    Returns:
        Tuple of (lowers, uppers) arrays
    """
    if _lib is None:
        raise RuntimeError("POUQ library not loaded")
    
    size = len(data_map)
    
    data_map_ptr = data_map.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    freq_map_ptr = freq_map.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    lowers = np.zeros(size, dtype=np.float32)
    uppers = np.zeros(size, dtype=np.float32)
    
    lowers_ptr = lowers.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    uppers_ptr = uppers.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    _lib.segment(
        data_map_ptr,
        freq_map_ptr,
        size,
        do_count_freq,
        lowers_ptr,
        uppers_ptr,
    )
    
    return lowers, uppers


def optimize(
    init_lower: float,
    init_upper: float,
    data_map: np.ndarray,
    freq_map: np.ndarray,
    parameter: Optional[Parameter] = None,
    do_count_freq: bool = True,
) -> Bound:
    """
    Optimize bounds
    
    Args:
        init_lower: Initial lower bound
        init_upper: Initial upper bound
        data_map: Data map array
        freq_map: Frequency map array
        parameter: Optimization parameters (optional)
        do_count_freq: Whether to count frequency
    
    Returns:
        Bound structure with optimized lower and upper
    """
    if _lib is None:
        raise RuntimeError("POUQ library not loaded")
    
    size = len(data_map)
    data_map_ptr = data_map.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    freq_map_ptr = freq_map.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    if parameter is None:
        parameter = Parameter(
            max_iter=100,
            particle_count=30,
            scale_factor=0.1,
            init_inertia=0.9,
            final_inertia=0.4,
            c1=2.0,
            c2=2.0,
        )
    
    return _lib.optimize(
        init_lower,
        init_upper,
        data_map_ptr,
        freq_map_ptr,
        size,
        parameter,
        do_count_freq,
    )


def get_sorted_data(
    data: np.ndarray,
    d: int,
    dim: int,
) -> np.ndarray:
    """
    Get sorted data
    
    Args:
        data: Input data array
        d: Dimension index
        dim: Total dimensions
    
    Returns:
        Sorted data array
    """
    if _lib is None:
        raise RuntimeError("POUQ library not loaded")
    
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float32)
    else:
        data = data.astype(np.float32)
    
    size = data.shape[0]
    sorted_data = np.zeros(size, dtype=np.float32)
    
    data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    sorted_data_ptr = sorted_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    _lib.get_sorted_data(data_ptr, size, d, dim, sorted_data_ptr)
    
    return sorted_data


def count_freq(
    sorted_data: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Count frequency of sorted data
    
    Args:
        sorted_data: Sorted data array
    
    Returns:
        Tuple of (data_map, freq_map)
    """
    if _lib is None:
        raise RuntimeError("POUQ library not loaded")
    
    sorted_data_size = len(sorted_data)
    # Allocate output arrays (size needs to be determined from implementation)
    # This is a placeholder - adjust based on actual C implementation
    max_size = sorted_data_size
    data_map = np.zeros(max_size, dtype=np.float32)
    freq_map = np.zeros(max_size, dtype=np.int32)
    
    sorted_data_ptr = sorted_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    data_map_ptr = data_map.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    freq_map_ptr = freq_map.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    result_size = _lib.count_freq(
        sorted_data_ptr,
        sorted_data_size,
        data_map_ptr,
        freq_map_ptr,
    )
    
    # Trim arrays to actual size
    if result_size > 0 and result_size < max_size:
        data_map = data_map[:result_size]
        freq_map = freq_map[:result_size]
    
    return data_map, freq_map


# Export Parameter and Bound classes for easy access
__all__ = [
    "train",
    "distance",
    "decode",
    "segment",
    "optimize",
    "get_sorted_data",
    "count_freq",
    "Parameter",
    "Bound",
    "Result",
    "CodeUnit",
    "RecPara",
]

