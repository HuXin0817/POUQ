"""
POUQ Python Bindings

This module provides Python bindings for the POUQ C library.
"""

from .pouq import (
    train,
    distance,
    decode,
    segment,
    optimize,
    get_sorted_data,
    count_freq,
    Parameter,
    Bound,
    CodeUnit,
    RecPara,
)

__version__ = "0.1.0"
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
    "CodeUnit",
    "RecPara",
]

