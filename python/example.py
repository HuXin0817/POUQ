#!/usr/bin/env python3
"""
Example usage of POUQ Python bindings
"""

import faulthandler

import numpy as np
from pouq import Quantizer

faulthandler.enable()


def main():
    data = np.random.rand(100, 10)
    quantizer = Quantizer(dim=10)
    quantizer.train(data)

    vec1 = quantizer.decode(0)
    vec2 = data[0, :]

    print(np.linalg.norm(vec1 - vec2))


if __name__ == "__main__":
    main()
