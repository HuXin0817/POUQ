#!/usr/bin/env python3
"""
Example usage of POUQ Python bindings
"""


import numpy as np

from pouq import Quantizer

n_samples = 1000
dim = 16

def main():
    data = np.random.rand(n_samples, dim).astype(np.float32)
    quantizer = Quantizer(dim=dim)
    quantizer.train(data)

    vec1 = quantizer.decode(0)
    vec2 = data[0, :]

    print(np.linalg.norm(vec1 - vec2))


if __name__ == "__main__":
    main()
