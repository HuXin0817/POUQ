#!/usr/bin/env python3
"""
Example usage of POUQ Python bindings
"""

import numpy as np
from pouq import Quantizer


def main():
    data = np.random.rand(10, 10)
    quantizer = Quantizer(dim=10)
    quantizer.train(data=data)
    print(quantizer.decode(1).tolist())
    print(data[1].tolist())


if __name__ == "__main__":
    main()
