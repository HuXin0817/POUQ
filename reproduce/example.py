import sys

import numpy as np

from pypouq import QVector


def read_fvecs(filename, c_contiguous=True) -> np.ndarray:
    print(f"Reading from {filename}.")
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv


if len(sys.argv) != 2:
    print(f"usage: {sys.argv[0]} <dataset_name>")
    exit(0)
dataset_name = sys.argv[1]
data = read_fvecs(f"../data/{dataset_name}/{dataset_name}_base.fvecs")

N, Dim = data.shape
print(f"N={N}, Dim={Dim}")


def print_err(method: str, qvector: QVector):
    err = 0
    for i in range(N):
        for j in range(Dim):
            err += (data[i, j] - qvector[i, j]) ** 2
    print(f"Method: {method}, Error: {err / N}")


print_err("UQ", QVector(data, c_bit=0, q_bit=8, optimize_bound=False))
print_err("OUQ", QVector(data, c_bit=0, q_bit=8, optimize_bound=True))
print_err("PUQ", QVector(data, c_bit=4, q_bit=4, optimize_bound=False))
print_err("POUQ", QVector(data, c_bit=4, q_bit=4, optimize_bound=True))
