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


dataset_name = sys.argv[1]
data = read_fvecs(f"../data/{dataset_name}/{dataset_name}_base.fvecs")

N, Dim = data.shape


def print_err(q):
    err = 0
    for i in range(N):
        for j in range(Dim):
            err += (data[i, j] - q[i, j]) ** 2
    print(err / N)


print_err(QVector(data, c_bit=0, q_bit=8, optimize_bound=False))
print_err(QVector(data, c_bit=0, q_bit=8, optimize_bound=True))
print_err(QVector(data, c_bit=4, q_bit=4, optimize_bound=False))
print_err(QVector(data, c_bit=4, q_bit=4, optimize_bound=True))
