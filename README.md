# [ESWA 2025 With Editor] POUQ: A Clustering and Swarm-Optimized Framework for Precision-Driven Uniform Quantization of Non-uniform Data

## Prerequisites

- [OpenMP](https://www.openmp.org/) is required

## Directory Structure

    ../
    ├── data/       # datasets and indices
    ├── libpouq/    # cpp implements
    ├── pypouq/     # python package for pouq
    ├── python/     # python bindings
    └── reproduce/  # code for reproduction

## Python Bindings (recommended)

### Bindings installation

- Install from sources in Python env (recommended version: 3.10):

```bash
cd python/ && bash build.sh
```

### API Description

- `QVector(data, c_bit, q_bit, optimize_bound=True, max_iter=128, grid_side_length=8, grid_scale_factor=0.1, initial_inertia=0.9, final_inertia=0.4, c1=1.8, c2=1.8)` -
  initializes a quantization vector with specified parameters
  - `data` numpy array of input data
  - `c_bit` number of cluster bits
  - `q_bit` number of quantization bits
  - `optimize_bound` flag to enable optimize bound (default: True)
  - `max_iter` maximum number of iterations (default: 128)
  - `grid_side_length` grid side length (default: 8)
  - `grid_scale_factor` grid scale factor (default: 0.1)
  - `initial_inertia` initial inertia value (default: 0.9)
  - `final_inertia` final inertia value (default: 0.4)
  - `c1` cognitive learning factor (default: 1.8)
  - `c2` social learning factor (default: 1.8)

`QVector` methods:

- `ndim()` - Returns the number of dimensions of the quantum vector
- `shape(i)` - Returns the size of the i-th dimension
  - `i` dimension index
- `__getitem__(item)` - Returns the value at the specified index
  - `item` index or tuple of indices

### Example

For examples on real-world datasets, please refer to `./reproduce`

```python
import sys

import numpy as np

from pypouq import QVector


# read_fvecs sourced from https://github.com/gaoj0017/RaBitQ/blob/main/data/utils/io.py
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
```

## Reproduce

- For downloading datasets and preprocessing, please refer to `./data/README.md`

## Comparison Table of Quantization Errors between Scalar Quantization (SQ) and POUQ Method across Different Datasets

| Dataset  | Dimension | Sample Size | Scalar Quantization (SQ) | POUQ (Ours)  |
| -------- | --------- | ----------- | ------------------------ | ------------ |
| SIFT     | 128       | 1,000,000   | 1.78e-02                 | **1.11e-03** |
| GIST     | 960       | 1,000,000   | 1.34e-07                 | **3.02e-08** |
| Deep1M   | 256       | 1,000,000   | 7.08e-08                 | **4.87e-08** |
| ImageNet | 150       | 1,000,000   | 1.71e-07                 | **2.73e-08** |
| MSong    | 420       | 994,185     | 3.40e-02                 | **3.50e-04** |
| Word2Vec | 300       | 1,000,000   | 4.51e-06                 | **5.81e-07** |

## License

[Apache License 2.0](./LICENSE)
