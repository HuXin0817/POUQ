# [ESWA 2025 With Editor] POUQ: A Clustering and Swarm-Optimized Framework for Precision-Driven Uniform Quantization of Non-uniform Data

## Prerequisites

- OpenMP is required

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
cd python/ && pip install .
```

### API Description

- `QVector(data, c_bit, q_bit, learn_step_size=True, max_iter=128, grid_side_length=8, grid_scale_factor=0.1, initial_inertia=0.9, final_inertia=0.4, c1=1.8, c2=1.8)` -
  Initializes a quantum vector with specified parameters
  - `data` numpy array of input data
  - `c_bit` number of classical bits
  - `q_bit` number of quantum bits
  - `learn_step_size` flag to enable learning step size (default: True)
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


print_err(QVector(data, c_bit=0, q_bit=8, learn_step_size=False))
print_err(QVector(data, c_bit=0, q_bit=8, learn_step_size=True))
print_err(QVector(data, c_bit=4, q_bit=4, learn_step_size=False))
print_err(QVector(data, c_bit=4, q_bit=4, learn_step_size=True))
```

## Reproduce

- For downloading datasets and preprocessing, please refer to `./data/README.md`

## License

[Apache License 2.0](./LICENSE)
