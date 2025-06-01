# [ESWA 2025 With Editor] POUQ: A Clustering and Swarm-Optimized Framework for Precision-Driven Uniform Quantization of Non-uniform Data

![](GraphicalAbstract.svg)

## Prerequisites

- [OpenMP](https://www.openmp.org/) is required

## Directory Structure

```
../
├── data/       # datasets and indices
├── example/    # cpp example
├── libpouq/    # cpp implements
├── python/     # python bindings
└── reproduce/  # code for reproduction
```

## Python Bindings (recommended)

### Bindings installation

- Install from sources in Python env (recommended version: 3.10):

```bash
cd python/ && bash build.sh
```

### API Description

#### POUQuantizer

- `POUQuantizer(c_bit: int, q_bit: int, groups: int = 1)` - Initializes a POUQ quantizer with specified parameters
  - `c_bit`: Number of cluster id bits (0-16)
  - `q_bit`: Number of quantization bits (1-16)
  - `groups`: Number of quantization groups (default: `1`)

`POUQuantizer` methods:

- `train(data: np.ndarray) -> None` - Trains the quantizer with input data
  - `data`: Input numpy array of float32 values
- `size() -> int` - Returns the size of the quantization vector
- `__getitem__(i: int) -> float` - Returns the quantized value at the specified index
  - `i`: Index to retrieve

#### ScaledQuantizer

- `ScaledQuantizer(q_bit: int, groups: int = 1)` - Initializes a scaled quantizer with specified parameters
  - `q_bit`: Number of quantization bits (1-16)
  - `groups`: Number of quantization groups (default: `1`)

`ScaledQuantizer` methods:

- `train(data: np.ndarray) -> None` - Trains the quantizer with input data
  - `data`: Input numpy array of float32 values
- `size() -> int` - Returns the size of the quantization vector
- `__getitem__(i: int) -> float` - Returns the quantized value at the specified index
  - `i`: Index to retrieve

### Example

For examples on real-world datasets, please refer to `./reproduce`

```python
import sys

import numpy as np
from pouq import Quantizer, compute_mse


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


def print_err(method: str, quantizer: Quantizer):
  print(f"Method: {method}, Error: {compute_mse(data, quantizer)}")


print_err("SQ", Quantizer(data, c_bit=0, q_bit=8, groups=Dim, opt_bound=False))
print_err("POUQ", Quantizer(data, c_bit=4, q_bit=4, groups=Dim, opt_bound=True))
```

## Reproduce

- For downloading datasets and preprocessing, please refer to `./data/README.md`

## Results

Comparison table of quantization mean squared error (MSE) between scalar quantization (SQ) and POUQ method across different datasets (quantization bit depth `cbit + qbit = 8`) **as follows**:

| Dataset  | Dimension | Size      | SQ       | POUQ (Ours) | Reduction  |
| -------- | --------- | --------- | -------- | ----------- | ---------- |
| SIFT     | 128       | 1,000,000 | 1.78e-02 | 1.11e-03    | **93.76%** |
| GIST     | 960       | 1,000,000 | 1.34e-07 | 3.02e-08    | **77.46%** |
| Deep1M   | 256       | 1,000,000 | 7.08e-08 | 4.87e-08    | **31.21%** |
| ImageNet | 150       | 2,340,373 | 1.02e-07 | 2.12e-08    | **79.22%** |
| MSong    | 420       | 994,185   | 3.40e-02 | 3.50e-04    | **98.97%** |
| Word2Vec | 300       | 1,000,000 | 4.51e-06 | 5.81e-07    | **87.12%** |

## Reference

Reference to cite when you use POUQ in a research paper:

```latex
@article{hu2025pouq,
  title={POUQ: A Clustering and Swarm-Optimized Framework for Precision-Driven Uniform Quantization of Non-uniform Data},
  author={Xin Hu},
  year={2025},
}
```

## Contributors

For any questions, feel free to open an issue or contact us:

- 202219120810@stu.cdut.edu.cn

## License

[Apache License 2.0](./LICENSE)
