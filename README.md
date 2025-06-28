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

### Example

For examples on random datasets, please refer to `./example.cpp`

```cpp
#include "../libpouq/quantizer.h"
#include "../libpouq/utils.h"

#include <chrono>
#include <iomanip>
#include <iostream>

constexpr size_t N = 1e6;

template <typename DataType>
void print_vector(const char *prefix, const DataType &data) {
  std::cout << std::left << std::setw(18) << prefix << "[";
  std::cout << std::fixed << std::setprecision(3);
  for (size_t i = 0; i < 5; ++i) {
    std::cout << data[i];
    if (i < 4) {
      std::cout << ", ";
    } else {
      std::cout << "...]\n";
    }
  }
  std::cout << std::defaultfloat;
}

int main() {
  std::random_device             rd;
  std::mt19937                   gen(rd());
  std::uniform_real_distribution dis(0.0f, 256.0f);

  std::vector<float> data(N);
  for (auto &d : data) {
    d = dis(gen);
  }

  pouq::POUQQuantizer quantizer(4, 4, 256);

  const auto start_time = std::chrono::high_resolution_clock::now();
  quantizer.train(data.data(), N);
  const auto end_time = std::chrono::high_resolution_clock::now();
  const auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);

  std::cout << std::left << std::setw(18) << "Training time:" << duration.count() << "s" << std::endl;
  std::cout << std::left << std::setw(18) << "Error:" << compute_mse(data, quantizer, N) << std::endl;

  print_vector("Origin Vector:", data);
  print_vector("Quantized Vector:", quantizer);
}
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
