# _POUQ: A High-Precision Quantization via Segmentation and Swarm Optimization_

POUQ is a high-performance C++ library for vector quantization that combines advanced optimization techniques with SIMD acceleration to achieve superior compression and fast similarity search.

![Demo](POUQ.png)

## Overview

POUQ implements a sophisticated vector quantization scheme that uses:

- **Particle Swarm Optimization (PSO)** for optimal quantization parameter tuning
- **Dynamic Programming** with Knuth-Yao speedup for optimal data segmentation
- **AVX2 SIMD instructions** for accelerated distance computation
- **OpenMP parallelization** for multi-threaded training

## Key Features

### 🚀 High Performance

- **SIMD-optimized distance computation**: ~8x faster than scalar implementations using AVX2
- **Parallel training**: Multi-threaded processing with OpenMP
- **Memory-aligned data structures**: Optimized for cache efficiency
- **Compressed storage**: Efficient bit-packing reduces memory footprint

### 🎯 Advanced Algorithms

- **Particle Swarm Optimization**: Automatically finds optimal quantization parameters
- **Dynamic Programming Segmentation**: O(kn) optimal clustering using Knuth-Yao optimization
- **4-level per-dimension quantization**: Balances compression ratio and accuracy

### 🔧 Easy Integration

- **Header-only library**: Simple integration into existing projects
- **Modern C++17**: Clean, well-documented API
- **CMake support**: Standard build system integration

## Architecture

The library consists of three main components:

### 1. [Segmenter](./libpouq/segmenter.hpp)

Optimal data partitioning using dynamic programming with Knuth-Yao speedup:

- Partitions each dimension into 4 optimal segments
- Minimizes total quantization cost
- O(kn) time complexity instead of O(kn²)

### 2. [Optimizer](./libpouq/optimizer.hpp)

Particle Swarm Optimization for quantization parameter tuning:

- Optimizes lower bounds and step sizes for each segment
- Time-varying inertia and acceleration coefficients
- Minimizes weighted quantization loss

### 3. [Quantizer](./libpouq/quantizer.hpp)

Main quantization engine with SIMD optimization:

- Multi-dimensional vector quantization
- AVX2-accelerated distance computation
- Compressed code storage and fast reconstruction

## Requirements

- **C++17** compatible compiler
- **AVX2** instruction set support
- **OpenMP** for parallel processing
- **CMake 3.12+** for building

## Installation

### Using CMake

```bash
git clone https://github.com/HuXin0817/POUQ
cd POUQ
mkdir build && cd build
cmake ..
make
```

### Integration

POUQ is a header-only library. Simply include the headers and link with OpenMP:

```cpp
#include "libpouq/quantizer.hpp"

// Your code here
```

## Usage Example

```cpp
#include "libpouq/quantizer.hpp"
#include <vector>
#include <iostream>

int main() {
    // Initialize quantizer for 128-dimensional vectors
    pouq::Quantizer quantizer(128);

    // Prepare training data (num_vectors * dimension)
    std::vector<float> training_data = /* your data */;

    // Train the quantizer
    quantizer.train(training_data.data(), training_data.size());

    // Compute distance to a query vector
    std::vector<float> query = /* your query vector */;
    float distance = quantizer.l2distance(query.data(), vector_offset);

    std::cout << "L2 distance: " << distance << std::endl;

    return 0;
}
```

## Performance Characteristics

### Training Complexity

- **Time**: O(n×d×log(n) + d×k×n) where n=vectors, d=dimensions, k=4
- **Space**: O(n×d) for temporary storage + O(n/4) for compressed codes

### Query Performance

- **SIMD acceleration**: Processes 8 dimensions per AVX2 instruction
- **Memory efficiency**: ~4x compression ratio with minimal accuracy loss
- **Cache optimization**: Memory-aligned data structures

## Algorithm Details

### Training Process

1. **Data Analysis**: Extract and sort values for each dimension
2. **Segmentation**: Use dynamic programming to find 4 optimal clusters per dimension
3. **Optimization**: Apply PSO to optimize quantization parameters within each cluster
4. **Quantization**: Convert all vectors to compressed 4-level codes
5. **Precomputation**: Generate SIMD-optimized reconstruction parameters

### Distance Computation

1. **Code Loading**: Extract compressed cluster IDs and quantization codes
2. **Parameter Lookup**: Retrieve precomputed reconstruction parameters
3. **SIMD Reconstruction**: Decode 8 dimensions simultaneously using AVX2
4. **Distance Calculation**: Compute squared differences with FMA instructions
5. **Accumulation**: Horizontal sum for final L2 distance

## Technical Specifications

- **Quantization Levels**: 4 levels per dimension (2 bits)
- **Compression Ratio**: ~4x (32-bit float → 8-bit codes + parameters)
- **SIMD Width**: 256-bit AVX2 (8 floats)
- **Memory Alignment**: 256-byte aligned for optimal SIMD performance
- **Parallel Training**: OpenMP-accelerated dimension processing

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](./LICENSE) file for details.

## Contributing

Contributions are welcome! Please ensure your code:

- Follows the existing code style
- Includes appropriate documentation
- Maintains SIMD optimization where applicable
- Passes all tests

## Citation

If you use POUQ in your research, please cite:

```bibtex
@software{xin2025pouq,
  title={POUQ: A High-Precision Quantization via Segmentation and Swarm Optimization},
  author={Xin Hu},
  year={2025},
}
```
