# POUQ Python Bindings

This directory contains Python bindings for the POUQ C library.

## Installation

### Prerequisites

- Python 3.6 or higher
- NumPy
- GCC compiler with AVX2 and FMA support
- OpenMP support

### Build and Install

```bash
cd python
pip install -e .
```

Or to build the shared library manually:

```bash
cd python
python setup.py build_ext
```

The shared library (`libpouq.so` on Linux, `libpouq.dylib` on macOS) will be built and placed in the `pouq` package directory.

## Usage

### Basic Example

```python
import numpy as np
import pouq

# Prepare data
dim = 128
n_samples = 10000
data = np.random.randn(n_samples, dim).astype(np.float32)

# Train the model
code, rec_para = pouq.train(dim, data)

# Calculate distance
query = np.random.randn(dim).astype(np.float32)
dist = pouq.distance(dim, code, rec_para, query.flatten())

print(f"Distance: {dist}")
```

### With Custom Parameters

```python
import numpy as np
import pouq

# Create custom training parameters
parameter = pouq.Parameter(
    max_iter=200,
    particle_count=50,
    scale_factor=0.15,
    init_inertia=0.95,
    final_inertia=0.3,
    c1=2.5,
    c2=2.5,
)

# Train with custom parameters
dim = 128
data = np.random.randn(10000, dim).astype(np.float32)
code, rec_para = pouq.train(dim, data, parameter=parameter)
```

### Utility Functions

```python
import numpy as np
import pouq

# Get sorted data for a specific dimension
data = np.random.randn(1000, 64).astype(np.float32)
sorted_data = pouq.get_sorted_data(data, d=0, dim=64)

# Count frequency
data_map, freq_map = pouq.count_freq(sorted_data)

# Segment data
lowers, uppers = pouq.segment(data_map, freq_map, do_count_freq=True)

# Optimize bounds
bound = pouq.optimize(
    init_lower=0.0,
    init_upper=1.0,
    data_map=data_map,
    freq_map=freq_map,
    do_count_freq=True,
)
print(f"Optimized bounds: [{bound.lower}, {bound.upper}]")
```

## API Reference

### Main Functions

#### `train(dim, data, parameter=None)`

Train a POUQ model.

**Parameters:**

- `dim` (int): Dimension of the data
- `data` (np.ndarray): Input data array with shape `[n_samples, dim]`
- `parameter` (Parameter, optional): Training parameters

**Returns:**

- `code` (np.ndarray): CodeUnit array with shape `[n_samples // 8, 3]`
- `rec_para` (np.ndarray): RecPara array with shape `[dim * 64, 8]`

#### `distance(dim, code, rec_para, data, offset=0)`

Calculate distance using POUQ code.

**Parameters:**

- `dim` (int): Dimension of the data
- `code` (np.ndarray): CodeUnit array
- `rec_para` (np.ndarray): RecPara array
- `data` (np.ndarray): Input data array (flattened)
- `offset` (int): Offset in the data array

**Returns:**

- `float`: Distance value

#### `decode(dim, code, rec_para, dist, offset=0)`

Decode using POUQ code.

**Parameters:**

- `dim` (int): Dimension of the data
- `code` (np.ndarray): CodeUnit array
- `rec_para` (np.ndarray): RecPara array
- `dist` (np.ndarray): Output distance array (modified in place)
- `offset` (int): Offset in the distance array

### Utility Functions

#### `get_sorted_data(data, d, dim)`

Get sorted data for a specific dimension.

#### `count_freq(sorted_data)`

Count frequency of sorted data.

#### `segment(data_map, freq_map, do_count_freq=True)`

Segment data.

#### `optimize(init_lower, init_upper, data_map, freq_map, parameter=None, do_count_freq=True)`

Optimize bounds.

### Data Structures

#### `Parameter`

Training parameters structure:

- `max_iter` (int): Maximum iterations
- `particle_count` (int): Number of particles
- `scale_factor` (float): Scale factor
- `init_inertia` (float): Initial inertia
- `final_inertia` (float): Final inertia
- `c1` (float): Cognitive parameter
- `c2` (float): Social parameter

#### `Bound`

Bound structure:

- `lower` (float): Lower bound
- `upper` (float): Upper bound

## Notes

- The library requires x86_64 architecture with AVX2 and FMA support
- All data arrays should be `float32` (np.float32)
- The library uses OpenMP for parallelization

## Troubleshooting

### Library Not Found

If you get an error about the library not being found:

1. Make sure you've built the library: `python setup.py build_ext`
2. Check that the shared library exists in the `pouq` directory
3. On macOS, you may need to set `DYLD_LIBRARY_PATH`

### Compilation Errors

If you encounter compilation errors:

1. Ensure you have GCC with AVX2 and FMA support
2. Check that OpenMP is installed
3. Verify your CPU supports AVX2 and FMA instructions
