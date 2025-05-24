import numpy
import pouq

__doc__ = pouq.__doc__


class QVector(pouq.QVector):
    """
    Quantum vector class inheriting from pouq.QVector, providing multidimensional indexing
    and parameter optimization capabilities
    """

    def ndim(self) -> int:
        """Return the number of dimensions of the vector"""
        return super().ndim()

    def shape(self, i: int) -> int:
        """Return the size of the specified dimension"""
        if i < 0 or i >= self.ndim():
            raise ValueError(f"Dimension index out of range: {i}")
        return super().shape(i)

    def __getitem__(self, item):
        """Support integer indexing and multidimensional tuple indexing"""
        if isinstance(item, int):
            if self.ndim() != 1:
                raise IndexError("Integer index used on non-1D vector")
            if item < 0 or item >= self.shape(0):
                raise IndexError(
                    f"Index {item} out of range (dimension size: {self.shape(0)})"
                )
            return super().at(item)

        elif isinstance(item, (tuple, list)):
            if len(item) != self.ndim():
                raise IndexError(
                    f"Index dimension ({len(item)}) does not match vector dimension ({self.ndim()})"
                )

            for i, idx in enumerate(item):
                if idx < 0 or idx >= self.shape(i):
                    raise IndexError(
                        f"Index {idx} out of range for dimension {i} (size: {self.shape(i)})"
                    )

            linear_idx = 0
            multiplier = 1
            for i in range(self.ndim() - 1, -1, -1):
                linear_idx += item[i] * multiplier
                multiplier *= self.shape(i)

            return super().at(linear_idx)

        else:
            raise TypeError("Index must be int or tuple of ints")

    def __init__(
        self,
        data: numpy.ndarray,
        c_bit: int,
        q_bit: int,
        optimize_bound: bool = True,
        max_iter: int = 128,
        grid_side_length: int = 8,
        grid_scale_factor: float = 0.1,
        initial_inertia: float = 0.9,
        final_inertia: float = 0.4,
        c1: float = 1.8,
        c2: float = 1.8,
    ):
        """
        Initialize a quantum vector

        Parameters:
            data: Input numpy array
            c_bit: Number of classical bits
            q_bit: Number of quantum bits
            optimize_bound: Whether to optimize bounds
            max_iter: Maximum number of iterations
            grid_side_length: Grid side length
            grid_scale_factor: Grid scaling factor
            initial_inertia: Initial inertia weight
            final_inertia: Final inertia weight
            c1, c2: Learning factors
        """
        if optimize_bound:
            if max_iter < 1:
                raise ValueError("Maximum iterations must be at least 1")
            if grid_side_length < 1:
                raise ValueError("Grid side length must be at least 1")
            if grid_scale_factor <= 0:
                raise ValueError("Grid scale factor must be positive")
            if grid_scale_factor > 0.5:
                grid_scale_factor = 0.4999  # Avoid exact 0.5
            if initial_inertia < 0:
                raise ValueError("Initial inertia must be non-negative")
            if final_inertia < 0:
                raise ValueError("Final inertia must be non-negative")
            if c1 < 0 or c2 < 0:
                raise ValueError("Learning factors must be non-negative")

        super().__init__(
            data=data,
            c_bit=c_bit,
            q_bit=q_bit,
            optimize_bound=optimize_bound,
            max_iter=max_iter,
            grid_side_length=grid_side_length,
            grid_scale_factor=grid_scale_factor,
            initial_inertia=initial_inertia,
            final_inertia=final_inertia,
            c1=c1,
            c2=c2,
        )
