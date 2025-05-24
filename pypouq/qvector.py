import numpy
import pouq


class QVector:
    def ndim(self) -> int:
        return self._qvector.ndim()

    def shape(self, i: int):
        if i < 0 or i >= self.ndim():
            raise ValueError(f"QVector dimension out of range: {i}")
        return self._qvector.shape(i)

    def __getitem__(self, item):
        if isinstance(item, int):
            if self.ndim() != 1:
                raise IndexError("Invalid index: 1D index used on non-1D QVector")
            if item < 0 or item >= self._qvector.shape(0):
                raise IndexError(
                    f"Index {item} out of range for QVector with shape ({self._qvector.shape(0)})"
                )
            return self._qvector.at(item)
        elif isinstance(item, tuple) or isinstance(item, list):
            if len(item) != self.ndim():
                raise IndexError(
                    f"Index tuple length {len(item)} does not match QVector dimension {self.ndim()}"
                )
            for i in range(self.ndim()):
                if item[i] < 0 or item[i] >= self._qvector.shape(i):
                    raise IndexError(
                        f"Index {item[i]} out of range for dimension {i} with size {self._qvector.shape(i)}"
                    )

            index = 0
            mul = 1
            for i in range(self.ndim() - 1, -1, -1):
                index += item[i] * mul
                mul *= self.shape(i)
            return self._qvector.at(index)
        else:
            raise TypeError("Invalid index type. Must be int or tuple of ints")

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
        if optimize_bound:
            if max_iter < 1:
                raise ValueError("Invalid optimize_bound: max_iter must be at least 1")
            if grid_side_length < 1:
                raise ValueError("Invalid grid_side_length: must be at least 1")
            if grid_scale_factor <= 0:
                raise ValueError("Invalid grid_scale_factor: must be positive")
            if grid_scale_factor > 0.5:
                grid_scale_factor = 0.4999
            if initial_inertia < 0:
                raise ValueError("Invalid initial_inertia: must be non-negative")
            if final_inertia < 0:
                raise ValueError("Invalid final_inertia: must be non-negative")
            if c1 < 0 or c2 < 0:
                raise ValueError("Invalid c1 or c2: must be non-negative")
        self._qvector = pouq.QVector(
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
