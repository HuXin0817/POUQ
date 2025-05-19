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
                raise IndexError()
            if item < 0 or item >= self._qvector.shape(0):
                raise IndexError()
            return self._qvector.at(item)
        elif isinstance(item, tuple) or isinstance(item, list):
            if len(item) != self.ndim():
                raise IndexError()
            for i in range(self.ndim()):
                if item[i] < 0 or item[i] >= self._qvector.shape(i):
                    raise IndexError()

            index = 0
            mul = 1
            for i in range(self.ndim() - 1, -1, -1):
                index += item[i] * mul
                mul *= self.shape(i)
            return self._qvector.at(index)
        else:
            raise TypeError("Invalid index type")

    def __init__(
        self,
        data: numpy.ndarray,
        c_bit: int,
        q_bit: int,
        learn_step_size: bool = True,
        max_iter: int = 128,
        grid_side_length: int = 8,
        grid_scale_factor: float = 0.1,
        initial_inertia: float = 0.9,
        final_inertia: float = 0.4,
        c1: float = 1.8,
        c2: float = 1.8,
    ):
        if learn_step_size:
            if max_iter < 1:
                raise ValueError("Invalid learn_step_size")
            if grid_side_length < 1:
                raise ValueError("Invalid grid_side_length")
            if grid_scale_factor <= 0:
                raise ValueError("Invalid grid_scale_factor")
            if grid_scale_factor > 0.5:
                grid_scale_factor = 0.4999
            if initial_inertia < 0:
                raise ValueError("Invalid initial_inertia")
            if final_inertia < 0:
                raise ValueError("Invalid final_inertia")
            if c1 < 0 or c2 < 0:
                raise ValueError("Invalid c1 or c2")
        self._qvector = pouq.QVector(
            data=data,
            c_bit=c_bit,
            q_bit=q_bit,
            learn_step_size=learn_step_size,
            max_iter=max_iter,
            grid_side_length=grid_side_length,
            grid_scale_factor=grid_scale_factor,
            initial_inertia=initial_inertia,
            final_inertia=final_inertia,
            c1=c1,
            c2=c2,
        )
