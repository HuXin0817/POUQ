from typing import List


class ScalarQuantizer:
    def __init__(self, k_level: int = 15):
        self.k_level: int = k_level
        self.n_dim: int = 0
        self.n_sample: int = 0
        self.lower: List[float] = []
        self.step_size: List[float] = []
        self.code: List[List[int]] = []

    def train(self, data: List[List[float]]) -> None:
        self.n_sample = len(data)
        if self.n_sample == 0:
            return

        self.n_dim = len(data[0])
        if self.n_dim == 0:
            return

        if not all(len(vec) == self.n_dim for vec in data):
            raise ValueError()

        self.lower = [float("inf")] * self.n_dim
        upper = [float("-inf")] * self.n_dim

        for vec in data:
            for i in range(self.n_dim):
                val = vec[i]
                if val < self.lower[i]:
                    self.lower[i] = val
                if val > upper[i]:
                    upper[i] = val

        self.step_size = [0.0] * self.n_dim
        for i in range(self.n_dim):
            rng = upper[i] - self.lower[i]
            self.step_size[i] = rng / self.k_level if rng > 0 else 1.0

        self.code = [[0] * self.n_dim for _ in range(self.n_sample)]

        for j, vec in enumerate(data):
            for i in range(self.n_dim):
                idx = int((vec[i] - self.lower[i]) / self.step_size[i] + 0.5)
                self.code[j][i] = max(0, min(self.k_level - 1, idx))

    def decode(self, n: int) -> List[float]:
        return [
            self.lower[i] + (self.code[n][i] + 0.5) * self.step_size[i]
            for i in range(self.n_dim)
        ]
