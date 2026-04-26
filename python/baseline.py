import numpy as np


class ScalarQuantizer:
    def __init__(self, k_level: int = 15):
        self.k_level: int = k_level
        self.n_dim: int = 0
        self.n_sample: int = 0
        self.lower: np.ndarray = None
        self.step_size: np.ndarray = None
        self.code: np.ndarray = None

    def train(self, data: np.ndarray) -> None:
        self.n_sample = data.shape[0]
        if self.n_sample == 0:
            return

        self.n_dim = data.shape[1]
        if self.n_dim == 0:
            return

        self.lower = np.full(self.n_dim, np.inf)
        upper = np.full(self.n_dim, -np.inf)

        for vec in data:
            vec_lower = vec < self.lower
            vec_upper = vec > upper
            self.lower = np.where(vec_lower, vec, self.lower)
            upper = np.where(vec_upper, vec, upper)

        rng = upper - self.lower
        self.step_size = np.where(rng > 0, rng / self.k_level, 1.0)

        self.code = np.zeros((self.n_sample, self.n_dim), dtype=int)

        for j, vec in enumerate(data):
            idx = ((vec - self.lower) / self.step_size + 0.5).astype(int)
            idx = np.clip(idx, 0, self.k_level - 1)
            self.code[j] = idx

    def decode(self, n: int, arr: np.ndarray) -> None:
        arr[:] = self.lower + (self.code[n] + 0.5) * self.step_size
