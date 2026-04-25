import math
import random
from typing import List

from pypouq import Quantizer

from baseline import ScalarQuantizer


def random_data_2d_unifrom(x: int, y: int) -> List[List[float]]:
    return [[random.random() for _ in range(y)] for _ in range(x)]


def box_muller_transform() -> float:
    u1 = random.random()
    u2 = random.random()
    z0 = (-2 * math.log(u1)) ** 0.5 * math.cos(2 * math.pi * u2)
    return z0


def random_data_2d_gmm(
    x: int, y: int, n_cluster: int, n_sigma: float
) -> List[List[float]]:
    cluster_centers = [[random.random() for _ in range(y)] for _ in range(n_cluster)]
    data = []
    for _ in range(x):
        cluster = random.randint(0, n_cluster - 1)
        center = cluster_centers[cluster]
        point = [center[i] + n_sigma * box_muller_transform() for i in range(y)]
        data.append(point)
    return data


def run_impl(quantizer, n_sample: int, n_dim: int, data: List[List[float]]):
    quantizer.train(data)

    max_error = 0.0
    mae = 0.0
    mse = 0.0

    for i in range(n_sample):
        decode: List[float] = quantizer.decode(i)
        for j in range(n_dim):
            diff = abs(data[i][j] - decode[j])
            max_error = max(max_error, diff)
            mae += diff
            mse += diff * diff

    n_size = n_sample * n_dim
    print(f"    Max Error: {max_error:.3f}")
    print(f"    MAE: {mae / n_size:.3f}")
    print(f"    MSE: {mse / n_size:.3f}")


def run(n_sample: int, n_dim: int, data: List[List[float]]):
    print("  BaseLine:")
    run_impl(ScalarQuantizer(), n_sample, n_dim, data)
    print("  Ours:")
    run_impl(Quantizer(), n_sample, n_dim, data)


if __name__ == "__main__":
    n_sample = 10000
    n_dim = 256

    unifrom_data = random_data_2d_unifrom(n_sample, n_dim)
    print(f"Unifrom(n_sample={n_sample},n_dim={n_dim}):")
    run(n_sample, n_dim, unifrom_data)

    gmm_cases = [
        (2, 20.0),
        (4, 100.0),
        (4, 20.0),
        (4, 5.0),
        (8, 20.0),
    ]

    for n_cluster, n_sigma in gmm_cases:
        gmm_data = random_data_2d_gmm(n_sample, n_dim, n_cluster, n_sigma)
        print(
            f"GMM(n_sample={n_sample},n_dim={n_dim},n_cluster={n_cluster},n_sigma={n_sigma}):"
        )
        run(n_sample, n_dim, gmm_data)
