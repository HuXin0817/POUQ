import argparse
import math
import random
import time

import numpy as np
from pypouq import Quantizer

from baseline import ScalarQuantizer


def random_data_2d_unifrom(x: int, y: int) -> np.ndarray:
    return np.random.random((x, y))


def box_muller_transform() -> float:
    u1 = random.random()
    u2 = random.random()
    z0 = (-2 * math.log(u1)) ** 0.5 * math.cos(2 * math.pi * u2)
    return z0


def random_data_2d_gmm(x: int, y: int, n_cluster: int, n_sigma: float) -> np.ndarray:
    cluster_centers = np.random.random((n_cluster, y))
    data = np.zeros((x, y))
    for i in range(x):
        cluster = random.randint(0, n_cluster - 1)
        center = cluster_centers[cluster]
        noise = np.array([n_sigma * box_muller_transform() for _ in range(y)])
        data[i] = center + noise
    return data


def run_impl(quantizer, n_sample: int, n_dim: int, data: np.ndarray):
    start_time = time.time()
    quantizer.train(data)
    train_time = time.time() - start_time
    print(f"    Train Time:  {train_time:.3f}s")

    decoded = np.zeros((n_sample, n_dim), dtype=np.float32)
    start_time = time.time()
    for i in range(n_sample):
        quantizer.decode(i, decoded[i])
    decode_time = time.time() - start_time
    print(f"    Decode Time: {decode_time:.3f}s")

    diff = np.abs(data.astype(np.float32) - decoded)

    max_error = np.max(diff)
    mae = np.sum(diff) / (n_sample * n_dim)
    mse = np.sum(diff**2) / (n_sample * n_dim)

    print(f"    Max Error:   {max_error:.3e}")
    print(f"    MAE:         {mae:.3e}")
    print(f"    MSE:         {mse:.3e}")


def run(n_sample: int, n_dim: int, data: np.ndarray):
    print("  BaseLine:")
    run_impl(ScalarQuantizer(), n_sample, n_dim, data)
    print("  Ours:")
    run_impl(Quantizer(), n_sample, n_dim, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sample", type=int, default=10000, help="Number of samples")
    parser.add_argument("--n_dim", type=int, default=256, help="Number of dimensions")
    args = parser.parse_args()
    n_sample = args.n_sample
    n_dim = args.n_dim

    unifrom_data = random_data_2d_unifrom(n_sample, n_dim)
    print(f"Unifrom(sample={n_sample},dim={n_dim}):")
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
            f"GMM(sample={n_sample},dim={n_dim},cluster={n_cluster},sigma={n_sigma}):"
        )
        run(n_sample, n_dim, gmm_data)
