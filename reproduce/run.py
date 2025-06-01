import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from pouq import LloydMaxQuantizer, POUQuantizer, ScaledQuantizer, compute_mse


def read_fvecs(filename, c_contiguous=True) -> np.ndarray:
    print(f"Reading from {filename}.")
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv


def evaluate_method(method: str, data: np.ndarray, quantizer):
    start_time = time.time()
    quantizer.train(data)
    end_time = time.time()
    training_time = end_time - start_time
    mse = compute_mse(data, quantizer)
    print(
        f"Method: {method}, " f"Error: {mse}, " f"Training Time: {training_time:.4f}s"
    )
    return training_time, mse


def run(data: np.ndarray, results_dict):
    N, Dim = data.shape
    print(f"N={N}, Dim={Dim}")

    methods = [
        ("SQ", ScaledQuantizer(q_bit=8, groups=Dim)),
        ("POUQ", POUQuantizer(c_bit=4, q_bit=4, groups=Dim)),
        ("LloydMax", LloydMaxQuantizer(c_bit=8, groups=Dim)),
    ]

    for method_name, quantizer in methods:
        training_time, mse = evaluate_method(method_name, data, quantizer)

        if method_name not in results_dict:
            results_dict[method_name] = {"times": [], "mses": [], "sizes": []}

        results_dict[method_name]["times"].append(training_time)
        results_dict[method_name]["mses"].append(mse)
        results_dict[method_name]["sizes"].append(N)

    print()


def plot_tradeoff(results_dict, dataset_name):
    result_dir = "../result"
    os.makedirs(result_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))

    colors = {"SQ": "blue", "POUQ": "red", "LloydMax": "green"}
    markers = {"SQ": "o", "POUQ": "s", "LloydMax": "^"}

    for method_name, data in results_dict.items():
        plt.scatter(
            data["times"],
            data["mses"],
            c=colors.get(method_name, "black"),
            marker=markers.get(method_name, "o"),
            label=method_name,
            s=60,
            alpha=0.7,
        )

        plt.plot(
            data["times"],
            data["mses"],
            c=colors.get(method_name, "black"),
            alpha=0.3,
            linewidth=1,
        )

    plt.xlabel("Training Time (seconds)")
    plt.ylabel("MSE")
    plt.title(f"Training Time vs MSE Trade-off ({dataset_name})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.yscale("log")
    plt.xscale("log")

    output_path = os.path.join(result_dir, f"{dataset_name}_tradeoff.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Trade-off plot saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <dataset_name>")
        exit(0)
    dataset_name = sys.argv[1]
    data = read_fvecs(f"../data/{dataset_name}/{dataset_name}_base.fvecs")

    N, Dim = data.shape
    results_dict = {}

    i = 100
    while i < N:
        run(data[:i, :], results_dict)
        i *= 10

    run(data, results_dict)
    plot_tradeoff(results_dict, dataset_name)
