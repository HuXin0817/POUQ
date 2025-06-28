import json
import os
import sys
import time

import numpy as np
from pouq import *


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
    print(f"Method: {method}, Error: {mse}, Training Time: {training_time:.4f}s")
    return training_time, mse


def run(data: np.ndarray, results_dict):
    N, Dim = data.shape
    print(f"N={N}, Dim={Dim}")

    methods = [
        ("SQ", SQQuantizer(q_bit=8, groups=Dim)),
        ("OSQ", OSQQuantizer(q_bit=8, groups=Dim)),
        ("POUQ", POUQQuantizer(c_bit=4, q_bit=4, groups=Dim)),
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


def save_results_to_json(results_dict, dataset_name):
    result_dir = "../result"
    os.makedirs(result_dir, exist_ok=True)
    output_path = os.path.join(result_dir, f"{dataset_name}_results.json")
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"Results saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <dataset_name>")
        exit(0)
    dataset_name = sys.argv[1]
    data = read_fvecs(f"../data/{dataset_name}/{dataset_name}_base.fvecs")

    N, Dim = data.shape
    results_dict = {}

    log_min = np.log10(1000)
    log_max = np.log10(N)
    log_samples = np.linspace(log_min, log_max, 10)

    for log_sample in log_samples:
        sample_size = int(10**log_sample)
        sample_size = min(sample_size, N)
        run(data[:sample_size, :], results_dict)

    save_results_to_json(results_dict, dataset_name)
