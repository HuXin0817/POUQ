import os
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatter
from pouq import (LloydMaxQuantizer, OptimizedScaledQuantizer, POUQuantizer,
                  ScaledQuantizer, compute_mse)


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
        ("OSQ", OptimizedScaledQuantizer(q_bit=8, groups=Dim)),
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

    min_threshold = 1e-9
    marker_config = {
        "SQ": {
            "color": "#1f77b4",
            "marker": "o",
            "linestyles": ["-", "--", ":"],
            "fill_styles": ["full", "full", "full"],
        },
        "OSQ": {
            "color": "#ff7f0e",
            "marker": "s",
            "linestyles": ["-", "--", ":"],
            "fill_styles": ["full", "full", "full"],
        },
        "POUQ": {
            "color": "#2ca02c",
            "marker": "^",
            "linestyles": ["-", "--", ":"],
            "fill_styles": ["full", "full", "full"],
        },
        "LloydMax": {
            "color": "#d62728",
            "marker": "D",
            "linestyles": ["-", "--", ":"],
            "fill_styles": ["full", "full", "full"],
        },
    }

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(12, 7))

    legend_handles = []

    has_zero_mse = False
    min_nonzero_mse = float("inf")
    max_mse = 0
    all_nonzero_mses = []

    for method_name, data in results_dict.items():
        for mse in data["mses"]:
            if mse == 0:
                has_zero_mse = True
            elif mse > 0:
                min_nonzero_mse = min(min_nonzero_mse, mse)
                max_mse = max(max_mse, mse)
                all_nonzero_mses.append(mse)

    if has_zero_mse and all_nonzero_mses:
        log_min = np.log10(min_nonzero_mse)
        log_max = np.log10(max_mse)
        log_range = log_max - log_min

        if log_range > 4:
            zero_replacement = 10 ** (log_min - 1)
        elif log_range > 2:
            zero_replacement = 10 ** (log_min - 0.5)
        elif log_range > 1:
            zero_replacement = 10 ** (log_min - 0.3)
        else:
            zero_replacement = 10 ** (log_min - 0.1)

        # 确保零MSE阈值不会过小，至少为最小非零MSE的1/1000
        zero_replacement = max(zero_replacement, min_nonzero_mse / 1000)
    else:
        zero_replacement = (
            min_nonzero_mse / 100 if min_nonzero_mse != float("inf") else min_threshold
        )

    for method_name, data in results_dict.items():
        times = data["times"]
        adjusted_mses = [mse if mse > 0 else zero_replacement for mse in data["mses"]]

        if not times:
            continue

        config = marker_config.get(
            method_name,
            {
                "color": "#000000",
                "marker": "o",
                "linestyles": ["-"],
                "fill_styles": ["full"],
            },
        )

        (line,) = ax.plot(
            times,
            adjusted_mses,
            label=method_name,
            color=config["color"],
            marker=config["marker"],
            markersize=8,
            linewidth=2,
            linestyle=config["linestyles"][0],
            alpha=0.8,
        )

        legend_handles.append(line)

        fill_style = config["fill_styles"][0]
        if fill_style == "full":
            line.set_markerfacecolor(config["color"])
            line.set_markeredgecolor(config["color"])
        elif fill_style == "none":
            line.set_markerfacecolor("white")
            line.set_alpha(0.9)
            line.set_markeredgecolor(config["color"])
        elif fill_style == "top":
            line.set_markerfacecolor(config["color"])
            line.set_alpha(0.5)
            line.set_markeredgecolor(config["color"])

    if has_zero_mse:
        ax.axhline(
            y=zero_replacement,
            color="#00008B",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="Zero MSE Threshold",
            zorder=-1,
        )

        xlim = ax.get_xlim()
        ax.text(
            x=xlim[1] * 0.95,
            y=zero_replacement * 1.1,
            s="Zero MSE Threshold",
            color="#00008B",
            fontsize=14,
            fontweight="bold",
            verticalalignment="bottom",
            horizontalalignment="right",
            zorder=-1,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    class ExponentOnlyFormatter(LogFormatter):
        def __call__(self, x, pos=None):
            if x <= 0:
                return ""
            exponent = int(np.log10(x))
            return f"{exponent}"

    ax.xaxis.set_major_formatter(ExponentOnlyFormatter())
    ax.yaxis.set_major_formatter(ExponentOnlyFormatter())

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.tick_params(axis="both", which="minor", labelsize=16)

    ax.grid(True, which="major", linestyle="-", linewidth=2, alpha=0.3)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.2)

    ax.set_xlabel(r"Training Time ($\log_{10}$ seconds)", fontsize=20)
    ax.set_ylabel(r"MSE Error ($\log_{10}$)", fontsize=20)
    ax.set_title(dataset_name, fontsize=20)

    plt.subplots_adjust(right=0.75)

    output_path = os.path.join(result_dir, f"{dataset_name}_tradeoff_plot.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Trade-off plot saved to: {output_path}")
    subprocess.run(["open", output_path])

    fig_legend = plt.figure(figsize=(12, 1.5))
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis("off")

    legend_labels = [line.get_label() for line in legend_handles]
    if has_zero_mse:
        legend_labels.append("Zero MSE Threshold")

    legend = fig_legend.legend(
        handles=legend_handles + ([ax.lines[-1]] if has_zero_mse else []),
        labels=legend_labels,
        loc="center",
        fontsize=12,
        frameon=False,
        ncol=len(legend_labels),
    )

    legend_path = os.path.join(result_dir, f"{dataset_name}_legend.pdf")
    plt.savefig(legend_path, dpi=300, bbox_inches="tight")


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

    plot_tradeoff(results_dict, dataset_name)
