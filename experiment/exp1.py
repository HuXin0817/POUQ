import csv
import multiprocessing
import os
import signal
import sys
import time
from multiprocessing import Process, Queue

# Set OpenMP thread count to current device CPU core count
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())

import faiss

# Set faiss to use all available CPU cores
faiss.omp_set_num_threads(os.cpu_count())

print("faiss version:", faiss.__version__)
print(f"Available CPU cores: {os.cpu_count()}")
print(f"faiss OpenMP threads: {faiss.omp_get_max_threads()}")

import numpy as np
import pouq
from util.io import fvecs_read

# List to store results
results = []


# Subprocess training function
def train_worker(name, bitwidth, quantizer, data_f32, result_queue):
    """Execute training in subprocess"""
    try:
        start_time = time.time()

        # Execute training
        quantizer.train(data_f32)
        codes = quantizer.compute_codes(data_f32)

        train_time = time.time() - start_time

        # Calculate MSE
        reconstructed = quantizer.decode(codes)
        if isinstance(reconstructed, list):
            mse = np.mean((data_f32.reshape(-1) - reconstructed) ** 2)
        else:
            mse = np.mean((data_f32 - reconstructed) ** 2)

        # Put result into queue
        result_queue.put(
            {
                "success": True,
                "method": name,
                "bitwidth": bitwidth,
                "train_time": train_time,
                "mse": mse,
            }
        )

    except Exception as e:
        # If exception occurs, also put into queue
        result_queue.put(
            {"success": False, "method": name, "bitwidth": bitwidth, "error": str(e)}
        )


def run(name, bitwidth, quantizer, data_f32):
    """Run function in main process, manages subprocess training"""
    # 12-hour time limit (43200 seconds)
    max_train_time = 12 * 60 * 60  # 12 hours = 43200 seconds

    # Create inter-process communication queue
    result_queue = Queue()

    # Create subprocess
    process = Process(
        target=train_worker, args=(name, bitwidth, quantizer, data_f32, result_queue)
    )

    # print(f"Starting training: {name} (bitwidth={bitwidth})")
    start_time = time.time()

    # Start subprocess
    process.start()

    # Wait for subprocess to complete or timeout
    process.join(timeout=max_train_time)

    if process.is_alive():
        # If subprocess is still running, it means timeout occurred
        print(
            f"Warning: {name} (bitwidth={bitwidth}) training time exceeded 12 hours, forcibly terminating training"
        )

        # Forcibly terminate subprocess
        process.terminate()
        process.join(timeout=5)  # Wait 5 seconds for graceful exit

        if process.is_alive():
            # If still not exited, force kill
            process.kill()
            process.join()

        print(f"{name} (bitwidth={bitwidth}) training has been terminated")
        return

    # Check if subprocess completed normally
    if process.exitcode == 0:
        # Get training results
        try:
            result = result_queue.get_nowait()
            if result["success"]:
                train_time = result["train_time"]
                mse = result["mse"]

                print(f"| {name} | {bitwidth} | {train_time:.4f} | {mse:.6e} |")

                # Add result to list
                results.append(
                    {
                        "method": name,
                        "bitwidth": bitwidth,
                        "train_time": train_time,
                        "mse": mse,
                    }
                )
            else:
                print(
                    f"Error: {name} (bitwidth={bitwidth}) training failed: {result['error']}"
                )
        except:
            print(f"Error: {name} (bitwidth={bitwidth}) unable to get training results")
    else:
        print(
            f"Error: {name} (bitwidth={bitwidth}) subprocess exited abnormally (exit code: {process.exitcode})"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python exp2.py <dataset_name>")
        sys.exit(1)

    dataset_name = sys.argv[1]
    data = fvecs_read(f"../data/{dataset_name}/{dataset_name}_base.fvecs")
    # data = data[:1000]

    n, d = data.shape
    data_f32 = data.astype(np.float32)

    print(f"Dataset: {dataset_name}, Shape: {data.shape}")
    print("=" * 50)

    for i in range(4, 9):
        q = pouq.SQQuantizer(c_bit=0, q_bit=i, sub=d)
        run("SQ", i, q, data_f32)

    for i in range(4, 9):
        q = pouq.OSQQuantizer(c_bit=0, q_bit=i, sub=d)
        run("OSQ-Baseline", i, q, data_f32)

    for i in range(4, 9):
        q = pouq.OSQ2Quantizer(c_bit=0, q_bit=i, sub=d)
        run("OSQ-PSO", i, q, data_f32)

    for i in range(2, 5):
        q = pouq.POUQKMeansQuantizer(c_bit=i, q_bit=4, sub=d)
        run("POUQ-KMeans-MinMax", i + 4, q, data_f32)

    for i in range(2, 5):
        q = pouq.POUQKRangeQuantizer(c_bit=i, q_bit=4, sub=d)
        run("POUQ-KRange-MinMax", i + 4, q, data_f32)

    for i in range(4, 9):
        q = pouq.POUQQuantizer(c_bit=i // 2, q_bit=i - (i // 2), sub=d)
        run("POUQ", i, q, data_f32)

    for M in range(2, d // 4 + 1):
        if d % M > 0:
            continue
        q = faiss.ProductQuantizer(d, M, 8)
        run("PQ", M * 8 / d, q, data_f32)

    for M in range(2, d // 4 + 1):
        if d % M > 0:
            continue
        q = faiss.LocalSearchQuantizer(d, M, 8)
        run("LSQ", M * 8 / d, q, data_f32)

    rabitq = faiss.RaBitQuantizer(d, faiss.METRIC_L2)
    run("RaBitQ", 1, rabitq, data_f32)

    # Create result directory (if it doesn't exist)
    result_dir = "../result"
    os.makedirs(result_dir, exist_ok=True)

    # Save results to CSV file
    csv_file = f"../result/exp1_{dataset_name}.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["method", "bitwidth", "train_time", "mse"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {csv_file}")
