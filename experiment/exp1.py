import os
import sys
import time

# 设置OpenMP线程数为当前设备CPU核心数
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())

import faiss

# 设置faiss使用所有可用CPU核心
faiss.omp_set_num_threads(os.cpu_count())

print("faiss version:", faiss.__version__)
print(f"Available CPU cores: {os.cpu_count()}")
print(f"faiss OpenMP threads: {faiss.omp_get_max_threads()}")

import numpy as np
import posq
from util.io import fvecs_read


def run(name, bitwidth, quantizer, data_f32):
    start_time = time.time()
    quantizer.train(data_f32)
    codes = quantizer.compute_codes(data_f32)
    train_time = time.time() - start_time

    # 计算MSE
    reconstructed = quantizer.decode(codes)
    if isinstance(reconstructed, list):
        mse = np.mean((data_f32.reshape(-1) - reconstructed) ** 2)
    else:
        mse = np.mean((data_f32 - reconstructed) ** 2)

    print(f"| {name} | {bitwidth} | {train_time:.4f} | {mse:.6e} |")


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    data = fvecs_read(f"../data/{dataset_name}/{dataset_name}_base.fvecs")
    data = data[:10000]

    n, d = data.shape
    data_f32 = data.astype(np.float32)

    print(f"Dataset: {dataset_name}, Shape: {data.shape}")
    print("=" * 50)

    for i in range(4, 9):
        q = posq.SQQuantizer(c_bit=0, q_bit=i, sub=d)
        run("SQ", i, q, data_f32)

    for i in range(4, 9):
        q = posq.LSQQuantizer(c_bit=0, q_bit=i, sub=d)
        run("OSQ-Baseline", i, q, data_f32)

    for i in range(4, 9):
        q = posq.LSQ2Quantizer(c_bit=0, q_bit=i, sub=d)
        run("OSQ-PSO", i, q, data_f32)

    for i in range(2, 5):
        q = posq.PLSQKMeansQuantizer(c_bit=i, q_bit=4, sub=d)
        run("POSQ-KMeans-MinMax", i + 4, q, data_f32)

    for i in range(2, 5):
        q = posq.PLSQKRangeQuantizer(c_bit=i, q_bit=4, sub=d)
        run("PLSQ-KRange-MinMax", i + 4, q, data_f32)

    for i in range(4, 9):
        q = posq.PLSQQuantizer(c_bit=i // 2, q_bit=i - (i // 2), sub=d)
        run("PLSQ (Ours)", i, q, data_f32)

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
