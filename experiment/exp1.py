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
import plsq
from util.io import fvecs_read


def run(name, bitwidth, quantizer, data_f32):
    print(f"{name} bitwidth: {bitwidth}")

    start_time = time.time()
    quantizer.train(data_f32)
    codes = quantizer.compute_codes(data_f32)
    train_time = time.time() - start_time
    print(f"{name} training time: {train_time:.4f} seconds")

    # 计算MSE
    start_time = time.time()
    reconstructed = quantizer.decode(codes)
    decode_time = time.time() - start_time
    print(f"{name} decode time: {decode_time:.4f} seconds")
    if isinstance(reconstructed, list):
        mse = np.mean((data_f32.reshape(-1) - reconstructed) ** 2)
    else:
        mse = np.mean((data_f32 - reconstructed) ** 2)
    print(f"{name} MSE: {mse:.6f}")

    print()


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    data = fvecs_read(f"../data/{dataset_name}/{dataset_name}_base.fvecs")
    query_data = fvecs_read(f"../data/{dataset_name}/{dataset_name}_query.fvecs")

    n, d = data.shape
    data_f32 = data.astype(np.float32)

    print(f"Dataset: {dataset_name}, Shape: {data.shape}")
    print("=" * 50)

    # PLSQ8量化器
    plsq8 = plsq.PLSQQuantizer(c_bit=4, q_bit=4, sub=1)
    run("PLSQ", 8, plsq8, data_f32)

    # Faiss Scalar Quantizer (类似SQ8的基线)
    sq = faiss.ScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit)
    run("SQ", 8, sq, data_f32)

    # Faiss PQ (Product Quantization)
    m = max(1, d // 4)  # 子空间数量
    pq = faiss.ProductQuantizer(d, m, 8)
    run("PQ", 2, pq, data_f32)

    # RaBit Quantizer
    rabitq = faiss.RaBitQuantizer(d, faiss.METRIC_L2)
    run("RaBitQ", 1, rabitq, data_f32)
