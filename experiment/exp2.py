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
import posq

from util.io import fvecs_read


def calculate_recall(gt_indices, search_indices, k):
    """计算recall@k"""
    if len(search_indices) == 0:
        return 0.0

    # 确保我们只比较前k个结果
    gt_k = gt_indices[:k]
    search_k = search_indices[:k]

    # 计算交集
    intersection = len(set(gt_k) & set(search_k))
    return intersection / k


def get_ground_truth(data, queries, k):
    """使用brute force搜索获取ground truth"""
    print("Computing ground truth using brute force search...")

    # 使用faiss的IndexFlatL2进行精确搜索
    index_flat = faiss.IndexFlatL2(data.shape[1])
    index_flat.add(data.astype("float32"))

    _, gt_indices = index_flat.search(queries.astype("float32"), k)
    return gt_indices


def benchmark_index(index, queries, gt_indices, k, index_name, search_param=None):
    """对索引进行基准测试"""
    param_str = f" (param={search_param})" if search_param is not None else ""
    print(f"\nBenchmarking {index_name}{param_str}...")

    # 设置搜索参数
    if search_param is not None:
        if hasattr(index, "nprobe"):
            index.nprobe = search_param
        elif hasattr(index, "hnsw") and hasattr(index.hnsw, "efSearch"):
            index.hnsw.efSearch = search_param

    # QPS测试
    start_time = time.time()

    # 为保证公平性，所有索引都使用单个查询向量的方式
    all_results = []
    for i in range(len(queries)):
        if index_name.startswith("POSQ"):
            result_indices = index.search(queries[i].astype("float32"), k, search_param)
            # print(result_indices)
            # print(gt_indices[i])
            all_results.append(result_indices)
        else:
            _, result_indices = index.search(queries[i : i + 1].astype("float32"), k)
            all_results.append(result_indices[0])  # 取出第一个结果

    search_results = np.array(
        [r[:k] if len(r) >= k else r + [0] * (k - len(r)) for r in all_results]
    )

    end_time = time.time()

    # 计算QPS
    total_time = end_time - start_time
    qps = len(queries) / total_time

    # 计算recall
    recalls = []
    for i in range(len(queries)):
        recall = calculate_recall(gt_indices[i], search_results[i], k)
        recalls.append(recall)

    avg_recall = np.mean(recalls)

    print(f"{index_name}{param_str} Results:")
    print(f"  QPS: {qps:.2f}")
    print(f"  Recall@{k}: {avg_recall:.4f}")
    print(f"  Total time: {total_time:.4f}s")

    return qps, avg_recall, total_time


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python exp2.py <dataset_name>")
        sys.exit(1)

    dataset_name = sys.argv[1]

    # 读取数据
    print(f"Loading dataset: {dataset_name}")
    data = fvecs_read(f"../data/{dataset_name}/{dataset_name}_base.fvecs")
    query_data = fvecs_read(f"../data/{dataset_name}/{dataset_name}_query.fvecs")

    # 为了快速测试，使用较小的数据集
    data = data[:10000]
    query_data = query_data[:100]

    print(f"Data shape: {data.shape}")
    print(f"Query shape: {query_data.shape}")

    k = 10  # 搜索top-k

    # 获取ground truth
    gt_indices = get_ground_truth(data, query_data, k)

    # 准备结果存储
    results = []

    # 1. 测试POSQ IVF索引 - 不同nprobe参数
    print("\n=== Testing POSQ IVF Index with different nprobe values ===")
    try:
        posq_index = posq.IvfIndex(nlist=100, dim=data.shape[1])
        posq_index.train(data.astype("float32"))

        # 测试不同的nprobe值（对应POSQ的搜索参数）
        nprobe_values = [5, 10, 20, 50]
        for nprobe in nprobe_values:
            qps, recall, total_time = benchmark_index(
                posq_index, query_data, gt_indices, k, "POSQ-IVF", nprobe
            )
            results.append(
                [f"POSQ-IVF", qps, recall, total_time, nprobe]
            )
    except Exception as e:
        print(f"POSQ IVF test failed: {e}")

    # 2. 测试Faiss IndexIVFSQ - 不同nprobe参数
    print("\n=== Testing Faiss IndexIVFSQ with different nprobe values ===")
    try:
        nlist = 100
        # Use ScalarQuantizer type instead of separate m and nbits
        qtype = faiss.ScalarQuantizer.QT_8bit  # 8-bit scalar quantization

        quantizer = faiss.IndexFlatL2(data.shape[1])
        index_ivfsq = faiss.IndexIVFScalarQuantizer(
            quantizer, data.shape[1], nlist, qtype
        )
        index_ivfsq.train(data.astype("float32"))
        index_ivfsq.add(data.astype("float32"))

        # 测试不同的nprobe值
        nprobe_values = [5, 10, 20, 50]
        for nprobe in nprobe_values:
            qps, recall, total_time = benchmark_index(
                index_ivfsq, query_data, gt_indices, k, "IVFSQ", nprobe
            )
            results.append([f"IVFSQ", qps, recall, total_time, nprobe])
    except Exception as e:
        print(f"Faiss IVFSQ test failed: {e}")


    # 3. 测试Faiss IndexHNSW - 不同ef_search参数
    print("\n=== Testing Faiss IndexHNSW with different ef_search values ===")
    try:
        M = 16  # HNSW参数
        index_hnsw = faiss.IndexHNSWFlat(data.shape[1], M)
        index_hnsw.add(data.astype("float32"))

        # 测试不同的ef_search值
        ef_search_values = [ 32, 64, 128, 256, 512]
        for ef_search in ef_search_values:
            qps, recall, total_time = benchmark_index(
                index_hnsw, query_data, gt_indices, k, "HNSW", ef_search
            )
            results.append([f"HNSW", qps, recall, total_time, ef_search])
    except Exception as e:
        print(f"Faiss HNSW test failed: {e}")

    # 4. 测试Faiss IndexIVFPQFastScan - 不同nprobe参数
    print("\n=== Testing Faiss IndexIVFPQFastScan with different nprobe values ===")
    try:
        nlist = 100
        m = 4  # 子量化器数量
        nbits = 4  # 每个子量化器的位数

        quantizer = faiss.IndexFlatL2(data.shape[1])
        index_ivfpqfs = faiss.IndexIVFPQFastScan(
            quantizer, data.shape[1], nlist, m, nbits
        )
        index_ivfpqfs.train(data.astype("float32"))
        index_ivfpqfs.add(data.astype("float32"))

        # 测试不同的nprobe值
        nprobe_values = [1, 5, 10, 20, 50]
        for nprobe in nprobe_values:
            qps, recall, total_time = benchmark_index(
                index_ivfpqfs, query_data, gt_indices, k, "IVFPQFastScan", nprobe
            )
            results.append(
                [f"IVFPQFastScan", qps, recall, total_time, nprobe]
            )
    except Exception as e:
        print(f"Faiss IVFPQFastScan test failed: {e}")

    # 5. 测试Faiss IndexIVFRaBitQ - 不同nprobe参数 (作为基准)
    print("\n=== Testing Faiss IndexIVFRaBitQ with different nprobe values ===")
    try:
        nlist = 100
        quantizer = faiss.IndexFlatL2(data.shape[1])
        index_ivf = faiss.IndexIVFRaBitQ(quantizer, data.shape[1], nlist)
        index_ivf.train(data.astype("float32"))
        index_ivf.add(data.astype("float32"))

        # 测试不同的nprobe值
        nprobe_values = [5, 10, 20, 50]
        for nprobe in nprobe_values:
            qps, recall, total_time = benchmark_index(
                index_ivf, query_data, gt_indices, k, "IndexIVFRaBitQ", nprobe
            )
            results.append(
                [f"IndexIVFRaBitQ", qps, recall, total_time, nprobe]
            )
    except Exception as e:
        print(f"Faiss IndexIVFRaBitQ test failed: {e}")

    # Create result directory (if it doesn't exist)
    result_dir = "../result"
    os.makedirs(result_dir, exist_ok=True)

    # 保存结果到CSV文件
    output_file = f"../result/exp2_{dataset_name}.csv"
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["Index", "QPS", f"Recall@{k}", "Total_Time(s)", "Search_Param"]
        )
        writer.writerows(results)

    print(f"\n=== Summary ===")
    print(f"Results saved to: {output_file}")
    print("\nPerformance Comparison:")
    print(
        f"{'Index':<25} {'QPS':<10} {'Recall@'+str(k):<12} {'Time(s)':<10} {'Param':<8}"
    )
    print("-" * 70)
    for result in results:
        print(
            f"{result[0]:<25} {result[1]:<10.2f} {result[2]:<12.4f} {result[3]:<10.4f} {result[4]:<8}"
        )

    # 生成参数扫描分析
    print("\n=== Parameter Sweep Analysis ===")
    
    # 按索引类型分组分析
    index_types = {}
    for result in results:
        # 直接使用索引名称作为分组依据，不进行分割
        index_base = result[0]  # 直接使用完整的索引名称
        
        if index_base not in index_types:
            index_types[index_base] = []
        index_types[index_base].append(result)
    
    for index_type, type_results in index_types.items():
        print(f"\n{index_type} Parameter Analysis:")
        print(f"{'Param':<8} {'QPS':<10} {'Recall':<10} {'QPS/Recall Trade-off':<20}")
        print("-" * 50)
        for result in sorted(type_results, key=lambda x: x[4]):
            trade_off = result[1] / result[2] if result[2] > 0 else 0
            print(
                f"{result[4]:<8} {result[1]:<10.2f} {result[2]:<10.4f} {trade_off:<20.2f}"
            )
