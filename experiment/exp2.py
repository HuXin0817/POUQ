import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

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


def calculate_distance_ratio(gt_distances, search_distances, k):
    """计算L2距离比的精度指标 - 使用 1-abs(1-Dist_Ratio) 公式"""
    if len(search_distances) == 0:
        return 0.0

    # 确保我们只比较前k个结果
    gt_k = gt_distances[:k]
    search_k = search_distances[:k]

    # 计算原始距离比：搜索结果的平均距离 / 真实最近邻的平均距离
    if np.mean(gt_k) == 0:
        raw_ratio = 1.0 if np.mean(search_k) == 0 else float("inf")
    else:
        raw_ratio = np.mean(search_k) / np.mean(gt_k)

    # 转换为精度指标：1 - abs(1 - ratio)
    # 当ratio=1时，精度=1（完美）
    # 当ratio偏离1时，精度降低
    if raw_ratio == float("inf"):
        return 0.0

    precision_score = 1 - abs(1 - raw_ratio)
    return max(0.0, precision_score)  # 确保不为负数


def get_ground_truth(data, queries, k):
    """使用brute force搜索获取ground truth，返回索引和距离"""
    print("Computing ground truth using brute force search...")

    # 使用faiss的IndexFlatL2进行精确搜索
    index_flat = faiss.IndexFlatL2(data.shape[1])
    index_flat.add(data.astype("float32"))

    gt_distances, gt_indices = index_flat.search(queries.astype("float32"), k)
    return gt_indices, gt_distances


def calculate_memory_usage(index, data, index_name):
    """计算索引的内存占用（MB）"""
    # 原始数据大小（未压缩）
    original_size_mb = data.nbytes / (1024 * 1024)

    if index_name == "IVFPOUQ":
        # POUQ索引的压缩比估算（基于8位量化）
        compression_ratio = 0.25  # 8位量化约为原始32位的1/4
        memory_mb = original_size_mb * compression_ratio
    elif index_name == "IvfSQ8Index":
        # IvfSQ8Index 8位标量量化
        compression_ratio = 0.25
        memory_mb = original_size_mb * compression_ratio
    elif index_name == "IVFPQ":
        # IVFPQ 4个子量化器，每个8位
        compression_ratio = 0.125  # 4*8/32 = 1/4，但PQ更高效
        memory_mb = original_size_mb * compression_ratio
    elif index_name == "HNSW":
        # HNSW存储完整向量加图结构
        compression_ratio = 1.05  # 比原始数据稍大（包含图结构）
        memory_mb = original_size_mb * compression_ratio
    elif index_name == "IndexIVFRaBitQ":
        # RaBitQ使用1位量化
        compression_ratio = 1 / 32.0  # 1/32
        memory_mb = original_size_mb * compression_ratio
    else:
        # 默认情况
        compression_ratio = 1.0
        memory_mb = original_size_mb

    return memory_mb


def benchmark_index(
    index, queries, data, gt_indices, gt_distances, k, index_name, search_param=None
):
    """对索引进行基准测试，包括召回率、距离比和内存占用评估"""
    param_str = f" (param={search_param})" if search_param is not None else ""
    print(f"\nBenchmarking {index_name}{param_str}...")

    # 设置搜索参数
    if search_param is not None:
        if hasattr(index, "nprobe"):
            index.nprobe = search_param
        elif hasattr(index, "hnsw") and hasattr(index.hnsw, "efSearch"):
            index.hnsw.efSearch = search_param

    # 计算内存占用
    memory_usage = calculate_memory_usage(index, data, index_name)

    # QPS测试
    start_time = time.time()
    # 为保证公平性，所有索引都使用单个查询向量的方式
    for i in range(len(queries)):
        if index_name == "IVFPOUQ" or index_name == "IvfSQ8Index":
            index.search(queries[i : i + 1].astype("float32"), k, search_param)
        else:
            index.search(queries[i : i + 1].astype("float32"), k)
    end_time = time.time()

    # 为保证公平性，所有索引都使用单个查询向量的方式
    all_results = []
    all_distances = []
    for i in range(len(queries)):
        if index_name == "IVFPOUQ" or index_name == "IvfSQ8Index":
            assert search_param is not None
            ret = index.search(queries[i : i + 1].astype("float32"), k, search_param)
            result = []
            distances = []
            for r, d in ret:
                result.append(r)
                distances.append(d)
            all_results.append(result)
            all_distances.append(distances)
        else:
            distances, result_indices = index.search(
                queries[i : i + 1].astype("float32"), k
            )
            all_results.append(result_indices[0])  # 取出第一个结果
            all_distances.append(distances[0])  # 取出第一个距离结果

    search_results = np.array(
        [r[:k] if len(r) >= k else r + [0] * (k - len(r)) for r in all_results]
    )

    search_distances = np.array(
        [
            d[:k] if len(d) >= k else np.concatenate([d, [float("inf")] * (k - len(d))])
            for d in all_distances
        ]
    )

    # 计算QPS
    total_time = end_time - start_time
    qps = len(queries) / total_time

    # 在 benchmark_index 函数中
    # 计算recall
    recalls = []
    precision_scores = []  # 改名
    for i in range(len(queries)):
        recall = calculate_recall(gt_indices[i], search_results[i], k)
        recalls.append(recall)

        # 计算精度分数
        precision_score = calculate_distance_ratio(
            gt_distances[i], search_distances[i], k
        )
        precision_scores.append(precision_score)

    avg_recall = np.mean(recalls)
    avg_precision_score = np.mean(precision_scores)  # 改名

    print(f"{index_name}{param_str} Results:")
    print(f"  QPS: {qps:.2f}")
    print(f"  Recall@{k}: {avg_recall:.4f}")
    print(f"  Precision Score: {avg_precision_score:.4f}")
    print(f"  Memory Usage: {memory_usage:.2f} MB")
    print(f"  Total time: {total_time:.4f}s")

    return qps, avg_recall, avg_precision_score, memory_usage, total_time


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
    # data = data[:100000]
    # query_data = query_data[:100]

    print(f"Data shape: {data.shape}")
    print(f"Query shape: {query_data.shape}")

    k = 10  # 搜索top-k

    # 获取ground truth（包括距离）
    gt_indices, gt_distances = get_ground_truth(data, query_data, k)

    # 准备结果存储
    results = []

    # 1. 测试POUQ IVF索引 - 8个不同nprobe参数
    print("\n=== Testing IvfSQ8Index with different nprobe values ===")
    try:
        nlist = 1024  # 修改为1024
        pouq_index = pouq.IvfSQ8Index(nlist=nlist, dim=data.shape[1])
        pouq_index.train(data.astype("float32"))

        # 测试8个不同的nprobe值（对应POUQ的搜索参数）
        nprobe_values = [5, 10, 20, 40, 80, 160, 320]
        for nprobe in nprobe_values:
            qps, recall, distance_ratio, memory_usage, total_time = benchmark_index(
                pouq_index,
                query_data,
                data,
                gt_indices,
                gt_distances,
                k,
                "IvfSQ8Index",
                nprobe,
            )
            results.append(
                [
                    f"IvfSQ8Index",
                    qps,
                    recall,
                    distance_ratio,
                    memory_usage,
                    total_time,
                    nprobe,
                ]
            )
    except Exception as e:
        print(f"IvfSQ8Index test failed: {e}")

    # 1. 测试POUQ IVF索引 - 8个不同nprobe参数
    print("\n=== Testing POUQ IVF Index with different nprobe values ===")
    try:
        nlist = 1024  # 修改为1024
        pouq_index = pouq.IvfIndex(nlist=nlist, dim=data.shape[1])
        pouq_index.train(data.astype("float32"))

        # 测试8个不同的nprobe值（对应POUQ的搜索参数）
        nprobe_values = [5, 10, 20, 40, 80, 160, 320]
        for nprobe in nprobe_values:
            qps, recall, distance_ratio, memory_usage, total_time = benchmark_index(
                pouq_index,
                query_data,
                data,
                gt_indices,
                gt_distances,
                k,
                "IVFPOUQ",
                nprobe,
            )
            results.append(
                [
                    f"IVFPOUQ",
                    qps,
                    recall,
                    distance_ratio,
                    memory_usage,
                    total_time,
                    nprobe,
                ]
            )
    except Exception as e:
        print(f"POUQ IVF test failed: {e}")

    # 3. 测试Faiss IndexHNSW - 8个不同ef_search参数
    print("\n=== Testing Faiss IndexHNSW with different ef_search values ===")
    try:
        M = 16  # HNSW参数
        index_hnsw = faiss.IndexHNSWFlat(data.shape[1], M)
        index_hnsw.add(data.astype("float32"))

        # 测试8个不同的ef_search值
        ef_search_values = [16, 32, 64, 128, 256, 512, 1024, 2048]
        for ef_search in ef_search_values:
            qps, recall, distance_ratio, memory_usage, total_time = benchmark_index(
                index_hnsw,
                query_data,
                data,
                gt_indices,
                gt_distances,
                k,
                "HNSW",
                ef_search,
            )
            results.append(
                [
                    f"HNSW",
                    qps,
                    recall,
                    distance_ratio,
                    memory_usage,
                    total_time,
                    ef_search,
                ]
            )
    except Exception as e:
        print(f"Faiss HNSW test failed: {e}")

    # 4. 测试Faiss IndexIVFPQR - 8个不同nprobe参数 (带重排序的IVFPQ)
    print("\n=== Testing Faiss IndexIVFPQR with different nprobe values ===")
    try:
        nlist = 1024  # 修改为1024
        m = 4  # 子量化器数量
        nbits = 8  # 每个子量化器的位数
        m_refine = 4  # 重排序的子量化器数量
        nbits_refine = 8  # 重排序每个子量化器的位数

        quantizer = faiss.IndexFlatL2(data.shape[1])
        index_ivfpqr = faiss.IndexIVFPQR(
            quantizer, data.shape[1], nlist, m, nbits, m_refine, nbits_refine
        )
        index_ivfpqr.train(data.astype("float32"))
        index_ivfpqr.add(data.astype("float32"))

        # 测试8个不同的nprobe值
        nprobe_values = [5, 10, 20, 40, 80, 160, 320]
        for nprobe in nprobe_values:
            qps, recall, distance_ratio, memory_usage, total_time = benchmark_index(
                index_ivfpqr,
                query_data,
                data,
                gt_indices,
                gt_distances,
                k,
                "IVFPQR",
                nprobe,
            )
            results.append(
                [
                    f"IVFPQR",
                    qps,
                    recall,
                    distance_ratio,
                    memory_usage,
                    total_time,
                    nprobe,
                ]
            )
    except Exception as e:
        print(f"Faiss IVFPQR test failed: {e}")

    # 5. 测试Faiss IndexIVFRaBitQ - 8个不同nprobe参数 (作为基准)
    print("\n=== Testing Faiss IndexIVFRaBitQ with different nprobe values ===")
    try:
        nlist = 1024  # 修改为1024
        quantizer = faiss.IndexFlatL2(data.shape[1])
        index_ivf = faiss.IndexIVFRaBitQ(quantizer, data.shape[1], nlist)
        index_ivf.train(data.astype("float32"))
        index_ivf.add(data.astype("float32"))

        # 测试8个不同的nprobe值
        nprobe_values = [5, 10, 20, 40, 80, 160, 320]
        for nprobe in nprobe_values:
            qps, recall, distance_ratio, memory_usage, total_time = benchmark_index(
                index_ivf,
                query_data,
                data,
                gt_indices,
                gt_distances,
                k,
                "IndexIVFRaBitQ",
                nprobe,
            )
            results.append(
                [
                    f"IndexIVFRaBitQ",
                    qps,
                    recall,
                    distance_ratio,
                    memory_usage,
                    total_time,
                    nprobe,
                ]
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
            [
                "Index",
                "QPS",
                f"Recall@{k}",
                "Distance_Ratio",
                "Memory_Usage(MB)",
                "Total_Time(s)",
                "Search_Param",
            ]
        )
        writer.writerows(results)

    print(f"\n=== Summary ===")
    print(f"Results saved to: {output_file}")
    print("\nPerformance Comparison:")
    print(
        f"{'Index':<25} {'QPS':<10} {'Recall@'+str(k):<12} {'Dist_Ratio':<12} {'Memory(MB)':<12} {'Time(s)':<10} {'Param':<8}"
    )
    print("-" * 95)
    for result in results:
        print(
            f"{result[0]:<25} {result[1]:<10.2f} {result[2]:<12.4f} {result[3]:<12.4f} {result[4]:<12.2f} {result[5]:<10.4f} {result[6]:<8}"
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
        print(
            f"{'Param':<8} {'QPS':<10} {'Recall':<10} {'Dist_Ratio':<12} {'Memory(MB)':<12} {'QPS/Recall':<12}"
        )
        print("-" * 75)
        for result in sorted(type_results, key=lambda x: x[6]):
            trade_off = result[1] / result[2] if result[2] > 0 else 0
            print(
                f"{result[6]:<8} {result[1]:<10.2f} {result[2]:<10.4f} {result[3]:<12.4f} {result[4]:<12.2f} {trade_off:<12.2f}"
            )
