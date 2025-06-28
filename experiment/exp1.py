import csv
import multiprocessing
import os
import signal
import sys
import time
from multiprocessing import Process, Queue

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

# 存储结果的列表
results = []


# 子进程训练函数
def train_worker(name, bitwidth, quantizer, data_f32, result_queue):
    """在子进程中执行训练"""
    try:
        start_time = time.time()

        # 执行训练
        quantizer.train(data_f32)
        codes = quantizer.compute_codes(data_f32)

        train_time = time.time() - start_time

        # 计算MSE
        reconstructed = quantizer.decode(codes)
        if isinstance(reconstructed, list):
            mse = np.mean((data_f32.reshape(-1) - reconstructed) ** 2)
        else:
            mse = np.mean((data_f32 - reconstructed) ** 2)

        # 将结果放入队列
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
        # 如果出现异常，也放入队列
        result_queue.put(
            {"success": False, "method": name, "bitwidth": bitwidth, "error": str(e)}
        )


def run(name, bitwidth, quantizer, data_f32):
    """主进程中的run函数，管理子进程训练"""
    # 12小时的时间限制（43200秒）
    max_train_time = 12 * 60 * 60  # 12小时 = 43200秒

    # 创建进程间通信队列
    result_queue = Queue()

    # 创建子进程
    process = Process(
        target=train_worker, args=(name, bitwidth, quantizer, data_f32, result_queue)
    )

    # print(f"开始训练: {name} (bitwidth={bitwidth})")
    start_time = time.time()

    # 启动子进程
    process.start()

    # 等待子进程完成或超时
    process.join(timeout=max_train_time)

    if process.is_alive():
        # 如果子进程仍在运行，说明超时了
        print(f"警告: {name} (bitwidth={bitwidth}) 训练时间超过12小时，强制终止训练")

        # 强制终止子进程
        process.terminate()
        process.join(timeout=5)  # 等待5秒让进程优雅退出

        if process.is_alive():
            # 如果还是没有退出，强制杀死
            process.kill()
            process.join()

        print(f"{name} (bitwidth={bitwidth}) 训练已被终止")
        return

    # 检查子进程是否正常完成
    if process.exitcode == 0:
        # 获取训练结果
        try:
            result = result_queue.get_nowait()
            if result["success"]:
                train_time = result["train_time"]
                mse = result["mse"]

                print(f"| {name} | {bitwidth} | {train_time:.4f} | {mse:.6e} |")

                # 将结果添加到列表中
                results.append(
                    {
                        "method": name,
                        "bitwidth": bitwidth,
                        "train_time": train_time,
                        "mse": mse,
                    }
                )
            else:
                print(f"错误: {name} (bitwidth={bitwidth}) 训练失败: {result['error']}")
        except:
            print(f"错误: {name} (bitwidth={bitwidth}) 无法获取训练结果")
    else:
        print(
            f"错误: {name} (bitwidth={bitwidth}) 子进程异常退出 (退出码: {process.exitcode})"
        )


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    data = fvecs_read(f"../data/{dataset_name}/{dataset_name}_base.fvecs")
    data = data[:1000]

    n, d = data.shape
    data_f32 = data.astype(np.float32)

    print(f"Dataset: {dataset_name}, Shape: {data.shape}")
    print("=" * 50)

    for i in range(4, 9):
        q = posq.SQQuantizer(c_bit=0, q_bit=i, sub=d)
        run("SQ", i, q, data_f32)

    for i in range(4, 9):
        q = posq.OSQQuantizer(c_bit=0, q_bit=i, sub=d)
        run("OSQ-Baseline", i, q, data_f32)

    for i in range(4, 9):
        q = posq.OSQ2Quantizer(c_bit=0, q_bit=i, sub=d)
        run("OSQ-PSO", i, q, data_f32)

    for i in range(2, 5):
        q = posq.POSQKMeansQuantizer(c_bit=i, q_bit=4, sub=d)
        run("POSQ-KMeans-MinMax", i + 4, q, data_f32)

    for i in range(2, 5):
        q = posq.POSQKRangeQuantizer(c_bit=i, q_bit=4, sub=d)
        run("POSQ-KRange-MinMax", i + 4, q, data_f32)

    for i in range(4, 9):
        q = posq.POSQQuantizer(c_bit=i // 2, q_bit=i - (i // 2), sub=d)
        run("POSQ", i, q, data_f32)

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

    # 创建结果目录（如果不存在）
    result_dir = "../result"
    os.makedirs(result_dir, exist_ok=True)

    # 保存结果到CSV文件
    csv_file = f"../result/exp1_{dataset_name}.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["method", "bitwidth", "train_time", "mse"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\n结果已保存到: {csv_file}")
