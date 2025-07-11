import json
import os
import subprocess
from argparse import ArgumentParser
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

# 定义 input/output 长度
INPUT_LEN = 6000
OUTPUT_LEN = 1000

max_concurrency = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
num_prompts = [50, 50, 50, 60, 60, 80, 100, 120, 140, 160, 180, 200]

# 开始时间
start_time = datetime.now()

parser = ArgumentParser(description="Benchmark the online serving throughput.")
parser.add_argument(
    "--engine",
    type=str,
    default="SGLang",
    choices=["SGLang", "TensorRT", "xLLM", "vLLM", "KsanaLLM"],
    help='one of "SGLang", "TensorRT", "xLLM", "vLLM", "KsanaLLM"',
)
parser.add_argument(
    "--benchmark",
    type=str,
    default="SGLang",
    choices=["SGLang", "vLLM", "KsanaLLM"],
    help='the benchmark script of "SGLang", "vLLM", "KsanaLLM"',
)
parser.add_argument(
    "--gpu",
    type=str,
    default="H200",
    choices=["H20-96G", "H20-141G", "H200", "B200"],
    help='one of "H20-96G", "H20-141G", "H200", "B200"',
)
parser.add_argument(
    "--cluster",
    type=str,
    default="1Node",
    help='one of "1Node", "xNodesLB", "xPxD"',
)
parser.add_argument(
    "--model",
    type=str,
    default="/root/.cache/huggingface/DeepSeek-R1",
    help="Name or path of the model. If not set, the default model will request /v1/models for conf.",
)
parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Default host is 0.0.0.0.")
parser.add_argument(
    "--port",
    type=str,
    default="8000",
    help="If not set, the default port is configured according to its default value for different LLM Inference Engines.",
)
parser.add_argument(
    "--backend",
    type=str,
    default="sglang-oai",
    help="sglang-oai, vllm, sglang, ksana, chat",
)
parser.add_argument(
    "--tokenizer",
    type=str,
    default="/root/.cache/huggingface/DeepSeek-R1",
    help="Name or path of the tokenizer. If not set, the default tokenizer will use model value,"
         "if model start with a pathlike str.",
)
parser.add_argument(
    "--dataset-path",
    type=str,
    default="/root/.cache/huggingface/ShareGPT_V3_unfiltered_cleaned_split.json",
    help="Path to the dataset."
)
parser.add_argument(
    "--dataset-name", type=str, default="random", help="longbenchV2noCtx or longbenchV2withCtx or random"
)
in_args = parser.parse_args()

_tokenizer = in_args.tokenizer or (in_args.model if in_args.model and in_args.model.startswith("/") else "/root/.cache/huggingface/DeepSeek-R1")
_model_name = in_args.model.split("/")[-1]
current_time = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S")
output_file = f"{in_args.engine}_{in_args.gpu}_{in_args.cluster}_{in_args.benchmark}_{_model_name}_{in_args.dataset_name}_{in_args.backend}_benchmark_results_{current_time}.xlsx"

print(f"=== result will save to: {output_file} ===")

_result_dir = f"logs/{in_args.benchmark}/{in_args.backend}"
# 创建 logs 目录（如果不存在）
os.makedirs(f"{_result_dir}", exist_ok=True)

# 运行 benchmark 测试
for i in range(len(max_concurrency)):
    CONCURRENCY = max_concurrency[i]
    PROMPTS = num_prompts[i]
    SEED = i + 1

    print(f"\n\n --- start benchmark for input_len={INPUT_LEN}, output_len={OUTPUT_LEN}, concurrency={CONCURRENCY}, num_prompts={PROMPTS} --- \n")

    result_file_name_prefix = f"{CONCURRENCY}_{_model_name}_{in_args.dataset_name}_{OUTPUT_LEN}_{current_time}"

    ksana_cmd = [
        "python", "/workspace/KsanaLLM/benchmarks/benchmark_throughput.py",
        "--dataset_name", in_args.dataset_name,
        "--dataset_path", in_args.dataset_path,
        "--backend", in_args.backend,
        "--model_type", "deepseek_r1",
        "--mode", "async",
        "--output_csv", f"{_result_dir}/{result_file_name_prefix}_output_res.csv",
        "--perf_csv", f"{_result_dir}/{result_file_name_prefix}_perf_res.csv",
        "--tokenizer_path", _tokenizer,
        "--shuffle",
        "--stream",
        "--host", in_args.host,
        "--port", in_args.port,
        "--concurrency", str(CONCURRENCY),
        "--prompt_num", str(PROMPTS),
        "--seed", str(SEED),
        "--max_new_tokens", str(OUTPUT_LEN)
    ]

    sglang_cmd = [
        "python3", "-m", "sglang.bench_serving",
        "--backend", in_args.backend,
        "--dataset-name", in_args.dataset_name,
        "--dataset-path", in_args.dataset_path,
        "--random-input", str(INPUT_LEN),
        "--random-output", str(OUTPUT_LEN),
        "--random-range-ratio", "1",
        "--host", in_args.host,
        "--port", in_args.port,
        "--model", in_args.model,
        "--tokenizer", _tokenizer,
        "--max-concurrency", str(CONCURRENCY),
        "--num-prompts", str(PROMPTS),
        "--output-file", f"{_result_dir}/{result_file_name_prefix}.jsonl",
    ]

    if in_args.benchmark == "SGLang":
        with open(f"{_result_dir}/{result_file_name_prefix}.txt", "w") as f:
            process = subprocess.Popen(sglang_cmd, stdout=f, stderr=subprocess.STDOUT)
            process.wait()
        # 读取jsonl文件，获取throughput数据;demo 如下
        # {
        #     "request_rate": Infinity,
        #     "random_range_ratio": 1.0,
        #     "duration": 3103.9254839839414,
        #     "completed": 12000,
        #     "request_throughput": 3.8660721921061696,
        #     "input_throughput": 23196.43315263702,
        #     "output_throughput": 3866.0721921061695,
        #     "mean_ttft_ms": 850.8626157734543,
        #     "mean_tpot_ms": 50.85409914053094,
        #     "concurrency": 199.6985092214196,
        #     ...
        # }

        file_path = f"{_result_dir}/{result_file_name_prefix}.jsonl"
        last_line = None

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last_line = line  # 只读取最后一行；因为输出结果是jsonl格式，每行是一个结果对象，最后一行是最新数据

        if not last_line:
            raise ValueError(f"Invalid or empty JSONL file: {file_path}")

        _result = json.loads(last_line)
        _total_latency = _result["duration"]
        entry = {
            "seed": SEED,
            "Input len": INPUT_LEN,
            "Output len": OUTPUT_LEN,
            "batch size": CONCURRENCY,
            "request rate": str(_result["request_rate"]),
            "requests": PROMPTS,
            "failed requests": PROMPTS - _result["completed"],
            "TTFT(ms)": _result["mean_ttft_ms"],
            "TPOT(ms)": _result["mean_tpot_ms"],
            "Throughput(tokens/s)": _result["input_throughput"] + _result["output_throughput"]
        }

    elif in_args.benchmark == "KsanaLLM":
        with open(f"{_result_dir}/{result_file_name_prefix}.txt", "w") as f:
            process = subprocess.Popen(ksana_cmd, stdout=f, stderr=subprocess.STDOUT)
            process.wait()
        # 读取perf_res.csv，获取TTFT/TPOT/throughput/success requests数据;demo 如下
        # Request rate,Request throughput,Total latency,Token throughput,Avg TTFT,Avg TPOT,...
        # inf,5.84291,34.2295,4835.68269,0.89407,0.03252
        perf_res = pd.read_csv(f"{_result_dir}/{result_file_name_prefix}_perf_res.csv")
        throughput = perf_res["Token throughput"].values[0]
        ttft = perf_res["Avg TTFT"].values[0]
        tpot = perf_res["Avg TPOT"].values[0]
        request_rate = perf_res["Request rate"].values[0]
        _request_throughput = perf_res["Request throughput"].values[0]
        _total_latency = perf_res["Total latency"].values[0]

        entry = {
            "seed": SEED,
            "Input len": "not set",
            "Output len": OUTPUT_LEN,
            "batch size": CONCURRENCY,
            "request rate": request_rate,
            "requests": PROMPTS,
            "failed requests": PROMPTS - round(_request_throughput * _total_latency),
            "TTFT(ms)": ttft*1000,
            "TPOT(ms)": tpot*1000,
            "Throughput(tokens/s)": throughput
        }

    # 将结果保存到Excel文件
    df = pd.DataFrame([entry])
    if not os.path.exists(output_file):
        df.to_excel(output_file, index=False, header=True)
    else:
        with pd.ExcelWriter(output_file, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
    print(f"+++ finish benchmark for output_len={OUTPUT_LEN}, concurrency={CONCURRENCY}, num_prompts={PROMPTS}, Total latency={_total_latency}s; save to execl. +++ \n")

# 结束时间
end_time = datetime.now()
print(f"Total time: {end_time - start_time}")
