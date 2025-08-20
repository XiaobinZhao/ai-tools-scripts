import json
import os
import subprocess
from argparse import ArgumentParser
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from tqdm import tqdm

try:
    import openpyxl
except ImportError:
    raise ImportError("Please install openpyxl first. run `pip install openpyxl -i https://mirrors.aliyun.com/pypi/simple/  --trusted-host mirrors.aliyun.com`")

in_params = {
    # 完整random测试参数设置
    "default": {
        "input_output_pairs": [(1024, 256), (1024, 1024), (2048, 2048)],
        "concurrency_values": [1, 16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1024],
        "num_prompts": [50, 50, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 1024, 2048],
        "random_range_ratio": 0.1
    },
    # 专为腾讯测试使用
    "tx": {
        "input_output_pairs": [(6000, 1000)],
        "concurrency_values": [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200],
        "num_prompts": [50, 50, 50, 60, 60, 80, 100,  120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400],
        "random_range_ratio": 1
    },
    # 专为本脚本测试使用
    "test": {
        "input_output_pairs": [(10, 10), (20, 20)],
        "concurrency_values": [1, 2],
        "num_prompts": [2, 2],
        "random_range_ratio": 1
    },
    # 专为火山测试使用
    # 火山测试的文档 https://www.volcengine.com/docs/6459/1462763#sglang%E7%9A%84%E6%B5%8B%E8%AF%95%E6%96%B9%E6%B3%95
    # declare -a request_rates=(1 4 8 16 32 48 64 80 128 160)
    # declare -a max_concurrency=(1 4 8 16 32 48 64 80 128 160)
    # declare -a num_prompts=(4 16 32 64 128 192 256 320 512 640)
    # "fire": {
    #     "input_output_pairs": [(6000, 1000)],
    #     "request_rates": [1, 4, 8, 16, 30, 30],
    #     "concurrency_values": [1, 4, 8, 16, 32, 40],
    #     "num_prompts": [4, 16, 32, 64, 128, 160],
    #     "random_range_ratio": 1
    # }
    "fire": {
        "input_output_pairs": [(6000, 1000)],
        "request_rates": [1, 4, 8, 16, 32, 48, 64, 80, 128, 160],
        "concurrency_values": [1, 4, 8, 16, 32, 48, 64, 80, 128, 160],
        "num_prompts": [4, 16, 32, 64, 128, 192, 256, 320, 512, 640],
        "random_range_ratio": 1
    },
    "fire3.5": {
        "input_output_pairs": [(3500, 1000)],
        "request_rates": [1, 4, 8, 16, 32, 48, 64, 80, 128, 160],
        "concurrency_values": [1, 4, 8, 16, 32, 48, 64, 80, 128, 160],
        "num_prompts": [4, 16, 32, 64, 128, 192, 256, 320, 512, 640],
        "random_range_ratio": 1
    }
}

# 开始时间
start_time = datetime.now()

parser = ArgumentParser(description="Benchmark the online serving throughput.")
parser.add_argument(
    "--mode",
    type=str,
    required=True,
    choices=["test", "default", "tx", "fire", "fire3.5"],  # 只允许这两个值
    help="default is 1024-in/256-out to 2048-in/2048-out and concurrency from 16 to 1024, tx is for tencent test.fire is for huoshan test",
)
parser.add_argument(
    "--engine",
    type=str,
    default="SGLang",
    choices=["SGLang", "TensorRT", "xLLM", "vLLM", "KsanaLLM"],
    help='one of "SGLang", "TensorRT", "xLLM", "vLLM", "KsanaLLM", default is SGLang',
)
parser.add_argument(
    "--benchmark",
    type=str,
    default="SGLang",
    choices=["SGLang", "vLLM", "KsanaLLM"],
    help='the benchmark script of "SGLang", "vLLM", "KsanaLLM", default is SGLang',
)
parser.add_argument(
    "--gpu",
    type=str,
    default="H200",
    choices=["H20-96G", "H20-141G", "H200", "B200"],
    help='one of "H20-96G", "H20-141G", "H200", "B200", default is H200',
)
parser.add_argument(
    "--cluster",
    type=str,
    default="1Node",
    help='one of "1Node", "xNodesLB", "xPxD", default is 1Node',
)
parser.add_argument(
    "--model",
    type=str,
    default="/root/.cache/huggingface/DeepSeek-R1",
    help="Name or path of the model. If not set, the default model will request /v1/models for conf. Default is /root/.cache/huggingface/DeepSeek-R1",
)
parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Default host is 0.0.0.0.")
parser.add_argument(
    "--port",
    type=str,
    default="9000",
    help="If not set, the default port is configured according to its default value for different LLM Inference Engines. Default is 9000",
)
parser.add_argument(
    "--backend",
    type=str,
    default="sglang-oai-chat",
    help="sglang-oai, sglang-oai-chat, vllm, sglang, ksana, default is sglang-oai-chat",
)
parser.add_argument(
    "--tokenizer",
    type=str,
    default="/root/.cache/huggingface/DeepSeek-R1",
    help="Name or path of the tokenizer. If not set, the default tokenizer will use model value,"
         "if model start with a pathlike str. Default is /root/.cache/huggingface/DeepSeek-R1",
)
parser.add_argument(
    "--dataset-path",
    type=str,
    default="/root/.cache/huggingface/ShareGPT_V3_unfiltered_cleaned_split.json",
    help="Path to the dataset. Default is /root/.cache/huggingface/ShareGPT_V3_unfiltered_cleaned_split.json"
)
parser.add_argument(
    "--dataset-name", type=str, default="random", help="longbenchV2noCtx,longbenchV2withCtx,random,tx; Default is random"
)
in_args = parser.parse_args()

_tokenizer = in_args.tokenizer or (in_args.model if in_args.model and in_args.model.startswith("/") else "/root/.cache/huggingface/DeepSeek-R1")
_model_name = in_args.model.split("/")[-1]
current_time = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S")
today = datetime.today().date()
output_file = f"{in_args.engine}_{in_args.gpu}_{in_args.cluster}_{in_args.benchmark}_{_model_name}_{in_args.mode}_{in_args.dataset_name}_{in_args.backend}_benchmark_results_{current_time}.xlsx"
_result_dir = f"./logs/{in_args.benchmark}/{in_args.backend}/{today}"

print(f"=== result will save to: ./{output_file} ===")

# 创建 logs 目录（如果不存在）
os.makedirs(f"{_result_dir}", exist_ok=True)

input_output_pairs = in_params[in_args.mode]["input_output_pairs"]
concurrency_values = in_params[in_args.mode]["concurrency_values"]
request_rates_values = in_params[in_args.mode].get("request_rates", [])
num_prompts = in_params[in_args.mode]["num_prompts"]
random_range_ratio = in_params[in_args.mode]["random_range_ratio"]

seed = 0
# 运行 benchmark 测试
for input_len, output_len in tqdm(input_output_pairs, desc="处理输入输出组合"):
    for index, concurrency in enumerate(tqdm(concurrency_values, desc=f"处理并发组合In-{input_len}:Out-{output_len}")):
        seed += 1
        prompts = num_prompts[index]
        request_rate = request_rates_values[index] if request_rates_values else float("inf")  # 默认为正无穷大

        print(f"\n\n --- start benchmark for input_len={input_len}, output_len={output_len}, concurrency={concurrency}, num_prompts={prompts} ---")

        result_file_name_prefix = f"{concurrency}_{_model_name}_{in_args.dataset_name}_{input_len}_{output_len}_{current_time}"

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
            "--concurrency", str(concurrency),
            "--request_rate", str(request_rate),
            "--prompt_num", str(prompts),
            "--seed", str(seed),
            "--max_new_tokens", str(output_len)
        ]

        sglang_cmd = [
            "python3", "-m", "sglang.bench_serving",
            "--backend", in_args.backend,
            "--dataset-name", in_args.dataset_name,
            "--dataset-path", in_args.dataset_path,
            "--random-input", str(input_len),
            "--random-output", str(output_len),
            "--random-range-ratio", str(random_range_ratio),
            "--output-details",
            "--request-rate", str(request_rate),
            "--host", in_args.host,
            "--port", in_args.port,
            "--model", in_args.model,
            "--tokenizer", _tokenizer,
            "--max-concurrency", str(concurrency),
            "--num-prompts", str(prompts),
            "--output-file", f"{_result_dir}/{result_file_name_prefix}.jsonl",
        ]

        _run_benchmark_failed = False

        if in_args.benchmark == "SGLang":
            with open(f"{_result_dir}/{result_file_name_prefix}.txt", "w") as f:
                process = subprocess.Popen(sglang_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in process.stdout:
                    print(line.strip())  # 打印到终端
                    if "bench_serving.py: error" in line:
                        _run_benchmark_failed = line
                    f.write(line)  # 写入文件
                    f.flush()  # 确保实时写入文件
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
            if _run_benchmark_failed:
                raise ValueError(f"{_run_benchmark_failed}\n command: {sglang_cmd}")
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
                "seed": seed,
                "Input len": input_len,
                "Output len": output_len,
                "batch size": concurrency,
                "request rate": str(_result["request_rate"]),
                "requests": prompts,
                "failed requests": prompts - _result["completed"],
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
                "seed": seed,
                "Input len": "not set",
                "Output len": output_len,
                "batch size": concurrency,
                "request rate": request_rate,
                "requests": prompts,
                "failed requests": prompts - round(_request_throughput * _total_latency),
                "TTFT(ms)": ttft*1000,
                "TPOT(ms)": tpot*1000,
                "Throughput(tokens/s)": throughput
            }
        else:
            print(f"参数 --benchmark:{in_args.benchmark} 输入有误，目前只支持SGLang和KsanaLLM两种参数")
        # 将结果保存到Excel文件
        df = pd.DataFrame([entry])
        if not os.path.exists(output_file):
            df.to_excel(output_file, index=False, header=True)
        else:
            with pd.ExcelWriter(output_file, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
        print(f"+++ finish benchmark for output_len={output_len}, concurrency={concurrency}, num_prompts={prompts}, Total latency={_total_latency}s; save to execl. +++ \n\n")

# 结束时间
print(f"=== result save to: ./{output_file} ===")
end_time = datetime.now()
print(f"=== Total time: {end_time - start_time} ===")
