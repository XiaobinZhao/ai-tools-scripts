"""
批量使用sglang.bench_serving random dataset 压测推理后端，收集数据到excel
pip install sglang
"""

import argparse
import os

from argparse import ArgumentParser
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
# 需要预先安装sglang
from sglang.bench_serving import run_benchmark, set_global_args

# 定义参数
params = {
    # 完整random测试参数设置
    "full": {
        "input_output_pairs": [(1024, 256), (1024, 1024), (2048, 2048)],
        # 并发值设置,有一个1025的目的是为了实现并发和请求数都是1024的case
        "concurrency_values": [1, 16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1024],
        "num_prompts": [50, 50, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 1024, 2048],
        "random_range_ratio": 0.1
    },
    # 转为腾讯测试使用
    "tx": {
        "input_output_pairs": [(6000, 1000)],
        "concurrency_values": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200],
        "num_prompts": [50, 60, 60, 80, 100,  120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400],
        "random_range_ratio": 1
    }
}

results = []

parser = ArgumentParser(description="Benchmark the online serving throughput.")
parser.add_argument(
    "--mode",
    type=str,
    required=True,
    choices=["full", "tx"],  # 只允许这两个值
    help="full is 1024-in/256-out to 2048-in/2048-out and concurrency from 16 to 1024, tx is for tencent test.",
)
parser.add_argument(
    "--model",
    type=str,
    help="Name or path of the model. If not set, the default model will request /v1/models for conf.",
)
parser.add_argument(
    "--port",
    type=str,
    help="If not set, the default port is configured according to its default value for different LLM Inference Engines.",
)
parser.add_argument(
    "--backend",
    type=str,
    default="vllm",
    help="Must specify a backend, depending on the LLM Inference Engine.",
)
parser.add_argument(
    "--tokenizer",
    type=str,
    help="Name or path of the tokenizer. If not set, the default tokenizer will use model value,"
         "if model start with a pathlike str.",
)

parser.add_argument(
    "--dataset-path", type=str, default="", help="Path to the dataset."
)
in_args = parser.parse_args()

# 设置全局参数
args = argparse.Namespace(
    backend=in_args.backend,
    dataset_name="random",
    dataset_path=in_args.dataset_path or "/root/.cache/huggingface/ShareGPT_V3_unfiltered_cleaned_split.json",
    host="0.0.0.0",
    port=in_args.port or 8000,
    base_url=None,
    tokenizer=in_args.tokenizer or in_args.model if in_args.model and in_args.model.startswith("/") else "/root/.cache/huggingface/DeepSeek-R1",   # 分词器用来做token长度计算
    model=in_args.model or "/root/.cache/huggingface/DeepSeek-R1",
    random_range_ratio=params[in_args.mode]["random_range_ratio"],
    request_rate=float("inf"),
    seed=1,
    profile=False,
    extra_request_body=None,
    disable_tqdm=False,
    disable_stream=False,
    disable_ignore_eos=False,
    apply_chat_template=False,
    lora_name=None,
    pd_separated=False,
    flush_cache=False,
    prompt_suffix="",
    warmup_requests=1,
    random_input_len=1024,
    random_output_len=1024,
    max_concurrency=1024,
    output_file=None,
    num_prompts=1024,
    sharegpt_output_len=256,
    gsp_num_groups=64,
    gsp_prompts_per_group=16,
    gsp_system_prompt_len=2048,
    gsp_question_len=128,
    gsp_output_len=256,
    output_details=True,
)

# 设置全局args
set_global_args(args)

# 使用ExcelWriter来逐次写入结果
current_time = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S")
output_file = f"batch_{args.dataset_name}_{args.backend}_{in_args.mode}_benchmark_results_{current_time}.xlsx"
i = 1

input_output_pairs = params[in_args.mode]["input_output_pairs"]
concurrency_values = params[in_args.mode]["concurrency_values"]
num_prompts = params[in_args.mode]["num_prompts"]

for input_len, output_len in input_output_pairs:
    for index, concurrency in enumerate(concurrency_values):
        args.seed = i
        i += 1
        args.random_input_len = input_len
        args.random_output_len = output_len
        args.max_concurrency = concurrency
        args.num_prompts = num_prompts[index]

        print(
            f"\n\n === start benchmark for input_len={input_len}, output_len={output_len}, concurrency={concurrency}, num_prompts={args.num_prompts} === \n\n")

        result = run_benchmark(args)  # 调用sglang的run_benchmark函数

        # 提取需要的字段
        total_throughput = (result["total_input_tokens"] + sum(result["output_lens"])) / result["duration"]
        entry = {
            "seed": args.seed,
            "Input len": input_len,
            "Output len": output_len,
            "batch size": args.max_concurrency,
            "requests": args.num_prompts,
            "TTFT(ms)": result["mean_ttft_ms"],
            "TPOT(ms)": result["mean_itl_ms"],
            "Throughput(tokens/s)": total_throughput
        }
        results.append(entry)

        # 将结果保存到Excel文件
        df = pd.DataFrame([entry])
        if not os.path.exists(output_file):
            df.to_excel(output_file, index=False, header=True)
        else:
            with pd.ExcelWriter(output_file, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
                print(f"save {input_len}_{output_len}_{concurrency}_{args.num_prompts} result to {output_file}")
