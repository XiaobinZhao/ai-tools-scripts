"""
批量使用sglang.bench_serving random dataset 压测推理后端，收集数据到excel
pip install sglang
"""

import argparse
import os

from argparse import ArgumentParser
import pandas as pd
# 需要预先安装sglang
from sglang.bench_serving import run_benchmark, set_global_args

# 定义参数
input_output_pairs = [(1024, 256), (1024, 1024), (2048, 2048)]  # 输入/输出长度设置
# 并发值设置,有一个1025的目的是为了实现并发和请求数都是1024的case
concurrency_values = [1, 16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1025,
                      1024]
results = []

parser = ArgumentParser(description="Benchmark the online serving throughput.")
parser.add_argument(
    "--model",
    type=str,
    help="Name or path of the model. If not set, the default model will request /v1/models for conf.",
)

parser.add_argument(
    "--dataset-path", type=str, default="", help="Path to the dataset."
)
in_args = parser.parse_args()

# 设置全局参数
args = argparse.Namespace(
    backend="vllm",
    dataset_name="random",
    dataset_path=in_args.dataset_path or "/root/.cache/huggingface/ShareGPT_V3_unfiltered_cleaned_split.json",
    host="0.0.0.0",
    port=8000,
    base_url=None,
    tokenizer=None,
    model=in_args.model or "/root/.cache/huggingface/DeepSeek-R1",
    random_range_ratio=0.1,
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
)

# 设置全局args
set_global_args(args)

# 使用ExcelWriter来逐次写入结果
output_file = f"batch_{args.dataset_name}_{args.backend}_benchmark_results.xlsx"
i = 1
for input_len, output_len in input_output_pairs:
    for concurrency in concurrency_values:
        args.seed = i
        i += 1
        args.random_input_len = input_len
        args.random_output_len = output_len
        args.max_concurrency = concurrency
        # 根据并发数设置请求数
        if concurrency in [1, 16]:
            args.num_prompts = 50
        elif concurrency in [1025]:
            args.num_prompts = concurrency - 1
            args.max_concurrency = concurrency - 1
        else:
            args.num_prompts = concurrency * 2  # 请求个数是并发个数的2倍

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
