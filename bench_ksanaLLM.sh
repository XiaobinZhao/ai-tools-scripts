#!/bin/bash

# 定义output 长度
OUTPUT_LEN=1000

# 创建 logs 目录（如果不存在）
mkdir -p logs

# 定义 request-rate 和 max-concurrency
declare -a seed=(1 2 3 4 5 6 7 8 9 10 11 12)
declare -a max_concurrency=(1 5 10 20 30 40 50 60 70 80 90 100)
declare -a num_prompts=(50 50 50 60 60 80 100 120 140 160 180 200)

# 运行 benchmark 测试
for i in "${!max_concurrency[@]}"; do
    CONCURRENCY=${max_concurrency[$i]}
    PROMPTS=${num_prompts[$i]}
    SEED=${seed[$i]}

    echo "Running benchmark with concurrency $CONCURRENCY, prompt num $PROMPTS"

    python /workspace/KsanaLLM/benchmarks/benchmark_throughput.py \
        --dataset_name longbenchV2noCtx \
        --dataset_path /workspace/KsanaLLM/benchmarks/LongBench-v2.json \
        --backend ksana --model_type deepseek_r1 --mode async  \
        --output_csv output_res.csv --perf_csv perf_res.csv \
        --tokenizer_path /root/.cache/huggingface/DeepSeek-R1  --shuffle  --stream \
        --host localhost --port 8000 \
        --concurrency $CONCURRENCY --prompt_num $PROMPTS  --seed $SEED --max_new_tokens $OUTPUT_LEN \
        > logs/${CONCURRENCY}_fp8_r1_longbenchV2noCtx_${OUTPUT_LEN}.txt 2>&1

    wait
done