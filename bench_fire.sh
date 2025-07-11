#!/bin/bash

# 定义 input 和 output 长度
INPUT_LEN=3500
OUTPUT_LEN=1500

# 创建 logs 目录（如果不存在）
mkdir -p logs

# 定义 request-rate 和 max-concurrency
declare -a request_rates=(1 4 8 16 32 48 64 80 128 160)
declare -a max_concurrency=(1 4 8 16 32 48 64 80 128 160)
declare -a num_prompts=(4 16 32 64 128 192 256 320 512 640)

# 运行 benchmark 测试
for i in "${!request_rates[@]}"; do
    RATE=${request_rates[$i]}
    CONCURRENCY=${max_concurrency[$i]}
    PROMPTS=${num_prompts[$i]}

    echo "Running benchmark with request rate $RATE, concurrency $CONCURRENCY, prompts $PROMPTS"

    python3 -m sglang.bench_serving --backend sglang \
        --dataset-name random \
        --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
        --random-input-len $INPUT_LEN \
        --random-output-len $OUTPUT_LEN \
        --random-range-ratio 1 \
        --request-rate $RATE \
        --max-concurrency $CONCURRENCY \
        --num-prompts $PROMPTS \
        --host 0.0.0.0 --port 40000 \
        > logs/${CONCURRENCY}_fp8_r1_${INPUT_LEN}_${OUTPUT_LEN}.txt 2>&1

    wait
done