#!/bin/bash

# 20
python3 -m sglang.bench_serving --backend sglang-oai \
--dataset-name random --random-input 6000 --random-output 1000 --random-range-ratio 1 \
--host 0.0.0.0 --port 8000 \
--model /root/.cache/huggingface/DeepSeek-R1 \
--dataset-path /root/.cache/huggingface/ShareGPT_V3_unfiltered_cleaned_split.json \
--max-concurrency 20 \
--num-prompts 2000 --seed 20

sleep 1

# 30
python3 -m sglang.bench_serving --backend sglang-oai \
--dataset-name random --random-input 6000 --random-output 1000 --random-range-ratio 1 \
--host 0.0.0.0 --port 8000 \
--model /root/.cache/huggingface/DeepSeek-R1 \
--dataset-path /root/.cache/huggingface/ShareGPT_V3_unfiltered_cleaned_split.json \
--max-concurrency 30 \
--num-prompts 2000 --seed 30

sleep 1


# 40
python3 -m sglang.bench_serving --backend sglang-oai \
--dataset-name random --random-input 6000 --random-output 1000 --random-range-ratio 1 \
--host 0.0.0.0 --port 8000 \
--model /root/.cache/huggingface/DeepSeek-R1 \
--dataset-path /root/.cache/huggingface/ShareGPT_V3_unfiltered_cleaned_split.json \
--max-concurrency 40 \
--num-prompts 2000 --seed 40

sleep 1

# 50
python3 -m sglang.bench_serving --backend sglang-oai \
--dataset-name random --random-input 6000 --random-output 1000 --random-range-ratio 1 \
--host 0.0.0.0 --port 8000 \
--model /root/.cache/huggingface/DeepSeek-R1 \
--dataset-path /root/.cache/huggingface/ShareGPT_V3_unfiltered_cleaned_split.json \
--max-concurrency 50 \
--num-prompts 2000 --seed 50

sleep 1


# 60
python3 -m sglang.bench_serving --backend sglang-oai \
--dataset-name random --random-input 6000 --random-output 1000 --random-range-ratio 1 \
--host 0.0.0.0 --port 8000 \
--model /root/.cache/huggingface/DeepSeek-R1 \
--dataset-path /root/.cache/huggingface/ShareGPT_V3_unfiltered_cleaned_split.json \
--max-concurrency 60 \
--num-prompts 2000 --seed 60

sleep 1

# 70
python3 -m sglang.bench_serving --backend sglang-oai \
--dataset-name random --random-input 6000 --random-output 1000 --random-range-ratio 1 \
--host 0.0.0.0 --port 8000 \
--model /root/.cache/huggingface/DeepSeek-R1 \
--dataset-path /root/.cache/huggingface/ShareGPT_V3_unfiltered_cleaned_split.json \
--max-concurrency 70 \
--num-prompts 2000 --seed 70

sleep 1


# 80
python3 -m sglang.bench_serving --backend sglang-oai \
--dataset-name random --random-input 6000 --random-output 1000 --random-range-ratio 1 \
--host 0.0.0.0 --port 8000 \
--model /root/.cache/huggingface/DeepSeek-R1 \
--dataset-path /root/.cache/huggingface/ShareGPT_V3_unfiltered_cleaned_split.json \
--max-concurrency 80 \
--num-prompts 2000 --seed 80

sleep 1


# 90
python3 -m sglang.bench_serving --backend sglang-oai \
--dataset-name random --random-input 6000 --random-output 1000 --random-range-ratio 1 \
--host 0.0.0.0 --port 8000 \
--model /root/.cache/huggingface/DeepSeek-R1 \
--dataset-path /root/.cache/huggingface/ShareGPT_V3_unfiltered_cleaned_split.json \
--max-concurrency 90 \
--num-prompts 2000 --seed 90

sleep 1


# 100
python3 -m sglang.bench_serving --backend sglang-oai \
--dataset-name random --random-input 6000 --random-output 1000 --random-range-ratio 1 \
--host 0.0.0.0 --port 8000 \
--model /root/.cache/huggingface/DeepSeek-R1 \
--dataset-path /root/.cache/huggingface/ShareGPT_V3_unfiltered_cleaned_split.json \
--max-concurrency 100 \
--num-prompts 2000 --seed 100