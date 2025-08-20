# Adapted from https://github.com/vllm-project/vllm/blob/6366efc67b0aedd2c1721c14385370e50b297fb3/benchmarks/backend_request_func.py
# Adapted from https://github.com/vllm-project/vllm/blob/6366efc67b0aedd2c1721c14385370e50b297fb3/benchmarks/benchmark_serving.py

"""
Benchmark online serving with dynamic requests.

Usage:
python3 -m sglang.bench_serving --backend sglang --num-prompt 10

python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 3000 --random-input 1024 --random-output 1024 --random-range-ratio 0.5
"""

import argparse
import asyncio
import json
import os
import random
import resource
import sys
import time
import traceback
import warnings
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from json import JSONDecodeError
from typing import AsyncGenerator, List, Optional

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

ASSISTANT_SUFFIX = "Assistant:"

global args


def _create_bench_client_session():
    # When the pressure is big, the read buffer could be full before aio thread read
    # the content. We increase the read_bufsize from 64K to 10M.
    # Define constants for timeout and buffer size for clarity and maintainability
    BENCH_AIOHTTP_TIMEOUT_SECONDS = 6 * 60 * 60  # 6 hours
    BENCH_AIOHTTP_READ_BUFSIZE_BYTES = 10 * 1024**2  # 10 MB

    aiohttp_timeout = aiohttp.ClientTimeout(total=BENCH_AIOHTTP_TIMEOUT_SECONDS)
    return aiohttp.ClientSession(
        timeout=aiohttp_timeout, read_bufsize=BENCH_AIOHTTP_READ_BUFSIZE_BYTES
    )


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    seed: int
    image_data: str = ""
    num_inference_steps: int = 20
    guidance_scale: float = 3.5
    save_disk_path: str = "/data/output"



@dataclass
class RequestFuncOutput:
    """
    {
        "message": "Image generated successfully",
        "elapsed_time": "1.05 sec",
        "output": "/data/output/generated_image_20250819-060731.png",
        "save_to_disk": true
    }
    """
    message: str = ""
    success: bool = False
    save_to_disk: bool = True
    prompt_latency: float = 0.0
    e2e_latency: float = 0.0
    output_file_path: str = ""


async def async_request_generate(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "generate"
    ), "XDit API URL must end with 'generate'."


    async with _create_bench_client_session() as session:
        payload = {
            "num_inference_steps": request_func_input.num_inference_steps,
            "prompt": request_func_input.prompt,
            "cfg": request_func_input.guidance_scale,
            "seed": request_func_input.seed,
            "save_disk_path": request_func_input.save_disk_path
        }
        output = RequestFuncOutput()
        st = time.perf_counter()
        try:
            async with session.post(
                url=api_url, json=payload, headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    latency = time.perf_counter() - st
                    response_json = await response.json()
                    output.success = True
                    output.message = response_json["message"]
                    output.save_to_disk = bool(response_json["save_to_disk"])
                    output.prompt_latency = float(response_json["elapsed_time"].replace(" sec", ""))
                    output.e2e_latency = latency
                    output.output_file_path = response_json["output"]
                else:
                    output.message = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.message = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output



@dataclass
class BenchmarkMetrics:
    completed: int
    request_throughput: float
    e2e_latency: float
    mean_latency: float
    median_latency: float
    std_latency: float
    p99_latency: float
    concurrency: float


def is_file_valid_json(path):
    if not os.path.isfile(path):
        return False

    # TODO can fuse into the real file open later
    try:
        with open(path) as f:
            json.load(f)
        return True
    except JSONDecodeError as e:
        print(
            f"{path} exists but json loading fails ({e=}), thus treat as invalid file"
        )
        return False


@dataclass
class DatasetRow:
    prompt: str
    prompt_len: int
    output_len: int
    image_data: Optional[str] = None


def get_dataset():
    with open(args.dataset_path) as f:
        dataset = json.load(f)

    # Shuffle the dataset.
    random.shuffle(dataset)
    if len(dataset) < args.num_prompts:
        print(
            f"{len(dataset)=} < {args.num_prompts=}, using all dataset instead of {args.num_prompts} prompts"
        )
        args.num_prompts = len(dataset)
    filtered_dataset = dataset[: args.num_prompts]

    return filtered_dataset


async def get_request(
    input_requests: List[str],
    request_rate: float,
) -> AsyncGenerator[DatasetRow, None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
    outputs: List[RequestFuncOutput],
    dur_s: float,
) -> BenchmarkMetrics:
    completed = 0
    prompt_latencies: List[float] = []
    e2e_latencies: List[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            prompt_latencies.append(outputs[i].prompt_latency)
            e2e_latencies.append(outputs[i].e2e_latency)
            completed += 1

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )
    metrics = BenchmarkMetrics(
        completed=completed,
        request_throughput=completed / dur_s,
        e2e_latency=np.mean(e2e_latencies),
        mean_latency=np.mean(prompt_latencies),
        median_latency=np.median(prompt_latencies),
        std_latency=np.std(prompt_latencies),
        p99_latency=np.percentile(prompt_latencies, 99),
        concurrency=np.sum(prompt_latencies) / dur_s,
    )

    return metrics


async def benchmark(
    api_url: str,
    input_requests: List[DatasetRow],
    request_rate: float,
    max_concurrency: Optional[int],
    warmup_requests: int = 1,
):
    # Limit concurrency
    # From https://github.com/vllm-project/vllm/pull/9390
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await async_request_generate(request_func_input=request_func_input, pbar=pbar)
        async with semaphore:
            return await async_request_generate(request_func_input=request_func_input, pbar=pbar)

    # Warmup
    print(f"Starting warmup with {warmup_requests} sequences...")

    # Use the first request for all warmup iterations
    test_request = input_requests[0]

    # Create the test input once
    test_input = RequestFuncInput(
        prompt=test_request,
        api_url=api_url,
        seed=args.seed
    )

    # Run warmup requests
    warmup_tasks = []
    for _ in range(warmup_requests):
        warmup_tasks.append(
            asyncio.create_task(async_request_generate(request_func_input=test_input))
        )

    warmup_outputs = await asyncio.gather(*warmup_tasks)

    # Check if at least one warmup request succeeded
    if warmup_requests > 0 and not any(output.success for output in warmup_outputs):
        raise ValueError(
            "Warmup failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {warmup_outputs[0].message}"
        )
    else:
        print(
            f"Warmup completed with {args.warmup_requests} sequences. Starting main benchmark run..."
        )


    time.sleep(1.0)


    pbar = tqdm(total=len(input_requests))

    # Run all requests
    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate):

        request_func_input = RequestFuncInput(
            prompt=request,
            api_url=api_url,
            seed=args.seed
        )

        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input, pbar=pbar)
            )
        )
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)


    if pbar is not None:
        pbar.close()

    # Compute metrics and print results
    benchmark_duration = time.perf_counter() - benchmark_start_time
    metrics = calculate_metrics(
        outputs=outputs,
        dur_s=benchmark_duration
    )

    print("\n{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Traffic request rate:", request_rate))
    print(
        "{:<40} {:<10}".format(
            "Max request concurrency:",
            max_concurrency if max_concurrency else "not set",
        )
    )
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    print("{:<40} {:<10.2f}".format("Concurrency:", metrics.concurrency))
    print("{s:{c}^{n}}".format(s="Latency", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean Prompt Latency (s):", metrics.mean_latency))
    print("{:<40} {:<10.2f}".format("Median Prompt Latency (s):", metrics.median_latency))
    print("{:<40} {:<10.2f}".format("Mean E2E Latency (s):", metrics.e2e_latency))
    print("=" * 50)

    if metrics.completed > 0:
        result = {
            "request_rate": request_rate,
            "max_concurrency": max_concurrency,
            "duration": benchmark_duration,
            "completed": metrics.completed,
            "request_throughput": metrics.request_throughput,
            "e2e_latency": metrics.e2e_latency,
            "mean_latency": metrics.mean_latency,
            "median_latency": metrics.median_latency,
            "std_latency": metrics.std_latency,
            "p99_latency": metrics.p99_latency,
            "concurrency": metrics.concurrency
        }
    else:
        print(f"Error running benchmark for request rate: {request_rate}")
        print("-" * 30)

    # Determine output file name
    if args.output_file:
        output_file_name = args.output_file
    else:
        now = datetime.now().strftime("%m%d")
        output_file_name = f"XDit_{now}_{args.seed}_{args.max_concurrency}_{args.num_prompts}.jsonl"

    result_details = {
        "output_file_path": [output.output_file_path for output in outputs],
        "message": [output.message for output in outputs],
    }

    # Append results to a JSONL file
    with open(output_file_name, "a") as file:
        if args.output_details:
            result_for_dump = result | result_details
        else:
            result_for_dump = result
        file.write(json.dumps(result_for_dump) + "\n")

    return result | result_details


def set_global_args(args_: argparse.Namespace):
    """Set the global args."""
    global args
    args = args_


def run_benchmark(args_: argparse.Namespace):
    global args
    args = args_

    # Set default value for max_concurrency if not present
    if not hasattr(args, "max_concurrency"):
        args.max_concurrency = None

    # Set default value for warmup_requests if not present
    if not hasattr(args, "warmup_requests"):
        args.warmup_requests = 1

    if not hasattr(args, "output_details"):
        args.output_details = False

    print(f"benchmark_args={args}")

    # Set global environments
    set_ulimit()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Set url
    base_url = args.base_url if args.base_url else "http://127.0.0.1:6000"
    api_url = f"{base_url}/generate"
    # Read dataset
    input_requests = get_dataset()


    return asyncio.run(
        benchmark(
            api_url=api_url,
            input_requests=input_requests,
            request_rate=args.request_rate,
            max_concurrency=args.max_concurrency,
            warmup_requests=args.warmup_requests,
        )
    )


def set_ulimit(target_soft_limit=65535):
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            print(f"Fail to set RLIMIT_NOFILE: {e}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:6000",
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="seed.",
    )
    parser.add_argument(
        "--dataset-path", type=str, default="", help="Path to the dataset."
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process. Default is 1000.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize the request arrival times. Default is inf.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up.",
    )
    parser.add_argument("--output-file", type=str, help="Output JSONL file name.")
    parser.add_argument(
        "--output-details", action="store_true", help="Output details of benchmarking."
    )
    args = parser.parse_args()
    run_benchmark(args)
