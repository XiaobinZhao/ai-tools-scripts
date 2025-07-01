import os
import re
import json
import glob
import traceback
from tqdm import tqdm
import ast


def extract_prompts_from_logs(log_dirs):
    # 查找所有日志文件
    log_files = []
    for log_dir in log_dirs:
        log_files.extend(glob.glob(os.path.join(log_dir, '*.log')))
    all_conversations = []
    conversations_index = 1
    for file_path in tqdm(log_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    prompt_max_tokens = re.findall(r"request_json={'max_tokens': (\d+)", line)
                    prompt_messages = re.search(r"'messages': (\[{.+}\])", line, re.DOTALL)

                    if not prompt_messages:
                        continue

                    all_conversations.append({
                        "id": str(conversations_index).zfill(5),  # 使用随机生成的 10 位字符串替换 uuid
                        "conversations": json.loads(json.dumps(ast.literal_eval(prompt_messages.group(1)), ensure_ascii=False)),
                        "max_tokens": int(prompt_max_tokens[0]) if prompt_max_tokens else None
                    })

                    conversations_index += 1

        except Exception as e:
            traceback.print_exc()
            print(f"Error processing {file_path}: {str(e)}")

    # 将字典转换为列表
    return all_conversations


def main():
    log_dir = r'D:\zxb94\desktop\AI-算力\tx-bench-0624sglang日志\tx-bench'
    log_dir2 = r'D:\zxb94\desktop\AI-算力\tx-bench-0624-02sglang日志\tx-bench-0624'
    output_file = f'tx_bench_longtext_dataset.json'

    datasets = extract_prompts_from_logs([log_dir, log_dir2])

    print(f"Found {len(datasets)} valid dialogue sequences")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(datasets, f, ensure_ascii=False, indent=2)

    print(f"Dataset saved to {output_file}")


if __name__ == "__main__":
    main()
