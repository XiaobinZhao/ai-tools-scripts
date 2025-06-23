import os
import re
import json
import glob
import traceback
from tqdm import tqdm


def format_conversations(full_prompt):
    # 提取对话中的内容
    conversations = []
    # 去除对话开头的标识符
    full_prompt_after_replace_prefix = re.sub(r"<\｜begin▁of▁sentence\｜>[0-9a-z\-]+", "", full_prompt)
    # 去除对话结尾的标识符
    full_prompt_after_replace_end = re.sub(r"<\｜Assistant\｜><think>", "", full_prompt_after_replace_prefix)
    # 根据<｜end▁of▁sentence｜>分割字符串；最后一个元素往往只有用户提问
    _conversations = full_prompt_after_replace_end.split("<｜end▁of▁sentence｜>")
    conversations = []
    for conversation in _conversations:
        q_a = conversation.split("<｜Assistant｜>")
        if len(q_a) == 2:
            _q = q_a[0].replace("<｜User｜>", "")
            _a = q_a[1]
        else:
            _q = q_a[0].replace("<｜User｜>", "")
            _a = ""
        conversations.append({
            "role": "user",
            "content": _q
        })
        if _a:
            conversations.append({
                "role": "assistant",
                "content": _a
            })
    return conversations


def extract_prompts_from_logs(log_dir, mode="dataset"):
    # 查找所有日志文件
    log_files = glob.glob(os.path.join(log_dir, '*.log'))
    all_conversations = []
    conversations_index = 1
    for file_path in tqdm(log_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    # (?<!\\) 是一个负向零宽断言（negative lookbehind），表示匹配的位置前面不能有反斜杠（\）。
                    # 匹配单引号中的内容, 忽略转义字符,包含匹配的换行符
                    match = re.findall(r"(?<!\\)prompt: '(.*?)(?<!\\)', params: SamplingParams", line, re.DOTALL)
                    if not match:
                        continue

                    full_prompt = match[0]

                    if mode == "prompt":
                        all_conversations.append({
                            "id": str(conversations_index).zfill(5),  # 使用随机生成的 10 位字符串替换 uuid
                            "prompt": full_prompt
                        })
                    else:
                        all_conversations.append({
                            "id": str(conversations_index).zfill(5),  # 使用随机生成的 10 位字符串替换 uuid
                            "conversations": format_conversations(full_prompt)
                        })

                    conversations_index += 1

        except Exception as e:
            traceback.print_exc()
            print(f"Error processing {file_path}: {str(e)}")

    # 将字典转换为列表
    return all_conversations


def main(mode="dataset"):
    log_dir = r'D:\zxb94\desktop\AI-算力\tx-bench-0429vllm日志'
    output_file = f'tx_bench_{mode}.json'

    datasets = extract_prompts_from_logs(log_dir, mode)

    print(f"Found {len(datasets)} valid dialogue sequences")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(datasets, f, ensure_ascii=False, indent=2)

    print(f"Dataset saved to {output_file}")


if __name__ == "__main__":
    main("dataset")  # dataset 或者 prompt; dataset会格式化为ShareGPT那样的数据集格式；prompt直接提取prompt文本
