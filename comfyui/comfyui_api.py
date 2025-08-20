import json
import random
import sys
import threading
import time
import traceback
import urllib.request
import uuid
from urllib.parse import urlparse

import requests
import websocket
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def queue_workflow(workflow_json, client_id, server_address):
    payload = {
        "prompt": workflow_json,
        "client_id": client_id
    }
    headers = {'Content-Type': 'application/json'}
    data = json.dumps(payload).encode('utf-8')
    url = f"http://{server_address}/prompt"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req) as response:
        response_data = response.read().decode('utf-8')
        return json.loads(response_data)


def get_result(server_address, prompt_id, prompt_results):
    headers = {'Content-Type': 'application/json'}
    url = f"http://{server_address}/history/{prompt_id}"
    logger.info(f"发起结果请求`{url}`")

    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            response_data = response.json()
            logger.info(f"发起结果请求成功")

            prompt_results[prompt_id]["result"] = response_data

            if (prompt_id in response_data and
                    "outputs" in response_data[prompt_id] and
                    "9" in response_data[prompt_id]["outputs"] and
                    len(response_data[prompt_id]["outputs"]["9"]["images"]) > 0):
                logger.info(
                    f'获取结果: {prompt_results[prompt_id]['index']} - {response_data[prompt_id]["outputs"]["9"]["images"][0]["filename"]}')

                # 确保status和messages存在
                if ("status" in response_data[prompt_id] and
                        "messages" in response_data[prompt_id]["status"] and
                        len(response_data[prompt_id]["status"]["messages"]) > 2):
                    prompt_results[prompt_id].update({
                        "execution_start": response_data[prompt_id]["status"]["messages"][0][1]["timestamp"],
                        "execution_end": response_data[prompt_id]["status"]["messages"][2][1]["timestamp"]
                    })

                # 标记此任务成功完成（成功获取到结果）
                prompt_results[prompt_id]["success"] = True
            else:
                prompt_results[prompt_id]["success"] = False
                logger.warning(f"结果数据不完整: {prompt_results[prompt_id]['index']}-{prompt_id} - 重试")
                return "retry"

        else:
            raise requests.HTTPError(f"请求 {url} 失败，状态码: {response.status_code}, text: {response.text}; reason: {response.reason}")
    except requests.exceptions.RequestException as e:
        logger.error(f"请求 {url} 时发生错误: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"解析响应JSON时发生错误: {e}")
        raise
    except Exception as e:
        logger.error(f"获取结果时发生未知错误: {e}")
        raise


def run_test(total_tasks, server_address, workflow_json, dataset):
    # 为每个测试创建独立的变量，实现线程隔离
    client_id = uuid.uuid4().hex
    prompt_results = {}
    finished = 0
    # 创建事件用于通知所有任务完成
    all_done = threading.Event()

    # 使用线程锁保护共享资源
    finished_lock = threading.Lock()

    # 创建WebSocket连接（只连接可用的服务器）
    websocket_urls = [
        f"ws://127.0.0.1:3000/ws?clientId={client_id}",
        f"ws://127.0.0.1:6001/ws?clientId={client_id}",
        f"ws://127.0.0.1:6002/ws?clientId={client_id}",
        f"ws://127.0.0.1:6003/ws?clientId={client_id}",
        f"ws://127.0.0.1:6004/ws?clientId={client_id}",
        f"ws://127.0.0.1:6005/ws?clientId={client_id}",
        f"ws://127.0.0.1:6006/ws?clientId={client_id}",
        f"ws://127.0.0.1:6007/ws?clientId={client_id}"
    ]

    websockets = []
    ws_threads = []

    # 跟踪连接状态
    connected_websockets = 0

    def on_message(ws, message, url):
        nonlocal finished
        try:
            msg = json.loads(message)
            # print(msg)
            if msg.get("type") == "execution_success" and msg["data"].get("prompt_id", None) in prompt_results.keys():

                end_time = time.perf_counter()
                prompt_id = msg["data"]["prompt_id"]

                # 检查prompt_id是否存在于prompt_results中
                if prompt_id in prompt_results:
                    prompt_results[prompt_id]["end_at"] = end_time
                    logger.info(f"任务{prompt_results[prompt_id]['index']}-{prompt_id}完成！")

                    try:
                        result = get_result(url, prompt_id, prompt_results)
                        if result == "retry":
                            logger.info(f"任务{prompt_results[prompt_id]['index']}-{prompt_id}重试中...")
                            time.sleep(1)
                            get_result(url, prompt_id, prompt_results)
                    except Exception as e:
                        exc_info = sys.exc_info()
                        logger.error(f"获取结果 {prompt_results[prompt_id]['index']} 时出错: {"".join(traceback.format_exception(*exc_info))}")
                        prompt_results[prompt_id]["success"] = False
                    finally:
                        # 使用锁保护共享变量
                        with finished_lock:
                            finished = finished + 1
                            current_finished = finished

                    # 检查是否所有任务都已完成
                    if current_finished >= total_tasks:
                        all_done.set()  # 设置事件，表示所有任务完成
                        # 关闭所有WebSocket连接
                        for ws_conn in websockets:
                            try:
                                ws_conn.close()
                            except:
                                pass
                else:
                    logger.warning(f"收到未知prompt_id的消息: {prompt_id}")
        except json.JSONDecodeError:
            logger.error(f"无法解析WebSocket消息: {message}")
        except Exception as e:
            logger.error(f"处理WebSocket消息时出错: {e}")

    def make_callback(url):
        def on_message_wrapper(ws, message):
            return on_message(ws, message, urlparse(url).netloc)

        return on_message_wrapper

    # 创建并启动WebSocket连接
    for i, ws_url in enumerate(websocket_urls):
        try:
            ws = websocket.WebSocketApp(ws_url, on_message=make_callback(ws_url))
            ws_thread = threading.Thread(target=ws.run_forever, name=f"WebSocketThread-{i}")
            ws_thread.daemon = True
            ws_thread.start()

            websockets.append(ws)
            ws_threads.append(ws_thread)
            connected_websockets += 1
            time.sleep(0.1)  # 稍微间隔一下，避免同时连接造成压力
        except Exception as e:
            logger.error(f"无法连接到WebSocket {ws_url}: {e}")

    logger.info(f"成功建立 {connected_websockets} 个WebSocket连接")

    # 触发任务
    for i in range(total_tasks):
        workflow_json["25"]["inputs"]["noise_seed"] = random.randint(0, 1000000000)
        workflow_json["6"]["inputs"]["text"] = dataset[random.randint(0, len(dataset) - 1)]
        logger.info("正在触发第 %d 次任务...", i + 1)
        start_time = time.perf_counter()
        # 触发工作流
        try:
            response = queue_workflow(workflow_json, client_id, server_address)
            logger.info("API响应：%s", response)
            prompt_results[response["prompt_id"]] = {"index": i, "start_at": start_time}
        except Exception as e:
            logger.error(f"触发任务时出错: {e}")

    # 等待所有任务完成（设置超时）
    logger.info("等待所有任务完成...")
    if not all_done.wait(timeout=300):  # 5分钟超时
        logger.warning("等待超时，任务可能未全部完成")

    # 计算成功请求数量（基于是否成功获取到结果）
    successful_requests = len([p for p in prompt_results if prompt_results[p].get("success", False)])

    # 计算平均时延（仅计算已完成的任务）
    completed_tasks = len([p for p in prompt_results if "end_at" in prompt_results[p]])
    if completed_tasks > 0:
        avg_e2e_delay = sum(prompt_results[prompt_id]["end_at"] - prompt_results[prompt_id]["start_at"]
                            for prompt_id in prompt_results if "end_at" in prompt_results[prompt_id]) / completed_tasks

        # 只计算有执行时间数据的任务
        valid_tasks = len([p for p in prompt_results if "execution_end" in prompt_results[p]])
        if valid_tasks > 0:
            avg_prompt_delay = sum(prompt_results[prompt_id]["execution_end"] - prompt_results[prompt_id]["execution_start"]
                                   for prompt_id in prompt_results if "execution_end" in prompt_results[prompt_id]) / 1000 / valid_tasks
        else:
            avg_prompt_delay = 0

        logger.info(f"平均端到端延迟时间：{avg_e2e_delay:.2f}秒")
        logger.info(f"平均prompt执行时间：{avg_prompt_delay:.2f}秒")
        logger.info(f"任务完成情况：{completed_tasks}/{total_tasks}")
        logger.info(f"请求成功个数：{successful_requests}/{total_tasks}")
    else:
        logger.error("没有任务成功完成")
        avg_e2e_delay = 0
        avg_prompt_delay = 0
        successful_requests = 0

    logger.info("测试完成！")
    # 保存结果
    with open(f"results_{total_tasks}.json", "w") as f:
        json.dump(prompt_results, f, indent=4)

    return avg_e2e_delay, avg_prompt_delay, successful_requests


def main():
    server_address = "127.0.0.1:40008"

    # 读取工作流JSON
    with open('./flux_WaveSpeed_wf.json', 'r', encoding='utf-8') as f:
        workflow_json = json.load(f)
    with open('./xdit_dataset.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # 定义要测试的任务数量
    task_counts = [1, 2, 4, 8, 16, 32, 64, 128]
    # task_counts = [1, 2, 4, 8]
    results = []

    # 对每个任务数量运行测试
    for count in task_counts:
        logger.info(f"=====================================开始测试 {count} 个任务====================================")
        avg_e2e, avg_prompt, successful_requests = run_test(count, server_address, workflow_json, dataset)
        results.append({
            'tasks': count,
            'e2e_delay': avg_e2e,
            'prompt_delay': avg_prompt,
            'successful_requests': successful_requests
        })

    # 输出Markdown表格，包含请求成功个数
    print("\n## 测试结果")
    print("| 任务数量 | 请求成功个数 | 平均端到端延迟(秒) | 平均Prompt执行时间(秒) |")
    print("|---------|------------|------------------|---------------------|")
    for result in results:
        print(f"| {result['tasks']} | {result['successful_requests']} | {result['e2e_delay']:.2f} | {result['prompt_delay']:.2f} |")


if __name__ == "__main__":
    main()
