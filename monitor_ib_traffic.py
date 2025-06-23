"""
perfquery可以观测ib网卡的数据流量。但是这个命令只能运行一次得到一次结果。
本脚本可以实现定时更新输出结果到屏幕上，方便观察。其结果以表格方式输出

pip install tabulate
"""

import logging
import re
import sys
import time
import subprocess
from tabulate import tabulate


METRIC_NAMES = ["PortXmitData", "PortRcvData"]
metrics = {}


def decode_str_list(line_list):
    return [x.decode("utf-8") for x in line_list]


def get_cmd_out(cmd):
    return decode_str_list(subprocess.Popen(cmd, stdout=subprocess.PIPE).stdout.readlines())


def ibstat_ports():
    lid7port = []
    ibstat = get_cmd_out("ibstat")
    for index, line in enumerate(ibstat):
        line = line.strip()
        match = re.match("Port [0-9]\:", line)
        if match:
            link_layer = ibstat[index + 9].split(':')[1].strip()
            if link_layer != "InfiniBand":
                continue
            number = line.split(' ')[1].replace(':', '')
            state = ibstat[index + 1].split(':')[1].strip()
            an = re.match("Active", state)
            if an:
                lid = ibstat[index + 4].split(':')[1].strip()

                nic_str = re.match("CA \'([a-zA-Z0-9]+)\'", ibstat[index - 7])
                nic = None
                if nic_str:
                    nic = nic_str.group(1)

                lid7port.append((nic or "", lid, number))
    return lid7port


# Return a key-value pair, eventually empty if the line didn't match
def parse_counter_line(line, keys):
    if re.match("^[a-zA-Z0-9]*\:\.\.\.*[0-9]*$", line):
        line = line.split(':')
        key = line[0]
        if key in keys:
            value = line[1].replace('.','').strip()
            return (key, int(value))
    return ("", 0)


# Parse the complete input from perfquery for lines matching counters,
# and return all counters and their values as dictionary
def parse_counters(counters, keys):
    counts = {}
    for line in counters:
        key, value = parse_counter_line(line, keys)
        # Omit empty return values...
        if key:
            logging.debug("[parse_counters] Found counter: %s=%s", key, value)
            counts[key] = value
    return counts


# Call perfquery for extended traffic counters, and reset the counters
def traffic_counter(lid, port=1):
    command = ["/usr/sbin/perfquery", "-x", "-r", lid, port]
    logging.debug("[traffic_counters] Execute command: %s", " ".join(command))
    counters = get_cmd_out(command)
    return parse_counters(counters, METRIC_NAMES)


def init_metric():
    metrics["last_update"] = time.time()


def update_metric():
    global metrics

    # NOTE: time_since_last_update is not calculated precisely
    time_since_last_update = time.time() - metrics["last_update"]
    logging.debug("[update_metrics] Update metrics after %ss", time_since_last_update)

    for nic, lid, port in ibstat_ports():
        metric2counts = traffic_counter(lid, port)
        metrics[lid] = {port: metric2counts, "nic": nic}
        for metric in METRIC_NAMES:
            # Data port counters indicate octets divided by 4 rather than just octets.
            #
            # It's consistent with what the IB spec says (IBA 1.2 vol 1 p.948) as to
            # how these quantities are counted. They are defined to be octets divided
            # by 4 so the choice is to display them the same as the actual quantity
            # (which is why they are named Data rather than Octets) or to multiply by
            # 4 for Octets. The former choice was made.
            #
            # For simplification the values are multiplied by 4 to represent octets/bytes
            num_bytes = metric2counts[metric] * 4
            metrics[lid][port][metric.replace("Data", "Bytes")] = num_bytes
            metrics[lid][port][metric.replace("Data", "GB/s")] = num_bytes / (time_since_last_update * 1024 * 1024 * 1024)

    metrics["last_update"] = time.time()


if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)
    update_interval = 1 if len(sys.argv) == 1 else int(sys.argv[1])  # default is 1s

    init_metric()

    while True:
        update_metric()

        # 使用 ANSI 转义序列实现局部刷新
        print("\033[H\033[J", end="")  # 将光标移动到屏幕顶部并清除屏幕内容

        # 将 metrics 数据以表格形式打印
        table_data = []
        for lid, ports in metrics.items():
            if lid == "last_update":
                continue
            nic = metrics[lid]["nic"]
            for port, data in ports.items():
                if port == "nic":
                    continue
                row = [
                    nic,
                    lid,
                    port,
                    data.get("PortXmitBytes", 0),
                    data.get("PortRcvBytes", 0),
                    data.get("PortXmitGB/s", 0),
                    data.get("PortRcvGB/s", 0),
                ]
                table_data.append(row)

        headers = ["NIC", "LID", "Port", "Xmit Bytes", "Rcv Bytes", "Xmit GB/s", "Rcv GB/s"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        time.sleep(update_interval)
