
import re
import subprocess
import pandas as pd

import logging
logger = logging.getLogger('')


def get_number_of_idle_nodes(hostname, partition, username):
    retry = True
    while retry:
        table = subprocess.run(
            f"ssh {username}@{hostname} \"sinfo -p {partition} --states idle -O NODES\"",
            capture_output=True,
            shell=True
        )

        response = str(table.stdout)
        error_str = str(table.stderr)
        if 'unbound variable' not in error_str:
            retry = True
            logger.error(f'Retrying because of : {error_str}')
        else:
            retry = False

    try:
        response = response.split("NODES")[1]
        print(f'NODES {response=}')
        number_of_idle_nodes = int(response.split(r"\n")[1])
    except:
        print(f'{response=}')
        number_of_idle_nodes = 0

    print(f'Number of idle nodes is {number_of_idle_nodes} on {partition}.')
    return number_of_idle_nodes


def get_available_resources(username, hostname, requested_partition='abc_a100'):
    resources = {}

    nodes = str(subprocess.run(
        f'ssh {username}@{hostname} "scontrol show nodes"',
        capture_output=True,
        shell=True
    ).stdout)

    node_names = list(map(lambda x: f"{x.replace('NodeName=', '')}.abc0", re.findall(r"NodeName=\w+", str(nodes))))
    node_partitions = list(map(lambda x: x.replace('Partitions=', ''), re.findall(r"Partitions=\w+", str(nodes))))
    configured = list(map(lambda x: int(x.replace('CPUTot=', '')), re.findall(r"CPUTot=\d+", str(nodes))))
    allocated = list(map(lambda x: int(x.replace('CPUAlloc=', '')), re.findall(r"CPUAlloc=\d+", str(nodes))))

    for name, partition, total_cpus, allocated_cpus in zip(node_names, node_partitions, configured, allocated):

        if partition == requested_partition:
            available_cpus = total_cpus - allocated_cpus

            # Remove once RMA is completed for missing A00 GPUs
            if name == 'g0003.abc0' or name == 'g0006.abc0':
                available_gpus = 3
            else:
                available_gpus = available_cpus // 4 if available_cpus != 0 else 0

            resources[name] = {
                "total_cpus": total_cpus,
                "available_cpus": available_cpus,
                "total_mem": f"{30 * total_cpus}GB",  # 30GB per core
                "available_mem": f"{30 * available_cpus}GB" if available_cpus != 0 else 0,  # 30GB per core
                "total_gpus": total_cpus // 4,  # 4 cores per gpu
                "available_gpus": available_gpus,  # 4 cores per gpu
            }

    return pd.DataFrame.from_dict(resources, orient='index')
