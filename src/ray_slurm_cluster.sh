#!/bin/bash

############################## CHECK ARGS

function get_free_port()
{
    CHECK="do while"
    while [[ ! -z $CHECK ]]; do
        port=$(( ( RANDOM % 40000 ) + 19999 ))
        CHECK=$(netstat -a | grep $port)
    done
    echo $port
}

while getopts ":c:" option;do
    case "${option}" in
    c) c=${OPTARG}
        workload=$c
    ;;
    *) echo "Did not supply the correct arguments"
    ;;
    esac
done


############################## FIND NODES/HOSTS

set -x

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)


HEAD_NODE=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)
export HEAD_NODE

############################## SETUP PORTS

RAY_PORT=6379
if [[ ! -z $(netstat -a | grep $RAY_PORT) ]]; then
  RAY_PORT=$(get_free_port)
fi
echo "Head node will use port: $RAY_PORT"
export RAY_PORT

HEAD_ADDRESS="${HEAD_NODE}:${RAY_PORT}"
echo "Head address: $HEAD_ADDRESS"
export HEAD_ADDRESS

RAY_DASHBOARD_PORT=8265
if [[ ! -z $(netstat -a | grep $RAY_PORT) ]]; then
  RAY_DASHBOARD_PORT=$(get_free_port)
fi
echo "Dashboard will use port: $RAY_DASHBOARD_PORT"
export RAY_DASHBOARD_PORT

############################## START HEAD NODE

echo "Starting HEAD at $HEAD_NODE"
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    ray start --head --node-ip-address="$HEAD_ADDRESS" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block &

############################## ADD WORKER NODES

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$HEAD_ADDRESS" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block &
    sleep 5
done

############################## RUN WORKLOAD

echo "Starting workload: $workload"
$workload