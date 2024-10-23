#!/bin/bash

# conda create -n ray -c conda-forge python=3.9 "ray-default" "ray-core" "ray-dashboard" -y
source ~/.bashrc
conda activate ray
echo "Activated conda environment: ray"

while getopts ":o:w:" option;do
    case "${option}" in
    o) o=${OPTARG}
        outdir=$o
    ;;
    w) w=${OPTARG}
        workload=$w
    ;;
    *) echo "Did not supply the correct arguments"
    ;;
    esac
done

ln -sf $outdir /tmp/ray_symlink
echo "Create symlink: ray"

############################## SETUP PORTS

#bias to selection of higher range ports
function getfreeport()
{
    CHECK="do while"
    while [[ ! -z $CHECK ]]; do
        port=$(( ( RANDOM % 40000 )  + 20000 ))
        CHECK=$(netstat -a | grep $port)
    done
    echo $port
}

port=$(getfreeport)
echo "Head node will use port: $port"
export port

dashboard_port=$(getfreeport)
echo "Dashboard will use port: $dashboard_port"
export dashboard_port


############################## FIND NODES/HOSTS

set -x

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)
cluster_address="$head_node_ip:$port"

export head_node
export head_node_ip
export cluster_address

############################## START HEAD NODE

job="srun --nodes=1 --ntasks=1 -w $head_node bash ray_start_cluster.sh -i $head_node_ip -p $port -d $dashboard_port -c $head_cpus -g $head_gpus"
echo $job
$job &

############################## ADD WORKER NODES

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    worker_job="srun --nodes=1 --ntasks=1 -w $node_i bash ray_start_worker.sh -a $cluster_address -c $head_cpus -g $head_gpus"
    echo $worker_job
    $worker_job &
done

############################## RUN WORKLOAD

echo "Starting workload: $workload"
$workload
