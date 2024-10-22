#!/bin/bash

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

source ~/.bashrc
conda activate ray
echo "Activated conda environment: ray"


ln -sf $outdir /tmp/ray_symlink
echo "Create symlink: ray"

port=$(getfreeport)
echo "Head node will use port: $port"
export port

dashboard_port=$(getfreeport)
echo "Dashboard will use port: $dashboard_port"
export dashboard_port


redis_password=$(uuidgen)
export redis_password

head_node=$(hostname)
head_node_ip=$(hostname --ip-address)
cluster_address="$head_node_ip:$port"
client_server_port=10001

export head_node
export head_node_ip
export cluster_address
export client_server_port


if [ -z "$CUDA_VISIBLE_DEVICES" ]
then
    num_gpu_for_head=0
else
    num_gpu_for_head=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," "{print NF}")
fi


if [ -z $num_cpu_for_head ]
then
    echo "Using all cores available on $head_node"
    num_cpu_for_head=$(nproc)
else
    echo "The object store memory in bytes is: $num_cpu_for_head"
fi

echo "STARTING HEAD at $head_node"
echo "Head node IP: $head_node_ip"
./ray_start_cluster.sh $head_node_ip $port &


#Run workload
echo $workload