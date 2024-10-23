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

head_node=$(hostname)
head_node_ip=$(hostname --ip-address)
cluster_address="$head_node_ip:$port"

export head_node
export head_node_ip
export cluster_address

gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," "{print NF}")
cpus=$(nproc)

############################## START HEAD NODE

bash ray_start_cluster.sh -i $head_node_ip -p $port -c $cpus -g $gpus &

rpids=$(pgrep -u $USER ray)
echo "Ray head node PID:"
echo $rpids

############################## RUN WORKLOAD

#Run workload
echo $workload
$workload
