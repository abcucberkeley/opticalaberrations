#!/bin/bash

# conda create -n ray -c conda-forge python=3.9 "ray-default" "ray-core" "ray-dashboard" -y
source ~/.bashrc
conda activate ray
echo "Activated conda environment: ray"

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

while getopts ":i:p:d:c:g:" option;do
    case "${option}" in
    i) i=${OPTARG}
       ip=$i
       echo ip=$ip
    ;;
    p) p=${OPTARG}
       port=$p
       echo port=$port
    ;;
    d) d=${OPTARG}
       dashboard_port=$d
       echo dashboard_port=$dashboard_port
    ;;
    c) c=${OPTARG}
       cpus=$c
       echo cpus=$cpus
    ;;
    g) g=${OPTARG}
       gpus=$g
       echo gpus=$gpus
    ;;
    *) echo "Did not supply the correct arguments"
    ;;
    esac
done

cluster_address="$ip:$port"

echo "Starting ray head node @ $(hostname)::$cluster_address with CPUs[$cpus] & GPUs [$gpus]"
job="ray start --head --node-ip-address=$ip --port=$port --dashboard-port=$dashboard_port --dashboard-host=0.0.0.0 --temp-dir=/tmp/ray_symlink --num-cpus=$cpus --num-gpus=$gpus"
echo $job
$job

check="ray status --address $cluster_address"
while ! $check
do
    sleep 3
done
