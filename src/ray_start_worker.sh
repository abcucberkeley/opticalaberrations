#!/bin/bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

while getopts ":i:p:c:g:" option;do
    case "${option}" in
    i) i=${OPTARG}
       ip=$i
       echo ip=$ip
    ;;
    p) p=${OPTARG}
       port=$p
       echo port=$port
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

source ~/.bashrc
conda activate ray
echo "Activated conda environment: ray"

cluster_address="$ip:$port"


echo "Starting ray worker @ $(hostname) with CPUs[$cpus] & GPUs [$gpus] => $cluster_address"
job="ray start --address=$cluster_address --num-cpus=$cpus --num-gpus=$gpus"
echo $job
$job


check="ray status --address  $cluster_address"
while ! $check
do
    sleep 3
done


sleep infinity