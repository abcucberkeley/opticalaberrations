#!/bin/bash

# conda create -n ray -c conda-forge python=3.9 "ray-default" "ray-core" "ray-dashboard" -y
source ~/.bashrc
conda activate ray
echo "Activated conda environment: ray"

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

while getopts ":a:c:g:" option;do
    case "${option}" in
    a) a=${OPTARG}
       cluster_address=$a
       echo cluster_address=$cluster_address
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

echo "Starting ray worker @ $(hostname) with CPUs[$cpus] & GPUs [$gpus] => $cluster_address"
job="ray start --address=$cluster_address --num-cpus=$cpus --num-gpus=$gpus"
echo $job
$job


check="ray status --address  $cluster_address"
while ! $check
do
    sleep 3
done
