#!/bin/bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

cluster_address="$1:$2"

echo "Starting ray head node $cluster_address"
# Launch the head node
ray start --head --node-ip-address=$1 --port=$2 --dashboard-host 0.0.0.0 --temp-dir /tmp/ray_symlink

echo "Ray head node PID: $!"

command_check_up="ray status --address $cluster_address"
while ! $command_check_up
do
    sleep 3
done

sleep infinity
