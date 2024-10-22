#!/bin/bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

echo "starting ray worker node"
ray start --address $1

command_check_up="ray status --address $1"
while ! $command_check_up
do
    sleep 3
done


sleep infinity