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

hosts=()
for host in `cat $LSB_DJOB_HOSTFILE | uniq`
do
        echo "Adding host: $host"
        hosts+=($host)
done

echo "The host list is: ${hosts[@]}"

port=$(getfreeport)
echo "Head node will use port: $port"

export port

dashboard_port=$(getfreeport)
echo "Dashboard will use port: $dashboard_port"

export dashboard_port

# Compute number of cores allocated to hosts
# Format of each line in file $LSB_AFFINITY_HOSTFILE:
#   host_name core_id_list NUMA_node_id_list memory_policy
# core_id_list is comma separeted core IDs. e.g.
#   host1 1,2,3,4,5,6,7
#   host2 0,2,3,4,6,7,8
#   host2 19,21,22,23,24,26,27
#   host2 28,29,37,41,48,49,50
# First, count up number of cores for each line (slot), then sum up for same host.
declare -A associative
while read -a line
do
    host=${line[0]}
    num_cpu=`echo ${line[1]} | tr , ' ' | wc -w`
    ((associative[$host]+=$num_cpu))
done < $LSB_AFFINITY_HOSTFILE
for host in ${!associative[@]}; do
    echo host=$host cores=${associative[$host]}
done

#Assumption only one head node and more than one
#workers will connect to head node

head_node=${hosts[0]}
cluster_address="$head_node:$port"
client_server_port=10001

export head_node
export cluster_address
export client_server_port

echo "Object store memory for the cluster is set to 4GB"

echo "Starting ray head node on: ${hosts[0]}"

if [ -z $object_store_mem ]
then
    echo "using default object store mem of 4GB make sure your cluster has mem greater than 4GB"
    object_store_mem=4000000000
else
    echo "The object store memory in bytes is: $object_store_mem"
fi


if [ -z "$CUDA_VISIBLE_DEVICES" ]
then
    num_gpu_for_head=0
else
    num_gpu_for_head=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," "{print NF}")
fi

num_cpu_for_head=${associative[$head_node]}
# Number of GPUs available for each host is detected "ray start" command
command_launch="blaunch -z ${hosts[0]} ray start --head --port $port --dashboard-port $dashboard_port --temp-dir /tmp/ray_symlink --num-cpus $num_cpu_for_head --num-gpus $num_gpu_for_head --object-store-memory $object_store_mem"

$command_launch &



sleep 10

command_check_up="ray status --address $cluster_address"

while ! $command_check_up
do
    sleep 3
done



workers=("${hosts[@]:1}")

echo "adding the workers to head node: ${workers[*]}"
#run ray on worker nodes and connect to head
for host in "${workers[@]}"
do
    echo "starting worker on: $host and using master node: $head_node"

    sleep 10
    num_cpu=${associative[$host]}
    command_for_worker="blaunch -z $host ray start --temp-dir /tmp/ray_symlink --address $cluster_address --num-cpus $num_cpu --object-store-memory $object_store_mem"


    $command_for_worker &
    sleep 10
    command_check_up_worker="blaunch -z $host ray  status --address $cluster_address"
    while ! $command_check_up_worker
    do
        sleep 3
    done
done

#Run workload
echo "Running user workload"
echo $workload
$workload


if [ $? != 0 ]; then
    echo "Failure: $?"
    exit $?
else
    echo "Done"
    echo "Shutting down the Job"
    bkill $LSB_JOBID
fi
