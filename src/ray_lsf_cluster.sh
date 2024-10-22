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

#port=6379
port=$(getfreeport)
echo "Head node will use port: $port"
export port

dashboard_port=$(getfreeport)
echo "Dashboard will use port: $dashboard_port"
export dashboard_port


redis_password=$(uuidgen)
export redis_password

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

num_nodes=${#hosts[@]}
head_node=${hosts[0]}
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

num_cpu_for_head=${associative[$head_node]}

command_launch="blaunch -z $head_node ray start --head --redis-password $redis_password --port $port --dashboard-host 0.0.0.0 --dashboard-port $dashboard_port --temp-dir /tmp/ray_symlink --num-cpus $num_cpu_for_head --num-gpus $num_gpu_for_head --object-store-memory $object_store_mem --block"
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

    command_for_worker="blaunch -z $host ray start --temp-dir /tmp/ray_symlink --redis-password $redis_password --address $cluster_address --num-cpus $num_cpu --object-store-memory $object_store_mem --block"


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


echo "Ray cluster with $num_nodes nodes is now running at ray://$cluster_address with a dashboard at http://$head_node_ip:$dashboard_port/"


if [ $? != 0 ]; then
    echo "Failure: $?"
    exit $?
else
    echo "Done"
    echo "Shutting down the Job"
    bkill $LSB_JOBID
fi
