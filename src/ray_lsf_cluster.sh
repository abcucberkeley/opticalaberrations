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

hosts=()
for host in `cat $LSB_DJOB_HOSTFILE | uniq`
do
        echo "Adding host: $host"
        hosts+=($host)
done
echo "The host list is: ${hosts[@]}"

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
head_node_ip=$(getent hosts $head_node | awk '{ print $1 }')
cluster_address="$head_node_ip:$port"

export head_node
export head_node_ip
export cluster_address

head_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," "{print NF}")
head_cpus=${associative[$head_node]}

############################## START HEAD NODE

job="blaunch -z $head_node bash ray_start_cluster.sh -i $head_node_ip -p $port -c $head_cpus -g $head_gpus"
echo $job
$job &

rpids=$(blaunch -z $head_node pgrep -u $USER ray)
echo "Ray head node PID:"
echo $rpids

############################## ADD WORKER NODES

workers=("${hosts[@]:1}")
for host in "${workers[@]}"
do
    num_cpu=${associative[$host]}
    worker_job="blaunch -z $host bash ray_start_worker.sh -i $head_node_ip -p $port -c $head_cpus -g $head_gpus"
    echo $worker_job
    $worker_job &
done

############################## RUN WORKLOAD

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
