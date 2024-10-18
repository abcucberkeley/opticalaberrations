#!/bin/bash
#
# Run a Ray cluster job on the Janelia LSF cluster.
# https://github.com/JaneliaSciComp/ray-janelia/blob/main/ray-janelia.sh
# Adapted from https://github.com/IBMSpectrumComputing/ray-integration
#
# Parameters:
# -w : user command to run on the cluster (if not specified, the cluster will be left running)
#
# Example usage:
#
# bsub -o std%J.out -e std%J.out -n 20 -R "span[ptile=4]" bash -i $PWD/ray-janelia.sh \
#   -w "python /path/to/job.py --option" -n "ray-python"
#
# This will allocate 20 slots on the cluster, divide them into 20/4=5 nodes, and run the job.py

# Timeout (in seconds) whenever we're waiting on the cluster to converge to a new state
TIMEOUT_SEC=180

# How often to check cluster state
SLEEP_DELAY_SEC=5

# conda create -n ray -c conda-forge python=3.9 "ray-default" "ray-core" "ray-dashboard" -y
eval "$(conda shell.bash hook)"
conda activate ray
echo "Activated conda environment: ray"


# Shut down the cluster by killing the job id
function shutdown_cluster()
{
    echo "Shutting down the cluster"
    bkill $LSB_JOBID
}

# Wait for the cluster at address $1 to have $2 nodes available, with timeout.
function wait_for_nodes()
{
    local _address=$1
    local _num_nodes=$2
    start_time="$(date -u +%s)"
    status_cmd="ray status --address $_address"
    echo "Waiting for $_num_nodes nodes to be ready on cluster $_address"

    while true; do

        # from https://stackoverflow.com/questions/12321469/retry-a-bash-command-with-timeout
        current_time="$(date -u +%s)"
        elapsed_seconds=$(($current_time-$start_time))
        if [ $elapsed_seconds -gt $TIMEOUT_SEC ]; then
            echo "Cluster timeout after $TIMEOUT_SEC seconds"
            shutdown_cluster
            exit 1
        fi

        STATUS_OUTPUT=$($status_cmd)
        STATUS_RC=$?

        if [ $STATUS_RC -ne 0 ]; then
            echo "Cluster status command failed with exit code $STATUS_RC"
            shutdown_cluster
            exit 1
        fi

        num_ready=$(echo "$STATUS_OUTPUT" | awk '/Healthy:/{ f = 1; next } /Pending:/{ f = 0 } f' | wc -l)
        if [ $_num_nodes -eq $num_ready ]; then
            echo "Cluster is ready with $num_ready nodes"
            return 0
        else
            echo "$num_ready cluster nodes are ready (waiting for $_num_nodes)"
        fi

        sleep $SLEEP_DELAY_SEC
    done
}

# Find an available network port to use
function get_free_port()
{
    CHECK="do while"
    while [[ ! -z $CHECK ]]; do
        port=$(( ( RANDOM % 40000 ) + 19999 ))
        CHECK=$(netstat -a | grep $port)
    done
    echo $port
}

while getopts ":w:" option;do
    case "${option}" in
    w) w=${OPTARG}
        workload=$w
    ;;
    *) echo "Did not supply the correct arguments"
    ;;
    esac
done

hosts=()
for host in `cat $LSB_DJOB_HOSTFILE | uniq`
do
    echo "Adding host: $host"
    hosts+=($host)
done

echo
port=$(get_free_port)
echo "Head node will use port: " $port
export port
dashboard_port=$(get_free_port)
echo "Dashboard will use port: " $dashboard_port
echo

IFS=' ' read -r -a array <<< "$LSB_MCPU_HOSTS"
declare -A cpus_for_node
i=0
len=${#array[@]}
while [ $i -lt $len ]
do
    key=${array[$i]}
    value=${array[$i+1]}
    cpus_for_node[$key]+=$value
    i=$((i=i+2))
done
echo

if [ -z "$CUDA_VISIBLE_DEVICES" ]
then
    num_gpu_for_head=0
else
    num_gpu_for_head=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," "{print NF}")
fi
num_gpu_for_worker=0

export head_node=${hosts[0]}
cluster_address="$head_node:$port"
client_server_port=10001

echo "Starting Ray head node on $head_node"

if [ -z $object_store_mem ]
then
    echo "  Using default object store mem of 4GB. Make sure your cluster has more than 4GB of memory."
    object_store_mem=4000000000
fi

echo "  The object store memory: $object_store_mem bytes"

num_cpu_for_head=${cpus_for_node[$head_node]}
command_launch="blaunch -z $head_node ray start --head --port $port --ray-client-server-port $client_server_port --dashboard-host 0.0.0.0 --dashboard-port $dashboard_port --min-worker-port 18999 --max-worker-port 19999 --num-cpus $num_cpu_for_head --num-gpus $num_gpu_for_head --object-store-memory $object_store_mem"
echo $command_launch
$command_launch &

# Wait for the head node to start up
wait_for_nodes "$cluster_address" 1

echo "The dashboard is now available at http://$head_node:$dashboard_port"

workers=("${hosts[@]:1}")

echo "Starting workers: ${workers[*]}"
#run ray on worker nodes and connect to head
for host in "${workers[@]}"
do
    echo "Starting worker on $host using master node $head_node"
    num_cpu=${cpus_for_node[$host]}
    command_for_worker="blaunch -z $host ray start --address $cluster_address --num-cpus $num_cpu --num-gpus $num_gpu_for_worker --object-store-memory $object_store_mem"
    echo $command_for_worker
    $command_for_worker &
done

# Wait for the head node to start up
num_nodes=${#hosts[@]}
wait_for_nodes "$cluster_address" "$num_nodes"

# Run the user command if specified
if [ -n "$workload" ]; then

    echo "Running user workload"
    echo $workload
    $workload

    retVal=$?
    if [ $retVal -ne 0 ]; then
        echo "Error: user command exited with code $retVal"
        echo "Cluster will keep running to allow debugging"
        exit $retVal
    else
        echo "Done"
        shutdown_cluster
    fi

else
    echo "Ray cluster with $num_nodes nodes is now running at ray://$head_node:$client_server_port with a dashboard at http://$head_node:$dashboard_port/"
fi