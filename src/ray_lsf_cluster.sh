#!/bin/bash

############################## CHECK ARGS

function getfreeport()
{
    CHECK="do while"
    while [[ ! -z $CHECK ]]; do
        port=$(( ( RANDOM % 40000 )  + 20000 ))
        CHECK=$(netstat -a | grep $port)
    done
    echo $port
}

function usage() {
    echo "Usage: $0 [-c number of cpus per node <int>] [-g number of gpus per node <int>] [-p python command <string>]"
    exit 1
}

while getopts ":c:g:p:" opt;do
    case "${opt}" in
      c) CPUS=${OPTARG};;
      g) GPUS=${OPTARG};;
      p) PYTHON_COMMAND=${OPTARG};;
      ?) usage;;
    esac
done
shift $((OPTIND-1))

if [ -z "${c}" ] || [ -z "${g}" ] || [ -z "${p}" ]; then
    usage
fi

############################## SETUP PORTS

RAY_PORT=6379
if [[ ! -z $(netstat -a | grep $RAY_PORT) ]]; then
  RAY_PORT=$(getfreeport)
fi
echo "Head node will use port: $RAY_PORT"
export RAY_PORT

HEAD_ADDRESS="${HEAD_NODE}:${RAY_PORT}"
echo "Head address: $HEAD_ADDRESS"
HEAD_ADDRESS RAY_PORT

RAY_DASHBOARD_PORT=8265
if [[ ! -z $(netstat -a | grep $RAY_PORT) ]]; then
  RAY_DASHBOARD_PORT=$(getfreeport)
fi
echo "Dashboard will use port: $RAY_DASHBOARD_PORT"
export RAY_DASHBOARD_PORT


############################## FIND NODES/HOSTS

echo "LSB_MCPU_HOSTS=$LSB_MCPU_HOSTS"
echo "---- LSB_AFFINITY_HOSTFILE=$LSB_AFFINITY_HOSTFILE"
cat $LSB_AFFINITY_HOSTFILE
echo "---- End of LSB_AFFINITY_HOSTFILE"
echo "---- LSB_DJOB_HOSTFILE=$LSB_DJOB_HOSTFILE"
cat $LSB_DJOB_HOSTFILE
echo "---- End of LSB_DJOB_HOSTFILE"

export RAY_TMPDIR="/tmp/ray-$USER"
echo "RAY_TMPDIR=$RAY_TMPDIR"
mkdir -p $RAY_TMPDIR

hosts=()
for host in `cat $LSB_DJOB_HOSTFILE | uniq`
do
        echo "Adding host: $host"
        hosts+=($host)
done

HEAD_NODE=${hosts[0]}
WORKERS=("${hosts[@]:1}")
export HEAD_NODE

############################## START HEAD NODE

echo "Starting ray head node on: $HEAD_NODE"

blaunch -z $HEAD_NODE ray start --head --port $RAY_PORT --dashboard-port $RAY_DASHBOARD_PORT --num-cpus $CPUS --num-gpus $GPUS &

sleep 20

while ! ray status --address $HEAD_ADDRESS
do
    sleep 3
done


############################## ADD WORKER NODES

echo "Adding the workers to head node: ${WORKERS[*]}"

for host in "${WORKERS[@]}"
do
    echo "starting worker on: $host and using master node: $HEAD_NODE"

    sleep 10
    blaunch -z $host ray start --address $HEAD_ADDRESS --num-cpus $CPUS --num-gpus $GPUS &

    sleep 10
    while ! blaunch -z $host ray status --address $HEAD_ADDRESS
    do
        sleep 3
    done
done


############################## RUN WORKLOAD

$PYTHON_COMMAND


############################## STOP WORKERS

if [ $? != 0 ]; then
    echo "Failure: $?"
    exit $?
else
    echo "Done"
    echo "Shutting down the Job"
    bkill $LSB_JOBID
fi