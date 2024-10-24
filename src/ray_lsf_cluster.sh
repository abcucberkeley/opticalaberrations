#!/bin/bash

# conda create -n ray -c conda-forge python=3.9 "ray-default" "ray-core" "ray-dashboard" -y
source ~/.bashrc
conda activate ray
echo "Activated conda environment: ray"

while getopts ":n:c:g:o:w:" option;do
    case "${option}" in
    n) n=${OPTARG}
       nodes=$n
       echo nodes=$nodes
    ;;
    c) c=${OPTARG}
       cpus=$c
       echo cpus=$cpus
    ;;
    g) g=${OPTARG}
       gpus=$g
       echo gpus=$gpus
    ;;
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

############################## START HEAD NODE

head_node=$(cat $LSB_DJOB_HOSTFILE | uniq)
head_node_ip=$(getent hosts $head_node | awk '{ print $1 }')
cluster_address="$head_node_ip:$port"

export head_node
export head_node_ip
export cluster_address

bash ray_start_cluster.sh -i $head_node_ip -p $port -d $dashboard_port -c $cpus -g $gpus &

############################## ADD WORKER NODES

worker_ids=()
num_workers=$((nodes - 1))
for i in $(seq 1 $num_workers)
do
    echo "Adding worker: ${outdir}/ray_worker_${i}"
    worker="bash ray_start_worker.sh -a $cluster_address -c $cpus -g gpus"
    bsub -cwd "$(pwd)" -q $LSB_QUEUE -J "${outdir}/ray_worker_${i}" -n $cpus -gpu "num=$gpus:mode=shared" -o "${outdir}/ray_worker_${i}.log" $worker

    jid=$(bjobs -J "${outdir}/ray_worker_${i}" | awk '/<none>/ {print $1}')
    while [ -z "$jid" ]
    do
        sleep 2
        jid=$(bjobs -J "${outdir}/ray_worker_${i}" | awk '/<none>/ {print $1}')
    done

    worker_ids+=("jid")
    echo "Running ray_worker_${i} @ ${jid}"
done
############################## RUN WORKLOAD

echo "Running user workload"
echo $workload
$workload

############################## CLEANUP

if [ $? != 0 ]; then
    echo "Failure: $?"
    exit $?
else
    echo "Shutting down the Job"

    for jid in "${worker_ids[@]}"
    do
        bkill $jid
    done

    bkill $LSB_JOBID
fi
