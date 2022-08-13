#!/bin/bash

HANDLER=slurm
ENV=~/anaconda3/envs/deep/bin/python

OUTDIR='/clusterfs/nvme/thayer/allencell/aics/label-free-imaging-collection/dataset/'
KERNELS='/clusterfs/nvme/thayer/allencell/aics/label-free-imaging-collection/kernels/'
DATASET='/clusterfs/nvme/thayer/allencell/aics/label-free-imaging-collection/channels/golgi_apparatus'

for S in $(find "${DATASET}" -type f -name "*.tif")
do
    echo $S
    while [ $(squeue -u thayeralshaabi -h -t pending -r | wc -l) -gt 500 ]
    do
      sleep 10s
    done

    j="${ENV} allencell.py dataset"
    j="${j} --sample ${S}"
    j="${j} --kernels ${KERNELS}"
    j="${j} --savedir ${OUTDIR}"

    task="/usr/bin/sbatch"
    task="${task} --qos=abc_normal"
    task="${task} --partition=abc"
    task="${task} --cpus-per-task=24"
    task="${task} --mem=500"
    task="${task} --job-name=psnr#${S}"
    task="${task} --time=24:00:00"
    task="${task} --export=ALL"
    task="${task} --wrap=\"${j}\""
    echo $task | bash

    echo "ABC : R[$(squeue -u thayeralshaabi -h -t running -r -p abc | wc -l)], P[$(squeue -u thayeralshaabi -h -t pending -r -p abc | wc -l)]"
done
