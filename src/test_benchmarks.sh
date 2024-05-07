#!/bin/bash

DX=97
DY=97
DZ=200
SHAPE=64
MODES=15
ITERS=5
MAX=10000
OUTDIR='../benchmarks'
DATASET="97nm_dataset_extended"
EVALSIGN="signed"
NA=1.0
TIMELIMIT='24:00:00'  #hh:mm:ss
APPTAINER="--apptainer ../develop_TF_CUDA_12_3.sif"
CLUSTER='slurm'

DENOISE=false
DENOISER='../pretrained_models/denoise/20231107_simulatedBeads_v3_32_64_64/'

for (( i=1; i<=$ITERS; i++ ))
do
    JOB="benchmark.py --timelimit $TIMELIMIT --dependency singleton --partition abc --constraint 'titan' --mem '256GB' --cpus 10 --gpus 2"
    CONFIG="/clusterfs/nvme/thayer/dataset/$DATASET/test/YuMB_lambda510/z$DZ-y$DY-x$DX/z$SHAPE-y$SHAPE-x$SHAPE/z$MODES"
    CONFIG="${CONFIG} --n_samples ${MAX}"
    CONFIG="${CONFIG} --niter ${i}"
    CONFIG="${CONFIG} --num_beads 1"
    CONFIG="${CONFIG} --eval_sign ${EVALSIGN}"

    if $DENOISE; then
      CONFIG="${CONFIG} --denoiser ${DENOISER}"
    fi

    python manager.py $CLUSTER $APPTAINER $JOB \
    --task "phasenet_heatmap ${CONFIG} --simulate_psf_only" \
    --taskname $NA \
    --name ${OUTDIR}/${DATASET}/phasenet/${EVALSIGN}/psf

    python manager.py $CLUSTER $APPTAINER $JOB \
    --task "phasenet_heatmap ${CONFIG}" \
    --taskname $NA \
    --name ${OUTDIR}/${DATASET}/phasenet/${EVALSIGN}/beads-1

    python manager.py $CLUSTER $APPTAINER $JOB \
    --task "phaseretrieval_heatmap ${CONFIG} --simulate_psf_only" \
    --taskname $NA \
    --name ${OUTDIR}/${DATASET}/phaseretrieval/${EVALSIGN}/psf

    python manager.py $CLUSTER $APPTAINER $JOB \
    --task "phaseretrieval_heatmap ${CONFIG}" \
    --taskname $NA \
    --name ${OUTDIR}/${DATASET}/phaseretrieval/${EVALSIGN}/beads-1

    echo
done
