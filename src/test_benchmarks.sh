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
APPTAINER="--apptainer ../develop_CUDA_12_3.sif"

DENOISE=false
DENOISER='../pretrained_models/denoise/20231107_simulatedBeads_v3_32_64_64/'

for (( i=1; i<=$ITERS; i++ ))
do
    JOB="benchmark.py --timelimit $TIMELIMIT --dependency singleton --partition abc_a100 --mem '125GB' --cpus 4 --gpus 1"
    CONFIG="/groups/betzig/betziglab/thayer/dataset/$DATASET/test/YuMB_lambda510/z$DZ-y$DY-x$DX/z$SHAPE-y$SHAPE-x$SHAPE/z$MODES"
    CONFIG="${CONFIG} --n_samples ${MAX}"
    CONFIG="${CONFIG} --niter ${i}"
    CONFIG="${CONFIG} --eval_sign ${EVALSIGN}"

    if $DENOISE; then
      CONFIG="${CONFIG} --denoiser ${DENOISER}"
    fi

    for EXP in "psf" "bead-1"
    do
        if [ $EXP = "bead-1" ];then
          CONFIG="${CONFIG} --num_beads 1"
        fi

        python manager.py ${CLUSTER} $APPTAINER $JOB \
        --task "phasenet_heatmap ${CONFIG}" \
        --taskname $NA \
        --name ${OUTDIR}/${DATASET}/phasenet/${EVALSIGN}/${EXP}

        python manager.py ${CLUSTER} $APPTAINER $JOB \
        --task "phaseretrieval_heatmap ${CONFIG}" \
        --taskname $NA \
        --name ${OUTDIR}/${DATASET}/phaseretrieval/${EVALSIGN}/${EXP}

        echo
    done
done
