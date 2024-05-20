#!/bin/bash

OUTDIR='../benchmarks'
PRETRAINED="../models/new/variable_object_size_fourier_filter_125nm_dataset"
CLUSTER='slurm'
TIMELIMIT='24:00:00'  #hh:mm:ss
APPTAINER="--apptainer ../develop_TF_CUDA_12_3.sif"
JOB="benchmark.py --timelimit $TIMELIMIT --partition abc_a100 --mem=125GB --cpus 4 --gpus 1"

TRAINED_MODELS=(
  "vit-S16 vit/vit-15-YuMB_lambda510-S16"
  "vit-B16 vit/vit-15-YuMB_lambda510-B16"
  "vit-S32 vit/vit-15-YuMB_lambda510-S32"
  "vit-B32 vit/vit-15-YuMB_lambda510-B32"
  "vit-L32 vit/vit-15-YuMB_lambda510-L32"
  "baseline-T baseline/baseline-15-YuMB_lambda510-T"
  "baseline-S baseline/baseline-15-YuMB_lambda510-S"
  "baseline-B baseline/baseline-15-YuMB_lambda510-B"
  "baseline-L baseline/baseline-15-YuMB_lambda510-L"
  "opticalnet-S3216 scaling/opticalnet-15-YuMB_lambda510-S3216"
  "opticalnet-B3216 scaling/opticalnet-15-YuMB_lambda510-B3216"
  "opticalnet-M3216 scaling/opticalnet-15-YuMB_lambda510-M3216"
  "opticalnet-L3216 scaling/opticalnet-15-YuMB_lambda510-L3216"
  "opticalnet-H3216 scaling/opticalnet-15-YuMB_lambda510-H3216"
)

for S in `seq 1 ${#TRAINED_MODELS[@]}`
do
    set -- ${TRAINED_MODELS[$S-1]}
    M=$1
    P=$2

    echo Profile $M
    echo python manager.py $CLUSTER $APPTAINER $JOB \
    --task \""profile_models ../ --outdir ${OUTDIR} --model_codename ${M} --model_predictions ${PRETRAINED}/${P}"\" \
    --taskname profile_models \
    --name ${OUTDIR}/${M}

    echo '----------------'
    echo
done

