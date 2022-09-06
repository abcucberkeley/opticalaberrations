#!/bin/bash

PSF_TYPE='confocal'
DEPTH=1.0
xVOXEL=.15
yVOXEL=.15
zVOXEL=.6
LAMBDA=.920
MAXAMP=1
mSNR=1
xSNR=100
SHAPE=64
MODES=60
CPUS=-1
GPUS=-1
BATCH=512
DATA="/clusterfs/nvme/thayer/dataset/${PSF_TYPE}/train/x150-y150-z600/"

for OPT in adamw adam
do
  for LR in .005 .0005
  do
    nohup python manager.py default train.py --flags \
    "--network opticaltransformer --opt Adamw --patch_size '32-16-8-8' --max_amplitude $MAXAMP --batch_size $BATCH --cpu_workers $CPUS --gpu_workers $GPUS --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --modes $MODES --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
    --name new/embedding/opticaltransformer/$OPT/$LR &
  done
done