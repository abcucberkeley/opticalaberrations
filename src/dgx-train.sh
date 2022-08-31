#!/bin/bash

DEPTH=1.0
xVOXEL=.15
yVOXEL=.15
zVOXEL=.6
MAXAMP=1
mSNR=1
xSNR=100
SHAPE=64
MODES=60
CPUS=-1
GPUS=-1
BATCH=512
DATA='/clusterfs/nvme/thayer/allencell/aics/label-free-imaging-collection/dataset/train/golgi_apparatus/x150-y150-z600/'


nohup python manager.py default train.py --flags \
"--network opticaltransformer --opt Adamw --patch_size '32-16-8-8' --max_amplitude $MAXAMP --batch_size $BATCH --cpu_workers $CPUS --gpu_workers $GPUS --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --modes $MODES --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
--name new/embedding/opticaltransformer &
