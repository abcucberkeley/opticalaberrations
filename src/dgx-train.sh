#!/bin/bash

DEPTH=1.0
xVOXEL=.15
yVOXEL=.15
zVOXEL=.6
MODES=60
CPUS=-1
GPUS=-1
BATCH=512
SHAPE=64
DATA='/clusterfs/nvme/thayer/dataset/embedding/train/x150-y150-z600/'


for MODEL in baseline widekernel multikernel opticalresnet
do
  nohup python manager.py default train.py --flags \
  "--dataset $DATA/i$SHAPE --batch_size $BATCH --network $MODEL --depth_scalar $DEPTH --modes $MODES --input_shape $SHAPE --cpu_workers $CPUS --gpu_workers $GPUS  --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL " \
  --name new/embedding/$MODEL &
done
