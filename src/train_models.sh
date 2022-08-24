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
BATCH=512
DATA='/clusterfs/nvme/thayer/allencell/aics/label-free-imaging-collection/dataset/train/golgi_apparatus/x150-y150-z600/'


#### Fourier-space models
python multinode_manager.py train.py --partition abc_a100 --nodes 4 --gpus 4 --cpus 16 \
--task "--network opticaltransformer --multinode --opt Adamw --patch_size '32-16-8-8' --max_amplitude $MAXAMP --batch_size $BATCH --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --modes $MODES --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
--taskname p32-p16-p8x2 \
--name new/allencell/opticaltransformer

# python multinode_manager.py train.py --partition abc_a100 --nodes 4 --gpus 4 --cpus 16 \
#--task "--network opticalresnet --multinode --mul --batch_size $BATCH --max_amplitude $MAXAMP --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --modes $MODES --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
#--taskname multikernel \
#--name new/allencell/opticalresnet


#### Real-space models
#python manager.py slurm train.py --partition dgx --mem '1950GB' --gpus 8 --cpus 128 \
#--task "--network phasenet --fixedlr --input_shape $SHAPE --batch_size 32 --dist uniform --max_amplitude .075 --modes $MODES --min_psnr $mSNR --max_psnr $xSNR --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL " \
#--taskname "phasenet-uniform-p075" \
#--task "--network phasenet --fixedlr --input_shape $SHAPE --batch_size 32 --dist uniform --max_amplitude .15 --modes $MODES --min_psnr $mSNR --max_psnr $xSNR --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL " \
#--taskname "phasenet-uniform-p15" \
#--task "--network phasenet --fixedlr --input_shape $SHAPE --batch_size 32 --dist dirichlet --max_amplitude .2 --modes $MODES --min_psnr $mSNR --max_psnr $xSNR --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL " \
#--taskname "phasenet-dirichlet-p2" \
#--task "--network phasenet --fixedlr --input_shape $SHAPE --batch_size 32 --dist mixed --max_amplitude .3 --modes $MODES --min_psnr $mSNR --max_psnr $xSNR --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL " \
#--taskname "phasenet-mixed" \
#--task "--network baseline --input_shape $SHAPE --batch_size 32 --dist mixed --max_amplitude .3 --depth_scalar $DEPTH --modes $MODES --min_psnr $mSNR --max_psnr $xSNR --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL " \
#--taskname "baseline-mixed" \
#--name new/realspace
