#!/bin/bash

PSF_TYPE='confocal'
DEPTH=1.0
xVOXEL=.15
yVOXEL=.15
zVOXEL=.6
MAXAMP=1
SHAPE=64
MODES=60
BATCH=512
LAMBDA=.920
DATA="/clusterfs/nvme/thayer/dataset/${PSF_TYPE}/train/x150-y150-z600/"

#### Single-node
for OPT in adamw adam
do
  for LR in .005 .0005
  do
    python manager.py slurm train.py --partition abc_a100 --mem '500GB' --gpus 4 --cpus 16 \
    --task "--network opticaltransformer --opt $OPT --lr $LR --patch_size '32-16-8-8' --max_amplitude $MAXAMP --batch_size $BATCH --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --modes $MODES --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
    --taskname p32-p16-p8x2 \
    --name new/$PSF_TYPE/opticaltransformer/$OPT/$LR
  done
done

#### Multi-node
#python multinode_manager.py train.py --partition abc_a100 --mem '500GB' --nodes 4 --gpus 4 --cpus 16 \
#--task "--network opticaltransformer --multinode --opt Adamw --patch_size '32-16-8-8' --max_amplitude $MAXAMP --batch_size $BATCH --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --modes $MODES --psf_type $PSF_TYPE --wavelength $LAMBDA  --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
#--taskname p32-p16-p8x2 \
#--name new/$PSF_TYPE/opticaltransformer

#python multinode_manager.py train.py --partition abc --constraint 'titan' --mem '500GB' --nodes 3 --gpus 4 --cpus 20 \
#--task "--network opticalresnet --multinode --mul --batch_size $BATCH --max_amplitude $MAXAMP --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --modes $MODES --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
#--taskname multikernel \
#--name new/$PSF_TYPE/opticalresnet


### Optimal settings
## Widefield
#--task "--network opticaltransformer --multinode --opt Adamw --patch_size '32-16-8-8' --max_amplitude $MAXAMP --batch_size $BATCH --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --modes $MODES --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
#--taskname p32-p16-p8x2 \
#--task "--network opticalresnet --multinode --mul --batch_size $BATCH --max_amplitude $MAXAMP --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --modes $MODES --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
#--taskname multikernel \
