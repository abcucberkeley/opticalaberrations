#!/bin/bash

PSF_TYPE='../lattice/HexRect_NAlattice0.25_NAAnnulusMax0.60_NAsigma0.08.mat'
DEPTH=1.0
xVOXEL=.108
yVOXEL=.108
zVOXEL=.268
SHAPE=64
BATCH=1024
LAMBDA=.510
DIFFICULTY='hard'
PHASE='--no_phase'
DATASET='lattice_objects'
DATA="/clusterfs/nvme/thayer/dataset/$DATASET/$DIFFICULTY/train/x108-y108-z268/"

if [ "$DIFFICULTY" = "easy" ];then
  MODES=15
  MAXAMP=.2
else
  MODES=60
  MAXAMP=.5
fi

#python manager.py slurm train.py --partition dgx --mem '1950GB' --gpus 8 --cpus 128 \
#python multinode_manager.py train.py --partition abc --constraint 'titan' --mem '500GB' --nodes 3 --gpus 4 --cpus 20 \
python multinode_manager.py train.py --partition abc_a100 --mem '500GB' --nodes 4 --gpus 4 --cpus 16 \
--task "--network opticaltransformer $PHASE --opt Adamw --patch_size '32-16-8-8' --max_amplitude $MAXAMP --batch_size $BATCH --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --modes $MODES --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
--taskname p32-p16-p8x2 \
--name new/$DATASET/$DIFFICULTY/opticaltransformer


### Optimal settings
## Widefield
#--task "--network opticaltransformer --multinode --opt Adamw --patch_size '32-16-8-8' --max_amplitude $MAXAMP --batch_size $BATCH --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --modes $MODES --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
#--taskname p32-p16-p8x2 \
#--task "--network opticalresnet --multinode --mul --batch_size $BATCH --max_amplitude $MAXAMP --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --modes $MODES --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
#--taskname multikernel \
