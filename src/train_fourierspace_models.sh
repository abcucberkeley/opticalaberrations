#!/bin/bash

#PSF_TYPE='../lattice/HexRect_NAlattice0.25_NAAnnulusMax0.60_NAsigma0.08.mat'
PSF_TYPE='../lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat'
DEPTH=1.0
xVOXEL=.108
yVOXEL=.108
zVOXEL=.200
SHAPE=64
BATCH=1024
LAMBDA=.510
PHASE='--no_phase'
DATASET='yumb'
MAXAMP=.5


DIFFICULTY='easy'
MODES=15
DATA="/clusterfs/nvme/thayer/dataset/$DATASET/$DIFFICULTY/train/x108-y108-z200/"

python manager.py slurm train.py --partition abc_a100 --mem '500GB' --gpus 4 --cpus 16 \
--task "--network opticalnet $PHASE --patch_size '32-16-8-8' --max_amplitude $MAXAMP --batch_size $BATCH --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --modes $MODES --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
--taskname opticalnet \
--task "--network otfnet $PHASE --max_amplitude $MAXAMP --batch_size $BATCH --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --modes $MODES --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
--taskname otfnet \
--name new/$DATASET/$DIFFICULTY


DIFFICULTY='hard'
MODES=55
DATA="/clusterfs/nvme/thayer/dataset/$DATASET/$DIFFICULTY/train/x108-y108-z200/"

python manager.py slurm train.py --partition abc_a100 --mem '500GB' --gpus 4 --cpus 16 \
--task "--network opticalnet $PHASE --patch_size '32-16-8-8' --max_amplitude $MAXAMP --batch_size $BATCH --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --modes $MODES --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
--taskname opticalnet \
--task "--network otfnet $PHASE --max_amplitude $MAXAMP --batch_size $BATCH --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --modes $MODES --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
--taskname otfnet \
--name new/$DATASET/$DIFFICULTY


#python manager.py slurm train.py --partition dgx --mem '1950GB' --gpus 8 --cpus 128 \
#python manager.py slurm train.py --partition abc_a100 --mem '500GB' --gpus 4 --cpus 16 \
#python multinode_manager.py train.py --partition abc --constraint 'titan' --mem '500GB' --nodes 3 --gpus 4 --cpus 20 \
#python multinode_manager.py train.py --partition abc_a100 --mem '500GB' --nodes 3 --gpus 4 --cpus 16 \

### Optimal settings
## Widefield
#--task "--network opticalnet --multinode --opt AdamW --patch_size '32-16-8-8' --max_amplitude $MAXAMP --batch_size $BATCH --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --modes $MODES --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
#--taskname p32-p16-p8x2 \
#--task "--network opticalresnet --multinode --mul --batch_size $BATCH --max_amplitude $MAXAMP --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --modes $MODES --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
#--taskname multikernel \
