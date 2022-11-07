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
DIFFICULTY='hard'
PHASE='--no_phase'
DATASET='yumb'
DATA="/clusterfs/nvme/thayer/dataset/$DATASET/$DIFFICULTY/train/x108-y108-z200/"

if [ "$DIFFICULTY" = "easy" ];then
  MODES=15
  MAXAMP=.25
else
  MODES=60
  MAXAMP=.5
fi

python manager.py slurm train.py --partition abc_a100 --mem '500GB' --gpus 4 --cpus 16 \
--task "--network opticaltransformer $PHASE --opt Adamw --patch_size '32-16-8-8' --max_amplitude $MAXAMP --batch_size $BATCH --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --modes $MODES --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
--taskname p32-p16-p8x2 \
--name new/$DATASET/$DIFFICULTY/opticaltransformer

BATCH=2048
python multinode_manager.py train.py --partition abc_a100 --mem '500GB' --nodes 2 --gpus 4 --cpus 16 \
--task "--network opticaltransformer --multinode $PHASE --opt Adamw --patch_size '32-16-8-8' --max_amplitude $MAXAMP --batch_size $BATCH --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --modes $MODES --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
--taskname p32-p16-p8x2 \
--name new/$DATASET/$DIFFICULTY/opticaltransformer_multinode

#python manager.py slurm train.py --partition dgx --mem '1950GB' --gpus 8 --cpus 128 \
#python manager.py slurm train.py --partition abc_a100 --mem '500GB' --gpus 4 --cpus 16 \
#python multinode_manager.py train.py --partition abc --constraint 'titan' --mem '500GB' --nodes 3 --gpus 4 --cpus 20 \
#python multinode_manager.py train.py --partition abc_a100 --mem '500GB' --nodes 3 --gpus 4 --cpus 16 \

### Optimal settings
## Widefield
#--task "--network opticaltransformer --multinode --opt Adamw --patch_size '32-16-8-8' --max_amplitude $MAXAMP --batch_size $BATCH --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --modes $MODES --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
#--taskname p32-p16-p8x2 \
#--task "--network opticalresnet --multinode --mul --batch_size $BATCH --max_amplitude $MAXAMP --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --modes $MODES --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
#--taskname multikernel \