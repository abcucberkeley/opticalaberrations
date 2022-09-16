#!/bin/bash

PSF_TYPE='/clusterfs/nvme/thayer/dataset/lattice/simulations/NAlattice0.25/HexRect/NAAnnulusMax0.60/NAsigma0.08/decon_simulation/PSF_OTF_simulation.mat'
DEPTH=1.0
xVOXEL=.108
yVOXEL=.108
zVOXEL=.268
MAXAMP=1
SHAPE=64
MODES=60
BATCH=512
LAMBDA=.510
DATA="/clusterfs/nvme/thayer/dataset/lattice_multipoints/train/x108-y108-z268/"

python multinode_manager.py train.py --partition abc --constraint 'titan' --mem '500GB' --nodes 3 --gpus 4 --cpus 20 \
--task "--network opticaltransformer --no_phase --multinode --opt Adamw --patch_size '24-12-6-6' --roi '6-48-48' --max_amplitude $MAXAMP --batch_size $BATCH --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --modes $MODES --psf_type $PSF_TYPE --wavelength $LAMBDA  --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
--taskname p24-p12-p6x2 \
--name new/multipoints/lattice/opticaltransformer-roi-nophase

python multinode_manager.py train.py --partition abc_a100 --mem '500GB' --nodes 4 --gpus 4 --cpus 16 \
--task "--network opticalresnet --no_phase --multinode --mul --batch_size $BATCH --max_amplitude $MAXAMP --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --modes $MODES --psf_type $P    SF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
--taskname multikernel \
--name new/multipoints/lattice/opticalresnet-nophase

python manager.py slurm train.py --partition dgx --mem '1950GB' --gpus 8 --cpus 128 \
--task "--network opticaltransformer -opt Adamw --patch_size '32-16-8-8' --max_amplitude $MAXAMP --batch_size $BATCH --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --modes $MODES --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
--taskname p32-p16-p8x2 \
--task "--network opticaltransformer --opt Adamw --patch_size '24-12-6-6' --roi '6-48-48' --max_amplitude $MAXAMP --batch_size $BATCH --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --modes $MODES --psf_type $PSF_TYPE --wavelength $LAMBDA  --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
--taskname p24-p12-p6x2 \
--task "--network opticalresnet --mul --batch_size $BATCH --max_amplitude $MAXAMP --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --modes $MODES --psf_type $P    SF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
--taskname multikernel \
--name new/multipoints/lattice/


### Optimal settings
## Widefield
#--task "--network opticaltransformer --multinode --opt Adamw --patch_size '32-16-8-8' --max_amplitude $MAXAMP --batch_size $BATCH --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --modes $MODES --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
#--taskname p32-p16-p8x2 \
#--task "--network opticalresnet --multinode --mul --batch_size $BATCH --max_amplitude $MAXAMP --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --modes $MODES --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
#--taskname multikernel \
