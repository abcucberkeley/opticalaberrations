#!/bin/bash

#PSF_TYPE='../lattice/HexRect_NAlattice0.25_NAAnnulusMax0.60_NAsigma0.08.mat'
PSF_TYPE='../lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat'
DEPTH=1.0
xVOXEL=.108
yVOXEL=.108
zVOXEL=.200
SHAPE=64
LAMBDA=.510
PHASE='--no_phase'
DATASET='yumb_new_embeddings'
MAXAMP=.5
DATA='/clusterfs/nvme/thayer/dataset/$DATASET/train/x108-y108-z200/'

BATCH=1024
for EMBEDDING in principle_planes rotary_slices spatial_quadrants
do
  MODES=15
  python python manager.py slurm train.py --partition abc --constraint 'titan' --mem '500GB' --gpus 4 --cpus 20 \
  --task "--network opticalnet $PHASE --embedding $EMBEDDING --patch_size '32-16-8-8' --max_amplitude $MAXAMP --batch_size $BATCH --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --modes $MODES --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
  --taskname $EMBEDDING \
  --name new/$DATASET/z$MODES/opticalnet

  MODES=35
  python python manager.py slurm train.py --partition abc_a100 --mem '500GB' --gpus 4 --cpus 16 \
  --task "--network opticalnet $PHASE --embedding $EMBEDDING --patch_size '32-16-8-8' --max_amplitude $MAXAMP --batch_size $BATCH --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --modes $MODES --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
  --taskname $EMBEDDING \
  --name new/$DATASET/z$MODES/opticalnet
done


MODES=55
python manager.py slurm train.py --partition dgx --mem '1950GB' --gpus 8 --cpus 128 \
--task "--network opticalnet $PHASE --embedding principle_planes --patch_size '32-16-8-8' --modes $MODES --max_amplitude $MAXAMP --batch_size $BATCH --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
--taskname principle_planes \
--task "--network opticalnet $PHASE --embedding spatial_quadrants --patch_size '32-16-8-8' --modes $MODES --max_amplitude $MAXAMP --batch_size $BATCH --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
--taskname spatial_quadrants \
--task "--network opticalnet $PHASE --embedding rotary_slices --patch_size '32-16-8-8' --modes $MODES --max_amplitude $MAXAMP --batch_size $BATCH --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
--taskname rotary_slices \
--name new/$DATASET/z$MODES/opticalnet


#python multinode_manager.py train.py --partition abc_a100 --mem '500GB' --nodes 2 --gpus 4 --cpus 16 \
#--task "--network opticalnet --multinode $PHASE --embedding $EMBEDDING --patch_size '32-16-8-8' --max_amplitude $MAXAMP --batch_size $BATCH --dataset $DATA/i$SHAPE --input_shape $SHAPE --depth_scalar $DEPTH --modes $MODES --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
#--taskname opticalnet \
#--name new/$DATASET/z$MODES/opticalnet


#python manager.py slurm train.py --partition dgx --mem '1950GB' --gpus 8 --cpus 128 \
#python manager.py slurm train.py --partition abc_a100 --mem '500GB' --gpus 4 --cpus 16 \
#python multinode_manager.py train.py --partition abc --constraint 'titan' --mem '500GB' --nodes 3 --gpus 4 --cpus 20 \
#python multinode_manager.py train.py --partition abc_a100 --mem '500GB' --nodes 3 --gpus 4 --cpus 16 \