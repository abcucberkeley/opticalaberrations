#!/bin/bash
#--partition dgx --mem '1950GB' --gpus 8 --cpus 128 \
#--partition abc_a100 --mem '500GB' --nodes 1 --gpus 4 --cpus 16 \
#--partition abc --constraint 'titan' --mem '500GB' --nodes 3 --gpus 4 --cpus 20 \


#PSF_TYPE='../lattice/HexRect_NAlattice0.25_NAAnnulusMax0.60_NAsigma0.08.mat'
PSF_TYPE='../lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat'
DEPTH=1.0
xVOXEL=.108
yVOXEL=.108
zVOXEL=.200
SHAPE=64
LAMBDA=.510
MAXAMP=.5
NO_PHASE='--no_phase'
DEFOCUS='--lls_defocus'
EMB="spatial_planes"
DATASET='lls_defocus_embeddings'
DATA="/clusterfs/nvme/thayer/dataset/$DATASET/train/x108-y108-z200/"
BATCH=2048


for MODES in 15 28
do
  python multinode_manager.py train.py --partition abc_a100 --mem '500GB' --nodes 2 --gpus 4 --cpus 16 \
  --task "--network opticalnet $DEFOCUS --embedding $EMB --patch_size '32-16-8-8' --modes $MODES --max_amplitude $MAXAMP --batch_size $BATCH --dataset $DATA/i$SHAPE/z$MODES --input_shape $SHAPE --depth_scalar $DEPTH --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL --multinode" \
  --taskname opticalnet \
  --name new/$DATASET/opticalnet-z$MODES
done


MODES=45
python manager.py slurm train.py --partition dgx --mem '1950GB' --gpus 8 --cpus 128 \
--task "--network opticalnet $DEFOCUS --embedding $EMB --patch_size '32-16-8-8' --modes $MODES --max_amplitude $MAXAMP --batch_size $BATCH --dataset $DATA/i$SHAPE/z$MODES --input_shape $SHAPE --depth_scalar $DEPTH --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL" \
--taskname opticalnet \
--name new/$DATASET/opticalnet-z$MODES