#!/bin/bash
#--partition dgx --mem '1950GB' --gpus 8 --cpus 128 \
#--partition abc_a100 --mem '500GB' --nodes 1 --gpus 4 --cpus 16 \
#--partition abc --constraint 'titan' --mem '500GB' --nodes 3 --gpus 4 --cpus 20 \

DEPTH=1.0
SHAPE=64
MAXAMP=.5
NO_PHASE='--no_phase'
DEFOCUS='--lls_defocus'
DEFOCUS_ONLY='--defocus_only'
EMB="spatial_planes"
DATASET='new_modalities'
BATCH=1024
NETWORK=opticalnet


for PSF in YuMB_lambda510 confocal_lambda510 2photon_lambda920 widefield_lambda510
do
  DATA="/clusterfs/nvme/thayer/dataset/new_modalities/train/$PSF/x108-y108-z200/"
  python multinode_manager.py train.py --partition abc_a100 --mem '500GB' --nodes 1 --gpus 4 --cpus 16 \
  --task "--multinode --network $NETWORK --embedding $EMB --patch_size '32-16-8-8' --modes $MODES --max_amplitude $MAXAMP --batch_size $BATCH --dataset $DATA/i$SHAPE/z$MODES --input_shape $SHAPE --depth_scalar $DEPTH" \
  --taskname $NETWORK \
  --name new/$DATASET/$NETWORK-$MODES-$PSF
done


#python multinode_manager.py train.py --partition abc_a100 --mem '500GB' --nodes 2 --gpus 4 --cpus 16 \
#--task "--multinode --network $NETWORK --embedding $EMB --patch_size '32-16-8-8' --modes $MODES --max_amplitude $MAXAMP --batch_size $BATCH --dataset $DATA/i$SHAPE/z$MODES --input_shape $SHAPE --depth_scalar $DEPTH" \
#--taskname $NETWORK \
#--name new/$DATASET/$NETWORK-$MODES
#
#python manager.py slurm train.py --partition dgx --mem '1950GB' --gpus 8 --cpus 128 \
#--task "--network $NETWORK --embedding $EMB --patch_size '32-16-8-8' --modes $MODES --max_amplitude $MAXAMP --batch_size $BATCH --dataset $DATA/i$SHAPE/z$MODES --input_shape $SHAPE --depth_scalar $DEPTH" \
#--taskname $NETWORK \
#--name new/$DATASET/$NETWORK-$MODES
