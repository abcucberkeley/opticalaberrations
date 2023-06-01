#!/bin/bash

xVOXEL=.108
yVOXEL=.108
zVOXEL=.200
LAMBDA=.510
SHAPE=64
PSF_TYPE='../lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat'
DATA="/clusterfs/nvme/thayer/dataset/eval_dataset/test/x108-y108-z200/i$SHAPE/z15"

ROTATIONS='--digital_rotations'
ITERS=10
MAX=10000
PRETRAINED="../pretrained_models/lattice_yumb_x108um_y108um_z200um/"

for EVALSIGN in signed #positive_only
do
  for MODES in 15 #28 45
  do
      MODEL="$PRETRAINED/opticalnet-$MODES"

       BATCH=128
#      python manager.py slurm test.py --partition abc --constraint 'titan' --mem '125GB' --cpus 5 --gpus 1 \
#      --task "$MODEL.h5 --eval_sign $EVALSIGN --digital_rotations --batch_size $BATCH random" \
#      --taskname random \
#      --name $MODEL/$EVALSIGN/samples

#      python manager.py slurm test.py --partition abc --constraint 'titan' --mem '125GB' --cpus 5 --gpus 1 \
#      --task "$MODEL.h5 --eval_sign $EVALSIGN --digital_rotations --batch_size $BATCH modes" \
#      --taskname evalmodes \
#      --name $MODEL/$EVALSIGN/evalmodes/'num_objs_1'
#
#      python manager.py slurm test.py --partition abc --constraint 'titan' --mem '125GB' --cpus 5 --gpus 1 \
#      --task "$MODEL.h5 --eval_sign $EVALSIGN --num_objs 5 --n_samples 5 --digital_rotations --batch_size $BATCH modes" \
#      --taskname evalmodes \
#      --name $MODEL/$EVALSIGN/evalmodes/'num_objs_5'

      BATCH=3500
      for NA in 1.
      do
        python manager.py slurm test.py --partition abc_a100 --mem '500GB' --cpus 16 --gpus 4 \
        --task "$MODEL.h5 --datadir $DATA --na $NA --batch_size $BATCH --niter 1 --n_samples $MAX --eval_sign $EVALSIGN $ROTATIONS snrheatmap" \
        --taskname $NA \
        --name $MODEL/$EVALSIGN/snrheatmaps

        python manager.py slurm test.py --partition abc_a100 --mem '500GB' --cpus 16 --gpus 4 \
        --task "$MODEL.h5 --datadir $DATA --na $NA --batch_size $BATCH --niter 1 --n_samples $MAX --eval_sign $EVALSIGN $ROTATIONS densityheatmap" \
        --taskname $NA \
        --name $MODEL/$EVALSIGN/densityheatmaps

        python manager.py slurm test.py --partition abc_a100 --mem '500GB' --cpus 16 --gpus 4 \
        --task "$MODEL.h5 --datadir $DATA --na $NA --batch_size $BATCH --niter $ITERS --n_samples $MAX --eval_sign $EVALSIGN $ROTATIONS iterheatmap" \
        --taskname $NA \
        --name $MODEL/$EVALSIGN/iterheatmaps
      done
  done
done
