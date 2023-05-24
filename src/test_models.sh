#!/bin/bash

xVOXEL=.108
yVOXEL=.108
zVOXEL=.200
LAMBDA=.510
SHAPE=64
PSF_TYPE='../lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat'
DATA="../../dataset/spatial_planes_embeddings/test/x108-y108-z200/i$SHAPE/z15"
BATCH=512
ROTATIONS='--digital_rotations'
ITERS=1
MAX=10000
PRETRAINED="../pretrained_models/lattice_yumb_x108um_y108um_z200um/"

for EVALSIGN in positive_only signed
do
  for MODES in 15 28 45
  do
      MODEL="$PRETRAINED/opticalnet-$MODES"

      python manager.py slurm test.py --partition abc --constraint 'titan' --mem '125GB' --cpus 5 --gpus 1 \
      --task "$MODEL.h5 --eval_sign $EVALSIGN --digital_rotations --batch_size 128 random" \
      --taskname random \
      --name $MODEL/$EVALSIGN/samples

      python manager.py slurm test.py --partition abc --constraint 'titan' --mem '125GB' --cpus 5 --gpus 1 \
      --task "$MODEL.h5 --eval_sign $EVALSIGN --digital_rotations --batch_size 128 modes" \
      --taskname evalmodes \
      --name $MODEL/$EVALSIGN/evalmodes

#      for NA in 1.
#      do
#        for COV in 1.0
#        do
#          python manager.py slurm test.py --partition abc_a100 --mem '500GB' --cpus 16 --gpus 4 \
#          --task "$MODEL.h5 --datadir $DATA --input_coverage $COV --na $NA --batch_size $BATCH --niter $ITERS --eval_sign $EVALSIGN --n_samples $MAX densityheatmap" \
#          --taskname $NA \
#          --name $MODEL/$EVALSIGN/densityheatmaps_${COV}
#
#          python manager.py slurm test.py --partition abc_a100 --mem '500GB' --cpus 16 --gpus 4 \
#          --task "$MODEL.h5 --datadir $DATA --input_coverage $COV --na $NA --batch_size $BATCH  --niter 10 --eval_sign $EVALSIGN --n_samples $MAX iterheatmap" \
#          --taskname $NA \
#          --name $MODEL/$EVALSIGN/iterheatmaps_${COV}
#
#          python manager.py slurm test.py --partition dgx --mem '500GB' --cpus 32 --gpus 2 \
#          --task "$MODEL.h5 --datadir $DATA --input_coverage $COV --na $NA --batch_size $BATCH --niter $ITERS --eval_sign $EVALSIGN --n_samples $MAX snrheatmap" \
#          --taskname $NA \
#          --name $MODEL/$EVALSIGN/snrheatmaps_${COV}
#        done
#      done
  done
done
