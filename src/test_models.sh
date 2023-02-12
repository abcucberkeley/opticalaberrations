#!/bin/bash

xVOXEL=.108
yVOXEL=.108
zVOXEL=.200
LAMBDA=.510
SHAPE=64
DATASET='spatial_planes_embeddings'
PSF_TYPE='../lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat'
DATA="/clusterfs/nvme/thayer/dataset/$DATASET/test/x108-y108-z200/i$SHAPE/z15"
BATCH=512
ROTATIONS='--digital_rotations'
ITERS=1
MAX=10000

for EVALSIGN in positive_only signed
do
  for MODES in 15 28 45
  do
    for M in phase/opticalnet
    do
      MODEL="../models/new/$DATASET/z$MODES/$M"

      python manager.py slurm test.py --partition abc --mem '250GB' --cpus 12 --gpus 0 \
      --task "$MODEL --eval_sign $EVALSIGN modes" \
      --taskname 'test' \
      --name $MODEL/$EVALSIGN/evalmodes

      python manager.py slurm test.py --partition abc --mem '250GB' --cpus 12 --gpus 0 \
      --task "$MODEL --eval_sign $EVALSIGN random" \
      --taskname random \
      --name $MODEL/$EVALSIGN/samples

      for NA in 1.
      do
        for COV in 1.0
        do
          python manager.py slurm test.py --partition abc_a100 --mem '500GB' --cpus 16 --gpus 4 \
          --task "$MODEL --datadir $DATA --input_coverage $COV --na $NA --batch_size $BATCH --niter $ITERS --eval_sign $EVALSIGN --n_samples $MAX densityheatmap" \
          --taskname $NA \
          --name $MODEL/$EVALSIGN/densityheatmaps_${COV}

          python manager.py slurm test.py --partition abc_a100 --mem '500GB' --cpus 16 --gpus 4 \
          --task "$MODEL --datadir $DATA --input_coverage $COV --na $NA --batch_size $BATCH  --niter 10 --eval_sign $EVALSIGN --n_samples $MAX iterheatmap" \
          --taskname $NA \
          --name $MODEL/$EVALSIGN/iterheatmaps_${COV}

          python manager.py slurm test.py --partition dgx --mem '500GB' --cpus 32 --gpus 2 \
          --task "$MODEL --datadir $DATA --input_coverage $COV --na $NA --batch_size $BATCH --niter $ITERS --eval_sign $EVALSIGN --n_samples $MAX snrheatmap" \
          --taskname $NA \
          --name $MODEL/$EVALSIGN/snrheatmaps_${COV}
        done
      done
    done
  done
done
