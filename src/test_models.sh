#!/bin/bash

xVOXEL=.108
yVOXEL=.108
zVOXEL=.200
LAMBDA=.510
SHAPE=64
DATASET='spatial_planes_embeddings'
PSF_TYPE='../lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat'
DATA="/clusterfs/nvme/thayer/dataset/$DATASET/test/x108-y108-z200/i$SHAPE/z15"
EVALSIGN="signed"  ## options: "positive_only", "dual_stage", "signed"
BATCH=2048

for MODES in 15 28 45
do
  for M in phase/opticalnet compact/opticalnet
  do
    MODEL="../models/new/$DATASET/z$MODES/$M"

    python manager.py slurm test.py --partition abc --mem '250GB' --cpus 12 --gpus 0 \
    --task "$MODEL --eval_sign $EVALSIGN modes" \
    --taskname 'test' \
    --name $MODEL/$EVALSIGN/evalmodes

    python manager.py slurm test.py --partition abc --mem '64GB' --cpus 4 --gpus 0 \
    --task "$MODEL --eval_sign $EVALSIGN random" \
    --taskname random \
    --name $MODEL/$EVALSIGN/samples

    for NA in 1.
    do
      for COV in 1.0
      do
        python manager.py slurm test.py --partition abc --mem '1000GB' --cpus 48 --gpus 0 \
        --task "$MODEL --datadir $DATA --input_coverage $COV --na $NA --batch_size $BATCH --eval_sign $EVALSIGN densityheatmap" \
        --taskname $NA \
        --name $MODEL/$EVALSIGN/densityheatmaps_${COV}

        python manager.py slurm test.py --partition abc --mem '500GB' --cpus 24 --gpus 4 \
        --task "$MODEL --datadir $DATA --input_coverage $COV --na $NA --batch_size $BATCH --eval_sign $EVALSIGN iterheatmap" \
        --taskname $NA \
        --name $MODEL/$EVALSIGN/iterheatmaps_${COV}

        python manager.py slurm test.py --partition dgx --mem '500GB' --cpus 32 --gpus 2 \
        --task "$MODEL --datadir $DATA --input_coverage $COV --na $NA --batch_size $BATCH --eval_sign $EVALSIGN --n_samples 10000 snrheatmap" \
        --taskname $NA \
        --name $MODEL/$EVALSIGN/snrheatmaps_${COV}
      done
    done
  done
done
