#!/bin/bash

xVOXEL=.108
yVOXEL=.108
zVOXEL=.200
LAMBDA=.510
SHAPE=64
SAMPLES=100
DATASET='new_embeddings'
PSF_TYPE='../lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat'
DATA="/clusterfs/nvme/thayer/dataset/$DATASET/test/x108-y108-z200/"


for MODES in 15 28 45
do
  for M in principle_planes/opticalnet
  do
    MODEL="../models/new/$DATASET/z$MODES/$M"

    python manager.py slurm test.py --partition abc --mem '250GB' --cpus 12 --gpus 0 \
    --task "$MODEL modes" \
    --taskname 'test' \
    --name $MODEL/evalmodes

    python manager.py slurm test.py --partition abc --mem '64GB' --cpus 4 --gpus 0 \
    --task "$MODEL random" \
    --taskname random \
    --name $MODEL/samples

    for NA in 1. .9 .8
    do
      for COV in 1.0 0.5
      do
        #python manager.py slurm test.py --partition abc_a100 --mem '500GB' --cpus 16 --gpus 4 \
        #python manager.py slurm test.py --partition dgx --mem '250GB' --cpus 16 --gpus 1 \
        python manager.py slurm test.py --partition abc --mem '250GB' --cpus 12 --gpus 0 \
        --task "$MODEL --datadir $DATA/i$SHAPE --input_coverage $COV --n_samples $SAMPLES --na $NA  snrheatmap" \
        --taskname $NA \
        --name $MODEL/snrheatmaps_${COV}

        python manager.py slurm test.py --partition abc --mem '250GB' --cpus 12 --gpus 0 \
        --task "$MODEL --datadir $DATA/i$SHAPE --input_coverage $COV --n_samples $SAMPLES --na $NA  densityheatmap" \
        --taskname $NA \
        --name $MODEL/densityheatmaps_${COV}

        python manager.py slurm test.py --partition abc --mem '250GB' --cpus 12 --gpus 0 \
        --task "$MODEL --datadir $DATA/i$SHAPE --input_coverage $COV --n_samples $SAMPLES --na $NA  iterheatmap" \
        --taskname $NA \
        --name $MODEL/iterheatmaps_${COV}
      done
    done
  done
done
