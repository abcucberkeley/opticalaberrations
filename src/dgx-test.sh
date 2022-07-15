#!/bin/bash

SHAPE=64
DATA='/clusterfs/nvme/thayer/dataset/bimodal_embedding/test/x150-y150-z600/'

declare -a models=(
'../models/new/bimodal_embedding/adam/gelu/multikernel/'
)

for MODEL in "${models[@]}"
do
  for DIST in / powerlaw dirichlet
  do
    for TEST in evalheatmap iterheatmap
    do
      nohup python test.py $MODEL --datadir $DATA/i$SHAPE --dist $DIST $TEST &
    done
  done
done
