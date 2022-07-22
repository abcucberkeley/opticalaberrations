#!/bin/bash

xVOXEL=.15
yVOXEL=.15
zVOXEL=.6
SHAPE=64
MAXAMP=1.
DATA='/clusterfs/nvme/thayer/dataset/embeddings/test/x150-y150-z600/'

declare -a models=(
'../models/new/embeddings/transformers/p32-p16-p8x2/'
)

for MODEL in "${models[@]}"
do
  for NA in 1 .95 .9 .85 .8
  do
    for REF in 'single_point' 'two_points' 'five_points' \
    '10_points' '25_points' '50_points' '75_points' '100_points' \
    'line' 'sheet' 'sphere' 'cylinder' \
    'point_and_line' 'point_and_sheet' 'point_and_cylinder' \
    'several_points_and_line' 'several_points_and_sheet' 'several_points_and_cylinder'
    do
      #python manager.py slurm test.py --partition abc_a100 --mem '500GB' --gpus 3 --cpus 16 \
      #python manager.py slurm test.py --partition abc --constraint titan --mem '500GB' --gpus 4 --cpus 20 \
      python manager.py slurm test.py --partition abc --mem '500GB' --cpus 24 --gpus 0 \
      --task "$MODEL --na $NA --reference ../data/shapes/$REF.tif evalsample" \
      --taskname $NA \
      --name $MODEL/shapes/$REF

#      python manager.py slurm test.py --partition abc --mem '500GB' --cpus 24 --gpus 0 \
#      --task "$MODEL --datadir $DATA/i$SHAPE --na $NA --n_samples 20 --max_amplitude $MAXAMP --reference ../data/shapes/$REF.tif iterheatmap" \
#      --taskname $NA \
#      --name $MODEL/shapes/$REF
    done
  done
done
