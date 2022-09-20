#!/bin/bash

xVOXEL=.108
yVOXEL=.108
zVOXEL=.268
LAMBDA=.510
SHAPE=64
MAXAMP=1.
PSF_TYPE='/clusterfs/nvme/thayer/dataset/lattice/simulations/NAlattice0.25/HexRect/NAAnnulusMax0.60/NAsigma0.08/decon_simulation/PSF_OTF_simulation.mat'
DATA="/clusterfs/nvme/thayer/dataset/lattice/test/x108-y108-z268/"

declare -a models=(
'../models/new/lattice/opticaltransformer'
)

for SIZE in 1. .75 .5
do
  for MODEL in "${models[@]}"
  do
    for NA in 1 .95 .9 .85 .8
    do
      #python manager.py slurm test.py --partition abc_a100 --mem '500GB' --gpus 4 --cpus 16 \
      #python manager.py slurm test.py --partition abc --constraint titan --mem '500GB' --gpus 4 --cpus 20 \
      python manager.py slurm test.py --partition abc --mem '500GB' --cpus 24 --gpus 0\
      --task "$MODEL --input_coverage $SIZE --datadir $DATA/i$SHAPE --n_samples 100 --na $NA --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL --max_amplitude $MAXAMP evalheatmap" \
      --taskname $NA \
      --name $MODEL/evalheatmaps_$SIZE

      #python manager.py slurm test.py --partition abc_a100 --mem '500GB' --gpus 4 --cpus 16 \
      #python manager.py slurm test.py --partition abc --constraint titan --mem '500GB' --gpus 4 --cpus 20 \
      python manager.py slurm test.py --partition abc --mem '500GB' --cpus 24 --gpus 0 \
      --task "$MODEL --datadir $DATA/i$SHAPE --n_samples 20 --na $NA --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL --max_amplitude $MAXAMP iterheatmap" \
      --taskname $NA \
      --name $MODEL/iterheatmaps_$SIZE
    done

    python manager.py slurm predict.py --partition abc --mem '500GB' --cpus 4 --gpus 0 \
    --task "$MODEL --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL random" \
    --taskname random \
    --name $MODEL/samples_$SIZE
  done
done