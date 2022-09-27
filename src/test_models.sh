#!/bin/bash

xVOXEL=.108
yVOXEL=.108
zVOXEL=.268
LAMBDA=.510
SHAPE=64
SAMPLES=100
MAXAMP=.5
PHASE='--no_phase'

PSF_TYPE='/clusterfs/nvme/thayer/dataset/lattice/simulations/NAlattice0.25/HexRect/NAAnnulusMax0.60/NAsigma0.08/decon_simulation/PSF_OTF_simulation.mat'
DATA="/clusterfs/nvme/thayer/dataset/lattice_multipoints/test/x108-y108-z268/"

declare -a models=(
'../models/new/multipoints/lattice/nophase/opticaltransformer-multinode'
)

for MODEL in "${models[@]}"
do
  for NA in 1 .95 .9 .85 .8
  do
    python manager.py slurm test.py --partition abc --mem '250GB' --cpus 12 --gpus 0 \
    --task "$MODEL $PHASE --datadir $DATA/i$SHAPE --n_samples $SAMPLES --na $NA --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL --max_amplitude $MAXAMP evalheatmap" \
    --taskname $NA \
    --name $MODEL/evalheatmaps_1.0

    #python manager.py slurm test.py --partition abc --mem '250GB' --cpus 12 --gpus 0 \
    #--task "$MODEL $PHASE --datadir $DATA/i$SHAPE --n_samples $SAMPLES --na $NA --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL --max_amplitude $MAXAMP iterheatmap" \
    #--taskname $NA \
    #--name $MODEL/iterheatmaps_1.0

    for N in 2 3 4 5
    do
      python manager.py slurm test.py --partition abc --mem '250GB' --cpus 12 --gpus 0 \
      --task "$MODEL $PHASE --num_neighbor $N --datadir $DATA/i$SHAPE --n_samples $SAMPLES --na $NA --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL --max_amplitude $MAXAMP distheatmap" \
      --taskname $N \
      --name $MODEL/distheatmaps_neighbor_$N
    done
  done

  python manager.py slurm predict.py --partition abc --mem '500GB' --cpus 4 --gpus 0 \
  --task "$MODEL $PHASE --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL random" \
  --taskname random \
  --name $MODEL/samples
done
