#!/bin/bash

xVOXEL=.108
yVOXEL=.108
zVOXEL=.268
LAMBDA=.510
SHAPE=64
SAMPLES=100
MAXAMP=.5
MODES=60

DATASET='lattice_objects'
PSF_DATA="/clusterfs/nvme/thayer/dataset/lattice/test/x108-y108-z268/"
PSF_TYPE='../lattice/HexRect_NAlattice0.25_NAAnnulusMax0.60_NAsigma0.08.mat'
DATA="/clusterfs/nvme/thayer/dataset/$DATASET/easy/test/x108-y108-z268/"

declare -a models=(
'../models/new/lattice_objects/easy/opticaltransformer'
)

for MODEL in "${models[@]}"
do
  for NA in 1 .95 .9 .85 .8
  do
    python manager.py slurm test.py --partition abc --mem '250GB' --cpus 12 --gpus 0 \
    --task "$MODEL --datadir $DATA/i$SHAPE --modes $MODES --n_samples $SAMPLES --na $NA --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL --max_amplitude $MAXAMP evalheatmap" \
    --taskname $NA \
    --name $MODEL/evalheatmaps_1.0

    #python manager.py slurm test.py --partition abc --mem '250GB' --cpus 12 --gpus 0 \
    #--task "$MODEL --datadir $PSF_DATA/i$SHAPE --modes $MODES --n_samples $SAMPLES --na $NA --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL --max_amplitude $MAXAMP iterheatmap" \
    #--taskname $NA \
    #--name $MODEL/iterheatmaps_1.0

    python manager.py slurm test.py --partition abc --mem '250GB' --cpus 12 --gpus 0 \
    --task "$MODEL --datadir $DATA/i$SHAPE --modes $MODES --n_samples $SAMPLES --na $NA --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL --max_amplitude $MAXAMP distheatmap" \
    --taskname all \
    --name $MODEL/distheatmaps_neighbor_None

    #python manager.py slurm test.py --partition abc --mem '250GB' --cpus 12 --gpus 0 \
    #--task "$MODEL --datadir $DATA/i$SHAPE --modes $MODES --n_samples $SAMPLES --na $NA --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL --max_amplitude $MAXAMP evalpoints" \
    #--taskname all \
    #--name $MODEL/evalpoints_neighbor_None

    for N in 1 2 3 4 5
    do
      python manager.py slurm test.py --partition abc --mem '250GB' --cpus 12 --gpus 0 \
      --task "$MODEL --num_neighbor $N --datadir $DATA/i$SHAPE --modes $MODES --n_samples $SAMPLES --na $NA --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL --max_amplitude $MAXAMP distheatmap" \
      --taskname $N \
      --name $MODEL/distheatmaps_neighbor_$N

      #python manager.py slurm test.py --partition abc --mem '250GB' --cpus 12 --gpus 0 \
      #--task "$MODEL --num_neighbor $N --datadir $DATA/i$SHAPE --modes $MODES --n_samples $SAMPLES --na $NA --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL --max_amplitude $MAXAMP evalpoints" \
      #--taskname $N \
      #--name $MODEL/evalpoints_neighbor_$N
    done
  done

  python manager.py slurm predict.py --partition abc --mem '64GB' --cpus 4 --gpus 0 \
  --task "$MODEL --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL random" \
  --taskname random \
  --name $MODEL/samples
done
