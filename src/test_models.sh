#!/bin/bash

xVOXEL=.108
yVOXEL=.108
zVOXEL=.200
LAMBDA=.510
SHAPE=64
SAMPLES=25
MAXAMP=.5
MODES=15

DIFFICULTY='easy'
DATASET='yumb_lattice_objects'
PSF_TYPE='../lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat'
DATA="/clusterfs/nvme/thayer/dataset/$DATASET/$DIFFICULTY/test/x108-y108-z200/"

declare -a models=(
'../models/new/yumb_lattice_objects/easy/opticaltransformer'
)

for MODEL in "${models[@]}"
do
  for NA in 1. .9 .8
  do
    for COV in 1.0 0.75 0.5
    do
      python manager.py slurm test.py --partition abc --mem '250GB' --cpus 12 --gpus 0 \
      --task "$MODEL --datadir $DATA/i$SHAPE --input_coverage $COV --modes $MODES --n_samples $SAMPLES --na $NA --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL --max_amplitude $MAXAMP evalheatmap" \
      --taskname $NA \
      --name $MODEL/evalheatmaps_${COV}

      #python manager.py slurm test.py --partition abc --mem '250GB' --cpus 12 --gpus 0 \
      #--task "$MODEL --datadir $PSF_DATA/i$SHAPE --input_coverage $COV --modes $MODES --n_samples $SAMPLES --na $NA --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL --max_amplitude $MAXAMP iterheatmap" \
      #--taskname $NA \
      #--name $MODEL/iterheatmaps_${COV}

      python manager.py slurm test.py --partition abc --mem '250GB' --cpus 12 --gpus 0 \
      --task "$MODEL --datadir $DATA/i$SHAPE --input_coverage $COV --modes $MODES --n_samples $SAMPLES --na $NA --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL --max_amplitude $MAXAMP distheatmap" \
      --taskname all \
      --name $MODEL/distheatmaps_neighbor_None_${COV}

      #python manager.py slurm test.py --partition abc --mem '250GB' --cpus 12 --gpus 0 \
      #--task "$MODEL --datadir $DATA/i$SHAPE --input_coverage $COV --modes $MODES --n_samples $SAMPLES --na $NA --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL --max_amplitude $MAXAMP evalpoints" \
      #--taskname all \
      #--name $MODEL/evalpoints_neighbor_None
    done

    for N in 2 3 4 5
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

  for COV in 1.0 0.75 0.5
  do
    python manager.py slurm predict.py --partition abc --mem '64GB' --cpus 4 --gpus 0 \
    --task "$MODEL --psf_type $PSF_TYPE --input_coverage $COV --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL random" \
    --taskname random \
    --name $MODEL/samples
  done

done
