#!/bin/bash

xVOXEL=.108
yVOXEL=.108
zVOXEL=.200
LAMBDA=.510
SHAPE=64
SAMPLES=100
MAXAMP=.5
MODES=55
DIFFICULTY='hard'
DATASET='yumb'
PSF_TYPE='../lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat'
DATA="/clusterfs/nvme/thayer/dataset/$DATASET/$DIFFICULTY/test/x108-y108-z200/"


for M in opticalnet otfnet
do
  MODEL="../models/new/$DATASET/$DIFFICULTY/$M"
  echo $MODEL

  python manager.py slurm predict.py --partition abc --mem '64GB' --cpus 4 --gpus 0 \
  --task "$MODEL --psf_type $PSF_TYPE --n_modes $MODES --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL metadata" \
  --taskname metadata \
  --name $MODEL/metadata

  python manager.py slurm test.py --partition abc --mem '250GB' --cpus 12 --gpus 0 \
  --task "$MODEL modes" \
  --taskname 'test' \
  --name $MODEL/evalmodes

  for NA in 1. .9 .8
  do
    for COV in 1.0 0.5
    do
      #python manager.py slurm test.py --partition abc_a100 --mem '500GB' --cpus 16 --gpus 4 \
      #python manager.py slurm test.py --partition dgx --mem '250GB' --cpus 16 --gpus 1 \
      python manager.py slurm test.py --partition abc --mem '250GB' --cpus 12 --gpus 0 \
      --task "$MODEL --datadir $DATA/i$SHAPE --input_coverage $COV --modes $MODES --n_samples $SAMPLES --na $NA --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL --max_amplitude $MAXAMP evalheatmap" \
      --taskname $NA \
      --name $MODEL/evalheatmaps_${COV}

      python manager.py slurm test.py --partition abc --mem '250GB' --cpus 12 --gpus 0 \
      --task "$MODEL --datadir $DATA/i$SHAPE --input_coverage $COV --modes $MODES --n_samples $SAMPLES --na $NA --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL --max_amplitude $MAXAMP distheatmap" \
      --taskname $NA \
      --name $MODEL/distheatmaps_${COV}

      python manager.py slurm test.py --partition abc --mem '250GB' --cpus 12 --gpus 0 \
      --task "$MODEL --datadir $DATA/i$SHAPE --input_coverage $COV --modes $MODES --n_samples $SAMPLES --na $NA --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL --max_amplitude $MAXAMP densityheatmap" \
      --taskname $NA \
      --name $MODEL/densityheatmaps_${COV}

      #python manager.py slurm test.py --partition abc --mem '250GB' --cpus 12 --gpus 0 \
      #--task "$MODEL --datadir $DATA/i$SHAPE --input_coverage $COV --modes $MODES --n_samples $SAMPLES --na $NA --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL --max_amplitude $MAXAMP evalpoints" \
      #--taskname $NA \
      #--name $MODEL/evalpoints_${COV}
    done

    #for N in 2 3 4 5
    #do
      #python manager.py slurm test.py --partition abc --mem '500GB' --cpus 24 --gpus 0 \
      #--task "$MODEL --num_neighbor $N --datadir $DATA/i$SHAPE --modes $MODES --n_samples $SAMPLES --na $NA --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL --max_amplitude $MAXAMP distheatmap" \
      #--taskname $NA \
      #--name $MODEL/distheatmaps_neighbor_${N}
    #done
  done

  for COV in 1.0 0.5
  do
    python manager.py slurm predict.py --partition abc --mem '64GB' --cpus 4 --gpus 0 \
    --task "$MODEL --input_coverage $COV random" \
    --taskname $COV \
    --name $MODEL/samples
  done

done
