#!/bin/bash

xVOXEL=.15
yVOXEL=.15
zVOXEL=.6
LAMBDA=.920
SHAPE=64
MAXAMP=1.
PSF_TYPE='confocal'
DATA="/clusterfs/nvme/thayer/dataset/${PSF_TYPE}/test/x150-y150-z600/"

declare -a models=(
'../models/new/confocal/opticaltransformer'
'../models/new/confocal/opticalresnet'
)

for MODEL in "${models[@]}"
do
  for NA in 1 .95 .9 .85 .8
  do
    #python manager.py slurm test.py --partition abc_a100 --mem '500GB' --gpus 4 --cpus 16 \
    #python manager.py slurm test.py --partition abc --constraint titan --mem '500GB' --gpus 4 --cpus 20 \
    python manager.py slurm test.py --partition abc --mem '500GB' --cpus 24 --gpus 0\
    --task "$MODEL --datadir $DATA/i$SHAPE --n_samples 100 --na $NA --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL --max_amplitude $MAXAMP evalheatmap" \
    --taskname $NA \
    --name $MODEL/evalheatmaps

    #python manager.py slurm test.py --partition abc_a100 --mem '500GB' --gpus 4 --cpus 16 \
    #python manager.py slurm test.py --partition abc --constraint titan --mem '500GB' --gpus 4 --cpus 20 \
    python manager.py slurm test.py --partition abc --mem '500GB' --cpus 24 --gpus 0 \
    --task "$MODEL --datadir $DATA/i$SHAPE --n_samples 20 --na $NA --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL --max_amplitude $MAXAMP iterheatmap" \
    --taskname $NA \
    --name $MODEL/iterheatmaps
  done

  python manager.py slurm predict.py --partition abc --constraint titan --mem '125GB' --gpus 1 --cpus 5 \
  --task "$MODEL --psf_type $PSF_TYPE --wavelength $LAMBDA --x_voxel_size $xVOXEL --y_voxel_size $yVOXEL --z_voxel_size $zVOXEL random" \
  --taskname random \
  --name $MODEL/samples
done
