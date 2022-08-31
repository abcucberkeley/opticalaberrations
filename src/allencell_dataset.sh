#!/bin/bash

HANDLER=slurm
ENV=~/anaconda3/envs/deep/bin/python

OUTDIR='/clusterfs/nvme/thayer/allencell/aics/label-free-imaging-collection/dataset/train/golgi_apparatus'
DATASET='/clusterfs/nvme/thayer/allencell/aics/label-free-imaging-collection/channels/fovs/golgi_apparatus'

SHAPE=64
xVOXEL=.15
yVOXEL=.15
zVOXEL=.6
MODES=60
GAMMA=.75
LAMBDA=.605
NA=1.0
DTYPE='widefield'

amps1=($(seq 0 .05 .45))
amps2=($(seq .05 .05 .5))

ABERRATIONS=1
FILES=$(find "${DATASET}" -type f -name "*.tif")
FILES=($(sort --version-sort <<<"${FILES[*]}"))
FILES=${FILES[@]:0:5000}


for DIST in single
do
  for AMP in `seq 1 ${#amps1[@]}`
  do
    while [ $(squeue -u thayeralshaabi -h -t pending -r | wc -l) -gt 500 ]
    do
      sleep 10s
    done

    for IMG in $FILES
    do
      j="${ENV} allencell_dataset.py"
      j="${j} --iters ${ABERRATIONS}"
      j="${j} --sample ${IMG}"
      j="${j} --outdir ${OUTDIR}"
      j="${j} --kernels"
      j="${j} --bimodal"
      j="${j} --dist ${DIST}"
      j="${j} --gamma ${GAMMA}"
      j="${j} --modes ${MODES}"
      j="${j} --input_shape ${SHAPE}"
      j="${j} --min_amplitude ${amps1[$AMP-1]}"
      j="${j} --max_amplitude ${amps2[$AMP-1]}"
      j="${j} --x_voxel_size ${xVOXEL}"
      j="${j} --y_voxel_size ${yVOXEL}"
      j="${j} --z_voxel_size ${zVOXEL}"
      j="${j} --na_detection ${NA}"
      j="${j} --lam_detection ${LAMBDA}"

      task="/usr/bin/sbatch"
      task="${task} --qos=abc_normal"
      task="${task} --partition=abc"
      task="${task} --cpus-per-task=1"
      task="${task} --mem=15G"
      task="${task} --job-name=psnr#${SNR}-amp#${AMP}-iter#${S}"
      task="${task} --time=1:00:00"
      task="${task} --export=ALL"
      task="${task} --wrap=\"${j}\""
      echo $task | bash
      echo "R[$(squeue -u thayeralshaabi -h -t running -r -p abc | wc -l)], P[$(squeue -u thayeralshaabi -h -t pending -r -p abc | wc -l)]"
    done
  done
done
