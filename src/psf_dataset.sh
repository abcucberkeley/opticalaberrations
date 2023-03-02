#!/bin/bash

HANDLER=slurm
ENV=~/anaconda3/envs/ml/bin/python

#PSF_TYPE='widefield'
#PSF_TYPE='confocal'
#PSF_TYPE='../lattice/HexRect_NAlattice0.25_NAAnnulusMax0.60_NAsigma0.08.mat'
PSF_TYPE='../lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat'
xVOXEL=.108
yVOXEL=.108
zVOXEL=.200
LAMBDA=.510
NA=1.0
CPUS=1
MEM='20G'
TIMELIMIT='1:00:00'
SHAPE=64
MIN_LLS_OFFSET=-1
MAX_LLS_OFFSET=1

MODES=15
TITLE='psfs'
MODE_DIST='pyramid'
OUTDIR="/clusterfs/nvme/thayer/dataset/${TITLE}"

SAMPLES_PER_JOB=100
SAMPLES_PER_BIN=500
mPSNR=($(seq 11 10 51))
xPSNR=($(seq 20 10 60))
amps1=($(seq 0 .01 .29))
amps2=($(seq .01 .01 .3))
SAMPLES=($(seq 1 $SAMPLES_PER_JOB $SAMPLES_PER_BIN))
DISTRIBUTIONS=(single bimodal powerlaw dirichlet)


for DIST in `seq 1 ${#DISTRIBUTIONS[@]}`
do
  for SNR in `seq 1 ${#xPSNR[@]}`
  do
    for AMP in `seq 1 ${#amps1[@]}`
    do
      for S in `seq 1 ${#SAMPLES[@]}`
      do
          while [ $(squeue -u $USER -h -t pending -r | wc -l) -gt 300 ]
          do
            sleep 10s
          done

          j="${ENV} psf_dataset.py ${TYPE}"
          j="${j} --psf_type ${PSF_TYPE}"
          j="${j} --dist ${DISTRIBUTIONS[$DIST-1]}"
          j="${j} --mode_dist ${MODE_DIST}"
          j="${j} --iters ${SAMPLES_PER_JOB}"
          j="${j} --signed"
          j="${j} --rotate"
          j="${j} --noise"
          j="${j} --normalize"
          j="${j} --outdir ${OUTDIR}"
          j="${j} --modes ${MODES}"
          j="${j} --input_shape ${SHAPE}"
          j="${j} --min_psnr ${mPSNR[$SNR-1]}"
          j="${j} --max_psnr ${xPSNR[$SNR-1]}"
          j="${j} --min_amplitude ${amps1[$AMP-1]}"
          j="${j} --max_amplitude ${amps2[$AMP-1]}"
          j="${j} --min_lls_defocus_offset $MIN_LLS_OFFSET"
          j="${j} --max_lls_defocus_offset $MAX_LLS_OFFSET"
          j="${j} --filename ${SAMPLES[$S-1]}"
          j="${j} --x_voxel_size ${xVOXEL}"
          j="${j} --y_voxel_size ${yVOXEL}"
          j="${j} --z_voxel_size ${zVOXEL}"
          j="${j} --na_detection ${NA}"
          j="${j} --lam_detection ${LAMBDA}"
          j="${j} --cpu_workers ${CPUS}"

          task="/usr/bin/sbatch"
          task="${task} --qos=abc_normal --nice=1111111111"
          task="${task} --partition=abc"
          task="${task} --cpus-per-task=${CPUS}"
          task="${task} --mem='${MEM}'"
          task="${task} --job-name=${TITLE}-${DISTRIBUTIONS[$DIST-1]}-psnr${xPSNR[$SNR-1]}-amp${amps2[$AMP-1]}-iter#${S}"
          task="${task} --time=${TIMELIMIT}"
          task="${task} --export=ALL"
          task="${task} --wrap=\"${j}\""
          echo $task | bash
          echo "ABC : R[$(squeue -u $USER -h -t running -r -p abc | wc -l)], P[$(squeue -u $USER -h -t pending -r -p abc | wc -l)]"
      done
    done
  done
done