#!/bin/bash

HANDLER=slurm
ENV=~/anaconda3/envs/ml/bin/python

PSF_TYPE='../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat'

xVOXEL=.108
yVOXEL=.108
zVOXEL=.200
LAMBDA=.510
NA=1.0
CPUS=1
MEM='20G'
TIMELIMIT='1:00:00'
SHAPE=64

MODES=15
TITLE='lls_defocus_only'
MODE_DIST='pyramid'
OUTDIR="../dataset/${TITLE}"

LOGS="${OUTDIR}/logs"
mkdir -p $OUTDIR
mkdir -p $LOGS

SAMPLES_PER_JOB=100
SAMPLES_PER_BIN=200
defocus1=($(seq -2 .2 1.8))
defocus2=($(seq -1.8 .2 2))
mPH=($(seq 1 50000 460000))
xPH=($(seq 50000 50000 500000))
SAMPLES=($(seq 1 $SAMPLES_PER_JOB $SAMPLES_PER_BIN))


for PH in `seq 1 ${#xPH[@]}`
do
  for OFFSET in `seq 1 ${#defocus1[@]}`
  do
    for S in `seq 1 ${#SAMPLES[@]}`
    do
        j="${ENV} psf_dataset.py"
        j="${j} --psf_type ${PSF_TYPE}"
        j="${j} --dist mixed"
        j="${j} --mode_dist ${MODE_DIST}"
        j="${j} --iters ${SAMPLES_PER_JOB}"
        j="${j} --signed"
        j="${j} --rotate"
        j="${j} --noise"
        j="${j} --normalize"
        j="${j} --outdir ${OUTDIR}"
        j="${j} --modes ${MODES}"
        j="${j} --input_shape ${SHAPE}"
        j="${j} --min_photons ${mPH[$PH-1]}"
        j="${j} --max_photons ${xPH[$PH-1]}"
        j="${j} --min_amplitude 0."
        j="${j} --max_amplitude 0."
        j="${j} --min_lls_defocus_offset ${defocus1[$OFFSET-1]}"
        j="${j} --max_lls_defocus_offset ${defocus2[$OFFSET-1]}"
        j="${j} --filename ${SAMPLES[$S-1]}"
        j="${j} --x_voxel_size ${xVOXEL}"
        j="${j} --y_voxel_size ${yVOXEL}"
        j="${j} --z_voxel_size ${zVOXEL}"
        j="${j} --na_detection ${NA}"
        j="${j} --lam_detection ${LAMBDA}"
        j="${j} --cpu_workers ${CPUS}"

        if [ $HANDLER = 'slurm' ];then
            while [ $(squeue -u $USER -h -t pending -r | wc -l) -gt 300 ]
            do
              sleep 10s
            done

            JOB="${TITLE}-${MODES}-photons${xPH[$PH-1]}-defocus${defocus2[$OFFSET-1]}-iter#${S}"

            task="/usr/bin/sbatch"
            task="${task} --qos=abc_normal --nice=1111111111"
            task="${task} --partition=abc"
            task="${task} --cpus-per-task=${CPUS}"
            task="${task} --mem='${MEM}'"
            task="${task} --job-name=${JOB}"
            task="${task} --time=${TIMELIMIT}"
            task="${task} --output=${LOGS}/${JOB}.log"
            task="${task} --export=ALL"
            task="${task} --wrap=\"${j}\""

            echo $task | bash
            echo "ABC : R[$(squeue -u $USER -h -t running -r -p abc | wc -l)], P[$(squeue -u $USER -h -t pending -r -p abc | wc -l)]"
        else
          echo $j | bash
        fi

    done
  done
done