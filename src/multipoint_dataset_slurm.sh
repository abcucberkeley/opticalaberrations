#!/bin/bash

ENV=~/anaconda3/envs/ml/bin/python
NODES='abc'

xVOXEL=.108
yVOXEL=.108
zVOXEL=.200
LAMBDA=.510
NA=1.0
ALPHA='abs'
PHI='angle'
CPUS=1
MEM='20G'
TIMELIMIT='2:00:00'
SHAPE=64
MAX_LLS_OFFSET=0
RAND_VSIZE=false

MODES=15
TITLE='fixed_density'
DATASET='test'

MODE_DIST='pyramid'
OUTDIR="/clusterfs/nvme/thayer/dataset/${TITLE}/${DATASET}"
LOGS="${OUTDIR}/logs"
mkdir -p $OUTDIR
mkdir -p $LOGS

if [ "$DATASET" = "train" ];then
  TYPE='--emb'
  SAMPLES_PER_JOB=200
  SAMPLES_PER_BIN=400
  OBJS=(1 2 3 4 5)
  mPH=($(seq 1 25000 250000))
  xPH=($(seq 25000 25000 250000))
  amps1=($(seq 0 .01 .24))
  amps2=($(seq .01 .01 .25))
  SAMPLES=($(seq 1 $SAMPLES_PER_JOB $SAMPLES_PER_BIN))
  DISTRIBUTIONS=(single bimodal powerlaw dirichlet)
  FILL_RADIUS=0.66
else
  TYPE=''
  SAMPLES_PER_JOB=25
  SAMPLES_PER_BIN=25
  OBJS=(1 2 3 5 10 25 50 100 125 150)
  mPH=($(seq     1 25000 525000))
  xPH=($(seq 25000 25000 525000))
  amps1=($(seq    0 .025 .475))
  amps2=($(seq .025 .025 .50))
  SAMPLES=($(seq 1 $SAMPLES_PER_JOB $SAMPLES_PER_BIN))
  DISTRIBUTIONS=(mixed)
  FILL_RADIUS=0.66
fi

TOTAL_SAMPLES=$(( ${#DISTRIBUTIONS[@]} * ${#mPH[@]} * ${#amps1[@]} * ${#OBJS[@]} * $SAMPLES_PER_BIN ))
TOTAL_JOBS=$(( $TOTAL_SAMPLES / $SAMPLES_PER_JOB ))

JOB_COUNTER=0
for DIST in `seq 1 ${#DISTRIBUTIONS[@]}`
do
  for PH in `seq 1 ${#xPH[@]}`
  do
    for AMP in `seq 1 ${#amps1[@]}`
    do
      for N in `seq 1 ${#OBJS[@]}`
      do
        for S in `seq 1 ${#SAMPLES[@]}`
        do
            while [ $(squeue -u $USER -h -t pending -r | wc -l) -gt 500 ]
            do
              sleep 10s
            done

            (( JOB_COUNTER=JOB_COUNTER+1 ))

            j="${ENV} multipoint_dataset.py ${TYPE}"
            j="${j} --npoints ${OBJS[$N-1]}"
            j="${j} --alpha_val ${ALPHA}"
            j="${j} --phi_val ${PHI}"
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
            j="${j} --min_photons ${mPH[$PH-1]}"
            j="${j} --max_photons ${xPH[$PH-1]}"
            j="${j} --min_amplitude ${amps1[$AMP-1]}"
            j="${j} --max_amplitude ${amps2[$AMP-1]}"
            j="${j} --min_lls_defocus_offset -$MAX_LLS_OFFSET"
            j="${j} --max_lls_defocus_offset $MAX_LLS_OFFSET"
            j="${j} --filename ${SAMPLES[$S-1]}"
            j="${j} --x_voxel_size ${xVOXEL}"
            j="${j} --y_voxel_size ${yVOXEL}"
            j="${j} --z_voxel_size ${zVOXEL}"
            j="${j} --na_detection ${NA}"
            j="${j} --fill_radius ${FILL_RADIUS}"
            j="${j} --lam_detection ${LAMBDA}"
            j="${j} --cpu_workers ${CPUS}"

            if $RAND_VSIZE; then
              j="${j} --randomize_voxel_size"
            fi

            for e in spatial_planes
            do
              j="${j} --embedding_option ${e}"
            done

            if [ "$DATASET" = "train" ];then
              for psf in "../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat" "../lattice/v2Hex_NAexc0p50_NAsigma0p075_annulus0p60-0p40_FWHM53p0.mat" widefield confocal 2photon
              do
                j="${j} --psf_type ${psf}"
              done
            fi

            task="/usr/bin/sbatch"
            task="${task} --qos=abc_normal --nice=1111111111"

            if [ "$NODES" = "all" ];then
              if [ $(squeue -u $USER -h -t pending -r -p dgx | wc -l) -lt 100 ];then
                task="${task} --partition=dgx"
              elif [ $(squeue -u $USER -h -t pending -r -p abc_a100 | wc -l) -lt 100 ];then
                task="${task} --partition=abc_a100"
              else
                task="${task} --partition=abc"
              fi
            elif [ "$NODES" = "gpus" ];then
              if [ $(squeue -u $USER -h -t pending -r -p dgx | wc -l) -lt 100 ];then
                task="${task} --partition=dgx"
              elif [ $(squeue -u $USER -h -t pending -r -p abc_a100 | wc -l) -lt 100 ];then
                task="${task} --partition=abc_a100"
              else
                task="${task} --partition=abc --constraint titan"
              fi
            else
              task="${task} --partition=abc"
            fi

            JOB="${TITLE}-${DATASET}-${MODES}-${DISTRIBUTIONS[$DIST-1]}-photons${xPH[$PH-1]}-amp${amps2[$AMP-1]}-objs${OBJS[$N-1]}-iter#${S}"
            task="${task} --cpus-per-task=${CPUS}"
            task="${task} --mem='${MEM}'"
            task="${task} --job-name=${JOB}"
            task="${task} --time=${TIMELIMIT}"
            task="${task} --output=${LOGS}/${JOB}.log"
            task="${task} --export=ALL"
            task="${task} --wrap=\"${j}\""

            echo $task | bash
            echo "DGX : Running[$(squeue -u $USER -h -t running -r -p dgx | wc -l)], Pending[$(squeue -u $USER -h -t pending -r -p dgx | wc -l)]"
            echo "A100: Running[$(squeue -u $USER -h -t running -r -p abc_a100 | wc -l)], Pending[$(squeue -u $USER -h -t pending -r -p abc_a100 | wc -l)]"
            echo "ABC : Running[$(squeue -u $USER -h -t running -r -p abc | wc -l)], Pending[$(squeue -u $USER -h -t pending -r -p abc | wc -l)]"
            printf "JOBS: [ %'d / %'d ] \n" $(($TOTAL_JOBS - $JOB_COUNTER)) $TOTAL_JOBS
            printf "SAMPLES: [ %'d / %'d ] \n" $((($TOTAL_JOBS - $JOB_COUNTER) * $SAMPLES_PER_JOB)) $TOTAL_SAMPLES

        done
      done
    done
  done
done
