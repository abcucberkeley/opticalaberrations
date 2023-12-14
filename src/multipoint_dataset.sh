#!/bin/bash

HANDLER=lsf
ENV=~/anaconda3/envs/ml/bin/python

xVOXEL=.125
yVOXEL=.125
zVOXEL=.200
LAMBDA=.510
NA=1.0
ALPHA='abs'
PHI='angle'
MODE_DIST='pyramid'
CPUS=1
TIMELIMIT='1:00'
SHAPE=64
MAX_LLS_OFFSET=0
RAND_VSIZE=false
SKIP_PREPROCESSING=true
USE_THEORETICAL_WIDEFIELD_SIMULATOR=true

MODES=15
TITLE='fit_125nm_dataset'
DATASET='train'
MODE_DIST='pyramid'

MODALITIES=(
  "../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat"
#  "../lattice/v2Hex_NAexc0p50_NAsigma0p075_annulus0p60-0p40_FWHM53p0.mat"
#  "widefield"
#  "confocal"
#  "2photon"
)

if [ $HANDLER = 'lsf' ];then
  OUTDIR="/groups/betzig/betziglab/thayer/dataset/${TITLE}/${DATASET}"
elif [ $HANDLER = 'slurm' ]; then
  OUTDIR="/clusterfs/nvme/thayer/dataset/${TITLE}/${DATASET}"
else
  OUTDIR="~/dataset/${TITLE}/${DATASET}"
fi

LOGS="${OUTDIR}/logs"
mkdir -p $OUTDIR
mkdir -p $LOGS

if [ "$DATASET" = "train" ];then  # 2M samples
  TYPE='--emb'
  SAMPLES_PER_JOB=400
  SAMPLES_PER_BIN=400
  SAMPLES=($(seq 1 $SAMPLES_PER_JOB $SAMPLES_PER_BIN))
  OBJS=(1 2 3 4 5)  # 5 bins
  mPH=($(seq 100000 50000 550000)) # 10 bins
  xPH=($(seq 150000 50000 600000))
  amps1=($(seq 0 .01 .24))  # 25 bins
  amps2=($(seq .01 .01 .25))
  DISTRIBUTIONS=(single bimodal powerlaw dirichlet) # 4 bins
  FILL_RADIUS=0.66
else
  TYPE=''
  SAMPLES_PER_JOB=25
  SAMPLES_PER_BIN=25
  SAMPLES=($(seq 1 $SAMPLES_PER_JOB $SAMPLES_PER_BIN))
  OBJS=(1 2 3 5 10 25 50 100 125 150)
  mPH=($(seq     1 25000 500000))
  xPH=($(seq 25000 25000 500000))
  amps1=($(seq    0 .025 .475))
  amps2=($(seq .025 .025 .50))
  DISTRIBUTIONS=(mixed)
  FILL_RADIUS=0.66
fi

BINS=$(( ${#DISTRIBUTIONS[@]} * ${#mPH[@]} * ${#amps1[@]} * ${#OBJS[@]} ))
TOTAL_SAMPLES=$(( $BINS * $SAMPLES_PER_BIN ))
TOTAL_JOBS=$(( $TOTAL_SAMPLES / $SAMPLES_PER_JOB ))

printf "DISTRIBUTIONS: [ %'d ] bins \n" ${#DISTRIBUTIONS[@]}
printf "PHOTONS:       [ %'d ] bins \n" ${#mPH[@]}
printf "AMPLITUDES:    [ %'d ] bins \n" ${#amps1[@]}
printf "BEADS:         [ %'d ] bins \n" ${#OBJS[@]}
printf "SAMPLES:       [ %'d bins x %'d samples ] = %'d samples \n" $BINS $SAMPLES_PER_BIN $TOTAL_SAMPLES

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

#            if [ $HANDLER = 'lsf' ];then
#                while [ $(bjobs -u $USER | wc -l) -gt 25000 ]
#                do
#                  sleep 10s
#                done
#
#            elif [ $HANDLER = 'slurm' ]; then
#                while [ $(squeue -u $USER -h -t pending -r | wc -l) -gt 500 ]
#                do
#                  sleep 10s
#                done
#
#            else
#                sleep 10s
#
#            fi


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

            if $USE_THEORETICAL_WIDEFIELD_SIMULATOR; then
              j="${j} --use_theoretical_widefield_simulator"
            fi

            if $SKIP_PREPROCESSING; then
              j="${j} --skip_preprocessing"
            fi

            if $RAND_VSIZE; then
              j="${j} --randomize_voxel_size"
            fi

            for e in spatial_planes
            do
              j="${j} --embedding_option ${e}"
            done

            if [ "$DATASET" = "train" ];then
              for psf in ${MODALITIES[@]}
              do
                j="${j} --psf_type ${psf}"
              done
            fi

            if [ $HANDLER = 'lsf' ];then
                task="bsub"
                #task="${task} -q local "
                task="${task} -n ${CPUS}"

                JOB="${TITLE}-${DATASET}-${MODES}-${DISTRIBUTIONS[$DIST-1]}-photons${xPH[$PH-1]}-amp${amps2[$AMP-1]}-objs${OBJS[$N-1]}-iter#${S}"
                task="${task} -J ${JOB}"

                task="${task} -We ${TIMELIMIT}"
                task="${task} -o ${LOGS}/${JOB}.log"
                task="${task} \"${j}\""

                echo $task | bash
                echo "$(bjobs -u $USER -sum)"

            elif [ $HANDLER = 'slurm' ]; then
                task="/usr/bin/sbatch"
                task="${task} --qos=abc_normal --nice=1111111111"
                task="${task} --partition=abc"

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

            else
                echo $j | bash
            fi

            printf "JOBS: [ %'d / %'d ] \n" $(($TOTAL_JOBS - $JOB_COUNTER)) $TOTAL_JOBS
            printf "SAMPLES: [ %'d / %'d ] \n" $((($TOTAL_JOBS - $JOB_COUNTER) * $SAMPLES_PER_JOB)) $TOTAL_SAMPLES

        done
      done
    done
  done
done
