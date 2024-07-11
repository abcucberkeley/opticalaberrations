#!/bin/bash

DX=97
DY=97
DZ=200
SHAPE=64
MODES=15
ROTATIONS='--digital_rotations'
ITERS=3
MAX=10000
OUTDIR='../evaluations'
PRETRAINED="../pretrained_models"
DATASET="97nm_dataset_extended"
EVALSIGN="signed"
NA=1.0
ABC_A100_NODES=( "g0003.abc0" "g0004.abc0" "g0005.abc0" "g0006.abc0" )
CLUSTER='slurm'
TIMELIMIT='24:00:00'  #hh:mm:ss
SKIP_REMOVE_BACKGROUND=false
APPTAINER="--apptainer ../tensorflow_TF_CUDA_12_3.sif"
ESTIMATED_OBJECT_GAUSSIAN_SIGMA=0
DENOISER='../pretrained_models/denoise/20231107_simulatedBeads_v3_32_64_64/'
BATCH=2048 #-1

TRAINED_MODELS=(
  "opticalnet-T"
  "opticalnet-S"
  "opticalnet-B"
  "opticalnet-L"
  "opticalnet-H"
  "baseline-T"
  "baseline-S"
  "baseline-B"
  "baseline-L"
  "vit-S32"
  "vit-B32"
  "vit-L32"
  "vit-S16"
  "vit-B16"
)

for M in ${TRAINED_MODELS[@]}
do
    MODEL="$PRETRAINED/$M"

    if [[ $M == *"YuMB"* ]];then
      declare -a PSFS=(
        "YuMB ../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat"
        "YuMB5 ../lattice/YuMB_NAlattice0p5_NAAnnulusMax0p40_NAsigma0p1.mat"
        #"Gaussian ../lattice/Gaussian_NAexc0p21_NAsigma0p21_annulus0p4-0p2_crop0p1_FWHM51p0.mat"
        #"MBSq ../lattice/MBSq_NAexc0p30_annulus0p375-0p225_FWHM48p5.mat"
        #"Sinc ../lattice/Sinc_by_lateral_SW_NAexc0p32_NAsigma5p0_annulus0p4-0p2_realSLM_FWHM51p5.mat"
      )

    elif [[ $M == *"v2Hex"* ]];then
      declare -a PSFS=(
        "v2Hex ../lattice/v2Hex_NAexc0p50_NAsigma0p075_annulus0p60-0p40_FWHM53p0.mat"
        #"ACHex ../lattice/ACHex_NAexc0p40_NAsigma0p075_annulus0p6-0p2_crop0p1_FWHM52p0.mat"
        #"MBHex ../lattice/MBHex_NAexc0p43_annulus0p47_0p40_crop0p08_FWHM48p0.mat"
        #"v2HexRect ../lattice/v2HexRect_NAexc0p50_NAsigma0p15_annulus0p60-0p40_FWHM_56p0.mat"
      )

    elif [[ $M == *"widefield"* ]];then
      declare -a PSFS=( "widefield widefield" )

    elif [[ $M == *"2photon"* ]];then
      declare -a PSFS=( "2photon 2photon" )

    elif [[ $M == *"confocal"* ]];then
      declare -a PSFS=( "confocal confocal" )

    else
      declare -a PSFS=( "YuMB ../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat" )

    fi

    for S in `seq 1 ${#PSFS[@]}`
    do
      NODE="${ABC_A100_NODES[$(( $S % ${#ABC_A100_NODES[@]} ))]}"  #--nodelist $NODE
      set -- ${PSFS[$S-1]}
      PTYPE=$1
      PSF_TYPE=$2

      echo Eval $M on $PTYPE
      echo

      if [[ $PTYPE = "2photon" ]];then
        LAM=.920
      else
        LAM=.510
      fi


      for (( i=1; i<=$ITERS; i++ ))
      do
        if [[ $CLUSTER = 'slurm' ]];then
          DATA="/clusterfs/nvme/thayer/dataset/$DATASET/test/YuMB_lambda510/z$DZ-y$DY-x$DX/z$SHAPE-y$SHAPE-x$SHAPE/z$MODES"
          JOB="test.py --timelimit $TIMELIMIT --dependency singleton --partition abc_a100 --mem=500GB --cpus 16 --gpus 4 --exclusive"
        else
          DATA="/groups/betzig/betziglab/thayer/dataset/$DATASET/test/YuMB_lambda510/z$DZ-y$DY-x$DX/z$SHAPE-y$SHAPE-x$SHAPE/z$MODES"
          JOB="test.py --timelimit $TIMELIMIT --dependency singleton --partition gpu_a100 --cpus 8 --gpus 4"
        fi

        for SIM in '' #'--use_theoretical_widefield_simulator'
        do  
          for PREP in '' #'--skip_remove_background'
          do
              CONFIG="${SIM} ${PREP} ${ROTATIONS}"
              CONFIG="${CONFIG} --simulate_samples"
              CONFIG="${CONFIG} --batch_size ${BATCH}"
              CONFIG="${CONFIG} --n_samples ${MAX}"
              CONFIG="${CONFIG} --datadir ${DATA}"
              CONFIG="${CONFIG} --niter ${i}"
              CONFIG="${CONFIG} --wavelength ${LAM}"
              CONFIG="${CONFIG} --psf_type ${PSF_TYPE}"
              CONFIG="${CONFIG} --na ${NA}"
              CONFIG="${CONFIG} --eval_sign ${EVALSIGN}"
              CONFIG="${CONFIG} --estimated_object_gaussian_sigma ${ESTIMATED_OBJECT_GAUSSIAN_SIGMA}"

              if [[ $M == *"denoise"* ]]; then
                CONFIG="${CONFIG} --denoiser ${DENOISER}"
              fi

              #python manager.py $CLUSTER $APPTAINER $JOB \
              #--task "${MODEL}.h5 --num_beads 1 --simulate_psf_only ${CONFIG} snrheatmap" \
              #--taskname na_$NA \
              #--name ${OUTDIR}/${DATASET}${SIM}${PREP}/${M}/${EVALSIGN}/snrheatmaps/mode-${PTYPE}/psf

              python manager.py $CLUSTER $APPTAINER $JOB \
              --task "${MODEL}.h5 --num_beads 1 ${CONFIG} snrheatmap" \
              --taskname na_$NA \
              --name ${OUTDIR}/${DATASET}${SIM}${PREP}/${M}/${EVALSIGN}/snrheatmaps/mode-${PTYPE}/beads-1

              #python manager.py $CLUSTER $APPTAINER $JOB \
              #--task "${MODEL}.h5  ${CONFIG} densityheatmap" \
              #--taskname na_$NA \
              #--name ${OUTDIR}/${DATASET}${SIM}${PREP}/${M}/${EVALSIGN}/densityheatmaps/mode-${PTYPE}

              #python manager.py $CLUSTER $APPTAINER $JOB \
              #--task "${MODEL}.h5  --num_beads 1 ${CONFIG} objectsizeheatmap" \
              #--taskname na_$NA \
              #--name ${OUTDIR}/${DATASET}${SIM}${PREP}/${M}/${EVALSIGN}/objectsizeheatmaps/mode-${PTYPE}

              #python manager.py $CLUSTER $APPTAINER $JOB \
              #--task "${MODEL}.h5 $CONFIG snrheatmap" \
              #--taskname na_$NA \
              #--name ${OUTDIR}/${DATASET}${SIM}${PREP}/${M}/${EVALSIGN}/snrheatmaps/mode-${PTYPE}/beads

              echo
          done
        done
      done

      echo '----------------'
      echo

    done


  #  BATCH=128
  #  python manager.py slurm test.py --partition abc --constraint 'titan' --mem '125GB' --cpus 5 --gpus 1 \
  #  --task "$MODEL.h5 --eval_sign $EVALSIGN $ROTATIONS --batch_size $BATCH random" \
  #  --taskname random \
  #  --name $MODEL/$EVALSIGN/samples
  #
  #  python manager.py slurm test.py --partition abc --constraint 'titan' --mem '125GB' --cpus 5 --gpus 1 \
  #  --task "$MODEL.h5 --eval_sign $EVALSIGN $ROTATIONS --batch_size $BATCH modes" \
  #  --taskname evalmodes \
  #  --name $MODEL/$EVALSIGN/evalmodes/'num_objs_1'
  #
  #  python manager.py slurm test.py --partition abc --constraint 'titan' --mem '125GB' --cpus 5 --gpus 1 \
  #  --task "$MODEL.h5 --eval_sign $EVALSIGN --num_objs 5 --n_samples 5 $ROTATIONS --batch_size $BATCH modes" \
  #  --taskname evalmodes \
  #  --name $MODEL/$EVALSIGN/evalmodes/'num_objs_5'
  #
  #  python manager.py slurm test.py --partition abc --constraint 'titan' --mem '125GB' --cpus 5 --gpus 1 \
  #  --task "$MODEL.h5 --eval_sign $EVALSIGN $ROTATIONS --batch_size $BATCH modalities" \
  #  --taskname modalities \
  #  --name $MODEL/$EVALSIGN/modalities

  done
