#!/bin/bash

DX=97
DY=97
DZ=200
SHAPE=64
MODES=15
ROTATIONS='--digital_rotations'
ITERS=5
MAX=10000
OUTDIR='../evaluations'
PRETRAINED="../pretrained_models"
DATASET="97nm_dataset_extended"
EVALSIGN="signed"
NA=1.0
ABC_A100_NODES=( "g0003.abc0" "g0004.abc0" "g0005.abc0" "g0006.abc0" )
CLUSTER='slurm'
TIMELIMIT='24:00:00'  #hh:mm:ss
NETWORK='opticalnet'
SKIP_REMOVE_BACKGROUND=false
APPTAINER="--apptainer ../develop_CUDA_12_3.sif"

TRAINED_MODELS=(
  "YuMB-lambda510-R1242"
  "YuMB-lambda510-R1462"
  "YuMB-lambda510-R2462"
)

for M in ${TRAINED_MODELS[@]}
do
    MODEL="$PRETRAINED/$NETWORK-$MODES-$M"

    if [ "${M:0:4}" = YuMB ];then
      declare -a PSFS=(
        "YuMB ../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat"
        #"Gaussian ../lattice/Gaussian_NAexc0p21_NAsigma0p21_annulus0p4-0p2_crop0p1_FWHM51p0.mat"
        #"MBSq ../lattice/MBSq_NAexc0p30_annulus0p375-0p225_FWHM48p5.mat"
        #"Sinc ../lattice/Sinc_by_lateral_SW_NAexc0p32_NAsigma5p0_annulus0p4-0p2_realSLM_FWHM51p5.mat"
      )

    elif [ "${M:0:5}" == v2Hex ];then
      declare -a PSFS=(
        "v2Hex ../lattice/v2Hex_NAexc0p50_NAsigma0p075_annulus0p60-0p40_FWHM53p0.mat"
        #"ACHex ../lattice/ACHex_NAexc0p40_NAsigma0p075_annulus0p6-0p2_crop0p1_FWHM52p0.mat"
        #"MBHex ../lattice/MBHex_NAexc0p43_annulus0p47_0p40_crop0p08_FWHM48p0.mat"
        #"v2HexRect ../lattice/v2HexRect_NAexc0p50_NAsigma0p15_annulus0p60-0p40_FWHM_56p0.mat"
      )

    elif [ $M = "widefield_lambda510" ];then
      declare -a PSFS=( "widefield widefield" )

    elif [ $M = "2photon_lambda920" ];then
      declare -a PSFS=( "2photon 2photon" )

    elif [ $M = "confocal_lambda510" ];then
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

      if [ $PTYPE = '2photon' ];then
        LAM=.920
      else
        LAM=.510
      fi


      for (( i=1; i<=$ITERS; i++ ))
      do
        if [ $CLUSTER = 'slurm' ];then
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
              CONFIG=" $SIM $PREP --datadir $DATA --niter $i --wavelength $LAM --psf_type $PSF_TYPE --na $NA --eval_sign $EVALSIGN $ROTATIONS "

              python manager.py ${CLUSTER} $APPTAINER $JOB \
              --task "${MODEL}.h5 --num_beads 1 $CONFIG snrheatmap" \
              --taskname na_$NA \
              --name $OUTDIR/${DATASET}${SIM}${PREP}/$NETWORK-$MODES-$M/$EVALSIGN/snrheatmaps/mode-$PTYPE/beads-1

              #python manager.py ${CLUSTER} $APPTAINER $JOB \
              #--task "${MODEL}.h5  $CONFIG densityheatmap" \
              #--taskname na_$NA \
              #--name $OUTDIR/${DATASET}${SIM}${PREP}/$NETWORK-$MODES-$M/$EVALSIGN/densityheatmaps/mode-$PTYPE

              #python manager.py ${CLUSTER} $APPTAINER $JOB \
              #--task "${MODEL}.h5 $CONFIG snrheatmap" \
              #--taskname na_$NA \
              #--name $OUTDIR/$DATASET/$NETWORK-$MODES-$M/$EVALSIGN/snrheatmaps/mode-$PTYPE/beads
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


#python manager.py slurm benchmark.py --partition abc --constraint 'titan' --mem $MEM --cpus $CPUS --gpus $GPUS \
#--task "phasenet_heatmap $DATA --no_beads --n_samples $MAX --eval_sign $EVALSIGN" \
#--taskname $NA \
#--name ../src/phasenet_repo/$EVALSIGN/psf
#
#
#python manager.py slurm benchmark.py --partition abc --constraint 'titan' --mem $MEM --cpus $CPUS --gpus $GPUS \
#--task "phasenet_heatmap $DATA --n_samples $MAX --eval_sign $EVALSIGN" \
#--taskname $NA \
#--name ../src/phasenet_repo/$EVALSIGN/bead
