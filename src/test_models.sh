#!/bin/bash

DX=108
DY=108
DZ=200
SHAPE=64
MODES=15
ROTATIONS='--digital_rotations'
ITERS=3
MAX=10000
PRETRAINED="../pretrained_models/"
DATASET="new_modalities"
DATA="/clusterfs/nvme/thayer/dataset/$DATASET/test/YuMB_lambda510/z$DZ-y$DY-x$DX/z$SHAPE-y$SHAPE-x$SHAPE/z$MODES"
EVALSIGN="signed"
NA=1.0
A100_NODES=( "g0003.abc0" "g0004.abc0" "g0005.abc0" "g0006.abc0" )
GPUS=4
CPUS=16

TRAINED_MODELS=(
  "YuMB_lambda510-nostem"
  "YuMB_lambda510-nostem-radial-encoding-p16"
  "YuMB_lambda510-nostem-radial-encoding-p4"
  "YuMB_lambda510-nostem-radial-encoding-p1-round2"
#  "YuMB_lambda510"
#  "v2Hex_lambda510"
  "2photon_lambda920"
  "confocal_lambda510"
  "widefield_lambda510"
)

for M in ${TRAINED_MODELS[@]}
do
  MODEL="$PRETRAINED/opticalnet-$MODES-$M"

  BATCH=128
  python manager.py slurm test.py --partition abc --constraint 'titan' --mem '125GB' --cpus 5 --gpus 1 \
  --task "$MODEL.h5 --eval_sign $EVALSIGN $ROTATIONS --batch_size $BATCH random" \
  --taskname random \
  --name $MODEL/$EVALSIGN/samples

  python manager.py slurm test.py --partition abc --constraint 'titan' --mem '125GB' --cpus 5 --gpus 1 \
  --task "$MODEL.h5 --eval_sign $EVALSIGN $ROTATIONS --batch_size $BATCH modes" \
  --taskname evalmodes \
  --name $MODEL/$EVALSIGN/evalmodes/'num_objs_1'

  python manager.py slurm test.py --partition abc --constraint 'titan' --mem '125GB' --cpus 5 --gpus 1 \
  --task "$MODEL.h5 --eval_sign $EVALSIGN --num_objs 5 --n_samples 5 $ROTATIONS --batch_size $BATCH modes" \
  --taskname evalmodes \
  --name $MODEL/$EVALSIGN/evalmodes/'num_objs_5'

  python manager.py slurm test.py --partition abc --constraint 'titan' --mem '125GB' --cpus 5 --gpus 1 \
  --task "$MODEL.h5 --eval_sign $EVALSIGN $ROTATIONS --batch_size $BATCH modalities" \
  --taskname modalities \
  --name $MODEL/$EVALSIGN/modalities

  BATCH=$(( 896 * $GPUS ))

  if [ "${M:0:4}" = YuMB ];then
    declare -a PSFS=( "YuMB" "Gaussian" "MBSq" "Sinc" )
    declare -a PATHS=(
      "../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat"
      "../lattice/Gaussian_NAexc0p21_NAsigma0p21_annulus0p4-0p2_crop0p1_FWHM51p0.mat"
      "../lattice/MBSq_NAexc0p30_annulus0p375-0p225_FWHM48p5.mat"
      "../lattice/Sinc_by_lateral_SW_NAexc0p32_NAsigma5p0_annulus0p4-0p2_realSLM_FWHM51p5.mat"
      "widefield"
    )
  elif [ "${M:0:5}" == v2Hex ];then
    declare -a PSFS=( "v2Hex" "ACHex" "MBHex" "v2HexRect" )
    declare -a PATHS=(
      "../lattice/v2Hex_NAexc0p50_NAsigma0p075_annulus0p60-0p40_FWHM53p0.mat"
      "../lattice/ACHex_NAexc0p40_NAsigma0p075_annulus0p6-0p2_crop0p1_FWHM52p0.mat"
      "../lattice/MBHex_NAexc0p43_annulus0p47_0p40_crop0p08_FWHM48p0.mat"
      "../lattice/v2HexRect_NAexc0p50_NAsigma0p15_annulus0p60-0p40_FWHM_56p0.mat"
    )
  elif [ $M = "widefield" ];then
    declare -a PSFS=( "widefield" )
    declare -a PATHS=( "widefield" )
  else
    declare -a PSFS=( "confocal" "2photon" )
    declare -a PATHS=( "confocal" "2photon" )
  fi

  for S in `seq 1 ${#PSFS[@]}`
  do
    NODE="${A100_NODES[$(( $S % ${#A100_NODES[@]} ))]}"  #--nodelist $NODE
    PTYPE="${PSFS[$S-1]}"
    PSF_TYPE="${PATHS[$S-1]}"

    echo Eval $M on $PTYPE

    if [ $PTYPE = '2photon' ];then
      LAM=.920
    else
      LAM=.510
    fi

    for (( i=1; i<=$ITERS; i++ ))
    do
      python manager.py slurm test.py --dependency singleton --partition abc_a100 --mem '500GB' --cpus $CPUS --gpus $GPUS \
      --task "$MODEL.h5 --niter $i --num_beads 1 --datadir $DATA --wavelength $LAM --psf_type $PSF_TYPE --na $NA --batch_size $BATCH --eval_sign $EVALSIGN $ROTATIONS snrheatmap" \
      --taskname $NA \
      --name $MODEL/$EVALSIGN/snrheatmaps/mode-$PTYPE/beads-1

      python manager.py slurm test.py --dependency singleton --partition abc_a100 --mem '500GB' --cpus $CPUS --gpus $GPUS \
      --task "$MODEL.h5 --niter $i --datadir $DATA --n_samples $MAX --wavelength $LAM --psf_type $PSF_TYPE --na $NA --batch_size $BATCH --eval_sign $EVALSIGN $ROTATIONS snrheatmap" \
      --taskname $NA \
      --name $MODEL/$EVALSIGN/snrheatmaps/mode-$PTYPE/beads
    done

    python manager.py slurm test.py --partition abc_a100 --mem '500GB' --cpus $CPUS --gpus $GPUS \
    --task "$MODEL.h5 --datadir $DATA --wavelength $LAM --psf_type $PSF_TYPE --na $NA --batch_size $BATCH --niter 1 --eval_sign $EVALSIGN $ROTATIONS densityheatmap" \
    --taskname $NA \
    --name $MODEL/$EVALSIGN/densityheatmaps/mode-$PTYPE

    echo '----------------'
  done
done


#python manager.py slurm benchmark.py --partition abc --constraint 'titan' --mem '500GB' --cpus 20 --gpus $GPUS \
#--task "phasenet_heatmap $DATA --no_beads --n_samples $MAX --eval_sign $EVALSIGN" \
#--taskname $NA \
#--name ../src/phasenet_repo/$EVALSIGN/psf
#
#
#python manager.py slurm benchmark.py --partition abc --constraint 'titan' --mem '500GB' --cpus 20 --gpus $GPUS \
#--task "phasenet_heatmap $DATA --n_samples $MAX --eval_sign $EVALSIGN" \
#--taskname $NA \
#--name ../src/phasenet_repo/$EVALSIGN/bead
