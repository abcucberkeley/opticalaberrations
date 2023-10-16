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
DATASET="fixed_density"
EVALSIGN="signed"
NA=1.0
ABC_A100_NODES=( "g0003.abc0" "g0004.abc0" "g0005.abc0" "g0006.abc0" )
CLUSTER=slurm

TRAINED_MODELS=(
  "spatial_planes10"
  "YuMB_lambda510-nostem"
  "YuMB_lambda510-nostem-radial-encoding-p16"
#  "v2Hex_lambda510"
#  "2photon_lambda920"
#  "confocal_lambda510"
#  "widefield_lambda510"
)


if [ $CLUSTER = 'slurm' ];then
  DATA="/clusterfs/nvme/thayer/dataset/$DATASET/test/YuMB_lambda510/z$DZ-y$DY-x$DX/z$SHAPE-y$SHAPE-x$SHAPE/z$MODES"
  PARTITION="abc_a100"
  GPUS=4
  CPUS=16
  MEM="500GB"
else
  DATA="/groups/betzig/betziglab/thayer/dataset/$DATASET/test/YuMB_lambda510/z$DZ-y$DY-x$DX/z$SHAPE-y$SHAPE-x$SHAPE/z$MODES"
  PARTITION="gpu_a100"
  GPUS=4
  CPUS=8
  MEM="320GB"
fi


for M in ${TRAINED_MODELS[@]}
do
  MODEL="$PRETRAINED/opticalnet-$MODES-$M"

  BATCH=$(( 896 * $GPUS ))

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

    if [ $PTYPE = '2photon' ];then
      LAM=.920
    else
      LAM=.510
    fi

    for (( i=1; i<=$ITERS; i++ ))
    do
      python manager.py $CLUSTER test.py --dependency singleton  --partition $PARTITION --mem $MEM --cpus $CPUS --gpus $GPUS \
      --task "$MODEL.h5 --niter $i --num_beads 1 --datadir $DATA --wavelength $LAM --psf_type $PSF_TYPE --na $NA --batch_size $BATCH --eval_sign $EVALSIGN $ROTATIONS snrheatmap" \
      --taskname $NA \
      --name $MODEL/$EVALSIGN/snrheatmaps/mode-$PTYPE/beads-1

      python manager.py $CLUSTER test.py --dependency singleton --partition $PARTITION --mem $MEM --cpus $CPUS --gpus $GPUS \
      --task "$MODEL.h5 --niter $i --datadir $DATA --wavelength $LAM --psf_type $PSF_TYPE --na $NA --batch_size $BATCH --eval_sign $EVALSIGN $ROTATIONS densityheatmap" \
      --taskname $NA \
      --name $MODEL/$EVALSIGN/densityheatmaps/mode-$PTYPE

      #python manager.py $CLUSTER test.py --dependency singleton --partition $PARTITION --mem $MEM --cpus $CPUS --gpus $GPUS \
      #--task "$MODEL.h5 --niter $i --datadir $DATA --n_samples $MAX --wavelength $LAM --psf_type $PSF_TYPE --na $NA --batch_size $BATCH --eval_sign $EVALSIGN $ROTATIONS snrheatmap" \
      #--taskname $NA \
      #--name $MODEL/$EVALSIGN/snrheatmaps/mode-$PTYPE/beads
    done

    echo '----------------'
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
