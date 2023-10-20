#!/bin/bash
#--partition gpu_a100 --gpus 4 --cpus 8 \
#--partition gpu_rtx8000 --gpus 8 --cpus 16 \


DEPTH=1.0
SHAPE=64
MAXAMP=1
DZ=200
DY=108
DX=108
SAM='--sam'
RADIAL_ENCODING='--radial_encoding'
RADIAL_ENCODING_PERIOD='--radial_encoding_period'
RADIAL_ENCODING_ORDER='--radial_encoding_nth_order'
POSITIONAL_ENCODING_SCHEME='--positional_encoding_scheme'
NO_PHASE='--no_phase'
DEFOCUS='--lls_defocus'
DEFOCUS_ONLY='--defocus_only'
EMB="spatial_planes"
BATCH=2048
NETWORK=opticalnet
MODES=15
WARMUP=25
EPOCHS=500
NODES=1
CLUSTER=lsf
INCREASE_DROPOUT='--increase_dropout_depth'
DECREASE_DROPOUT='--decrease_dropout_depth'


SUBSET='new_modalities'
if [ $CLUSTER = 'slurm' ];then
  DATASET="/clusterfs/nvme/thayer/dataset"
else
  DATASET="/groups/betzig/betziglab/thayer/dataset/"
fi

declare -a PSF_DATASETS=(
  "YuMB_lambda510"
#  "v2Hex_lambda510"
#  "widefield_lambda510"
#  "confocal_lambda510"
#  "2photon_lambda920"
)
declare -a PSF_TYPES=(
  "../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat"
#  "../lattice/v2Hex_NAexc0p50_NAsigma0p075_annulus0p60-0p40_FWHM53p0.mat"
#  "widefield"
#  "confocal"
#  "2photon"
)

for S in `seq 1 ${#PSF_DATASETS[@]}`
do
  DIR="${PSF_DATASETS[$S-1]}"
  PTYPE="${PSF_TYPES[$S-1]}"
  DATA="$DATASET/$SUBSET/train/$DIR/z$DZ-y$DY-x$DX/z$SHAPE-y$SHAPE-x$SHAPE/z$MODES"

  if [ $PTYPE = '2photon' ];then
    LAM=.920
  else
    LAM=.510
  fi

  for DROPOUT in $INCREASE_DROPOUT $DECREASE_DROPOUT
  do
    for OPT in lion adamw #sam
    do
      for i in "5e-4 5e-5" "5e-5 5e-4"
      do
        set -- $i
        LR=$1
        WD=$2

        if [ $OPT = 'sam' ];then
          python manager.py $CLUSTER train.py --partition gpu_a100 --gpus 4 --cpus 8 \
          --task "$DROPOUT $SAM --opt adamw --lr $LR --wd $WD $RADIAL_ENCODING $RADIAL_ENCODING_PERIOD 16 $POSITIONAL_ENCODING_SCHEME rotational_symmetry --psf_type $PTYPE --wavelength $LAM --network $NETWORK --embedding $EMB --patch_size '32-16-8-8' --modes $MODES --max_amplitude $MAXAMP --batch_size $BATCH --dataset $DATA --input_shape $SHAPE --depth_scalar $DEPTH --epochs $EPOCHS --warmup $WARMUP" \
          --taskname $NETWORK \
          --name new/$SUBSET/$NETWORK-$MODES-$DIR-$OPT-$LR-$DROPOUT
        else
          python manager.py $CLUSTER train.py --partition gpu_a100 --gpus 4 --cpus 8 \
          --task "$DROPOUT --opt $OPT --lr $LR --wd $WD $RADIAL_ENCODING $RADIAL_ENCODING_PERIOD 16 $POSITIONAL_ENCODING_SCHEME rotational_symmetry --psf_type $PTYPE --wavelength $LAM --network $NETWORK --embedding $EMB --patch_size '32-16-8-8' --modes $MODES --max_amplitude $MAXAMP --batch_size $BATCH --dataset $DATA --input_shape $SHAPE --depth_scalar $DEPTH --epochs $EPOCHS --warmup $WARMUP" \
          --taskname $NETWORK \
          --name new/$SUBSET/$NETWORK-$MODES-$DIR-$OPT-$LR-$DROPOUT
        fi

      done
    done
  done
done


#### FOR ABC CLUSTER

#--partition dgx --mem '1950GB' --gpus 8 --cpus 128 \
#--partition abc_a100 --mem '500GB' --nodes 1 --gpus 4 --cpus 16 \
#--partition abc --constraint 'titan' --mem '500GB' --nodes 3 --gpus 4 --cpus 20 \

#for S in `seq 1 ${#PSF_DATASETS[@]}`
#do
#  DIR="${PSF_DATASETS[$S-1]}"
#  PTYPE="${PSF_TYPES[$S-1]}"
#  DATA="/clusterfs/nvme/thayer/dataset/$SUBSET/train/$DIR/z$DZ-y$DY-x$DX/z$SHAPE-y$SHAPE-x$SHAPE/z$MODES"
#
#  if [ $PTYPE = '2photon' ];then
#    LAM=.920
#  else
#    LAM=.510
#  fi
#
#  python multinode_manager.py train.py --partition abc_a100 --mem '500GB' --nodes $NODES --gpus 4 --cpus 16 \
#  --task "--multinode --psf_type $PTYPE --wavelength $LAM --network $NETWORK --embedding $EMB --patch_size '32-16-8-8' --modes $MODES --max_amplitude $MAXAMP --batch_size $BATCH --dataset $DATA --input_shape $SHAPE --depth_scalar $DEPTH --epochs $EPOCHS --warmup $WARMUP" \
#  --taskname $NETWORK \
#  --name new/$SUBSET/$NETWORK-$MODES-$DIR
#done


#python manager.py slurm train.py --partition dgx --mem '1950GB' --gpus 8 --cpus 128 \
#--task "--psf_type $PSF_T --network $NETWORK --embedding $EMB --patch_size '32-16-8-8' --modes $MODES --max_amplitude $MAXAMP --batch_size $BATCH --dataset $DATA --input_shape $SHAPE --depth_scalar $DEPTH --epochs $EPOCHS --warmup $WARMUP" \
#--taskname $NETWORK \
#--name new/$SUBSET/$NETWORK-$MODES
