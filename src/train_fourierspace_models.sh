#!/bin/bash
#--partition gpu_a100 --gpus 4 --cpus 8 \
#--partition gpu_rtx8000 --gpus 8 --cpus 16 \

SHAPE=64
DZ=200
DY=125
DX=125
DEFOCUS='--lls_defocus'
DEFOCUS_ONLY='--defocus_only'
NETWORK=opticalnet
MODES=15
CLUSTER='lsf'
DEFAULT='--positional_encoding_scheme default --fixed_precision --batch_size 1024 --lr 5e-4 --wd 5e-6 --opt adamw'

SUBSET='125nm_dataset'
if [ $CLUSTER = 'slurm' ];then
  DATASET="/clusterfs/nvme/thayer/dataset"
else
  DATASET="/groups/betzig/betziglab/thayer/dataset"
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

  CONFIG=" --psf_type ${PTYPE} --wavelength ${LAM} --network ${NETWORK} --modes ${MODES} --dataset ${DATA} --input_shape ${SHAPE} "

  python manager.py $CLUSTER train.py --partition gpu_a100 --gpus 4 --cpus 8 \
  --task "$CONFIG $DEFAULT" \
  --taskname $NETWORK \
  --name new/$SUBSET/$NETWORK-$MODES-$DIR-default

  python manager.py $CLUSTER train.py --partition gpu_a100 --gpus 4 --cpus 8 \
  --task "$CONFIG --fixed_precision --batch_size 1024 --lr 5e-4 --wd 5e-6 --opt adamw" \
  --taskname $NETWORK \
  --name new/$SUBSET/$NETWORK-$MODES-$DIR-adamw-fp

  python manager.py $CLUSTER train.py --partition gpu_a100 --gpus 4 --cpus 8 \
  --task "$CONFIG --batch_size 2048 --lr 5e-4 --wd 5e-6 --opt adamw" \
  --taskname $NETWORK \
  --name new/$SUBSET/$NETWORK-$MODES-$DIR-adamw-amp

  python manager.py $CLUSTER train.py --partition gpu_a100 --gpus 4 --cpus 8 \
  --task "$CONFIG --batch_size 2048 --lr 1e-3 --wd 1e-2 --opt lamb" \
  --taskname $NETWORK \
  --name new/$SUBSET/$NETWORK-$MODES-$DIR-lamb-amp

  python manager.py $CLUSTER train.py --partition gpu_a100 --gpus 4 --cpus 8 \
  --task "$CONFIG --depth_scalar 1.5 --batch_size 1024 --lr 1e-3 --wd 1e-2 --opt lamb" \
  --taskname $NETWORK \
  --name new/$SUBSET/$NETWORK-$MODES-$DIR-lamb-amp-1p5x

  python manager.py $CLUSTER train.py --partition gpu_a100 --gpus 4 --cpus 8 \
  --task "$CONFIG --depth_scalar 2.0 --batch_size 1024 --lr 1e-3 --wd 1e-2 --opt lamb" \
  --taskname $NETWORK \
  --name new/$SUBSET/$NETWORK-$MODES-$DIR-lamb-amp-2x

done


#### FOR ABC CLUSTER

#--partition dgx --mem '1950GB' --gpus 8 --cpus 128 \
#--partition abc_a100 --mem '500GB' --nodes 1 --gpus 4 --cpus 16 \
#--partition abc --constraint 'titan' --mem '500GB' --nodes 3 --gpus 4 --cpus 20 \

#  python multinode_manager.py train.py --partition abc_a100 --mem '500GB' --nodes 1 --gpus 4 --cpus 16 \
#  --task "--multinode $CONFIG" \
#  --taskname $NETWORK \
#  --name new/$SUBSET/$NETWORK-$MODES-$DIR

#python manager.py slurm train.py --partition dgx --mem '1950GB' --gpus 8 --cpus 128 \
#  --task "$CONFIG" \
#  --taskname $NETWORK \
#  --name new/$SUBSET/$NETWORK-$MODES-$DIR
