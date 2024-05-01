#!/bin/bash

SHAPE=64
DZ=200
DY=125
DX=125
DEFOCUS='--lls_defocus'
DEFOCUS_ONLY='--defocus_only'
NETWORK='opticalnet'
MODES=15
CLUSTER='lsf'
DEFAULT='--positional_encoding_scheme default --lr 5e-4 --wd 5e-6 --opt adamw'
LAMB='--lr 1e-3 --wd 1e-2 --opt lamb'
APPTAINER="--apptainer ../develop_TF_CUDA_12_3.sif"
H100="--partition gpu_h100 --gpus 8 --cpus 16"
A100="--partition gpu_a100 --gpus 4 --cpus 8"
BS=4096

SUBSET='10m_125nm_dataset'
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
  DATA="$DATASET/$SUBSET/train/$DIR/z$SHAPE-y$SHAPE-x$SHAPE/z$MODES"

  if [ $PTYPE = '2photon' ];then
    LAM=.920
  else
    LAM=.510
  fi
  
  CONFIG=" --psf_type ${PTYPE} --wavelength ${LAM} --network ${NETWORK} --modes ${MODES} --dataset ${DATA} --input_shape ${SHAPE} "

  python manager.py $CLUSTER $APPTAINER train.py $H100 \
  --task "$CONFIG $LAMB --batch_size $BS --patches '32-16' --repeats '3-3' --heads '6-6' --samplelimit 500000" \
  --taskname $NETWORK \
  --name new/$SUBSET/scaling/$NETWORK-$MODES-$DIR-S3216-5e5

  python manager.py $CLUSTER $APPTAINER train.py $H100 \
  --task "$CONFIG $LAMB --batch_size $BS --patches '32-16' --repeats '3-3' --heads '6-6' --samplelimit 1000000" \
  --taskname $NETWORK \
  --name new/$SUBSET/scaling/$NETWORK-$MODES-$DIR-S3216-1e6

  python manager.py $CLUSTER $APPTAINER train.py $H100 \
  --task "$CONFIG $LAMB --batch_size $BS --patches '32-16' --repeats '3-3' --heads '6-6' --samplelimit 2000000" \
  --taskname $NETWORK \
  --name new/$SUBSET/scaling/$NETWORK-$MODES-$DIR-S3216-2e6

  python manager.py $CLUSTER $APPTAINER train.py $H100 \
  --task "$CONFIG $LAMB --batch_size $BS --patches '32-16' --repeats '3-3' --heads '6-6' --samplelimit 4000000" \
  --taskname $NETWORK \
  --name new/$SUBSET/scaling/$NETWORK-$MODES-$DIR-S3216-4e6

  python manager.py $CLUSTER $APPTAINER train.py $H100 \
  --task "$CONFIG $LAMB --batch_size $BS --patches '32-16' --repeats '3-3' --heads '6-6' --samplelimit 6000000" \
  --taskname $NETWORK \
  --name new/$SUBSET/scaling/$NETWORK-$MODES-$DIR-S3216-6e6

  python manager.py $CLUSTER $APPTAINER train.py $H100 \
  --task "$CONFIG $LAMB --batch_size $BS --patches '32-16' --repeats '3-3' --heads '6-6' --samplelimit 8000000" \
  --taskname $NETWORK \
  --name new/$SUBSET/scaling/$NETWORK-$MODES-$DIR-S3216-8e6

#  python manager.py $CLUSTER $APPTAINER train.py $H100 \
#  --task "$CONFIG $LAMB --batch_size $BS --patches '32-16' --repeats '3-3' --heads '6-6'" \
#  --taskname $NETWORK \
#  --name new/$SUBSET/scaling/$NETWORK-$MODES-$DIR-S3216
#
#  python manager.py $CLUSTER $APPTAINER train.py $H100 \
#  --task "$CONFIG $LAMB --batch_size $BS --patches '32-16' --repeats '4-4' --heads '8-8'" \
#  --taskname $NETWORK \
#  --name new/$SUBSET/scaling/$NETWORK-$MODES-$DIR-B3216
#
#  python manager.py $CLUSTER $APPTAINER train.py $H100 \
#  --task "$CONFIG $LAMB --batch_size $BS --patches '32-16' --repeats '6-6' --heads '12-12'" \
#  --taskname $NETWORK \
#  --name new/$SUBSET/scaling/$NETWORK-$MODES-$DIR-M3216
#
#  python manager.py $CLUSTER $APPTAINER train.py $H100 \
#  --task "$CONFIG $LAMB --batch_size $BS --patches '32-16' --repeats '12-12' --heads '16-16'" \
#  --taskname $NETWORK \
#  --name new/$SUBSET/scaling/$NETWORK-$MODES-$DIR-L3216
#
#  python manager.py $CLUSTER $APPTAINER train.py $H100 \
#  --task "$CONFIG $LAMB --batch_size $BS --patches '32-16' --repeats '16-16' --heads '16-16'" \
#  --taskname $NETWORK \
#  --name new/$SUBSET/scaling/$NETWORK-$MODES-$DIR-H3216
#
#  python manager.py $CLUSTER $APPTAINER train.py $H100 \
#  --task "$CONFIG $LAMB --batch_size $BS --patches '32-16' --repeats '24-24' --heads '24-24'" \
#  --taskname $NETWORK \
#  --name new/$SUBSET/scaling/$NETWORK-$MODES-$DIR-G3216
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
