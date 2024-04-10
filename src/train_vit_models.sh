#!/bin/bash

SHAPE=64
DZ=200
DY=125
DX=125
DEFOCUS='--lls_defocus'
DEFOCUS_ONLY='--defocus_only'
NETWORK='vit'
MODES=15
CLUSTER='lsf'
DEFAULT='--positional_encoding_scheme default --lr 5e-4 --wd 5e-6 --opt adamw'
LAMB='--lr 1e-3 --wd 1e-2 --opt lamb'
APPTAINER="--apptainer ../develop_TF_CUDA_12_3.sif"
H100="--partition gpu_h100 --gpus 8 --cpus 16"
A100="--partition gpu_a100 --gpus 4 --cpus 8"
BS=4096

SUBSET='variable_object_size_fourier_filter_125nm_dataset'
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
  --task "$CONFIG $LAMB --batch_size $BS --patches '8' --repeats '12' --heads '6' --hidden_size 384" \
  --taskname $NETWORK \
  --name new/$SUBSET/vit/$NETWORK-$MODES-$DIR-S8

  python manager.py $CLUSTER $APPTAINER train.py $H100 \
  --task "$CONFIG $LAMB --batch_size $BS --patches '16' --repeats '12' --heads '6' --hidden_size 384" \
  --taskname $NETWORK \
  --name new/$SUBSET/vit/$NETWORK-$MODES-$DIR-S16

  python manager.py $CLUSTER $APPTAINER train.py $H100 \
  --task "$CONFIG $LAMB --batch_size $BS --patches '32' --repeats '12' --heads '6' --hidden_size 384" \
  --taskname $NETWORK \
  --name new/$SUBSET/vit/$NETWORK-$MODES-$DIR-S32

  python manager.py $CLUSTER $APPTAINER train.py $H100 \
  --task "$CONFIG $LAMB --batch_size $BS --patches '8' --repeats '12' --heads '12' --hidden_size 768" \
  --taskname $NETWORK \
  --name new/$SUBSET/vit/$NETWORK-$MODES-$DIR-B8

  python manager.py $CLUSTER $APPTAINER train.py $H100 \
  --task "$CONFIG $LAMB --batch_size $BS --patches '16' --repeats '12' --heads '12' --hidden_size 768" \
  --taskname $NETWORK \
  --name new/$SUBSET/vit/$NETWORK-$MODES-$DIR-B16

  python manager.py $CLUSTER $APPTAINER train.py $H100 \
  --task "$CONFIG $LAMB --batch_size $BS --patches '32' --repeats '12' --heads '12' --hidden_size 768" \
  --taskname $NETWORK \
  --name new/$SUBSET/vit/$NETWORK-$MODES-$DIR-B32

#  python manager.py $CLUSTER $APPTAINER train.py $H100 \
#  --task "$CONFIG $LAMB --batch_size $BS --patches '8' --repeats '24' --heads '16' --hidden_size 1024" \
#  --taskname $NETWORK \
#  --name new/$SUBSET/vit/$NETWORK-$MODES-$DIR-L8
#
#  python manager.py $CLUSTER $APPTAINER train.py $H100 \
#  --task "$CONFIG $LAMB --batch_size $BS --patches '16' --repeats '24' --heads '16' --hidden_size 1024" \
#  --taskname $NETWORK \
#  --name new/$SUBSET/vit/$NETWORK-$MODES-$DIR-L16
#
#  python manager.py $CLUSTER $APPTAINER train.py $H100 \
#  --task "$CONFIG $LAMB --batch_size $BS --patches '32' --repeats '24' --heads '16' --hidden_size 1024" \
#  --taskname $NETWORK \
#  --name new/$SUBSET/vit/$NETWORK-$MODES-$DIR-L32

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
