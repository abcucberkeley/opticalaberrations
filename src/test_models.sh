#!/bin/bash

DX=108
DY=108
DZ=200
SHAPE=64
MODES=15
ROTATIONS='--digital_rotations'
ITERS=5
MAX=10000
PRETRAINED="../pretrained_models/"
DATASET="new_modalities"
DATA="/clusterfs/nvme/thayer/dataset/$DATASET/test/YuMB_lambda510/z$DZ-y$DY-x$DX/z$SHAPE-y$SHAPE-x$SHAPE/z$MODES"
EVALSIGN="signed"
NA=1.0

for M in YuMB_lambda510 v2Hex_lambda510
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

  BATCH=2048
  if [ $M = 'YuMB_lambda510' ];then
    declare -a PSFS=( "YuMB" "Gaussian" "MBSq" "Sinc" "widefield" )
    declare -a PATHS=(
      "../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat"
      "../lattice/Gaussian_NAexc0p21_NAsigma0p21_annulus0p4-0p2_crop0p1_FWHM51p0.mat"
      "../lattice/MBSq_NAexc0p30_annulus0p375-0p225_FWHM48p5.mat"
      "../lattice/Sinc_by_lateral_SW_NAexc0p32_NAsigma5p0_annulus0p4-0p2_realSLM_FWHM51p5.mat"
      "widefield"
    )

  else
    declare -a PSFS=( "v2Hex" "ACHex" "MBHex" "v2HexRect")
    declare -a PATHS=(
      "../lattice/v2Hex_NAexc0p50_NAsigma0p075_annulus0p60-0p40_FWHM53p0.mat"
      "../lattice/ACHex_NAexc0p40_NAsigma0p075_annulus0p6-0p2_crop0p1_FWHM52p0.mat"
      "../lattice/MBHex_NAexc0p43_annulus0p47_0p40_crop0p08_FWHM48p0.mat"
      "../lattice/v2HexRect_NAexc0p50_NAsigma0p15_annulus0p60-0p40_FWHM_56p0.mat"
    )
  fi

  for S in `seq 1 ${#PSFS[@]}`
  do
    PTYPE="${PSFS[$S-1]}"
    PSF_TYPE="${PATHS[$S-1]}"

    if [ $PTYPE = '2photon' ];then
      LAM=.920
    else
      LAM=.510
    fi

    python manager.py slurm test.py --partition abc_a100 --mem '500GB' --cpus 16 --gpus 4 \
    --task "$MODEL.h5 --datadir $DATA --wavelength $LAM --psf_type $PSF_TYPE --na $NA --batch_size $BATCH --n_samples $MAX --niter $ITERS --eval_sign $EVALSIGN $ROTATIONS snrheatmap" \
    --taskname $NA \
    --name $MODEL/$EVALSIGN/snrheatmaps/mode-$PTYPE

    python manager.py slurm test.py --partition abc_a100 --mem '500GB' --cpus 16 --gpus 4 \
    --task "$MODEL.h5 --datadir $DATA --wavelength $LAM --psf_type $PSF_TYPE --na $NA --batch_size $BATCH --n_samples $MAX --niter 1 --eval_sign $EVALSIGN $ROTATIONS densityheatmap" \
    --taskname $NA \
    --name $MODEL/$EVALSIGN/densityheatmaps/mode-$PTYPE
  done
done


python manager.py slurm test.py --partition abc --constraint 'titan' --mem '125GB' --cpus 5 --gpus 1 \
--task "$PRETRAINED/opticalnet-$MODES-$M.h5 --no_beads --datadir $DATA --n_samples $MAX --niter $ITERS --eval_sign $EVALSIGN phasenet" \
--taskname $NA \
--name phasenetrepo/$EVALSIGN/psf


python manager.py slurm test.py --partition abc --constraint 'titan' --mem '125GB' --cpus 5 --gpus 1 \
--task "$PRETRAINED/opticalnet-$MODES-$M.h5 --datadir $DATA --n_samples $MAX --niter $ITERS --eval_sign $EVALSIGN phasenet" \
--taskname $NA \
--name phasenetrepo/$EVALSIGN/bead
