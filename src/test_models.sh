#!/bin/bash

DX=.108
DY=.108
DZ=.200
LAMBDA=.510
SHAPE=64
MODES=15
ROTATIONS='--digital_rotations'
ITERS=5
MAX=10000
PRETRAINED="../pretrained_models/"
PSF="widefield_lambda510"
DATA="/clusterfs/nvme/thayer/dataset/new_modalities/test/$PSF/z$DZ-y$DY-x$DX/z$SHAPE-y$SHAPE-x$SHAPE/z$MODES"
EVALSIGN="signed"

for PSF in YuMB_lambda510 widefield_lambda510 confocal_lambda510 2photon_lambda920
do
  MODEL="$PRETRAINED/opticalnet-$MODES-$PSF"

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
  --task "$MODEL.h5 --eval_sign $EVALSIGN --num_objs 5 --n_samples 5 $ROTATIONS --batch_size $BATCH modes" \
  --taskname eval_modalities \
  --name $MODEL/$EVALSIGN/eval_modalities


  BATCH=2048
  for NA in 1.
  do
    python manager.py slurm test.py --partition abc_a100 --mem '250GB' --cpus 8 --gpus 2 \
    --task "$MODEL.h5 --datadir $DATA --na $NA --batch_size $BATCH --n_samples $MAX --niter $ITERS --eval_sign $EVALSIGN $ROTATIONS snrheatmap" \
    --taskname $NA \
    --name $MODEL/$EVALSIGN/snrheatmaps/mode-$PSF

    python manager.py slurm test.py --partition abc_a100 --mem '250GB' --cpus 8 --gpus 2 \
    --task "$MODEL.h5 --datadir $DATA --na $NA --batch_size $BATCH --n_samples $MAX --niter 1 --n_samples $MAX --eval_sign $EVALSIGN $ROTATIONS densityheatmap" \
    --taskname $NA \
    --name $MODEL/$EVALSIGN/densityheatmaps/mode-$PSF
  done

done



#declare -a PSFS=( "YuMB" "ACHex" "Gaussian" "MBHex" "MBSq" "Sinc" "v2Hex" "v2HexRect" "widefield")
#declare -a PSFS_PATHS=(
#  "../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat"
#  "../lattice/ACHex_NAexc0p40_NAsigma0p075_annulus0p6-0p2_crop0p1_FWHM52p0.mat"
#  "../lattice/Gaussian_NAexc0p21_NAsigma0p21_annulus0p4-0p2_crop0p1_FWHM51p0.mat"
#  "../lattice/MBHex_NAexc0p43_annulus0p47_0p40_crop0p08_FWHM48p0.mat"
#  "../lattice/MBSq_NAexc0p30_annulus0p375-0p225_FWHM48p5.mat"
#  "../lattice/Sinc_by_lateral_SW_NAexc0p32_NAsigma5p0_annulus0p4-0p2_realSLM_FWHM51p5.mat"
#  "../lattice/v2Hex_NAexc0p50_NAsigma0p075_annulus0p60-0p40_FWHM53p0.mat"
#  "../lattice/v2HexRect_NAexc0p50_NAsigma0p15_annulus0p60-0p40_FWHM_56p0.mat"
#  "widefield"
#)
#
#for S in `seq 1 ${#PSFS[@]}`
#do
#  PSF_TYPE="${PSFS_PATHS[$S-1]}"
#  PTYPE="${PSFS[$S-1]}"
#
#  BATCH=2048
#  for NA in 1.
#  do
#    python manager.py slurm test.py --partition abc_a100 --mem '500GB' --cpus 16 --gpus 4 \
#    --task "$MODEL.h5 --datadir $DATA --psf_type $PSF_TYPE --na $NA --batch_size $BATCH --n_samples $MAX --niter $ITERS --eval_sign $EVALSIGN $ROTATIONS snrheatmap" \
#    --taskname $NA \
#    --name $MODEL/$EVALSIGN/snrheatmaps/mode-$PTYPE
#
#    python manager.py slurm test.py --partition abc_a100 --mem '500GB' --cpus 16 --gpus 4 \
#    --task "$MODEL.h5 --datadir $DATA --psf_type $PSF_TYPE --na $NA --batch_size $BATCH --n_samples $MAX --niter 1 --n_samples $MAX --eval_sign $EVALSIGN $ROTATIONS densityheatmap" \
#    --taskname $NA \
#    --name $MODEL/$EVALSIGN/densityheatmaps/mode-$PTYPE
#
#    python manager.py slurm test.py --partition abc_a100 --mem '500GB' --cpus 16 --gpus 4 \
#    --task "$MODEL.h5 --datadir $DATA --psf_type $PSF_TYPE --na $NA --batch_size $BATCH --n_samples $MAX --niter $ITERS --eval_sign $EVALSIGN $ROTATIONS iterheatmap" \
#    --taskname $NA \
#    --name $MODEL/$EVALSIGN/iterheatmaps/mode-$PTYPE
#  done
#done
