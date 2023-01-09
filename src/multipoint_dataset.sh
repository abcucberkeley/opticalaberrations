#!/bin/bash

HANDLER=slurm
ENV=~/anaconda3/envs/deep/bin/python
NODES='all'

#PSF_TYPE='widefield'
#xVOXEL=.15
#yVOXEL=.15
#zVOXEL=.6
#LAMBDA=.605
#NA=1.0

#PSF_TYPE='confocal'
#xVOXEL=.1
#yVOXEL=.1
#zVOXEL=.5
#LAMBDA=.920
#NA=1.0

#PSF_TYPE='../lattice/HexRect_NAlattice0.25_NAAnnulusMax0.60_NAsigma0.08.mat'
PSF_TYPE='../lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat'
xVOXEL=.108
yVOXEL=.108
zVOXEL=.200
LAMBDA=.510
NA=1.0
ALPHA='abs'
PHI='angle'
CPUS=1

SHAPE=64
RCROP=32

MODES=15
TITLE='new_embeddings'
DATASET='test'

MODE_DIST='pyramid'
OUTDIR="/clusterfs/nvme/thayer/dataset/${TITLE}/${DATASET}"

if [ "$DATASET" = "train" ];then
  TYPE='--emb'
  ITERS=100
  SAMPLES_PER_BIN=500
  OBJS=(1 2 5 10 25)
  mPSNR=($(seq 11 10 51))
  xPSNR=($(seq 20 10 60))
  SAMPLES=($(seq 1 $ITERS $SAMPLES_PER_BIN))
  amps1=($(seq 0 .01 .29))
  amps2=($(seq .01 .01 .3))

  #  difractionlimit=($(seq 0 .005 .05))
  #  small=($(seq .05 .0025 .1))
  #  large=($(seq .1 .01 .25))
  #  amps=( "${difractionlimit[@]}" "${small[@]}" "${large[@]}" )
  #  echo ${amps[@]}
  #  echo ${#amps[@]}
  #  amps1=( "${difractionlimit[@]}" "${small[@]}" "${large[@]:0:${#large[@]}-1}" )
  #  amps2=( "${difractionlimit[@]:1}" "${small[@]}" "${large[@]}" )

else
  TYPE=''
  ITERS=25
  SAMPLES_PER_BIN=25
  OBJS=(1 2 3 4 5 10 15 20 25 30)
  mPSNR=($(seq 1 10 91))
  xPSNR=($(seq 10 10 100))
  SAMPLES=($(seq 1 $ITERS $SAMPLES_PER_BIN))
  amps1=($(seq 0 .01 .49))
  amps2=($(seq .01 .01 .50))
fi


for DIST in single bimodal powerlaw dirichlet
do
  for SNR in `seq 1 ${#xPSNR[@]}`
  do
    for AMP in `seq 1 ${#amps1[@]}`
    do
      for N in `seq 1 ${#OBJS[@]}`
      do
          for S in `seq 1 ${#SAMPLES[@]}`
          do
            while [ $(squeue -u thayeralshaabi -h -t pending -r | wc -l) -gt 300 ]
            do
              sleep 10s
            done

            j="${ENV} multipoint_dataset.py ${TYPE}"
            j="${j} --npoints ${OBJS[$N-1]}"
            j="${j} --psf_type ${PSF_TYPE}"
            j="${j} --alpha_val ${ALPHA}"
            j="${j} --phi_val ${PHI}"
            j="${j} --dist ${DIST}"
            j="${j} --mode_dist ${MODE_DIST}"
            j="${j} --iters ${ITERS}"
            j="${j} --signed"
            j="${j} --rotate"
            j="${j} --noise"
            j="${j} --normalize"
            j="${j} --gamma .75"
            j="${j} --outdir ${OUTDIR}"
            j="${j} --filename ${SAMPLES[$S-1]}"
            j="${j} --modes ${MODES}"
            j="${j} --input_shape ${SHAPE}"
            j="${j} --min_psnr ${mPSNR[$SNR-1]}"
            j="${j} --max_psnr ${xPSNR[$SNR-1]}"
            j="${j} --min_amplitude ${amps1[$AMP-1]}"
            j="${j} --max_amplitude ${amps2[$AMP-1]}"
            j="${j} --x_voxel_size ${xVOXEL}"
            j="${j} --y_voxel_size ${yVOXEL}"
            j="${j} --z_voxel_size ${zVOXEL}"
            j="${j} --na_detection ${NA}"
            j="${j} --lam_detection ${LAMBDA}"
            j="${j} --cpu_workers ${CPUS}"

            for e in principle_planes
            do
              j="${j} --embedding_option ${e}"
            done

            if [ "$DATASET" = "train" ];then
              j="${j} --random_crop ${RCROP}"
            fi

            task="/usr/bin/sbatch"
            task="${task} --qos=abc_normal --nice=1111111111"

            if [ "$NODES" = "all" ];then
              if [ $(squeue -u thayeralshaabi -h -t pending -r -p dgx | wc -l) -lt 128 ];then
                task="${task} --partition=dgx"
              elif [ $(squeue -u thayeralshaabi -h -t pending -r -p abc_a100 | wc -l) -lt 64 ];then
                task="${task} --partition=abc_a100"
              else
                task="${task} --partition=abc"
              fi
            else
              task="${task} --partition=abc"
            fi

            task="${task} --cpus-per-task=${CPUS}"
            task="${task} --mem=15G"
            task="${task} --job-name=psnr#${SNR}-amp#${AMP}-iter#${S}"
            task="${task} --time=1:00:00"
            task="${task} --export=ALL"
            task="${task} --wrap=\"${j}\""
            echo $task | bash

            echo "DGX : R[$(squeue -u thayeralshaabi -h -t running -r -p dgx | wc -l)], P[$(squeue -u thayeralshaabi -h -t pending -r -p dgx | wc -l)]"
            echo "A100: R[$(squeue -u thayeralshaabi -h -t running -r -p abc_a100 | wc -l)], P[$(squeue -u thayeralshaabi -h -t pending -r -p abc_a100 | wc -l)]"
            echo "ABC : R[$(squeue -u thayeralshaabi -h -t running -r -p abc | wc -l)], P[$(squeue -u thayeralshaabi -h -t pending -r -p abc | wc -l)]"

          done
        done
    done
  done
done
