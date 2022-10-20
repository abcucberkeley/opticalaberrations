#!/bin/bash

HANDLER=slurm
ENV=~/anaconda3/envs/deep/bin/python
NODES='abc'

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

DIFFICULTY='easy'
DATASET='test'
SHAPE=64
RCROP=32
OUTDIR="/clusterfs/nvme/thayer/dataset/yumb_lattice_objects/${DIFFICULTY}/${DATASET}"


if [ "$DATASET" = "train" ];then
  TYPE='--emb'
  ITERS=100
  NEIGHBORS=($(seq 1 1 5))
  mPSNR=($(seq 11 10 51))
  xPSNR=($(seq 20 10 60))

  if [ "$DIFFICULTY" = "easy" ];then
    MODES=15
    amps1=($(seq 0 .01 .25))
    amps2=($(seq .01 .01 .25))
    SAMPLES=($(seq 1 100 300))
  else
    MODES=60
    SAMPLES=($(seq 1 100 300))
    difractionlimit=($(seq 0 .005 .05))
    small=($(seq .05 .0025 .1))
    large=($(seq .1 .005 .2))
    extreme=($(seq .2 .02 .5))
    amps=( "${difractionlimit[@]}" "${small[@]}" "${large[@]}" "${extreme[@]}" )
    echo ${amps[@]}
    echo ${#amps[@]}
    amps1=( "${difractionlimit[@]}" "${small[@]}" "${large[@]}" "${extreme[@]:0:${#extreme[@]}-1}" )
    amps2=( "${difractionlimit[@]:1}" "${small[@]}" "${large[@]}" "${extreme[@]}" )
  fi
else
  TYPE=''
  ITERS=10
  NEIGHBORS=(1 5 10 15 20 25 30 35 40 50)
  mPSNR=($(seq 1 10 91))
  xPSNR=($(seq 10 10 100))
  SAMPLES=($(seq 1 10 10))

  if [ "$DIFFICULTY" = "easy" ];then
    MODES=15
    amps1=($(seq 0 .025 .25))
    amps2=($(seq .025 .025 .25))
  else
    MODES=60
    amps1=($(seq 0 .025 .5))
    amps2=($(seq .025 .025 .5))
  fi
fi


for DIST in powerlaw dirichlet
do
  for SNR in `seq 1 ${#xPSNR[@]}`
  do
    for AMP in `seq 1 ${#amps1[@]}`
    do
      for R in 0 1
      do
        for N in `seq 1 ${#NEIGHBORS[@]}`
        do
          for S in `seq 1 ${#SAMPLES[@]}`
          do
            while [ $(squeue -u thayeralshaabi -h -t pending -r | wc -l) -gt 500 ]
            do
              sleep 10s
            done

            j="${ENV} multipoint_dataset.py ${TYPE}"
            j="${j} --npoints ${N}"
            j="${j} --sphere ${R}"
            j="${j} --psf_type ${PSF_TYPE}"
            j="${j} --dist ${DIST}"
            j="${j} --iters ${ITERS}"
            j="${j} --bimodal"
            j="${j} --noise"
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

            if [ "$DATASET" = "train" ];then
              j="${j} --random_crop ${RCROP}"
            fi

            task="/usr/bin/sbatch"
            task="${task} --qos=abc_normal"

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

            task="${task} --cpus-per-task=1"
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
done
