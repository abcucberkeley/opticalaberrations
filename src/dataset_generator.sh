#!/bin/bash

HANDLER=slurm
ENV=~/anaconda3/envs/deep/bin/python

OUTDIR='/clusterfs/nvme/thayer/dataset/embeddings'

SHAPE=64
xVOXEL=.15
yVOXEL=.15
zVOXEL=.6
MODES=60
GAMMA=.75
LAMBDA=.605
NA=1.0
DTYPE='widefield'

TYPE='--otf'
OUTDIR="${OUTDIR}/train"
difractionlimit=($(seq 0 .01 .05))
small=($(seq .055 .005 .1))
large=($(seq .11 .01 .4))
extreme=($(seq .45 .05 .65))
amps=( "${difractionlimit[@]}" "${small[@]}" "${large[@]}" "${extreme[@]}" )
echo ${amps[@]}
echo ${#amps[@]}
amps1=( "${difractionlimit[@]}" "${small[@]}" "${large[@]}" "${extreme[@]:0:${#extreme[@]}-1}" )
amps2=( "${difractionlimit[@]:1}" "${small[@]}" "${large[@]}" "${extreme[@]}" )
mPSNR=($(seq 1 20 81))
xPSNR=($(seq 20 20 100))
SAMPLES=($(seq 1 100 1000))

#TYPE=''
#OUTDIR="${OUTDIR}/test"
#amps1=($(seq 0 .025 .5))
#amps2=($(seq .025 .025 .5))
#mPSNR=($(seq 1 10 91))
#xPSNR=($(seq 10 10 100))
#SAMPLES=($(seq 1 100 100))

for DIST in powerlaw dirichlet
do
  for SNR in `seq 1 ${#xPSNR[@]}`
  do
    for AMP in `seq 1 ${#amps1[@]}`
    do
      for S in `seq 1 ${#SAMPLES[@]}`
      do
        while [ $(squeue -u thayeralshaabi -h -t pending -r | wc -l) -gt 500 ]
        do
          sleep 10s
        done

        j="${ENV} dataset.py ${TYPE}"
        j="${j} --dist ${DIST}"
        j="${j} --bimodal"
        j="${j} --noise"
        j="${j} --gamma ${GAMMA}"
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

        task="/usr/bin/sbatch"
        task="${task} --qos=abc_normal"

        if [ $(squeue -u thayeralshaabi -h -t pending -r -p dgx | wc -l) -lt 128 ];then
            task="${task} --partition=dgx"
        elif [ $(squeue -u thayeralshaabi -h -t pending -r -p abc_a100 | wc -l) -lt 64 ];then
            task="${task} --partition=abc_a100"
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