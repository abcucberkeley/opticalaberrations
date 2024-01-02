#!/bin/bash
cd $PWD

datadir=$1
env=~/anaconda3/envs/ml/bin/python
calibration="../calibration/aang/15_mode_calibration.csv"
model="../pretrained_models/opticalnet-15-YuMB-lambda510.h5"
current_dm="None"
script="ao.py"

prefix="clta_"
before="${prefix}before"
clusters=("cluster0" "cluster1" "cluster2" "cluster3")

window_size="64-64-64"
config="--wavelength 0.51 --lateral_voxel_size 0.097 --axial_voxel_size 0.2"
predict_tiles_flags=" ${calibration} ${config} --window_size ${window_size} --batch_size 750 --plot --plot_rotations"
aggregate_predictions_flags=" ${calibration} --current_dm ${current_dm} --plot --prediction_threshold 0.3"

echo Initial prediction....
${env} ${script} predict_tiles ${model} ${datadir}/${before}.tif ${predict_tiles_flags}
${env} ${script} aggregate_predictions ${datadir}/${before}_tiles_predictions.csv ${aggregate_predictions_flags}

echo Predict_tiles....
for c in ${clusters[@]}
do
  file="${datadir}/${prefix}${c}.tif"
  echo $file

  if [ $c = ${clusters[-1]} ]; then
      ${env} ${script} predict_tiles ${model} ${file} ${predict_tiles_flags} --cluster
  else
      konsole -e "${env} ${script} predict_tiles ${model} ${file} ${predict_tiles_flags} --cluster" &
  fi
done


echo Aggregate_predictions....
for c in ${clusters[@]}
do
  file="${datadir}/${prefix}${c}_tiles_predictions.csv"
  ${env} ${script} aggregate_predictions ${file} ${aggregate_predictions_flags}
done


echo Combine_tiles....
cmd="${env} ${script} combine_tiles ${datadir}/${before}_tiles_predictions_aggregated_corrected_actuators.csv"

for c in ${clusters[@]}
do
  cmd="${cmd} --corrections ${datadir}/${prefix}${c}_tiles_predictions_aggregated_p2v_error.tif "
done

$cmd
