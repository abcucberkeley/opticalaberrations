from pathlib import Path
from subprocess import call

python = Path('~/anaconda3/envs/deep/bin/python')
repo = Path('~/Gitlab/opticalaberrations/')
script = repo/'src/ao.py'

sample = repo/'data/agarose/exp1.tif'
points = repo/'data/agarose/results/Detection3D.mat'

dm = repo/'examples/Zernike_Korra_Bax273.csv'
model = repo/'pretrained_models/z60_modes/lattice_yumb/x108-y108-z200/opticaltransformer.h5'
psf_type = repo/'lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat'

# common flags
state = None
prev = None
wavelength = .510
scalar = .75
lateral_voxel_size = .108
axial_voxel_size = .1
model_lateral_voxel_size = .108
model_axial_voxel_size = .2
sign_threshold = .5
prediction_threshold = .1
window_size = 128
plot = True

# aggregate_predictions_flags flags
majority_threshold = .5
min_percentile = 10
max_percentile = 90
final_prediction = 'mean'

# predict_rois flags
num_rois = 10
min_intensity = 200
minimum_distance = 1.


predict_rois = f"{python} {script} predict_rois"
predict_rois += f" {model} {sample} {points}"
predict_rois += f" --num_rois {num_rois}"
predict_rois += f" --min_intensity {min_intensity}"
predict_rois += f" --minimum_distance {minimum_distance}"
predict_rois += f" --wavelength {wavelength}"
predict_rois += f" --lateral_voxel_size {lateral_voxel_size}"
predict_rois += f" --axial_voxel_size {axial_voxel_size}"
predict_rois += f" --model_lateral_voxel_size {model_lateral_voxel_size}"
predict_rois += f" --model_axial_voxel_size {model_axial_voxel_size}"
predict_rois += f" --psf_type {psf_type}"
predict_rois += f" --window_size {window_size}"
predict_rois += f" --prev {prev}"
predict_rois += f" --prediction_threshold 0."
predict_rois += f" --sign_threshold {sign_threshold}"

predict_tiles = f"{python} {script} predict_tiles"
predict_tiles += f" {model} {sample}"
predict_tiles += f" --wavelength {wavelength}"
predict_tiles += f" --lateral_voxel_size {lateral_voxel_size}"
predict_tiles += f" --axial_voxel_size {axial_voxel_size}"
predict_tiles += f" --model_lateral_voxel_size {model_lateral_voxel_size}"
predict_tiles += f" --model_axial_voxel_size {model_axial_voxel_size}"
predict_tiles += f" --psf_type {psf_type}"
predict_tiles += f" --window_size {window_size}"
predict_tiles += f" --prev {prev}"
predict_tiles += f" --prediction_threshold 0."
predict_tiles += f" --sign_threshold {sign_threshold}"


aggregate_predictions_flags = f" --state {state}"
aggregate_predictions_flags += f" --scalar {scalar}"
aggregate_predictions_flags += f" --prediction_threshold {prediction_threshold}"
aggregate_predictions_flags += f" --majority_threshold {majority_threshold}"
aggregate_predictions_flags += f" --min_percentile {min_percentile}"
aggregate_predictions_flags += f" --max_percentile {max_percentile}"
aggregate_predictions_flags += f" --final_prediction {final_prediction}"
aggregate_predictions_flags += f" --plot" if plot else ""

roi_predictions = f"{sample.with_suffix('')}_rois_predictions.csv"
aggregate_roi_predictions = f"{python} {script} aggregate_predictions"
aggregate_roi_predictions += f" {roi_predictions} {sample} {dm}"

tile_predictions = f"{sample.with_suffix('')}_tiles_predictions.csv"
aggregate_tile_predictions = f"{python} {script} aggregate_predictions"
aggregate_tile_predictions += f" {tile_predictions} {sample} {dm}"


call(predict_rois, shell=True)
call(aggregate_roi_predictions, shell=True)
call(predict_tiles, shell=True)
call(aggregate_tile_predictions, shell=True)
