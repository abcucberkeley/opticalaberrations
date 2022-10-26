from pathlib import Path
from subprocess import call

# required flags
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
wavelength = .510
scalar = .75
lateral_voxel_size = .108
axial_voxel_size = .1
model_lateral_voxel_size = .108
model_axial_voxel_size = .2
sign_threshold = .5
prediction_threshold = .1
window_size = 64
plot = True

# extra `aggregate_predictions` flags
majority_threshold = .5
min_percentile = 10
max_percentile = 90
final_prediction = 'mean'

# extra `predict_rois` flags
num_rois = 10
min_intensity = 200
minimum_distance = 1.

# extra `deskew` flags
skew_angle = 32.45
flipz = False


# extra `detect_rois` flags
psf = None

deskew = f"{python} {script} deskew"
deskew += f" {sample}"
deskew += f" --flipz {num_rois}"
deskew += f" --skew_angle {skew_angle}"
deskew += f" --lateral_voxel_size {lateral_voxel_size}"
deskew += f" --axial_voxel_size {axial_voxel_size}"


detect_rois = f"{python} {script} detect_rois"
detect_rois += f" {sample} {psf}"
detect_rois += f" --lateral_voxel_size {lateral_voxel_size}"
detect_rois += f" --axial_voxel_size {axial_voxel_size}"


phase_retrieval = f"{python} {script} predict_sample"
phase_retrieval += f" {model} {sample} {dm}"
phase_retrieval += f" --state {state}"
phase_retrieval += f" --scalar {scalar}"
phase_retrieval += f" --wavelength {wavelength}"
phase_retrieval += f" --lateral_voxel_size {lateral_voxel_size}"
phase_retrieval += f" --axial_voxel_size {axial_voxel_size}"
phase_retrieval += f" --model_lateral_voxel_size {model_lateral_voxel_size}"
phase_retrieval += f" --model_axial_voxel_size {model_axial_voxel_size}"
phase_retrieval += f" --psf_type {psf_type}"
phase_retrieval += f" --prediction_threshold 0."
phase_retrieval += f" --sign_threshold {sign_threshold}"
phase_retrieval += f" --plot" if plot else ""

prev = None  # replace with initial predictions .csv file
sample = repo/'data/agarose/exp1.tif'  # replace with second sample
phase_retrieval_signed = f"{phase_retrieval} --prev {prev}"

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
predict_rois += f" --prediction_threshold 0."
predict_rois += f" --sign_threshold {sign_threshold}"
predict_rois += f" --prev {prev}"
predict_rois += f" --plot" if plot else ""

prev = None  # replace with initial predictions .csv file
sample = repo/'data/agarose/exp1.tif'  # replace with second sample
predict_rois_signed = f"{predict_rois} --prev {prev}"

predict_tiles = f"{python} {script} predict_tiles"
predict_tiles += f" {model} {sample}"
predict_tiles += f" --wavelength {wavelength}"
predict_tiles += f" --lateral_voxel_size {lateral_voxel_size}"
predict_tiles += f" --axial_voxel_size {axial_voxel_size}"
predict_tiles += f" --model_lateral_voxel_size {model_lateral_voxel_size}"
predict_tiles += f" --model_axial_voxel_size {model_axial_voxel_size}"
predict_tiles += f" --psf_type {psf_type}"
predict_tiles += f" --window_size {window_size}"
predict_tiles += f" --prediction_threshold 0."
predict_tiles += f" --sign_threshold {sign_threshold}"
predict_tiles += f" --prev {prev}"
predict_tiles += f" --plot" if plot else ""

prev = None  # replace with initial predictions .csv file
sample = repo/'data/agarose/exp1.tif'  # replace with second sample
predict_tiles_signed = f"{predict_tiles} --prev {prev}"


aggregate_predictions_flags = f" --state {state}"
aggregate_predictions_flags += f" --scalar {scalar}"
aggregate_predictions_flags += f" --prediction_threshold {prediction_threshold}"
aggregate_predictions_flags += f" --majority_threshold {majority_threshold}"
aggregate_predictions_flags += f" --min_percentile {min_percentile}"
aggregate_predictions_flags += f" --max_percentile {max_percentile}"
aggregate_predictions_flags += f" --final_prediction {final_prediction}"
aggregate_predictions_flags += f" --plot" if plot else ""

roi_predictions = f"{sample.with_suffix('')}_rois_predictions.csv"
aggregate_roi_predictions = f"{python} {script} aggregate_predictions {roi_predictions} {sample} {dm} {aggregate_predictions_flags}"

tile_predictions = f"{sample.with_suffix('')}_tiles_predictions.csv"
aggregate_tile_predictions = f"{python} {script} aggregate_predictions {tile_predictions} {sample} {dm} {aggregate_predictions_flags}"


# call(deskew, shell=True)
# call(detect_rois, shell=True)
# call(predict_rois, shell=True)
# call(aggregate_roi_predictions, shell=True)

call(predict_tiles, shell=True)
call(aggregate_tile_predictions, shell=True)
