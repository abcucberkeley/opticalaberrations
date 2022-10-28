from pathlib import Path
from subprocess import call

# required flags
python = Path('~/anaconda3/envs/deep/bin/python')
repo = Path('~/Gitlab/opticalaberrations/')
script = repo/'src/ao.py'

n = 'neg'
sample = repo/f'examples/simulated/{n}/{n}.tif'
points = repo/f'examples/simulated/{n}/results/Detection3D.mat'

dm = repo/'examples/Zernike_Korra_Bax273.csv'
model = repo/'pretrained_models/z60_modes/lattice_yumb/x108-y108-z200/opticaltransformer.h5'
psf_type = repo/'lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat'

# common flags
prev = None
state = None
wavelength = .510
scalar = .75
lateral_voxel_size = .108
axial_voxel_size = .1
model_lateral_voxel_size = .108
model_axial_voxel_size = .2
sign_threshold = .5
prediction_threshold = .1
num_predictions = 10
window_size = 64
batch_size = 256
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

# extra `decon` flags
decon_iters = 10

# extra `detect_rois` flags
psf = repo/'examples/simulated/psf.tif'

deskew = f"{python} {script} deskew"
deskew += f" {sample}"
deskew += f" --flipz {num_rois}"
deskew += f" --skew_angle {skew_angle}"
deskew += f" --lateral_voxel_size {lateral_voxel_size}"
deskew += f" --axial_voxel_size {axial_voxel_size}"


detect_rois = f"{python} {script} detect_rois"
detect_rois += f" {sample}"
detect_rois += f" --psf {psf}"
detect_rois += f" --lateral_voxel_size {lateral_voxel_size}"
detect_rois += f" --axial_voxel_size {axial_voxel_size}"


predict_sample = f"{python} {script} predict_sample"
predict_sample += f" {model} {sample} {dm}"
predict_sample += f" --state {state}"
predict_sample += f" --scalar {scalar}"
predict_sample += f" --wavelength {wavelength}"
predict_sample += f" --lateral_voxel_size {lateral_voxel_size}"
predict_sample += f" --axial_voxel_size {axial_voxel_size}"
predict_sample += f" --model_lateral_voxel_size {model_lateral_voxel_size}"
predict_sample += f" --model_axial_voxel_size {model_axial_voxel_size}"
predict_sample += f" --psf_type {psf_type}"
predict_sample += f" --prediction_threshold 0."
predict_sample += f" --sign_threshold {sign_threshold}"
predict_sample += f" --num_predictions {num_predictions}"
predict_sample += f" --batch_size {batch_size}"
predict_sample += f" --prev {prev}"
predict_sample += f" --plot" if plot else ""

prev = None  # replace with initial predictions .csv file (*_predictions_zernike_coffs.csv)
# sample = repo/'data/agarose/exp1.tif'  # replace with second sample
predict_sample_signed = f"{predict_sample} --prev {prev}"

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
predict_rois += f" --num_predictions {num_predictions}"
predict_rois += f" --batch_size {batch_size}"
predict_rois += f" --prev {prev}"
predict_rois += f" --plot" if plot else ""

prev = None  # replace with initial predictions .csv file (*_predictions_zernike_coffs.csv)
# sample = repo/'data/agarose/exp1.tif'  # replace with second sample
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
predict_tiles += f" --num_predictions {num_predictions}"
predict_tiles += f" --batch_size {batch_size}"
predict_tiles += f" --prev {prev}"
predict_tiles += f" --plot" if plot else ""

prev = None  # replace with initial predictions .csv file (*_predictions_zernike_coffs.csv)
# sample = repo/'data/agarose/exp1.tif'  # replace with second sample
predict_tiles_signed = f"{predict_tiles} --prev {prev}"

aggregate_predictions_flags = f" --state {state}"
aggregate_predictions_flags += f" --scalar {scalar}"
aggregate_predictions_flags += f" --prediction_threshold {prediction_threshold}"
aggregate_predictions_flags += f" --majority_threshold {majority_threshold}"
aggregate_predictions_flags += f" --min_percentile {min_percentile}"
aggregate_predictions_flags += f" --max_percentile {max_percentile}"
aggregate_predictions_flags += f" --final_prediction {final_prediction}"
aggregate_predictions_flags += f" --lateral_voxel_size {lateral_voxel_size}"
aggregate_predictions_flags += f" --axial_voxel_size {axial_voxel_size}"
aggregate_predictions_flags += f" --psf_type {psf_type}"
aggregate_predictions_flags += f" --wavelength {wavelength}"
aggregate_predictions_flags += f" --plot" if plot else ""

roi_predictions = f"{sample.with_suffix('')}_rois_predictions.csv"
aggregate_roi_predictions = f"{python} {script} aggregate_predictions {roi_predictions} {sample} {dm} {aggregate_predictions_flags}"

tile_predictions = f"{sample.with_suffix('')}_tiles_predictions.csv"
aggregate_tile_predictions = f"{python} {script} aggregate_predictions {tile_predictions} {sample} {dm} {aggregate_predictions_flags}"


decon_sample_predictions = f"{python} {script} decon"
decon_sample_predictions += f" {sample}"
decon_sample_predictions += f" {sample.with_suffix('')}_predictions_psf.tif"
decon_sample_predictions += f" --iters {decon_iters}"

decon_roi_predictions = f"{python} {script} decon"
decon_roi_predictions += f" {sample}"
decon_roi_predictions += f" {sample.with_suffix('')}_rois_predictions_aggregated_psf.tif"
decon_roi_predictions += f" --iters {decon_iters}"

decon_tiles_predictions = f"{python} {script} decon"
decon_tiles_predictions += f" {sample}"
decon_tiles_predictions += f" {sample.with_suffix('')}_tiles_predictions_aggregated_psf.tif"
decon_tiles_predictions += f" --iters {decon_iters}"


# call(deskew, shell=True)

call(predict_sample, shell=True)
call(decon_sample_predictions, shell=True)

call(detect_rois, shell=True)
call(predict_rois, shell=True)
call(aggregate_roi_predictions, shell=True)
call(decon_roi_predictions, shell=True)

call(predict_tiles, shell=True)
call(aggregate_tile_predictions, shell=True)
call(decon_tiles_predictions, shell=True)
