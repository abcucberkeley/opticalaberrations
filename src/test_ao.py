from pathlib import Path
from subprocess import call
import platform

# required flags
python = Path('~/anaconda3/envs/deep/bin/python')
repo = Path('~/Github/opticalaberrations/')
script = repo/'src/ao.py'

if platform.system() == "Windows":
    import win32api  # pip install pywin32
    python  = Path('python.exe')    # Where to find python
    repo    = r'C:\SPIM\Common\Calculations\Python\Phase Retrieval ML\opticalaberrations'  # Top folder of repo (path with spaces still fails.
    repo    = Path(win32api.GetShortPathName(repo))   # shorten name to get rid of spaces.
    script  = repo/'src/ao.py'  # Script to run


n = 'single'                                            # The subfolder within /examples where the test data should exist.
image = repo/f'examples/{n}/{n}.tif'                   # Test image file
pois = repo/f'examples/{n}/results/Detection3D.mat'

dm_calibration = repo/'examples/Zernike_Korra_Bax273.csv'           # Deformable Mirror offsets that produce the Zernike functions
model = repo/'pretrained_models/z60_modes/lattice_yumb/x108-y108-z200/opticaltransformer.h5'
psf_type = repo/'lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat'    # excitation PSF being used.  This is sythesized.


# extra `detect_rois` flags
psf = repo/'examples/psf.tif'

# common flags
prev = None
current_dm = None
wavelength = .510
dm_damping_scalar = .75
lateral_voxel_size = .108
axial_voxel_size = .1
model_lateral_voxel_size = .108
model_axial_voxel_size = .2
sign_threshold = .5
freq_strength_threshold = .01
prediction_threshold = .1
num_predictions = 10
window_size = 64
batch_size = 256
plot = True
estimate_sign_with_decon = False
ignore_modes = [21, 27, 28, 35, 36, 44, 45, 54, 55]

# extra `aggregate_predictions` flags
majority_threshold = .5
min_percentile = 1
max_percentile = 99
final_prediction = 'mean'
ignore_tiles = ['z0-y0-x0', 'z0-y0-x3', 'z0-y3-x0', 'z0-y3-x3', 'z0-y3-x2', 'z1-y0-x3']

# extra `predict_rois` flags
num_rois = 10
min_intensity = 200
minimum_distance = .5

# extra `deskew` flags
skew_angle = 32.45
flipz = False

# extra `decon` flags
decon_iters = 10


## Construct the commands as strings using the flags we assigned above

deskew = f"{python} {script} deskew"
deskew += f" {image}"
deskew += f" --skew_angle {skew_angle}"
deskew += f" --lateral_voxel_size {lateral_voxel_size}"
deskew += f" --axial_voxel_size {axial_voxel_size}"
deskew += f" --flipz" if flipz else ""


detect_rois = f"{python} {script} detect_rois"
detect_rois += f" {image}"
detect_rois += f" --psf {psf}"
detect_rois += f" --lateral_voxel_size {lateral_voxel_size}"
detect_rois += f" --axial_voxel_size {axial_voxel_size}"


predict_sample = f"{python} {script} predict_sample"
predict_sample += f" {model} {image} {dm_calibration}"
predict_sample += f" --current_dm {current_dm}"
predict_sample += f" --dm_damping_scalar {dm_damping_scalar}"
predict_sample += f" --wavelength {wavelength}"
predict_sample += f" --lateral_voxel_size {lateral_voxel_size}"
predict_sample += f" --axial_voxel_size {axial_voxel_size}"
predict_sample += f" --prediction_threshold {prediction_threshold}"
predict_sample += f" --freq_strength_threshold {freq_strength_threshold}"
predict_sample += f" --sign_threshold {sign_threshold}"
predict_sample += f" --num_predictions {num_predictions}"
predict_sample += f" --batch_size {batch_size}"
predict_sample += f" --prev {prev}"
predict_sample += f" --plot" if plot else ""
predict_sample += f" --estimate_sign_with_decon" if estimate_sign_with_decon else ""

for mode in ignore_modes:
    predict_sample += f" --ignore_mode {mode}"


prev = None  # replace with initial predictions .csv file (*_predictions_zernike_coffs.csv)
# image = repo/'data/agarose/exp1.tif'  # replace with second image
predict_sample_signed = f"{predict_sample} --prev {prev}"

predict_rois = f"{python} {script} predict_rois"
predict_rois += f" {model} {image} {pois}"
predict_rois += f" --num_rois {num_rois}"
predict_rois += f" --min_intensity {min_intensity}"
predict_rois += f" --minimum_distance {minimum_distance}"
predict_rois += f" --wavelength {wavelength}"
predict_rois += f" --lateral_voxel_size {lateral_voxel_size}"
predict_rois += f" --axial_voxel_size {axial_voxel_size}"
predict_rois += f" --window_size {window_size}"
predict_rois += f" --prediction_threshold 0."
predict_rois += f" --freq_strength_threshold {freq_strength_threshold}"
predict_rois += f" --sign_threshold {sign_threshold}"
predict_rois += f" --num_predictions {num_predictions}"
predict_rois += f" --batch_size {batch_size}"
predict_rois += f" --prev {prev}"
predict_rois += f" --plot" if plot else ""
predict_rois += f" --estimate_sign_with_decon" if estimate_sign_with_decon else ""

for mode in ignore_modes:
    predict_rois += f" --ignore_mode {mode}"


prev = None  # replace with initial predictions .csv file (*_predictions_zernike_coffs.csv)
# image = repo/'data/agarose/exp1.tif'  # replace with second image
predict_rois_signed = f"{predict_rois} --prev {prev}"

predict_tiles = f"{python} {script} predict_tiles"
predict_tiles += f" {model} {image}"
predict_tiles += f" --wavelength {wavelength}"
predict_tiles += f" --lateral_voxel_size {lateral_voxel_size}"
predict_tiles += f" --axial_voxel_size {axial_voxel_size}"
predict_tiles += f" --window_size {window_size}"
predict_tiles += f" --prediction_threshold 0."
predict_tiles += f" --freq_strength_threshold {freq_strength_threshold}"
predict_tiles += f" --sign_threshold {sign_threshold}"
predict_tiles += f" --num_predictions {num_predictions}"
predict_tiles += f" --batch_size {batch_size}"
predict_tiles += f" --prev {prev}"
predict_tiles += f" --plot" if plot else ""
predict_tiles += f" --estimate_sign_with_decon" if estimate_sign_with_decon else ""

for mode in ignore_modes:
    predict_tiles += f" --ignore_mode {mode}"


prev = None  # replace with initial predictions .csv file (*_predictions_zernike_coffs.csv)
# image = repo/'data/agarose/exp1.tif'  # replace with second image
predict_tiles_signed = f"{predict_tiles} --prev {prev}"

aggregate_predictions_flags = f" --current_dm {current_dm}"
aggregate_predictions_flags += f" --dm_damping_scalar {dm_damping_scalar}"
aggregate_predictions_flags += f" --prediction_threshold {prediction_threshold}"
aggregate_predictions_flags += f" --majority_threshold {majority_threshold}"
aggregate_predictions_flags += f" --min_percentile {min_percentile}"
aggregate_predictions_flags += f" --max_percentile {max_percentile}"
aggregate_predictions_flags += f" --final_prediction {final_prediction}"
aggregate_predictions_flags += f" --lateral_voxel_size {lateral_voxel_size}"
aggregate_predictions_flags += f" --axial_voxel_size {axial_voxel_size}"
aggregate_predictions_flags += f" --wavelength {wavelength}"
aggregate_predictions_flags += f" --plot" if plot else ""

roi_predictions = f"{image.with_suffix('')}_rois_predictions.csv"
aggregate_roi_predictions = f"{python} {script} aggregate_predictions {model} {roi_predictions} {dm_calibration} {aggregate_predictions_flags}"

tile_predictions = f"{image.with_suffix('')}_tiles_predictions.csv"
aggregate_tile_predictions = f"{python} {script} aggregate_predictions {model} {tile_predictions} {dm_calibration} {aggregate_predictions_flags}"

for tile in ignore_tiles:
    aggregate_tile_predictions += f" --ignore_tile {tile}"

decon_sample_predictions = f"{python} {script} decon"
decon_sample_predictions += f" {image}"
decon_sample_predictions += f" {image.with_suffix('')}_sample_predictions_psf.tif"
decon_sample_predictions += f" --iters {decon_iters}"
decon_sample_predictions += f" --plot" if plot else ""

decon_roi_predictions = f"{python} {script} decon"
decon_roi_predictions += f" {image}"
decon_roi_predictions += f" {image.with_suffix('')}_rois_predictions_aggregated_psf.tif"
decon_roi_predictions += f" --iters {decon_iters}"
decon_roi_predictions += f" --plot" if plot else ""

decon_tiles_predictions = f"{python} {script} decon"
decon_tiles_predictions += f" {image}"
decon_tiles_predictions += f" {image.with_suffix('')}_tiles_predictions_aggregated_psf.tif"
decon_tiles_predictions += f" --iters {decon_iters}"
decon_tiles_predictions += f" --plot" if plot else ""


## Execute the calls to each.  You can comment out the ones you don't want to run,
# but some (e.g., 'decon') need the outputs from the preceeding call

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
