# Multiscale attention networks for sensorless detection of aberrations in adaptive optics

# Table of Contents
* [Installation](#installation)
   * [For Linux or Windows](#for-linux-or-windows)
   * [For MOSAIC Microscope](#for-mosaic-microscope)
* [Utilities](#utilities)
   * [Simple predictions](#simple-predictions)
   * [ROI-based predictions](#roi-based-predictions)
   * [Tile-based predictions](#tile-based-predictions)
   * [Aggregate predictions](#aggregate-predictions)
   * [Deconvolution](#deconvolution)
   * [Point detection](#point-detection)
   * [Deskew](#deskew)



## Installation

We recommend using Anaconda 
(https://docs.anaconda.com/anaconda/install/index.html)
to install the required packages for running our models. 

> **Note:** Please make sure you have Git LFS installed to download our pretrained models:
https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage

### For `Linux` or `Windows`: 

Once you have Anaconda installed on your system, clone the repo using the following commands:
```shell
git clone https://github.com/abcucberkeley/opticalaberrations.git
```

Create a new `conda` environment using the following commands (will create an environment named "ml"):
```shell
cd opticalaberrations
conda env create -f requirements.yml
conda activate ml
```

Finally, clone the LLSM3D tools repository for additional functions such as `decon`, `deskew` and `point detection`
```shell
git clone --branch dev https://github.com/abcucberkeley/LLSM3DTools.git
```


### For `MOSAIC` microscope: 

SVN update (repo will be downloaded to `"C:\SPIM\Common\Calculations\Python\Phase Retrieval ML\opticalaberrations\"`)
```
svn update c:\spim
```

Create environment "ml" and install dependencies
```
conda env create -f “C:\SPIM\Common\Calculations\Python\Phase Retrieval ML\opticalaberrations\requirements.yml”
```


## Utilities

The [`src/python ao.py`](src/python ao.py) script provides a CLI 
for running our models on a given 3D stack (`.tif` file). 

> **Note:** Make sure to activate your conda `env` before running the script, or use the full filepath to your `python` environment.


### Simple predictions

For each successful run, the script will output the following files:
- `*_predictions_psf.tif`: predicted PSF
- `*_predictions_zernike_coffs.csv`: predicted Zernike modes 
- `*_predictions_pupil_displacement.tif`: predicted wavefront
- `*_predictions_corrected_actuators.csv`: a new vector describing the new positions for the DM's actuators

#### Example Usage (Sample):
```shell
python ao.py predict_sample [--optional_flags] model input pattern
```

The script takes 3 positional arguments and a few optional ones described below. 

#### Positional arguments:

|           | Description                                                     |
|-----------|-----------------------------------------------------------------|
| `model`   | path to pretrained TensorFlow model                             |
| `input`   | path to input (.tif file)                                       |
| `pattern` | path DM pattern mapping matrix (e.g., Zernike_Korra_Bax273.csv) |


#### Optional arguments:

|                            | Description                                                                                                       |
|----------------------------|-------------------------------------------------------------------------------------------------------------------|
| `help`                     | show this help message and exit                                                                                   |
| `state`                    | optional path to current DM state .csv file (Default: `blank mirror`)                                             |
| `prev`                     | previous predictions .csv file (Default: `None`)                                                                  |
| `lateral_voxel_size`       | lateral voxel size in microns for X (Default: `0.108`)                                                            |
| `axial_voxel_size`         | axial voxel size in microns for Z (Default: `0.1`)                                                                |
| `wavelength`               | wavelength in microns (Default: `0.51`)                                                                           |
| `scalar`                   | scale DM actuators by an arbitrary multiplier (Default: `0.75`)                                                   |
| `prediction_threshold`     | set predictions below threshold to zero (waves) (Default: `0.`)                                                   |
| `sign_threshold`           | flip sign of modes above given threshold <br/> [fractional value relative to previous prediction] (Default: `0.`) |
| `num_predictions`          | number of predictions per sample to estimate model's confidence (Default: `10`)                                   |
| `plot`                     | a toggle for plotting predictions                                                                                 |



### ROI-based predictions 

For each successful run, the script will output the following files:
- `*_rois_predictions_stats.csv`: a statistical summary of the selected candidate ROIs
- `*_rois_predictions.csv`: a statistical summary of the predictions for each ROI

#### Example Usage (ROI-based):
```shell
python ao.py predict_rois [--optional_flags] model input peaks
```

The script takes 3 positional arguments and a few optional ones described below. 

#### Positional arguments:

|         | Description                                 |
|---------|---------------------------------------------|
| `model` | path to pretrained TensorFlow model         |
| `input` | path to input (.tif file)                   |
| `peaks` | path to point detection results (.mat file) |


#### Optional arguments:

|                            | Description                                                                                                       |
|----------------------------|-------------------------------------------------------------------------------------------------------------------|
| `help`                     | show this help message and exit                                                                                   |
| `window_size`              | size of the window to crop around each point of interest (Default: `64`)                                          |
| `num_rois`                 | max number of detected points to use for estimating aberrations (Default: `10`)                                   |
| `min_intensity`            | minimum intensity desired for detecting peaks of interest (Default: `200`)                                        |
| `minimum_distance`         | minimum distance to the nearest neighbor (microns) (Default: `1.0`)                                               |
| `prev`                     | previous predictions .csv file (Default: `None`)                                                                  |
| `lateral_voxel_size`       | lateral voxel size in microns for X (Default: `0.108`)                                                            |
| `axial_voxel_size`         | axial voxel size in microns for Z (Default: `0.1`)                                                                |
| `wavelength`               | wavelength in microns (Default: `0.51`)                                                                           |
| `prediction_threshold`     | set predictions below threshold to zero (waves) (Default: `0.`)                                                   |
| `sign_threshold`           | flip sign of modes above given threshold <br/> [fractional value relative to previous prediction] (Default: `0.`) |
| `num_predictions`          | number of predictions per sample to estimate model's confidence (Default: `10`)                                   |
| `batch_size`               | maximum batch size for the model (Default: `100`)                                                                 |
| `plot`                     | a toggle for plotting predictions                                                                                 |




### Tile-based predictions 

For each successful run, the script will output the following files:
- `*_tiles_predictions.csv`: a statistical summary of the predictions for each tile

#### Example Usage (Tiles):
```shell
python ao.py predict_tiles [--optional_flags] model input
```

The script takes 2 positional arguments and a few optional ones described below.

#### Positional arguments:

|         | Description                         |
|---------|-------------------------------------|
| `model` | path to pretrained TensorFlow model |
| `input` | path to input (.tif file)           |


#### Optional arguments:

|                            | Description                                                                                                       |
|----------------------------|-------------------------------------------------------------------------------------------------------------------|
| `help`                     | show this help message and exit                                                                                   |
| `window_size`              | size of the window to crop for each tile (Default: `64`)                                                          |
| `prev`                     | previous predictions .csv file (Default: `None`)                                                                  |
| `lateral_voxel_size`       | lateral voxel size in microns for X (Default: `0.108`)                                                            |
| `axial_voxel_size`         | axial voxel size in microns for Z (Default: `0.1`)                                                                |
| `wavelength`               | wavelength in microns (Default: `0.51`)                                                                           |
| `prediction_threshold`     | set predictions below threshold to zero (waves) (Default: `0.`)                                                   |
| `sign_threshold`           | flip sign of modes above given threshold <br/> [fractional value relative to previous prediction] (Default: `0.`) |
| `num_predictions`          | number of predictions per sample to estimate model's confidence (Default: `10`)                                   |
| `batch_size`               | maximum batch size for the model (Default: `100`)                                                                 |
| `plot`                     | a toggle for plotting predictions                                                                                 |


### Aggregate predictions 

For each successful run, the script will output the following files:
- `*_predictions_aggregated_psf.tif`: predicted PSF
- `*_predictions_aggregated.csv`: a statistical summary of the predictions
- `*_predictions_aggregated_zernike_coffs.csv`: predicted zernike modes 
- `*_predictions_aggregated_pupil_displacement.tif`: predicted wavefront
- `*_predictions_aggregated_corrected_actuators.csv`: a new vector describing the new positions for the DM's actuators


#### Example Usage (Aggregate):
```shell
python ao.py aggregate_predictions [--optional_flags] model predictions pattern
```

The script takes 3 positional arguments and a few optional ones described below. 

#### Positional arguments:

|               | Description                                                   |
|---------------|---------------------------------------------------------------|
| `model`       | path to pretrained TensorFlow model                           |
| `predictions` | path to model's predictions (.csv file)                       |
| `pattern`     | path DM pattern mapping matrix (eg. Zernike_Korra_Bax273.csv) |



#### Optional arguments:

|                        | Description                                                                                              |
|------------------------|----------------------------------------------------------------------------------------------------------|
| `help`                 | show this help message and exit                                                                          |
| `state`                | optional path to current DM state .csv file (Default: `blank mirror`)                                    |
| `scalar`               | scale DM actuators by an arbitrary multiplier (Default: `0.75`)                                          |
| `prediction_threshold` | set predictions below threshold to zero (waves) (Default: `0.`)                                          |
| `majority_threshold`   | majority rule to use to determine dominant modes among ROIs (Default: `0.5`)                             |
| `final_prediction`     | rule to use to calculate final prediction [mean, median, min, max] (Default: `mean`)                     |
| `min_percentile`       | minimum percentile to filter out outliers (Default: `10`)                                                |
| `max_percentile`       | maximum percentile to filter out outliers (Default: `90`)                                                |
| `lateral_voxel_size`   | lateral voxel size in microns for X (Default: `0.108`)                                                   |
| `axial_voxel_size`     | axial voxel size in microns for Z (Default: `0.1`)                                                       |
| `wavelength`           | wavelength in microns (Default: `0.51`)                                                                  |
| `plot`                 | a toggle for plotting predictions                                                                        |



### Deconvolution

For each successful run, the script will output the following files:
- `*_decon.tif`: results of deconvolving the given input with the desired PSF

#### Example Usage (Deconvolution)
```shell
python ao.py decon [--optional_flags] input psf
```

The script takes 3 positional arguments and a few optional ones described below. 

#### Positional arguments:

|         | Description               |
|---------|---------------------------|
| `input` | path to input (.tif file) |
| `psf`   | path to PSF (.tif file)   |



#### Optional arguments:

|         | Description                                                            |
|---------|------------------------------------------------------------------------|
| `help`  | show this help message and exit                                        |
| `iters` | number of iterations for Richardson-Lucy deconvolution (Default: `10`) |
| `plot`  | a toggle for plotting results                                          |


### Point detection

For each successful run, the script will output the following files:
- `/results/Detection3D.mat`: predicted points

#### Example Usage (Detect ROIs):
```shell
python ao.py detect_rois [--optional_flags] input
```

The script takes 1 positional argument and a few optional ones described below. 

#### Positional arguments:

|         | Description               |
|---------|---------------------------|
| `input` | path to input (.tif file) |


#### Optional arguments:

|                      | Description                                            |
|----------------------|--------------------------------------------------------|
| `help`               | show this help message and exit                        |
| `psf`                | path to the experimental PSF (.tif file)               |
| `lateral_voxel_size` | lateral voxel size in microns for X (Default: `0.108`) |
| `axial_voxel_size`   | axial voxel size in microns for Z (Default: `0.1`)     |



### Deskew

For each successful run, the script will output the following files:
- `/DS/*.tif`: de-skewed image with the desired skew angle


#### Example Usage (Deskew):
```shell
python ao.py deskew [--optional_flags] input
```

The script takes 1 positional argument and a few optional ones described below. 

#### Positional arguments:

|         | Description               |
|---------|---------------------------|
| `input` | path to input (.tif file) |


#### Optional arguments:

|                      | Description                                            |
|----------------------|--------------------------------------------------------|
| `help`               | show this help message and exit                        |
| `lateral_voxel_size` | lateral voxel size in microns for X (Default: `0.108`) |
| `axial_voxel_size`   | axial voxel size in microns for Z (Default: `0.1`)     |
| `skew_angle`         | skew angle (Default: `32.45`)                          |
| `flipz`              | a toggle to flip Z axis                                |
