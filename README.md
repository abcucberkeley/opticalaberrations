# Multiscale attention networks for sensorless detection of aberrations in adaptive optics

[![python](https://img.shields.io/badge/python-3.8+-3776AB.svg?style=flat&logo=python&logoColor=3776AB)](https://www.python.org/)
[![tensorflow](https://img.shields.io/badge/tensorFlow-2.5+-FF6F00.svg?style=flat&logo=tensorflow)](https://www.tensorflow.org/)
[![license](https://img.shields.io/github/license/abcucberkeley/opticalaberrations.svg?style=flat&logo=git&logoColor=white)](https://opensource.org/license/bsd-2-clause/)
[![issues](https://img.shields.io/github/issues/abcucberkeley/opticalaberrations.svg?style=flat&logo=github)](https://github.com/abcucberkeley/opticalaberrations/issues)
[![pr](https://img.shields.io/github/issues-pr/abcucberkeley/opticalaberrations.svg?style=flat&logo=github)](https://github.com/abcucberkeley/opticalaberrations/pulls)

|  | [![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)](https://ubuntu.com/)                                                                                                             | [![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)](https://www.microsoft.com/en-us/windows)                                        | ![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white) |
|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `main`    | [![Ubuntu-master](https://github.com/abcucberkeley/opticalaberrations/actions/workflows/ubuntu-build.yaml/badge.svg?event=push)](https://github.com/abcucberkeley/opticalaberrations/actions/workflows/ubuntu-build.yaml)                 | [![Windows-master](https://github.com/abcucberkeley/opticalaberrations/actions/workflows/windows-build.yaml/badge.svg?event=push)](https://github.com/abcucberkeley/opticalaberrations/actions/workflows/windows-build.yaml)                | [![Docker-ubuntu-build](https://github.com/abcucberkeley/opticalaberrations/actions/workflows/docker_action.yml/badge.svg?event=push)](https://github.com/abcucberkeley/opticalaberrations/actions/workflows/docker_action.yml) |
| `develop` | [![Ubuntu-develop](https://github.com/abcucberkeley/opticalaberrations/actions/workflows/ubuntu-build.yaml/badge.svg?branch=develop&event=push)](https://github.com/abcucberkeley/opticalaberrations/actions/workflows/ubuntu-build.yaml) | [![Windows-develop](https://github.com/abcucberkeley/opticalaberrations/actions/workflows/windows-build.yaml/badge.svg?branch=develop&event=push)](https://github.com/abcucberkeley/opticalaberrations/actions/workflows/windows-build.yaml) | [![Docker-ubuntu-build](https://github.com/abcucberkeley/opticalaberrations/actions/workflows/docker_action.yml/badge.svg?branch=develop&event=push)](https://github.com/abcucberkeley/opticalaberrations/actions/workflows/docker_action.yml) |


# Table of Contents

* [Installation](#installation)
  * [For Linux or Windows](#for-linux-or-windows)
  * [For MOSAIC Microscope](#for-mosaic-microscope)
* [Utilities](#utilities)
  * [Simple predictions](#simple-predictions)
  * [Tile-based predictions](#tile-based-predictions)
  * [Aggregate predictions](#aggregate-predictions)
  * [Deconvolution](#deconvolution)

![zernike_pyramid](examples/zernikes/10th_zernike_pyramid.png)

## Installation

We recommend using Anaconda
(<https://docs.anaconda.com/anaconda/install/index.html>)
to install the required packages for running our models.

> **Note:** Please make sure you have Git LFS installed to download our pretrained models:
<https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage>

Once you have Anaconda installed on your system, clone the repo using the following commands:

```shell
git clone -b develop --recurse-submodules https://github.com/abcucberkeley/opticalaberrations.git
```

Create a new `conda` environment using the following commands (will create an environment named "ml"):

| System  | Requirements                 |
|---------|------------------------------|
| Ubuntu  | [`ubuntu.yml`](ubuntu.yml)     |
| Windows | [`windows.yml`](windows.yml) |

```shell
# Ubuntu via .yml
cd opticalaberrations
conda env create -f ubuntu.yml
conda activate ml
```
```shell
# Windows via .yml
cd opticalaberrations
conda env create -f windows.yml
conda activate ml
```
>....to later update to the latest packages in `*.yml`:
>
>```shell
>conda env update --file ***.yml 
>```

```shell
# Windows via conda and pip.  If you abscond from installing from the .yml file, you may try this at your own risk:
conda create python=3.10 cudatoolkit=11.2 cudnn=8.1.0  matplotlib astropy seaborn numpy scikit-image scikit-learn scikit-spatial pandas ipython pytest ujson zarr conda pycudadecon -c conda-forge -n ml --yes
conda activate ml
pip install tensorflow=2.10 keras=2.10 
pip install cupy-cuda11x tensorflow_addons dphtools csbdeep line-profiler line-profiler-pycharm tifffile=2023.9.18 imagecodecs==2023.9.18 
```

### Pre-trained models

To make sure you have the latest pre-trained models:

```shell
git lfs fetch --all
git lfs pull 
```

## Utilities

The [`src/python ao.py`](src/python ao.py) script provides a CLI
for running our models on a given 3D stack (`.tif` file).

> **Note:** Make sure to activate the new `ml` env before running the script,
> or use the full filepath to your `python` environment.

### Simple predictions

For each successful run, the script will output the following files:
* `*_sample_predictions_psf.tif`: predicted PSF
* `*_sample_predictions_zernike_coefficients.csv`: predicted Zernike modes
* `*_sample_predictions_pupil_displacement.tif`: predicted wavefront
* `*_sample_predictions_corrected_actuators.csv`: a new vector describing the new positions for the DM's actuators

#### Example Usage (Sample)

```shell
python ao.py predict_sample [--optional_flags] model input dm_calibration
```

The script takes 3 positional arguments and a few optional ones described below.

#### Positional arguments

|                  | Description                                                         |
|------------------|---------------------------------------------------------------------|
| `model`          | path to pretrained TensorFlow model                                 |
| `input`          | path to input (.tif file)                                           |
| `dm_calibration` | path DM calibration mapping matrix (e.g., Zernike_Korra_Bax273.csv) |

#### Optional arguments

|                            | Description                                                                                                                      |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| `help`                     | show this help message and exit                                                                                                  |
| `current_dm`               | optional path to current DM state .csv file (Default: `blank mirror`)                                                            |
| `prev`                     | previous predictions .csv file (Default: `None`)                                                                                 |
| `lateral_voxel_size`       | lateral voxel size in microns for X (Default: `0.108`)                                                                           |
| `axial_voxel_size`         | axial voxel size in microns for Z (Default: `0.1`)                                                                               |
| `wavelength`               | wavelength in microns (Default: `0.51`)                                                                                          |
| `dm_damping_scalar`        | scale DM actuators by an arbitrary multiplier (Default: `0.75`)                                                                  |
| `freq_strength_threshold`  | minimum frequency threshold in fourier space [fractional values below that will be set to the desired minimum] (Default: `0.01`) |
| `prediction_threshold`     | set predictions below threshold to zero (waves) (Default: `.1`)                                                                  |
| `sign_threshold`           | flip sign of modes above given threshold <br/> [fractional value relative to previous prediction] (Default: `.9`)                |
| `num_predictions`          | number of predictions per sample to estimate model's confidence (Default: `10`)                                                  |
| `plot`                     | a toggle for plotting predictions                                                                                                |
| `estimate_sign_with_decon` | a toggle for estimating signs of each Zernike mode via decon                                                                     |
| `ignore_mode`              | ANSI index for mode you wish to ignore  (Default: [0, 1, 2, 4])                                                                  |


### Tile-based predictions

For each successful run, the script will output the following files:
* `*_tiles_predictions.csv`: a statistical summary of the predictions for each tile

> Note: You need to run [aggregate predictions](#aggregate-predictions) to get the final prediction

#### Example Usage (Tiles)

```shell
python ao.py predict_tiles [--optional_flags] model input
```

The script takes 2 positional arguments and a few optional ones described below.

#### Positional arguments

|         | Description                         |
|---------|-------------------------------------|
| `model` | path to pretrained TensorFlow model |
| `input` | path to input (.tif file)           |

#### Optional arguments

|                            | Description                                                                                                                      |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| `help`                     | show this help message and exit                                                                                                  |
| `window_size`              | size of the window to crop for each tile (Default: `64`)                                                                         |
| `prev`                     | previous predictions .csv file (Default: `None`)                                                                                 |
| `lateral_voxel_size`       | lateral voxel size in microns for X (Default: `0.108`)                                                                           |
| `axial_voxel_size`         | axial voxel size in microns for Z (Default: `0.1`)                                                                               |
| `wavelength`               | wavelength in microns (Default: `0.51`)                                                                                          |
| `freq_strength_threshold`  | minimum frequency threshold in fourier space [fractional values below that will be set to the desired minimum] (Default: `0.01`) |
| `prediction_threshold`     | set predictions below threshold to zero (waves) (Default: `0.`)                                                                  |
| `sign_threshold`           | flip sign of modes above given threshold <br/> [fractional value relative to previous prediction] (Default: `.9`)                |
| `num_predictions`          | number of predictions per sample to estimate model's confidence (Default: `10`)                                                  |
| `batch_size`               | maximum batch size for the model (Default: `100`)                                                                                |
| `ignore_tile`              | IDs [e.g., "z0-y0-x0"] for tiles you wish to ignore  (Default: None)                                                             |
| `plot`                     | a toggle for plotting predictions                                                                                                |
| `estimate_sign_with_decon` | a toggle for estimating signs of each Zernike mode via decon                                                                     |
| `ignore_mode`              | ANSI index for mode you wish to ignore  (Default: [0, 1, 2, 4])                                                                  |

### Aggregate predictions

For each successful run, the script will output the following files:
* `*_predictions_aggregated_psf.tif`: predicted PSF
* `*_predictions_aggregated.csv`: a statistical summary of the predictions
* `*_predictions_aggregated_zernike_coefficients.csv`: predicted zernike modes
* `*_predictions_aggregated_pupil_displacement.tif`: predicted wavefront
* `*_predictions_aggregated_corrected_actuators.csv`: a new vector describing the new positions for the DM's actuators

> Note: You need to run `aggregate_predictions`
> for both [ROI-based predictions](#roi-based-predictions) and
> [Tile-based predictions](#tile-based-predictions)
> to get the final prediction

#### Example Usage (Aggregate)

```shell
python ao.py aggregate_predictions [--optional_flags] model predictions dm_calibration
```

The script takes 3 positional arguments and a few optional ones described below.

#### Positional arguments

|                  | Description                                                       |
|------------------|-------------------------------------------------------------------|
| `model`          | path to pretrained TensorFlow model                               |
| `predictions`    | path to model's predictions (.csv file)                           |
| `dm_calibration` | path DM calibration mapping matrix (eg. Zernike_Korra_Bax273.csv) |

#### Optional arguments

|                        | Description                                                                          |
|------------------------|--------------------------------------------------------------------------------------|
| `help`                 | show this help message and exit                                                      |
| `current_dm`           | optional path to current DM state .csv file (Default: `blank mirror`)                |
| `dm_damping_scalar`    | scale DM actuators by an arbitrary multiplier (Default: `0.75`)                      |
| `prediction_threshold` | set predictions below threshold to zero (waves) (Default: `0.`)                      |
| `majority_threshold`   | majority rule to use to determine dominant modes among ROIs (Default: `0.5`)         |
| `aggregation_rule`     | rule to use to calculate final prediction [mean, median, min, max] (Default: `mean`) |
| `min_percentile`       | minimum percentile to filter out outliers (Default: `10`)                            |
| `max_percentile`       | maximum percentile to filter out outliers (Default: `90`)                            |
| `lateral_voxel_size`   | lateral voxel size in microns for X (Default: `0.108`)                               |
| `axial_voxel_size`     | axial voxel size in microns for Z (Default: `0.1`)                                   |
| `wavelength`           | wavelength in microns (Default: `0.51`)                                              |
| `plot`                 | a toggle for plotting predictions                                                    |

### Deconvolution

For each successful run, the script will output the following files:
* `*_decon.tif`: results of deconvolving the given input with the desired PSF

#### Example Usage (Deconvolution)

```shell
python ao.py decon [--optional_flags] input psf
```

The script takes 3 positional arguments and a few optional ones described below.

#### Positional arguments

|         | Description               |
|---------|---------------------------|
| `input` | path to input (.tif file) |
| `psf`   | path to PSF (.tif file)   |

#### Optional arguments

|         | Description                                                            |
|---------|------------------------------------------------------------------------|
| `help`  | show this help message and exit                                        |
| `iters` | number of iterations for Richardson-Lucy deconvolution (Default: `10`) |
| `plot`  | a toggle for plotting results                                          |
