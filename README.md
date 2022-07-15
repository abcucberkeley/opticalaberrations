# Sensorless detection of aberrations for adaptive optics 


## Installation

We recommend using Anaconda 
(https://docs.anaconda.com/anaconda/install/index.html)
to install the required packages for running our models. 

Once you have Anaconda installed on your system, 
clone the repo and create a new `conda` environment 
using the following commands (will create an environment named "ml"):

For **Linux**: 
```
git clone git@github.com:abcucberkeley/opticalaberrations.git
cd opticalaberrations
conda create --name ml --file requirements.txt
conda activate ml
```


For **Windows**: 
```
git clone git@github.com:abcucberkeley/opticalaberrations.git
cd opticalaberrations
conda env create -f requirements.yml
conda activate ml
```


For **MOSAIC Microcope**: 

SVN update (repo will be downloaded to `"C:\SPIM\Common\Calculations\Python\Phase Retrieval ML\opticalaberrations\"`)
```
svn update c:\spim
```

Create environment "ml" and install dependencies
```
conda env create -f “C:\SPIM\Common\Calculations\Python\Phase Retrieval ML\opticalaberrations\requirements.yml”
```

## Usage

### Phase retrieval 

The [`phase_retrieval.py`](src/phase_retrieval.py) script provides a CLI 
for running phase retrieval on a given 3D PSF (`.tif` file). 
The script takes 3 positional arguments and a few optional ones described below. 

For each successful run, the script will output 3 files:
- `_pred.png`: visualization of the model predictions 
- `_diagnosis.png`: in-depth analysis of the model predictions 
- `_zernike_coffs.csv`: predicted zernike modes 
- `_pupil_displacement.tif`: predicted pupil displacement 
- `_corrected_actuators.csv`: a new vector describing the new positions for the DM's actuators

```shell
usage: phase_retrieval.py [-h] 
                          [--state STATE] 
                          [--x_voxel_size X_VOXEL_SIZE] 
                          [--y_voxel_size Y_VOXEL_SIZE] 
                          [--z_voxel_size Z_VOXEL_SIZE] 
                          [--wavelength WAVELENGTH] 
                          [--scalar SCALAR]
                          [--threshold THRESHOLD] 
                          [--plot] 
                          [--verbose]
                          model input pattern


Copyright (c) 2021 ABC. Licensed under the BSD 2-Clause License.

positional arguments:
  model                 path to pretrained tensorflow model
  input                 path to input .tif file
  pattern               path DM pattern mapping matrix (eg. Zernike_Korra_Bax273.csv)

optional arguments:
  -h, --help            show this help message and exit
  --state STATE         optional path to current DM state .csv file (Default: `blank mirror`)
  --x_voxel_size X_VOXEL_SIZE
                        lateral voxel size in microns for X (Default: `0.15`)
  --y_voxel_size Y_VOXEL_SIZE
                        lateral voxel size in microns for Y (Default: `0.15`)
  --z_voxel_size Z_VOXEL_SIZE
                        axial voxel size in microns for Z (Default: `0.6`)
  --wavelength WAVELENGTH
                        wavelength in microns (Default: `0.605`)
  --scalar SCALAR       scale DM actuators by an arbitrary multiplier (Default: `1`)
  --threshold THRESHOLD
                        set predictions below threshold to zero (microns) (Default: `1e-05`)
  --plot                a toggle for plotting predictions
  --verbose             a toggle for a progress bar
```

> **Note:** Make sure to activate your conda `env` before running the script, or use the full filepath to your `python` environment.

```
conda activate ml
```

An example of using the [`phase_retrieval.py`](src/phase_retrieval.py) script:
```
~/anaconda3/envs/ml/bin/python src/phase_retrieval.py pretrained_models/fourier_space/i64/z60_modes/x150-y150-z600/opticaltransformer examples/PSF_z7_p01_1.tif  examples/Zernike_Korra_Bax273.csv --state examples/DM_z7_p01_1.csv
```

For ***Windows*** (started in the \opticalabberations folder)
```
python.exe .\src\phase_retrieval.py .\pretrained_models\fourier_space\i64\z60_modes\x150-y150-z600\opticaltransformer .\examples\PSF_z7_p01_1.tif .\examples\Zernike_Korra_Bax273.csv --state .\examples\DM_z7_p01_1.csv
```


![example](examples/PSF_z7_p01_1_pred.png)
![example](examples/PSF_z7_p01_1_diagnosis.png)

An example of running the script without a DM state input is also provided below. 
```
~/anaconda3/envs/ml/bin/python src/phase_retrieval.py pretrained_models/fourier_space/i64/z60_modes/x150-y150-z600/opticaltransformer examples/PSF_blank.tif examples/Zernike_Korra_Bax273.csv
```

![example](examples/PSF_blank_pred.png)
![example](examples/PSF_blank_diagnosis.png)


### Training 
### Evaluations
### Models
 
