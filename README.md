# Multiscale attention networks for sensorless detection of aberrations in adaptive optics


## Installation

We recommend using Anaconda 
(https://docs.anaconda.com/anaconda/install/index.html)
to install the required packages for running our models. 


### For `Linux` or `Windows`: 

Once you have Anaconda installed on your system, clone the repo using the following commands:
```shell
git clone https://github.com/abcucberkeley/opticalaberrations.git
```

Also, clone the LLSM3D tools repository for additional utils such as `deskew` and `point detection`
```shell
git clone https://github.com/abcucberkeley/LLSM3DTools.git
```

Create a new `conda` environment using the following commands (will create an environment named "ml"):
```shell
cd opticalaberrations
conda env create -f requirements.yml
conda activate ml
```


### For `MOSAIC Microcope`: 

SVN update (repo will be downloaded to `"C:\SPIM\Common\Calculations\Python\Phase Retrieval ML\opticalaberrations\"`)
```
svn update c:\spim
```

Create environment "ml" and install dependencies
```
conda env create -f “C:\SPIM\Common\Calculations\Python\Phase Retrieval ML\opticalaberrations\requirements.yml”
```


## Usage

The [`ao.py`](src/ao.py) script provides a CLI 
for running our models on a given 3D stack (`.tif` file). 

> **Note:** Make sure to activate your conda `env` before running the script, or use the full filepath to your `python` environment.
> 
```shell
usage: ao.py [-h] {deskew,points,predict,predict_rois,predict_tiles} ...
positional arguments: {deskew,points,predict,predict_rois,predict_tiles} 
```

## Phase retrieval 

The script takes 3 positional arguments and a few optional ones described below. 

For each successful run, the script will output the following files:
- `*_pred.png`: visualization of the model predictions 
- `*_zernike_coffs.csv`: predicted zernike modes 
- `*_pupil_displacement.tif`: predicted pupil displacement 
- `*_corrected_actuators.csv`: a new vector describing the new positions for the DM's actuators

```shell
usage: ao.py predict    [-h] 
                        [--state STATE] 
                        [--prev PREV] 
                        [--psf_type PSF_TYPE] 
                        [--lateral_voxel_size LATERAL_VOXEL_SIZE] 
                        [--axial_voxel_size AXIAL_VOXEL_SIZE] 
                        [--model_lateral_voxel_size MODEL_LATERAL_VOXEL_SIZE] 
                        [--model_axial_voxel_size MODEL_AXIAL_VOXEL_SIZE] 
                        [--wavelength WAVELENGTH] 
                        [--scalar SCALAR] 
                        [--threshold THRESHOLD] 
                        [--plot] 
                        model input pattern

positional arguments:
  model                       path to pretrained tensorflow model
  input                       path to input .tif file
  pattern                     path DM pattern mapping matrix (eg. Zernike_Korra_Bax273.csv)

optional arguments:
  -h, --help                  show this help message and exit
  --state               
                              optional path to current DM state .csv file (Default: `blank mirror`)
  --prev                
                              previous predictions .csv file (Default: `None`)
  --psf_type            
                              type of the desired PSF 
                              (Default: `../lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat`)
  --lateral_voxel_size 
                              lateral voxel size in microns for X (Default: `0.108`)
  --axial_voxel_size 
                              axial voxel size in microns for Z (Default: `0.1`)
  --model_lateral_voxel_size 
                              lateral voxel size in microns for X (Default: `0.108`)
  --model_axial_voxel_size 
                              axial voxel size in microns for Z (Default: `0.2`)
  --wavelength 
                              wavelength in microns (Default: `0.51`)
  --scalar              
                              scale DM actuators by an arbitrary multiplier (Default: `0.75`)
  --threshold THRESHOLD
                              set predictions below threshold to zero (microns) (Default: `0.01`)
  --plot                
                              a toggle for plotting predictions
```

### Example

```
conda activate ml
```

An example of using the [`ao.py`](src/ao.py) script:
```
~/anaconda3/envs/ml/bin/python src/ao.py predict 
pretrained_models/z15_modes/lattice_yumb/x108-y108-z200/opticaltransformer.h5 
examples/phase_retrieval/-.tif 
examples/Zernike_Korra_Bax273.csv 
--psf_type lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat
--state examples/phase_retrieval/DM-.csv 
--plot
```
![example](examples/phase_retrieval/-_pred.png)  

Example of a followup prediction to eval signs:
```
~/anaconda3/envs/ml/bin/python src/ao.py predict 
pretrained_models/z15_modes/lattice_yumb/x108-y108-z200/opticaltransformer.h5 
examples/phase_retrieval/-_2.tif 
examples/Zernike_Korra_Bax273.csv 
--psf_type lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat
--state examples/phase_retrieval/DM-.csv 
--prev examples/phase_retrieval/-_zernike_coffs.csv
--plot
```
![example](examples/phase_retrieval/-_2_sign_correction.png)  
![example](examples/phase_retrieval/-_2_pred.png)  



## ROI-based Prediction 

The script takes 4 positional arguments and a few optional ones described below. 

```shell
usage: ao.py predict_rois   [-h] 
                            [--state STATE] 
                            [--prev PREV] 
                            [--psf_type PSF_TYPE] 
                            [--window_size WINDOW_SIZE] 
                            [--num_rois NUM_ROIS] 
                            [--min_intensity MIN_INTENSITY] 
                            [--lateral_voxel_size LATERAL_VOXEL_SIZE] 
                            [--axial_voxel_size AXIAL_VOXEL_SIZE] 
                            [--model_lateral_voxel_size MODEL_LATERAL_VOXEL_SIZE] 
                            [--model_axial_voxel_size MODEL_AXIAL_VOXEL_SIZE] 
                            [--wavelength WAVELENGTH] 
                            [--scalar SCALAR] 
                            [--threshold THRESHOLD]
                            [--plot]
                            model input peaks pattern

positional arguments:
  model                       path to pretrained tensorflow model
  input                       path to input .tif file
  peaks                       path to point detection results (.mat file)
  pattern                     path DM pattern mapping matrix (eg. Zernike_Korra_Bax273.csv)

optional arguments:
  -h, --help                  show this help message and exit
  --state               
                              optional path to current DM state .csv file (Default: `blank mirror`)
  --prev            
                              previous predictions .csv file (Default: `None`)
  --psf_type    
                              type of the desired PSF 
                              (Default: `../lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat`)
  --window_size 
                              size of the window to crop around each point of interest (Default: `64`)
  --num_rois    
                              max number of detected points to use for estimating aberrations (Default: `10`)
  --min_intensity 
                              minimum intensity desired for detecting peaks of interest (Default: `200`)
  --lateral_voxel_size 
                              lateral voxel size in microns for X (Default: `0.108`)
  --axial_voxel_size 
                              axial voxel size in microns for Z (Default: `0.1`)
  --model_lateral_voxel_size 
                              lateral voxel size in microns for X (Default: `0.108`)
  --model_axial_voxel_size 
                              axial voxel size in microns for Z (Default: `0.2`)
  --wavelength 
                              wavelength in microns (Default: `0.51`)
  --scalar        
                              scale DM actuators by an arbitrary multiplier (Default: `0.75`)
  --threshold 
                              set predictions below threshold to zero (microns) (Default: `0.01`)
  --plot                
                              a toggle for plotting predictions
```

### Example

```
conda activate ml

~/anaconda3/envs/ml/bin/python src/ao.py predict_rois 
/pretrained_models/z60_modes/lattice_yumb/x108-y108-z200/opticaltransformer.h5
examples/agarose/exp1.tif
examples/experimental/agarose/results/Detection3D.mat 
examples/Zernike_Korra_Bax273.csv 
--psf_type lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat
--state examples/agarose/DM_system.csv 
--window_size 64
--plot
```
![example](examples/agarose/exp1_rois.png)  
![example](examples/agarose/exp1_selected_points.png)  
![example](examples/agarose/exp1_rois_pred.png)  


## Tile-based Prediction 

The script takes 3 positional arguments and a few optional ones described below. 

```shell
usage: ao.py predict_rois   [-h] 
                            [--state STATE] 
                            [--prev PREV] 
                            [--psf_type PSF_TYPE] 
                            [--window_size WINDOW_SIZE] 
                            [--num_rois NUM_ROIS] 
                            [--min_intensity MIN_INTENSITY] 
                            [--lateral_voxel_size LATERAL_VOXEL_SIZE] 
                            [--axial_voxel_size AXIAL_VOXEL_SIZE] 
                            [--model_lateral_voxel_size MODEL_LATERAL_VOXEL_SIZE] 
                            [--model_axial_voxel_size MODEL_AXIAL_VOXEL_SIZE] 
                            [--wavelength WAVELENGTH] 
                            [--scalar SCALAR] 
                            [--threshold THRESHOLD]
                            [--plot]
                            model input pattern

positional arguments:
  model                       path to pretrained tensorflow model
  input                       path to input .tif file
  pattern                     path DM pattern mapping matrix (eg. Zernike_Korra_Bax273.csv)

optional arguments:
  -h, --help                  show this help message and exit
  --state               
                              optional path to current DM state .csv file (Default: `blank mirror`)
  --prev            
                              previous predictions .csv file (Default: `None`)
  --psf_type    
                              type of the desired PSF 
                              (Default: `../lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat`)
  --window_size 
                              size of the window to crop around each point of interest (Default: `64`)
  --num_rois    
                              max number of detected points to use for estimating aberrations (Default: `10`)
  --min_intensity 
                              minimum intensity desired for detecting peaks of interest (Default: `200`)
  --lateral_voxel_size 
                              lateral voxel size in microns for X (Default: `0.108`)
  --axial_voxel_size 
                              axial voxel size in microns for Z (Default: `0.1`)
  --model_lateral_voxel_size 
                              lateral voxel size in microns for X (Default: `0.108`)
  --model_axial_voxel_size 
                              axial voxel size in microns for Z (Default: `0.2`)
  --wavelength 
                              wavelength in microns (Default: `0.51`)
  --scalar        
                              scale DM actuators by an arbitrary multiplier (Default: `0.75`)
  --threshold 
                              set predictions below threshold to zero (microns) (Default: `0.01`)
  --plot                
                              a toggle for plotting predictions
```

### Example

```
conda activate ml

~/anaconda3/envs/ml/bin/python src/ao.py predict_tiles 
/pretrained_models/z60_modes/lattice_yumb/x108-y108-z200/opticaltransformer.h5
examples/agarose/exp1.tif
examples/Zernike_Korra_Bax273.csv 
--psf_type lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat
--state examples/agarose/DM_system.csv 
--window_size 64
--plot
```
![example](examples/agarose/exp1_tiles_pred.png)
![example](examples/agarose/exp1_tiles_predictions.png)  