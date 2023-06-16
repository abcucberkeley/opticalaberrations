@echo off
REM change working directory to where this batch file is (aka the \src folder)
cd /D "%~dp0"

echo Copy latest models
copy "V:\thayer\opticalaberrations\models\new\new_embeddings\opticalnet-15-spatial_planes1020\opticalnet\2023-06-10-08-03.h5" "C:\SPIM\Common\Calculations\Python\Phase Retrieval ML\opticalaberrations\pretrained_models\lattice_yumb_x108um_y108um_z200um\opticalnet-15-spatial_planes1020.h5"
copy "V:\thayer\opticalaberrations\models\new\new_embeddings\opticalnet-15-spatial_planes10\opticalnet\2023-06-10-08-03.h5"   "C:\SPIM\Common\Calculations\Python\Phase Retrieval ML\opticalaberrations\pretrained_models\lattice_yumb_x108um_y108um_z200um\opticalnet-15-spatial_planes10.h5"
copy "V:\thayer\opticalaberrations\models\new\new_embeddings\opticalnet-15-spatial_planes20\opticalnet\2023-06-10-08-03.h5"   "C:\SPIM\Common\Calculations\Python\Phase Retrieval ML\opticalaberrations\pretrained_models\lattice_yumb_x108um_y108um_z200um\opticalnet-15-spatial_planes20.h5"

echo:
echo Add metadata
echo for Model 1020
C:\Users\Aang\miniconda3\envs\ml\python.exe "C:\SPIM\Common\Calculations\Python\Phase Retrieval ML\opticalaberrations\src\predict.py" ../pretrained_models/lattice_yumb_x108um_y108um_z200um/opticalnet-15-spatial_planes1020.h5 --psf_type ../lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat --n_modes 15 --wavelength .510 --x_voxel_size .108 --y_voxel_size .108 --z_voxel_size .2 --embedding_option spatial_planes1020 metadata
echo:
echo for Model 20
C:\Users\Aang\miniconda3\envs\ml\python.exe "C:\SPIM\Common\Calculations\Python\Phase Retrieval ML\opticalaberrations\src\predict.py" ../pretrained_models/lattice_yumb_x108um_y108um_z200um/opticalnet-15-spatial_planes20.h5   --psf_type ../lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat --n_modes 15 --wavelength .510 --x_voxel_size .108 --y_voxel_size .108 --z_voxel_size .2 --embedding_option spatial_planes20 metadata
echo:
echo for Model 10
C:\Users\Aang\miniconda3\envs\ml\python.exe "C:\SPIM\Common\Calculations\Python\Phase Retrieval ML\opticalaberrations\src\predict.py" ../pretrained_models/lattice_yumb_x108um_y108um_z200um/opticalnet-15-spatial_planes10.h5   --psf_type ../lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat --n_modes 15 --wavelength .510 --x_voxel_size .108 --y_voxel_size .108 --z_voxel_size .2 --embedding_option spatial_planes10 metadata

echo copy to cluster
copy "C:\SPIM\Common\Calculations\Python\Phase Retrieval ML\opticalaberrations\pretrained_models\lattice_yumb_x108um_y108um_z200um\opticalnet-15-spatial_planes*.h5" V:\thayer\opticalaberrations\pretrained_models\lattice_yumb_x108um_y108um_z200um
