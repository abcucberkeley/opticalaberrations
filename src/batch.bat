@set mypath=U:\Data\TestsForThayer\20230509_multicolorfish_24hpf\new_fish2

@set suffix=_rotated
@set fileA=before%suffix%
@set fileB=after0%suffix%
@set fileC=after1%suffix%
@set fileD=after2%suffix%

@start python.exe "C:\SPIM\Common\Calculations\Python\Phase Retrieval ML\opticalaberrations\src\ao.py" predict_tiles ../pretrained_models/lattice_yumb_x108um_y108um_z200um/opticalnet-15.h5 %mypath%\%fileA%.tif ../calibration/aang/15_mode_calibration.csv --wavelength 0.51 --lateral_voxel_size 0.097 --axial_voxel_size 0.2 --window_size 64-64-64 --freq_strength_threshold 0.01 --batch_size 450 --prev None --plot --plot_rotations --cluster
@start python.exe "C:\SPIM\Common\Calculations\Python\Phase Retrieval ML\opticalaberrations\src\ao.py" predict_tiles ../pretrained_models/lattice_yumb_x108um_y108um_z200um/opticalnet-15.h5 %mypath%\%fileB%.tif ../calibration/aang/15_mode_calibration.csv --wavelength 0.51 --lateral_voxel_size 0.097 --axial_voxel_size 0.2 --window_size 64-64-64 --freq_strength_threshold 0.01 --batch_size 450 --prev None --plot --plot_rotations --cluster
@start python.exe "C:\SPIM\Common\Calculations\Python\Phase Retrieval ML\opticalaberrations\src\ao.py" predict_tiles ../pretrained_models/lattice_yumb_x108um_y108um_z200um/opticalnet-15.h5 %mypath%\%fileC%.tif ../calibration/aang/15_mode_calibration.csv --wavelength 0.51 --lateral_voxel_size 0.097 --axial_voxel_size 0.2 --window_size 64-64-64 --freq_strength_threshold 0.01 --batch_size 450 --prev None --plot --plot_rotations --cluster
@      python.exe "C:\SPIM\Common\Calculations\Python\Phase Retrieval ML\opticalaberrations\src\ao.py" predict_tiles ../pretrained_models/lattice_yumb_x108um_y108um_z200um/opticalnet-15.h5 %mypath%\%fileD%.tif ../calibration/aang/15_mode_calibration.csv --wavelength 0.51 --lateral_voxel_size 0.097 --axial_voxel_size 0.2 --window_size 64-64-64 --freq_strength_threshold 0.01 --batch_size 450 --prev None --plot --plot_rotations --cluster

@start python.exe "C:\SPIM\Common\Calculations\Python\Phase Retrieval ML\opticalaberrations\src\ao.py" aggregate_predictions %mypath%\%fileA%_tiles_predictions.csv ../calibration/aang/15_mode_calibration.csv --current_dm \\AANG\opticalaberrations\examples\DM_State_510nm.csv
@start python.exe "C:\SPIM\Common\Calculations\Python\Phase Retrieval ML\opticalaberrations\src\ao.py" aggregate_predictions %mypath%\%fileB%_tiles_predictions.csv ../calibration/aang/15_mode_calibration.csv --current_dm \\AANG\opticalaberrations\examples\DM_State_510nm.csv
@start python.exe "C:\SPIM\Common\Calculations\Python\Phase Retrieval ML\opticalaberrations\src\ao.py" aggregate_predictions %mypath%\%fileC%_tiles_predictions.csv ../calibration/aang/15_mode_calibration.csv --current_dm \\AANG\opticalaberrations\examples\DM_State_510nm.csv
       python.exe "C:\SPIM\Common\Calculations\Python\Phase Retrieval ML\opticalaberrations\src\ao.py" aggregate_predictions %mypath%\%fileD%_tiles_predictions.csv ../calibration/aang/15_mode_calibration.csv --current_dm \\AANG\opticalaberrations\examples\DM_State_510nm.csv

python.exe "C:\SPIM\Common\Calculations\Python\Phase Retrieval ML\opticalaberrations\src\ao.py" combine_tiles  %mypath%\%fileA%_tiles_predictions_aggregated_clusters.csv ^
--corrections  %mypath%\%fileA%_tiles_predictions_aggregated_p2v_error.tif ^
--corrections  %mypath%\%fileB%_tiles_predictions_aggregated_p2v_error.tif ^
--corrections  %mypath%\%fileC%_tiles_predictions_aggregated_p2v_error.tif ^
--corrections  %mypath%\%fileD%_tiles_predictions_aggregated_p2v_error.tif
