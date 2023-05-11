@echo off
set mypath=U:\Data\TestsForThayer\20230510_Clta-mNG_cox8a_mChilada_multicolorfish_48hpf\largefov\exp1
REM change working directory to where this batch file is (aka the \src folder)
cd /D "%~dp0"

set prefix=clta_
set suffix=
set before=%prefix%before%suffix%

set fileA=%prefix%cluster0%suffix%
set fileB=%prefix%cluster1%suffix%
set fileC=%prefix%cluster2%suffix%
set fileD=%prefix%cluster3%suffix%
set ao.py="ao.py"

REM -----Flags for predict_tiles------
set model=../pretrained_models/lattice_yumb_x108um_y108um_z200um/opticalnet-15.h5
set predict_tiles_flags=--wavelength 0.51 --lateral_voxel_size 0.097 --axial_voxel_size 0.2 --window_size 64-64-64 --freq_strength_threshold 0.01 --batch_size 450 --prev None --plot --plot_rotations --cluster

REM -----Flags for aggregate_predictions-----
set aggregate_predictions_flags= --current_dm \\AANG\opticalaberrations\examples\DM_State_510nm.csv --plot --prediction_threshold 0.3


echo:
echo Predict_tiles....
REM asynch launch all but the last one. Pause one second. Launch final and wait for it to complete.
start "%fileA%"       python.exe %ao.py% predict_tiles %model% %mypath%\%fileA%.tif ../calibration/aang/15_mode_calibration.csv %predict_tiles_flags%
start "%fileB%"       python.exe %ao.py% predict_tiles %model% %mypath%\%fileB%.tif ../calibration/aang/15_mode_calibration.csv %predict_tiles_flags%
start "%fileC%"       python.exe %ao.py% predict_tiles %model% %mypath%\%fileC%.tif ../calibration/aang/15_mode_calibration.csv %predict_tiles_flags%
timeout 1 > NUL
start "%fileD%" /wait python.exe %ao.py% predict_tiles %model% %mypath%\%fileD%.tif ../calibration/aang/15_mode_calibration.csv %predict_tiles_flags%

echo:
echo Aggregate_predictions....
REM asynch launch all but the last one. Pause ten seconds. Launch final and wait for it to complete.
start python.exe %ao.py% aggregate_predictions %mypath%\%fileA%_tiles_predictions.csv ../calibration/aang/15_mode_calibration.csv %aggregate_predictions_flags%
start python.exe %ao.py% aggregate_predictions %mypath%\%fileB%_tiles_predictions.csv ../calibration/aang/15_mode_calibration.csv %aggregate_predictions_flags%
start python.exe %ao.py% aggregate_predictions %mypath%\%fileC%_tiles_predictions.csv ../calibration/aang/15_mode_calibration.csv %aggregate_predictions_flags%
timeout 10 > NUL
      python.exe %ao.py% aggregate_predictions %mypath%\%fileD%_tiles_predictions.csv ../calibration/aang/15_mode_calibration.csv %aggregate_predictions_flags%

echo:
echo Combine_tiles....
python.exe %ao.py% combine_tiles  %mypath%\%before%_tiles_predictions_aggregated_corrected_actuators.csv ^
--corrections  %mypath%\%fileA%_tiles_predictions_aggregated_p2v_error.tif ^
--corrections  %mypath%\%fileB%_tiles_predictions_aggregated_p2v_error.tif ^
--corrections  %mypath%\%fileC%_tiles_predictions_aggregated_p2v_error.tif ^
--corrections  %mypath%\%fileD%_tiles_predictions_aggregated_p2v_error.tif

echo:
echo All done.  Output here:    %mypath%
echo: