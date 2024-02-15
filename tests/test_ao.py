
import logging
import sys

sys.path.append('.')
sys.path.append('./src')
sys.path.append('./tests')

import warnings
warnings.filterwarnings("ignore")

import pytest
from pathlib import Path
from src import experimental


@pytest.mark.run(order=1)
def test_predict_tiles(kargs):
    logging.info(f"Pytest will assert that 'tile_predictions' has output shape of: "
                 f"(num_modes={kargs['num_modes']}, num_tiles={kargs['num_tiles']}), "
                 f"since window_size={kargs['window_size']}.")
    tile_predictions = experimental.predict_tiles(
        model=kargs['model'],
        img=kargs['inputs'],
        prev=kargs['prev'],
        dm_calibration=kargs['dm_calibration'],
        dm_state=kargs['dm_state'],
        axial_voxel_size=kargs['axial_voxel_size'],
        lateral_voxel_size=kargs['lateral_voxel_size'],
        wavelength=kargs['wavelength'],
        plot=False,
        plot_rotations=False,
        batch_size=kargs['batch_size'],
        ignore_modes=kargs['ignore_modes'],
        window_size=kargs['window_size'],
        min_psnr=kargs['min_psnr']
    )
    tile_predictions = tile_predictions.drop(columns=['mean', 'median', 'min', 'max', 'std'])
    assert tile_predictions.shape == (kargs['num_modes'], kargs['num_tiles']), f'{tile_predictions=}'


@pytest.mark.run(order=2)
def test_aggregate_tiles(kargs):
    zernikes = experimental.aggregate_predictions(
        model_pred=Path(f"{kargs['inputs'].with_suffix('')}_tiles_predictions.csv"),
        dm_calibration=kargs['dm_calibration'],
        dm_state=kargs['dm_state'],
        prediction_threshold=kargs['prediction_threshold'],
        majority_threshold=kargs['majority_threshold'],
        min_percentile=kargs['min_percentile'],
        max_percentile=kargs['max_percentile'],
        aggregation_rule=kargs['aggregation_rule'],
        max_isoplanatic_clusters=kargs['max_isoplanatic_clusters'],
        ignore_tile=kargs['ignore_tile'],
        plot=kargs['plot'],
    )
    assert zernikes.shape[1] == kargs['num_modes'] + 2

@pytest.mark.run(order=3)
def test_predict_sample(kargs):
    zernikes = experimental.predict_sample(
        model=kargs['model'],
        img=kargs['inputs'],
        dm_calibration=kargs['dm_calibration'],
        dm_state=kargs['dm_state'],
        prev=kargs['prev'],
        axial_voxel_size=kargs['axial_voxel_size'],
        lateral_voxel_size=kargs['lateral_voxel_size'],
        wavelength=kargs['wavelength'],
        plot=kargs['plot'],
        plot_rotations=kargs['plot_rotations'],
        batch_size=kargs['batch_size'],
        ignore_modes=kargs['ignore_modes'],
        prediction_threshold=kargs['prediction_threshold'],
        min_psnr=kargs['min_psnr']
    )
    assert zernikes.shape[0] == kargs['num_modes']

@pytest.mark.run(order=4)
def test_phase_retrieval(kargs):
    zernikes = experimental.phase_retrieval(
        img=kargs['inputs'],
        num_modes=kargs['num_modes'],
        dm_calibration=kargs['dm_calibration'],
        dm_state=kargs['dm_state'],
        axial_voxel_size=kargs['axial_voxel_size'],
        lateral_voxel_size=kargs['lateral_voxel_size'],
        wavelength=kargs['wavelength'],
        ignore_modes=kargs['ignore_modes'],
        plot=kargs['plot'],
        prediction_threshold=kargs['prediction_threshold'],
    )
    assert zernikes.shape[0] == kargs['num_modes']


@pytest.mark.run(order=5)
def test_predict_large_fov(kargs):
    zernikes = experimental.predict_large_fov(
        model=kargs['model'],
        img=kargs['inputs'],
        dm_calibration=kargs['dm_calibration'],
        dm_state=kargs['dm_state'],
        prev=kargs['prev'],
        axial_voxel_size=kargs['axial_voxel_size'],
        lateral_voxel_size=kargs['lateral_voxel_size'],
        wavelength=kargs['wavelength'],
        plot=kargs['plot'],
        plot_rotations=kargs['plot_rotations'],
        batch_size=kargs['batch_size'],
        ignore_modes=kargs['ignore_modes'],
        prediction_threshold=kargs['prediction_threshold'],
        min_psnr=kargs['min_psnr']
    )
    assert zernikes.shape[0] == kargs['num_modes']


@pytest.mark.run(order=6)
def test_predict_large_fov_with_interpolated_embeddings(kargs):
    zernikes = experimental.predict_large_fov(
        model=kargs['model'],
        img=kargs['inputs'],
        dm_calibration=kargs['dm_calibration'],
        dm_state=kargs['dm_state'],
        prev=kargs['prev'],
        axial_voxel_size=kargs['axial_voxel_size'],
        lateral_voxel_size=kargs['lateral_voxel_size'],
        wavelength=kargs['wavelength'],
        plot=kargs['plot'],
        plot_rotations=kargs['plot_rotations'],
        batch_size=kargs['batch_size'],
        ignore_modes=kargs['ignore_modes'],
        prediction_threshold=kargs['prediction_threshold'],
        min_psnr=kargs['min_psnr'],
        interpolate_embeddings=True
    )
    assert zernikes.shape[0] == kargs['num_modes']


@pytest.mark.run(order=7)
def test_predict_folder(kargs):
    test_folder = Path(f"{kargs['repo']}/dataset/experimental_zernikes/psfs")
    number_of_files = len(sorted(test_folder.glob(kargs['prediction_filename_pattern'])))
    
    logging.info(
        f"Pytest will assert that 'folder_predictions' has output shape of: "
        f"(num_modes={kargs['num_modes']}, number_of_files={number_of_files})"
    )
    
    predictions = experimental.predict_folder(
        model=kargs['model'],
        folder=test_folder,
        filename_pattern=kargs['prediction_filename_pattern'],
        prev=kargs['prev'],
        dm_calibration=kargs['dm_calibration'],
        dm_state=kargs['dm_state'],
        axial_voxel_size=kargs['axial_voxel_size'],
        lateral_voxel_size=kargs['lateral_voxel_size'],
        wavelength=kargs['wavelength'],
        plot=kargs['plot'],
        plot_rotations=kargs['plot_rotations'],
        batch_size=kargs['batch_size'],
        ignore_modes=kargs['ignore_modes'],
        min_psnr=kargs['min_psnr']
    )
    predictions = predictions.drop(columns=['mean', 'median', 'min', 'max', 'std'])
    assert predictions.shape == (kargs['num_modes'], number_of_files)


@pytest.mark.run(order=8)
def test_denoise(kargs):
    denoised_image = experimental.denoise(
        input_path=kargs['inputs'],
        model_path=kargs['denoiser'],
        window_size=kargs['window_size'],
        batch_size=kargs['batch_size'],
    )
    assert denoised_image.shape == kargs['input_shape']


@pytest.mark.run(order=9)
def test_predict_sample_with_denoising(kargs):
    zernikes = experimental.predict_sample(
        model=kargs['model'],
        img=kargs['inputs'],
        dm_calibration=kargs['dm_calibration'],
        dm_state=kargs['dm_state'],
        prev=kargs['prev'],
        axial_voxel_size=kargs['axial_voxel_size'],
        lateral_voxel_size=kargs['lateral_voxel_size'],
        wavelength=kargs['wavelength'],
        plot=kargs['plot'],
        plot_rotations=kargs['plot_rotations'],
        batch_size=kargs['batch_size'],
        ignore_modes=kargs['ignore_modes'],
        prediction_threshold=kargs['prediction_threshold'],
        min_psnr=kargs['min_psnr'],
        denoiser=kargs['denoiser'],
        denoiser_window_size=kargs['window_size'],
    )
    assert zernikes.shape[0] == kargs['num_modes']


@pytest.mark.run(order=10)
def test_predict_folder_with_denoising(kargs):
    test_folder = Path(f"{kargs['repo']}/dataset/experimental_zernikes/psfs")
    number_of_files = len(sorted(test_folder.glob(kargs['prediction_filename_pattern'])))
    
    logging.info(
        f"Pytest will assert that 'folder_predictions' has output shape of: "
        f"(num_modes={kargs['num_modes']}, number_of_files={number_of_files})"
    )
    
    predictions = experimental.predict_folder(
        model=kargs['model'],
        folder=test_folder,
        filename_pattern=kargs['prediction_filename_pattern'],
        prev=kargs['prev'],
        dm_calibration=kargs['dm_calibration'],
        dm_state=kargs['dm_state'],
        axial_voxel_size=kargs['axial_voxel_size'],
        lateral_voxel_size=kargs['lateral_voxel_size'],
        wavelength=kargs['wavelength'],
        plot=kargs['plot'],
        plot_rotations=kargs['plot_rotations'],
        batch_size=kargs['batch_size'],
        ignore_modes=kargs['ignore_modes'],
        min_psnr=kargs['min_psnr'],
        denoiser=kargs['denoiser'],
        denoiser_window_size=kargs['window_size'],
    )
    predictions = predictions.drop(columns=['mean', 'median', 'min', 'max', 'std'])
    assert predictions.shape == (kargs['num_modes'], number_of_files)


@pytest.mark.run(order=11)
def test_predict_rois(kargs):

    roi_predictions = experimental.predict_rois(
        model=kargs['model'],
        img=kargs['inputs'],
        prev=kargs['prev'],
        dm_calibration=kargs['dm_calibration'],
        dm_state=kargs['dm_state'],
        axial_voxel_size=kargs['axial_voxel_size'],
        lateral_voxel_size=kargs['lateral_voxel_size'],
        wavelength=kargs['wavelength'],
        plot=False,
        plot_rotations=False,
        batch_size=kargs['batch_size'],
        ignore_modes=kargs['ignore_modes'],
        window_size=kargs['window_size'],
        min_psnr=kargs['min_psnr'],
    )
    assert not roi_predictions.empty

@pytest.mark.run(order=12)
def test_aggregate_rois(kargs):
    zernikes = experimental.aggregate_predictions(
        model_pred=Path(f"{kargs['inputs'].with_suffix('')}_rois_predictions.csv"),
        dm_calibration=kargs['dm_calibration'],
        dm_state=kargs['dm_state'],
        prediction_threshold=kargs['prediction_threshold'],
        majority_threshold=kargs['majority_threshold'],
        min_percentile=kargs['min_percentile'],
        max_percentile=kargs['max_percentile'],
        aggregation_rule=kargs['aggregation_rule'],
        max_isoplanatic_clusters=kargs['max_isoplanatic_clusters'],
        ignore_tile=kargs['ignore_tile'],
        plot=kargs['plot'],
    )
    assert not zernikes.empty
