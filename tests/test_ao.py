
import sys
sys.path.append('.')
sys.path.append('./src')
sys.path.append('./tests')

import warnings
warnings.filterwarnings("ignore")

import logging
logger = logging.getLogger('')

import pytest
from pathlib import Path

from src import experimental


@pytest.mark.run(order=1)
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


@pytest.mark.run(order=2)
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
        plot_rotations=kargs['plot'],
        batch_size=kargs['batch_size'],
        ignore_modes=kargs['ignore_modes'],
        prediction_threshold=kargs['prediction_threshold'],
    )
    assert zernikes.shape[0] == kargs['num_modes']


@pytest.mark.run(order=3)
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
        plot_rotations=kargs['plot'],
        batch_size=kargs['batch_size'],
        ignore_modes=kargs['ignore_modes'],
        prediction_threshold=kargs['prediction_threshold'],
        rolling_strides=kargs['rolling_strides'],
    )
    assert zernikes.shape[0] == kargs['num_modes']


@pytest.mark.run(order=4)
def test_predict_tiles(kargs):
    tile_predictions = experimental.predict_tiles(
        model=kargs['model'],
        img=kargs['inputs'],
        prev=kargs['prev'],
        dm_calibration=kargs['dm_calibration'],
        dm_state=kargs['dm_state'],
        axial_voxel_size=kargs['axial_voxel_size'],
        lateral_voxel_size=kargs['lateral_voxel_size'],
        wavelength=kargs['wavelength'],
        plot=kargs['plot'],
        plot_rotations=kargs['plot'],
        batch_size=kargs['batch_size'],
        ignore_modes=kargs['ignore_modes'],
        window_size=kargs['window_size'],
        rolling_strides=kargs['rolling_strides'],
    )

    assert tile_predictions.shape == kargs['tiles_shape']


@pytest.mark.run(order=5)
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

    assert zernikes.shape[0] == kargs['num_modes']
