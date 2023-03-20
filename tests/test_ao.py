
import sys
sys.path.append('.')
sys.path.append('./src')

import warnings
warnings.filterwarnings("ignore")

import logging
logger = logging.getLogger('')

import math
import pytest
from pathlib import Path
from operator import floordiv
import numpy as np

from src import experimental
from src import preprocessing


@pytest.fixture(scope="session")
def kargs():
    repo = Path.cwd()
    num_modes = 15
    input_shape = (256, 256, 256)  # z-y-x
    window_size = (64, 64, 64)  # z-y-x
    num_tiles = np.prod(tuple(map(floordiv, input_shape, window_size)))
    digital_rotations = range(0, 361)

    kargs = dict(
        inputs=repo / f'examples/single/single.tif',
        input_shape=input_shape,
        embeddings_shape=(6, 64, 64, 1),
        digital_rotations=digital_rotations,
        rotations_shape=(len(digital_rotations), 6, 64, 64, 1),
        window_size=window_size,
        num_tiles=num_tiles,
        tiles_shape=(num_modes, num_tiles+5),
        num_modes=num_modes,
        zernikes_shape=(num_modes, 3),
        model=repo / f'pretrained_models/lattice_yumb_x108um_y108um_z200um/opticalnet-{num_modes}.h5',
        dm_calibration=repo/'calibration/aang/28_mode_calibration.csv',
        psf_type=repo/'lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat',
        pois=repo/f'examples/single/results/Detection3D.mat',
        ideal_psf=repo/'examples/psf.tif',
        prev=None,
        dm_state=None,
        wavelength=.510,
        dm_damping_scalar=1.0,
        lateral_voxel_size=.108,
        axial_voxel_size=.2,
        freq_strength_threshold=.01,
        prediction_threshold=0.,
        num_predictions=1,
        batch_size=128,
        plot=True,
        plot_rotations=True,
        ignore_modes=[],
        # extra `aggregate_predictions` flags
        majority_threshold=.5,
        min_percentile=1,
        max_percentile=99,
        final_prediction='mean',
        ignore_tile=[],
        # extra `predict_rois` flags
        num_rois=10,
        min_intensity=200,
        minimum_distance=.5,
    )

    return kargs


def test_load_sample(kargs):
    sample = experimental.load_sample(kargs['inputs'])
    assert sample.shape == kargs['input_shape']


def test_psnr(kargs):
    sample = experimental.load_sample(kargs['inputs'])

    psnr = preprocessing.prep_sample(
        sample,
        remove_background=True,
        return_psnr=True,
        plot=None,
        normalize=False,
        edge_filter=False,
        filter_mask_dilation=False,
    )
    assert math.isclose(psnr, 30, rel_tol=1)


def test_preprocessing(kargs):
    sample_voxel_size = (
        kargs['axial_voxel_size'],
        kargs['lateral_voxel_size'],
        kargs['lateral_voxel_size']
    )
    sample = experimental.load_sample(kargs['inputs'])

    sample = preprocessing.prep_sample(
        sample,
        sample_voxel_size=sample_voxel_size,
        remove_background=True,
        normalize=True,
        edge_filter=False,
        filter_mask_dilation=False,
        plot=kargs['inputs'].with_suffix('') if kargs['plot'] else None,
    )
    assert sample.shape == kargs['input_shape']


def test_fourier_embeddings(kargs):
    emb = experimental.generate_embeddings(
        file=kargs['inputs'],
        model=kargs['model'],
        axial_voxel_size=kargs['axial_voxel_size'],
        lateral_voxel_size=kargs['lateral_voxel_size'],
        wavelength=kargs['wavelength'],
        plot=kargs['plot'],
        match_model_fov=True
    )
    assert emb.shape == kargs['embeddings_shape']


def test_rolling_fourier_embeddings(kargs):
    emb = experimental.generate_embeddings(
        file=kargs['inputs'],
        model=kargs['model'],
        axial_voxel_size=kargs['axial_voxel_size'],
        lateral_voxel_size=kargs['lateral_voxel_size'],
        wavelength=kargs['wavelength'],
        plot=kargs['plot'],
        match_model_fov=False
    )
    assert emb.shape == kargs['embeddings_shape']


def test_embeddings_with_digital_rotations(kargs):
    emb = experimental.generate_embeddings(
        file=kargs['inputs'],
        model=kargs['model'],
        axial_voxel_size=kargs['axial_voxel_size'],
        lateral_voxel_size=kargs['lateral_voxel_size'],
        wavelength=kargs['wavelength'],
        plot=kargs['plot'],
        digital_rotations=kargs['digital_rotations'],
        match_model_fov=False
    )
    assert emb.shape == kargs['rotations_shape']


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
    assert zernikes.shape == kargs['zernikes_shape']


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
    assert zernikes.shape == kargs['zernikes_shape']


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
    )
    assert zernikes.shape == kargs['zernikes_shape']


def test_predict_tiles(kargs):
    tile_predictions = experimental.predict_tiles(
        model=kargs['model'],
        img=kargs['inputs'],
        prev=kargs['prev'],
        axial_voxel_size=kargs['axial_voxel_size'],
        lateral_voxel_size=kargs['lateral_voxel_size'],
        wavelength=kargs['wavelength'],
        plot=kargs['plot'],
        plot_rotations=kargs['plot'],
        batch_size=kargs['batch_size'],
        ignore_modes=kargs['ignore_modes'],
        window_size=kargs['window_size'],
    )

    assert tile_predictions.shape == kargs['tiles_shape']

    zernikes = experimental.aggregate_predictions(
        model=kargs['model'],
        model_pred=Path(f"{kargs['inputs'].with_suffix('')}_tiles_predictions.csv"),
        dm_calibration=kargs['dm_calibration'],
        dm_state=kargs['dm_state'],
        axial_voxel_size=kargs['axial_voxel_size'],
        lateral_voxel_size=kargs['lateral_voxel_size'],
        wavelength=kargs['wavelength'],
        prediction_threshold=kargs['prediction_threshold'],
        majority_threshold=kargs['majority_threshold'],
        min_percentile=kargs['min_percentile'],
        max_percentile=kargs['max_percentile'],
        final_prediction=kargs['final_prediction'],
        ignore_tile=kargs['ignore_tile'],
        plot=kargs['plot'],
    )

    assert zernikes.shape == kargs['zernikes_shape']
