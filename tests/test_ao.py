
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

from src import experimental
from src import preprocessing


@pytest.fixture(scope="session")
def kargs():
    repo = Path.cwd()

    kargs = dict(
        inputs=repo / f'examples/single/single.tif',
        input_shape=(256, 256, 256),
        embeddings_shape=(6, 64, 64, 1),
        digital_rotations=range(0, 361),
        rotations_shape=(361, 6, 64, 64, 1),
        dm_calibration=repo/'calibration/aang/28_mode_calibration.csv',
        model=repo/'pretrained_models/lattice_yumb_x108um_y108um_z200um/opticalnet-28.h5',
        psf_type=repo/'lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat',
        pois=repo/f'examples/single/results/Detection3D.mat',
        ideal_psf=repo/'examples/psf.tif',
        prev=None,
        current_dm=None,
        wavelength=.510,
        dm_damping_scalar=1.0,
        lateral_voxel_size=.108,
        axial_voxel_size=.2,
        freq_strength_threshold=.01,
        prediction_threshold=0.,
        num_predictions=1,
        window_size='64-64-64',  # z-y-x
        batch_size=128,
        plot=True,
        plot_rotations=True,
        ignore_modes=[],
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
