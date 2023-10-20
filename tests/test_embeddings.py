
import sys
sys.path.append('.')
sys.path.append('./src')

import warnings
warnings.filterwarnings("ignore")

import logging
logger = logging.getLogger('')

import pytest
from src import experimental


@pytest.mark.run(order=1)
def test_fourier_embeddings(kargs):
    emb = experimental.generate_embeddings(
        file=kargs['inputs'],
        model=kargs['model'],
        axial_voxel_size=kargs['axial_voxel_size'],
        lateral_voxel_size=kargs['lateral_voxel_size'],
        wavelength=kargs['wavelength'],
        plot=kargs['plot'],
        min_psnr=kargs['min_psnr'],
        fov_is_small=True,
    )
    assert emb.shape == kargs['embeddings_shape']


@pytest.mark.run(order=2)
def test_rolling_fourier_embeddings(kargs):
    emb = experimental.generate_embeddings(
        file=kargs['inputs'],
        model=kargs['model'],
        axial_voxel_size=kargs['axial_voxel_size'],
        lateral_voxel_size=kargs['lateral_voxel_size'],
        wavelength=kargs['wavelength'],
        plot=kargs['plot'],
        min_psnr=kargs['min_psnr'],
        fov_is_small=False
    )
    assert emb.shape == kargs['embeddings_shape']


@pytest.mark.run(order=3)
def test_embeddings_with_digital_rotations(kargs):
    emb = experimental.generate_embeddings(
        file=kargs['inputs'],
        model=kargs['model'],
        axial_voxel_size=kargs['axial_voxel_size'],
        lateral_voxel_size=kargs['lateral_voxel_size'],
        wavelength=kargs['wavelength'],
        plot=kargs['plot'],
        digital_rotations=kargs['digital_rotations'],
        min_psnr=kargs['min_psnr'],
        fov_is_small=False
    )
    assert emb.shape == kargs['rotations_shape']
