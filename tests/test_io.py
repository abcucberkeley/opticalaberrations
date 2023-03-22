
import sys
sys.path.append('.')
sys.path.append('./src')

import warnings
warnings.filterwarnings("ignore")

import logging
logger = logging.getLogger('')

import math
import pytest

from src import experimental
from src import preprocessing


@pytest.mark.run(order=1)
def test_load_sample(kargs):
    sample = experimental.load_sample(kargs['inputs'])
    assert sample.shape == kargs['input_shape']


@pytest.mark.run(order=2)
def test_model(kargs):
    model, modelpsfgen = experimental.reloadmodel_if_needed(modelpath=kargs['model'], preloaded=None)
    model.summary()
    assert model.name


@pytest.mark.run(order=3)
def test_ideal_empirical_psf(kargs):
    model, modelpsfgen = experimental.reloadmodel_if_needed(
        modelpath=kargs['model'],
        preloaded=None,
        ideal_empirical_psf=kargs['ideal_psf'],
        ideal_empirical_psf_voxel_size=(
            kargs['axial_voxel_size'],
            kargs['lateral_voxel_size'],
            kargs['lateral_voxel_size'],
        )
    )
    assert modelpsfgen.ipsf.shape == modelpsfgen.psf_shape


@pytest.mark.run(order=4)
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


@pytest.mark.run(order=5)
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

