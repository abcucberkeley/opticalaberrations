
import sys
sys.path.append('.')
sys.path.append('./src')

import warnings
warnings.filterwarnings("ignore")

import pytest

from src import experimental
from src import backend


@pytest.mark.run(order=1)
def test_load_sample(kargs):
    sample = backend.load_sample(kargs['inputs'])
    assert sample.shape == kargs['input_shape']


@pytest.mark.run(order=2)
def test_load_metadata(kargs):
    gen = backend.load_metadata(model_path=kargs['model'])
    assert hasattr(gen, 'psf_type')


@pytest.mark.run(order=3)
def test_reloadmodel_if_needed(kargs):
    model, modelpsfgen = experimental.reloadmodel_if_needed(modelpath=kargs['model'], preloaded=None)
    model.summary()


@pytest.mark.run(order=4)
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