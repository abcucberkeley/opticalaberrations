
import sys
sys.path.append('.')
sys.path.append('./src')

import warnings
warnings.filterwarnings("ignore")

import logging
logger = logging.getLogger('')

import pytest
from pathlib import Path

from src import psf_dataset
from src import multipoint_dataset
from src.synthetic import SyntheticPSF


@pytest.mark.run(order=1)
def test_psf_aberrated_dataset(kargs):
    gen = SyntheticPSF(
        order='ansi',
        n_modes=kargs['num_modes'],
        distribution='mixed',
        mode_weights='pyramid',
        signed=True,
        rotate=True,
        psf_type='../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat',
        lam_detection=kargs['wavelength'],
        psf_shape=[64, 64, 64],
        x_voxel_size=kargs['lateral_voxel_size'],
        y_voxel_size=kargs['lateral_voxel_size'],
        z_voxel_size=kargs['axial_voxel_size'],
    )

    sample = psf_dataset.create_synthetic_sample(
        filename='1',
        gen=gen,
        savedir=Path(f"{kargs['repo']}/dataset/aberrations"),
        noise=True,
        normalize=True,
        min_amplitude=.1,
        max_amplitude=.2,
        min_lls_defocus_offset=0,
        max_lls_defocus_offset=0,
        min_photons=100000,
        max_photons=200000,
    )

    assert sample.shape == (64, 64, 64)


@pytest.mark.run(order=2)
def test_psf_lls_defocus_dataset(kargs):
    gen = SyntheticPSF(
        order='ansi',
        n_modes=kargs['num_modes'],
        distribution='mixed',
        mode_weights='pyramid',
        signed=True,
        rotate=True,
        psf_type='../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat',
        lam_detection=kargs['wavelength'],
        psf_shape=[64, 64, 64],
        x_voxel_size=kargs['lateral_voxel_size'],
        y_voxel_size=kargs['lateral_voxel_size'],
        z_voxel_size=kargs['axial_voxel_size'],
    )

    sample = psf_dataset.create_synthetic_sample(
        filename='1',
        gen=gen,
        savedir=Path(f"{kargs['repo']}/dataset/lls"),
        noise=True,
        normalize=True,
        min_amplitude=0.,
        max_amplitude=0.,
        min_lls_defocus_offset=-2,
        max_lls_defocus_offset=2,
        min_photons=100000,
        max_photons=200000,
    )

    assert sample.shape == (64, 64, 64)


@pytest.mark.run(order=3)
def test_psf_dataset(kargs):
    gen = SyntheticPSF(
        order='ansi',
        n_modes=kargs['num_modes'],
        distribution='mixed',
        mode_weights='pyramid',
        signed=True,
        rotate=True,
        psf_type='../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat',
        lam_detection=kargs['wavelength'],
        psf_shape=[64, 64, 64],
        x_voxel_size=kargs['lateral_voxel_size'],
        y_voxel_size=kargs['lateral_voxel_size'],
        z_voxel_size=kargs['axial_voxel_size'],
    )

    sample = psf_dataset.create_synthetic_sample(
        filename='1',
        gen=gen,
        savedir=Path(f"{kargs['repo']}/dataset/psfs"),
        noise=True,
        normalize=True,
        min_amplitude=.1,
        max_amplitude=.2,
        min_lls_defocus_offset=-2,
        max_lls_defocus_offset=2,
        min_photons=100000,
        max_photons=200000,
    )

    assert sample.shape == (64, 64, 64)


@pytest.mark.run(order=4)
def test_multipoint_dataset(kargs):
    gen = SyntheticPSF(
        order='ansi',
        n_modes=kargs['num_modes'],
        distribution='mixed',
        mode_weights='pyramid',
        signed=True,
        rotate=True,
        psf_type='../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat',
        lam_detection=kargs['wavelength'],
        psf_shape=[64, 64, 64],
        x_voxel_size=kargs['lateral_voxel_size'],
        y_voxel_size=kargs['lateral_voxel_size'],
        z_voxel_size=kargs['axial_voxel_size'],
    )

    sample = multipoint_dataset.create_synthetic_sample(
        filename='1',
        npoints=5,
        fill_radius=.5,
        generators={str(kargs['psf_type']): gen},
        upsampled_generators={str(kargs['psf_type']): gen},
        modes=kargs['num_modes'],
        savedir=Path(f"{kargs['repo']}/dataset/beads"),
        distribution=gen.distribution,
        mode_dist=gen.mode_weights,
        gamma=.75,
        randomize_voxel_size=False,
        emb=False,
        signed=True,
        random_crop=None,
        embedding_option=set(),
        rotate=True,
        noise=True,
        normalize=True,
        min_amplitude=.1,
        max_amplitude=.2,
        min_lls_defocus_offset=0,
        max_lls_defocus_offset=0,
        min_photons=100000,
        max_photons=200000,
    )

    assert sample.shape == (64, 64, 64)

