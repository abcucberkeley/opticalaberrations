
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
        fill_radius=.75,
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
        min_lls_defocus_offset=-2,
        max_lls_defocus_offset=2,
        min_photons=100000,
        max_photons=200000,
    )

    assert sample.shape == (1, 64, 64, 64)


@pytest.mark.run(order=5)
def test_multimodal_dataset(kargs):

    psf_types = [
        '../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat',
        '../lattice/Gaussian_NAexc0p21_NAsigma0p21_annulus0p4-0p2_crop0p1_FWHM51p0.mat',
        '../lattice/MBSq_NAexc0p30_annulus0p375-0p225_FWHM48p5.mat',
        '../lattice/Sinc_by_lateral_SW_NAexc0p32_NAsigma5p0_annulus0p4-0p2_realSLM_FWHM51p5.mat',
        '../lattice/v2Hex_NAexc0p50_NAsigma0p075_annulus0p60-0p40_FWHM53p0.mat',
        '../lattice/v2HexRect_NAexc0p50_NAsigma0p15_annulus0p60-0p40_FWHM_56p0.mat',
        '../lattice/ACHex_NAexc0p40_NAsigma0p075_annulus0p6-0p2_crop0p1_FWHM52p0.mat',
        '../lattice/MBHex_NAexc0p43_annulus0p47_0p40_crop0p08_FWHM48p0.mat',
        '2photon',
        'confocal',
        'widefield',
    ]

    generators, upsampled_generators = {}, {}
    for psf in psf_types:
        generators[psf] = SyntheticPSF(
            psf_shape=[64, 64, 64],
            order='ansi',
            n_modes=kargs['num_modes'],
            distribution='mixed',
            mode_weights='pyramid',
            signed=True,
            rotate=True,
            psf_type=psf,
            lam_detection=.920 if psf == '2photon' else kargs['wavelength'],
            x_voxel_size=kargs['lateral_voxel_size'],
            y_voxel_size=kargs['lateral_voxel_size'],
            z_voxel_size=kargs['axial_voxel_size'],
        )

        upsampled_generators[psf] = SyntheticPSF(
            psf_shape=[128, 128, 128],
            order=generators[psf].order,
            n_modes=generators[psf].n_modes,
            distribution=generators[psf].distribution,
            mode_weights=generators[psf].mode_weights,
            signed=generators[psf].signed,
            rotate=generators[psf].rotate,
            psf_type=generators[psf].psf_type,
            lam_detection=generators[psf].lam_detection,
            x_voxel_size=generators[psf].x_voxel_size,
            y_voxel_size=generators[psf].y_voxel_size,
            z_voxel_size=generators[psf].z_voxel_size,
        )

    samples = multipoint_dataset.create_synthetic_sample(
        filename='1',
        npoints=5,
        fill_radius=.75,
        generators=generators,
        upsampled_generators=upsampled_generators,
        modes=kargs['num_modes'],
        savedir=Path(f"{kargs['repo']}/dataset/modalities"),
        distribution='mixed',
        mode_dist='pyramid',
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
        min_lls_defocus_offset=-2,
        max_lls_defocus_offset=2,
        min_photons=100000,
        max_photons=200000,
    )

    assert samples.shape == (len(psf_types), 64, 64, 64)

