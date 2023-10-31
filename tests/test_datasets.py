
import sys

import numpy as np

sys.path.append('.')
sys.path.append('./src')

import warnings
warnings.filterwarnings("ignore")

import logging

import pytest
from pathlib import Path

from src import psf_dataset
from src import multipoint_dataset
from src.synthetic import SyntheticPSF
from src.wavefront import Wavefront


def get_synthetic_generator(kargs):
    return SyntheticPSF(
        order='ansi',
        n_modes=15,
        distribution='mixed',
        mode_weights='pyramid',
        signed=True,
        rotate=True,
        psf_type='../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat',
        lam_detection=.510,
        psf_shape=[64, 64, 64],
        x_voxel_size=.108,
        y_voxel_size=.108,
        z_voxel_size=.2,
    )


@pytest.mark.run(order=1)
def test_zernike_modes(kargs):

    gen = get_synthetic_generator(kargs)

    amplitude = .1
    zernikes = np.zeros(15)

    for z in range(3, 15):  # zernikes from (3,..,14)

        if z == 4:  # skip defocus
            continue

        aberration = zernikes.copy()
        aberration[z] = amplitude

        phi = Wavefront(
            amplitudes=aberration,
            order=gen.order,
            distribution=gen.distribution,
            mode_weights=gen.mode_weights,
            modes=gen.n_modes,
            gamma=gen.gamma,
            signed=gen.signed,
            rotate=gen.rotate,
            lam_detection=gen.lam_detection,
        )

        np.testing.assert_array_equal(phi.amplitudes, aberration)

        sample = psf_dataset.simulate_psf(
            filename=f'z{z}',
            outdir=Path(f"{kargs['repo']}/dataset/zernikes/psfs"),
            gen=gen,
            phi=phi,
            emb=False,
            photons=100000,
            noise=True,
            normalize=True,
            lls_defocus_offset=(0, 0)
        )

        assert sample.shape == gen.psf_shape

        embeddings = psf_dataset.simulate_psf(
            filename=f'z{z}',
            outdir=Path(f"{kargs['repo']}/dataset/zernikes/embeddings"),
            gen=gen,
            phi=phi,
            emb=True,
            photons=100000,
            noise=True,
            normalize=True,
            lls_defocus_offset=(0, 0),
            plot=True
        )

        assert embeddings.shape == (6, gen.psf_shape[1], gen.psf_shape[2])


@pytest.mark.run(order=2)
def test_random_aberrated_psf(kargs):

    gen = get_synthetic_generator(kargs)

    phi = Wavefront(
        amplitudes=(.1, .2),
        order=gen.order,
        distribution=gen.distribution,
        mode_weights=gen.mode_weights,
        modes=gen.n_modes,
        gamma=gen.gamma,
        signed=gen.signed,
        rotate=gen.rotate,
        lam_detection=gen.lam_detection,
    )

    sample = psf_dataset.simulate_psf(
        filename='1',
        outdir=Path(f"{kargs['repo']}/dataset/aberrations"),
        gen=gen,
        phi=phi,
        emb=False,
        photons=100000,
        noise=True,
        normalize=True,
        lls_defocus_offset=(0, 0)
    )

    assert sample.shape == gen.psf_shape

    embeddings = psf_dataset.simulate_psf(
        filename='2',
        outdir=Path(f"{kargs['repo']}/dataset/aberrations"),
        gen=gen,
        phi=phi,
        emb=True,
        photons=100000,
        noise=True,
        normalize=True,
        lls_defocus_offset=(0, 0),
        plot=True
    )

    assert embeddings.shape == (6, gen.psf_shape[1], gen.psf_shape[2])


@pytest.mark.run(order=3)
def test_random_defocused_psf(kargs):
    gen = get_synthetic_generator(kargs)

    phi = Wavefront(
        amplitudes=(0, 0),
        order=gen.order,
        distribution=gen.distribution,
        mode_weights=gen.mode_weights,
        modes=gen.n_modes,
        gamma=gen.gamma,
        signed=gen.signed,
        rotate=gen.rotate,
        lam_detection=gen.lam_detection,
    )

    sample = psf_dataset.simulate_psf(
        filename='1',
        outdir=Path(f"{kargs['repo']}/dataset/lls"),
        gen=gen,
        phi=phi,
        emb=False,
        photons=100000,
        noise=True,
        normalize=True,
        lls_defocus_offset=(1, 2)
    )

    assert sample.shape == gen.psf_shape

    embeddings = psf_dataset.simulate_psf(
        filename='2',
        outdir=Path(f"{kargs['repo']}/dataset/lls"),
        gen=gen,
        phi=phi,
        emb=True,
        photons=100000,
        noise=True,
        normalize=True,
        lls_defocus_offset=(1, 2),
        plot=True
    )

    assert embeddings.shape == (6, gen.psf_shape[1], gen.psf_shape[2])


@pytest.mark.run(order=4)
def test_random_aberrated_defocused_psf(kargs):
    gen = get_synthetic_generator(kargs)

    phi = Wavefront(
        amplitudes=(.1, .2),
        order=gen.order,
        distribution=gen.distribution,
        mode_weights=gen.mode_weights,
        modes=gen.n_modes,
        gamma=gen.gamma,
        signed=gen.signed,
        rotate=gen.rotate,
        lam_detection=gen.lam_detection,
    )

    sample = psf_dataset.simulate_psf(
        filename='1',
        outdir=Path(f"{kargs['repo']}/dataset/psfs"),
        gen=gen,
        phi=phi,
        emb=False,
        photons=100000,
        noise=True,
        normalize=True,
        lls_defocus_offset=(1, 2)
    )

    assert sample.shape == gen.psf_shape

    embeddings = psf_dataset.simulate_psf(
        filename='2',
        outdir=Path(f"{kargs['repo']}/dataset/psfs"),
        gen=gen,
        phi=phi,
        emb=True,
        photons=100000,
        noise=True,
        normalize=True,
        lls_defocus_offset=(1, 2),
        plot=True
    )

    assert embeddings.shape == (6, gen.psf_shape[1], gen.psf_shape[2])


@pytest.mark.run(order=5)
def test_psf_dataset(kargs):
    gen = get_synthetic_generator(kargs)

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

    assert sample.shape == gen.psf_shape


@pytest.mark.run(order=6)
def test_multipoint_dataset(kargs):
    gen = get_synthetic_generator(kargs)

    sample = multipoint_dataset.create_synthetic_sample(
        filename='1',
        npoints=5,
        fill_radius=.66,
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

    assert sample.shape == (1, *gen.psf_shape)


@pytest.mark.run(order=7)
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
        fill_radius=.66,
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

