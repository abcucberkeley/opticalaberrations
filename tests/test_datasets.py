
import sys

import numpy as np
from tqdm import trange

sys.path.append('.')
sys.path.append('./src')

import warnings
warnings.filterwarnings("ignore")

import pytest
import h5py
from pathlib import Path
import matplotlib.pyplot as plt

from src import psf_dataset
from src import multipoint_dataset
from src.synthetic import SyntheticPSF
from src.wavefront import Wavefront


def get_synthetic_generator(kargs):
    return SyntheticPSF(
        signed=True,
        rotate=True,
        psf_shape=[64, 64, 64],
        n_modes=kargs['num_modes'],
        distribution='mixed',
        mode_weights='pyramid',
        psf_type=kargs['psf_type'],
        lam_detection=kargs['wavelength'],
        x_voxel_size=kargs['lateral_voxel_size'],
        y_voxel_size=kargs['lateral_voxel_size'],
        z_voxel_size=kargs['axial_voxel_size'],
    )


@pytest.mark.run(order=1)
def test_theoretical_widefield_simulator(kargs):

    gen = SyntheticPSF(
        signed=True,
        rotate=True,
        psf_shape=[64, 64, 64],
        n_modes=kargs['num_modes'],
        distribution='mixed',
        mode_weights='pyramid',
        psf_type=kargs['psf_type'],
        lam_detection=kargs['wavelength'],
        x_voxel_size=kargs['lateral_voxel_size'],
        y_voxel_size=kargs['lateral_voxel_size'],
        z_voxel_size=kargs['axial_voxel_size'],
        use_theoretical_widefield_simulator=True,
        skip_remove_background_ideal_psf=False
    )

    amplitude = .1
    zernikes = np.zeros(15)

    for z in range(3, 15):  # zernikes from (3,..,14)

        if z == 4:  # skip defocus
            z = 'ideal'
            aberration = zernikes.copy()
            noise = False
            photons = 10000000
        else:
            aberration = zernikes.copy()
            aberration[z] = amplitude
            noise = True
            photons = 100000

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
            outdir=Path(f"{kargs['repo']}/dataset/theoretical_zernikes/psfs"),
            gen=gen,
            phi=phi,
            emb=False,
            photons=photons,
            noise=noise,
            normalize=True,
            lls_defocus_offset=(0, 0)
        )

        assert sample.shape == gen.psf_shape

        embeddings = psf_dataset.simulate_psf(
            filename=f'z{z}',
            outdir=Path(f"{kargs['repo']}/dataset/theoretical_zernikes/embeddings"),
            gen=gen,
            phi=phi,
            emb=True,
            photons=photons,
            noise=noise,
            normalize=True,
            lls_defocus_offset=(0, 0),
            plot=True,
            skip_remove_background=False
        )

        assert embeddings.shape == (6, gen.psf_shape[1], gen.psf_shape[2])


@pytest.mark.run(order=2)
def test_experimental_widefield_simulator(kargs):

    gen = SyntheticPSF(
        signed=True,
        rotate=True,
        psf_shape=[64, 64, 64],
        n_modes=kargs['num_modes'],
        distribution='mixed',
        mode_weights='pyramid',
        psf_type=kargs['psf_type'],
        lam_detection=kargs['wavelength'],
        x_voxel_size=kargs['lateral_voxel_size'],
        y_voxel_size=kargs['lateral_voxel_size'],
        z_voxel_size=kargs['axial_voxel_size'],
        use_theoretical_widefield_simulator=False,
        skip_remove_background_ideal_psf=False
    )

    amplitude = .1
    zernikes = np.zeros(15)

    for z in range(3, 15):  # zernikes from (3,..,14)

        if z == 4:  # skip defocus
            z = 'ideal'
            aberration = zernikes.copy()
            noise = False
            photons = 10000000
        else:
            aberration = zernikes.copy()
            aberration[z] = amplitude
            noise = True
            photons = 100000

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
            outdir=Path(f"{kargs['repo']}/dataset/experimental_zernikes/psfs"),
            gen=gen,
            phi=phi,
            emb=False,
            photons=photons,
            noise=noise,
            normalize=True,
            lls_defocus_offset=(0, 0)
        )

        assert sample.shape == gen.psf_shape

        embeddings = psf_dataset.simulate_psf(
            filename=f'z{z}',
            outdir=Path(f"{kargs['repo']}/dataset/experimental_zernikes/embeddings"),
            gen=gen,
            phi=phi,
            emb=True,
            photons=photons,
            noise=noise,
            normalize=True,
            lls_defocus_offset=(0, 0),
            plot=True,
            skip_remove_background=False
        )

        assert embeddings.shape == (6, gen.psf_shape[1], gen.psf_shape[2])


@pytest.mark.run(order=3)
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
        outdir=Path(f"{kargs['repo']}/dataset/aberrated"),
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
        outdir=Path(f"{kargs['repo']}/dataset/aberrated"),
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


@pytest.mark.run(order=4)
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
        outdir=Path(f"{kargs['repo']}/dataset/defocused"),
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
        outdir=Path(f"{kargs['repo']}/dataset/defocused"),
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
        outdir=Path(f"{kargs['repo']}/dataset/aberrated_defocused"),
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
        outdir=Path(f"{kargs['repo']}/dataset/aberrated_defocused"),
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


@pytest.mark.run(order=6)
def test_psf_dataset(kargs):
    gen = get_synthetic_generator(kargs)
    
    for i in trange(10, file=sys.stdout):
        sample = psf_dataset.create_synthetic_sample(
            filename=f'{i + 1}',
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


@pytest.mark.run(order=7)
def test_multipoint_dataset(kargs):
    gen = get_synthetic_generator(kargs)
    
    for i in trange(10, file=sys.stdout):
        sample = multipoint_dataset.create_synthetic_sample(
            filename=f'{i + 1}',
            npoints=5,
            fill_radius=.66,
            generators={str(kargs['psf_type']): gen},
            upsampled_generators={str(kargs['psf_type']): gen},
            modes=kargs['num_modes'],
            savedir=Path(f"{kargs['repo']}/dataset/beads"),
            distribution=gen.distribution,
            mode_dist=gen.mode_weights,
            gamma=.75,
            randomize_object_size=False,
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

    assert sample[kargs['psf_type']].shape == gen.psf_shape


@pytest.mark.run(order=8)
def test_randomize_object_size_dataset(kargs):
    gen = get_synthetic_generator(kargs)
    
    for i in trange(10, file=sys.stdout):
        sample = multipoint_dataset.create_synthetic_sample(
            randomize_object_size=True,
            filename=f'{i + 1}',
            npoints=10,
            fill_radius=.66,
            generators={str(kargs['psf_type']): gen},
            upsampled_generators={str(kargs['psf_type']): gen},
            modes=kargs['num_modes'],
            savedir=Path(f"{kargs['repo']}/dataset/objects"),
            distribution=gen.distribution,
            mode_dist=gen.mode_weights,
            gamma=.75,
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
    
    assert sample[kargs['psf_type']].shape == gen.psf_shape


@pytest.mark.run(order=9)
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

    for f in psf_types:
        f = Path(f"{Path(__file__).parent.parent.resolve()}/lattice/{Path(f).name}")

        with h5py.File(f, 'r') as file:
            lls_excitation_profile = file.get('DitheredxzPSFCrossSection')[:, 0]

            fig, ax = plt.subplots(figsize=(2, 6))
            exc_profile = np.squeeze(lls_excitation_profile)
            # rrange = np.linspace(-13, 13, exc_profile.shape[0])
            rrange = np.arange(-1*exc_profile.shape[0]//2, exc_profile.shape[0]//2, 1) * .488 / 1.33 * .1

            mat = ax.plot(exc_profile, rrange, 'k', linewidth=2)

            ax.set_xlim(right=0, left=1)
            ax.set_xticks(np.arange(0, 1.2, .2))
            # ax.set_ylim(-.2, .2)
            ax.set_ylim(-6, 6)
            # ax.set_yticks(np.arange(0, exc_profile.shape[0]+1, 4))
            # ax.set_yticks([]) #np.arange(-.2, .22, .02)
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.yaxis.tick_right()
            ax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

            plt.subplots_adjust(top=.9, bottom=.1, left=.1, right=.9, hspace=.15, wspace=.15)
            plt.savefig(f.parent / f"{f.stem}.svg", dpi=300, bbox_inches='tight', pad_inches=.25, transparent=True)

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
    
    for i in trange(10, file=sys.stdout):
        samples = multipoint_dataset.create_synthetic_sample(
            filename=f'{i + 1}',
            npoints=5,
            fill_radius=.66,
            generators=generators,
            upsampled_generators=upsampled_generators,
            modes=kargs['num_modes'],
            savedir=Path(f"{kargs['repo']}/dataset/modalities"),
            distribution='mixed',
            mode_dist='pyramid',
            gamma=.75,
            randomize_object_size=False,
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

    for psf in psf_types:
        assert samples[psf].shape == generators[psf].psf_shape
