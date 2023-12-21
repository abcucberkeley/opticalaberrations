
import matplotlib
matplotlib.use('Agg')

import re
import logging
import sys
import os
import time
import ujson
from functools import partial
from typing import Any
from pathlib import Path
from tifffile import imwrite
import numpy as np
from scipy import stats as st

import matplotlib.pyplot as plt
plt.set_loglevel('error')

import cli
from utils import multiprocess, add_noise, randuniform, electrons2counts, photons2electrons
from synthetic import SyntheticPSF
from wavefront import Wavefront
from preprocessing import prep_sample
from embeddings import fourier_embeddings

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_synthetic_sample(
    savepath,
    inputs,
    amps,
    photons,
    counts,
    p2v,
    gen,
    lls_defocus_offset=0.,
    realspace=None,
    counts_mode=None,
    counts_percentiles=None,
    sigma_background_noise=None,
    mean_background_offset=None,
    electrons_per_count=None,
    quantum_efficiency=None,
    psf_type=None,
):

    if realspace is not None:
        imwrite(f"{savepath}_realspace.tif", realspace.astype(np.float32), compression='deflate', dtype=np.float32)

    imwrite(f"{savepath}.tif", inputs.astype(np.float32), compression='deflate', dtype=np.float32)
    # logger.info(f"Saved: {savepath.resolve()}.tif")

    with Path(f"{savepath}.json").open('w') as f:
        json = dict(
            path=f"{savepath}.tif",
            shape=inputs.shape,
            n_modes=int(gen.n_modes),
            order=str(gen.order),
            lls_defocus_offset=float(lls_defocus_offset),
            zernikes=amps.tolist(),
            photons=int(photons),
            counts=int(counts),
            counts_mode=int(counts_mode),
            counts_percentiles=counts_percentiles.tolist(),
            peak2peak=float(p2v),
            mean_background_offset=int(mean_background_offset),
            sigma_background_noise=float(sigma_background_noise),
            electrons_per_count=float(electrons_per_count),
            quantum_efficiency=float(quantum_efficiency),
            x_voxel_size=float(gen.x_voxel_size),
            y_voxel_size=float(gen.y_voxel_size),
            z_voxel_size=float(gen.z_voxel_size),
            wavelength=float(gen.lam_detection),
            na_detection=float(gen.na_detection),
            refractive_index=float(gen.refractive_index),
            mode_weights=str(gen.mode_weights),
            embedding_option=str(gen.embedding_option),
            distribution=str(gen.distribution),
            psf_type=str(psf_type)
        )

        ujson.dump(
            json,
            f,
            indent=4,
            sort_keys=False,
            ensure_ascii=False,
            escape_forward_slashes=False
        )


def simulate_psf(
    filename: str,
    outdir: Path,
    gen: SyntheticPSF,
    phi: Wavefront,
    photons: int,
    noise: bool = True,
    normalize: bool = True,
    emb: bool = False,
    lls_defocus_offset: Any = 0.,
    sigma_background_noise=40,
    mean_background_offset=100,
    electrons_per_count: float = .22,
    quantum_efficiency: float = .82,
    plot: bool = False,
    skip_remove_background: bool = False,
):
    outdir.mkdir(exist_ok=True, parents=True)
    np.random.seed(os.getpid()+np.random.randint(low=0, high=10**6))

    # aberrated PSF without noise
    kernel, amps, lls_defocus_offset = gen.single_psf(
        phi=phi,
        lls_defocus_offset=lls_defocus_offset,
        normed=True,
        meta=True,
    )
    kernel /= np.sum(kernel)
    kernel *= photons

    p2v = phi.peak2valley(na=1.0)

    if noise:  # convert to electrons to add shot noise and dark read noise, then convert to counts
        inputs = add_noise(
            kernel,
            mean_background_offset=mean_background_offset,
            sigma_background_noise=sigma_background_noise,
            quantum_efficiency=quantum_efficiency,
            electrons_per_count=electrons_per_count,
        )
    else:  # convert image to counts
        inputs = photons2electrons(kernel, quantum_efficiency=quantum_efficiency)
        inputs = electrons2counts(inputs, electrons_per_count=electrons_per_count)

    counts = np.sum(inputs)
    counts_mode = st.mode(inputs, axis=None).mode
    counts_mode = int(counts_mode[0]) if isinstance(counts_mode, (list, tuple, np.ndarray)) else int(counts_mode)
    counts_percentiles = np.array([np.percentile(inputs, p) for p in range(1, 101)], dtype=int)

    if normalize:
        inputs /= np.max(inputs)

    if emb:
        embeddings = prep_sample(
            inputs,
            sample_voxel_size=gen.voxel_size,
            model_fov=gen.psf_fov,
            remove_background=False if skip_remove_background else True,
            normalize=normalize,
            min_psnr=0,
            plot=outdir/filename if plot else None,
            na_mask=gen.na_mask
        )

        embeddings = np.squeeze(fourier_embeddings(
            inputs=embeddings,
            iotf=gen.iotf,
            na_mask=gen.na_mask,
            plot=outdir/filename if plot else None
        ))

        save_synthetic_sample(
            outdir / filename,
            embeddings,
            realspace=inputs,
            amps=amps,
            photons=photons,
            counts=counts,
            counts_mode=counts_mode,
            counts_percentiles=counts_percentiles,
            p2v=p2v,
            gen=gen,
            lls_defocus_offset=lls_defocus_offset,
            sigma_background_noise=sigma_background_noise,
            mean_background_offset=mean_background_offset,
            electrons_per_count=electrons_per_count,
            quantum_efficiency=quantum_efficiency,
            psf_type=gen.psf_type,
        )

        return embeddings

    else:
        save_synthetic_sample(
            outdir / filename,
            inputs,
            amps=amps,
            photons=photons,
            counts=counts,
            counts_mode=counts_mode,
            counts_percentiles=counts_percentiles,
            p2v=p2v,
            gen=gen,
            lls_defocus_offset=lls_defocus_offset,
            sigma_background_noise=sigma_background_noise,
            mean_background_offset=mean_background_offset,
            electrons_per_count=electrons_per_count,
            quantum_efficiency=quantum_efficiency,
            psf_type=gen.psf_type,
        )

        return inputs


def create_synthetic_sample(
    filename: str,
    gen: SyntheticPSF,
    savedir: Path,
    min_amplitude: float,
    max_amplitude: float,
    min_photons: int,
    max_photons: int,
    noise: bool,
    normalize: bool,
    min_lls_defocus_offset: float = 0.,
    max_lls_defocus_offset: float = 0.,
    emb: bool = False,
    skip_remove_background: bool = False,
):
    outdir = savedir / rf"{re.sub(r'.*/lattice/', '', str(gen.psf_type)).split('_')[0]}_lambda{round(gen.lam_detection * 1000)}"
    outdir = outdir / f"z{round(gen.z_voxel_size * 1000)}-y{round(gen.y_voxel_size * 1000)}-x{round(gen.x_voxel_size * 1000)}"
    outdir = outdir / f"z{gen.psf_shape[0]}-y{gen.psf_shape[0]}-x{gen.psf_shape[0]}"
    outdir = outdir / f"z{gen.n_modes}"

    if gen.distribution == 'powerlaw':
        outdir = outdir / f"powerlaw_gamma_{str(round(gen.gamma, 2)).replace('.', 'p')}"
    else:
        outdir = outdir / f"{gen.distribution}"

    outdir = outdir / f"photons_{min_photons}-{max_photons}"
    outdir = outdir / f"amp_{str(round(min_amplitude, 3)).replace('0.', 'p').replace('-', 'neg')}" \
                      f"-{str(round(max_amplitude, 3)).replace('0.', 'p').replace('-', 'neg')}"

    if min_lls_defocus_offset != 0 or max_lls_defocus_offset != 0:
        outdir = outdir / f"defocus_{str(round(min_lls_defocus_offset, 3)).replace('.', 'p').replace('-', 'neg')}" \
                          f"-{str(round(max_lls_defocus_offset, 3)).replace('.', 'p').replace('-', 'neg')}"

    outdir.mkdir(exist_ok=True, parents=True)

    phi = Wavefront(
        amplitudes=(min_amplitude, max_amplitude),
        order=gen.order,
        distribution=gen.distribution,
        mode_weights=gen.mode_weights,
        modes=gen.n_modes,
        gamma=gen.gamma,
        signed=gen.signed,
        rotate=gen.rotate,
        lam_detection=gen.lam_detection,
    )
    photons = randuniform((min_photons, max_photons))

    return simulate_psf(
        filename=filename,
        outdir=outdir,
        gen=gen,
        phi=phi,
        emb=emb,
        photons=photons,
        noise=noise,
        normalize=normalize,
        skip_remove_background=skip_remove_background,
        lls_defocus_offset=(min_lls_defocus_offset, max_lls_defocus_offset)
    )


def parse_args(args):
    parser = cli.argparser()

    parser.add_argument("--filename", type=str, default='1')
    parser.add_argument("--outdir", type=Path, default='../dataset')

    parser.add_argument(
        '--emb', action='store_true',
        help='toggle to save embeddings only'
    )

    parser.add_argument(
        "--iters", default=10, type=int,
        help='number of samples'
    )

    parser.add_argument(
        '--kernels', action='store_true',
        help='toggle to save raw kernels'
    )

    parser.add_argument(
        '--noise', action='store_true',
        help='toggle to add random background and shot noise to the generated PSFs'
    )

    parser.add_argument(
        '--normalize', action='store_true',
        help='toggle to scale the generated PSFs to 1.0'
    )

    parser.add_argument(
        "--x_voxel_size", default=.125, type=float,
        help='lateral voxel size in microns for X'
    )

    parser.add_argument(
        "--y_voxel_size", default=.125, type=float,
        help='lateral voxel size in microns for Y'
    )

    parser.add_argument(
        "--z_voxel_size", default=.2, type=float,
        help='axial voxel size in microns for Z'
    )

    parser.add_argument(
        "--input_shape", default=64, type=int,
        help="PSF input shape"
    )

    parser.add_argument(
        "--modes", default=55, type=int,
        help="number of modes to describe aberration"
    )

    parser.add_argument(
        "--min_photons", default=5000, type=int,
        help="minimum photons for training samples"
    )

    parser.add_argument(
        "--max_photons", default=10000, type=int,
        help="maximum photons for training samples"
    )

    parser.add_argument(
        "--psf_type", default='../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat', type=str,
        help='widefield, 2photon, confocal, or a path to an LLS excitation profile '
    )

    parser.add_argument(
        "--dist", default='single', type=str,
        help="distribution of the zernike amplitudes"
    )

    parser.add_argument(
        "--mode_dist", default='pyramid', type=str,
        help="distribution of the zernike modes"
    )

    parser.add_argument(
        "--gamma", default=.75, type=float,
        help="exponent for the powerlaw distribution"
    )

    parser.add_argument(
        '--signed', action='store_true',
        help='optional flag to generate a symmetric (pos/neg) semi-distributions for the given range of amplitudes'
    )

    parser.add_argument(
        '--rotate', action='store_true',
        help='optional flag to introduce a random radial rotation to each zernike mode'
    )

    parser.add_argument(
        "--min_amplitude", default=0, type=float,
        help="min amplitude for the zernike coefficients"
    )

    parser.add_argument(
        "--max_amplitude", default=.25, type=float,
        help="max amplitude for the zernike coefficients"
    )

    parser.add_argument(
        "--min_lls_defocus_offset", default=0, type=float,
        help="min value for the offset between the excitation and detection focal plan (microns)"
    )

    parser.add_argument(
        "--max_lls_defocus_offset", default=0, type=float,
        help="max value for the offset between the excitation and detection focal plan (microns)"
    )

    parser.add_argument(
        "--refractive_index", default=1.33, type=float,
        help="the quotient of the speed of light as it passes through two media"
    )

    parser.add_argument(
        "--na_detection", default=1.0, type=float,
        help="Numerical aperture"
    )

    parser.add_argument(
        "--lam_detection", default=.510, type=float,
        help='wavelength in microns'
    )

    parser.add_argument(
        "--cpu_workers", default=-1, type=int,
        help='number of CPU cores to use'
    )

    parser.add_argument(
        '--use_theoretical_widefield_simulator', action='store_true',
        help='optional toggle to use an experimental complex pupil '
             'to estimate amplitude attenuation (cosine factor)'
    )

    parser.add_argument(
        '--skip_remove_background', action='store_true',
        help='optional toggle to skip preprocessing input data using the DoG filter'
    )

    return parser.parse_args(args)


def main(args=None):
    timeit = time.time()
    args = parse_args(args)
    logger.info(args)

    gen = SyntheticPSF(
        order='ansi',
        amplitude_ranges=(args.min_amplitude, args.max_amplitude),
        cpu_workers=args.cpu_workers,
        n_modes=args.modes,
        distribution=args.dist,
        mode_weights=args.mode_dist,
        gamma=args.gamma,
        signed=args.signed,
        rotate=args.rotate,
        psf_type=args.psf_type,
        lam_detection=.920 if args.psf_type == '2photon' else args.lam_detection,
        psf_shape=3 * [args.input_shape],
        x_voxel_size=args.x_voxel_size,
        y_voxel_size=args.y_voxel_size,
        z_voxel_size=args.z_voxel_size,
        refractive_index=args.refractive_index,
        na_detection=args.na_detection,
        use_theoretical_widefield_simulator=args.use_theoretical_widefield_simulator,
        skip_remove_background_ideal_psf=args.skip_remove_background
    )

    sample = partial(
        create_synthetic_sample,
        gen=gen,
        emb=args.emb,
        savedir=args.outdir,
        noise=args.noise,
        normalize=args.normalize,
        min_amplitude=args.min_amplitude,
        max_amplitude=args.max_amplitude,
        min_lls_defocus_offset=args.min_lls_defocus_offset,
        max_lls_defocus_offset=args.max_lls_defocus_offset,
        min_photons=args.min_photons,
        max_photons=args.max_photons,
        skip_remove_background=args.skip_remove_background
    )

    jobs = [f"{int(args.filename)+k}" for k in range(args.iters)]
    multiprocess(func=sample, jobs=jobs, cores=args.cpu_workers)
    logging.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
