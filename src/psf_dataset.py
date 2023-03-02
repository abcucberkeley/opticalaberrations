
import numexpr
numexpr.set_num_threads(numexpr.detect_number_of_cores())

import matplotlib
matplotlib.use('Agg')

import logging
import sys
import os
import time
import ujson
from functools import partial
from typing import Any
from pathlib import Path
from tifffile import imsave
import numpy as np

import matplotlib.pyplot as plt
plt.set_loglevel('error')

import cli
from preprocessing import remove_background_noise
from utils import multiprocess
from synthetic import SyntheticPSF
from wavefront import Wavefront

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
    snr,
    maxcounts,
    p2v,
    gen,
    lls_defocus_offset=0.
):
    logger.info(f"Saved: {savepath}")
    imsave(f"{savepath}.tif", inputs)

    with Path(f"{savepath}.json").open('w') as f:
        json = dict(
            path=f"{savepath}.tif",
            n_modes=int(gen.n_modes),
            order=str(gen.order),
            zernikes=amps.tolist(),
            lls_defocus_offset=float(lls_defocus_offset),
            snr=int(snr),
            shape=inputs.shape,
            maxcounts=int(maxcounts),
            peak2peak=float(p2v),
            x_voxel_size=float(gen.x_voxel_size),
            y_voxel_size=float(gen.y_voxel_size),
            z_voxel_size=float(gen.z_voxel_size),
            wavelength=float(gen.lam_detection),
            na_detection=float(gen.na_detection),
            refractive_index=float(gen.refractive_index),
            mode_weights=str(gen.mode_weights),
            distribution=str(gen.distribution),
        )

        ujson.dump(
            json,
            f,
            indent=4,
            sort_keys=False,
            ensure_ascii=False,
            escape_forward_slashes=False
        )


def sim(
    filename: str,
    outdir: Path,
    gen: SyntheticPSF,
    snr: tuple,
    noise: bool = True,
    normalize: bool = True,
    remove_background: bool = True,
    lls_defocus_offset: Any = 0.
):
    np.random.seed(os.getpid()+np.random.randint(low=0, high=10**6))

    # aberrated PSF without noise
    kernel, amps, estsnr, maxcounts, lls_defocus_offset = gen.single_psf(
        phi=gen.amplitude_ranges,
        lls_defocus_offset=lls_defocus_offset,
        normed=True,
        noise=False,
        meta=True,
    )

    p2v = Wavefront(amps, lam_detection=gen.lam_detection).peak2valley(na=1.0)
    psnr = gen._randuniform(snr)
    img = kernel * psnr ** 2

    if noise:
        rand_noise = gen._random_noise(
            image=img,
            mean=gen.mean_background_noise,
            sigma=gen.sigma_background_noise
        )
        noisy_img = rand_noise + img
        maxcounts = np.max(noisy_img)
    else:
        maxcounts = np.max(img)
        noisy_img = img

    if remove_background:
        noisy_img = remove_background_noise(noisy_img)

    if normalize:
        noisy_img /= np.max(noisy_img)

    save_synthetic_sample(
        outdir/filename,
        noisy_img,
        amps=amps,
        snr=psnr,
        maxcounts=maxcounts,
        p2v=p2v,
        gen=gen,
        lls_defocus_offset=lls_defocus_offset
    )


def create_synthetic_sample(
    filename: str,
    outdir: Path,
    input_shape: int,
    modes: int,
    psf_type: str,
    distribution: str,
    mode_dist: str,
    gamma: float,
    signed: bool,
    rotate: bool,
    min_amplitude: float,
    max_amplitude: float,
    x_voxel_size: float,
    y_voxel_size: float,
    z_voxel_size: float,
    min_psnr: int,
    max_psnr: int,
    lam_detection: float,
    refractive_index: float,
    na_detection: float,
    cpu_workers: int,
    noise: bool,
    normalize: bool,
    min_lls_defocus_offset: float = 0.,
    max_lls_defocus_offset: float = 0.,
):
    gen = SyntheticPSF(
        order='ansi',
        cpu_workers=cpu_workers,
        n_modes=modes,
        snr=1000,
        psf_type=psf_type,
        distribution=distribution,
        mode_weights=mode_dist,
        gamma=gamma,
        signed=signed,
        rotate=rotate,
        amplitude_ranges=(min_amplitude, max_amplitude),
        lam_detection=lam_detection,
        psf_shape=3*[input_shape],
        x_voxel_size=x_voxel_size,
        y_voxel_size=y_voxel_size,
        z_voxel_size=z_voxel_size,
        refractive_index=refractive_index,
        na_detection=na_detection,
    )

    outdir = outdir / f"x{round(x_voxel_size * 1000)}-y{round(y_voxel_size * 1000)}-z{round(z_voxel_size * 1000)}"
    outdir = outdir / f"i{input_shape}"
    outdir = outdir / f"z{modes}"

    if distribution == 'powerlaw':
        outdir = outdir / f"powerlaw_gamma_{str(round(gamma, 2)).replace('.', 'p')}"
    else:
        outdir = outdir / f"{distribution}"

    outdir = outdir / f"psnr_{min_psnr}-{max_psnr}"
    outdir = outdir / f"amp_{str(round(min_amplitude, 3)).replace('0.', 'p').replace('-', 'neg')}" \
                          f"-{str(round(max_amplitude, 3)).replace('0.', 'p').replace('-', 'neg')}"

    outdir.mkdir(exist_ok=True, parents=True)

    sim(
        filename=filename,
        outdir=outdir,
        gen=gen,
        snr=(min_psnr, max_psnr),
        noise=noise,
        normalize=normalize,
        lls_defocus_offset=(min_lls_defocus_offset, max_lls_defocus_offset)
    )


def parse_args(args):
    parser = cli.argparser()

    parser.add_argument("--filename", type=str, default='1')
    parser.add_argument("--outdir", type=Path, default='../dataset')

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
        "--x_voxel_size", default=.108, type=float,
        help='lateral voxel size in microns for X'
    )

    parser.add_argument(
        "--y_voxel_size", default=.108, type=float,
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
        "--min_psnr", default=10, type=int,
        help="minimum PSNR for training samples"
    )

    parser.add_argument(
        "--max_psnr", default=50, type=int,
        help="maximum PSNR for training samples"
    )

    parser.add_argument(
        "--psf_type", default='widefield', type=str,
        help="widefield or confocal"
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

    return parser.parse_args(args)


def main(args=None):
    timeit = time.time()
    args = parse_args(args)
    logger.info(args)

    sample = partial(
        create_synthetic_sample,
        outdir=args.outdir,
        noise=args.noise,
        normalize=args.normalize,
        modes=args.modes,
        input_shape=args.input_shape,
        psf_type=args.psf_type,
        distribution=args.dist,
        mode_dist=args.mode_dist,
        gamma=args.gamma,
        signed=args.signed,
        rotate=args.rotate,
        min_amplitude=args.min_amplitude,
        max_amplitude=args.max_amplitude,
        min_lls_defocus_offset=args.min_lls_defocus_offset,
        max_lls_defocus_offset=args.max_lls_defocus_offset,
        x_voxel_size=args.x_voxel_size,
        y_voxel_size=args.y_voxel_size,
        z_voxel_size=args.z_voxel_size,
        min_psnr=args.min_psnr,
        max_psnr=args.max_psnr,
        lam_detection=args.lam_detection,
        refractive_index=args.refractive_index,
        na_detection=args.na_detection,
        cpu_workers=args.cpu_workers,
    )

    jobs = [f"{int(args.filename)+k}" for k in range(args.iters)]
    multiprocess(sample, jobs=jobs, cores=args.cpu_workers)
    logging.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
