import numexpr
numexpr.set_num_threads(numexpr.detect_number_of_cores())

import matplotlib
matplotlib.use('Agg')

import logging
import sys
import time
import ujson
from functools import partial
from typing import Any
from pathlib import Path
from tifffile import imsave
import numpy as np
import scipy.stats as st
import raster_geometry as rg

import cli
from preprocessing import resize_with_crop_or_pad
from utils import peak_aberration, fftconvolution, multiprocess
from synthetic import SyntheticPSF

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_synthetic_sample(savepath, inputs, amps, snr, maxcounts, npoints=1):

    logger.info(f"Saved: {savepath}")
    imsave(f"{savepath}.tif", inputs)

    with Path(f"{savepath}.json").open('w') as f:
        json = dict(
            path=f"{savepath}.tif",
            zernikes=amps.tolist(),
            snr=int(snr),
            shape=inputs.shape,
            maxcounts=int(maxcounts),
            npoints=int(npoints),
            peak2peak=float(peak_aberration(amps))
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
    savepath: Path,
    gen: SyntheticPSF,
    npoints: int,
    kernel: np.array,
    amps: np.array,
    snr: tuple,
    emb: bool = True,
    noise: bool = True,
    normalize: bool = True,
    random_crop: Any = None,
    radius: float = .4,
    alpha_val: str = 'abs',
    phi_val: str = 'angle',
):
    reference = np.zeros(gen.psf_shape)
    for i in range(npoints):
        sphere_radius = np.random.choice([0, 1, 2], p=[.95, .04, .01])

        if sphere_radius > 0:
            reference += rg.sphere(
                shape=gen.psf_shape,
                radius=sphere_radius,
                position=np.random.uniform(low=.2, high=.8, size=3)
            ).astype(np.float) * np.random.random()
        else:
            reference[
                np.random.randint(int(gen.psf_shape[0] * (.5 - radius)), int(gen.psf_shape[0] * (.5 + radius))),
                np.random.randint(int(gen.psf_shape[1] * (.5 - radius)), int(gen.psf_shape[1] * (.5 + radius))),
                np.random.randint(int(gen.psf_shape[2] * (.5 - radius)), int(gen.psf_shape[2] * (.5 + radius)))
            ] += np.random.random()

    reference /= np.max(reference)
    img = fftconvolution(reference, kernel)
    snr = gen._randuniform(snr)
    img *= snr ** 2

    if noise:
        rand_noise = gen._random_noise(
            image=img,
            mean=gen.mean_background_noise,
            sigma=gen.sigma_background_noise
        )
        noisy_img = rand_noise + img
        psnr = np.sqrt(np.max(noisy_img))
        maxcounts = np.max(noisy_img)
    else:
        psnr = np.mean(np.array(snr))
        maxcounts = np.max(img)
        noisy_img = img

    if random_crop is not None:
        mode = np.abs(st.mode(noisy_img, axis=None).mode[0])
        crop = int(np.random.uniform(low=random_crop, high=gen.psf_shape[0]+1))
        noisy_img = resize_with_crop_or_pad(noisy_img, crop_shape=[crop]*3)
        noisy_img = resize_with_crop_or_pad(noisy_img, crop_shape=gen.psf_shape, constant_values=mode)

    if normalize:
        noisy_img /= np.max(noisy_img)

    if emb:
        noisy_img = gen.embedding(
            psf=noisy_img,
            principle_planes=True,
            alpha_val=alpha_val,
            phi_val=phi_val,
            plot=f"{savepath}_embedding"
        )

    save_synthetic_sample(
        savepath,
        noisy_img,
        amps=amps,
        snr=psnr,
        maxcounts=maxcounts,
        npoints=npoints,
    )


def create_synthetic_sample(
    filename: str,
    npoints: int,
    outdir: Path,
    input_shape: int,
    modes: int,
    psf_type: str,
    distribution: str,
    mode_dist: str,
    gamma: float,
    bimodal: bool,
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
    random_crop: Any,
    noise: bool,
    normalize: bool,
    emb: bool,
    alpha_val: str = 'abs',
    phi_val: str = 'angle',
):
    gen = SyntheticPSF(
        order='ansi',
        cpu_workers=cpu_workers,
        n_modes=modes,
        snr=1000,
        max_jitter=0,
        psf_type=psf_type,
        distribution=distribution,
        mode_weights=mode_dist,
        gamma=gamma,
        bimodal=bimodal,
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

    # theoretical kernel without noise
    kernel, amps, phi, maxcounts = gen.single_psf(
        phi=(min_amplitude, max_amplitude),
        normed=True,
        noise=False,
        augmentation=False,
        meta=True,
    )

    outdir = outdir / f"x{round(x_voxel_size * 1000)}-y{round(y_voxel_size * 1000)}-z{round(z_voxel_size * 1000)}"
    outdir = outdir / f"i{input_shape}"

    if distribution == 'powerlaw':
        outdir = outdir / f"powerlaw_gamma_{str(round(gamma, 2)).replace('.', 'p')}"
    else:
        outdir = outdir / f"{distribution}"

    savepath = outdir / f"z{modes}"
    savepath = savepath / f"psnr_{min_psnr}-{max_psnr}"
    savepath = savepath / f"amp_{str(round(min_amplitude, 3)).replace('0.', 'p').replace('-', 'neg')}" \
                          f"-{str(round(max_amplitude, 3)).replace('0.', 'p').replace('-', 'neg')}"

    savepath = savepath / f"npoints_{npoints}"
    savepath.mkdir(exist_ok=True, parents=True)
    savepath = savepath / filename

    sim(
        savepath=savepath,
        gen=gen,
        npoints=npoints,
        kernel=kernel,
        amps=amps,
        snr=(min_psnr, max_psnr),
        emb=emb,
        random_crop=random_crop,
        noise=noise,
        normalize=normalize,
        alpha_val=alpha_val,
        phi_val=phi_val,
    )


def parse_args(args):
    parser = cli.argparser()

    parser.add_argument("--filename", type=str, default='sample')
    parser.add_argument("--npoints", type=int, default=1)
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
        "--random_crop", default=None, type=int,
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
        "--mode_dist", default='uniform', type=str,
        help="distribution of the zernike modes"
    )

    parser.add_argument(
        "--gamma", default=.75, type=float,
        help="exponent for the powerlaw distribution"
    )

    parser.add_argument(
        '--bimodal', action='store_true',
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
        "--alpha_val", default='abs', type=str,
        help="values to use for the `alpha` embedding [options: real, abs]"
    )

    parser.add_argument(
        "--phi_val", default='angle', type=str,
        help="values to use for the `phi` embedding [options: angle, imag, abs]"
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
        emb=args.emb,
        alpha_val=args.alpha_val,
        phi_val=args.phi_val,
        npoints=args.npoints,
        outdir=args.outdir,
        noise=args.noise,
        normalize=args.normalize,
        modes=args.modes,
        input_shape=args.input_shape,
        psf_type=args.psf_type,
        distribution=args.dist,
        mode_dist=args.mode_dist,
        random_crop=args.random_crop,
        gamma=args.gamma,
        bimodal=args.bimodal,
        rotate=args.rotate,
        min_amplitude=args.min_amplitude,
        max_amplitude=args.max_amplitude,
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
