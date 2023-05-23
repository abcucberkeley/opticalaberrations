
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
from typing import Any, Optional
from pathlib import Path
from tifffile import imwrite
import numpy as np
import raster_geometry as rg

import matplotlib.pyplot as plt
plt.set_loglevel('error')

import cli
from utils import mean_min_distance, randuniform, add_noise
from preprocessing import prep_sample, resize_with_crop_or_pad
from utils import fftconvolution, multiprocess
from synthetic import SyntheticPSF
from embeddings import fourier_embeddings
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
    photons,
    maxcounts,
    p2v,
    avg_min_distance,
    gen,
    lls_defocus_offset=0.,
    npoints=1,
    gt=None,
    realspace=None
):

    if gt is not None:
        imwrite(f"{savepath}_gt.tif", gt)

    if realspace is not None:
        imwrite(f"{savepath}_realspace.tif", realspace)

    logger.info(f"Saved: {savepath}")
    imwrite(f"{savepath}.tif", inputs)

    with Path(f"{savepath}.json").open('w') as f:
        json = dict(
            path=f"{savepath}.tif",
            n_modes=int(gen.n_modes),
            order=str(gen.order),
            zernikes=amps.tolist(),
            lls_defocus_offset=float(lls_defocus_offset),
            photons=int(photons),
            shape=inputs.shape,
            maxcounts=int(maxcounts),
            npoints=int(npoints),
            peak2peak=float(p2v),
            avg_min_distance=float(avg_min_distance),
            x_voxel_size=float(gen.x_voxel_size),
            y_voxel_size=float(gen.y_voxel_size),
            z_voxel_size=float(gen.z_voxel_size),
            wavelength=float(gen.lam_detection),
            na_detection=float(gen.na_detection),
            refractive_index=float(gen.refractive_index),
            mode_weights=str(gen.mode_weights),
            embedding_option=str(gen.embedding_option),
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


def beads(
    image_shape: tuple,
    object_size: Optional[int] = 0,
    num_objs: int = 1,
    fill_radius: float = .4,
):
    """
    Args:
        image_shape: image size
        object_size: bead size (0 for diffraction-limited beads)
        num_objs: number of beads
        fill_radius: (0 for a single bead at the center of the image)
    """
    np.random.seed(os.getpid()+np.random.randint(low=0, high=10**6))
    reference = np.zeros(image_shape)
    rng = np.random.default_rng()

    for i in range(num_objs):
        if object_size is None:
            object_size = np.random.choice([0, 1, 2], p=[.95, .04, .01])

        if object_size > 0:
            reference += rg.sphere(
                shape=image_shape,
                radius=object_size,
                position=rng.integers(
                    int(image_shape[0] * (.5 - fill_radius)), int(image_shape[0] * (.5 + fill_radius)),
                    3
                ),
            ).astype(np.float32)
        else:
            if fill_radius > 0:
                reference[
                    rng.integers(int(image_shape[0] * (.5 - fill_radius)), int(image_shape[0] * (.5 + fill_radius))),
                    rng.integers(int(image_shape[1] * (.5 - fill_radius)), int(image_shape[0] * (.5 + fill_radius))),
                    rng.integers(int(image_shape[2] * (.5 - fill_radius)), int(image_shape[2] * (.5 + fill_radius))),
                ] = 1
            else:
                reference[image_shape[0] // 2, image_shape[1] // 2, image_shape[2] // 2] = 1

    return reference


def sim(
    filename: str,
    outdir: Path,
    gen: SyntheticPSF,
    npoints: int,
    photons: tuple,
    emb: bool = True,
    noise: bool = True,
    normalize: bool = True,
    remove_background: bool = True,
    random_crop: Any = None,
    embedding_option: list = (),
    alpha_val: str = 'abs',
    phi_val: str = 'angle',
    lls_defocus_offset: Any = 0.,
    sigma_background_noise=40,
    mean_background_offset=100,
    electrons_per_count: float = .22,
    quantum_efficiency: float = .82,
):
    photons = randuniform(photons)
    reference = photons * beads(
        image_shape=gen.psf_shape,
        object_size=None,
        num_objs=npoints,
    )

    # aberrated PSF without noise
    kernel, amps, lls_defocus_offset = gen.single_psf(
        phi=gen.amplitude_ranges,
        lls_defocus_offset=lls_defocus_offset,
        normed=True,
        meta=True,
    )
    kernel /= np.sum(kernel)

    img = fftconvolution(sample=reference, kernel=kernel)  # image in photons

    p2v = Wavefront(amps, lam_detection=gen.lam_detection).peak2valley(na=1.0)
    avg_min_distance = mean_min_distance(reference, voxel_size=gen.voxel_size) if npoints > 1 else 0.

    if noise:
        inputs = add_noise(
            img,
            mean_background_offset=mean_background_offset,
            sigma_background_noise=sigma_background_noise,
            quantum_efficiency=quantum_efficiency,
            electrons_per_count=electrons_per_count,
        )
    else:  # convert image to counts
        inputs = img / electrons_per_count

    maxcounts = np.max(inputs)

    if random_crop is not None:
        crop = int(np.random.uniform(low=random_crop, high=gen.psf_shape[0]+1))
        inputs = resize_with_crop_or_pad(inputs, crop_shape=[crop]*3)

    if emb:
        for e in set(embedding_option):
            odir = outdir/e
            odir.mkdir(exist_ok=True, parents=True)

            inputs = prep_sample(
                inputs,
                sample_voxel_size=gen.voxel_size,
                model_fov=gen.psf_fov,
                remove_background=remove_background,
                normalize=normalize,
                edge_filter=False,
                filter_mask_dilation=False,
                read_noise_bias=5,
                plot=odir/filename,
            )

            embeddings = np.squeeze(fourier_embeddings(
                inputs=inputs,
                iotf=gen.iotf,
                embedding_option=e,
                alpha_val=alpha_val,
                phi_val=phi_val,
                plot=odir/filename
            ))

            save_synthetic_sample(
                odir/filename,
                embeddings,
                amps=amps,
                photons=photons,
                maxcounts=maxcounts,
                npoints=npoints,
                p2v=p2v,
                gt=reference,
                gen=gen,
                realspace=inputs,
                avg_min_distance=avg_min_distance,
                lls_defocus_offset=lls_defocus_offset
            )
    else:
        inputs = prep_sample(
            inputs,
            sample_voxel_size=gen.voxel_size,
            model_fov=gen.psf_fov,
            remove_background=remove_background,
            normalize=normalize,
            edge_filter=False,
            filter_mask_dilation=False,
            read_noise_bias=5,
            plot=outdir/filename,
        )

        save_synthetic_sample(
            outdir/filename,
            inputs,
            amps=amps,
            photons=photons,
            maxcounts=maxcounts,
            npoints=npoints,
            avg_min_distance=avg_min_distance,
            p2v=p2v,
            gt=reference,
            gen=gen,
            lls_defocus_offset=lls_defocus_offset
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
    signed: bool,
    randomize_voxel_size: bool,
    rotate: bool,
    min_amplitude: float,
    max_amplitude: float,
    x_voxel_size: float,
    y_voxel_size: float,
    z_voxel_size: float,
    min_photons: int,
    max_photons: int,
    lam_detection: float,
    refractive_index: float,
    na_detection: float,
    cpu_workers: int,
    random_crop: Any,
    noise: bool,
    normalize: bool,
    fog: bool,
    emb: bool,
    embedding_option: list,
    alpha_val: str = 'abs',
    phi_val: str = 'angle',
    min_lls_defocus_offset: float = 0.,
    max_lls_defocus_offset: float = 0.,
):
    if randomize_voxel_size:
        x_voxel_size = np.random.uniform(low=x_voxel_size-.025, high=x_voxel_size+.025)
        y_voxel_size = np.random.uniform(low=y_voxel_size - .025, high=y_voxel_size + .025)
        z_voxel_size = np.random.uniform(low=z_voxel_size - .05, high=z_voxel_size + .05)

    gen = SyntheticPSF(
        order='ansi',
        cpu_workers=cpu_workers,
        n_modes=modes,
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

    #outdir = outdir / f"x{round(x_voxel_size * 1000)}-y{round(y_voxel_size * 1000)}-z{round(z_voxel_size * 1000)}"
    outdir = outdir / f"i{input_shape}"
    outdir = outdir / f"z{modes}"

    if distribution == 'powerlaw':
        outdir = outdir / f"powerlaw_gamma_{str(round(gamma, 2)).replace('.', 'p')}"
    else:
        outdir = outdir / f"{distribution}"

    outdir = outdir / f"photons_{min_photons}-{max_photons}"
    outdir = outdir / f"amp_{str(round(min_amplitude, 3)).replace('0.', 'p').replace('-', 'neg')}" \
                      f"-{str(round(max_amplitude, 3)).replace('0.', 'p').replace('-', 'neg')}"

    outdir = outdir / f"npoints_{npoints}"
    outdir.mkdir(exist_ok=True, parents=True)

    sim(
        filename=filename,
        outdir=outdir,
        gen=gen,
        npoints=npoints,
        photons=(min_photons, max_photons),
        emb=emb,
        embedding_option=embedding_option,
        random_crop=random_crop,
        noise=noise,
        normalize=normalize,
        alpha_val=alpha_val,
        phi_val=phi_val,
        lls_defocus_offset=(min_lls_defocus_offset, max_lls_defocus_offset)
    )


def parse_args(args):
    parser = cli.argparser()

    parser.add_argument("--filename", type=str, default='1')
    parser.add_argument("--npoints", type=int, default=1)
    parser.add_argument("--outdir", type=Path, default='../dataset')

    parser.add_argument(
        '--emb', action='store_true',
        help='toggle to save embeddings only'
    )

    parser.add_argument(
        "--embedding_option", action='append', default=['spatial_planes'],
        help='type of embedding to use: ["spatial_planes", "principle_planes", "rotary_slices", "spatial_quadrants"]'
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
        '--fog', action='store_true',
        help='toggle to add a random hazy background'
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
        "--min_photons", default=5000, type=int,
        help="minimum photons for training samples"
    )

    parser.add_argument(
        "--max_photons", default=10000, type=int,
        help="maximum photons for training samples"
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
        '--randomize_voxel_size', action='store_true',
        help='optional flag to randomize voxel size during training'
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
        embedding_option=args.embedding_option,
        alpha_val=args.alpha_val,
        phi_val=args.phi_val,
        npoints=args.npoints,
        outdir=args.outdir,
        noise=args.noise,
        normalize=args.normalize,
        fog=args.fog,
        modes=args.modes,
        input_shape=args.input_shape,
        psf_type=args.psf_type,
        distribution=args.dist,
        mode_dist=args.mode_dist,
        random_crop=args.random_crop,
        gamma=args.gamma,
        signed=args.signed,
        randomize_voxel_size=args.randomize_voxel_size,
        rotate=args.rotate,
        min_amplitude=args.min_amplitude,
        max_amplitude=args.max_amplitude,
        min_lls_defocus_offset=args.min_lls_defocus_offset,
        max_lls_defocus_offset=args.max_lls_defocus_offset,
        x_voxel_size=args.x_voxel_size,
        y_voxel_size=args.y_voxel_size,
        z_voxel_size=args.z_voxel_size,
        min_photons=args.min_photons,
        max_photons=args.max_photons,
        lam_detection=args.lam_detection,
        refractive_index=args.refractive_index,
        na_detection=args.na_detection,
        cpu_workers=args.cpu_workers,
    )

    jobs = [f"{int(args.filename)+k}" for k in range(args.iters)]
    multiprocess(func=sample, jobs=jobs, cores=args.cpu_workers)
    logging.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
