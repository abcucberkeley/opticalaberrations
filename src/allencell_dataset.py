import logging
import sys
import time
import ujson
from pathlib import Path
from tifffile import imread, imsave
import numpy as np
from tqdm import trange
from scipy.signal import fftconvolve as scipy_fftconvolve
from skimage.filters import threshold_otsu

import cli
from synthetic import SyntheticPSF
from wavefront import Wavefront

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_synthetic_sample(savepath, inputs, kernel, amps, snr, maxcounts, p2v, save_kernel=False):

    logger.info(f"Saved: {savepath}")
    imsave(f"{savepath}.tif", inputs)
    if save_kernel:
        imsave(f"{savepath}_kernel.tif", kernel)

    with Path(f"{savepath}.json").open('w') as f:
        json = dict(
            path=f"{savepath}.tif",
            zernikes=amps.tolist(),
            snr=int(snr),
            shape=inputs.shape,
            maxcounts=int(maxcounts),
            peak2peak=float(p2v)
        )

        ujson.dump(
            json,
            f,
            indent=4,
            sort_keys=False,
            ensure_ascii=False,
            escape_forward_slashes=False
        )


def convolve(
    savepath: Path,
    gen: SyntheticPSF,
    sample: Path,
    kernel: np.array,
    amps: np.array,
    snr: int,
    maxcounts: int,
    save_kernel: bool,
    strides: int,
    log10: bool = True,
    apodization: bool = True,
    rolling_embedding: bool = True,
    debug: bool = False,
):
    # remove background
    sample = imread(sample)
    t = threshold_otsu(sample)
    sample[sample <= t] = 0.

    # convolve with a PSF
    conv = scipy_fftconvolve(sample, kernel, mode='full')
    conv /= np.nanpercentile(conv, 99.99)
    conv[conv > 1] = 1
    conv = np.nan_to_num(conv, nan=0)

    width = [(i // 2) for i in sample.shape]
    center = [(i // 2) + 1 for i in conv.shape]
    conv = conv[
       center[0] - width[0]:center[0] + width[0],
       center[1] - width[1]:center[1] + width[1],
       center[2] - width[2]:center[2] + width[2],
    ]

    conv = resize(
        conv,
        crop_shape=[conv.shape[i] if conv.shape[i] >= gen.psf_shape[i] else gen.psf_shape[i] for i in range(3)],
        voxel_size=gen.voxel_size,
        sample_voxel_size=gen.voxel_size,
    )
    # imsave(f"{savepath}_conv.tif", conv)

    kernel = gen.embedding(kernel, log10=True, plot=f"{savepath}_kernel_embedding" if debug else None)

    emb = gen.embedding(
        conv,
        log10=log10,
        plot=f"{savepath}_embedding" if debug else None
    )

    save_synthetic_sample(
        savepath,
        emb,
        kernel=kernel,
        amps=amps,
        snr=snr,
        maxcounts=maxcounts,
        save_kernel=save_kernel,
        p2v=Wavefront(amps, lam_detection=gen.lam_detection).peak2valley(),
    )


def create_synthetic_sample(
    filename: str,
    sample: Path,
    outdir: Path,
    input_shape: int,
    modes: int,
    psf_type: str,
    distribution: str,
    gamma: float,
    signed: bool,
    min_amplitude: float,
    max_amplitude: float,
    x_voxel_size: float,
    y_voxel_size: float,
    z_voxel_size: float,
    lam_detection: float,
    refractive_index: float,
    na_detection: float,
    cpu_workers: int,
    save_kernel: bool,
):
    gen = SyntheticPSF(
        order='ansi',
        cpu_workers=cpu_workers,
        n_modes=modes,
        snr=1000,
        psf_type=psf_type,
        distribution=distribution,
        gamma=gamma,
        signed=signed,
        amplitude_ranges=(min_amplitude, max_amplitude),
        lam_detection=lam_detection,
        psf_shape=3*[input_shape],
        x_voxel_size=x_voxel_size,
        y_voxel_size=y_voxel_size,
        z_voxel_size=z_voxel_size,
        refractive_index=refractive_index,
        na_detection=na_detection,
    )

    if distribution == 'single':
        outdir = outdir / f"x{round(x_voxel_size * 1000)}-y{round(y_voxel_size * 1000)}-z{round(z_voxel_size * 1000)}"
        outdir = outdir / f"i{input_shape}"

        if distribution == 'powerlaw':
            outdir = outdir / f"powerlaw_gamma_{str(round(gamma, 2)).replace('.', 'p')}"
        else:
            outdir = outdir / f"{distribution}"

        outdir = outdir / f"z{modes}"
        outdir = outdir / f"amp_{str(round(min_amplitude, 3)).replace('0.', 'p').replace('-', 'neg')}" \
                              f"-{str(round(max_amplitude, 3)).replace('0.', 'p').replace('-', 'neg')}"

        for i in range(5, 16):
            savepath = outdir / f"m{i}"

            phi = np.zeros(modes)
            phi[i] = np.random.uniform(min_amplitude, max_amplitude)

            kernel, amps, snr, maxcounts, lls_defocus_offset = gen.single_psf(
                phi=phi,
                normed=True,
                noise=False,
                meta=True
            )
            savepath = savepath / sample.stem
            savepath.mkdir(exist_ok=True, parents=True)
            savepath = savepath / filename

            convolve(
                savepath=savepath,
                gen=gen,
                sample=sample,
                kernel=kernel,
                amps=amps,
                snr=snr,
                maxcounts=maxcounts,
                save_kernel=save_kernel,
                strides=input_shape//2,
            )

    else:
        kernel, amps, snr, maxcounts, lls_defocus_offset = gen.single_psf(
            phi=(min_amplitude, max_amplitude),
            normed=True,
            noise=False,
            meta=True
        )

        outdir = outdir / f"x{round(x_voxel_size * 1000)}-y{round(y_voxel_size * 1000)}-z{round(z_voxel_size * 1000)}"
        outdir = outdir / f"i{input_shape}"

        if distribution == 'powerlaw':
            outdir = outdir / f"powerlaw_gamma_{str(round(gamma, 2)).replace('.', 'p')}"
        else:
            outdir = outdir / f"{distribution}"

        savepath = outdir / f"z{modes}"
        savepath = savepath / f"amp_{str(round(min_amplitude, 3)).replace('0.', 'p').replace('-', 'neg')}" \
                              f"-{str(round(max_amplitude, 3)).replace('0.', 'p').replace('-', 'neg')}"

        savepath = savepath / sample.stem
        savepath.mkdir(exist_ok=True, parents=True)
        savepath = savepath / filename

        convolve(
            savepath=savepath,
            gen=gen,
            sample=sample,
            kernel=kernel,
            amps=amps,
            snr=snr,
            maxcounts=maxcounts,
            save_kernel=save_kernel,
            strides=input_shape//2,
        )


def parse_args(args):
    parser = cli.argparser()

    parser.add_argument("--sample", type=Path, default='sample.tif')
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
        "--x_voxel_size", default=.15, type=float,
        help='lateral voxel size in microns for X'
    )

    parser.add_argument(
        "--y_voxel_size", default=.15, type=float,
        help='lateral voxel size in microns for Y'
    )

    parser.add_argument(
        "--z_voxel_size", default=.6, type=float,
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
        "--psf_type", default='widefield', type=str,
        help="widefield or confocal"
    )

    parser.add_argument(
        "--dist", default='powerlaw', type=str,
        help="distribution of the zernike amplitudes"
    )

    parser.add_argument(
        "--gamma", default=1.5, type=float,
        help="exponent for the powerlaw distribution"
    )

    parser.add_argument(
        '--signed', action='store_true',
        help='optional flag to generate a symmetric (pos/neg) semi-distributions for the given range of amplitudes'
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
        "--lam_detection", default=.605, type=float,
        help='wavelength in microns'
    )

    parser.add_argument(
        "--cpu_workers", default=1, type=int,
        help='number of CPU cores to use'
    )

    return parser.parse_args(args)


def main(args=None):
    timeit = time.time()
    args = parse_args(args)
    logger.info(args)

    def sample(k):
        return create_synthetic_sample(
            filename=f"{k}",
            sample=args.sample,
            outdir=args.outdir,
            save_kernel=args.kernels,
            modes=args.modes,
            input_shape=args.input_shape,
            psf_type=args.psf_type,
            distribution=args.dist,
            gamma=args.gamma,
            signed=args.signed,
            min_amplitude=args.min_amplitude,
            max_amplitude=args.max_amplitude,
            x_voxel_size=args.x_voxel_size,
            y_voxel_size=args.y_voxel_size,
            z_voxel_size=args.z_voxel_size,
            lam_detection=args.lam_detection,
            refractive_index=args.refractive_index,
            na_detection=args.na_detection,
            cpu_workers=args.cpu_workers,
        )

    for i in trange(1, args.iters+1):
        sample(k=i)

    logging.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
