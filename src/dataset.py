import logging
import sys
import time
import ujson
from pathlib import Path
from tifffile import imread, imsave
import numpy as np
from tqdm import trange


import cli
from utils import peak_aberration
from synthetic import SyntheticPSF

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_synthetic_sample(savepath, inputs, amps, snr, maxcounts):

    imsave(f"{savepath}.tif", inputs)
    # logger.info(f"Saved: {savepath}.tif")

    with Path(f"{savepath}.json").open('w') as f:
        json = dict(
            path=f"{savepath}.tif",
            zernikes=amps.tolist(),
            snr=int(snr),
            shape=inputs.shape,
            maxcounts=int(maxcounts),
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

    # logger.info(f"Saved: {savepath}.json")


def create_synthetic_sample(
    filename: str,
    outdir: Path,
    emb: bool,
    noise: bool,
    max_jitter: float,
    input_shape: int,
    modes: int,
    psf_type: str,
    distribution: str,
    gamma: float,
    bimodal: bool,
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
):
    gen = SyntheticPSF(
        order='ansi',
        cpu_workers=cpu_workers,
        n_modes=modes,
        max_jitter=max_jitter,
        snr=(min_psnr, max_psnr),
        dtype=psf_type,
        distribution=distribution,
        gamma=gamma,
        bimodal=bimodal,
        amplitude_ranges=(min_amplitude, max_amplitude),
        lam_detection=lam_detection,
        psf_shape=3*[input_shape],
        x_voxel_size=x_voxel_size,
        y_voxel_size=y_voxel_size,
        z_voxel_size=z_voxel_size,
        refractive_index=refractive_index,
        na_detection=na_detection,
    )

    outdir = outdir / f"x{round(x_voxel_size*1000)}-y{round(y_voxel_size*1000)}-z{round(z_voxel_size*1000)}"
    outdir = outdir / f"i{input_shape}"

    if distribution == 'powerlaw':
        outdir = outdir / f"powerlaw_gamma_{str(round(gamma, 2)).replace('.', 'p')}"
    else:
        outdir = outdir / f"{distribution}"

    if distribution == 'single':
        for i in range(5, modes):
            savepath = outdir / f"m{i}"
            savepath = savepath / f"psnr_{min_psnr}-{max_psnr}"
            savepath = savepath / f"amp_{str(round(min_amplitude, 3)).replace('0.', 'p').replace('-', 'neg')}" \
                                  f"-{str(round(max_amplitude, 3)).replace('0.', 'p').replace('-', 'neg')}"
            savepath.mkdir(exist_ok=True, parents=True)
            savepath = savepath / filename

            phi = np.zeros(modes)
            phi[i] = np.random.uniform(min_amplitude, max_amplitude)

            if emb:
                inputs, amps, snr, zplanes, maxcounts = gen.single_otf(
                    phi=phi,
                    zplanes=0,
                    normed=True,
                    noise=noise,
                    augmentation=noise,
                    meta=True,
                    na_mask=True,
                    ratio=True,
                    padsize=None,
                    plot=f"{savepath}_embedding"
                )
            else:
                inputs, amps, snr, zplanes, maxcounts = gen.single_psf(
                    phi=phi,
                    zplanes=0,
                    normed=True,
                    noise=noise,
                    augmentation=noise,
                    meta=True
                )

            save_synthetic_sample(savepath, inputs, amps, snr, maxcounts)
    else:
        savepath = outdir / f"z{modes}"
        savepath = savepath / f"psnr_{min_psnr}-{max_psnr}"
        savepath = savepath / f"amp_{str(round(min_amplitude, 3)).replace('0.', 'p').replace('-', 'neg')}" \
                              f"-{str(round(max_amplitude, 3)).replace('0.', 'p').replace('-', 'neg')}"
        savepath.mkdir(exist_ok=True, parents=True)
        savepath = savepath / filename

        if emb:
            inputs, amps, snr, zplanes, maxcounts = gen.single_otf(
                phi=(min_amplitude, max_amplitude),
                zplanes=0,
                normed=True,
                noise=True,
                augmentation=True,
                meta=True,
                na_mask=True,
                ratio=True,
                padsize=None,
                plot=f"{savepath}_embedding"
            )
        else:
            inputs, amps, snr, zplanes, maxcounts = gen.single_psf(
                phi=(min_amplitude, max_amplitude),
                zplanes=0,
                normed=True,
                noise=True,
                augmentation=True,
                meta=True
            )

        save_synthetic_sample(savepath, inputs, amps, snr, maxcounts)


def parse_args(args):
    parser = cli.argparser()

    parser.add_argument("--filename", type=str, default='sample')
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
        "--modes", default=60, type=int,
        help="number of modes to describe aberration"
    )

    parser.add_argument(
        "--min_psnr", default=10, type=int,
        help="minimum PSNR for training samples"
    )

    parser.add_argument(
        "--max_psnr", default=100, type=int,
        help="maximum PSNR for training samples"
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
        '--bimodal', action='store_true',
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
        "--max_jitter", default=1, type=float,
        help="max offset from center in microns"
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
            filename=f"{int(args.filename)+k}",
            outdir=args.outdir,
            emb=args.emb,
            noise=args.noise,
            modes=args.modes,
            input_shape=args.input_shape,
            psf_type=args.psf_type,
            distribution=args.dist,
            gamma=args.gamma,
            bimodal=args.bimodal,
            min_amplitude=args.min_amplitude,
            max_amplitude=args.max_amplitude,
            max_jitter=args.max_jitter,
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

    for i in trange(args.iters):
        sample(k=i)

    logging.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
