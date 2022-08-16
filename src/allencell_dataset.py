import logging
import sys
import time
import ujson
from pathlib import Path
from tifffile import imread, imsave
import numpy as np
from tqdm import trange
from scipy.signal import fftconvolve as scipy_fftconvolve
from skimage.filters import window, gaussian
from scipy.ndimage.morphology import distance_transform_edt


import cli
from preprocessing import resize
from utils import peak_aberration
from synthetic import SyntheticPSF

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_synthetic_sample(savepath, inputs, kernel, amps, snr, maxcounts, save_kernel=False):

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


def create_synthetic_sample(
    filename: str,
    sample: Path,
    outdir: Path,
    otf: bool,
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
    lam_detection: float,
    refractive_index: float,
    na_detection: float,
    cpu_workers: int,
    save_kernel: bool,
    debug: bool = True,
    apodization_dist: int = 10,
):
    gen = SyntheticPSF(
        order='ansi',
        cpu_workers=cpu_workers,
        n_modes=modes,
        max_jitter=0,
        snr=1000,
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

    savepath = outdir / f"z{modes}"
    savepath = savepath / f"amp_{str(round(min_amplitude, 3)).replace('0.', 'p').replace('-', 'neg')}" \
                          f"-{str(round(max_amplitude, 3)).replace('0.', 'p').replace('-', 'neg')}"
    savepath = savepath / sample.stem
    savepath.mkdir(exist_ok=True, parents=True)
    savepath = savepath / filename

    kernel, amps, snr, zplanes, maxcounts = gen.single_psf(
        phi=(min_amplitude, max_amplitude),
        zplanes=0,
        normed=True,
        noise=False,
        augmentation=True,
        meta=True
    )

    sample = imread(sample)
    conv = scipy_fftconvolve(sample, kernel, mode='full')
    conv /= np.nanpercentile(conv, 99.9)
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
        crop_shape=gen.psf_shape,
        voxel_size=gen.voxel_size,
        sample_voxel_size=gen.voxel_size,
    )

    circular_mask = window(('general_gaussian', apodization_dist/3, 2.5*apodization_dist), conv.shape)

    # corner_mask = np.zeros_like(conv, dtype=int)
    # corner_mask[1:-1, 1:-1, 1:-1] = 1.
    # corner_mask = distance_transform_edt(corner_mask, return_distances=True)
    # corner_mask = .5 - (.5 * np.cos((np.pi*corner_mask)/apodization_dist))
    # corner_mask[
    #     apodization_dist:conv.shape[0] - apodization_dist,
    #     apodization_dist:conv.shape[1] - apodization_dist,
    #     apodization_dist:conv.shape[2] - apodization_dist,
    # ] = 1.
    # # corner_mask = gaussian(corner_mask, sigma=2)
    #
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    # axes[0].imshow(circular_mask[circular_mask.shape[0]//2, :, :], cmap='magma')
    # axes[0].set_title('Circular mask')
    # axes[1].imshow(corner_mask[corner_mask.shape[0]//2, :, :], cmap='magma')
    # axes[1].set_title('Corner mask')
    # plt.show()

    inputs = conv * circular_mask

    if otf:
        inputs = gen.embedding(psf=inputs, plot=f"{savepath}_embedding" if debug else None)
        kernel = gen.embedding(psf=kernel, plot=f"{savepath}_kernel_embedding" if debug else None)

    save_synthetic_sample(
        savepath,
        inputs,
        kernel=kernel,
        amps=amps,
        snr=snr,
        maxcounts=maxcounts,
        save_kernel=save_kernel
    )


def parse_args(args):
    parser = cli.argparser()

    parser.add_argument("--filename", type=str, default='psf')
    parser.add_argument("--sample", type=Path, default='sample.tif')
    parser.add_argument("--outdir", type=Path, default='../dataset')

    parser.add_argument(
        '--otf', action='store_true',
        help='toggle to convert input to frequency space (OTF)'
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
        "--modes", default=60, type=int,
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
            sample=args.sample,
            outdir=args.outdir,
            otf=args.otf,
            save_kernel=args.kernels,
            modes=args.modes,
            input_shape=args.input_shape,
            psf_type=args.psf_type,
            distribution=args.dist,
            gamma=args.gamma,
            bimodal=args.bimodal,
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

    for i in trange(10):
        sample(k=i)

    logging.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
