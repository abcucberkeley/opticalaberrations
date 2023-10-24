
import matplotlib
matplotlib.use('Agg')

import logging
import sys
import os
import time
import ujson
from functools import partial
from typing import Any, Optional, Union
from pathlib import Path
from tifffile import TiffFile
from tifffile import imwrite
import numpy as np
import raster_geometry as rg
from scipy import stats as st

import matplotlib.pyplot as plt
plt.set_loglevel('error')

import cli
from utils import mean_min_distance, randuniform, add_noise
from utils import electrons2counts, photons2electrons, counts2electrons, electrons2photons
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
    counts,
    p2v,
    avg_min_distance,
    gen,
    lls_defocus_offset=0.,
    npoints=1,
    gt=None,
    realspace=None,
    counts_mode=None,
    counts_percentiles=None,
    sigma_background_noise=None,
    mean_background_offset=None,
    electrons_per_count=None,
    quantum_efficiency=None,
    psf_type=None,
):
    if gt is not None:
        imwrite(f"{savepath}_gt.tif", gt.astype(np.float32), compression='deflate')

    if realspace is not None:
        imwrite(f"{savepath}_realspace.tif", realspace.astype(np.float32), compression='deflate')

    imwrite(f"{savepath}.tif", inputs.astype(np.float32), compression='deflate')
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
            npoints=int(npoints),
            peak2peak=float(p2v),
            avg_min_distance=float(avg_min_distance),
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


def beads(
    image_shape: tuple,
    photons: int = 1,
    object_size: Optional[int] = 0,
    num_objs: int = 1,
    fill_radius: float = .66,           # .66 will be roughly a bit inside of the Tukey window
    zborder: int = 10
):
    """
    Args:
        image_shape: image size
        object_size: bead size (0 for diffraction-limited beads)
        num_objs: number of beads
        fill_radius: Fractional (0 for a single bead at the center of the image)
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
                    int(image_shape[0] * (.5 - fill_radius)), int(image_shape[0] * (.5 + fill_radius)), 3
                ),
            ).astype(np.float32) * photons
        else:
            # object_size = 0 diffraction-limited
            if fill_radius > 0:
                # make uniform distribution in polar
                r = np.sqrt(np.random.random(1)) * fill_radius * (image_shape[2] - 1) * 0.5
                theta = np.random.random(1) * 2 * np.pi
                x = r * np.cos(theta) + (image_shape[2] - 1) * 0.5
                y = r * np.sin(theta) + (image_shape[1] - 1) * 0.5
                reference[
                    rng.integers(zborder, int(image_shape[0] - zborder)),
                    np.round(y).astype(np.int32),
                    np.round(x).astype(np.int32)
                ] = photons
            else:
                # bead at center
                reference[image_shape[0] // 2, image_shape[1] // 2, image_shape[2] // 2] = photons

    return reference


def simulate_image(
    filename: str,
    reference: np.ndarray,
    outdir: Path,
    phi: Wavefront,
    gen: SyntheticPSF,
    upsampled_gen: SyntheticPSF,
    npoints: int,
    photons: tuple,
    emb: bool = True,
    noise: bool = True,
    normalize: bool = True,
    remove_background: bool = True,
    random_crop: Any = None,
    embedding_option: Union[set, list, tuple] = { 'spatial_planes' },
    alpha_val: str = 'abs',
    phi_val: str = 'angle',
    lls_defocus_offset: Any = 0.,
    sigma_background_noise=40,
    mean_background_offset=100,
    electrons_per_count: float = .22,
    quantum_efficiency: float = .82,
    model_psf_shape: tuple = (64, 64, 64),
    scale_by_maxcounts: Optional[int] = None
):

    # aberrated PSF without noise
    kernel = upsampled_gen.single_psf(
        phi=phi,
        lls_defocus_offset=lls_defocus_offset,
        normed=True,
    )

    if gen.psf_type == 'widefield':  # normalize PSF by the total energy in the focal plane
        kernel /= np.max(kernel)
    else:
        kernel /= np.sum(kernel)

    img = fftconvolution(sample=reference, kernel=kernel)  # image in photons
    img = resize_with_crop_or_pad(img, crop_shape=model_psf_shape, mode='constant')  # only center crop

    if gen.psf_type == 'widefield':
        img /= np.max(img)
        img *= electrons2photons(counts2electrons(scale_by_maxcounts))

    p2v = phi.peak2valley(na=1.0)
    if npoints > 1:
        avg_min_distance = np.nan_to_num(mean_min_distance(reference, voxel_size=gen.voxel_size), nan=0)
    else:
        avg_min_distance = 0.

    if noise:  # convert to electrons to add shot noise and dark read noise, then convert to counts
        inputs = add_noise(
            img,
            mean_background_offset=mean_background_offset,
            sigma_background_noise=sigma_background_noise,
            quantum_efficiency=quantum_efficiency,
            electrons_per_count=electrons_per_count,
        )
    else:  # convert image to counts
        inputs = photons2electrons(img, quantum_efficiency=quantum_efficiency)
        inputs = electrons2counts(inputs, electrons_per_count=electrons_per_count)

    counts = np.sum(inputs)
    counts_mode = int(st.mode(inputs.astype(int), axis=None).mode[0])
    counts_percentiles = np.array([np.percentile(inputs, p) for p in range(1, 101)], dtype=int)

    if random_crop is not None:
        crop = int(np.random.uniform(low=random_crop, high=gen.psf_shape[0]+1))
        inputs = resize_with_crop_or_pad(inputs, crop_shape=[crop]*3)

    if emb:
        for e in set(embedding_option):
            odir = outdir/e
            odir.mkdir(exist_ok=True, parents=True)

            embeddings = prep_sample(
                inputs,
                sample_voxel_size=gen.voxel_size,
                model_fov=gen.psf_fov,
                remove_background=remove_background,
                normalize=normalize,
                min_psnr=0,
                # plot=odir/filename,
            )

            embeddings = np.squeeze(fourier_embeddings(
                inputs=embeddings,
                iotf=gen.iotf,
                na_mask=gen.na_mask(),
                embedding_option=e,
                alpha_val=alpha_val,
                phi_val=phi_val,
                # plot=odir/filename
            ))

            save_synthetic_sample(
                odir/filename,
                embeddings,
                amps=phi.amplitudes,
                photons=photons,
                counts=counts,
                counts_mode=counts_mode,
                counts_percentiles=counts_percentiles,
                npoints=npoints,
                p2v=p2v,
                gt=reference,
                gen=gen,
                realspace=inputs,
                avg_min_distance=avg_min_distance,
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
            outdir/filename,
            inputs,
            gt=reference,
            amps=phi.amplitudes,
            photons=photons,
            counts=counts,
            counts_mode=counts_mode,
            counts_percentiles=counts_percentiles,
            npoints=npoints,
            avg_min_distance=avg_min_distance,
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
    generators: dict,
    upsampled_generators: dict,
    npoints: int,
    savedir: Path,
    modes: int,
    distribution: str,
    mode_dist: str,
    gamma: float,
    signed: bool,
    randomize_voxel_size: bool,
    rotate: bool,
    min_amplitude: float,
    max_amplitude: float,
    min_photons: int,
    max_photons: int,
    random_crop: Any,
    noise: bool,
    normalize: bool,
    emb: bool = False,
    embedding_option: set = {'spatial_planes'},
    alpha_val: str = 'abs',
    phi_val: str = 'angle',
    min_lls_defocus_offset: float = 0.,
    max_lls_defocus_offset: float = 0.,
    fill_radius: float = 0.,
    default_wavelength: float = .510,
):

    aberration = Wavefront(
        amplitudes=(min_amplitude, max_amplitude),
        order='ansi',
        distribution=distribution,
        mode_weights=mode_dist,
        modes=modes,
        gamma=gamma,
        signed=signed,
        rotate=rotate,
        lam_detection=.510,
    )

    photon_range = (min_photons, max_photons)
    photons = randuniform(photon_range)
    lls_defocus_offset = randuniform((min_lls_defocus_offset, max_lls_defocus_offset))

    reference = beads(
        image_shape=(64, 64, 64),    # Change this to change image size (e.g. 256,256,256).
        photons=photons,
        object_size=0,
        num_objs=npoints,
        fill_radius=fill_radius,
    )

    inputs = {}

    for k, (gen, upsampled_gen) in enumerate(zip(generators.values(), upsampled_generators.values())):
        if gen.psf_type == '2photon':
            # boost um RMS aberration amplitudes for '2photon', so we create equivalent p2v aberrations
            r = gen.lam_detection / default_wavelength
            phi = Wavefront(
                amplitudes=[r * z for z in aberration.amplitudes],
                order=gen.order,
                distribution=gen.distribution,
                mode_weights=gen.mode_weights,
                modes=gen.n_modes,
                gamma=gen.gamma,
                signed=gen.signed,
                rotate=gen.rotate,
                lam_detection=gen.lam_detection,
            )
        else:
            phi = Wavefront(
                amplitudes=aberration.amplitudes,
                order=gen.order,
                distribution=gen.distribution,
                mode_weights=gen.mode_weights,
                modes=gen.n_modes,
                gamma=gen.gamma,
                signed=gen.signed,
                rotate=gen.rotate,
                lam_detection=gen.lam_detection,
            )

        outdir = savedir / rf"{gen.psf_type.replace('../lattice/', '').split('_')[0]}_lambda{round(gen.lam_detection * 1000)}"

        if not randomize_voxel_size:
            outdir = outdir / f"z{round(gen.z_voxel_size * 1000)}-y{round(gen.y_voxel_size * 1000)}-x{round(gen.x_voxel_size * 1000)}"

        outdir = outdir / f"z{gen.psf_shape[0]}-y{gen.psf_shape[0]}-x{gen.psf_shape[0]}"
        outdir = outdir / f"z{gen.n_modes}"

        if gen.distribution == 'powerlaw':
            outdir = outdir / f"powerlaw_gamma_{str(round(gen.gamma, 2)).replace('.', 'p')}"
        else:
            outdir = outdir / f"{gen.distribution}"

        outdir = outdir / f"photons_{photon_range[0]}-{photon_range[1]}"
        outdir = outdir / f"amp_{str(round(min_amplitude, 3)).replace('0.', 'p').replace('-', 'neg')}" \
                          f"-{str(round(max_amplitude, 3)).replace('0.', 'p').replace('-', 'neg')}"

        outdir = outdir / f"npoints_{npoints}"
        outdir.mkdir(exist_ok=True, parents=True)

        if emb:
            try:  # check if file already exists and not corrupted
                for e in embedding_option:
                    path = Path(f"{outdir/e}/{filename}")

                    with open(path.with_suffix('.json')) as f:
                        ujson.load(f)

                    with TiffFile(path.with_suffix('.tif')) as tif:
                        inputs[gen.psf_type] = np.squeeze(tif.asarray())

            except Exception as e:
                inputs[gen.psf_type] = simulate_image(
                    filename=filename,
                    reference=reference,
                    model_psf_shape=reference.shape,
                    outdir=outdir,
                    phi=phi,
                    gen=gen,
                    upsampled_gen=upsampled_gen,
                    npoints=npoints,
                    photons=photons,
                    emb=emb,
                    embedding_option=embedding_option,
                    random_crop=random_crop,
                    noise=noise,
                    normalize=normalize,
                    alpha_val=alpha_val,
                    phi_val=phi_val,
                    lls_defocus_offset=lls_defocus_offset,
                    scale_by_maxcounts=np.max(inputs['../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat'])
                    if gen.psf_type == 'widefield' else None
                )
        else:
            inputs[gen.psf_type] = simulate_image(
                filename=filename,
                reference=reference,
                model_psf_shape=reference.shape,
                outdir=outdir,
                phi=phi,
                gen=gen,
                upsampled_gen=upsampled_gen,
                npoints=npoints,
                photons=photons,
                emb=emb,
                embedding_option=embedding_option,
                random_crop=random_crop,
                noise=noise,
                normalize=normalize,
                alpha_val=alpha_val,
                phi_val=phi_val,
                lls_defocus_offset=lls_defocus_offset,
                scale_by_maxcounts=np.max(inputs['../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat'])
                if gen.psf_type == 'widefield' else None
            )

    return np.stack(list(inputs.values()), axis=0)


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
        "--psf_type", action='append', default=['../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat'],
        help='widefield, 2photon, confocal, or a path to an LLS excitation profile'
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
        "--fill_radius", default=0.0, type=float,
        help="Fractional cylinder radius (0-1) that defines where a bead may be placed in X Y Z."
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

    generators, upsampled_generators = {}, {}
    for psf in args.psf_type:
        generators[psf] = SyntheticPSF(
            amplitude_ranges=(args.min_amplitude, args.max_amplitude),
            psf_shape=3 * [args.input_shape],
            order='ansi',
            cpu_workers=args.cpu_workers,
            n_modes=args.modes,
            distribution=args.dist,
            mode_weights=args.mode_dist,
            gamma=args.gamma,
            signed=args.signed,
            rotate=args.rotate,
            psf_type=psf,
            lam_detection=.920 if psf == '2photon' else args.lam_detection,
            x_voxel_size=args.x_voxel_size,
            y_voxel_size=args.y_voxel_size,
            z_voxel_size=args.z_voxel_size,
            refractive_index=args.refractive_index,
            na_detection=args.na_detection,
        )

        # just for the widefield case
        upsampled_generators[psf] = SyntheticPSF(
            amplitude_ranges=generators[psf].amplitude_ranges,
            psf_shape=3*[2 * args.input_shape],
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
            cpu_workers=generators[psf].cpu_workers,
            gamma=generators[psf].gamma,
            refractive_index=generators[psf].refractive_index,
            na_detection=generators[psf].na_detection,
        )

    sample = partial(
        create_synthetic_sample,
        generators=generators,
        upsampled_generators=upsampled_generators,
        emb=args.emb,
        embedding_option=set(args.embedding_option),
        alpha_val=args.alpha_val,
        phi_val=args.phi_val,
        npoints=args.npoints,
        savedir=args.outdir,
        noise=args.noise,
        normalize=args.normalize,
        modes=args.modes,
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
        min_photons=args.min_photons,
        max_photons=args.max_photons,
        fill_radius=args.fill_radius,
    )
    logger.info(f"Output folder: {Path(args.outdir).resolve()}")
    jobs = [f"{int(args.filename)+k}" for k in range(args.iters)]
    multiprocess(func=sample, jobs=jobs, cores=args.cpu_workers)
    logging.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
