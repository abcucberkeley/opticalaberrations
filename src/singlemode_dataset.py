import matplotlib

matplotlib.use('Agg')

import re
import logging
import sys
import os
import time
import uuid
import ujson
from functools import partial
from typing import Any, Optional, Union
from pathlib import Path
from tifffile import TiffFile
from tifffile import imwrite
import numpy as np
from scipy import stats as st
from csbdeep.models import CARE

import matplotlib.pyplot as plt

plt.set_loglevel('error')

import cli
from utils import randuniform, multiprocess
from synthetic import SyntheticPSF
from wavefront import Wavefront
from multipoint_dataset import beads, simulate_image

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_synthetic_sample(
        filename: str,
        target_mode_ansi: int,
        generators: dict,
        upsampled_generators: dict,
        npoints: int,
        savedir: Path,
        modes: int,
        mode_dist: str,
        gamma: float,
        signed: bool,
        randomize_object_size: bool,
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
        object_size: Optional[float] = 0.,
        default_wavelength: float = .510,
        override: bool = False,
        plot: bool = False,
        denoising_dataset: bool = False,
        uniform_background: int = 0,
        skip_remove_background: bool = False,
        denoiser: Optional[CARE] = None,
        denoiser_window_size: tuple = (32, 64, 64),
):
    amplitudes = np.zeros(modes)
    amplitudes[target_mode_ansi] = randuniform((min_amplitude, max_amplitude))

    aberration = Wavefront(
        amplitudes=amplitudes,
        order='ansi',
        distribution='single',
        mode_weights=mode_dist,
        modes=modes,
        gamma=gamma,
        signed=signed,
        rotate=False,
        lam_detection=.510,
    )

    photon_range = (min_photons, max_photons)
    photons = randuniform(photon_range)
    lls_defocus_offset = randuniform((min_lls_defocus_offset, max_lls_defocus_offset))
    image_shape = next(iter(generators.values())).psf_shape

    reference = beads(
        image_shape=image_shape,
        photons=photons,
        object_size=-1 if randomize_object_size else object_size,
        num_objs=npoints,
        fill_radius=fill_radius,
        uniform_background=uniform_background,
    )

    inputs, wavefronts = {}, {}
    template = None

    for k, (gen, upsampled_gen) in enumerate(zip(generators.values(), upsampled_generators.values())):

        if template is None:
            template = wavefronts.get("../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat")

        if denoising_dataset:
            basedir = savedir / rf"{re.sub(r'.*/lattice/', '', str(gen.psf_type)).split('_')[0]}_lambda{round(gen.lam_detection * 1000)}"
            basedir = basedir / f"z{gen.psf_shape[0]}-y{gen.psf_shape[0]}-x{gen.psf_shape[0]}"
            basedir = basedir / f"z{gen.n_modes}"

            outdir = basedir / 'noisy'
            outdir = outdir / f"{aberration.distribution}"
            outdir = outdir / f"photons_{photon_range[0]}-{photon_range[1]}"
            outdir = outdir / f"amp_{str(round(min_amplitude, 3)).replace('0.', 'p').replace('-', 'neg')}" \
                              f"-{str(round(max_amplitude, 3)).replace('0.', 'p').replace('-', 'neg')}"

            gtdir = basedir / 'gt'
            gtdir = gtdir / f"{aberration.distribution}"
            gtdir = gtdir / f"photons_{photon_range[0]}-{photon_range[1]}"
            gtdir = gtdir / f"amp_{str(round(min_amplitude, 3)).replace('0.', 'p').replace('-', 'neg')}" \
                            f"-{str(round(max_amplitude, 3)).replace('0.', 'p').replace('-', 'neg')}"

            outdir.mkdir(exist_ok=True, parents=True)
            gtdir.mkdir(exist_ok=True, parents=True)

        else:
            gtdir = None
            outdir = savedir / rf"{re.sub(r'.*/lattice/', '', str(gen.psf_type)).split('_')[0]}_lambda{round(gen.lam_detection * 1000)}"

            if not randomize_object_size:
                outdir = outdir / f"z{round(gen.z_voxel_size * 1000)}-y{round(gen.y_voxel_size * 1000)}-x{round(gen.x_voxel_size * 1000)}"

            outdir = outdir / f"z{gen.psf_shape[0]}-y{gen.psf_shape[0]}-x{gen.psf_shape[0]}"
            outdir = outdir / f"z{gen.n_modes}"
            outdir = outdir / f"{aberration.distribution}"
            outdir = outdir / f"photons_{photon_range[0]}-{photon_range[1]}"
            outdir = outdir / f"amp_{str(round(min_amplitude, 3)).replace('0.', 'p').replace('-', 'neg')}" \
                              f"-{str(round(max_amplitude, 3)).replace('0.', 'p').replace('-', 'neg')}"

            outdir = outdir / f"npoints_{npoints}"
            outdir.mkdir(exist_ok=True, parents=True)

        if gen.psf_type == '2photon':
            # boost um RMS aberration amplitudes for '2photon', so we create equivalent p2v aberrations
            r = gen.lam_detection / default_wavelength
            phi = Wavefront(
                amplitudes=[r * z for z in aberration.amplitudes],
                order=aberration.order,
                distribution=aberration.distribution,
                mode_weights=aberration.mode_weights,
                modes=aberration.modes,
                gamma=aberration.gamma,
                signed=aberration.signed,
                rotate=aberration.rotate,
                lam_detection=gen.lam_detection,
            )
        else:
            phi = Wavefront(
                amplitudes=aberration.amplitudes if template is None else template.amplitudes,
                order=aberration.order,
                distribution=aberration.distribution,
                mode_weights=aberration.mode_weights,
                modes=aberration.modes,
                gamma=aberration.gamma,
                signed=aberration.signed,
                rotate=aberration.rotate,
                lam_detection=gen.lam_detection,
            )

        if emb:
            try:  # check if file already exists and not corrupted
                if override:
                    raise Exception(f'Override {filename}')

                for e in embedding_option:
                    path = Path(f"{outdir/e}/{filename}")

                    with TiffFile(path.with_suffix('.tif')) as tif:
                        inputs[gen.psf_type] = np.squeeze(tif.asarray())

                    with open(path.with_suffix('.json')) as f:
                        hashtbl = ujson.load(f)
                        amplitudes = np.array(hashtbl['zernikes']).astype(np.float32)

                    w = Wavefront(
                        amplitudes=amplitudes,
                        order=aberration.order,
                        distribution=aberration.distribution,
                        mode_weights=aberration.mode_weights,
                        modes=aberration.modes,
                        gamma=aberration.gamma,
                        signed=aberration.signed,
                        rotate=aberration.rotate,
                        lam_detection=gen.lam_detection,
                    )

                    if template is None:
                        wavefronts[gen.psf_type] = w
                    else:
                        if gen.psf_type == '2photon':
                            r = gen.lam_detection / default_wavelength
                            template_amplitudes = r * template.amplitudes
                        else:
                            template_amplitudes = template.amplitudes

                        if np.allclose(template_amplitudes, w.amplitudes):
                            wavefronts[gen.psf_type] = w
                        else:
                            # Override wavefront to generate sample again
                            wavefronts[gen.psf_type] = Wavefront(
                                amplitudes=template_amplitudes,
                                order=aberration.order,
                                distribution=aberration.distribution,
                                mode_weights=aberration.mode_weights,
                                modes=aberration.modes,
                                gamma=aberration.gamma,
                                signed=aberration.signed,
                                rotate=aberration.rotate,
                                lam_detection=gen.lam_detection,
                            )
                            raise Exception("Wavefront does not match template. Creating sample again.")

            except Exception as exc:
                wavefronts[gen.psf_type] = phi

                inputs[gen.psf_type] = simulate_image(
                    filename=filename,
                    reference=reference,
                    outdir=outdir,
                    gtdir=gtdir,
                    phi=wavefronts[gen.psf_type],
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
                    if gen.psf_type == 'widefield' else None,
                    plot=plot,
                    skip_remove_background=skip_remove_background,
                    denoiser=denoiser,
                    denoiser_window_size=denoiser_window_size
                )
        else:
            wavefronts[gen.psf_type] = phi

            inputs[gen.psf_type] = simulate_image(
                filename=filename,
                reference=reference,
                outdir=outdir,
                gtdir=gtdir,
                phi=wavefronts[gen.psf_type],
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
                if gen.psf_type == 'widefield' else None,
                plot=plot,
                skip_remove_background=skip_remove_background,
                denoiser=denoiser,
                denoiser_window_size=denoiser_window_size
            )

    return inputs


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
        '--randomize_object_size', action='store_true',
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
        "--object_size", default=0.0, type=float,
        help="optional bead size (Default: 0 for diffraction-limited beads, -1 for beads with random sizes)"
    )

    parser.add_argument(
        "--uniform_background", default=0, type=int,
        help="optional uniform background value"
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

    parser.add_argument(
        '--override', action='store_true',
        help='optional toggle to override existing data'
    )

    parser.add_argument(
        '--plot', action='store_true',
        help='optional toggle to plot preprocessing'
    )

    parser.add_argument(
        '--denoising_dataset', action='store_true',
        help='optional toggle to create a dataset for training a denoising model'
    )

    parser.add_argument(
        '--use_theoretical_widefield_simulator', action='store_true',
        help='optional toggle to use an experimental complex pupil to estimate amplitude attenuation (cosine factor)'
    )

    parser.add_argument(
        '--skip_remove_background', action='store_true',
        help='optional toggle to skip preprocessing input data using the DoG filter'
    )

    parser.add_argument(
        '--denoiser', type=Path, default=None,
        help='path to denoiser model'
    )

    return parser.parse_args(args)


def main(args=None):
    timeit = time.time()
    args = parse_args(args)
    logger.info(args)

    generators, upsampled_generators = {}, {}
    for psf in sorted(set(args.psf_type)):
        generators[psf] = SyntheticPSF(
            amplitude_ranges=(args.min_amplitude, args.max_amplitude),
            psf_shape=3 * [args.input_shape],
            order='ansi',
            cpu_workers=args.cpu_workers,
            n_modes=args.modes,
            distribution='single',
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
            use_theoretical_widefield_simulator=args.use_theoretical_widefield_simulator,
            skip_remove_background_ideal_psf=args.skip_remove_background
        )

        # just for the widefield case
        upsampled_generators[psf] = SyntheticPSF(
            amplitude_ranges=generators[psf].amplitude_ranges,
            psf_shape=3 * [2 * args.input_shape],
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
            use_theoretical_widefield_simulator=args.use_theoretical_widefield_simulator,
            skip_remove_background_ideal_psf=args.skip_remove_background
        )

    if args.denoiser is not None:
        logger.info(f"Loading denoiser model: {args.denoiser}")
        denoiser = CARE(config=None, name=args.denoiser.name, basedir=args.denoiser.parent)
    else:
        denoiser = None

    for i in range(3, args.modes):
        if i == 4:
            continue

        outdir = args.outdir / f"z{i}"

        sample = partial(
            create_synthetic_sample,
            target_mode_ansi=i,
            generators=generators,
            upsampled_generators=upsampled_generators,
            emb=args.emb,
            embedding_option=set(args.embedding_option),
            alpha_val=args.alpha_val,
            phi_val=args.phi_val,
            npoints=args.npoints,
            savedir=outdir,
            noise=args.noise,
            normalize=args.normalize,
            modes=args.modes,
            mode_dist=args.mode_dist,
            random_crop=args.random_crop,
            gamma=args.gamma,
            signed=args.signed,
            randomize_object_size=args.randomize_object_size,
            min_amplitude=args.min_amplitude,
            max_amplitude=args.max_amplitude,
            min_lls_defocus_offset=args.min_lls_defocus_offset,
            max_lls_defocus_offset=args.max_lls_defocus_offset,
            min_photons=args.min_photons,
            max_photons=args.max_photons,
            fill_radius=args.fill_radius,
            object_size=args.object_size,
            override=args.override,
            plot=args.plot,
            denoising_dataset=args.denoising_dataset,
            uniform_background=args.uniform_background,
            skip_remove_background=args.skip_remove_background,
            denoiser=denoiser
        )
        logger.info(f"Output folder: {Path(outdir).resolve()}")

        if args.denoising_dataset:
            jobs = [f"{uuid.uuid4()}" for k in range(args.iters)]
        else:
            jobs = [f"{int(args.filename) + k}" for k in range(args.iters)]

        multiprocess(func=sample, jobs=jobs, cores=args.cpu_workers)
        logging.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
