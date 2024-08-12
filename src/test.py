import logging
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

import tensorflow as tf
import ujson

import cli
import eval

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args(args):
    parser = cli.argparser()

    parser.add_argument("model", type=Path, help="path of the model to evaluate")
    parser.add_argument("target", type=str, help="target of interest to evaluate")

    parser.add_argument(
        "--datadir", help='path to eval dataset. Can be a folder or a .csv', type=Path
    )

    parser.add_argument(
        "--outdir", default="../evaluations", type=Path, help='path to save eval'
    )

    parser.add_argument(
        "--niter", default=1, type=int, help='number of iterations'
    )

    parser.add_argument(
        "--digital_rotations", action='store_true', help='use digital rotations to estimate prediction confidence'
    )

    parser.add_argument(
        "--eval_sign", default="positive_only", type=str, help='path to save eval'
    )

    parser.add_argument(
        "--num_objs", default=None, type=int, help='number of beads to evaluate'
    )

    parser.add_argument(
        "--n_samples", default=None, type=int, help='number of samples to evaluate'
    )

    parser.add_argument(
        "--batch_size", default=-1, type=int, help='number of samples per batch'
    )

    parser.add_argument(
        "--dist", default='/', type=str, help='distribution to evaluate'
    )

    parser.add_argument(
        "--embedding", default='', type=str, help="embedding option to use for evaluation"
    )

    parser.add_argument(
        "--max_amplitude", default=1, type=float, help="max amplitude for the zernike coefficients"
    )

    parser.add_argument(
        "--cpu_workers", default=-1, type=int, help='number of CPU cores to use'
    )

    parser.add_argument(
        "--num_beads", default=None, type=int, help='number of beads in the fov'
    )

    parser.add_argument(
        "--na", default=1.0, type=float, help='numerical aperture of detection objective'
    )

    parser.add_argument(
        "--gpu_workers", default=1, type=int, help='number of GPUs to use'
    )

    parser.add_argument(
        "--min_photons", default=5e4, type=float, help='min number of photons to use'
    )

    parser.add_argument(
        "--max_photons", default=1e5, type=float, help='max number of photons to use'
    )

    parser.add_argument(
        "--plot", action='store_true', help='only plot, do not recompute errors, or a toggle for plotting predictions'
    )

    parser.add_argument(
        "--plot_rotations", action='store_true',
        help='a toggle for plotting predictions for digital rotations'
    )

    parser.add_argument(
        "--pois", default=None, help="matlab file that outlines peaks of interest coordinates"
    )

    parser.add_argument(
        "--psf_type", default=None, type=str,
        help='widefield, 2photon, confocal, or a path to an LLS excitation profile '
             '(Default: None; to keep default mode used during training)'
    )

    parser.add_argument(
        "--wavelength", default=None, type=float,
        help='detection wavelength in microns'
    )

    parser.add_argument(
        '--skip_remove_background', action='store_true',
        help='optional toggle to skip preprocessing input data using the DoG filter'
    )

    parser.add_argument(
        '--use_theoretical_widefield_simulator', action='store_true',
        help='optional toggle to calculate 3D PSF without amplitude attenuation (cosine factor)'
    )
    
    parser.add_argument(
        '--denoiser', type=Path, default=None,
        help='path to denoiser model'
    )
    
    parser.add_argument(
        '--simulate_samples', action='store_true',
        help='optional toggle to simulate PSFs to do iterative eval'
    )
    
    parser.add_argument(
        "--estimated_object_gaussian_sigma", default=0.0, type=float,
        help='size of object for creating an ideal psf (default: 0;  single pixel)'
    )
    
    parser.add_argument(
        "--simulate_psf_only", action='store_true', help='evaluate on PSFs only'
    )

    return parser.parse_args(args)


def main(args=None):
    args = parse_args(args)
    logger.info(args)

    if os.name == 'nt':
        mp.set_executable(subprocess.run("where python", capture_output=True).stdout.decode('utf-8').split()[0])

    timeit = time.time()

    tf.keras.backend.set_floatx('float32')
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    strategy = tf.distribute.MirroredStrategy(
        devices=[f"{physical_devices[i].device_type}:{i}" for i in range(len(physical_devices))]
    )

    gpu_workers = strategy.num_replicas_in_sync

    try:
        gpu_model = tf.config.experimental.get_device_details(physical_devices[0])['device_name']
    except IndexError:
        gpu_model = None

    if gpu_model is not None and args.batch_size == -1:
        if gpu_workers > 0 and gpu_model.find('A100') >= 0:  # update batchsize automatically
            batch_size = 896 * gpu_workers
        elif gpu_workers > 0 and gpu_model.find('RTX') >= 0:
            batch_size = 896 * gpu_workers
    else:
        batch_size = args.batch_size

    logging.info(f'Number of active GPUs: {gpu_workers}, {gpu_model}, batch_size={batch_size}')

    with strategy.scope():
        if args.target == 'modes':
            savepath = eval.evaluate_modes(
                args.model,
                outdir=args.outdir,
                eval_sign=args.eval_sign,
                num_objs=args.num_objs,
                batch_size=batch_size,
                digital_rotations=args.digital_rotations,
                denoiser=args.denoiser,
                estimated_object_gaussian_sigma=args.estimated_object_gaussian_sigma
            )
        elif args.target == 'templates':
            savepath = eval.plot_templates(args.model)
        elif args.target == 'sizes':
            savepath = eval.evaluate_object_sizes(
                args.model,
                outdir=args.outdir,
                eval_sign=args.eval_sign,
                batch_size=batch_size,
                digital_rotations=args.digital_rotations,
                denoiser=args.denoiser,
                estimated_object_gaussian_sigma=args.estimated_object_gaussian_sigma
            )
        elif args.target == 'background':
            savepath = eval.evaluate_uniform_background(
                args.model,
                eval_sign=args.eval_sign,
                batch_size=batch_size,
                digital_rotations=args.digital_rotations,
                estimated_object_gaussian_sigma=args.estimated_object_gaussian_sigma
            )
        elif args.target == "random":
            savepath = eval.random_samples(
                model=args.model,
                eval_sign=args.eval_sign,
                batch_size=batch_size,
                digital_rotations=args.digital_rotations,
                estimated_object_gaussian_sigma=args.estimated_object_gaussian_sigma
            )
        elif args.target == "modalities":
            savepath = eval.eval_modalities(
                model=args.model,
                eval_sign=args.eval_sign,
                batch_size=batch_size,
                digital_rotations=args.digital_rotations,
                estimated_object_gaussian_sigma=args.estimated_object_gaussian_sigma
            )
        elif args.target == "confidence":
            savepath = eval.eval_confidence(
                model=args.model,
                eval_sign=args.eval_sign,
                batch_size=batch_size,
                digital_rotations=args.digital_rotations,
                estimated_object_gaussian_sigma=args.estimated_object_gaussian_sigma
            )
        elif args.target == 'snrheatmap':
            savepath = eval.snrheatmap(
                iter_num=args.niter,
                modelpath=args.model,
                datadir=args.datadir,
                outdir=args.outdir,
                distribution=args.dist,
                samplelimit=args.n_samples,
                na=args.na,
                batch_size=batch_size,
                eval_sign=args.eval_sign,
                digital_rotations=args.digital_rotations,
                plot=args.plot,
                plot_rotations=args.plot_rotations,
                psf_type=args.psf_type,
                lam_detection=args.wavelength,
                num_beads=args.num_beads,
                skip_remove_background=args.skip_remove_background,
                use_theoretical_widefield_simulator=args.use_theoretical_widefield_simulator,
                denoiser=args.denoiser,
                simulate_samples=True if args.niter > 1 else args.simulate_samples,
                estimated_object_gaussian_sigma=args.estimated_object_gaussian_sigma,
                simulate_psf_only=args.simulate_psf_only
            )
        elif args.target == 'fscheatmap':
            savepath = eval.fscheatmap(
                iter_num=args.niter,
                modelpath=args.model,
                datadir=args.datadir,
                outdir=args.outdir,
                distribution=args.dist,
                samplelimit=args.n_samples,
                na=args.na,
                batch_size=batch_size,
                eval_sign=args.eval_sign,
                digital_rotations=args.digital_rotations,
                plot=args.plot,
                plot_rotations=args.plot_rotations,
                psf_type=args.psf_type,
                lam_detection=args.wavelength,
                num_beads=args.num_beads,
                skip_remove_background=args.skip_remove_background,
                use_theoretical_widefield_simulator=args.use_theoretical_widefield_simulator,
                denoiser=args.denoiser,
                simulate_samples=True if args.niter > 1 else args.simulate_samples,
                estimated_object_gaussian_sigma=args.estimated_object_gaussian_sigma,
                simulate_psf_only=args.simulate_psf_only
            )
        elif args.target == 'confidence_heatmap':
            savepath = eval.confidence_heatmap(
                iter_num=args.niter,
                modelpath=args.model,
                datadir=args.datadir,
                outdir=args.outdir,
                distribution=args.dist,
                samplelimit=args.n_samples,
                na=args.na,
                batch_size=batch_size,
                eval_sign=args.eval_sign,
                digital_rotations=args.digital_rotations,
                plot=args.plot,
                plot_rotations=args.plot_rotations,
                psf_type=args.psf_type,
                lam_detection=args.wavelength,
                skip_remove_background=args.skip_remove_background,
                use_theoretical_widefield_simulator=args.use_theoretical_widefield_simulator,
                denoiser=args.denoiser,
                simulate_samples=True if args.niter > 1 else args.simulate_samples,
                estimated_object_gaussian_sigma=args.estimated_object_gaussian_sigma
            )
        elif args.target == 'densityheatmap':
            savepath = eval.densityheatmap(
                iter_num=args.niter,
                modelpath=args.model,
                datadir=args.datadir,
                outdir=args.outdir,
                distribution=args.dist,
                samplelimit=args.n_samples,
                na=args.na,
                batch_size=batch_size,
                eval_sign=args.eval_sign,
                digital_rotations=args.digital_rotations,
                photons_range=(args.min_photons, args.max_photons),
                plot=args.plot,
                plot_rotations=args.plot_rotations,
                psf_type=args.psf_type,
                lam_detection=args.wavelength,
                num_beads=args.num_beads,
                skip_remove_background=args.skip_remove_background,
                use_theoretical_widefield_simulator=args.use_theoretical_widefield_simulator,
                denoiser=args.denoiser,
                simulate_samples=True if args.niter > 1 else args.simulate_samples,
                estimated_object_gaussian_sigma=args.estimated_object_gaussian_sigma
            )
        elif args.target == 'objectsizeheatmap':
            savepath = eval.objectsizeheatmap(
                iter_num=args.niter,
                modelpath=args.model,
                datadir=args.datadir,
                outdir=args.outdir,
                distribution=args.dist,
                samplelimit=args.n_samples,
                na=args.na,
                batch_size=batch_size,
                eval_sign=args.eval_sign,
                digital_rotations=args.digital_rotations,
                photons_range=(args.min_photons, args.max_photons),
                plot=args.plot,
                plot_rotations=args.plot_rotations,
                psf_type=args.psf_type,
                lam_detection=args.wavelength,
                num_beads=args.num_beads,
                skip_remove_background=args.skip_remove_background,
                use_theoretical_widefield_simulator=args.use_theoretical_widefield_simulator,
                denoiser=args.denoiser,
                simulate_samples=True if args.niter > 1 else args.simulate_samples,
                estimated_object_gaussian_sigma=args.estimated_object_gaussian_sigma
            )
        elif args.target == 'iterheatmap':
            savepath = eval.iterheatmap(
                iter_num=args.niter,
                modelpath=args.model,
                datadir=args.datadir,
                outdir=args.outdir,
                distribution=args.dist,
                samplelimit=args.n_samples,
                na=args.na,
                batch_size=batch_size,
                eval_sign=args.eval_sign,
                digital_rotations=args.digital_rotations,
                photons_range=(args.min_photons, args.max_photons),
                plot=args.plot,
                plot_rotations=args.plot_rotations,
                psf_type=args.psf_type,
                lam_detection=args.wavelength,
                skip_remove_background=args.skip_remove_background,
                use_theoretical_widefield_simulator=args.use_theoretical_widefield_simulator,
                denoiser=args.denoiser,
                simulate_samples=True if args.niter > 1 else args.simulate_samples,
                estimated_object_gaussian_sigma=args.estimated_object_gaussian_sigma
            )

        if savepath is not None:
            with Path(f"{savepath.with_suffix('')}_eval_settings.json").open('w') as f:
                json = dict(
                    iter_num=int(args.niter),
                    modelpath=str(args.model),
                    datadir=str(args.datadir),
                    distribution=str(args.dist),
                    samplelimit=int(args.n_samples) if args.n_samples is not None else None,
                    na=float(args.na),
                    batch_size=int(batch_size),
                    eval_sign=bool(args.eval_sign),
                    digital_rotations=bool(args.digital_rotations),
                    min_photons=float(args.min_photons),
                    max_photons=float(args.max_photons),
                    psf_type=args.psf_type,
                    lam_detection=args.wavelength,
                )

                ujson.dump(
                    json,
                    f,
                    indent=4,
                    sort_keys=False,
                    ensure_ascii=False,
                    escape_forward_slashes=False
                )
                logging.info(f"Saved: {f.name}")

    logging.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
