import os
import subprocess
import multiprocessing as mp

import logging
import sys
import time
from pathlib import Path
import tensorflow as tf

import cli
import eval
import ujson

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
        "--outdir", default="../models", type=Path, help='path to save eval'
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
        "--batch_size", default=512, type=int, help='number of samples per batch'
    )

    parser.add_argument(
        "--dist", default='/', type=str, help='distribution to evaluate'
    )

    parser.add_argument(
        "--embedding", default='', type=str, help="embedding option to use for evaluation"
    )

    parser.add_argument(
        "--max_amplitude", default=.5, type=float, help="max amplitude for the zernike coefficients"
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
        "--photons_min", default=1e5, type=float, help='min number of photons to use'
    )

    parser.add_argument(
        "--photons_max", default=1.5e5, type=float, help='max number of photons to use'
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
    gpu_model = tf.config.experimental.get_device_details(physical_devices[0])['device_name']

    if gpu_workers > 0 and gpu_model.find('A100') >= 0:  # update batchsize automatically
        batch_size = 256 * gpu_workers
    elif gpu_workers > 0 and gpu_model.find('RTX') >= 0:
        batch_size = 256 * gpu_workers
    else:
        batch_size = args.batch_size

    logging.info(f'Number of active GPUs: {gpu_workers}, {gpu_model}, batch_size={batch_size}')

    with strategy.scope():
        if args.target == 'modes':
            savepath = eval.evaluate_modes(
                args.model,
                eval_sign=args.eval_sign,
                num_objs=args.num_objs,
                batch_size=batch_size,
                digital_rotations=args.digital_rotations,
            )
        elif args.target == 'sizes':
            savepath = eval.evaluate_object_sizes(
                args.model,
                eval_sign=args.eval_sign,
                batch_size=batch_size,
                digital_rotations=args.digital_rotations,
            )
        elif args.target == 'background':
            savepath = eval.evaluate_uniform_background(
                args.model,
                eval_sign=args.eval_sign,
                batch_size=batch_size,
                digital_rotations=args.digital_rotations,
            )
        elif args.target == "random":
            savepath = eval.random_samples(
                model=args.model,
                eval_sign=args.eval_sign,
                batch_size=batch_size,
                digital_rotations=args.digital_rotations,
            )
        elif args.target == "modalities":
            savepath = eval.eval_modalities(
                model=args.model,
                eval_sign=args.eval_sign,
                batch_size=batch_size,
                digital_rotations=args.digital_rotations,
            )
        elif args.target == 'snrheatmap':
            savepath = eval.snrheatmap(
                iter_num=args.niter,
                modelpath=args.model,
                datadir=args.datadir,
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
                num_beads=args.num_beads
            )
        elif args.target == 'eval_folder':
            savepath = eval.predict_folder(
                iter_num=args.niter,
                modelpath=args.model,
                datadir=args.datadir,
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
                num_beads=args.num_beads
            )
        elif args.target == "confidence":
            savepath = eval.eval_confidence(
                model=args.model,
                eval_sign=args.eval_sign,
                batch_size=batch_size,
                digital_rotations=args.digital_rotations,
            )
        elif args.target == 'confidence_heatmap':
            savepath = eval.confidence_heatmap(
                iter_num=args.niter,
                modelpath=args.model,
                datadir=args.datadir,
                distribution=args.dist,
                samplelimit=args.n_samples,
                na=args.na,
                batch_size=batch_size,
                eval_sign=args.eval_sign,
                digital_rotations=args.digital_rotations,
                plot=args.plot,
                plot_rotations=args.plot_rotations,
                psf_type=args.psf_type,
                lam_detection=args.wavelength
            )
        elif args.target == 'densityheatmap':
            savepath = eval.densityheatmap(
                iter_num=args.niter,
                modelpath=args.model,
                datadir=args.datadir,
                distribution=args.dist,
                samplelimit=args.n_samples,
                na=args.na,
                batch_size=batch_size,
                eval_sign=args.eval_sign,
                digital_rotations=args.digital_rotations,
                photons_range=(args.photons_min, args.photons_max),
                plot=args.plot,
                plot_rotations=args.plot_rotations,
                psf_type=args.psf_type,
                lam_detection=args.wavelength,
                num_beads=args.num_beads
            )
        elif args.target == 'iterheatmap':
            savepath = eval.iterheatmap(
                iter_num=args.niter,
                modelpath=args.model,
                datadir=args.datadir,
                distribution=args.dist,
                samplelimit=args.n_samples,
                na=args.na,
                batch_size=batch_size,
                eval_sign=args.eval_sign,
                digital_rotations=args.digital_rotations,
                photons_range=(args.photons_min, args.photons_max),
                plot=args.plot,
                plot_rotations=args.plot_rotations,
                psf_type=args.psf_type,
                lam_detection=args.wavelength
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
                    photons_min=float(args.photons_min),
                    photons_max=float(args.photons_max),
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
