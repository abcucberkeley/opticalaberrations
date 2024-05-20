import logging
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

import tensorflow as tf

try:
    import cupy as cp
except ImportError as e:
    logging.warning(f"Cupy not supported on your system: {e}")

import cli
import experimental_benchmarks
from eval import compare_models, profile_models

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args(args):
    parser = cli.argparser()

    parser.add_argument("target", type=str, help="target of interest to evaluate")
    parser.add_argument(
        "inputs", help='path to eval dataset. Can be a folder or a .csv', type=Path
    )
    
    parser.add_argument(
        "--model_codename", default=None, action='append', help='path to _predictions.csv'
    )
    parser.add_argument(
        "--model_predictions", default=None, action='append', help='path to _predictions.csv'
    )

    parser.add_argument(
        "--outdir", default="../benchmarks", type=Path, help='path to save eval'
    )

    parser.add_argument(
        "--digital_rotations", action='store_true', help='use digital rotations to estimate prediction confidence'
    )

    parser.add_argument(
        "--eval_sign", default="signed", type=str, help='path to save eval'
    )

    parser.add_argument(
        "--batch_size", default=128, type=int, help='number of samples per batch'
    )

    parser.add_argument(
        "--dist", default='/', type=str, help='distribution to evaluate'
    )

    parser.add_argument(
        "--n_samples", default=10000, type=int, help='number of samples to evaluate'
    )

    parser.add_argument(
        "--niter", default=1, type=int, help='number of iterations'
    )

    parser.add_argument(
        "--na", default=1.0, type=float, help='numerical aperture of detection objective'
    )

    parser.add_argument(
        "--num_beads", default=None, type=int, help='number of beads in the fov'
    )
    
    parser.add_argument(
        "--plot", action='store_true', help='optional plot to show predictions for each sample'
    )
    
    parser.add_argument(
        "--simulate_psf_only", action='store_true', help='evaluate on PSFs only'
    )

    parser.add_argument(
        "--psf_type", default='widefield', type=str,
        help='widefield, 2photon, confocal, or a path to an LLS excitation profile '
    )

    parser.add_argument(
        "--lateral_voxel_size", default=.097, type=float, help='lateral voxel size in microns for X'
    )

    parser.add_argument(
        "--axial_voxel_size", default=.200, type=float, help='axial voxel size in microns for Z'
    )

    parser.add_argument(
        "--gpu_workers", default=1, type=int, help='number of GPUs to use'
    )

    parser.add_argument(
        "--cpu_workers", default=-1, type=int, help='number of CPU cores to use'
    )
    
    parser.add_argument(
        '--denoiser', type=Path, default=None,
        help='path to denoiser model'
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
        batch_size = 896 * gpu_workers
    elif gpu_workers > 0 and gpu_model.find('RTX') >= 0:
        batch_size = 896 * gpu_workers
    else:
        batch_size = args.batch_size

    logging.info(f'Number of active GPUs: {gpu_workers}, {gpu_model}, batch_size={batch_size}')

    with strategy.scope():
    
        if args.target == 'phasenet':
            experimental_benchmarks.predict_phasenet(
                inputs=args.inputs,
                plot=args.plot,
            )

        elif args.target == 'phaseretrieval':
            experimental_benchmarks.predict_phaseretrieval(
                inputs=args.inputs,
                plot=args.plot,
            )

        elif args.target == 'cocoa':
            experimental_benchmarks.predict_cocoa(
                inputs=args.inputs,
                axial_voxel_size=args.axial_voxel_size,
                lateral_voxel_size=args.lateral_voxel_size,
                na_detection=args.na,
                psf_type=args.psf_type,
                plot=args.plot,
            )

        elif args.target == 'phasenet_heatmap':
            experimental_benchmarks.phasenet_heatmap(
                inputs=args.inputs,
                outdir=args.outdir,
                distribution=args.dist,
                samplelimit=args.n_samples,
                na=args.na,
                batch_size=args.batch_size,
                eval_sign=args.eval_sign,
                num_beads=args.num_beads,
                iter_num=args.niter,
                denoiser=args.denoiser,
                simulate_psf_only=args.simulate_psf_only
            )

        elif args.target == 'phaseretrieval_heatmap':
            experimental_benchmarks.phaseretrieval_heatmap(
                inputs=args.inputs,
                outdir=args.outdir,
                distribution=args.dist,
                samplelimit=args.n_samples,
                na=args.na,
                batch_size=args.batch_size,
                eval_sign=args.eval_sign,
                num_beads=args.num_beads,
                iter_num=args.niter,
                denoiser=args.denoiser,
                simulate_psf_only=args.simulate_psf_only
            )

        elif args.target == 'profile_models':
            if args.model_codename is None:
                args.outdir = args.inputs
                args.model_codename, args.model_predictions = [], []
                for m in args.inputs.glob("*/"):
                    if m.is_dir():
                        args.model_codename.append(m.name.replace("-15-YuMB_lambda510", ""))
                        args.model_predictions.append(m)

            profile_models(
                models_codenames=args.model_codename,
                predictions_paths=args.model_predictions,
                outdir=args.outdir
            )
        else:
            compare_models(
                models_codenames=args.model_codename,
                predictions_paths=args.model_predictions,
                iter_num=args.niter,
                outdir=args.outdir
            )

    logging.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":

    main()
