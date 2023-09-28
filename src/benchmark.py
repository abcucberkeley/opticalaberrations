import atexit
import os
import subprocess
import multiprocessing as mp

import logging
import sys
import time
from pathlib import Path
import tensorflow as tf
from functools import partial


try:
    import cupy as cp
except ImportError as e:
    logging.warning(f"Cupy not supported on your system: {e}")

import cli
import experimental_benchmarks

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
        "--n_samples", default=None, type=int, help='number of samples to evaluate'
    )

    parser.add_argument(
        "--na", default=1.0, type=float, help='numerical aperture of detection objective'
    )

    parser.add_argument(
        "--no_beads", action='store_true', help='evaluate on PSFs only'
    )

    parser.add_argument(
        "--plot", action='store_true', help='evaluate on PSFs only'
    )

    parser.add_argument(
        "--lateral_voxel_size", default=.097, type=float, help='lateral voxel size in microns for X'
    )
    parser.add_argument(
        "--axial_voxel_size", default=.200, type=float, help='axial voxel size in microns for Z'
    )

    return parser.parse_args(args)


def main(args=None):
    args = parse_args(args)
    logger.info(args)

    if os.name == 'nt':
        mp.set_executable(subprocess.run("where python", capture_output=True).stdout.decode('utf-8').split()[0])

    timeit = time.time()

    if args.target == 'phasenet':
        experimental_benchmarks.predict_phasenet(
            inputs=args.inputs,
            plot=args.plot,
        )
    elif args.target == 'cocoa':
        experimental_benchmarks.predict_cocoa(
            inputs=args.inputs,
            axial_voxel_size=args.axial_voxel_size,
            lateral_voxel_size=args.lateral_voxel_size,
            plot=args.plot,
        )
    elif args.target == 'phasenet_heatmap':
        experimental_benchmarks.phasenet_heatmap(
            inputs=args.inputs,
            distribution=args.dist,
            samplelimit=args.n_samples,
            na=args.na,
            batch_size=args.batch_size,
            eval_sign=args.eval_sign,
            no_beads=args.no_beads,
        )

    logging.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":

    main()
