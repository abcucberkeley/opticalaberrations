import numexpr
numexpr.set_num_threads(numexpr.detect_number_of_cores())

import logging
import sys
import time
from pathlib import Path

import tensorflow as tf
import cupy as cp

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
        "--datadir", help='path to eval dataset', type=Path
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
        "--n_samples", default=None, type=int, help='number of samples to evaluate'
    )

    parser.add_argument(
        "--batch_size", default=128, type=int, help='number of samples per batch'
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
        "--num_neighbor", default=None, type=int, help='number of neighbors in the fov'
    )

    parser.add_argument(
        "--na", default=1.0, type=float, help='numerical aperture of detection objective'
    )

    parser.add_argument(
        "--input_coverage", default=1.0, type=float, help='faction of the image to feed into the model '
                                                          '(then padded to keep the original image size)'
    )

    parser.add_argument(
        "--gpu_workers", default=1, type=int, help='number of GPUs to use'
    )

    parser.add_argument(
        "--plot", action='store_true', help='only plot, do not recompute errors'
    )

    parser.add_argument(
        "--peaks", default=None, help="matlab file that outlines peaks-coordinates"
    )

    return parser.parse_args(args)


def main(args=None):
    timeit = time.time()
    args = parse_args(args)
    logger.info(args)

    physical_devices = tf.config.list_physical_devices('GPU')

    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    if len(physical_devices) > 1:
        cp.fft.config.use_multi_gpus = True
        cp.fft.config.set_cufft_gpus(list(range(len(physical_devices))))

    if args.target == 'modes':
        eval.evaluate_modes(
            args.model,
            eval_sign=args.eval_sign,
        )

    elif args.target == "random":
        eval.random_samples(
            model=args.model,
            eval_sign=args.eval_sign,
            digital_rotations=args.digital_rotations,
        )
    elif args.target == 'snrheatmap':
        eval.snrheatmap(
            niter=args.niter,
            modelpath=args.model,
            datadir=args.datadir,
            distribution=args.dist,
            samplelimit=args.n_samples,
            input_coverage=args.input_coverage,
            na=args.na,
            batch_size=args.batch_size,
            eval_sign=args.eval_sign,
            digital_rotations=args.digital_rotations,
        )
    elif args.target == 'densityheatmap':
        eval.densityheatmap(
            niter=args.niter,
            modelpath=args.model,
            datadir=args.datadir,
            distribution=args.dist,
            samplelimit=args.n_samples,
            input_coverage=args.input_coverage,
            na=args.na,
            batch_size=args.batch_size,
            eval_sign=args.eval_sign,
            digital_rotations=args.digital_rotations,
        )
    elif args.target == 'iterheatmap':
        eval.iterheatmap(
            niter=args.niter,
            modelpath=args.model,
            datadir=args.datadir,
            distribution=args.dist,
            samplelimit=args.n_samples,
            input_coverage=args.input_coverage,
            na=args.na,
            batch_size=args.batch_size,
            eval_sign=args.eval_sign,
            digital_rotations=args.digital_rotations,
        )
    logging.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
