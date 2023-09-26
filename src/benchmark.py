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
        "--niter", default=1, type=int, help='number of iterations'
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

    return parser.parse_args(args)


def run_task(iter_num, args):
    tf.keras.backend.set_floatx('float32')
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    try:
        if len(physical_devices) > 1:
            cp.fft.config.use_multi_gpus = True
            cp.fft.config.set_cufft_gpus(list(range(len(physical_devices))))

    except ImportError as e:
        logging.warning(f"Cupy not supported on your system: {e}")

    strategy = tf.distribute.MirroredStrategy(
        devices=[f"{physical_devices[i].device_type}:{i}" for i in range(len(physical_devices))]
    )

    gpu_workers = strategy.num_replicas_in_sync
    logging.info(f'Number of active GPUs: {gpu_workers}')

    with strategy.scope():
        if args.target == 'phasenet':
            savepath = experimental_benchmarks.predict_phasenet(
                inputs=args.inputs,
                plot=args.plot,
            )
        elif args.target == 'cocoa':
            savepath = experimental_benchmarks.predict_cocoa(
                inputs=args.inputs,
                iter_num=iter_num,
                plot=args.plot,
            )
        elif args.target == 'phasenet_heatmap':
            savepath = experimental_benchmarks.phasenet_heatmap(
                iter_num=iter_num,
                inputs=args.inputs,
                distribution=args.dist,
                samplelimit=args.n_samples,
                na=args.na,
                batch_size=args.batch_size,
                eval_sign=args.eval_sign,
                no_beads=args.no_beads,
            )

        atexit.register(strategy._extended._collective_ops._pool.close)


def main(args=None):
    args = parse_args(args)
    logger.info(args)

    if os.name == 'nt':
        mp.set_executable(subprocess.run("where python", capture_output=True).stdout.decode('utf-8').split()[0])

    timeit = time.time()
    mp.set_start_method('spawn', force=True)

    for k in range(1, args.niter + 1):
        t = time.time()
        # Need to shut down the process after each iteration to clear its context and vram 'safely'
        p = mp.Process(target=partial(run_task, iter_num=k, args=args), name=args.target)
        p.start()
        p.join()
        p.close()

        logging.info(
            f'Iteration #{k} took {(time.time() - t) / 60:.1f} minutes to run. '
            f'{(time.time() - t) / 60 * (args.niter - k):.1f} minutes left to go.'
        )

        logging.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":

    main()
