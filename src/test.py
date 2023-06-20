import atexit
import os
import re
import subprocess
import multiprocessing as mp

import numexpr
numexpr.set_num_threads(numexpr.detect_number_of_cores())

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
        "--num_neighbor", default=None, type=int, help='number of neighbors in the fov'
    )

    parser.add_argument(
        "--na", default=1.0, type=float, help='numerical aperture of detection objective'
    )

    parser.add_argument(
        "--gpu_workers", default=1, type=int, help='number of GPUs to use'
    )

    parser.add_argument(
        "--photons_min", default=5e5, type=float, help='min number of photons to use'
    )

    parser.add_argument(
        "--photons_max", default=6e5, type=float, help='max number of photons to use'
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
        "--cluster", action='store_true',
        help='a toggle to run predictions on our cluster'
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
        if args.target == 'modes':
            eval.evaluate_modes(
                args.model,
                eval_sign=args.eval_sign,
                num_objs=args.num_objs,
                batch_size=args.batch_size,
                digital_rotations=args.digital_rotations,
            )
        elif args.target == "random":
            eval.random_samples(
                model=args.model,
                eval_sign=args.eval_sign,
                batch_size=args.batch_size,
                digital_rotations=args.digital_rotations,
            )
        elif args.target == 'snrheatmap':
            eval.snrheatmap(
                iter_num=iter_num,
                modelpath=args.model,
                datadir=args.datadir,
                distribution=args.dist,
                samplelimit=args.n_samples,
                na=args.na,
                batch_size=args.batch_size,
                eval_sign=args.eval_sign,
                digital_rotations=args.digital_rotations,
                plot=args.plot,
                plot_rotations=args.plot_rotations,
            )
        elif args.target == 'densityheatmap':
            eval.densityheatmap(
                iter_num=iter_num,
                modelpath=args.model,
                datadir=args.datadir,
                distribution=args.dist,
                samplelimit=args.n_samples,
                na=args.na,
                batch_size=args.batch_size,
                eval_sign=args.eval_sign,
                digital_rotations=args.digital_rotations,
                plot=args.plot,
                plot_rotations=args.plot_rotations,
            )
        elif args.target == 'iterheatmap':
            savepath = eval.iterheatmap(
                iter_num=iter_num,
                modelpath=args.model,
                datadir=args.datadir,
                distribution=args.dist,
                samplelimit=args.n_samples,
                na=args.na,
                batch_size=args.batch_size,
                eval_sign=args.eval_sign,
                digital_rotations=args.digital_rotations,
                photons_range=(args.photons_min, args.photons_max),
                plot=args.plot,
                plot_rotations=args.plot_rotations,
            )
            with Path(f"{savepath.with_suffix('')}_eval_iterheatmap_settings.json").open('w') as f:
                json = dict(
                    iter_num=int(iter_num),
                    modelpath=str(args.model),
                    datadir=str(args.datadir),
                    distribution=str(args.dist),
                    samplelimit=int(args.n_samples),
                    na=float(args.na),
                    batch_size=int(args.batch_size),
                    eval_sign=bool(args.eval_sign),
                    digital_rotations=bool(args.digital_rotations),
                    photons_min=float(args.photons_min),
                    photons_max=float(args.photons_max),
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

        atexit.register(strategy._extended._collective_ops._pool.close)


def main(args=None):
    command_flags = sys.argv[1:] if args is None else args
    args = parse_args(args)
    logger.info(args)

    if args.cluster:
        hostname = "master.abc.berkeley.edu"
        username = "thayeralshaabi"
        partition = "abc_a100"

        cluster_env = f"~/anaconda3/envs/ml/bin/python"
        cluster_repo = f"/clusterfs/nvme/thayer/opticalaberrations"
        script = f"{cluster_repo}/src/test.py"

        flags = ' '.join(command_flags)
        flags = re.sub(pattern='--cluster', repl='', string=flags)
        flags = re.sub(pattern="\\\\", repl='/', string=flags)  # regex needs four backslashes to indicate one
        flags = flags.replace("..", cluster_repo)       # regex stinks at replacing ".."
        flags = re.sub(pattern='/home/supernova/nvme2/', repl='/clusterfs/nvme2/', string=flags)
        flags = re.sub(pattern='~/nvme2', repl='/clusterfs/nvme2/', string=flags)
        flags = re.sub(pattern='U:\\\\', repl='/clusterfs/nvme2/', string=flags)
        flags = re.sub(pattern='U:/', repl='/clusterfs/nvme2/', string=flags)
        flags = re.sub(pattern='V:\\\\', repl='/clusterfs/nvme/', string=flags)
        flags = re.sub(pattern='V:/', repl='/clusterfs/nvme/', string=flags)
        # flags = re.sub(pattern='--batch_size \d+', repl='--batch_size 300', string=flags)
        taskname = f"{args.target}_{Path(args.model).stem}"

        sjob = f"srun "
        sjob += f"--exclusive  "
        sjob += f"-p {partition} "
        sjob += f" --nodes=1 "
        sjob += f" --mem=500GB " #request basically all memory
        sjob += f"--job-name={taskname} "
        sjob += f"--pty {cluster_env} {script} {flags}"

        logger.info(f"ssh {username}@{hostname} \"{sjob}\"")
        subprocess.run(f"ssh {username}@{hostname} \"{sjob}\"", shell=True)
    else:
        if os.environ.get('SLURM_JOB_ID') is not None:
            logger.info(f"SLURM_JOB_ID = {os.environ.get('SLURM_JOB_ID')}")
        if os.environ.get('SLURMD_NODENAME') is not None:
            logger.info(f"SLURMD_NODENAME = {os.environ.get('SLURMD_NODENAME')}")
        if os.environ.get('SLURM_JOB_PARTITION') is not None:
            logger.info(f"SLURM_JOB_PARTITION = {os.environ.get('SLURM_JOB_PARTITION')}")

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
