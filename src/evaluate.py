import logging
import sys
import time
from pathlib import Path
import tensorflow as tf

import cli
import eval


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def parse_args(args):
    parser = cli.argparser()

    parser.add_argument("model", type=Path, help="path of the model to evaluate")

    parser.add_argument(
        "--outdir", default="../models", type=Path, help='path to save eval'
    )

    parser.add_argument(
        "--samples", default=100, type=int, help='number of samples to test on'
    )

    parser.add_argument(
        "--batch_size", default=100, type=int, help='number of samples per batch'
    )

    parser.add_argument(
        "--psf_shape", default=64, type=int, help='input shape to the model'
    )

    parser.add_argument(
        "--wavelength", default=.605, type=float, help='wavelength in microns'
    )

    parser.add_argument(
        "--x_voxel_size", default=.108, type=float, help='lateral voxel size in microns for X'
    )

    parser.add_argument(
        "--y_voxel_size", default=.108, type=float, help='lateral voxel size in microns for Y'
    )

    parser.add_argument(
        "--z_voxel_size", default=.1, type=float, help='axial voxel size in microns for Z'
    )

    parser.add_argument(
        "--cpu_workers", default=-1, type=int, help='number of CPU cores to use'
    )

    parser.add_argument(
        "--gpu_workers", default=1, type=int, help='number of GPUs to use'
    )

    parser.add_argument(
        "--plot", action='store_true', help='plot a wavefornt for each prediction'
    )

    subparsers = parser.add_subparsers(
        help="Arguments for specific action.", dest="dtype"
    )
    subparsers.required = True

    subparsers.add_parser("modes", help='test model on all aberration modes')
    subparsers.add_parser("psnr", help='test model on a range of PSNRs')
    subparsers.add_parser("jitter", help='test model on a range of jitters')

    compare = subparsers.add_parser("compare", help='compare a list of models in a dir')
    compare.add_argument('--psf_shape', type=int, default=64)

    compare_modes = subparsers.add_parser("compare_modes", help='compare a list of models in a dir')
    compare_modes.add_argument('--psf_shape', type=int, default=64)

    convergence = subparsers.add_parser("convergence", help='test the number of iters needed for convergence')
    convergence.add_argument('--datadir', type=str)
    convergence.add_argument('--psf_shape', type=int, default=64)

    return parser.parse_args(args)


def main(args=None):
    timeit = time.time()
    args = parse_args(args)

    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    if args.dtype == 'modes':
        eval.evaluate_modes(
            model=args.model,
            wavelength=args.wavelength,
            x_voxel_size=args.x_voxel_size,
            y_voxel_size=args.y_voxel_size,
            z_voxel_size=args.z_voxel_size,
            psf_shape=args.psf_shape
        )

    elif args.dtype == 'psnr':
        eval.evaluate_psnrs(
            model=args.model,
            wavelength=args.wavelength,
            x_voxel_size=args.x_voxel_size,
            y_voxel_size=args.y_voxel_size,
            z_voxel_size=args.z_voxel_size,
            cpu_workers=args.cpu_workers,
            n_samples=args.samples,
            batch_size=args.batch_size,
            plot=args.plot
        )

    elif args.dtype == 'compare':
        eval.compare_models(
            modelsdir=args.model,
            wavelength=args.wavelength,
            x_voxel_size=args.x_voxel_size,
            y_voxel_size=args.y_voxel_size,
            z_voxel_size=args.z_voxel_size,
            cpu_workers=args.cpu_workers,
            n_samples=args.samples,
            batch_size=args.batch_size,
            psf_shape=tuple(3*[args.psf_shape])
        )

    elif args.dtype == 'compare_modes':
        eval.compare_models_and_modes(
            modelsdir=args.model,
            wavelength=args.wavelength,
            x_voxel_size=args.x_voxel_size,
            y_voxel_size=args.y_voxel_size,
            z_voxel_size=args.z_voxel_size,
            cpu_workers=args.cpu_workers,
            n_samples=args.samples,
            batch_size=args.batch_size,
            psf_shape=tuple(3*[args.psf_shape])
        )

    elif args.dtype == 'convergence':
        eval.convergence(
            modelsdir=args.model,
            datadir=args.datadir,
            wavelength=args.wavelength,
            x_voxel_size=args.x_voxel_size,
            y_voxel_size=args.y_voxel_size,
            z_voxel_size=args.z_voxel_size,
            cpu_workers=args.cpu_workers,
            n_samples=args.samples,
            batch_size=args.batch_size,
            psf_shape=tuple(3*[args.psf_shape])
        )

    else:
        logging.error("Error: unknown action!")

    logging.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
