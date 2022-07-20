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
logger = logging.getLogger(__name__)


def parse_args(args):
    parser = cli.argparser()

    parser.add_argument("model", type=Path, help="path of the model to evaluate")
    parser.add_argument("target", type=str, help="target of interest to evaluate")

    parser.add_argument(
        "--datadir", help='path to eval dataset', type=Path
    )

    parser.add_argument(
        "--reference", default=None, type=Path, help='path to a reference sample'
    )

    parser.add_argument(
        "--outdir", default="../models", type=Path, help='path to save eval'
    )

    parser.add_argument(
        "--n_samples", default=100, type=int, help='number of samples to evaluate'
    )

    parser.add_argument(
        "--batch_size", default=100, type=int, help='number of samples per batch'
    )

    parser.add_argument(
        "--dist", default='/', type=str, help='distribution to evaluate'
    )

    parser.add_argument(
        "--max_amplitude", default=.25, type=float, help="max amplitude for the zernike coefficients"
    )

    parser.add_argument(
        "--wavelength", default=.605, type=float, help='wavelength in microns'
    )

    parser.add_argument(
        "--x_voxel_size", default=.15, type=float, help='lateral voxel size in microns for X'
    )

    parser.add_argument(
        "--y_voxel_size", default=.15, type=float, help='lateral voxel size in microns for Y'
    )

    parser.add_argument(
        "--z_voxel_size", default=.6, type=float, help='axial voxel size in microns for Z'
    )

    parser.add_argument(
        "--cpu_workers", default=-1, type=int, help='number of CPU cores to use'
    )

    parser.add_argument(
        "--na", default=1.0, type=float, help='numerical aperture of detection objective'
    )

    parser.add_argument(
        "--gpu_workers", default=1, type=int, help='number of GPUs to use'
    )

    parser.add_argument(
        "--plot", action='store_true', help='only plot, do not recompute errors'
    )

    parser.add_argument(
        "--dmodes", default=None, type=int, help='minimum number of dominant modes in each sample'
    )
    return parser.parse_args(args)


def main(args=None):
    timeit = time.time()
    args = parse_args(args)

    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    if args.target == 'evalheatmap':
        eval.evalheatmap(
            modelpath=args.model,
            datadir=args.datadir,
            distribution=args.dist,
            samplelimit=args.n_samples,
            max_amplitude=args.max_amplitude,
            na=args.na,
        )
    elif args.target == 'iterheatmap':
        eval.iterheatmap(
            modelpath=args.model,
            datadir=args.datadir,
            reference=args.reference,
            distribution=args.dist,
            samplelimit=args.n_samples,
            max_amplitude=args.max_amplitude,
            na=args.na,
        )
    elif args.target == 'evalsample':
        eval.evalsample(
            modelpath=args.model,
            reference=args.reference,
            na=args.na,
        )
    else:
        eval.evaluate(
            model=args.model,
            target=args.target,
            wavelength=args.wavelength,
            dist=args.dist,
            x_voxel_size=args.x_voxel_size,
            y_voxel_size=args.y_voxel_size,
            z_voxel_size=args.z_voxel_size,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            cpu_workers=args.cpu_workers,
            plot=args.plot,
            dominant_modes=args.dmodes,
        )
    logging.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
