import logging
import sys
import time
from pathlib import Path


import tensorflow as tf

import cli
import backend


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def parse_args(args):
    parser = cli.argparser()

    parser.add_argument("model", type=Path, help="path of the model to evaluate")
    parser.add_argument("target", type=str, help="target of interest to evaluate")

    parser.add_argument(
        "--outdir", default="../models", type=Path, help='path to save eval'
    )

    parser.add_argument(
        "--amplitude_range", default=.2, type=float, help='amplitude range for zernike modes in microns'
    )

    parser.add_argument(
        "--max_jitter", default=1, type=float, help='randomly move the center point within a given limit (microns)'
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
        "--psnr", default=100, type=float, help='peak signal-to-noise ratio'
    )

    parser.add_argument(
        "--cpu_workers", default=-1, type=int, help='number of CPU cores to use'
    )

    parser.add_argument(
        "--gpu_workers", default=1, type=int, help='number of GPUs to use'
    )

    return parser.parse_args(args)


def main(args=None):
    timeit = time.time()
    args = parse_args(args)
    logging.info(args)

    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    if args.target == "random":
        backend.predict(
            model=args.model,
            wavelength=args.wavelength,
            x_voxel_size=args.x_voxel_size,
            y_voxel_size=args.y_voxel_size,
            z_voxel_size=args.z_voxel_size,
            max_jitter=args.max_jitter,
            cpu_workers=args.cpu_workers
        )

    elif args.target == "compare":
        backend.compare(
            model=args.model,
            wavelength=args.wavelength,
            x_voxel_size=args.x_voxel_size,
            y_voxel_size=args.y_voxel_size,
            z_voxel_size=args.z_voxel_size,
            max_jitter=args.max_jitter,
            cpu_workers=args.cpu_workers
        )

    elif args.target == "featuremaps":
        backend.featuremaps(
            modelpath=args.model,
            wavelength=args.wavelength,
            amplitude_range=args.amplitude_range,
            psnr=args.psnr,
            x_voxel_size=args.x_voxel_size,
            y_voxel_size=args.y_voxel_size,
            z_voxel_size=args.z_voxel_size,
            cpu_workers=args.cpu_workers
        )

    elif args.target == "kernels":
        backend.kernels(modelpath=args.model)

    elif args.target == "deconstruct":
        backend.deconstruct(
            model=args.model,
        )

    else:
        print("Error: unknown action!")

    print(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
