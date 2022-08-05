import logging
import sys
import time
from pathlib import Path
import tensorflow as tf

import cli
import experimental
import shapes
import vis

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args(args):
    parser = cli.argparser()

    subparsers = parser.add_subparsers(
        help="Arguments for specific action.", dest="dtype"
    )
    subparsers.required = True

    subparsers.add_parser("inputs")
    subparsers.add_parser("fov")
    subparsers.add_parser("dist")
    subparsers.add_parser("signal")
    subparsers.add_parser("aberration")
    subparsers.add_parser("psnr")
    subparsers.add_parser("relratio")
    subparsers.add_parser("gaussian")
    subparsers.add_parser("simulation")
    subparsers.add_parser("shapes")
    subparsers.add_parser("similarity")

    data_parser = subparsers.add_parser("parse")
    data_parser.add_argument("dataset", type=Path, help="path to raw PSF directory")

    syn_parser = subparsers.add_parser("synthetic")
    syn_parser.add_argument("img", type=Path, help="path to a tif image")

    return parser.parse_args(args)


def main(args=None):
    timeit = time.time()
    args = parse_args(args)

    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    if args.dtype == "shapes":
        shapes.simobjects()

    elif args.dtype == "inputs":
        vis.plot_inputs()

    elif args.dtype == "dist":
        vis.plot_training_dist()

    elif args.dtype == "signal":
        vis.plot_signal()

    elif args.dtype == "fov":
        vis.plot_fov()

    elif args.dtype == "relratio":
        vis.plot_relratio()

    elif args.dtype == "gaussian":
        vis.plot_gaussian_filters()

    elif args.dtype == "simulation":
        vis.plot_simulation()

    elif args.dtype == "aberration":
        vis.plot_aberrations()

    elif args.dtype == "psnr":
        vis.plot_psnr()

    elif args.dtype == "similarity":
        shapes.similarity()

    elif args.dtype == "parse":
        experimental.create_dataset(data_path=args.dataset)

    else:
        logger.error("Error: unknown action!")

    logger.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
