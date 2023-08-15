import logging
import sys
import time
from pathlib import Path
import tensorflow as tf
import cli


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
        "--embedding_option", default='principle_planes', type=str, help="embedding option to use for evaluation"
    )

    parser.add_argument(
        "--amplitude_range", default=.2, type=float, help='amplitude range for zernike modes in microns'
    )

    parser.add_argument(
        "--wavelength", default=.510, type=float, help='wavelength in microns'
    )

    parser.add_argument(
        "--psf_type", default='widefield', help="type of the desired PSF"
    )

    parser.add_argument(
        "--x_voxel_size", default=.108, type=float, help='lateral voxel size in microns for X'
    )

    parser.add_argument(
        "--y_voxel_size", default=.108, type=float, help='lateral voxel size in microns for Y'
    )

    parser.add_argument(
        "--z_voxel_size", default=.2, type=float, help='axial voxel size in microns for Z'
    )

    parser.add_argument(
        "--n_modes", default=55, type=int, help='number zernike modes'
    )

    parser.add_argument(
        "--cpu_workers", default=-1, type=int, help='number of CPU cores to use'
    )

    parser.add_argument(
        "--gpu_workers", default=1, type=int, help='number of GPUs to use'
    )

    parser.add_argument(
        "--modelformat", default='trt', help="type of the desired model"
    )

    return parser.parse_args(args)


def main(args=None):
    timeit = time.time()
    args = parse_args(args)
    logging.info(args)

    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    if args.target == "metadata":

        import backend
        backend.save_metadata(
            filepath=args.model,
            n_modes=args.n_modes,
            wavelength=args.wavelength,
            psf_type=args.psf_type,
            x_voxel_size=args.x_voxel_size,
            y_voxel_size=args.y_voxel_size,
            z_voxel_size=args.z_voxel_size,
            embedding_option=args.embedding_option
        )
    elif args.target == "optimize":

        import convert
        convert.optimize_model(
            model_path=args.model,
            modelformat=args.modelformat
        )
    else:
        print("Error: unknown action!")

    print(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
