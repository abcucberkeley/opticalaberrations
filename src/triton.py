import logging
import sys
import time
import numpy as np
from pathlib import Path


import tensorflow as tf
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

import backend
import cli

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
tf.get_logger().setLevel(logging.ERROR)


def parse_args(args):
    parser = cli.argparser()
    parser.add_argument("model", type=Path, help="path to pretrained tensorflow model")
    parser.add_argument(
        "--current_dm", default=None, type=Path,
        help="optional path to current DM .csv file (Default: `blank mirror`)"
    )
    parser.add_argument(
        "--batch_size", default=128, type=int, help='maximum batch size for the model'
    )
    parser.add_argument(
        "--prev", default=None, type=Path,
        help="previous predictions .csv file (Default: `None`)"
    )
    parser.add_argument(
        "--lateral_voxel_size", default=.097, type=float, help='lateral voxel size in microns for X'
    )
    parser.add_argument(
        "--axial_voxel_size", default=.100, type=float, help='axial voxel size in microns for Z'
    )
    parser.add_argument(
        "--wavelength", default=.510, type=float,
        help='wavelength in microns'
    )
    parser.add_argument(
        "--freq_strength_threshold", default=.01, type=float,
        help='minimum frequency threshold in fourier space '
             '(percentages; values below that will be set to the desired minimum)'
    )
    parser.add_argument(
        "--confidence_threshold", default=0.015, type=float,
        help='optional threshold to flag unconfident predictions '
             'based on the standard deviations of the predicted amplitudes for all digital rotations (microns)'
    )
    parser.add_argument(
        "--sign_threshold", default=.9, type=float,
        help='flip sign of modes above given threshold relative to your initial prediction'
    )
    parser.add_argument(
        "--object_width", default=0.0, type=float,
        help='size of object for ideal psf. 0 (default) = single pixel. >0 gaussian width.'
    )
    parser.add_argument(
        "--plot", action='store_true',
        help='a toggle for plotting predictions'
    )
    parser.add_argument(
        "--plot_rotations", action='store_true',
        help='a toggle for plotting predictions for digital rotations'
    )
    parser.add_argument(
        "--ignore_mode", action='append', default=[0, 1, 2, 4],
        help='ANSI index for mode you wish to ignore'
    )
    parser.add_argument(
        "--cpu_workers", default=-1, type=int, help='number of CPU cores to use'
    )
    parser.add_argument(
        "--digital_rotations", default=361, type=int,
        help='optional flag for applying digital rotations'
    )

    return parser.parse_known_args(args)[0]



def get_verify_function():
    def verify_func(ys_runner, ys_expected):
        """Verify that at least 99% max probability tokens match on any given batch."""
        for y_runner, y_expected in zip(ys_runner, ys_expected):
            if not all(
                np.mean(a.argmax(axis=1) == b.argmax(axis=1)) > 0.99
                for a, b in zip(y_runner.values(), y_expected.values())
            ):
                return False
        return True

    return verify_func


def main(args=None):
    timeit = time.time()
    args = parse_args(args)
    logger.info(args) 

    if args.model.suffix == '.nav':
        import model_navigator as nav

        package = nav.package.load(path=args.model)
        pytriton_adapter = nav.pytriton.PyTritonAdapter(package=package, strategy=nav.MaxThroughputStrategy())
        runner = pytriton_adapter.runner
        runner.activate()

        @batch
        def infer_func(**inputs):
            return runner.infer(inputs)

        with Triton() as triton:
            triton.bind(
                model_name='trt-fp16',
                infer_func=infer_func,
                inputs=pytriton_adapter.inputs,
                outputs=pytriton_adapter.outputs,
                config=pytriton_adapter.config,
            )
            logger.info("Launching Triton inference server")
            triton.serve()
            
    elif args.model.ext == '.h5':
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        model = backend.load(args.model)
        
        @batch
        def infer_func(embeddings):
            batch = tf.convert_to_tensor(embeddings)
            zernikes = model.predict(batch)
            return [zernikes]
        
        with Triton() as triton:
            triton.bind(
                model_name=model.name,
                infer_func=infer_func,
                inputs=[Tensor(name="embeddings", dtype=np.float32, shape=model.input_shape[1:])],
                outputs=[Tensor(name="zernikes", dtype=np.float32, shape=model.output_shape[1:])],
                config=ModelConfig(max_batch_size=256),
            )
            logger.info("Launching Triton inference server")
            triton.serve()
    else:
        raise Exception(f'Unknown model format {args.model=}')

    logger.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")



if __name__ == "__main__":
    main()
