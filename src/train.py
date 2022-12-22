import numexpr
numexpr.set_num_threads(numexpr.detect_number_of_cores())

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
    train_parser = cli.argparser()

    train_parser.add_argument(
        "--network", default='opticaltransformer', type=str, help="codename for target network to train"
    )

    train_parser.add_argument(
        "--dataset", type=Path, help="path to dataset directory"
    )

    train_parser.add_argument(
        "--outdir", default="../models", type=Path, help='path to save trained models'
    )

    train_parser.add_argument(
        "--batch_size", default=512, type=int, help="number of images per batch"
    )

    train_parser.add_argument(
        "--patch_size", default='32-16-8-8', help="patch size for transformer-based model"
    )

    train_parser.add_argument(
        "--roi", default=None, help="region of interest to crop from the center of the input image"
    )

    train_parser.add_argument(
        "--psf_type", default='../lattice/HexRect_NAlattice0.25_NAAnnulusMax0.60_NAsigma0.08.mat', help="type of the desired PSF"
    )

    train_parser.add_argument(
        "--x_voxel_size", default=.108, type=float, help='lateral voxel size in microns for X'
    )

    train_parser.add_argument(
        "--y_voxel_size", default=.108, type=float, help='lateral voxel size in microns for Y'
    )

    train_parser.add_argument(
        "--z_voxel_size", default=.268, type=float, help='axial voxel size in microns for Z'
    )

    train_parser.add_argument(
        "--input_shape", default=64, type=int, help="PSF input shape"
    )

    train_parser.add_argument(
        "--modes", default=55, type=int, help="number of modes to describe aberration"
    )

    train_parser.add_argument(
        "--pmodes", default=None, type=int, help="number of modes to predict"
    )

    train_parser.add_argument(
        "--min_psnr", default=1, type=int, help="minimum PSNR for training samples"
    )

    train_parser.add_argument(
        "--max_psnr", default=100, type=int, help="maximum PSNR for training samples"
    )

    train_parser.add_argument(
        "--dist", default='/', type=str, help="distribution of the zernike amplitudes"
    )

    train_parser.add_argument(
        "--embedding", default='', type=str, help="embedding option to use for training"
    )

    train_parser.add_argument(
        "--samplelimit", default=None, type=int, help="max number of files to load from a dataset [per bin/class]"
    )

    train_parser.add_argument(
        "--max_amplitude", default=1., type=float, help="max amplitude for the zernike coefficients"
    )

    train_parser.add_argument(
        "--wavelength", default=.510, type=float, help='wavelength in microns'
    )

    train_parser.add_argument(
        "--depth_scalar", default=1., type=float, help='scale the number of blocks in the network'
    )

    train_parser.add_argument(
        "--width_scalar", default=1., type=float, help='scale the number of channels in each block'
    )

    train_parser.add_argument(
        '--fixedlr', action='store_true',
        help='toggle to use a fixed learning rate'
    )

    train_parser.add_argument(
        '--mul', action='store_true',
        help='toggle to multiply ratio (alpha) and phase (phi) in the STEM block'
    )

    train_parser.add_argument(
        "--lr", default=5e-4, type=float, help='initial learning rate'
    )

    train_parser.add_argument(
        "--wd", default=5e-6, type=float, help='initial weight decay'
    )

    train_parser.add_argument(
        "--opt", default='AdamW', type=str, help='optimizer to use for training'
    )

    train_parser.add_argument(
        "--activation", default='gelu', type=str, help='activation function for the model'
    )

    train_parser.add_argument(
        "--warmup", default=20, type=int, help='number of epochs for the initial linear warmup'
    )

    train_parser.add_argument(
        "--decay_period", default=300, type=int, help='number of epochs to decay over before restarting LR'
    )

    train_parser.add_argument(
        "--epochs", default=300, type=int, help="number of training epochs"
    )

    train_parser.add_argument(
        "--steps_per_epoch", default=100, type=int, help="number of steps per epoch"
    )

    train_parser.add_argument(
        "--cpu_workers", default=-1, type=int, help='number of CPU cores to use'
    )

    train_parser.add_argument(
        "--gpu_workers", default=-1, type=int, help='number of GPUs to use'
    )

    train_parser.add_argument(
        '--multinode', action='store_true',
        help='toggle for multi-node/multi-gpu training on a slurm-based cluster'
    )

    train_parser.add_argument(
        '--no_phase', action='store_true',
        help='toggle to use exclude phase from the model embeddings'
    )

    return train_parser.parse_known_args(args)[0]


def main(args=None):
    timeit = time.time()
    args = parse_args(args)
    logging.info(args)

    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    if args.multinode:
        strategy = tf.distribute.MultiWorkerMirroredStrategy(
            cluster_resolver=tf.distribute.cluster_resolver.SlurmClusterResolver(),
        )
    else:
        strategy = tf.distribute.MirroredStrategy()

    gpu_workers = strategy.num_replicas_in_sync
    logging.info(f'Number of active GPUs: {gpu_workers}')

    with strategy.scope():
        backend.train(
            epochs=args.epochs,
            dataset=args.dataset,
            embedding=args.embedding,
            outdir=args.outdir,
            network=args.network,
            input_shape=args.input_shape,
            batch_size=args.batch_size,
            patch_size=[int(i) for i in args.patch_size.split('-')],
            roi=[int(i) for i in args.roi.split('-')] if args.roi is not None else args.roi,
            steps_per_epoch=args.steps_per_epoch,
            psf_type=args.psf_type,
            x_voxel_size=args.x_voxel_size,
            y_voxel_size=args.y_voxel_size,
            z_voxel_size=args.z_voxel_size,
            modes=args.modes,
            activation=args.activation,
            mul=args.mul,
            opt=args.opt,
            lr=args.lr,
            wd=args.wd,
            fixedlr=args.fixedlr,
            warmup=args.warmup,
            decay_period=args.decay_period,
            pmodes=args.modes if args.pmodes is None else args.pmodes,
            min_psnr=args.min_psnr,
            max_psnr=args.max_psnr,
            max_amplitude=args.max_amplitude,
            distribution=args.dist,
            samplelimit=args.samplelimit,
            wavelength=args.wavelength,
            depth_scalar=args.depth_scalar,
            width_scalar=args.width_scalar,
            no_phase=args.no_phase,
        )

    logging.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
