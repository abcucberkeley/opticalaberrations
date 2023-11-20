import matplotlib
matplotlib.use('Agg')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import warnings
warnings.filterwarnings("ignore")

import logging
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import tensorflow as tf
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.experimental import AdamW
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, BackupAndRestore
from tensorflow_addons.optimizers import LAMB

from warmupcosinedecay import WarmupCosineDecay
from callbacks import LRLogger
from callbacks import Defibrillator
from callbacks import TensorBoardCallback

import utils
import data_utils
import opticalnet
import baseline
import otfnet
import cli
from roi import ROI

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
tf.get_logger().setLevel(logging.ERROR)
plt.set_loglevel('error')


def plot_patches(img: np.ndarray, outdir: Path, patch_size: list):
    for k, label in enumerate(['xy', 'xz', 'yz']):
        img = np.expand_dims(img[0], axis=0)
        original = np.squeeze(img[0, k])

        vmin = np.min(original)
        vmax = np.max(original)
        vcenter = (vmin + vmax) / 2
        step = .01

        highcmap = plt.get_cmap('YlOrRd', 256)
        lowcmap = plt.get_cmap('YlGnBu_r', 256)
        low = np.linspace(0, 1 - step, int(abs(vcenter - vmin) / step))
        high = np.linspace(0, 1 + step, int(abs(vcenter - vmax) / step))
        cmap = np.vstack((lowcmap(low), [1, 1, 1, 1], highcmap(high)))
        cmap = mcolors.ListedColormap(cmap)

        plt.figure(figsize=(4, 4))
        plt.imshow(original, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.axis("off")
        plt.title('Original')
        plt.savefig(f'{outdir}/{label}_original.png', dpi=300, bbox_inches='tight', pad_inches=.25)

        for p in patch_size:
            patches = opticalnet.Patchify(patch_size=p)(img)
            merged = opticalnet.Merge(patch_size=p)(patches)

            patches = patches[0, k]
            merged = np.squeeze(merged[0, k])

            plt.figure(figsize=(4, 4))
            plt.imshow(merged, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.axis("off")
            plt.title('Merged')
            plt.savefig(f'{outdir}/{label}_merged.png', dpi=300, bbox_inches='tight', pad_inches=.25)

            n = int(np.sqrt(patches.shape[0]))
            plt.figure(figsize=(4, 4))
            plt.title('Patches')
            for i, patch in enumerate(patches):
                ax = plt.subplot(n, n, i + 1)
                patch_img = tf.reshape(patch, (p, p)).numpy()
                ax.imshow(patch_img, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.axis("off")

            plt.savefig(f'{outdir}/{label}_patches_p{p}.png', dpi=300, bbox_inches='tight', pad_inches=.25)


def train_model(
    dataset: Path,
    outdir: Path,
    network: str = 'opticalnet',
    distribution: str = '/',
    embedding: str = 'spatial_planes',
    samplelimit: Optional[int] = None,
    max_amplitude: float = 1,
    input_shape: int = 64,
    batch_size: int = 32,
    patch_size: list = (32, 16, 8, 8),
    depth_scalar: float = 1.,
    width_scalar: float = 1.,
    activation: str = 'gelu',
    fixedlr: bool = False,
    opt: str = 'AdamW',
    lr: float = 5e-4,
    wd: float = 5e-6,
    dropout: float = .1,
    warmup: int = 2,
    epochs: int = 5,
    wavelength: float = .510,
    psf_type: str = '../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat',
    x_voxel_size: float = .097,
    y_voxel_size: float = .097,
    z_voxel_size: float = .2,
    modes: int = 15,
    pmodes: Optional[int] = None,
    min_photons: int = 1,
    max_photons: int = 1000000,
    roi: Any = None,
    refractive_index: float = 1.33,
    no_phase: bool = False,
    plot_patchfiy: bool = False,
    lls_defocus: bool = False,
    defocus_only: bool = False,
    radial_encoding_period: int = 16,
    radial_encoding_nth_order: int = 4,
    positional_encoding_scheme: str = 'rotational_symmetry',
    fixed_dropout_depth: bool = False,
    steps_per_epoch: Optional[int] = None,
    stem: bool = False,
    mul: bool = False,
    finetune: bool = False,
    strategy: Any = None
):
    outdir.mkdir(exist_ok=True, parents=True)
    network = network.lower()
    opt = opt.lower()
    restored = False

    if network == 'baseline':
        inputs = (input_shape, input_shape, input_shape, 1)
    else:
        inputs = (3 if no_phase else 6, input_shape, input_shape, 1)

    if dataset is None:
        config = dict(
            psf_type=psf_type,
            psf_shape=inputs,
            photons=(min_photons, max_photons),
            n_modes=modes,
            distribution=distribution,
            embedding_option=embedding,
            amplitude_ranges=(-max_amplitude, max_amplitude),
            lam_detection=wavelength,
            batch_size=batch_size,
            x_voxel_size=x_voxel_size,
            y_voxel_size=y_voxel_size,
            z_voxel_size=z_voxel_size,
            refractive_index=refractive_index,
            cpu_workers=-1
        )
        train_data = data_utils.create_dataset(config)
    else:
        train_data = data_utils.collect_dataset(
            dataset,
            metadata=False,
            modes=modes,
            distribution=distribution,
            embedding=embedding,
            samplelimit=samplelimit,
            max_amplitude=max_amplitude,
            no_phase=no_phase,
            lls_defocus=lls_defocus,
            photons_range=(min_photons, max_photons)
        )

        sample_writer = tf.summary.create_file_writer(f'{outdir}/train_samples/')
        with sample_writer.as_default():
            for s in range(10):
                fig = None
                for i, (img, y) in enumerate(train_data.take(5)):

                    if plot_patchfiy:
                        plot_patches(img=img, outdir=outdir, patch_size=patch_size)

                    img = np.squeeze(img, axis=-1)

                    if fig is None:
                        fig, axes = plt.subplots(5, img.shape[0], figsize=(8, 8))

                    for k in range(img.shape[0]):
                        if k > 2:
                            mphi = axes[i, k].imshow(img[k, :, :], cmap='coolwarm', vmin=-.5, vmax=.5)
                        else:
                            malpha = axes[i, k].imshow(img[k, :, :], cmap='Spectral_r', vmin=0, vmax=2)

                        axes[i, k].axis('off')

                    if img.shape[0] > 3:
                        cax = inset_axes(axes[i, 0], width="10%", height="100%", loc='center left', borderpad=-3)
                        cb = plt.colorbar(malpha, cax=cax)
                        cax.yaxis.set_label_position("left")

                        cax = inset_axes(axes[i, -1], width="10%", height="100%", loc='center right', borderpad=-2)
                        cb = plt.colorbar(mphi, cax=cax)
                        cax.yaxis.set_label_position("right")

                    else:
                        cax = inset_axes(axes[i, -1], width="10%", height="100%", loc='center right', borderpad=-2)
                        cb = plt.colorbar(malpha, cax=cax)
                        cax.yaxis.set_label_position("right")

                tf.summary.image("Training samples", utils.plot_to_image(fig), step=s)

        train_data = train_data.cache()
        train_data = train_data.shuffle(batch_size)
        train_data = train_data.batch(batch_size)
        train_data = train_data.prefetch(buffer_size=tf.data.AUTOTUNE)
        steps_per_epoch = tf.data.experimental.cardinality(train_data).numpy()

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        train_data = train_data.with_options(options)

    if fixedlr:
        scheduler = lr
        logger.info(f"Training steps: [{steps_per_epoch * epochs}]")
    else:
        warmup_steps = warmup * steps_per_epoch
        decay_steps = (epochs - warmup) * steps_per_epoch
        logger.info(f"Training steps [{steps_per_epoch * epochs}] = ({warmup_steps=}) + ({decay_steps=})")

        scheduler = WarmupCosineDecay(
            initial_learning_rate=0.,
            decay_steps=decay_steps,
            warmup_target=lr,
            warmup_steps=warmup_steps,
            alpha=.01,
        )

    if opt == 'lamb':
        optimizer = LAMB(learning_rate=scheduler, weight_decay=wd, beta_1=0.9, beta_2=0.99, clipnorm=1.0)
    elif opt == 'adamw':
        optimizer = AdamW(learning_rate=scheduler, weight_decay=wd, beta_1=0.9, beta_2=0.99, clipnorm=1.0)
    else:
        optimizer = Adam(learning_rate=scheduler, clipnorm=1.0)

    try:  # check if model already exists
        model_path = sorted(outdir.rglob('saved_model.pb'))[::-1][0].parent  # sort models to get the latest checkpoint

        custom_objects = {
            "ROI": ROI,
            "Stem": opticalnet.Stem,
            "Patchify": opticalnet.Patchify,
            "Merge": opticalnet.Merge,
            "PatchEncoder": opticalnet.PatchEncoder,
            "MLP": opticalnet.MLP,
            "StochasticDepth": opticalnet.StochasticDepth,
            "Transformer": opticalnet.Transformer,
            "WarmupCosineDecay": WarmupCosineDecay,
        }

        model = load_model(model_path, custom_objects=custom_objects)
        optimizer = model.optimizer

        if isinstance(model, tf.keras.Model):
            restored = True
            network = str(model_path)
            outdir = model_path
            training_history = pd.read_csv(model_path / 'logbook.csv', header=0, index_col=0)
            logger.info(f"Training history: {training_history}")
        else:
            outdir = outdir / f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
            outdir.mkdir(exist_ok=True, parents=True)

    except Exception as e:
        logger.warning(f"No model found in {outdir}; Creating a new model.")

    if not restored:  # Build a new model
        if defocus_only:  # only predict LLS defocus offset
            pmodes = 1
        elif lls_defocus:  # add LLS defocus offset to predictions
            pmodes = modes+1 if pmodes is None else pmodes+1
        else:
            pmodes = modes if pmodes is None else pmodes

        if network == 'opticalnet':
            model = opticalnet.OpticalTransformer(
                name='OpticalNet',
                roi=roi,
                stem=stem,
                patches=patch_size,
                modes=pmodes,
                depth_scalar=depth_scalar,
                width_scalar=width_scalar,
                dropout_rate=dropout,
                activation=activation,
                mul=mul,
                no_phase=no_phase,
                positional_encoding_scheme=positional_encoding_scheme,
                radial_encoding_period=radial_encoding_period,
                radial_encoding_nth_order=radial_encoding_nth_order,
                fixed_dropout_depth=fixed_dropout_depth,
            )

        elif network == 'baseline':
            model = baseline.Baseline(
                name='Baseline',
                modes=pmodes,
                depth_scalar=depth_scalar,
                width_scalar=width_scalar,
                activation=activation,
            )

        elif network == 'otfnet':
            model = otfnet.OTFNet(
                name='OTFNet',
                modes=pmodes
            )

        else:
            raise Exception(f'Network "{network}" is unknown.')

    if restored and not finetune:
        logger.info(f"Continue training {model.name} restored from {model_path}")
    else:
        if finetune:
            logger.info(f"Finetuning {model.name} using {optimizer.get_config()}")
        else:  # creating a new model
            model = model.build(input_shape=inputs)
            logger.info(model.summary())
            logger.info(optimizer.get_config())

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM),
            metrics=[tf.keras.metrics.RootMeanSquaredError(), 'mae', 'mse'],
        )

    tblogger = CSVLogger(
        f"{outdir}/logbook.csv",
        append=True,
    )

    pb_checkpoints = ModelCheckpoint(
        filepath=f"{outdir}",
        monitor="loss",
        save_best_only=True,
        verbose=1,
    )

    backup = BackupAndRestore(
        backup_dir=f"{outdir}",
        delete_checkpoint=False,
    )

    earlystopping = EarlyStopping(
        monitor='loss',
        min_delta=0,
        patience=50,
        verbose=1,
        mode='auto',
        restore_best_weights=True
    )

    defibrillator = Defibrillator(
        monitor='loss',
        patience=25,
        verbose=1,
    )

    tensorboard = TensorBoardCallback(
        log_dir=outdir,
        histogram_freq=1,
        profile_batch=100000000
    )

    lrlogger = LRLogger()

    model.fit(
        train_data,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=2,
        shuffle=True,
        callbacks=[
            tblogger,
            tensorboard,
            pb_checkpoints,
            earlystopping,
            defibrillator,
            backup,
            lrlogger
        ],
    )


def parse_args(args):
    train_parser = cli.argparser()

    train_parser.add_argument(
        "--network", default='opticalnet', type=str, help="codename for target network to train"
    )

    train_parser.add_argument(
        "--dataset", type=Path, help="path to dataset directory"
    )

    train_parser.add_argument(
        "--outdir", default="../models", type=Path, help='path to save trained models'
    )

    train_parser.add_argument(
        "--batch_size", default=1024, type=int, help="number of images per batch"
    )

    train_parser.add_argument(
        "--patch_size", default='32-16-8-8', help="patch size for transformer-based model"
    )

    train_parser.add_argument(
        "--roi", default=None, help="region of interest to crop from the center of the input image"
    )

    train_parser.add_argument(
        "--psf_type", default='../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat',
        help="type of the desired PSF"
    )

    train_parser.add_argument(
        "--x_voxel_size", default=.097, type=float, help='lateral voxel size in microns for X'
    )

    train_parser.add_argument(
        "--y_voxel_size", default=.097, type=float, help='lateral voxel size in microns for Y'
    )

    train_parser.add_argument(
        "--z_voxel_size", default=.2, type=float, help='axial voxel size in microns for Z'
    )

    train_parser.add_argument(
        "--input_shape", default=64, type=int, help="PSF input shape"
    )

    train_parser.add_argument(
        "--modes", default=15, type=int, help="number of modes to describe aberration"
    )

    train_parser.add_argument(
        "--pmodes", default=None, type=int, help="number of modes to predict"
    )

    train_parser.add_argument(
        "--min_photons", default=1, type=int, help="minimum photons for training samples"
    )

    train_parser.add_argument(
        "--max_photons", default=10000000, type=int, help="maximum photons for training samples"
    )

    train_parser.add_argument(
        "--dist", default='/', type=str, help="distribution of the zernike amplitudes"
    )

    train_parser.add_argument(
        "--embedding", default='spatial_planes', type=str, help="embedding option to use for training"
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
        "--dropout", default=0.1, type=float, help='initial dropout rate for stochastic depth'
    )

    train_parser.add_argument(
        "--opt", default='AdamW', type=str, help='optimizer to use for training'
    )

    train_parser.add_argument(
        "--activation", default='gelu', type=str, help='activation function for the model'
    )

    train_parser.add_argument(
        "--warmup", default=25, type=int, help='number of epochs for the initial linear warmup'
    )

    train_parser.add_argument(
        "--epochs", default=500, type=int, help="number of training epochs"
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

    train_parser.add_argument(
        '--lls_defocus', action='store_true',
        help='toggle to also predict the offset between the excitation and detection focal plan'
    )

    train_parser.add_argument(
        '--defocus_only', action='store_true',
        help='toggle to only predict the offset between the excitation and detection focal plan'
    )

    train_parser.add_argument(
        '--positional_encoding_scheme', default='rotational_symmetry', type=str,
        help='toggle to use different radial encoding types/schemes'
    )

    train_parser.add_argument(
        '--radial_encoding_period', default=16, type=int,
        help='toggle to add more periods for each sin/cos layer in the radial encodings'
    )

    train_parser.add_argument(
        '--radial_encoding_nth_order', default=4, type=int,
        help='toggle to define the max nth zernike order in the radial encodings'
    )

    train_parser.add_argument(
        '--stem', action='store_true',
        help='toggle to use a stem block'
    )

    train_parser.add_argument(
        '--fixed_dropout_depth', action='store_true',
        help='toggle to linearly increase dropout rate for deeper layers'
    )

    train_parser.add_argument(
        "--fixed_precision", action='store_true',
        help='optional toggle to disable automatic mixed precision training'
             '(https://www.tensorflow.org/guide/mixed_precision)'
    )

    return train_parser.parse_known_args(args)[0]


def main(args=None):
    timeit = time.time()
    args = parse_args(args)
    logger.info(args)

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
    logger.info(f'Number of active GPUs: {gpu_workers}')

    """
        To enable automatic mixed precision training for tensorflow:
        https://www.tensorflow.org/guide/mixed_precision
        https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#mptrain
        https://on-demand.gputechconf.com/gtc-taiwan/2018/pdf/5-1_Internal%20Speaker_Michael%20Carilli_PDF%20For%20Sharing.pdf
    """

    if not args.fixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

    with strategy.scope():
        train_model(
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
            dropout=args.dropout,
            fixedlr=args.fixedlr,
            warmup=args.warmup,
            epochs=args.epochs,
            pmodes=args.pmodes,
            min_photons=args.min_photons,
            max_photons=args.max_photons,
            max_amplitude=args.max_amplitude,
            distribution=args.dist,
            samplelimit=args.samplelimit,
            wavelength=args.wavelength,
            depth_scalar=args.depth_scalar,
            width_scalar=args.width_scalar,
            no_phase=args.no_phase,
            lls_defocus=args.lls_defocus,
            defocus_only=args.defocus_only,
            radial_encoding_period=args.radial_encoding_period,
            radial_encoding_nth_order=args.radial_encoding_nth_order,
            positional_encoding_scheme=args.positional_encoding_scheme,
            stem=args.stem,
            fixed_dropout_depth=args.fixed_dropout_depth,
            strategy=strategy
        )

    logger.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
