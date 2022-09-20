import numexpr
numexpr.set_num_threads(numexpr.detect_number_of_cores())

import matplotlib
matplotlib.use('TkAgg')

import logging
import sys
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from tqdm import trange, tqdm
from tifffile import imsave

import tensorflow as tf
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow_addons.optimizers import AdamW
from tensorflow_addons.optimizers import SGDW

from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.callbacks import EarlyStopping
from callbacks import Defibrillator
from callbacks import LearningRateScheduler
from callbacks import TensorBoardCallback

import utils
import vis
import data_utils
import preprocessing
import experimental

from synthetic import SyntheticPSF
from wavefront import Wavefront

from tensorflow.keras import Model
from phasenet import PhaseNet
from preprocessing import resize_with_crop_or_pad

from stem import Stem
from activation import MaskedActivation
from depthwiseconv import DepthwiseConv3D
from spatial import SpatialAttention
from roi import ROI
import opticalresnet
import opticaltransformer
from baseline import Baseline

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
tf.get_logger().setLevel(logging.ERROR)


def load(model_path: Path, mosaic=False) -> Model:
    model_path = Path(model_path)

    if mosaic:
        custom_objects = {
            "ROI": ROI,
            "Stem": opticaltransformer.Stem,
            "Patchify": opticaltransformer.Patchify,
            "Merge": opticaltransformer.Merge,
            "PatchEncoder": opticaltransformer.PatchEncoder,
            "MLP": opticaltransformer.MLP,
            "Transformer": opticaltransformer.Transformer,
        }
        if model_path.is_file() and model_path.suffix == '.h5':
            return load_model(str(model_path), custom_objects=custom_objects)
        else:
            return load_model(str(list(model_path.rglob('*.h5'))[0]), custom_objects=custom_objects)
    else:
        try:
            try:
                '''.pb format'''
                if model_path.is_file() and model_path.suffix == '.pb':
                    return load_model(str(model_path.parent))
                else:
                    return load_model(str(list(model_path.rglob('saved_model.pb'))[0].parent))

            except IndexError or FileNotFoundError or OSError:
                if 'opticaltransformer' in str(model_path):
                    custom_objects = {
                        "ROI": ROI,
                        "Stem": opticaltransformer.Stem,
                        "Patchify": opticaltransformer.Patchify,
                        "Merge": opticaltransformer.Merge,
                        "PatchEncoder": opticaltransformer.PatchEncoder,
                        "MLP": opticaltransformer.MLP,
                        "Transformer": opticaltransformer.Transformer,
                    }
                else:
                    custom_objects = {
                        "Stem": Stem,
                        "MaskedActivation": MaskedActivation,
                        "SpatialAttention": SpatialAttention,
                        "DepthwiseConv3D": DepthwiseConv3D,
                        "CAB": opticalresnet.CAB,
                        "TB": opticalresnet.TB,
                    }

                '''.h5/hdf5 format'''
                if model_path.is_file() and model_path.suffix == '.h5':
                    return load_model(str(model_path), custom_objects=custom_objects)
                else:
                    return load_model(str(list(model_path.rglob('*.h5'))[0]), custom_objects=custom_objects)

        except Exception as e:
            logger.exception(e)
            exit()


def train(
        dataset: Path,
        outdir: Path,
        network: str,
        distribution: str,
        samplelimit: int,
        max_amplitude: float,
        input_shape: int,
        batch_size: int,
        patch_size: list,
        steps_per_epoch: int,
        depth_scalar: int,
        width_scalar: int,
        activation: str,
        fixedlr: bool,
        opt: str,
        lr: float,
        wd: float,
        warmup: int,
        decay_period: int,
        wavelength: float,
        psf_type: str,
        x_voxel_size: float,
        y_voxel_size: float,
        z_voxel_size: float,
        modes: int,
        pmodes: int,
        min_psnr: int,
        max_psnr: int,
        epochs: int,
        mul: bool,
        roi: Any = None,
        refractive_index: float = 1.33,
        no_phase: bool = False,
        plot_patches: bool = False
):
    network = network.lower()
    opt = opt.lower()
    restored = False

    if network == 'opticaltransformer':
        model = opticaltransformer.OpticalTransformer(
            name='OpticalTransformer',
            roi=roi,
            patches=patch_size,
            modes=pmodes,
            na_det=1.0,
            refractive_index=refractive_index,
            psf_type=psf_type,
            lambda_det=wavelength,
            x_voxel_size=x_voxel_size,
            y_voxel_size=y_voxel_size,
            z_voxel_size=z_voxel_size,
            depth_scalar=depth_scalar,
            width_scalar=width_scalar,
            mask_shape=input_shape,
            activation=activation,
            mul=mul,
            no_phase=no_phase
        )
    elif network == 'opticalresnet':
        model = opticalresnet.OpticalResNet(
            name='OpticalResNet',
            modes=pmodes,
            na_det=1.0,
            refractive_index=refractive_index,
            lambda_det=wavelength,
            psf_type=psf_type,
            x_voxel_size=x_voxel_size,
            y_voxel_size=y_voxel_size,
            z_voxel_size=z_voxel_size,
            depth_scalar=depth_scalar,
            width_scalar=width_scalar,
            mask_shape=input_shape,
            activation=activation,
            mul=mul,
            no_phase=no_phase
        )
    elif network == 'baseline':
        model = Baseline(
            name='Baseline',
            modes=pmodes,
            depth_scalar=depth_scalar,
            width_scalar=width_scalar,
            activation=activation,
        )
    elif network == 'phasenet':
        model = PhaseNet(
            name='PhaseNet',
            modes=pmodes
        )
    else:
        model = load(Path(network))
        checkpoint = tf.train.Checkpoint(model)
        status = checkpoint.restore(network)

        if status:
            logger.info(f"Restored from {network}")
            restored = True
        else:
            logger.info("Initializing from scratch")

    if network == 'phasenet':
        inputs = (input_shape, input_shape, input_shape, 1)
        lr = .0003
        opt = Adam(learning_rate=lr)
        loss = 'mse'
    else:
        if network == 'baseline':
            inputs = (input_shape, input_shape, input_shape, 1)
        else:
            inputs = (3 if no_phase else 6, input_shape, input_shape, 1)

        loss = tf.losses.MeanSquaredError(reduction=tf.losses.Reduction.SUM)

        """
            Adam: A Method for Stochastic Optimization: https://arxiv.org/pdf/1412.6980
            SGDR: Stochastic Gradient Descent with Warm Restarts: https://arxiv.org/pdf/1608.03983
            Decoupled weight decay regularization: https://arxiv.org/pdf/1711.05101 
        """
        if opt.lower() == 'adam':
            opt = Adam(learning_rate=lr)
        elif opt == 'sgd':
            opt = SGD(learning_rate=lr, momentum=0.9)
        elif opt.lower() == 'adamw':
            opt = AdamW(learning_rate=lr, weight_decay=wd)
        elif opt == 'sgdw':
            opt = SGDW(learning_rate=lr, weight_decay=wd, momentum=0.9)
        else:
            raise ValueError(f'Unknown optimizer `{opt}`')

    if not restored:
        model = model.build(input_shape=inputs)
        model.compile(
            optimizer=opt,
            loss=loss,
            metrics=[tf.keras.metrics.RootMeanSquaredError(), 'mae', 'mse'],
        )

    outdir = outdir / f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    outdir.mkdir(exist_ok=True, parents=True)
    logger.info(model.summary())

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

    h5_checkpoints = ModelCheckpoint(
        filepath=f"{outdir}.h5",
        monitor="loss",
        save_best_only=True,
        verbose=1,
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
        patience=20,
        verbose=1,
    )

    features = LambdaCallback(
        on_epoch_end=lambda epoch, logs: featuremaps(
            modelpath=outdir,
            amplitude_range=.3,
            wavelength=wavelength,
            psf_type=psf_type,
            x_voxel_size=x_voxel_size,
            y_voxel_size=y_voxel_size,
            z_voxel_size=z_voxel_size,
            cpu_workers=-1,
            psnr=100,
        ) if epoch % 50 == 0 else epoch
    )

    tensorboard = TensorBoardCallback(
        log_dir=outdir,
        profile_batch='500,520',
        histogram_freq=1,
    )

    if fixedlr:
        lrscheduler = LearningRateScheduler(
            initial_learning_rate=lr,
            verbose=0,
            fixed=True
        )
    else:
        lrscheduler = LearningRateScheduler(
            initial_learning_rate=lr,
            weight_decay=wd,
            decay_period=epochs if decay_period is None else decay_period,
            warmup_epochs=0 if warmup is None else warmup,
            alpha=.01,
            decay_multiplier=2.,
            decay=.9,
            verbose=1,
        )

    if dataset is None:

        config = dict(
            dtype=psf_type,
            psf_shape=inputs,
            snr=(min_psnr, max_psnr),
            max_jitter=1,
            n_modes=modes,
            distribution=distribution,
            amplitude_ranges=(-max_amplitude, max_amplitude),
            lam_detection=wavelength,
            batch_size=batch_size,
            x_voxel_size=x_voxel_size,
            y_voxel_size=y_voxel_size,
            z_voxel_size=z_voxel_size,
            cpu_workers=-1
        )

        train_data = data_utils.create_dataset(config)
        training_steps = steps_per_epoch
    else:
        train_data = data_utils.collect_dataset(
            dataset,
            distribution=distribution,
            samplelimit=samplelimit,
            max_amplitude=max_amplitude,
            no_phase=no_phase
        )

        sample_writer = tf.summary.create_file_writer(f'{outdir}/train_samples/')
        with sample_writer.as_default():
            for s in range(10):
                fig = None
                for i, (img, y) in enumerate(train_data.shuffle(1000).take(5)):
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

        def configure_for_performance(ds):
            ds = ds.cache()
            ds = ds.shuffle(batch_size)
            ds = ds.batch(batch_size)
            ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
            return ds

        train_data = configure_for_performance(train_data)
        training_steps = tf.data.experimental.cardinality(train_data).numpy()

    for img, y in train_data.shuffle(buffer_size=100).take(1):
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Training steps: [{training_steps}] {img.numpy().shape}")

        if plot_patches:
            img = np.expand_dims(img[0], axis=0)
            original = np.squeeze(img[0, 1])

            vmin = np.min(original)
            vmax = np.max(original)
            vcenter = (vmin + vmax) / 2
            step = .01
            print(vmin, vmax, vcenter)

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

            for p in patch_size:
                sample = Stem(
                    kernel_size=3,
                    activation=activation,
                    mask_shape=input_shape,
                    refractive_index=refractive_index,
                    lambda_det=wavelength,
                    psf_type=psf_type,
                    x_voxel_size=x_voxel_size,
                    y_voxel_size=y_voxel_size,
                    z_voxel_size=z_voxel_size,
                    mul=mul,
                    no_phase=no_phase
                )(img)

                if roi is not None:
                    sample = ROI(crop_shape=roi)(sample)

                patches = opticaltransformer.Patchify(patch_size=p)(sample)
                merged = opticaltransformer.Merge(patch_size=p)(patches)
                print(patches.shape)
                print(merged.shape)

                patches = patches[0, 1]
                merged = np.squeeze(merged[0, 1])

                plt.figure(figsize=(4, 4))
                plt.imshow(merged, cmap=cmap, vmin=vmin, vmax=vmax)
                plt.axis("off")
                plt.title('Merged')

                n = int(np.sqrt(patches.shape[0]))
                plt.figure(figsize=(4, 4))
                plt.title('Patches')
                for i, patch in enumerate(patches):
                    ax = plt.subplot(n, n, i + 1)
                    patch_img = tf.reshape(patch, (p, p)).numpy()
                    ax.imshow(patch_img, cmap=cmap, vmin=vmin, vmax=vmax)
                    ax.axis("off")
            plt.show()

    try:
        model.fit(
            train_data,
            steps_per_epoch=training_steps,
            epochs=epochs,
            verbose=2,
            shuffle=True,
            callbacks=[
                tblogger,
                tensorboard,
                pb_checkpoints,
                h5_checkpoints,
                earlystopping,
                defibrillator,
                # features,
                lrscheduler,
            ],
        )
    except tf.errors.ResourceExhaustedError as e:
        logger.error(e)
        sys.exit(1)


def bootstrap_predict(
    model: tf.keras.Model,
    inputs: np.array,
    psfgen: SyntheticPSF,
    resize: Any = None,
    batch_size: int = 1,
    n_samples: int = 10,
    threshold: float = 1e-4,
    verbose: bool = True,
    desc: str = 'MiniBatch-probabilistic-predictions',
    plot: Any = None,
    gamma: float = 1.0,
    return_embeddings: bool = False,
    rolling_embedding: bool = False,
):
    """
    Average predictions and compute stdev

    Args:
        model: pre-trained keras model
        inputs: encoded tokens to be processed
        psfgen: Synthetic PSF object
        resize: optional resize voxel size
        n_samples: number of predictions of average
        batch_size: number of samples per batch
        threshold: set predictions below threshold to zero
        desc: test to display for the progressbar
        verbose: a toggle for progress bar
        gamma: apply a gamma to the embeddings

    Returns:
        average prediction, stdev
    """
    def batch_generator(arr, s):
        for k in range(0, len(arr), s):
            yield arr[k:k + s]

    if rolling_embedding:
        model_inputs = np.stack([
            np.expand_dims(
                psfgen.rolling_embedding(psf=np.squeeze(i), plot=plot, strides=16),
                axis=-1
            ) for i in inputs
        ], axis=0)
    else:
        if resize is not None:
            inputs = np.stack([
                np.expand_dims(
                    preprocessing.resize(
                        np.squeeze(i),
                        crop_shape=psfgen.psf_shape,
                        voxel_size=psfgen.voxel_size,
                        sample_voxel_size=resize,
                        debug=plot,
                    ),
                    axis=-1
                ) for i in inputs
            ], axis=0)

        # check z-axis to compute embeddings for fourier models0
        if (model.input_shape[1] != inputs.shape[1]):
            model_inputs = np.stack([
                psfgen.embedding(psf=i, plot=plot, gamma=gamma) for i in inputs
            ], axis=0)
        else:
            # pass raw PSFs to the model
            model_inputs = inputs

    preds = None
    total = n_samples * (len(model_inputs) // batch_size)
    if verbose:
        pbar = tqdm(total=total, desc=desc, unit='batch')

    for i in range(n_samples):
        b = []
        if verbose:
            pbar.update(len(model_inputs) // batch_size)
            gen = tqdm(batch_generator(model_inputs, s=batch_size), leave=False)
        else:
            gen = batch_generator(model_inputs, s=batch_size)

        for batch in gen:
            # if plot is not None:
            #     img = np.squeeze(batch[0])
            #     input_img = np.squeeze(inputs[0])
            #
            #     if img.shape[0] != img.shape[1]:
            #         vmin, vmax, vcenter, step = 0, 2, 1, .1
            #         highcmap = plt.get_cmap('YlOrRd', 256)
            #         lowcmap = plt.get_cmap('YlGnBu_r', 256)
            #         low = np.linspace(0, 1 - step, int(abs(vcenter - vmin) / step))
            #         high = np.linspace(0, 1 + step, int(abs(vcenter - vmax) / step))
            #         cmap = np.vstack((lowcmap(low), [1, 1, 1, 1], highcmap(high)))
            #         cmap = mcolors.ListedColormap(cmap)
            #
            #         fig, axes = plt.subplots(3, 3)
            #
            #         if img.shape[0] == 6:
            #             for t in range(3):
            #                 inner = gridspec.GridSpecFromSubplotSpec(
            #                     1, 2, subplot_spec=axes[0, t], wspace=0.1, hspace=0.1
            #                 )
            #                 ax = fig.add_subplot(inner[0])
            #                 m = ax.imshow(img[t].T if t == 2 else img[t], cmap=cmap, vmin=vmin, vmax=vmax)
            #                 ax.set_xticks([])
            #                 ax.set_yticks([])
            #                 ax.set_xlabel(r'$\alpha = |\tau| / |\hat{\tau}|$')
            #                 ax = fig.add_subplot(inner[1])
            #                 ax.imshow(img[t+3].T if t == 2 else img[t+3], cmap='coolwarm', vmin=vmin, vmax=vmax)
            #                 ax.set_xticks([])
            #                 ax.set_yticks([])
            #                 ax.set_xlabel(r'$\varphi = \angle \tau$')
            #         else:
            #             m = axes[0, 0].imshow(input_img[input_img.shape[0]//2, :, :], cmap='hot', vmin=0, vmax=1)
            #             axes[0, 1].imshow(input_img[:, input_img.shape[1]//2, :], cmap='hot', vmin=0, vmax=1)
            #             axes[0, 2].imshow(input_img[:, :, input_img.shape[2]//2].T, cmap='hot', vmin=0, vmax=1)
            #             cax = inset_axes(axes[0, 2], width="10%", height="100%", loc='center right', borderpad=-3)
            #             cb = plt.colorbar(m, cax=cax)
            #             cax.yaxis.set_label_position("right")
            #             cax.set_ylabel('Input (middle)')
            #
            #         m = axes[1, 0].imshow(img[0], cmap=cmap, vmin=vmin, vmax=vmax)
            #         axes[1, 1].imshow(img[1], cmap=cmap, vmin=vmin, vmax=vmax)
            #         axes[1, 2].imshow(img[2].T, cmap=cmap, vmin=vmin, vmax=vmax)
            #         cax = inset_axes(axes[1, 2], width="10%", height="100%", loc='center right', borderpad=-3)
            #         cb = plt.colorbar(m, cax=cax)
            #         cax.yaxis.set_label_position("right")
            #         cax.set_ylabel(r'Embedding ($\alpha$)')
            #
            #         m = axes[2, 0].imshow(img[3], cmap='coolwarm', vmin=-1, vmax=1)
            #         axes[2, 1].imshow(img[4], cmap='coolwarm', vmin=-1, vmax=1)
            #         axes[2, 2].imshow(img[5].T, cmap='coolwarm', vmin=-1, vmax=1)
            #         cax = inset_axes(axes[2, 2], width="10%", height="100%", loc='center right', borderpad=-3)
            #         cb = plt.colorbar(m, cax=cax)
            #         cax.yaxis.set_label_position("right")
            #         cax.set_ylabel(r'Embedding ($\varphi$)')
            #
            #         for ax in axes.flatten():
            #             ax.axis('off')
            #     else:
            #         fig, axes = plt.subplots(1, 3)
            #
            #         axes[1].set_title(str(img.shape))
            #         m = axes[0].imshow(np.max(img, axis=0), cmap='Spectral_r')
            #         axes[1].imshow(np.max(img, axis=1), cmap='Spectral_r')
            #         axes[2].imshow(np.max(img, axis=2).T, cmap='Spectral_r')
            #
            #         for ax in axes.flatten():
            #             ax.axis('off')
            #
            #         cax = inset_axes(axes[2], width="10%", height="100%", loc='center right', borderpad=-3)
            #         cb = plt.colorbar(m, cax=cax)
            #         cax.yaxis.set_label_position("right")
            #
            #     if plot == True:
            #         plt.show()
            #     else:
            #         plt.savefig(f'{plot}.png', dpi=300, bbox_inches='tight', pad_inches=.25)

            p = model(batch, training=True).numpy()
            p[:, [0, 1, 2, 4]] = 0.
            p[np.abs(p) <= threshold] = 0.
            b.extend(p)

        if preds is None:
            preds = np.zeros((len(b), len(b[0]), n_samples))

        preds[:, :, i] = b

    mu = np.mean(preds, axis=-1)
    mu = mu.flatten() if mu.shape[0] == 1 else mu

    sigma = np.std(preds, axis=-1)
    sigma = sigma.flatten() if sigma.shape[0] == 1 else sigma

    if return_embeddings:
        return mu, sigma, model_inputs
    else:
        return mu, sigma


def predict(
    model: Path,
    psf_type: str,
    wavelength: float,
    x_voxel_size: float,
    y_voxel_size: float,
    z_voxel_size: float,
    max_jitter: float,
    cpu_workers: int,
    input_coverage: float = 1.0
):
    m = load(model)
    m.summary()

    modes = m.layers[-1].output_shape[-1]
    input_shape = m.layers[0].input_shape[0][1:-1]

    for dist in ['powerlaw', 'dirichlet']:
        for amplitude_range in [(.05, .1), (.1, .2), (.2, .4), (.4, .6)]:
            for snr in [25, 50]:
                psfargs = dict(
                    dtype=psf_type,
                    order='ansi',
                    snr=snr,
                    n_modes=modes,
                    distribution=dist,
                    gamma=1.5,
                    bimodal=True,
                    lam_detection=wavelength,
                    amplitude_ranges=amplitude_range,
                    psf_shape=3 * [input_shape[-1]],
                    x_voxel_size=x_voxel_size,
                    y_voxel_size=y_voxel_size,
                    z_voxel_size=z_voxel_size,
                    batch_size=1,
                    max_jitter=max_jitter,
                    cpu_workers=cpu_workers,
                )

                gen = SyntheticPSF(**psfargs)
                for i, (psf, y, psnr, zplanes, maxcounts) in zip(range(10), gen.generator(debug=True, otf=False)):
                    waves = np.round(utils.microns2waves(amplitude_range[0], wavelength), 2)
                    save_path = Path(
                        f"{model}/{dist}/c{input_coverage}/lambda-{waves}/psnr-{snr}/"
                    )
                    save_path.mkdir(exist_ok=True, parents=True)

                    if input_coverage != 1.:
                        psf = resize_with_crop_or_pad(psf, crop_shape=[int(s * input_coverage) for s in gen.psf_shape])
                        psf = resize_with_crop_or_pad(psf, crop_shape=gen.psf_shape, mode='minimum')

                    p, std = bootstrap_predict(
                        m,
                        psfgen=gen,
                        inputs=psf,
                        batch_size=1,
                        n_samples=1,
                        plot=save_path / f'embeddings_{i}',
                    )

                    p = Wavefront(p, lam_detection=wavelength)
                    logger.info('Prediction')
                    pprint(p.zernikes)

                    logger.info('GT')
                    y = Wavefront(y.flatten(), lam_detection=wavelength)
                    pprint(y.zernikes)

                    diff = Wavefront(y - p, lam_detection=wavelength)

                    p_psf = gen.single_psf(p, zplanes=0)
                    gt_psf = gen.single_psf(y, zplanes=0)
                    corrected_psf = gen.single_psf(diff, zplanes=0)

                    psf = np.squeeze(psf[0], axis=-1)

                    imsave(save_path / f'psf_{i}.tif', psf)
                    imsave(save_path / f'corrected_psf_{i}.tif', corrected_psf)

                    vis.diagnostic_assessment(
                        psf=psf,
                        gt_psf=gt_psf,
                        predicted_psf=p_psf,
                        corrected_psf=corrected_psf,
                        wavelength=wavelength,
                        psnr=psnr,
                        maxcounts=maxcounts,
                        y=y,
                        pred=p,
                        save_path=save_path / f'{i}',
                        display=False
                    )


def compare(
        model: Path,
        wavelength: float,
        x_voxel_size: float,
        y_voxel_size: float,
        z_voxel_size: float,
        max_jitter: float,
        cpu_workers: int,
):
    m = load(model)
    m.summary()

    modes = m.layers[-1].output_shape[-1]
    input_shape = m.layers[0].input_shape[0][1:-1]

    for dist in ['powerlaw', 'dirichlet']:
        for amplitude_range in [(0, .2), (.2, .4), (.4, .6)]:
            psfargs = dict(
                order='ansi',
                snr=50,
                n_modes=modes,
                distribution=dist,
                gamma=1.5,
                bimodal=True,
                lam_detection=wavelength,
                amplitude_ranges=amplitude_range,
                psf_shape=3 * [input_shape[-1]],
                x_voxel_size=x_voxel_size,
                y_voxel_size=y_voxel_size,
                z_voxel_size=z_voxel_size,
                batch_size=1,
                max_jitter=max_jitter,
                cpu_workers=cpu_workers,
            )

            gen = SyntheticPSF(**psfargs)

            for i, (psf, y, psnr, zplanes, maxcounts) in zip(range(10), gen.generator(debug=True)):
                # rotate to match matlab and DM
                psf = np.squeeze(psf[0], axis=-1)

                # compute amp ratio and phase
                model_input = gen.embedding(psf)

                model_input = np.expand_dims(np.expand_dims(model_input, axis=0), axis=-1)

                p, std = bootstrap_predict(m, psfgen=gen, inputs=model_input, batch_size=1, n_samples=1)

                p = Wavefront(p, lam_detection=wavelength)
                logger.info('Prediction')
                pprint(p.zernikes)

                logger.info('GT')
                y = Wavefront(y.flatten(), lam_detection=wavelength)
                pprint(y.zernikes)

                diff = Wavefront(y - p, lam_detection=wavelength)

                p_psf = gen.single_psf(p, zplanes=0)
                corrected_psf = gen.single_psf(diff, zplanes=0)

                model_input = np.squeeze(model_input[0], axis=-1)
                waves = np.round(utils.microns2waves(amplitude_range[1], wavelength), 2)
                save_path = Path(
                    f"{model}/compare/{dist}/lambda-{waves}/psnr-{psfargs['snr']}/"
                )
                save_path.mkdir(exist_ok=True, parents=True)

                imsave(save_path / f'input_{i}.tif', model_input)
                imsave(save_path / f'input_psf_{i}.tif', psf)
                imsave(save_path / f'corrected_psf_{i}.tif', corrected_psf)

                logger.info('Matlab')
                pred_matlab = np.zeros_like(p.amplitudes)
                m_amps = np.array(experimental.phase_retrieval(
                    psf=save_path / f'input_psf_{i}.tif',
                    dx=x_voxel_size,
                    dz=z_voxel_size,
                    wavelength=wavelength,
                    n_modes=p.amplitudes.shape[0]
                ))
                pred_matlab[:m_amps.shape[0]] = m_amps
                pred_matlab = Wavefront(pred_matlab.flatten(), lam_detection=wavelength)
                matlab_diff = Wavefront(y - pred_matlab, lam_detection=wavelength)
                matlab_corrected_psf = gen.single_psf(matlab_diff, zplanes=0)
                imsave(save_path / f'matlab_corrected_psf_{i}.tif', matlab_corrected_psf)
                pprint(pred_matlab.zernikes)

                vis.matlab_diagnostic_assessment(
                    psf=model_input,
                    gt_psf=psf,
                    predicted_psf=p_psf,
                    corrected_psf=corrected_psf,
                    matlab_corrected_psf=matlab_corrected_psf,
                    psnr=psnr,
                    maxcounts=maxcounts,
                    wavelength=wavelength,
                    y=y,
                    pred=p,
                    pred_matlab=pred_matlab,
                    save_path=save_path / f'{i}',
                    display=False
                )


def deconstruct(
        model: Path,
        max_jitter: float = 1,
        cpu_workers: int = -1,
        x_voxel_size: float = .15,
        y_voxel_size: float = .15,
        z_voxel_size: float = .6,
        wavelength: float = .605,
        eval_distribution: str = 'powerlaw'
):
    m = load(model)
    m.summary()

    modes = m.layers[-1].output_shape[-1]
    input_shape = m.layers[0].input_shape[0][1:-1]

    for i in range(6):
        psfargs = dict(
            snr=100,
            n_modes=modes,
            distribution=eval_distribution,
            lam_detection=wavelength,
            amplitude_ranges=((.05*i), (.05*(i+1))),
            psf_shape=3 * [input_shape[-1]],
            x_voxel_size=x_voxel_size,
            y_voxel_size=y_voxel_size,
            z_voxel_size=z_voxel_size,
            batch_size=1,
            max_jitter=max_jitter,
            cpu_workers=cpu_workers,
        )
        gen = SyntheticPSF(**psfargs)
        psf, y, psnr, zplanes, maxcounts = next(gen.generator(debug=True))
        p, std = bootstrap_predict(m, psfgen=gen, inputs=psf, batch_size=1)

        p = Wavefront(p, lam_detection=wavelength)
        logger.info('Prediction')
        pprint(p.zernikes)

        logger.info('GT')
        y = Wavefront(y, lam_detection=wavelength)
        pprint(y.zernikes)

        diff = Wavefront(y - p, lam_detection=wavelength)

        p_psf = gen.single_psf(p, zplanes=0)
        gt_psf = gen.single_psf(y, zplanes=0)
        corrected_psf = gen.single_psf(diff, zplanes=0)

        psf = np.squeeze(psf[0], axis=-1)
        save_path = Path(f'{model}/deconstruct/{eval_distribution}/')
        save_path.mkdir(exist_ok=True, parents=True)

        vis.diagnostic_assessment(
            psf=psf,
            gt_psf=gt_psf,
            predicted_psf=p_psf,
            corrected_psf=corrected_psf,
            psnr=psnr,
            maxcounts=maxcounts,
            wavelength=wavelength,
            y=y,
            pred=p,
            save_path=save_path / f'{i}',
            display=False
        )

        vis.plot_dmodes(
            psf=psf,
            gen=gen,
            y=y,
            pred=p,
            save_path=save_path / f'{i}_dmodes',
        )


def featuremaps(
        modelpath: Path,
        amplitude_range: float,
        wavelength: float,
        psf_type: str,
        x_voxel_size: float,
        y_voxel_size: float,
        z_voxel_size: float,
        cpu_workers: int,
        psnr: int,
):
    plt.rcParams.update({
        'font.size': 25,
        'axes.titlesize': 30,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
    })

    logger.info(f"Models: {modelpath}")
    model = load(modelpath)

    input_shape = model.layers[1].output_shape[1:-1]
    modes = model.layers[-1].output_shape[-1]
    model = Model(
        inputs=model.input,
        outputs=[layer.output for layer in model.layers],
    )
    phi = np.zeros(modes)
    phi[10] = amplitude_range
    amplitude_range = Wavefront(phi, lam_detection=wavelength)

    psfargs = dict(
        n_modes=modes,
        dtype=psf_type,
        psf_shape=3 * [input_shape[1]],
        distribution='single',
        lam_detection=wavelength,
        amplitude_ranges=amplitude_range,
        x_voxel_size=x_voxel_size,
        y_voxel_size=y_voxel_size,
        z_voxel_size=z_voxel_size,
        batch_size=1,
        snr=psnr,
        max_jitter=1,
        cpu_workers=cpu_workers,
    )
    gen = SyntheticPSF(**psfargs)

    if input_shape[0] == 3 or input_shape[0] == 6:
        inputs = gen.single_otf(
            amplitude_range,
            zplanes=0,
            normed=True,
            noise=True,
            na_mask=True,
            ratio=True,
            augmentation=True
        )
    else:
        inputs = gen.single_psf(amplitude_range, zplanes=0, normed=True, noise=True, augmentation=True)

    inputs = np.expand_dims(np.stack(inputs, axis=0), 0)
    inputs = np.expand_dims(np.stack(inputs, axis=0), -1)

    layers = [layer.name for layer in model.layers[1:25]]
    maps = model.predict(inputs)[1:25]
    nrows = sum([1 for fmap in maps if len(fmap.shape[1:]) >= 3])

    fig = plt.figure(figsize=(150, 600))
    gs = fig.add_gridspec((nrows * input_shape[0]) + 1, 8)

    i = 0
    logger.info('Plotting featuremaps...')
    for name, fmap in zip(layers, maps):
        logger.info(f"Layer {name}: {fmap.shape}")
        fmap = fmap[0]

        if fmap.ndim < 2:
            continue

        if layers[0] == 'patchify':
            patches = fmap.shape[-2]
            features = fmap.shape[-1]
            fmap_size = patches // int(np.sqrt(maps[0].shape[-2]))
            fmap = np.reshape(fmap, (fmap.shape[0], fmap_size, fmap_size, features))

        if len(fmap.shape) == 4:
            if input_shape[0] == 3 or input_shape[0] == 6:

                i += 1
                features = fmap.shape[-1]
                window = fmap.shape[1], fmap.shape[2]
                grid = np.zeros((window[0] * input_shape[0], window[1] * features))

                for j in range(input_shape[0]):
                    for f in range(features):
                        vol = fmap[j, :, :, f]
                        grid[
                            j * window[0]:(j + 1) * window[0],
                            f * window[1]:(f + 1) * window[1]
                        ] = vol

                space = grid.flatten()
                vmin = np.nanquantile(space, .02)
                vmax = np.nanquantile(space, .98)
                vcenter = (vmin + vmax)/2
                step = .01

                highcmap = plt.get_cmap('YlOrRd', 256)
                lowcmap = plt.get_cmap('terrain', 256)
                low = np.linspace(0, 1 - step, int(abs(vcenter - vmin) / step))
                high = np.linspace(0, 1 + step, int(abs(vcenter - vmax) / step))
                cmap = np.vstack((lowcmap(low), [1, 1, 1, 1], highcmap(high)))
                cmap = mcolors.ListedColormap(cmap)

                ax = fig.add_subplot(gs[i, :])
                ax.set_title(f"{name.upper()} {fmap.shape}")
                m = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_aspect('equal')
                ax.axis('off')

                cax = inset_axes(ax, width="1%", height="100%", loc='center right', borderpad=-3)
                cb = plt.colorbar(m, cax=cax)
                cax.yaxis.set_label_position("right")
            else:
                for j in range(3):
                    i += 1
                    features = fmap.shape[-1]
                    if j == 0:
                        window = fmap.shape[1], fmap.shape[2]
                    elif j == 1:
                        window = fmap.shape[0], fmap.shape[2]
                    else:
                        window = fmap.shape[0], fmap.shape[1]

                    grid = np.zeros((window[0], window[1] * features))

                    for f in range(features):
                        vol = np.max(fmap[:, :, :, f], axis=j)
                        grid[:, f * window[1]:(f + 1) * window[1]] = vol

                    ax = fig.add_subplot(gs[i, :])

                    if j == 0:
                        ax.set_title(f"{name.upper()} {fmap.shape}")

                    ax.imshow(grid, cmap='hot', vmin=0, vmax=1)
                    ax.set_aspect('equal')
                    ax.axis('off')
        else:
            i += 1
            features = fmap.shape[-1]
            window = fmap.shape[0], fmap.shape[1]
            grid = np.zeros((window[0], window[1] * features))

            for f in range(features):
                vol = fmap[:, :, f]
                # vol = (vol - vol.min()) / (vol.max() - vol.min())
                grid[:, f * window[1]:(f + 1) * window[1]] = vol

            # from tifffile import imsave
            # imsave(f'{modelpath}/{name.upper()}_{i}.tif', grid)

            ax = fig.add_subplot(gs[i, :])
            ax.set_title(f"{name.upper()} {fmap.shape}")
            m = ax.imshow(grid, cmap='Spectral_r')
            ax.set_aspect('equal')
            ax.axis('off')

            cax = inset_axes(ax, width="1%", height="100%", loc='center right', borderpad=-3)
            cb = plt.colorbar(m, cax=cax)
            cax.yaxis.set_label_position("right")

    plt.subplots_adjust(top=0.95, right=0.95, wspace=.2)
    plt.savefig(f'{modelpath}/featuremaps.pdf', bbox_inches='tight', pad_inches=.25)
    return fig


def kernels(modelpath: Path, activation='relu'):
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.autolimit_mode': 'round_numbers'
    })

    def factorization(n):
        """Calculates kernel grid dimensions."""
        for i in range(int(np.sqrt(float(n))), 0, -1):
            if n % i == 0:
                return i, n // i

    logger.info(f"Model: {modelpath}")
    model = load(modelpath)
    layers = [layer for layer in model.layers if str(layer.name).startswith('conv')]

    if isinstance(activation, str):
        activation = tf.keras.layers.Activation(activation)

    logger.info("Plotting learned kernels")
    for i, layer in enumerate(layers):
        fig, ax = plt.subplots(figsize=(8, 11))
        grid, biases = layer.get_weights()
        logger.info(f"Layer [{layer.name}]: {layer.output_shape}, {grid.shape}")

        grid = activation(grid)
        grid = np.squeeze(grid, axis=0)

        with tf.name_scope(layer.name):
            low, high = tf.reduce_min(grid), tf.reduce_max(grid)
            grid = (grid - low) / (high - low)
            grid = tf.pad(grid, ((1, 1), (1, 1), (0, 0), (0, 0)))

            r, c, chan_in, chan_out = grid.shape
            grid_r, grid_c = factorization(chan_out)

            grid = tf.transpose(grid, (3, 0, 1, 2))
            grid = tf.reshape(grid, tf.stack([grid_c, r * grid_r, c, chan_in]))
            grid = tf.transpose(grid, (0, 2, 1, 3))
            grid = tf.reshape(grid, tf.stack([1, c * grid_c, r * grid_r, chan_in]))
            grid = tf.transpose(grid, (0, 2, 1, 3))

            while grid.shape[3] > 4:
                a, b = tf.split(grid, 2, axis=3)
                grid = tf.concat([a, b], axis=0)

            _, a, b, channels = grid.shape
            if channels == 2:
                grid = tf.concat([grid, tf.zeros((1, a, b, 1))], axis=3)

        img = grid[-1, :, :, 0]
        ax.imshow(img)
        ax.set_aspect('equal')
        ax.axis('off')

        plt.savefig(f'{modelpath}/kernels_{layer.name}.pdf', bbox_inches='tight', pad_inches=.25)
        plt.savefig(f'{modelpath}/kernels_{layer.name}.png', dpi=300, bbox_inches='tight', pad_inches=.25)
