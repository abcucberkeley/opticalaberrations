import matplotlib
matplotlib.use('Agg')

import numexpr
numexpr.set_num_threads(numexpr.detect_number_of_cores())

import logging
import sys
import os
import h5py
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Any, Union
from functools import partial
from line_profiler_pycharm import profile

import pandas as pd
from scipy import stats as st
from skimage.restoration import richardson_lucy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import scipy as sp
from tqdm import tqdm
from tifffile import imsave
import raster_geometry as rg

import tensorflow as tf
from tensorflow.keras.models import load_model
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
import opticalnet
import opticalresnet
import opticaltransformer
import baseline
import otfnet

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
tf.get_logger().setLevel(logging.ERROR)


@profile
def load_metadata(model_path: Path, psf_shape: tuple = (64, 64, 64), n_modes=None, **kwargs):
    # print(f"my suffix = {model_path.suffix}, my model = {model_path}")
    if not model_path.suffix == '.h5':       
        model_path = list(model_path.rglob('*.h5'))[0]

    with h5py.File(model_path, 'r') as file:
        psfgen = SyntheticPSF(
            psf_type=np.array(file.get('psf_type')[:]),
            psf_shape=psf_shape,
            n_modes=int(file.get('n_modes')[()]) if n_modes is None else n_modes,
            lam_detection=float(file.get('wavelength')[()]),
            x_voxel_size=float(file.get('x_voxel_size')[()]),
            y_voxel_size=float(file.get('y_voxel_size')[()]),
            z_voxel_size=float(file.get('z_voxel_size')[()]),
            **kwargs
        )
    return psfgen


@profile
def load(model_path: Path, mosaic=False):
    model_path = Path(model_path)

    if 'transformer' in str(model_path):
        custom_objects = {
            "ROI": ROI,
            "Stem": opticaltransformer.Stem,
            "Patchify": opticaltransformer.Patchify,
            "Merge": opticaltransformer.Merge,
            "PatchEncoder": opticaltransformer.PatchEncoder,
            "MLP": opticaltransformer.MLP,
            "Transformer": opticaltransformer.Transformer,
        }
    elif 'resnet' in str(model_path):
        custom_objects = {
            "Stem": Stem,
            "MaskedActivation": MaskedActivation,
            "SpatialAttention": SpatialAttention,
            "DepthwiseConv3D": DepthwiseConv3D,
            "CAB": opticalresnet.CAB,
            "TB": opticalresnet.TB,
        }
    else:
        custom_objects = {
            "ROI": ROI,
            "Stem": opticalnet.Stem,
            "Patchify": opticalnet.Patchify,
            "Merge": opticalnet.Merge,
            "PatchEncoder": opticalnet.PatchEncoder,
            "MLP": opticalnet.MLP,
            "Transformer": opticalnet.Transformer,
        }

    if mosaic:
        if model_path.is_file() and model_path.suffix == '.h5':
            model_path = Path(model_path)
        else:
            model_path = Path(list(model_path.rglob('*.h5'))[0])

        model = load_model(model_path, custom_objects=custom_objects)
        return model

    else:
        try:
            try:
                '''.pb format'''
                if model_path.is_file() and model_path.suffix == '.pb':
                    return load_model(str(model_path.parent))
                else:
                    return load_model(str(list(model_path.rglob('saved_model.pb'))[0].parent))

            except IndexError or FileNotFoundError or OSError:
                '''.h5/hdf5 format'''
                if model_path.is_file() and model_path.suffix == '.h5':
                    model_path = str(model_path)
                else:
                    model_path = str(list(model_path.rglob('*.h5'))[0])

                model = load_model(model_path, custom_objects=custom_objects)
                return model

        except Exception as e:
            logger.exception(e)
            exit()


@profile
def bootstrap_predict(
    model: tf.keras.Model,
    inputs: np.array,
    psfgen: SyntheticPSF,
    batch_size: int = 1,
    n_samples: int = 10,
    threshold: float = 0.1,
    freq_strength_threshold: float = .01,
    ignore_modes: list = (0, 1, 2, 4),
    verbose: bool = True,
    plot: Any = None,
    no_phase: bool = False,
    desc: str = 'MiniBatch-probabilistic-predictions',
):
    """
    Average predictions and compute stdev

    Args:
        model: pre-trained keras model
        inputs: encoded tokens to be processed
        psfgen: Synthetic PSF object
        n_samples: number of predictions of average
        ignore_modes: list of modes to ignore
        batch_size: number of samples per batch
        threshold: set predictions below threshold to zero (wavelength)
        desc: test to display for the progressbar
        verbose: a toggle for progress bar

    Returns:
        average prediction, stdev
    """
    threshold = utils.waves2microns(threshold, wavelength=psfgen.lam_detection)
    ignore_modes = list(map(int, ignore_modes))
    logger.info(f"Ignoring modes: {ignore_modes}")

    # check z-axis to compute embeddings for fourier models
    if len(inputs.shape) == 3:
        emb = model.input_shape[1] == inputs.shape[0]
    else:
        emb = model.input_shape[1] == inputs.shape[1]

    if not emb:
        logger.info(f"Generating embeddings")

        model_inputs = []
        for i in inputs:
            emb = psfgen.embedding(
                psf=np.squeeze(i),
                plot=plot,
                no_phase=no_phase,
                principle_planes=True,
                freq_strength_threshold=freq_strength_threshold
            )

            if no_phase and model.input_shape[1] == 6:
                phase_mask = np.zeros((3, model.input_shape[2], model.input_shape[3]))
                emb = np.concatenate([emb, phase_mask], axis=0)
            elif model.input_shape[1] == 3:
                emb = emb[:3]

            model_inputs.append(emb)

        model_inputs = np.stack(model_inputs, axis=0)
    else:
        # pass raw PSFs to the model
        model_inputs = inputs

    model_inputs = np.nan_to_num(model_inputs, nan=0, posinf=0, neginf=0)
    model_inputs = model_inputs[..., np.newaxis] if model_inputs.shape[-1] != 1 else model_inputs
    features = np.array([np.count_nonzero(s) for s in inputs])

    logger.info(f"[BS={batch_size}, n={n_samples}] {desc}")
    gen = tf.data.Dataset.from_tensor_slices(model_inputs).batch(batch_size).repeat(n_samples)
    preds = model.predict(gen, batch_size=batch_size, verbose=verbose)
    preds[:, ignore_modes] = 0.
    preds[np.abs(preds) <= threshold] = 0.
    preds = np.stack(np.split(preds, n_samples), axis=-1)

    mu = np.mean(preds, axis=-1)
    mu = mu.flatten() if mu.shape[0] == 1 else mu

    sigma = np.std(preds, axis=-1)
    sigma = sigma.flatten() if sigma.shape[0] == 1 else sigma

    mu[np.where(features == 0)[0]] = np.zeros_like(mu[0])
    sigma[np.where(features == 0)[0]] = np.zeros_like(sigma[0])

    return mu, sigma


@profile
def predict_sign(
    gen: SyntheticPSF,
    init_preds: np.ndarray,
    followup_preds: np.ndarray,
    sign_threshold: float = .9,
    plot: Any = None,
    bar_width: float = .35
):
    def pct_change(cur, prev):
        t = utils.waves2microns(.05, wavelength=gen.lam_detection)
        cur[cur < t] = 0
        prev[prev < t] = 0

        if np.array_equal(cur, prev):
            return np.zeros_like(prev)

        pct = ((cur - prev) / (prev+1e-6)) * 100.0
        pct[pct > 100] = 100
        pct[pct < -100] = -100
        return pct

    init_preds = np.abs(init_preds)
    followup_preds = np.abs(followup_preds)

    preds = init_preds.copy()
    pchange = pct_change(followup_preds, init_preds)

    # flip signs and make any necessary adjustments to the amplitudes based on the followup predictions
    # adj = pchange.copy()
    # adj[np.where(pchange > 0)] = -200
    # adj[np.where(pchange < 0)] += 100
    # preds += preds * (adj/100)

    # threshold-based sign prediction
    threshold = sign_threshold * init_preds
    flips = np.stack(np.where(followup_preds > threshold), axis=0)

    if len(np.squeeze(preds).shape) == 1:
        preds[flips[0]] *= -1
    else:
        preds[flips[0], flips[1]] *= -1

    if plot is not None:
        if len(np.squeeze(preds).shape) == 1:
            init_preds_wave = Wavefront(init_preds, lam_detection=gen.lam_detection).amplitudes
            init_preds_wave_error = Wavefront(np.zeros_like(init_preds), lam_detection=gen.lam_detection).amplitudes

            followup_preds_wave = Wavefront(followup_preds, lam_detection=gen.lam_detection).amplitudes
            followup_preds_wave_error = Wavefront(np.zeros_like(followup_preds),
                                                  lam_detection=gen.lam_detection).amplitudes

            preds_wave = Wavefront(preds, lam_detection=gen.lam_detection).amplitudes
            preds_error = Wavefront(np.zeros_like(preds), lam_detection=gen.lam_detection).amplitudes

            percent_changes = pchange
            percent_changes_error = np.zeros_like(pchange)
        else:
            init_preds_wave = Wavefront(np.mean(init_preds, axis=0), lam_detection=gen.lam_detection).amplitudes
            init_preds_wave_error = Wavefront(np.std(init_preds, axis=0), lam_detection=gen.lam_detection).amplitudes

            followup_preds_wave = Wavefront(np.mean(followup_preds, axis=0), lam_detection=gen.lam_detection).amplitudes
            followup_preds_wave_error = Wavefront(np.std(followup_preds, axis=0), lam_detection=gen.lam_detection).amplitudes

            preds_wave = Wavefront(np.mean(preds, axis=0), lam_detection=gen.lam_detection).amplitudes
            preds_error = Wavefront(np.std(preds, axis=0), lam_detection=gen.lam_detection).amplitudes

            percent_changes = np.mean(pchange, axis=0)
            percent_changes_error = np.std(pchange, axis=0)

        fig, axes = plt.subplots(3, 1, figsize=(16, 8))

        axes[0].bar(
            np.arange(len(preds_wave)) - bar_width / 2,
            init_preds_wave,
            yerr=init_preds_wave_error,
            capsize=5,
            alpha=.75,
            color='C0',
            align='center',
            ecolor='grey',
            label='Initial',
            width=bar_width
        )
        axes[0].bar(
            np.arange(len(preds_wave)) + bar_width / 2,
            followup_preds_wave,
            yerr=followup_preds_wave_error,
            capsize=5,
            alpha=.75,
            color='C1',
            align='center',
            ecolor='grey',
            label='Followup',
            width=bar_width
        )

        axes[0].legend(frameon=False, loc='upper left')
        axes[0].set_xlim((-1, len(preds_wave)))
        axes[0].set_xticks(range(0, len(preds_wave)))
        axes[0].spines.right.set_visible(False)
        axes[0].spines.left.set_visible(False)
        axes[0].spines.top.set_visible(False)
        axes[0].grid(True, which="both", axis='y', lw=1, ls='--', zorder=0)
        axes[0].set_ylabel(r'Zernike coefficients ($\mu$m)')

        axes[1].plot(np.zeros_like(percent_changes), '--', color='lightgrey')
        axes[1].bar(
            range(gen.n_modes),
            percent_changes,
            yerr=percent_changes_error,
            capsize=10,
            color='C2',
            alpha=.75,
            align='center',
            ecolor='grey',
        )
        axes[1].set_xlim((-1, len(preds_wave)))
        axes[1].set_xticks(range(0, len(preds_wave)))
        axes[1].set_ylim((-100, 100))
        axes[1].set_yticks(range(-100, 125, 25))
        axes[1].set_yticklabels(['-100+', '-75', '-50', '-25', '0', '25', '50', '75', '100+'])
        axes[1].spines.right.set_visible(False)
        axes[1].spines.left.set_visible(False)
        axes[1].spines.top.set_visible(False)
        axes[1].grid(True, which="both", axis='y', lw=1, ls='--', zorder=0)
        axes[1].set_ylabel(f'Percent change')

        axes[2].plot(np.zeros_like(preds_wave), '--', color='lightgrey')
        axes[2].bar(
            range(gen.n_modes),
            preds_wave,
            yerr=preds_error,
            capsize=10,
            alpha=.75,
            color='dimgrey',
            align='center',
            ecolor='grey',
            label='Predictions',
        )
        axes[2].set_xlim((-1, len(preds_wave)))
        axes[2].set_xticks(range(0, len(preds_wave)))
        axes[2].spines.right.set_visible(False)
        axes[2].spines.left.set_visible(False)
        axes[2].spines.top.set_visible(False)
        axes[2].grid(True, which="both", axis='y', lw=1, ls='--', zorder=0)
        axes[2].set_ylabel(r'Zernike coefficients ($\mu$m)')

        plt.tight_layout()
        plt.savefig(f'{plot}_sign_correction.svg', dpi=300, bbox_inches='tight', pad_inches=.25)

    return preds, pchange

@profile
def dual_stage_prediction(
    model: tf.keras.Model,
    inputs: np.array,
    gen: SyntheticPSF,
    modelgen: SyntheticPSF,
    plot: Any = None,
    verbose: bool = False,
    threshold: float = 0.,
    freq_strength_threshold: float = .01,
    sign_threshold: float = .9,
    n_samples: int = 1,
    batch_size: int = 1,
    ignore_modes: list = (0, 1, 2, 4),
    prev_pred: Any = None,
    estimate_sign_with_decon: bool = False,
    decon_iters: int = 5
):

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.autolimit_mode': 'round_numbers'
    })

    prev_pred = None if (prev_pred is None or str(prev_pred) == 'None') else prev_pred

    init_preds, stdev = bootstrap_predict(
        model,
        inputs,
        psfgen=modelgen,
        n_samples=n_samples,
        no_phase=True,
        verbose=verbose,
        batch_size=batch_size,
        threshold=threshold,
        ignore_modes=ignore_modes,
        freq_strength_threshold=freq_strength_threshold,
    )
    init_preds = np.abs(init_preds)

    if estimate_sign_with_decon:
        logger.info(f"Estimating signs w/ Decon")
        abrs = range(init_preds.shape[0]) if len(init_preds.shape) > 1 else range(1)
        make_psf = partial(gen.single_psf, normed=True, noise=False)
        psfs = np.stack(gen.batch(
            make_psf,
            [Wavefront(init_preds[i], lam_detection=gen.lam_detection) for i in abrs]
        ), axis=0)

        with sp.fft.set_workers(-1):
            followup_inputs = richardson_lucy(np.squeeze(inputs), np.squeeze(psfs), num_iter=decon_iters)

        if len(followup_inputs.shape) == 3:
            followup_inputs = followup_inputs[np.newaxis, ..., np.newaxis]
        else:
            followup_inputs = followup_inputs[..., np.newaxis]

        followup_preds, stdev = bootstrap_predict(
            model,
            followup_inputs,
            psfgen=modelgen,
            n_samples=n_samples,
            no_phase=True,
            verbose=False,
            batch_size=batch_size,
            threshold=threshold,
            ignore_modes=ignore_modes,
            freq_strength_threshold=freq_strength_threshold,
        )
        followup_preds = np.abs(followup_preds)

        preds, pchanges = predict_sign(
            gen=gen,
            init_preds=init_preds,
            followup_preds=followup_preds,
            sign_threshold=sign_threshold,
            plot=plot
        )

    elif prev_pred is not None:
        logger.info(f"Evaluating signs")
        followup_preds = init_preds.copy()
        init_preds = np.abs(pd.read_csv(prev_pred, header=0)['amplitude'].values)

        preds, pchanges = predict_sign(
            gen=gen,
            init_preds=init_preds,
            followup_preds=followup_preds,
            sign_threshold=sign_threshold,
            plot=plot
        )

    else:
        preds = init_preds
        pchanges = np.zeros_like(preds)

    return preds, stdev, pchanges


@profile
def eval_sign(
    model: tf.keras.Model,
    inputs: np.array,
    gen: SyntheticPSF,
    ys: np.array,
    batch_size: int,
    reference: Any = None,
    plot: Any = None,
    threshold: float = 0.,
    sign_threshold: float = .95,
    desc: str = 'Eval',
):
    init_preds, stdev = bootstrap_predict(
        model,
        inputs,
        psfgen=gen,
        batch_size=batch_size,
        n_samples=1,
        no_phase=True,
        threshold=threshold,
    )
    if len(init_preds.shape) > 1:
        init_preds = np.abs(init_preds)[:, :ys.shape[-1]]
    else:
        init_preds = np.abs(init_preds)[:ys.shape[-1]]

    res = ys - init_preds
    g = partial(
        gen.single_psf,
        normed=True,
        noise=False,
        meta=False
    )
    followup_inputs = np.expand_dims(np.stack(gen.batch(g, res), axis=0), -1)

    if reference is not None:
        conv = partial(utils.fftconvolution, sample=reference)
        followup_inputs = np.array(utils.multiprocess(conv, followup_inputs))

    followup_preds, stdev = bootstrap_predict(
        model,
        followup_inputs,
        psfgen=gen,
        batch_size=batch_size,
        n_samples=1,
        no_phase=True,
        threshold=threshold,
        desc=desc,
    )
    if len(init_preds.shape) > 1:
        followup_preds = np.abs(followup_preds)[:, :ys.shape[-1]]
    else:
        followup_preds = np.abs(followup_preds)[:ys.shape[-1]]

    preds, pchanges = predict_sign(
        gen=gen,
        init_preds=init_preds,
        followup_preds=followup_preds,
        sign_threshold=sign_threshold,
        plot=plot
    )

    return preds


@profile
def beads(
    gen: SyntheticPSF,
    object_size: float = 0,
    num_objs: int = 1,
    radius: float = .45,
):
    np.random.seed(os.getpid())
    reference = np.zeros(gen.psf_shape)
    np.random.seed(os.getpid())

    for i in range(num_objs):
        if object_size > 0:
            reference += rg.sphere(
                shape=gen.psf_shape,
                radius=object_size,
                position=np.random.uniform(low=.2, high=.8, size=3)
            ).astype(np.float) * np.random.random()
        else:
            if radius > 0:
                reference[
                    np.random.randint(int(gen.psf_shape[0] * (.5 - radius)), int(gen.psf_shape[0] * (.5 + radius))),
                    np.random.randint(int(gen.psf_shape[1] * (.5 - radius)), int(gen.psf_shape[1] * (.5 + radius))),
                    np.random.randint(int(gen.psf_shape[2] * (.5 - radius)), int(gen.psf_shape[2] * (.5 + radius)))
                ] += np.random.random()
            else:
                reference[gen.psf_shape[0]//2, gen.psf_shape[1]//2, gen.psf_shape[2]//2] += np.random.random()

    reference /= np.max(reference)
    return reference


@profile
def predict(model: Path, psnr: int = 30):
    m = load(model)
    m.summary()

    for dist in ['single', 'powerlaw', 'dirichlet']:
        for amplitude_range in [(.05, .1), (.1, .3)]:
            gen = load_metadata(
                model,
                snr=1000,
                bimodal=True,
                rotate=True,
                batch_size=1,
                amplitude_ranges=amplitude_range,
                distribution=dist,
                psf_shape=(64, 64, 64)
            )
            for s, (psf, y, snr, maxcounts) in zip(range(10), gen.generator(debug=True)):
                psf = np.squeeze(psf)

                for npoints in tqdm([1, 2, 5, 10]):
                    reference = beads(
                        gen=gen,
                        object_size=0,
                        num_objs=npoints
                    )

                    img = utils.fftconvolution(sample=reference, kernel=psf)
                    img *= psnr ** 2

                    rand_noise = gen._random_noise(
                        image=img,
                        mean=gen.mean_background_noise,
                        sigma=gen.sigma_background_noise
                    )
                    noisy_img = rand_noise + img
                    noisy_img /= np.max(noisy_img)

                    save_path = Path(
                        f"{model.with_suffix('')}/samples/{dist}/um-{amplitude_range[-1]}/npoints-{npoints}"
                    )
                    save_path.mkdir(exist_ok=True, parents=True)

                    p = eval_sign(
                        model=m,
                        inputs=noisy_img[np.newaxis, :, :, :, np.newaxis],
                        reference=reference,
                        gen=gen,
                        ys=y,
                        batch_size=1,
                        plot=save_path / f'embeddings_{s}',
                    )

                    p_wave = Wavefront(p, lam_detection=gen.lam_detection)
                    y_wave = Wavefront(y.flatten(), lam_detection=gen.lam_detection)
                    diff = y_wave - p_wave

                    p_psf = gen.single_psf(p_wave)
                    gt_psf = gen.single_psf(y_wave)
                    corrected_psf = gen.single_psf(diff)

                    imsave(save_path / f'psf_{s}.tif', noisy_img)
                    imsave(save_path / f'corrected_psf_{s}.tif', corrected_psf)

                    vis.diagnostic_assessment(
                        psf=noisy_img,
                        gt_psf=gt_psf,
                        predicted_psf=p_psf,
                        corrected_psf=corrected_psf,
                        wavelength=gen.lam_detection,
                        psnr=psnr,
                        maxcounts=maxcounts,
                        y=y_wave,
                        pred=p_wave,
                        save_path=save_path / f'{s}',
                        display=False
                    )


def deconstruct(
        model: Path,
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
            cpu_workers=cpu_workers,
        )
        gen = SyntheticPSF(**psfargs)
        psf, y, psnr, maxcounts = next(gen.generator(debug=True))
        p, std = bootstrap_predict(m, psfgen=gen, inputs=psf, batch_size=1)

        p = Wavefront(p, lam_detection=wavelength)
        logger.info('Prediction')
        pprint(p.zernikes)

        logger.info('GT')
        y = Wavefront(y, lam_detection=wavelength)
        pprint(y.zernikes)

        diff = Wavefront(y - p, lam_detection=wavelength)
        p_psf = gen.single_psf(p)
        gt_psf = gen.single_psf(y)
        corrected_psf = gen.single_psf(diff)

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


def save_metadata(
    filepath: Path,
    wavelength: float,
    psf_type: str,
    x_voxel_size: float,
    y_voxel_size: float,
    z_voxel_size: float,
    n_modes: int,
):
    def add_param(h5file, name, data):
        try:
            if name not in h5file.keys():
                h5file.create_dataset(name, data=data)
            else:
                del h5file[name]
                h5file.create_dataset(name, data=data)

            assert np.allclose(h5file[name].value, data), f"Failed to write {name}"

        except Exception as e:
            logger.error(e)

    file = filepath if str(filepath).endswith('.h5') else list(filepath.rglob('*.h5'))[0]
    with h5py.File(file, 'r+') as file:
        add_param(file, name='n_modes', data=n_modes)
        add_param(file, name='wavelength', data=wavelength)
        add_param(file, name='x_voxel_size', data=x_voxel_size)
        add_param(file, name='y_voxel_size', data=y_voxel_size)
        add_param(file, name='z_voxel_size', data=z_voxel_size)

        if isinstance(psf_type, str) or isinstance(psf_type, Path):
            with h5py.File(psf_type, 'r+') as f:
                add_param(file, name='psf_type', data=f.get('DitheredxzPSFCrossSection')[:, 0])


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

        plt.savefig(f'{modelpath}/kernels_{layer.name}.svg', dpi=300, bbox_inches='tight', pad_inches=.25)


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
        psf_type=psf_type,
        psf_shape=3 * [input_shape[1]],
        distribution='single',
        lam_detection=wavelength,
        amplitude_ranges=amplitude_range,
        x_voxel_size=x_voxel_size,
        y_voxel_size=y_voxel_size,
        z_voxel_size=z_voxel_size,
        batch_size=1,
        snr=psnr,
        cpu_workers=cpu_workers,
    )
    gen = SyntheticPSF(**psfargs)

    if input_shape[0] == 3 or input_shape[0] == 6:
        inputs = gen.single_otf(
            amplitude_range,
            normed=True,
            noise=True,
            na_mask=True,
            ratio=True,
        )
    else:
        inputs = gen.single_psf(amplitude_range, normed=True, noise=True)

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

    if isinstance(psf_type, str) or isinstance(psf_type, Path):
        with h5py.File(psf_type, 'r') as file:
            psf_type = file.get('DitheredxzPSFCrossSection')[:, 0]

    if network == 'opticalnet':
        model = opticalnet.OpticalTransformer(
            name='OpticalTransformer',
            roi=roi,
            patches=patch_size,
            modes=pmodes,
            depth_scalar=depth_scalar,
            width_scalar=width_scalar,
            activation=activation,
            mul=mul,
            no_phase=no_phase
        )

    elif network == 'opticaltransformer':
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
            Adam: A Method for Stochastic Optimization: httpsz://arxiv.org/pdf/1412.6980
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

    metadata = LambdaCallback(
        on_train_end=lambda logs: save_metadata(
            filepath=outdir,
            n_modes=pmodes,
            psf_type=psf_type,
            wavelength=wavelength,
            x_voxel_size=x_voxel_size,
            y_voxel_size=y_voxel_size,
            z_voxel_size=z_voxel_size,
        )
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
            psf_type=psf_type,
            psf_shape=inputs,
            snr=(min_psnr, max_psnr),
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
                metadata,
                earlystopping,
                defibrillator,
                lrscheduler,
            ],
        )
    except tf.errors.ResourceExhaustedError as e:
        logger.error(e)
        sys.exit(1)
