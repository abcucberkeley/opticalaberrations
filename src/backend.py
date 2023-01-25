import matplotlib
matplotlib.use('Agg')

import numexpr
numexpr.set_num_threads(numexpr.detect_number_of_cores())

import logging
import sys
import h5py
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Any, Union
from functools import partial
from line_profiler_pycharm import profile

import pandas as pd
from skimage.restoration import richardson_lucy
from skimage.transform import rotate
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import scipy as sp
from scipy import ndimage

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
from skimage.feature import peak_local_max

import utils
import vis
import data_utils

from synthetic import SyntheticPSF
from wavefront import Wavefront

from tensorflow.keras import Model
from phasenet import PhaseNet

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
def load_metadata(
        model_path: Path,
        psf_shape: Union[tuple, list] = (64, 64, 64),
        psf_type=None,
        n_modes=None,
        z_voxel_size=None,
        **kwargs
):
    """ The model .h5, HDF5 file, is also used to store metadata parameters (wavelength, x_voxel_size, etc) that 
    the model was trained with.  The metadata is read from file and is returned within the returned SyntheticPSF class.

    Args:
        model_path (Path): path to .h5 model, or path to the containing folder.
        psf_shape (Union[tuple, list], optional): dimensions of the SyntheticPSF to return. Defaults to (64, 64, 64).
        psf_type (str, optional): "widefield" or "confocal". Defaults to None which reads the value from the .h5 file.
        n_modes (int, optional): # of Zernike modes to describe abberation. Defaults to None which reads the value from the .h5 file.
        z_voxel_size (float, optional):  Defaults to None which reads the value from the .h5 file.
        **kwargs:  Get passed into SyntheticPSF generator.

    Returns:
        SyntheticPSF class: ideal PSF that the model was trained on.
    """
    # print(f"my suffix = {model_path.suffix}, my model = {model_path}")
    if not model_path.suffix == '.h5':       
        model_path = list(model_path.rglob('*.h5'))[0]  # locate the model if the parent folder path is given

    with h5py.File(model_path, 'r') as file:

        try:
            embedding_option = str(file.get('embedding_option').asstr()[()]).strip("\'").strip('\"')
        except Exception:
            embedding_option = 'principle_planes'

        psfgen = SyntheticPSF(
            psf_type=np.array(file.get('psf_type')[:]) if psf_type is None else psf_type,
            psf_shape=psf_shape,
            n_modes=int(file.get('n_modes')[()]) if n_modes is None else n_modes,
            lam_detection=float(file.get('wavelength')[()]),
            x_voxel_size=float(file.get('x_voxel_size')[()]),
            y_voxel_size=float(file.get('y_voxel_size')[()]),
            z_voxel_size=float(file.get('z_voxel_size')[()]) if z_voxel_size is None else z_voxel_size,
            embedding_option=embedding_option,
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
    no_phase: bool = False,
    padsize: Any = None,
    alpha_val: str = 'abs',
    phi_val: str = 'angle',
    peaks: Any = None,
    remove_interference: bool = True,
    ignore_modes: list = (0, 1, 2, 4),
    threshold: float = 0.05,
    freq_strength_threshold: float = .01,
    verbose: bool = True,
    plot: Any = None,
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
        no_phase: ignore/drop the phase component of the FFT
        padsize: pad the input to the desired size for the FFT
        alpha_val: use absolute values of the FFT `abs`, or the real portion `real`.
        phi_val: use the FFT phase in unwrapped radians `angle`, or the imaginary portion `imag`.
        remove_interference: a toggle to normalize out the interference pattern from the OTF
        peaks: masked array of the peaks of interest to compute the interference pattern
        threshold: set predictions below threshold to zero (wavelength)
        freq_strength_threshold: threshold to filter out frequencies below given threshold (percentage to peak)
        desc: test to display for the progressbar
        verbose: a toggle for progress bar
        plot: optional toggle to visualize embeddings

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
        generate_fourier_embeddings = partial(
            psfgen.embedding,
            plot=plot,
            padsize=padsize,
            no_phase=no_phase,
            alpha_val=alpha_val,
            phi_val=phi_val,
            peaks=peaks,
            remove_interference=remove_interference,
            embedding_option=psfgen.embedding_option,
            freq_strength_threshold=freq_strength_threshold,
        )
        model_inputs = utils.multiprocess(generate_fourier_embeddings, inputs, cores=-1)
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
def eval_rotation(
    init_preds: np.ndarray,
    rotations: np.ndarray,
    psfgen: SyntheticPSF,
    threshold: float = 0.01,
    plot: Any = None,
):
    """
        We can think of the mode and its twin as the X and Y basis, and the abberation being a
        vector on this coordinate system. A problem occurs when the ground truth vector lies near
        where one of the vectors flips sign (and the model was trained to only respond with positive
        values for that vector).  Either a discontinuity or a degeneracy occurs in this area leading to
        unreliable predictions.  The solution is to digitally rotate the input embeddings over a range
        of angles.  Converting from cartesian X Y to polar coordinates (rho, phi), the model predictions
        should map phi as a linear waveform (if all were perfect), a triangle waveform (if degenergancies
        exist because the model was trained to only return positive values), or a sawtooth waveform (if
        a discontinuity exists because the model was trained to only return positive values for the first
        mode in the twin pair).  These waveforms (with known period:given by the m value of the mode,
        known amplitude) can be curve-fit to determine the actual phi. Given rho and phi, we can
        return to cartesian coordinates, which give the amplitudes for mode & twin.

        Args:
            init_preds: predictions for each rotation angle
            rotations: list of rotations applied to the embeddings
            psfgen: Synthetic PSF object
            threshold: set predictions below threshold to zero (wavelength)
            plot: optional toggle to visualize embeddings
    """

    plt.style.use("default")
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.autolimit_mode': 'round_numbers'
    })
    def cart2pol(x, y):
        """Convert cartesian (x, y) to polar (rho, phi)
        """
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return (rho, phi)

    def pol2cart(rho, phi):
        """Convert polar (rho, phi) to cartesian (x, y)
        """
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return (x, y)

    def linear_fit_fixed_slope(x, y, m):
        return np.mean(y - m*x)

    threshold = utils.waves2microns(threshold, wavelength=psfgen.lam_detection)

    preds = np.zeros(psfgen.n_modes)
    wavefront = Wavefront(preds)

    if plot is not None:
        fig = plt.figure(figsize=(15, 20))
        plt.subplots_adjust(hspace=0.1)
        gs = fig.add_gridspec(len(wavefront.twins.keys()), 2)

    for row, (mode, twin) in enumerate(wavefront.twins.items()):
        if twin is not None:
            magnitude, phiangle = cart2pol(init_preds[:, mode.index_ansi], init_preds[:, twin.index_ansi])

            if np.mean(magnitude) > threshold:
                # the Zernike modes have m periods per 2pi. xdata is now the Twin angle
                xdata = rotations * np.abs(mode.m)
                rhos, ydata = cart2pol(init_preds[:, mode.index_ansi], init_preds[:, twin.index_ansi])
                ydata = np.degrees(ydata)
                rho = rhos[np.argmin(np.abs(ydata))]

                # exclude points near discontinuities (-90, +90, 450,..) based upon fit
                data_mask = np.ones(xdata.shape[0], dtype=bool)
                data_mask[(init_preds[:, mode.index_ansi] < rho/5) * (rho > threshold)] = 0.
                data_mask[rhos < rho/2] = 0.    # exclude if rho is unusually small (which can lead to small, but dominant primary mode near discontinuity)
                xdata = xdata[data_mask]
                ydata = ydata[data_mask] 
                offset = ydata[0]
                ydata = np.unwrap(ydata, period=180)
                ydata = ((ydata - offset - xdata) + 90) % 180 - 90 + offset + xdata


                m = 1
                b = linear_fit_fixed_slope(xdata, ydata, m)  # refit without bad data points
                fit = m * xdata + b

                mse = np.mean(np.square(fit-ydata))
                if mse > 700:
                    # reject if it doesn't show rotation that matches within +/- 26deg, equivalent to +/- 0.005 um on a 0.01 um mode.
                    rho = 0

                twin_angle = b   # evaluate the curve fit when there is no digital rotation.
                preds[mode.index_ansi], preds[twin.index_ansi] = pol2cart(rho, np.radians(twin_angle))

                if plot is not None:
                    ax = fig.add_subplot(gs[row, 0])
                    fit_ax = fig.add_subplot(gs[row, 1])

                    ax.plot(rotations, init_preds[:, mode.index_ansi], label=f"m{mode.index_ansi}")
                    ax.plot(rotations, init_preds[:, twin.index_ansi], '--', label=f"m{twin.index_ansi}")

                    ax.set_xlim(0, 360)
                    ax.set_xticks(range(0, 405, 45))
                    ax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)
                    ax.set_ylim(-np.max(rhos), np.max(rhos))
                    ax.legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(.5, 1.15))
                    ax.set_ylabel('Amplitude ($\mu$m RMS)')
                    ax.set_xlabel('Digital rotation (deg)')

                    title_color = 'g' if rho > 0 else 'r'
                    fit_ax.plot(xdata, m * xdata + b, color=title_color, lw='.75')
                    fit_ax.scatter(xdata, ydata, s=2, color='grey')
                    
                    fit_ax.set_title(
                        f'm{mode.index_ansi}={preds[mode.index_ansi]:.3f}, m{twin.index_ansi}={preds[twin.index_ansi]:.3f}'
                        f' [$b$={twin_angle:.1f}$^\circ$  $\\rho$={rho:.3f} $\mu$RMS   MSE={mse:.0f}]',
                        color=title_color
                    )

                    fit_ax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)
                    fit_ax.set_ylabel('Predicted Twin angle (deg)')
                    fit_ax.set_xlabel('Digitially rotated Twin angle (deg)')
                    fit_ax.set_xticks(range(0, int(np.max(xdata)), 90))
                    fit_ax.set_yticks(np.insert(np.arange(-90, np.max(ydata), 180), 0, 0))
                    fit_ax.set_xlim(0, 360 * np.abs(mode.m))
     
                    ax.scatter(rotations[~data_mask], rhos[~data_mask], s=1.5, color='pink', zorder=3)
                    ax.scatter(rotations[data_mask], rhos[data_mask], s=1.5, color='black', zorder=3)
            else:
                preds[mode.index_ansi] = 0
                preds[twin.index_ansi] = 0

                if plot is not None:
                    ax = fig.add_subplot(gs[row, 0])
                    ax.plot(rotations, init_preds[:, mode.index_ansi], label=f"m{mode.index_ansi}")
                    ax.plot(rotations, init_preds[:, twin.index_ansi], '--', label=f"m{twin.index_ansi}")

                    ax.set_xlim(0, 360)
                    ax.set_xticks(range(0, 405, 45))
                    ax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)
                    ax.set_ylim(np.min(init_preds), np.max(init_preds))
                    ax.legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(.5, 1.15))
                    ax.set_ylabel('Amplitude ($\mu$ RMS)')
                    ax.set_xlabel('Digital rotation (deg)')

        else:
            # mode has m=0 (spherical,...), or twin isn't within the 55 modes.
            rho = np.median(init_preds[:, mode.index_ansi])
            rho *= np.abs(rho) > threshold  # make sure it's above threshold, or else set to zero.
            preds[mode.index_ansi] = rho

            if plot is not None:
                ax = fig.add_subplot(gs[row, 0])
                ax.plot(rotations, init_preds[:, mode.index_ansi], label=f"m{mode.index_ansi}={rho:.3f}")

                ax.set_xlim(0, 360)
                ax.set_xticks(range(0, 405, 45))
                ax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)
                ax.set_ylim(np.min(init_preds), np.max(init_preds))
                ax.legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(.5, 1.15))
                ax.set_ylabel('Amplitude ($\mu$ RMS)')
                ax.set_xlabel('Digital rotation (deg)')

    if plot is not None:
        plt.tight_layout()
        plt.savefig(f'{plot}_rotations.svg', dpi=300, bbox_inches='tight', pad_inches=.25)

    return preds


@profile
def predict_rotation(
    model: tf.keras.Model,
    inputs: np.array,
    psfgen: SyntheticPSF,
    batch_size: int = 1,
    n_samples: int = 10,
    threshold: float = 0.01,
    freq_strength_threshold: float = .01,
    no_phase: bool = False,
    padsize: Any = None,
    alpha_val: str = 'abs',
    phi_val: str = 'angle',
    peaks: Any = None,
    remove_interference: bool = True,
    plot: Any = None,
    desc: str = 'Predict-rotations',
    rotations: np.ndarray = np.arange(0, 360+1, 1).astype(int)
):
    """
    Predict the fraction of the amplitude to be assigned each pair of modes (ie. mode & twin).

    Args:
        model: pre-trained keras model. Model must be trained with XY embeddings only so that we can rotate them.
        inputs: encoded tokens to be processed. (e.g. input images)
        psfgen: Synthetic PSF object
        n_samples: number of predictions of average
        batch_size: number of samples per batch
        no_phase: ignore/drop the phase component of the FFT
        padsize: pad the input to the desired size for the FFT
        alpha_val: use absolute values of the FFT `abs`, or the real portion `real`.
        phi_val: use the FFT phase in unwrapped radians `angle`, or the imaginary portion `imag`.
        remove_interference: a toggle to normalize out the interference pattern from the OTF
        peaks: masked array of the peaks of interest to compute the interference pattern
        threshold: set predictions below threshold to zero (wavelength)
        freq_strength_threshold: threshold to filter out frequencies below given threshold (percentage to peak)
        threshold: set predictions below threshold to zero (wavelength)
        freq_strength_threshold: threshold to filter out frequencies below given threshold (percentage to peak)
        desc: test to display for the progressbar
        plot: optional toggle to visualize embeddings
    """
    generate_fourier_embeddings = partial(
        psfgen.embedding,
        plot=plot,
        padsize=padsize,
        no_phase=no_phase,
        alpha_val=alpha_val,
        phi_val=phi_val,
        peaks=peaks,
        remove_interference=remove_interference,
        embedding_option=psfgen.embedding_option,
        freq_strength_threshold=freq_strength_threshold,
    )
    embeddings = np.array(utils.multiprocess(generate_fourier_embeddings, inputs, cores=-1))

    rotated_embs = []
    for emb in embeddings:
        for angle in rotations:
            rotated_embs.append(np.array([rotate(plane, angle=angle) for plane in emb]))
    rotated_embs = np.array(rotated_embs)

    init_preds, stdev = bootstrap_predict(
        model,
        rotated_embs,
        psfgen=psfgen,
        batch_size=batch_size,
        n_samples=n_samples,
        no_phase=True,
        threshold=0.,
        plot=plot,
        desc=desc
    )

    eval_mode_rotations = partial(
        eval_rotation,
        rotations=rotations,
        psfgen=psfgen,
        threshold=threshold,
        plot=plot,
    )

    init_preds = np.stack(np.split(init_preds, inputs.shape[0]), axis=0)
    return np.array(utils.multiprocess(eval_mode_rotations, init_preds, cores=-1))


@profile
def predict_sign(
    init_preds: np.ndarray,
    followup_preds: np.ndarray,
    sign_threshold: float = .99,
    plot: Any = None,
):
    def pct_change(cur, prev):
        cur = np.abs(cur)
        prev = np.abs(prev)

        if np.array_equal(cur, prev):
            return np.zeros_like(prev)
        pct = ((cur - prev) / (prev+1e-6)) * 100.0
        pct[pct > 100] = 100
        pct[pct < -100] = -100
        return pct

    preds = init_preds.copy()
    pchange = pct_change(followup_preds, init_preds)

    # flip signs and make any necessary adjustments to the amplitudes based on the followup predictions
    # adj = pchange.copy()
    # adj[np.where(pchange > 0)] = -200
    # adj[np.where(pchange < 0)] += 100
    # preds += preds * (adj/100)

    # threshold-based sign prediction
    # threshold = sign_threshold * init_preds
    # flips = np.stack(np.where(followup_preds > threshold), axis=0)

    # if len(np.squeeze(preds).shape) == 1:
    #     preds[flips[0]] *= -1
    # else:
    #     preds[flips[0], flips[1]] *= -1

    # flip sign only if amplitudes increased on the followup predictions
    preds[pchange > 0.] *= -1

    if plot is not None:
        if len(np.squeeze(preds).shape) == 1:
            init_preds_wave = np.squeeze(init_preds)
            init_preds_wave_error = np.zeros_like(init_preds_wave)

            followup_preds_wave = np.squeeze(followup_preds)
            followup_preds_wave_error = np.zeros_like(followup_preds_wave)

            preds_wave = np.squeeze(preds)
            preds_error = np.zeros_like(preds_wave)

            percent_changes = np.squeeze(pchange)
            percent_changes_error = np.zeros_like(percent_changes)
        else:
            init_preds_wave = np.mean(init_preds, axis=0)
            init_preds_wave_error = np.std(init_preds, axis=0)

            followup_preds_wave = np.mean(followup_preds, axis=0)
            followup_preds_wave_error = np.std(followup_preds, axis=0)

            preds_wave = np.mean(preds, axis=0)
            preds_error = np.std(preds, axis=0)

            percent_changes = np.mean(pchange, axis=0)
            percent_changes_error = np.std(pchange, axis=0)

        plt.style.use("default")
        vis.plot_sign_correction(
            init_preds_wave,
            init_preds_wave_error,
            followup_preds_wave,
            followup_preds_wave_error,
            preds_wave,
            preds_error,
            percent_changes,
            percent_changes_error,
            savepath=f'{plot}_sign_correction'
        )

        plt.style.use("dark_background")
        vis.plot_sign_correction(
            init_preds_wave,
            init_preds_wave_error,
            followup_preds_wave,
            followup_preds_wave_error,
            preds_wave,
            preds_error,
            percent_changes,
            percent_changes_error,
            savepath=f'{plot}_sign_correction_db'
        )

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
        plot=plot
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
def evaluate(
    model: tf.keras.Model,
    inputs: np.array,
    gen: SyntheticPSF,
    ys: np.array,
    psnr: int,
    batch_size: int,
    reference: Any = None,
    plot: Any = None,
    threshold: float = 0.01,
    eval_sign: str = 'positive_only',
):
    if isinstance(inputs, tf.Tensor):
        inputs = inputs.numpy()

    if isinstance(ys, tf.Tensor):
        ys = ys.numpy()

    if len(ys.shape) == 1:
        ys = ys[np.newaxis, :]

    if eval_sign == 'positive_only':
        ys = np.abs(ys)

        preds, stdev = bootstrap_predict(
            model,
            inputs,
            psfgen=gen,
            batch_size=batch_size,
            n_samples=1,
            no_phase=True,
            threshold=threshold,
            plot=plot
        )
        if len(preds.shape) > 1:
            preds = np.abs(preds)[:, :ys.shape[-1]]
        else:
            preds = np.abs(preds)[np.newaxis, :ys.shape[-1]]

    elif eval_sign == 'dual_stage':

        init_preds = predict_rotation(
            model=model,
            inputs=inputs,
            psfgen=gen,
            no_phase=True,
            batch_size=batch_size,
            n_samples=1,
            threshold=threshold,
            plot=plot
        )

        if len(init_preds.shape) > 1:
            init_preds = init_preds[:, :ys.shape[-1]]
        else:
            init_preds = init_preds[np.newaxis, :ys.shape[-1]]

        threshold = utils.waves2microns(threshold, wavelength=gen.lam_detection)
        ps = init_preds.copy()
        ps[ps > .1] = .1
        ps[ps < -.1] = -.1
        res = ys - ps
        followup_inputs = np.zeros_like(inputs)

        if reference is not None:
            for i in range(inputs.shape[0]):
                wavefront = Wavefront(
                    res[i],
                    modes=gen.n_modes,
                    rotate=False,
                    lam_detection=gen.lam_detection,
                )
                psf = gen.single_psf(
                    wavefront,
                    normed=True,
                    noise=False,
                    meta=False
                )

                fi = utils.fftconvolution(kernel=psf, sample=reference)
                fi *= psnr ** 2
                rand_noise = gen._random_noise(image=fi, mean=0, sigma=gen.sigma_background_noise)
                fi += rand_noise
                fi /= np.max(fi)
                followup_inputs[i] = fi[..., np.newaxis]

        # plt.style.use("default")
        # vis.plot_sign_eval(
        #     inputs=inputs,
        #     followup_inputs=followup_inputs,
        #     savepath=f'{plot}_sign_eval'
        # )
        #
        # plt.style.use("dark_background")
        # vis.plot_sign_eval(
        #     inputs=inputs,
        #     followup_inputs=followup_inputs,
        #     savepath=f'{plot}_sign_eval_db'
        # )

        followup_preds = predict_rotation(
            model=model,
            inputs=followup_inputs,
            psfgen=gen,
            no_phase=True,
            batch_size=batch_size,
            n_samples=1,
            threshold=threshold,
            plot=f"{plot}_followup"
        )

        if len(followup_preds.shape) > 1:
            followup_preds = followup_preds[:, :ys.shape[-1]]
        else:
            followup_preds = followup_preds[np.newaxis, :ys.shape[-1]]

        preds, pchanges = predict_sign(
            init_preds=init_preds,
            followup_preds=followup_preds,
            plot=plot
        )

    else:
        preds, stdev = bootstrap_predict(
            model,
            inputs,
            psfgen=gen,
            batch_size=batch_size,
            n_samples=1,
            no_phase=False,
            threshold=threshold,
            plot=plot
        )

        if len(preds.shape) > 1:
            preds = preds[:, :ys.shape[-1]]
        else:
            preds = preds[np.newaxis, :ys.shape[-1]]

    residuals = ys - preds
    return residuals, ys, preds


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
    embedding_option: str = 'principle_planes'
):
    def add_param(h5file, name, data):
        try:
            if name not in h5file.keys():
                h5file.create_dataset(name, data=data)
            else:
                del h5file[name]
                h5file.create_dataset(name, data=data)

            if isinstance(data, str):
                assert h5file.get(name).asstr()[()] == data, f"Failed to write {name}"
            else:
                assert np.allclose(h5file.get(name)[()], data), f"Failed to write {name}"

            logger.info(f"`{name}`: {h5file.get(name)[()]}")

        except Exception as e:
            logger.error(e)

    file = filepath if str(filepath).endswith('.h5') else list(filepath.rglob('*.h5'))[0]
    with h5py.File(file, 'r+') as file:
        add_param(file, name='n_modes', data=n_modes)
        add_param(file, name='wavelength', data=wavelength)
        add_param(file, name='x_voxel_size', data=x_voxel_size)
        add_param(file, name='y_voxel_size', data=y_voxel_size)
        add_param(file, name='z_voxel_size', data=z_voxel_size)
        add_param(file, name='embedding_option', data=embedding_option)

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
        embedding: str,
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
            name='OpticalNet',
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

    tensorboard = TensorBoardCallback(
        log_dir=outdir,
        histogram_freq=1,
        profile_batch=100000000
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
            embedding_option=embedding,
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
            modes=modes,
            distribution=distribution,
            embedding=embedding,
            samplelimit=samplelimit,
            max_amplitude=max_amplitude,
            no_phase=no_phase,
            snr_range=(min_psnr, max_psnr)
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
                lrscheduler,
            ],
        )
    except tf.errors.ResourceExhaustedError as e:
        logger.error(e)
        sys.exit(1)
