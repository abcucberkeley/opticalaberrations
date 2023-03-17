import matplotlib
matplotlib.use('Agg')

import logging
import multiprocessing as mp
import sys
from typing import Union, Any, List, Generator
from line_profiler_pycharm import profile
import subprocess

import matplotlib.pyplot as plt
plt.set_loglevel('error')

import io
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.special import binom
from skimage.feature import peak_local_max
from scipy.spatial import KDTree
from astropy import convolution

import vis
from preprocessing import resize_with_crop_or_pad
from synthetic import SyntheticPSF
from wavefront import Wavefront


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@profile
def multiprocess(jobs: Union[Generator, List, np.ndarray], func: Any, desc: str = 'Processing', cores: int = -1):
    """ Multiprocess a generic function
    Args:
        func: a python function
        jobs: a list of jobs for function `func`
        desc: description for the progress bar
        cores: number of cores to use

    Returns:
        an array of outputs for every function call
    """

    mp.set_start_method('spawn', force=True)
    jobs = list(jobs)
    if cores == 1 or len(jobs) == 1:
        logs = []
        for j in tqdm(jobs, total=len(jobs), desc=desc):
            logs.append(func(j))
    elif cores == -1 and len(jobs) > 0:
        with mp.Pool(min(mp.cpu_count(), len(jobs))) as p:
            logs = list(tqdm(p.imap(func, jobs), total=len(jobs), desc=desc))
    elif cores > 1 and len(jobs) > 0:
        with mp.Pool(cores) as p:
            logs = list(tqdm(p.imap(func, jobs), total=len(jobs), desc=desc))
    else:
        logging.error('Jobs must be a positive integer')
        return False
    return np.array(logs)


def microns2waves(a, wavelength):
    return a/wavelength


def waves2microns(a, wavelength):
    return a*wavelength


def mae(y: np.array, p: np.array, axis=0) -> np.array:
    error = np.abs(y - p)
    return np.mean(error[np.isfinite(error)], axis=axis)


def mse(y: np.array, p: np.array, axis=0) -> np.array:
    error = (y - p) ** 2
    return np.mean(error[np.isfinite(error)], axis=axis)


def rmse(y: np.array, p: np.array, axis=0) -> np.array:
    error = np.sqrt((y - p)**2)
    return np.mean(error[np.isfinite(error)], axis=axis)


def mape(y: np.array, p: np.array, axis=0) -> np.array:
    error = np.abs(y - p) / np.abs(y)
    return 100 * np.mean(error[np.isfinite(error)], axis=axis)


@profile
def p2v(zernikes, wavelength=.510, na=1.0):
    grid = 100
    r = np.linspace(-1, 1, grid)
    X, Y = np.meshgrid(r, r, indexing='ij')
    rho = np.hypot(X, Y)
    theta = np.arctan2(Y, X)

    Y, X = np.ogrid[:grid, :grid]
    dist_from_center = np.sqrt((X - grid//2) ** 2 + (Y - grid//2) ** 2)
    na_mask = dist_from_center <= (na * grid) / 2

    nm_pairs = set((n, m) for n in range(11) for m in range(-n, n + 1, 2))
    ansi_to_nm = dict(zip(((n * (n + 2) + m) // 2 for n, m in nm_pairs), nm_pairs))

    polynomials = []
    for ansi, a in enumerate(zernikes):
        n, m = ansi_to_nm[ansi]
        m0 = abs(m)

        radial = 0
        for k in range((n - m0) // 2 + 1):
            radial += (-1.) ** k * binom(n - k, k) * binom(n - 2 * k, (n - m0) // 2 - k) * rho ** (n - 2 * k)

        radial *= (rho <= 1.)
        prefac = 1. / np.sqrt((1. + (m == 0)) / (2. * n + 2))

        if (n - m) % 2 == 1:
            polynomials.append(0)
        elif m >= 0:
            polynomials.append(microns2waves(a, wavelength=wavelength) * prefac * radial * np.cos(m0 * theta))
        else:
            polynomials.append(microns2waves(a, wavelength=wavelength) * prefac * radial * np.sin(m0 * theta))

    wavefront = np.sum(np.array(polynomials), axis=0) * na_mask
    return abs(np.nanmax(wavefront) - np.nanmin(wavefront))


@profile
def peak2valley(w, wavelength: float = .510, na: float = 1.0) -> float:
    if not isinstance(w, Wavefront):
        w = Wavefront(w, lam_detection=wavelength)

    wavefront = w.wave(100)
    center = (int(wavefront.shape[0] / 2), int(wavefront.shape[1] / 2))
    Y, X = np.ogrid[:wavefront.shape[0], :wavefront.shape[1]]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= (na * wavefront.shape[0]) / 2
    wavefront *= mask
    return abs(np.nanmax(wavefront) - np.nanmin(wavefront))


def compute_signal_lost(phi, gen, res):
    hashtbl = {}
    w = Wavefront(phi, order='ansi')
    psf = gen.single_psf(w, normed=True, noise=False)
    abr = 0 if np.count_nonzero(phi) == 0 else round(w.peak2valley())

    for k, r in enumerate(res):
        window = resize_with_crop_or_pad(psf, crop_shape=tuple(3*[r]))
        hashtbl[abr][r] = np.sum(window)

    return hashtbl


def compute_error(y_true: pd.DataFrame, y_pred: pd.DataFrame, axis=None) -> pd.DataFrame:
    res = np.abs(y_true - y_pred).mean(axis=axis).to_frame('mae')
    res['mae'] = mae(y_true, y_pred, axis)
    res['mse'] = mse(y_true, y_pred, axis)
    res['mape'] = mape(y_true, y_pred, axis)
    res['rmse'] = rmse(y_true, y_pred, axis)
    return res


def eval(k: tuple, psfargs: dict):
    psf, y, pred, path, psnr, maxcounts = k

    if psf.ndim == 5:
        psf = np.squeeze(psf, axis=0)
        psf = np.squeeze(psf, axis=-1)
    elif psf.ndim == 4:
        psf = np.squeeze(psf, axis=-1)

    diff = y - pred
    y = Wavefront(y)
    pred = Wavefront(pred)
    diff = Wavefront(diff)

    psf_gen = SyntheticPSF(**psfargs)
    p_psf = psf_gen.single_psf(pred)
    gt_psf = psf_gen.single_psf(y)
    corrected_psf = psf_gen.single_psf(diff)

    vis.diagnostic_assessment(
        psf=psf,
        gt_psf=gt_psf,
        corrected_psf=corrected_psf,
        predicted_psf=p_psf,
        psnr=psnr,
        maxcounts=maxcounts,
        y=y,
        pred=pred,
        save_path=path,
        display=False
    )


def plot_to_image(figure):
    """
        Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call.
        https://www.tensorflow.org/tensorboard/image_summaries
    """
    import tensorflow as tf

    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


@profile
def mean_min_distance(sample: np.array, voxel_size: tuple, plot: bool = False):
    beads = peak_local_max(
        sample,
        min_distance=0,
        threshold_rel=0,
        exclude_border=False,
        p_norm=2,
    ).astype(np.float64)

    scaled_peaks = np.zeros_like(beads)
    scaled_peaks[:, 0] = beads[:, 0] * voxel_size[0]
    scaled_peaks[:, 1] = beads[:, 1] * voxel_size[1]
    scaled_peaks[:, 2] = beads[:, 2] * voxel_size[2]

    kd = KDTree(scaled_peaks)
    dists, idx = kd.query(scaled_peaks, k=2, workers=-1)

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False, sharex=False)
        for ax in range(3):
            axes[ax].imshow(
                np.nanmax(sample, axis=ax),
                aspect='auto',
                cmap='gray'
            )

            for p in range(dists.shape[0]):
                if ax == 0:
                    axes[ax].plot(beads[p, 2], beads[p, 1], marker='.', ls='', color=f'C{p}')
                elif ax == 1:
                    axes[ax].plot(beads[p, 2], beads[p, 0], marker='.', ls='', color=f'C{p}')
                else:
                    axes[ax].plot(beads[p, 1], beads[p, 0], marker='.', ls='', color=f'C{p}')

        plt.tight_layout()
        plt.show()

    return np.round(np.mean(dists), 1)


@profile
def fftconvolution(kernel, sample):
    if kernel.shape[0] == 1 or kernel.shape[-1] == 1:
        kernel = np.squeeze(kernel)

    if sample.shape[0] == 1 or sample.shape[-1] == 1:
        sample = np.squeeze(sample)

    conv = convolution.convolve_fft(sample, kernel, allow_huge=True)
    conv /= np.nanmax(conv)
    conv = np.nan_to_num(conv, nan=0, neginf=0, posinf=0)
    conv[conv < 0] = 0
    return conv
