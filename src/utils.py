import logging
import multiprocessing as mp
import sys
from functools import partial
from typing import Any, List

import io
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

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


def multiprocess(func: Any, jobs: List, desc: str = 'Processing', cores: int = -1):
    """ Multiprocess a generic function
    Args:
        func: a python function
        jobs: a list of jobs for function `func`
        desc: description for the progress bar
        cores: number of cores to use

    Returns:
        an array of outputs for every function call
    """
    jobs = list(jobs)
    if cores == 1:
        logs = []
        for j in tqdm(jobs, total=len(jobs), desc=desc):
            logs.append(func(j))
    elif cores == -1:
        with mp.Pool(mp.cpu_count()) as p:
            logs = list(tqdm(p.imap(func, jobs), total=len(jobs), desc=desc))
    elif cores > 1:
        with mp.Pool(cores) as p:
            logs = list(tqdm(p.imap(func, jobs), total=len(jobs), desc=desc))
    else:
        logging.error('Jobs must be a positive integer')
        return False
    return logs


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


def peak_aberration(w, na: float = 1.0) -> float:
    w = Wavefront(w).wave(100)
    radius = (na * w.shape[0]) / 2

    center = (int(w.shape[0] / 2), int(w.shape[1] / 2))
    Y, X = np.ogrid[:w.shape[0], :w.shape[1]]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    phi = w * mask

    mn = np.nanquantile(phi, .05)
    mx = np.nanquantile(phi, .95)
    return abs(mx-mn)


def peak2peak(y: np.array, na: float = 1.0) -> np.array:
    p2p = partial(peak_aberration, na=na)
    return np.array(multiprocess(p2p, list(y), desc='Compute peak2peak aberrations'))


def peak2peak_residuals(y: np.array, p: np.array, na: float = 1.0) -> np.array:
    p2p = partial(peak_aberration, na=na)
    error = np.abs(p2p(y) - p2p(p))
    return error


def microns2waves(phi, wavelength):
    return phi * (2 * np.pi / wavelength)


def waves2microns(phi, wavelength):
    return phi / (2 * np.pi / wavelength)


def compute_signal_lost(phi, gen, res):
    hashtbl = {}
    w = Wavefront(phi, order='ansi')
    psf = gen.single_psf(w, zplanes=0, normed=True, noise=False)
    abr = 0 if np.count_nonzero(phi) == 0 else round(peak_aberration(phi))

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
    psf, y, pred, path, psnr, zplanes, maxcounts = k

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
    p_psf = psf_gen.single_psf(pred, zplanes=zplanes)
    gt_psf = psf_gen.single_psf(y, zplanes=zplanes)
    corrected_psf = psf_gen.single_psf(diff, zplanes=zplanes)

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
