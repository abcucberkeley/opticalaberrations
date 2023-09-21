import matplotlib
matplotlib.use('Agg')

import io
import sys
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from scipy.special import binom
from skimage.feature import peak_local_max
from scipy.spatial import KDTree
from astropy import convolution
import multiprocessing as mp
from line_profiler_pycharm import profile
from typing import Any, List, Union, Optional, Generator

try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter
except ImportError as e:
    logging.warning(f"Cupy not supported on your system: {e}")

from preprocessing import resize_with_crop_or_pad
from wavefront import Wavefront

import matplotlib.pyplot as plt
plt.set_loglevel('error')

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@profile
def multiprocess(
    jobs: Union[Generator, List, np.ndarray],
    func: Any,
    desc: str = 'Processing',
    cores: int = -1,
    unit: str = 'it',
    pool: Optional[mp.Pool] = None,
):
    """ Multiprocess a generic function
    Args:
        func: a python function
        jobs: a list of jobs for function `func`
        desc: description for the progress bar
        cores: number of cores to use

    Returns:
        an array of outputs for every function call
    """

    cores = cores if mp.current_process().name == 'MainProcess' else 1
    mp.set_start_method('spawn', force=True)
    jobs = list(jobs)

    if cores == 1 or len(jobs) == 1:
        logs = []
        for j in tqdm(
                jobs,
                total=len(jobs),
                desc=desc,
                bar_format='{l_bar}{bar}{r_bar} {elapsed_s:.1f}s elapsed',
                unit=unit,
                file=sys.stdout,
        ):
            logs.append(func(j))
    elif cores == -1 and len(jobs) > 0:
        with pool if pool is not None else mp.Pool(min(mp.cpu_count(), len(jobs))) as p:
            logs = list(tqdm(
                p.imap(func, jobs),
                total=len(jobs),
                desc=desc,
                bar_format='{l_bar}{bar}{r_bar} {elapsed_s:.1f}s elapsed',
                unit=unit,
                file=sys.stdout,
            ))
    elif cores > 1 and len(jobs) > 0:
        with pool if pool is not None else mp.Pool(cores) as p:
            logs = list(tqdm(
                p.imap(func, jobs),
                total=len(jobs),
                desc=desc,
                bar_format='{l_bar}{bar}{r_bar} {elapsed_s:.1f}s elapsed',
                unit=unit,
                file=sys.stdout,
            ))
    else:
        logging.error('Jobs must be a positive integer')
        return False

    return np.array(logs)


def photons2electrons(image, quantum_efficiency: float = .82):
    return image * quantum_efficiency


def electrons2photons(image, quantum_efficiency: float = .82):
    return image / quantum_efficiency


def electrons2counts(image, electrons_per_count: float = .22):
    return image / electrons_per_count


def counts2electrons(image, electrons_per_count: float = .22):
    return image * electrons_per_count


def randuniform(var):
    """
    Returns a random number (uniform chance) in the range provided by var. If var is a scalar, var is simply returned.

    Args:
        var : (as scalar) Returned as is.
        var : (as list) Range to provide a random number

    Returns:
        _type_: ndarray or scalar. Random sample from the range provided.

    """
    var = (var, var) if np.isscalar(var) else var

    # star unpacks a list, so that var's values become the separate arguments here
    return np.random.uniform(*var)


def normal_noise(mean: float, sigma: float, size: tuple) -> np.array:
    mean = randuniform(mean)
    sigma = randuniform(sigma)
    return np.random.normal(loc=mean, scale=sigma, size=size).astype(np.float32)


def poisson_noise(image: np.ndarray) -> np.array:
    image = np.nan_to_num(image, nan=0)
    return np.random.poisson(lam=image).astype(np.float32) - image


def add_noise(
    image: np.ndarray,
    mean_background_offset: int = 100,
    sigma_background_noise: int = 40,
    quantum_efficiency: float = .82,
    electrons_per_count: float = .22,
):
    """

    Args:
        image: noise-free image in incident photons
        mean_background_offset: camera background offset
        sigma_background_noise: read noise from the camera
        quantum_efficiency: quantum efficiency of the camera
        electrons_per_count: conversion factor to go from electrons to counts

    Returns:
        noisy image in counts
    """
    image = photons2electrons(image, quantum_efficiency=quantum_efficiency)
    sigma_background_noise *= electrons_per_count  # electrons;  40 counts = 40 * .22 electrons per count
    dark_read_noise = normal_noise(mean=0, sigma=sigma_background_noise, size=image.shape)  # dark image in electrons
    shot_noise = poisson_noise(image)   # shot noise in electrons

    image += shot_noise + dark_read_noise
    image = electrons2counts(image, electrons_per_count=electrons_per_count)

    image += mean_background_offset    # add camera offset (camera offset in counts)
    image[image < 0] = 0
    return image


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
    psf = gen.single_psf(w, normed=True)
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
    ).astype(np.float32)

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

    conv = convolution.convolve_fft(
        sample,
        kernel,
        allow_huge=True,
        normalize_kernel=False,
        nan_treatment='fill',
        fill_value=0
    ).astype(sample.dtype)   # otherwise returns as float64
    conv[conv < 0] = 0  # clip negative small values
    return conv


def fft_decon(kernel, sample, iters):

    for k in range(kernel.ndim):
        kernel = np.roll(kernel, kernel.shape[k] // 2, axis=k)

    kernel = cp.array(kernel)
    sample = cp.array(sample)
    deconv = cp.array(sample)

    kernel = cp.fft.rfftn(kernel)

    for _ in range(iters):
        conv = cp.fft.irfftn(cp.fft.rfftn(deconv) * kernel)
        relative_blur = sample / conv
        deconv *= cp.fft.irfftn((cp.fft.rfftn(relative_blur).conj() * kernel).conj())

    return cp.asnumpy(deconv)


@profile
def load_dm(dm_state: Any) -> np.ndarray:
    if isinstance(dm_state, np.ndarray) or isinstance(dm_state, list):
        assert len(dm_state) == 69
    elif dm_state is None or str(dm_state) == 'None':
        dm_state = np.zeros(69, dtype=np.float64)
    else:
        dm_state = pd.read_csv(dm_state, header=None, dtype=np.float64).values[:, 0]
    return dm_state


@profile
def zernikies_to_actuators(
        coefficients: np.array,
        dm_calibration: Path,
        dm_state: Optional[Union[Path, str, np.array]] = None,
        scalar: float = 1
) -> np.ndarray:
    dm_state = load_dm(dm_state)
    dm_calibration = pd.read_csv(dm_calibration, header=None).values

    if dm_calibration.shape[-1] > coefficients.size:
        # if we have <55 coefficients, crop the calibration matrix columns
        dm_calibration = dm_calibration[:, :coefficients.size]
    else:
        # if we have >55 coefficients, crop the coefficients array
        coefficients = coefficients[:dm_calibration.shape[-1]]

    offset = np.dot(dm_calibration, coefficients)

    if dm_state is None:
        return - (offset * scalar)
    else:
        return dm_state - (offset * scalar)


@profile
def percentile_filter(data: np.ndarray, min_pct: int = 5, max_pct: int = 95) -> np.ndarray:
    minval, maxval = np.percentile(data, [min_pct, max_pct])
    return (data < minval) | (data > maxval)


def create_multiindex_tile_dataframe(
        path,
        wavelength: float = .510,
        return_wavefronts: bool = False,
        describe: bool = False,
):
    predictions = pd.read_csv(
        path,
        index_col=0,
        header=0,
        usecols=lambda col: col == 'ansi' or col.startswith('z')
    ).T
    # processes "z0-y0-x0" to z, y, x multindex (strip -, then letter, convert to int)
    predictions.index = pd.MultiIndex.from_tuples(predictions.index.str.split('-').to_list())
    predictions.index = pd.MultiIndex.from_arrays([
        predictions.index.get_level_values(0).str.lstrip('z').astype(np.int),
        predictions.index.get_level_values(1).str.lstrip('y').astype(np.int),
        predictions.index.get_level_values(2).str.lstrip('x').astype(np.int),
    ], names=('z', 'y', 'x'))

    wavefronts = {}
    predictions['p2v'] = np.nan
    for index, zernikes in predictions.iterrows():
        wavefronts[index] = Wavefront(
            np.nan_to_num(zernikes.values[:-1], nan=0),
            lam_detection=wavelength,
        )
        predictions.loc[index, 'p2v'] = wavefronts[index].peak2valley(na=1)

    if describe:
        logger.info(
            f'stats\npredictions dataframe\n'
            f'{predictions.describe(percentiles=[.01, .05, .1, .15, .2, .8, .85, .9, .95, .99])}\n'
        )

    if return_wavefronts:
        return predictions, wavefronts
    else:
        return predictions


def get_tile_confidence(
    predictions: pd.DataFrame,
    stdevs: pd.DataFrame,
    prediction_threshold: float = 0.25,
    ignore_tile: Optional[list] = None,
    ignore_modes: Optional[list] = [0, 1, 2, 4],
    verbose: bool = False,
):

    if ignore_tile is not None:
        for cc in ignore_tile:
            z, y, x = [int(s) for s in cc if s.isdigit()]
            predictions.loc[(z, y, x)] = np.nan
            stdevs.loc[(z, y, x)] = np.nan

    all_zeros = predictions == 0  # will label tiles that are any mix of (confident zero and unconfident).

    # filter out unconfident predictions (std deviation is too large)
    where_unconfident = stdevs == 0
    where_unconfident[ignore_modes] = False

    # filter out small predictions from KMeans cluster analysis, but keep these as an additional group
    # note: p2v threshold is computed for each tile
    unconfident_tiles = where_unconfident.copy()
    unconfident_tiles[ignore_modes] = True  # ignore these modes during agg
    all_zeros_tiles = all_zeros.agg('all', axis=1)  # 1D (one value for each tile)
    unconfident_tiles = unconfident_tiles.agg('all', axis=1)  # 1D (one value for each tile)
    zero_confident_tiles = predictions['p2v'] <= prediction_threshold  # 1D (one value for each tile)
    zero_confident_tiles = zero_confident_tiles * ~unconfident_tiles  # don't mark zero_confident if tile is unconfident

    if verbose:
        for z in predictions.index.levels[0].values:
            logger.info(
                f'Number of confident zero tiles {zero_confident_tiles.loc[z].sum():4}'
                f' out of {zero_confident_tiles.loc[z].count()} on z slab {z}'
            )
            logger.info(
                f'Number of unconfident tiles    {unconfident_tiles.loc[z].sum():4}'
                f' out of {unconfident_tiles.loc[z].count()} on z slab {z}'
            )
            logger.info(
                f'Number of all zeros tiles      {all_zeros_tiles.loc[z].sum():4}'
                f' out of {all_zeros_tiles.loc[z].count()} on z slab {z}'
            )
            logger.info(
                f'Number of non-zero tiles       {(~(unconfident_tiles.loc[z] | zero_confident_tiles.loc[z] | all_zeros_tiles.loc[z])).sum():4}'
                f' out of {all_zeros_tiles.loc[z].count()} on z slab {z}'
            )

    return unconfident_tiles, zero_confident_tiles, all_zeros_tiles


def convert_to_windows_file_string(f):
    return str(f).replace('/', '\\').replace("\\clusterfs\\nvme\\", "V:\\")
