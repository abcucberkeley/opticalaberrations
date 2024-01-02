import matplotlib
matplotlib.use('Agg')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings("ignore")

import re
import sys

from functools import partial
import ujson

import matplotlib.pyplot as plt
plt.set_loglevel('error')

import itertools
from pathlib import Path
import tensorflow as tf
from typing import Any, Union, Optional
import numpy as np

import pandas as pd
import seaborn as sns
from tifffile import imread, imwrite, TiffFile
from line_profiler_pycharm import profile
from tqdm import tqdm

import multiprocessing as mp
from sklearn.cluster import KMeans
from skimage.transform import resize
from sklearn.metrics import silhouette_score
from joblib import Parallel, delayed
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import shift, generate_binary_structure, binary_dilation
from scipy.signal import correlate
from scipy.optimize import minimize_scalar, minimize

import utils
import vis
import backend
from utils import round_to_even, round_to_odd

from synthetic import SyntheticPSF
from wavefront import Wavefront
from preloaded import Preloadedmodelclass
from embeddings import remove_interference_pattern
from preprocessing import prep_sample, optimal_rolling_strides, find_roi, get_tiles, resize_with_crop_or_pad

import logging
logger = logging.getLogger('')

try:
    import cupy as cp
except ImportError as e:
    logging.warning(f"Cupy not supported on your system: {e}")

from pycudadecon import decon as cuda_decon


@profile
def reloadmodel_if_needed(
    modelpath: Path,
    preloaded: Optional[Preloadedmodelclass] = None,
    ideal_empirical_psf: Union[Path, np.ndarray] = None,
    ideal_empirical_psf_voxel_size: Any = None,
    n_modes: Optional[int] = None,
    psf_type: Optional[Union[Path, str]] = None
):
    if preloaded is None:
        logger.info("Loading new model, because model didn't exist")
        preloaded = Preloadedmodelclass(
            modelpath,
            ideal_empirical_psf,
            ideal_empirical_psf_voxel_size,
            n_modes=n_modes,
            psf_type=psf_type,
        )

    if ideal_empirical_psf is None and preloaded.ideal_empirical_psf is not None:
        logger.info("Loading new model, because ideal_empirical_psf has been removed")
        preloaded = Preloadedmodelclass(
            modelpath,
            n_modes=n_modes,
            psf_type=psf_type,
        )

    elif preloaded.ideal_empirical_psf != ideal_empirical_psf:
        logger.info(
            f"Updating ideal psf with empirical, "
            f"because {chr(10)} {preloaded.ideal_empirical_psf} "
            f"of type {type(preloaded.ideal_empirical_psf)} "
            f"has been changed to {chr(10)} {ideal_empirical_psf} of type {type(ideal_empirical_psf)}"
        )
        if isinstance(ideal_empirical_psf, np.ndarray):
            # assume PSF has been pre-processed already
            ideal_empirical_preprocessed_psf = ideal_empirical_psf
        else:
            with TiffFile(ideal_empirical_psf) as tif:
                ideal_empirical_preprocessed_psf = prep_sample(
                    np.squeeze(tif.asarray()),
                    model_fov=preloaded.modelpsfgen.psf_fov,
                    sample_voxel_size=ideal_empirical_psf_voxel_size,
                    remove_background=True,
                    normalize=True,
                    min_psnr=0,
                    na_mask=preloaded.modelpsfgen.na_mask
                )

        preloaded.modelpsfgen.update_ideal_psf_with_empirical(ideal_empirical_preprocessed_psf)

    if psf_type is not None and preloaded.modelpsfgen.psf_type != psf_type:
        logger.info(f"Loading new PSF type: {psf_type}")
        preloaded = Preloadedmodelclass(
            modelpath,
            n_modes=n_modes,
            psf_type=psf_type,
        )

    return preloaded.model, preloaded.modelpsfgen


@profile
def estimate_and_save_new_dm(
    savepath: Path,
    coefficients: np.array,
    dm_calibration: Path,
    dm_state: np.array,
    dm_damping_scalar: float = 1
):
    dm = pd.DataFrame(utils.zernikies_to_actuators(
        coefficients,
        dm_calibration=dm_calibration,
        dm_state=dm_state,
        scalar=dm_damping_scalar
    ))
    dm.to_csv(savepath, index=False, header=False)
    return dm.values


def generate_embeddings(
    file: Union[tf.Tensor, Path, str],
    model: Union[tf.keras.Model, Path, str],
    axial_voxel_size: float,
    lateral_voxel_size: float,
    wavelength: float = .510,
    freq_strength_threshold: float = .01,
    remove_background: bool = True,
    read_noise_bias: float = 5,
    normalize: bool = True,
    plot: bool = False,
    fov_is_small: bool = True,
    preloaded: Preloadedmodelclass = None,
    ideal_empirical_psf: Any = None,
    digital_rotations: Optional[int] = None,
    psf_type: Optional[Union[str, Path]] = None,
    min_psnr: int = 5,
    object_gaussian_kernel_width: float = 0
):

    model, modelpsfgen = reloadmodel_if_needed(
        modelpath=model,
        preloaded=preloaded,
        psf_type=psf_type,
        ideal_empirical_psf=ideal_empirical_psf,
        ideal_empirical_psf_voxel_size=(axial_voxel_size, lateral_voxel_size, lateral_voxel_size)
    )

    sample = backend.load_sample(file)

    samplepsfgen = SyntheticPSF(
        psf_type=modelpsfgen.psf_type,
        psf_shape=sample.shape,
        n_modes=model.output_shape[1],
        lam_detection=wavelength,
        x_voxel_size=lateral_voxel_size,
        y_voxel_size=lateral_voxel_size,
        z_voxel_size=axial_voxel_size
    )

    return backend.preprocess(
        sample,
        modelpsfgen=modelpsfgen,
        samplepsfgen=samplepsfgen,
        freq_strength_threshold=freq_strength_threshold,
        remove_background=remove_background,
        normalize=normalize,
        fov_is_small=fov_is_small,
        digital_rotations=digital_rotations,
        read_noise_bias=read_noise_bias,
        plot=file.with_suffix('') if plot else None,
        min_psnr=min_psnr,
        object_gaussian_kernel_width=object_gaussian_kernel_width
    )


@profile
def reconstruct_wavefront_error_landscape(
    wavefronts: dict,
    xtiles: int,
    ytiles: int,
    ztiles: int,
    image: np.ndarray,
    save_path: Union[Path, str],
    window_size: tuple,
    lateral_voxel_size: float = .097,
    axial_voxel_size: float = .2,
    wavelength: float = .510,
    threshold: float = 0.,
    na: float = 1.0,
    tile_p2v: Optional[np.ndarray] = None,
):
    """
    Calculate the wavefront error landscape that would produce the wavefront error differences
    that we've measured between tiles.

    1. Calc wavefront error p2v difference between adjacent tiles. (e.g. wavefront error slope)
    2. Solve for the wavefront error using LS following wavefront reconstruction technique:
    W.H. Southwell, "Wave-front estimation from wave-front slope measurements," J. Opt. Soc. Am. 70, 998-1006 (1980)
    https://doi.org/10.1364/JOSA.70.000998

    S = A phi

    S = vector of slopes.  length = # of measurements
    A = matrix operator that calculates slopes (e.g. rise/run = (neighbor - current) / stride)
        number of rows = # of measurements
        number of cols = # of tile coordinates (essentially 3D meshgrid of coordinates flattened to 1D array)
        filled with all zeros except where slope is calculated (and we put -1 and +1 on the coordinate pair)
    phi = vector of the wavefront error at the coordinates. lenght = # of coordinates

    Args:
        wavefronts: wavefronts at each tile location
        na: Numerical aperature limit which to use for calculating p2v error

    Returns:
        terrain3d: wavefront error in units of waves

    """
    def get_neighbors(tile_coords: Union[tuple, np.array]):
        """
        Args:
            tile_coords: (z, y, x)

        Returns:
            The coordinates of the three bordering neighbors *forward* of the input tile (avoids double counting)
        """
        return [
            tuple(np.array(tile_coords) + np.array([1, 0, 0])),  # z neighbour
            tuple(np.array(tile_coords) + np.array([0, 1, 0])),  # y neighbour
            tuple(np.array(tile_coords) + np.array([0, 0, 1])),  # x neighbour
        ]

    num_coords = xtiles * ytiles * ztiles
    num_dimensions = 3
    num_measurements = num_coords * num_dimensions  # max limit of number of cube borders
    slopes = np.zeros(num_measurements)             # 1D vector of measurements
    A = np.zeros((num_measurements, num_coords))    # 2D matrix

    h = np.array(window_size) * (axial_voxel_size, lateral_voxel_size, lateral_voxel_size)
    h = utils.microns2waves(h, wavelength=wavelength)

    # center = (ztiles//2, ytiles//2, xtiles//2)
    # peak = predictions.apply(lambda x: x**2).groupby(['z', 'y', 'x']).sum().idxmax()
    if tile_p2v is None:
        tile_p2v = np.full((xtiles * ytiles * ztiles), np.nan)

    matrix_row = 0  # pointer to where we are writing
    for i, tile_coords in enumerate(itertools.product(range(ztiles), range(ytiles), range(xtiles))):
        neighbours = list(get_neighbors(tile_coords))
        tile_wavefront = wavefronts[tile_coords]
        if np.isnan(tile_p2v[i]):
            tile_p2v[i] = tile_wavefront.peak2valley(na=na)

        for k, neighbour_coords in enumerate(neighbours):  # ordered as (z, y, x) neighbours
            try:
                try:
                    j = np.ravel_multi_index(neighbour_coords, (ztiles, ytiles, xtiles))
                except ValueError:
                    continue

                neighbour_wavefront = wavefronts[neighbour_coords]
                if np.isnan(tile_p2v[j]):
                    tile_p2v[j] = neighbour_wavefront.peak2valley(na=na)

                diff_wavefront = Wavefront(tile_wavefront - neighbour_wavefront, lam_detection=wavelength)
                p2v = diff_wavefront.peak2valley(na=na)

                v1 = np.dot(tile_wavefront.amplitudes_ansi_waves, tile_wavefront.amplitudes_ansi_waves)
                v2 = np.dot(tile_wavefront.amplitudes_ansi_waves, neighbour_wavefront.amplitudes_ansi_waves)

                if v2 < v1:  # choose negative slope when neighbor has less aberration along the current aberration
                    p2v *= -1

                if tile_p2v[i] > threshold and tile_p2v[j] > threshold:
                    # rescale slopes with the distance between tiles (h)
                    slopes[matrix_row] = p2v / h[k]
                    A[matrix_row, j] = 1 / h[k]
                    A[matrix_row, i] = -1 / h[k]
                    matrix_row += 1

            except KeyError:
                continue    # e.g. if neighbor is beyond the border or that tile was dropped

    # clip out empty measurements
    slopes = slopes[:matrix_row]
    A = A[:matrix_row, :]

    # add row of ones to prevent singular solutions.
    # This basically amounts to pinning the average of terrain3d to zero
    A = np.append(A, np.ones((1, A.shape[1])), axis=0)
    slopes = np.append(slopes, 0)   # add a corresponding value of zero.

    # terrain in waves
    terrain, _, _, _ = np.linalg.lstsq(A, slopes, rcond=None)
    terrain3d = np.reshape(terrain, (ztiles, ytiles, xtiles))

    # upsample from tile coordinates back to the volume
    terrain3d = resize(terrain3d, image.shape, mode='edge')
    # terrain3d = resize(terrain3d, volume_shape, order=0, mode='constant')  # to show tiles

    isoplanatic_patch_colormap = sns.color_palette('hls', n_colors=256)
    isoplanatic_patch_colormap = np.array(isoplanatic_patch_colormap) * 255

    # isoplanatic_patch_colormap = pd.read_csv(
    #     Path.joinpath(Path(__file__).parent, '../CETperceptual/CET-C2.csv').resolve(),
    #     header=None,
    #     index_col=None,
    #     dtype=np.ubyte
    # ).values

    terrain3d *= 255    # convert waves to colormap cycles
    terrain3d = (terrain3d % 256).round(0).astype(np.ubyte)  # wrap if terrain's span is > 1 wave

    #  terrain3d is full brightness RGB color then use vol to determine brightness
    terrain3d = isoplanatic_patch_colormap[terrain3d] * image[..., np.newaxis]
    terrain3d = terrain3d.astype(np.ubyte)
    imwrite(save_path, terrain3d, photometric='rgb', compression='deflate', dtype=np.float32)

    return terrain3d


@profile
def predict_sample(
    img: Path,
    model: Path,
    dm_calibration: Any,
    dm_state: Any,
    axial_voxel_size: float,
    lateral_voxel_size: float,
    wavelength: float = .605,
    dm_damping_scalar: float = 1,
    prediction_threshold: float = 0.0,
    freq_strength_threshold: float = .01,
    confidence_threshold: float = .02,
    sign_threshold: float = .9,
    plot: bool = False,
    plot_rotations: bool = False,
    num_predictions: int = 1,
    batch_size: int = 1,
    prev: Any = None,
    estimate_sign_with_decon: bool = False,
    ignore_modes: list = (0, 1, 2, 4),
    preloaded: Preloadedmodelclass = None,
    ideal_empirical_psf: Any = None,
    digital_rotations: Optional[int] = 361,
    psf_type: Optional[Union[str, Path]] = None,
    cpu_workers: int = -1,
    min_psnr: int = 5,
    object_gaussian_kernel_width: float = 0
):
    lls_defocus = 0.
    dm_state = None if (dm_state is None or str(dm_state) == 'None') else dm_state

    preloadedmodel, preloadedpsfgen = reloadmodel_if_needed(
        modelpath=model,
        preloaded=preloaded,
        psf_type=psf_type,
        ideal_empirical_psf=ideal_empirical_psf,
        ideal_empirical_psf_voxel_size=(axial_voxel_size, lateral_voxel_size, lateral_voxel_size)
    )
    no_phase = True if preloadedmodel.input_shape[1] == 3 else False

    logger.info(f"Loading file: {img.name}")
    sample = backend.load_sample(img)
    logger.info(f"Sample: {sample.shape}")

    samplepsfgen = SyntheticPSF(
        psf_type=preloadedpsfgen.psf_type,
        psf_shape=sample.shape,
        n_modes=preloadedmodel.output_shape[1],
        lam_detection=wavelength,
        x_voxel_size=lateral_voxel_size,
        y_voxel_size=lateral_voxel_size,
        z_voxel_size=axial_voxel_size
    )

    embeddings = backend.preprocess(
        sample,
        modelpsfgen=preloadedpsfgen,
        samplepsfgen=samplepsfgen,
        digital_rotations=digital_rotations,
        remove_background=True,
        normalize=True,
        fov_is_small=True,
        min_psnr=min_psnr,
        plot=Path(f"{img.with_suffix('')}_sample_predictions") if plot else None,
        object_gaussian_kernel_width=object_gaussian_kernel_width
    )

    if no_phase:
        p, std, pchange = backend.dual_stage_prediction(
            preloadedmodel,
            inputs=embeddings,
            threshold=prediction_threshold,
            sign_threshold=sign_threshold,
            n_samples=num_predictions,
            gen=samplepsfgen,
            modelgen=preloadedpsfgen,
            batch_size=batch_size,
            prev_pred=prev,
            estimate_sign_with_decon=estimate_sign_with_decon,
            ignore_modes=ignore_modes,
            freq_strength_threshold=freq_strength_threshold,
            plot=Path(f"{img.with_suffix('')}_sample_predictions") if plot else None,
        )
    else:
        res = backend.predict_rotation(
            preloadedmodel,
            inputs=embeddings,
            psfgen=preloadedpsfgen,
            no_phase=False,
            batch_size=batch_size,
            threshold=prediction_threshold,
            ignore_modes=ignore_modes,
            freq_strength_threshold=freq_strength_threshold,
            confidence_threshold=confidence_threshold,
            digital_rotations=digital_rotations,
            plot=Path(f"{img.with_suffix('')}_sample_predictions") if plot else None,
            plot_rotations=Path(f"{img.with_suffix('')}_sample_predictions") if plot_rotations else None,
            save_path=Path(f"{img.with_suffix('')}_sample_predictions"),
        )
        try:
            p, std = res
        except ValueError:
            p, std, lls_defocus = res

    p = Wavefront(p, order='ansi', lam_detection=wavelength)
    std = Wavefront(std, order='ansi', lam_detection=wavelength)

    coefficients = [
        {'n': z.n, 'm': z.m, 'amplitude': a}
        for z, a in p.zernikes.items()
    ]
    df = pd.DataFrame(coefficients, columns=['n', 'm', 'amplitude'])
    df.index.name = 'ansi'
    df.to_csv(f"{img.with_suffix('')}_sample_predictions_zernike_coefficients.csv")

    if dm_calibration is not None:
        estimate_and_save_new_dm(
            savepath=Path(f"{img.with_suffix('')}_sample_predictions_corrected_actuators.csv"),
            coefficients=df['amplitude'].values,
            dm_calibration=dm_calibration,
            dm_state=dm_state,
            dm_damping_scalar=dm_damping_scalar
        )

    psf = samplepsfgen.single_psf(phi=p, normed=True)
    imwrite(f"{img.with_suffix('')}_sample_predictions_psf.tif", psf, compression='deflate', dtype=np.float32)
    imwrite(f"{img.with_suffix('')}_sample_predictions_wavefront.tif", p.wave(), compression='deflate', dtype=np.float32)

    with Path(f"{img.with_suffix('')}_sample_predictions_settings.json").open('w') as f:
        json = dict(
            path=str(img),
            model=str(model),
            input_shape=list(sample.shape),
            sample_voxel_size=list([axial_voxel_size, lateral_voxel_size, lateral_voxel_size]),
            model_voxel_size=list(preloadedpsfgen.voxel_size),
            psf_fov=list(preloadedpsfgen.psf_fov),
            wavelength=float(wavelength),
            dm_calibration=str(dm_calibration),
            dm_state=str(dm_state),
            dm_damping_scalar=float(dm_damping_scalar),
            prediction_threshold=float(prediction_threshold),
            freq_strength_threshold=float(freq_strength_threshold),
            prev=str(prev),
            ignore_modes=list(ignore_modes),
            ideal_empirical_psf=str(ideal_empirical_psf),
            lls_defocus=float(lls_defocus),
            zernikes=list(coefficients),
            psf_type=str(preloadedpsfgen.psf_type),
        )

        ujson.dump(
            json,
            f,
            indent=4,
            sort_keys=False,
            ensure_ascii=False,
            escape_forward_slashes=False
        )

    if plot:
        vis.diagnosis(
            pred=p,
            pred_std=std,
            save_path=Path(f"{img.with_suffix('')}_sample_predictions_diagnosis"),
            lls_defocus=lls_defocus
        )

    return df


@profile
def predict_large_fov(
    img: Path,
    model: Path,
    dm_calibration: Any,
    dm_state: Any,
    axial_voxel_size: float,
    lateral_voxel_size: float,
    wavelength: float = .605,
    dm_damping_scalar: float = 1,
    prediction_threshold: float = 0.0,
    freq_strength_threshold: float = .01,
    confidence_threshold: float = .02,
    sign_threshold: float = .9,
    plot: bool = False,
    plot_rotations: bool = False,
    num_predictions: int = 1,
    batch_size: int = 1,
    prev: Any = None,
    estimate_sign_with_decon: bool = False,
    ignore_modes: list = (0, 1, 2, 4),
    preloaded: Preloadedmodelclass = None,
    ideal_empirical_psf: Any = None,
    digital_rotations: Optional[int] = 361,
    psf_type: Optional[Union[str, Path]] = None,
    cpu_workers: int = -1,
    min_psnr: int = 5,
    object_gaussian_kernel_width: float = 0
):
    lls_defocus = 0.
    dm_state = None if (dm_state is None or str(dm_state) == 'None') else dm_state
    sample_voxel_size = (axial_voxel_size, lateral_voxel_size, lateral_voxel_size)

    preloadedmodel, preloadedpsfgen = reloadmodel_if_needed(
        modelpath=model,
        preloaded=preloaded,
        psf_type=psf_type,
        ideal_empirical_psf=ideal_empirical_psf,
        ideal_empirical_psf_voxel_size=sample_voxel_size
    )
    no_phase = True if preloadedmodel.input_shape[1] == 3 else False

    sample = backend.load_sample(img)
    logger.info(f"Sample: {sample.shape}")

    samplepsfgen = SyntheticPSF(
        psf_type=preloadedpsfgen.psf_type,
        psf_shape=sample.shape,
        n_modes=preloadedmodel.output_shape[1],
        lam_detection=wavelength,
        x_voxel_size=lateral_voxel_size,
        y_voxel_size=lateral_voxel_size,
        z_voxel_size=axial_voxel_size
    )

    embeddings = backend.preprocess(
        sample,
        modelpsfgen=preloadedpsfgen,
        samplepsfgen=samplepsfgen,
        digital_rotations=digital_rotations,
        no_phase=no_phase,
        freq_strength_threshold=freq_strength_threshold,
        remove_background=True,
        normalize=True,
        fov_is_small=False,
        min_psnr=min_psnr,
        rolling_strides=optimal_rolling_strides(preloadedpsfgen.psf_fov, sample_voxel_size, sample.shape),
        plot=Path(f"{img.with_suffix('')}_large_fov_predictions") if plot else None,
        object_gaussian_kernel_width=object_gaussian_kernel_width
    )

    res = backend.predict_rotation(
        preloadedmodel,
        inputs=embeddings,
        psfgen=preloadedpsfgen,
        no_phase=False,
        batch_size=batch_size,
        threshold=prediction_threshold,
        ignore_modes=ignore_modes,
        freq_strength_threshold=freq_strength_threshold,
        confidence_threshold=confidence_threshold,
        digital_rotations=digital_rotations,
        plot=Path(f"{img.with_suffix('')}_large_fov_predictions") if plot else None,
        plot_rotations=Path(f"{img.with_suffix('')}_large_fov_predictions") if plot_rotations else None,
        cpu_workers=cpu_workers,
        save_path=Path(f"{img.with_suffix('')}_large_fov_predictions"),
    )
    try:
        p, std = res
    except ValueError:
        p, std, lls_defocus = res

    p = Wavefront(p, order='ansi', lam_detection=wavelength)
    std = Wavefront(std, order='ansi', lam_detection=wavelength)

    coefficients = [
        {'n': z.n, 'm': z.m, 'amplitude': a}
        for z, a in p.zernikes.items()
    ]
    df = pd.DataFrame(coefficients, columns=['n', 'm', 'amplitude'])
    df.index.name = 'ansi'
    df.to_csv(f"{img.with_suffix('')}_large_fov_predictions_zernike_coefficients.csv")

    if dm_calibration is not None:
        estimate_and_save_new_dm(
            savepath=Path(f"{img.with_suffix('')}_large_fov_predictions_corrected_actuators.csv"),
            coefficients=df['amplitude'].values,
            dm_calibration=dm_calibration,
            dm_state=dm_state,
            dm_damping_scalar=dm_damping_scalar
        )

    psf = samplepsfgen.single_psf(phi=p, normed=True)
    imwrite(f"{img.with_suffix('')}_large_fov_predictions_psf.tif", psf, compression='deflate', dtype=np.float32)
    imwrite(f"{img.with_suffix('')}_large_fov_predictions_wavefront.tif", p.wave(), compression='deflate', dtype=np.float32)

    with Path(f"{img.with_suffix('')}_large_fov_predictions_settings.json").open('w') as f:
        json = dict(
            path=str(img),
            model=str(model),
            input_shape=list(sample.shape),
            sample_voxel_size=list([axial_voxel_size, lateral_voxel_size, lateral_voxel_size]),
            model_voxel_size=list(preloadedpsfgen.voxel_size),
            psf_fov=list(preloadedpsfgen.psf_fov),
            wavelength=float(wavelength),
            dm_calibration=str(dm_calibration),
            dm_state=str(dm_state),
            dm_damping_scalar=float(dm_damping_scalar),
            prediction_threshold=float(prediction_threshold),
            freq_strength_threshold=float(freq_strength_threshold),
            prev=str(prev),
            ignore_modes=list(ignore_modes),
            ideal_empirical_psf=str(ideal_empirical_psf),
            lls_defocus=float(lls_defocus),
            zernikes=list(coefficients),
            psf_type=str(preloadedpsfgen.psf_type),
        )

        ujson.dump(
            json,
            f,
            indent=4,
            sort_keys=False,
            ensure_ascii=False,
            escape_forward_slashes=False
        )

    if plot:
        vis.diagnosis(
            pred=p,
            pred_std=std,
            save_path=Path(f"{img.with_suffix('')}_large_fov_predictions_diagnosis"),
            lls_defocus=lls_defocus
        )

    return df


@profile
def predict_rois(
    img: Path,
    model: Path,
    pois: Any,
    dm_calibration: Any,
    dm_state: Any,
    axial_voxel_size: float,
    lateral_voxel_size: float,
    wavelength: float = .605,
    num_predictions: int = 1,
    batch_size: int = 1,
    window_size: tuple = (64, 64, 64),
    num_rois: int = 10,
    min_intensity: int = 200,
    prediction_threshold: float = 0.,
    freq_strength_threshold: float = .01,
    minimum_distance: float = 1.,
    plot: bool = False,
    plot_rotations: bool = False,
    ignore_modes: list = (0, 1, 2, 4),
    preloaded: Preloadedmodelclass = None,
    ideal_empirical_psf: Any = None,
    sign_threshold: float = .9,
    prev: Any = None,
    estimate_sign_with_decon: bool = False,
    digital_rotations: Optional[int] = 361,
    psf_type: Optional[Union[str, Path]] = None,
    cpu_workers: int = -1
):

    preloadedmodel, preloadedpsfgen = reloadmodel_if_needed(
        modelpath=model,
        preloaded=preloaded,
        psf_type=psf_type,
        ideal_empirical_psf=ideal_empirical_psf,
        ideal_empirical_psf_voxel_size=(axial_voxel_size, lateral_voxel_size, lateral_voxel_size)
    )

    outdir = Path(f"{img.with_suffix('')}_rois")
    outdir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Loading file: {img.name}")
    sample = backend.load_sample(img)
    logger.info(f"Sample: {sample.shape}")

    rois, ztiles, nrows, ncols = find_roi(
        sample,
        savepath=outdir,
        pois=pois,
        window_size=window_size,
        plot=f"{outdir}_predictions" if plot else None,
        num_rois=num_rois,
        min_dist=minimum_distance,
        max_dist=None,
        max_neighbor=20,
        min_intensity=min_intensity,
        voxel_size=(axial_voxel_size, lateral_voxel_size, lateral_voxel_size),
    )

    samplepsfgen = SyntheticPSF(
        psf_type=preloadedpsfgen.psf_type,
        psf_shape=sample.shape,
        n_modes=preloadedmodel.output_shape[1],
        lam_detection=wavelength,
        x_voxel_size=lateral_voxel_size,
        y_voxel_size=lateral_voxel_size,
        z_voxel_size=axial_voxel_size
    )

    with Path(f"{img.with_suffix('')}_rois_predictions_settings.json").open('w') as f:
        json = dict(
            path=str(img),
            model=str(model),
            input_shape=list(sample.shape),
            sample_voxel_size=list(samplepsfgen.voxel_size),
            model_voxel_size=list(preloadedpsfgen.voxel_size),
            psf_fov=list(preloadedpsfgen.psf_fov),
            wavelength=float(wavelength),
            prediction_threshold=float(prediction_threshold),
            freq_strength_threshold=float(freq_strength_threshold),
            prev=str(prev),
            ignore_modes=list(ignore_modes),
            ideal_empirical_psf=str(ideal_empirical_psf),
            number_of_tiles=int(len(rois)),
            ztiles=int(ztiles),
            ytiles=int(nrows),
            xtiles=int(ncols),
            dm_calibration=str(dm_calibration),
            psf_type=str(preloadedpsfgen.psf_type),
        )

        ujson.dump(
            json,
            f,
            indent=4,
            sort_keys=False,
            ensure_ascii=False,
            escape_forward_slashes=False
        )

    predictions, stdevs = backend.predict_files(
        paths=rois,
        outdir=outdir,
        model=preloadedmodel,
        modelpsfgen=preloadedpsfgen,
        samplepsfgen=samplepsfgen,
        dm_calibration=dm_calibration,
        dm_state=dm_state,
        prediction_threshold=prediction_threshold,
        batch_size=batch_size,
        wavelength=wavelength,
        ztiles=ztiles,
        nrows=nrows,
        ncols=ncols,
        ignore_modes=ignore_modes,
        freq_strength_threshold=freq_strength_threshold,
        plot=plot,
        plot_rotations=plot_rotations,
        digital_rotations=digital_rotations,
        cpu_workers=cpu_workers
    )
    return predictions


def predict_snr_map(
    img: Path,
    window_size: tuple = (64, 64, 64),
    save_files: bool = False
):

    logger.info(f"Loading file: {img.name}")
    sample = backend.load_sample(img)
    logger.info(f"Sample: {sample.shape}")

    outdir = Path(f"{img.with_suffix('')}_tiles")
    if not outdir.exists():
        save_files = True   # need to generate tile tiff files

    outdir.mkdir(exist_ok=True, parents=True)

    # obtain each tile filename. Skip saving to .tif if we have them already.
    tiles, ztiles, nrows, ncols = get_tiles(
        sample,
        savepath=outdir,
        strides=window_size,
        window_size=window_size,
        save_files=save_files,
    )
    rois = tiles['path'].values

    prep = partial(prep_sample, return_psnr=True, min_psnr=0)
    snrs = utils.multiprocess(func=prep, jobs=rois, desc=f'Calc PNSRs.', unit="tiles")
    snrs = np.reshape(snrs, (ztiles, nrows, ncols))
    snrs = resize(snrs, (snrs.shape[0], sample.shape[1], sample.shape[2]), order=1, mode='edge')
    snrs = resize(snrs, sample.shape, order=0, mode='edge')
    imwrite(Path(f"{img.with_suffix('')}_snrs.tif"), snrs.astype(np.float32), compression='deflate', dtype=np.float32)


@profile
def predict_tiles(
    img: Path,
    model: Path,
    dm_calibration: Any,
    dm_state: Any,
    axial_voxel_size: float,
    lateral_voxel_size: float,
    wavelength: float = .605,
    num_predictions: int = 1,
    batch_size: int = 1,
    window_size: tuple = (64, 64, 64),
    freq_strength_threshold: float = .01,
    confidence_threshold: float = .02,
    sign_threshold: float = .9,
    plot: bool = False,
    plot_rotations: bool = False,
    prev: Any = None,
    estimate_sign_with_decon: bool = False,
    ignore_modes: list = (0, 1, 2, 4),
    preloaded: Preloadedmodelclass = None,
    ideal_empirical_psf: Any = None,
    digital_rotations: Optional[int] = 361,
    cpu_workers: int = -1,
    shifting: tuple = (0, 0, 0),
    psf_type: Optional[Union[str, Path]] = None,
    min_psnr: int = 5,
    object_gaussian_kernel_width: float = 0
):
    # Begin spawning workers for Generate Fourier Embeddings (windows only). Must die to release their GPU memory.
    # if platform.system() == "Windows":
    #     if cpu_workers == 1:
    #         pool = None
    #     elif cpu_workers == -1:
    #         pool = mp.Pool(mp.cpu_count())
    #     else:
    #         pool = mp.Pool(processes=cpu_workers)
    # else:
    #     pool = None     #
    pool = None

    dm_state = utils.load_dm(dm_state)

    preloadedmodel, preloadedpsfgen = reloadmodel_if_needed(
        modelpath=model,
        preloaded=preloaded,
        psf_type=psf_type,
        ideal_empirical_psf=ideal_empirical_psf,
        ideal_empirical_psf_voxel_size=(axial_voxel_size, lateral_voxel_size, lateral_voxel_size)
    )

    logger.info(f"Loading file: {img.name}")
    sample = backend.load_sample(img)
    logger.info(f"Sample: {sample.shape}")

    if any(np.array(shifting) != 0):
        sample = shift(sample, shift=(-1*shifting[0], -1*shifting[1], -1*shifting[2]))
        img = Path(f"{img.with_suffix('')}_shifted_z{shifting[0]}_y{shifting[1]}_x{shifting[2]}.tif")
        imwrite(img, sample.astype(np.float32), compression='deflate', dtype=np.float32)

    outdir = Path(f"{img.with_suffix('')}_tiles")
    outdir.mkdir(exist_ok=True, parents=True)

    samplepsfgen = SyntheticPSF(
        psf_type=preloadedpsfgen.psf_type,
        psf_shape=window_size,
        n_modes=preloadedmodel.output_shape[1],
        lam_detection=wavelength,
        x_voxel_size=lateral_voxel_size,
        y_voxel_size=lateral_voxel_size,
        z_voxel_size=axial_voxel_size
    )

    fov_is_small = True if all(np.array(samplepsfgen.psf_fov) <= np.array(preloadedpsfgen.psf_fov)) else False

    if fov_is_small:  # only going to center crop and predict on that single FOV (fourier_embeddings)
        prep = partial(
            prep_sample,
            model_fov=preloadedpsfgen.psf_fov,              # this is what we will crop to
            sample_voxel_size=samplepsfgen.voxel_size,
            remove_background=True,
            normalize=True,
            min_psnr=min_psnr,
            na_mask=samplepsfgen.na_mask
        )
    else:
        prep = partial(
            prep_sample,
            sample_voxel_size=samplepsfgen.voxel_size,
            remove_background=True,
            normalize=True,
            min_psnr=min_psnr,
            na_mask=samplepsfgen.na_mask
        )

    # obtain each tile and save to .tif.
    tiles, ztiles, nrows, ncols = get_tiles(
        sample,
        savepath=outdir,
        strides=window_size,
        window_size=window_size,
        prep=prep,
        plot=plot
    )
    rois = tiles[tiles['ignored'] == False]['path'].values  # skip tiles with low snr or no signal
    logger.info(f" {rois.shape[0]} valid tiles found with sufficient SNR out of {tiles.shape[0]}")
    if rois.shape[0] == 0:
        raise Exception(f'No valid tiles found with sufficient SNR. Please use a different region.')
    
    template = pd.DataFrame(columns=tiles.index.values)

    with Path(f"{img.with_suffix('')}_tiles_predictions_settings.json").open('w') as f:
        json = dict(
            path=str(img),
            model=str(model),
            input_shape=list(sample.shape),
            sample_voxel_size=list(samplepsfgen.voxel_size),
            model_voxel_size=list(preloadedpsfgen.voxel_size),
            psf_fov=list(preloadedpsfgen.psf_fov),
            window_size=list(window_size),
            wavelength=float(wavelength),
            prediction_threshold=float(0),
            dm_state=list(dm_state),
            freq_strength_threshold=float(freq_strength_threshold),
            prev=str(prev),
            ignore_modes=list(ignore_modes),
            ideal_empirical_psf=str(ideal_empirical_psf),
            number_of_tiles=int(len(rois)),
            ztiles=int(ztiles),
            ytiles=int(nrows),
            xtiles=int(ncols),
            dm_calibration=str(dm_calibration),
            psf_type=str(preloadedpsfgen.psf_type),
        )

        ujson.dump(
            json,
            f,
            indent=4,
            sort_keys=False,
            ensure_ascii=False,
            escape_forward_slashes=False
        )

    predictions, stdevs = backend.predict_files(
        paths=rois,
        outdir=outdir,
        model=preloadedmodel,
        modelpsfgen=preloadedpsfgen,
        samplepsfgen=samplepsfgen,
        dm_calibration=dm_calibration,
        dm_state=dm_state,
        prediction_threshold=0,
        confidence_threshold=confidence_threshold,
        batch_size=batch_size,
        wavelength=wavelength,
        ignore_modes=ignore_modes,
        freq_strength_threshold=freq_strength_threshold,
        fov_is_small=fov_is_small,
        plot=plot,
        plot_rotations=plot_rotations,
        digital_rotations=digital_rotations,
        rolling_strides=optimal_rolling_strides(preloadedpsfgen.psf_fov, samplepsfgen.voxel_size, window_size),
        cpu_workers=cpu_workers,
        skip_prep_sample=True,
        template=template,
        pool=pool,
        object_gaussian_kernel_width=object_gaussian_kernel_width
    )

    return predictions


@profile
def predict_folder(
        folder: Path,
        model: Path,
        dm_calibration: Any,
        dm_state: Any,
        axial_voxel_size: float,
        lateral_voxel_size: float,
        wavelength: float = .605,
        num_predictions: int = 1,
        batch_size: int = 1,
        freq_strength_threshold: float = .01,
        confidence_threshold: float = .02,
        sign_threshold: float = .9,
        plot: bool = False,
        plot_rotations: bool = False,
        prev: Any = None,
        estimate_sign_with_decon: bool = False,
        ignore_modes: list = (0, 1, 2, 4),
        preloaded: Preloadedmodelclass = None,
        ideal_empirical_psf: Any = None,
        digital_rotations: Optional[int] = 361,
        cpu_workers: int = -1,
        shifting: tuple = (0, 0, 0),
        psf_type: Optional[Union[str, Path]] = None,
        min_psnr: int = 5,
        object_gaussian_kernel_width: float = 0,
        filename_pattern: str = r"*[!_gt|!_realspace|!_noisefree|!_predictions_psf|!_corrected_psf|!_reconstructed_psf].tif"
):
    pool = None
    dm_state = utils.load_dm(dm_state)

    outdir = folder / 'files'
    outdir.mkdir(exist_ok=True, parents=True)

    candidates = sorted(Path(folder).rglob(filename_pattern))
    with TiffFile(candidates[0]) as tif:
        input_shape = tif.asarray().shape

    preloadedmodel, preloadedpsfgen = reloadmodel_if_needed(
        modelpath=model,
        preloaded=preloaded,
        psf_type=psf_type,
        ideal_empirical_psf=ideal_empirical_psf,
        ideal_empirical_psf_voxel_size=(axial_voxel_size, lateral_voxel_size, lateral_voxel_size)
    )

    samplepsfgen = SyntheticPSF(
        psf_type=preloadedpsfgen.psf_type,
        psf_shape=input_shape,
        n_modes=preloadedmodel.output_shape[1],
        lam_detection=wavelength,
        x_voxel_size=lateral_voxel_size,
        y_voxel_size=lateral_voxel_size,
        z_voxel_size=axial_voxel_size
    )

    fov_is_small = True if all(np.array(samplepsfgen.psf_fov) <= np.array(preloadedpsfgen.psf_fov)) else False

    if fov_is_small:  # only going to center crop and predict on that single FOV (fourier_embeddings)
        prep = partial(
            prep_sample,
            model_fov=preloadedpsfgen.psf_fov,  # this is what we will crop to
            sample_voxel_size=samplepsfgen.voxel_size,
            remove_background=True,
            normalize=True,
            min_psnr=min_psnr,
            expand_dims=False,
            na_mask=samplepsfgen.na_mask
        )
    else:
        prep = partial(
            prep_sample,
            sample_voxel_size=samplepsfgen.voxel_size,
            remove_background=True,
            normalize=True,
            min_psnr=min_psnr,
            expand_dims=False,
            na_mask=samplepsfgen.na_mask
        )

    files = {}
    for path in tqdm(
        candidates,
        desc=f"Locating files: {len(candidates)}",
        bar_format='{l_bar}{bar}{r_bar} {elapsed_s:.1f}s elapsed',
        unit=' file',
        file=sys.stdout
    ):
        f = prep(path,  plot=outdir/path.stem if plot else None)
        if np.all(f == 0):
            files[path.stem] = dict(
                path=outdir / path.name,
                ignored=True,
            )
        else:
            files[path.stem] = dict(
                path=outdir / path.name,
                ignored=False,
            )
            imwrite(outdir / path.name, f, compression='deflate', dtype=np.float32)

    files = pd.DataFrame.from_dict(files, orient='index')
    rois = files[files['ignored'] == False]['path'].values  # skip files with low snr or no signal
    logger.info(
        f" {rois.shape[0]} valid files found that meet criteria [{filename_pattern}]"
        f" with sufficient SNR out of {files.shape[0]}"
    )

    if rois.shape[0] == 0:
        raise Exception(f'No valid files found with sufficient SNR. Please use a different folder.')

    template = pd.DataFrame(columns=files.index.values)

    with Path(f"{folder}/folder_predictions_settings.json").open('w') as f:
        json = dict(
            path=str(folder),
            model=str(model),
            input_shape=input_shape,
            sample_voxel_size=list(samplepsfgen.voxel_size),
            model_voxel_size=list(preloadedpsfgen.voxel_size),
            psf_fov=list(preloadedpsfgen.psf_fov),
            wavelength=float(wavelength),
            prediction_threshold=float(0),
            dm_state=list(dm_state),
            freq_strength_threshold=float(freq_strength_threshold),
            prev=str(prev),
            ignore_modes=list(ignore_modes),
            ideal_empirical_psf=str(ideal_empirical_psf),
            number_of_tiles=int(len(rois)),
            dm_calibration=str(dm_calibration),
            psf_type=str(preloadedpsfgen.psf_type),
        )

        ujson.dump(
            json,
            f,
            indent=4,
            sort_keys=False,
            ensure_ascii=False,
            escape_forward_slashes=False
        )

    predictions, stdevs = backend.predict_files(
        paths=rois,
        outdir=outdir,
        model=preloadedmodel,
        modelpsfgen=preloadedpsfgen,
        samplepsfgen=samplepsfgen,
        dm_calibration=dm_calibration,
        dm_state=dm_state,
        prediction_threshold=0,
        confidence_threshold=confidence_threshold,
        batch_size=batch_size,
        wavelength=wavelength,
        ignore_modes=ignore_modes,
        freq_strength_threshold=freq_strength_threshold,
        fov_is_small=fov_is_small,
        plot=plot,
        plot_rotations=plot_rotations,
        digital_rotations=digital_rotations,
        cpu_workers=cpu_workers,
        skip_prep_sample=True,
        template=template,
        pool=pool,
        object_gaussian_kernel_width=object_gaussian_kernel_width
    )

    return predictions


def kmeans_clustering(data, k):
    km = KMeans(
        init="k-means++",
        n_clusters=k
    ).fit(data)
    labels = km.fit_predict(data)
    silhouette = silhouette_score(data, labels)
    return silhouette


def cluster_tiles(
    predictions: pd.DataFrame,
    stdevs: pd.DataFrame,
    where_unconfident: pd.DataFrame,
    dm_calibration: Path,
    dm_state: Any,
    savepath: Path,
    plot: bool = False,
    wavelength: float = .510,
    aggregation_rule: str = 'mean',
    max_isoplanatic_clusters: int = 3,
    optimize_max_isoplanatic_clusters: bool = False,
    dm_damping_scalar: float = 1,
    postfix: str = 'aggregated',
    minimum_number_of_tiles_per_cluster: np.ndarray = np.array([3]),
):
    """
        Group tiles with similar wavefronts together,
        adding a new column to the `predictions` dataframe to indicate the predicted cluster ID for each tile

    Args:
        postfix: suffix to file names.  Used to designate 'aggregated' and 'consensus'
        predictions: dataframe of all predictions indexed by tile IDs
        stdevs: dataframe of all standard deviations of the predictions indexed by tile IDs
        where_unconfident: dataframe mask for unconfident tiles
        dm_calibration: DM calibration file
        dm_state: current DM
        savepath: path to save DMs for each selected cluster
        wavelength: detection wavelength
        aggregation_rule: metric to use to combine wavefronts of all tiles in a given cluster
        max_isoplanatic_clusters: max number of clusters
        optimize_max_isoplanatic_clusters: a toggle to find the optimal number of clusters automatically
        dm_damping_scalar: optional scalar to apply for the DM of each cluster
        plot: a toggle to plot the wavefront for each cluster
        minimum_number_of_tiles_per_cluster: if a cluster has less than this, those tiles in that clusters will be
            excluded as outliers, and the set will be reclustered.

    Returns:
        Updated prediction, stdevs dataframes
    """
    # create a new column for cluster ids.
    predictions['cluster'] = np.nan
    stdevs['cluster'] = np.nan

    pool = mp.Pool(processes=4)  # async pool for plotting

    # valid_predictions = predictions.loc[~(unconfident_tiles | zero_confident_tiles | all_zeros_tiles)]
    valid_predictions = predictions.groupby('z')

    # valid_stdevs = stdevs.loc[~(unconfident_tiles | zero_confident_tiles | all_zeros_tiles)]
    valid_stdevs = stdevs.groupby('z')

    wavefronts, coefficients, actuators = {}, {}, {}
    wavefronts_montage = np.zeros((len(valid_predictions.groups.keys())*64, (max_isoplanatic_clusters+1)*64)).astype(np.float32)
    for z in valid_predictions.groups.keys():  # basically loop through all ztiles, unless no valid predictions exist
        ztile_preds = valid_predictions.get_group(z)
        ztile_preds.drop(columns=['cluster', 'p2v'], errors='ignore', inplace=True)

        ztile_stds = valid_stdevs.get_group(z)
        ztile_stds.drop(columns=['cluster', 'p2v'], errors='ignore', inplace=True)

        if optimize_max_isoplanatic_clusters:
            logger.info('KMeans calculating...')
            ks = np.arange(2, max_isoplanatic_clusters + 1)
            ans = Parallel(n_jobs=-1, verbose=0)(delayed(kmeans_clustering)(ztile_preds.values, k) for k in ks)
            results = pd.DataFrame(ans, index=ks, columns=['silhouette'])
            max_silhouette = results['silhouette'].idxmax()
            max_isoplanatic_clusters = max_silhouette

        # weight zernike coefficients by their mth order for clustering
        features = ztile_preds.copy().fillna(0)
        for mode, twin in Wavefront(np.zeros(features.shape[1])).twins.items():
            if twin is not None:
                features[mode.index_ansi] /= abs(mode.m - 1)
                features[twin.index_ansi] /= twin.m + 1
            else:  # spherical modes
                features[mode.index_ansi] /= mode.m + 1

        n_clusters = min(max_isoplanatic_clusters, len(features)) + 1

        compute_clustering = True
        while compute_clustering:
            clustering = KMeans(n_clusters=n_clusters, max_iter=1000, random_state=0)
            clustering.fit(features)    # Cluster calculation

            ztile_preds['cluster'] = clustering.predict(features)   # Predict the closest cluster each tile belongs to

            if ztile_preds['cluster'].unique().size < max_isoplanatic_clusters:
                # We didn't have enough tiles to make all requested clusters. We're done.
                compute_clustering = False
            else:
                # Check if each cluster has enough tiles.
                tile_counts = ztile_preds['cluster'].value_counts()
                for cluster_id in ztile_preds['cluster'].unique():
                    if tile_counts[cluster_id] < minimum_number_of_tiles_per_cluster[z]:
                        outliers = ztile_preds[ztile_preds['cluster'] == cluster_id].index
                        features.loc[outliers] = 0  # Set outliers to yellow group, then recluster.
                        logger.info(f"Removed from clustering were {outliers.size} outliers from z slab {z}. "
                                    f"Min # of tiles per cluster = {minimum_number_of_tiles_per_cluster[z]}. "
                                    f"Outlier tiles={outliers.to_list()}")
                        compute_clustering = True
                        break
                    else:
                        compute_clustering = False

        # sort clusters by p2v
        centers_mag = [
            ztile_preds[ztile_preds['cluster'] == i].mask(where_unconfident).drop(columns='cluster').fillna(0).agg(
                aggregation_rule, axis=0
            ).values
            for i in range(n_clusters)
        ]
        centers_mag = np.array([Wavefront(np.nan_to_num(c, nan=0)).peak2valley() for c in centers_mag])
        ztile_preds['cluster'] = ztile_preds['cluster'].replace(dict(zip(np.argsort(centers_mag), range(n_clusters))))
        ztile_preds['cluster'] += z * (max_isoplanatic_clusters + 1)

        # assign KMeans cluster ids to full dataframes (untouched ones, remain NaN)
        predictions.loc[ztile_preds.index, 'cluster'] = ztile_preds['cluster']
        stdevs.loc[ztile_preds.index, 'cluster'] = ztile_preds['cluster']

        # remove the first (null) cluster from the dataframe
        # we'll the original DM for this cluster
        ztile_preds = ztile_preds[ztile_preds['cluster'] != z * (max_isoplanatic_clusters + 1)]

        print(f"\nNumber of tiles in each cluster of {postfix} map, z={z}")
        print("c    count")
        print(ztile_preds['cluster'].value_counts().sort_index().to_string())
        clusters = ztile_preds.groupby('cluster')
        for k in range(max_isoplanatic_clusters + 1):
            c = k + z * (max_isoplanatic_clusters + 1)

            if k == 0:  # "before" volume
                pred = np.zeros(features.shape[-1])  # "before" will not have a wavefront update here.
                pred_std = np.zeros(features.shape[-1])
            elif k >= n_clusters or c not in ztile_preds['cluster'].unique():  # if we didn't have enough tiles
                pred = np.zeros(features.shape[-1])  # these will not have a wavefront update here.
                pred_std = np.zeros(features.shape[-1])
                logger.warning(f'Not enough tiles to make another cluster.  '
                               f'This cluster will not have a wavefront update: z{z}_c{c}')
            else:  # "after" volumes
                g = clusters.get_group(c).index

                # come up with a pred for this cluster based on user's choice of metric ("mean", "median", ...)
                if aggregation_rule == 'centers':
                    pred = clustering.cluster_centers_[k - 1]
                    pred_std = ztile_stds.loc[g].mask(where_unconfident).agg('mean', axis=0)
                else:
                    pred = ztile_preds.loc[g].mask(where_unconfident).drop(columns='cluster').fillna(0).agg(aggregation_rule, axis=0)
                    if np.all(pred == 0):   # if 'aggregation_rule' fails (e.g. median on 3 tiles), fallback to 'mean'
                        pred = ztile_preds.loc[g].mask(where_unconfident).drop(columns='cluster').fillna(0).agg('mean', axis=0)
                        logger.info(f"Using 'mean' aggregation rule, because was getting all zeros from {aggregation_rule} on {ztile_preds.loc[g].shape[0]} tiles in cluster.")
                    pred_std = ztile_stds.loc[g].mask(where_unconfident).fillna(0).agg(aggregation_rule, axis=0)

            cluster = f'z{z}_c{c}'

            wavefronts[cluster] = Wavefront(
                np.nan_to_num(pred, nan=0, posinf=0, neginf=0),
                order='ansi',
                lam_detection=wavelength
            )
            wavefronts_montage[
                z*64:(z+1)*64,
                k*64:(k+1)*64,
            ] = wavefronts[cluster].wave(64).astype(np.float32)

            imwrite(Path(f"{savepath}_{postfix}_{cluster}_wavefront.tif"),
                    wavefronts_montage[z*64:(z+1)*64, k*64:(k+1)*64],
                    compression='deflate',
                    dtype=np.float32
            )

            pred_std = Wavefront(
                np.nan_to_num(pred_std, nan=0, posinf=0, neginf=0),
                order='ansi',
                lam_detection=wavelength
            )

            if plot:
                task = partial(
                    vis.diagnosis,
                    pred=wavefronts[cluster],
                    pred_std=pred_std,
                    save_path=Path(f"{savepath}_{postfix}_{cluster}_diagnosis"),
                )
                pool.apply_async(task)

            coefficients[cluster] = wavefronts[cluster].amplitudes

            actuators[cluster] = utils.zernikies_to_actuators(
                wavefronts[cluster].amplitudes,
                dm_calibration=dm_calibration,
                dm_state=dm_state,
                scalar=dm_damping_scalar
            )

    imwrite(Path(f"{savepath}_{postfix}_wavefronts_montage.tif"),
            wavefronts_montage,
            resolution=(64, 64),
            compression='deflate',
            dtype=np.float32
    )

    coefficients = pd.DataFrame.from_dict(coefficients)
    coefficients.index.name = 'ansi'
    coefficients.to_csv(f"{savepath}_{postfix}_zernike_coefficients.csv")

    actuators = pd.DataFrame.from_dict(actuators)
    actuators.index.name = 'actuators'
    actuators.to_csv(f"{savepath}_{postfix}_corrected_actuators.csv")
    logger.info(f"Saved {savepath}_{postfix}_corrected_actuators.csv")
    logger.info(f"with _corrected_actuators for :\ncluster  um_rms sum\n{coefficients.sum().round(3).to_string()}")

    return predictions, stdevs, coefficients


def color_clusters(
    heatmap,
    labels,
    savepath,
    xw,
    yw,
    colormap,
):
    scaled_heatmap = (heatmap - np.nanpercentile(heatmap[heatmap > 0], 1)) / \
                     (np.nanpercentile(heatmap[heatmap > 0], 99) - np.nanpercentile(heatmap[heatmap > 0], 1))
    scaled_heatmap = np.clip(scaled_heatmap, a_min=0, a_max=1)  # this helps see the volume data in _clusters.tif

    # scaled_heatmap = heatmap-np.min(heatmap) / (np.max(heatmap)-np.min(heatmap))

    rgb_map = colormap[labels.astype(np.ubyte)] * scaled_heatmap[..., np.newaxis]
    imwrite(
        savepath,
        rgb_map.astype(np.ubyte),
        photometric='rgb',
        resolution=(xw, yw),
        metadata={'axes': 'ZYXS'},
        compression='deflate',
    )
    logger.info(f'Saved {savepath}')


@profile
def aggregate_predictions(
    model_pred: Path,       # predictions  _tiles_predictions.csv
    dm_calibration: Path,
    dm_state: Any,
    majority_threshold: float = .5,
    min_percentile: int = 1,
    max_percentile: int = 99,
    prediction_threshold: float = 0.25,  # peak to valley in waves. you already have this diffraction limited data
    aggregation_rule: str = 'mean',     # metric to use to combine wavefronts of all tiles in a given cluster
    dm_damping_scalar: float = 1,
    max_isoplanatic_clusters: int = 3,
    optimize_max_isoplanatic_clusters: bool = False,
    plot: bool = False,
    ignore_tile: Any = None,
    clusters3d_colormap: str = 'tab20',
    zero_confident_color: tuple = (255, 255, 0),
    unconfident_color: tuple = (255, 255, 255),
    preloaded: Preloadedmodelclass = None,
    psf_type: Optional[Union[str, Path]] = None,
    postfix: str = 'aggregated'
):
    dm_state = utils.load_dm(dm_state)

    pd.options.display.width = 200
    pd.options.display.max_columns = 20

    vol = backend.load_sample(str(model_pred).replace('_tiles_predictions.csv', '.tif'))
    vol -= np.percentile(vol, 5)
    vol /= np.percentile(vol, 98)
    vol = np.clip(vol, 0, 1)

    pool = mp.Pool(processes=4)  # async pool for plotting

    with open(str(model_pred).replace('.csv', '_settings.json')) as f:
        predictions_settings = ujson.load(f)

    if vol.shape != tuple(predictions_settings['input_shape']):
        logger.error(f"vol.shape {vol.shape} != json's input_shape {tuple(predictions_settings['input_shape'])}")

    wavelength = predictions_settings['wavelength']
    axial_voxel_size = predictions_settings['sample_voxel_size'][0]
    lateral_voxel_size = predictions_settings['sample_voxel_size'][2]
    window_size = predictions_settings['window_size']

    samplepsfgen = SyntheticPSF(
        psf_type=predictions_settings['psf_type'] if psf_type is None else psf_type,
        psf_shape=window_size,
        lam_detection=wavelength,
        x_voxel_size=lateral_voxel_size,
        y_voxel_size=lateral_voxel_size,
        z_voxel_size=axial_voxel_size
    )

    # predict_snr_map(
    #     Path(str(model_pred).replace('_tiles_predictions.csv', '.tif')),
    #     window_size=window_size
    # )

    # tile id is the column header, rows are the predictions
    predictions, wavefronts = utils.create_multiindex_tile_dataframe(model_pred, return_wavefronts=True, describe=True)
    stdevs = utils.create_multiindex_tile_dataframe(str(model_pred).replace('_predictions.csv', '_stdevs.csv'))

    try:
        assert predictions_settings['ignore_modes']
    except KeyError:
        predictions_settings['ignore_modes'] = [0, 1, 2, 4]

    unconfident_tiles, zero_confident_tiles, all_zeros_tiles = utils.get_tile_confidence(
        predictions=predictions,
        stdevs=stdevs,
        prediction_threshold=prediction_threshold,
        ignore_tile=ignore_tile,
        ignore_modes=predictions_settings['ignore_modes'],
        verbose=True
    )

    where_unconfident = stdevs == 0
    where_unconfident[predictions_settings['ignore_modes']] = False

    ztiles = predictions.index.get_level_values('z').unique().shape[0]
    ytiles = predictions.index.get_level_values('y').unique().shape[0]
    xtiles = predictions.index.get_level_values('x').unique().shape[0]

    number_of_nonzero_tiles = np.array([(~(unconfident_tiles.loc[z] | zero_confident_tiles.loc[z] | all_zeros_tiles.loc[z])).sum() for z in range(ztiles)])

    errormapdf = predictions['p2v'].copy()
    nn_coords = np.array(errormapdf[~unconfident_tiles].index.to_list())
    nn_values = errormapdf[~unconfident_tiles].values
    try:
        myInterpolator = NearestNDInterpolator(nn_coords, nn_values)
        errormap = myInterpolator(np.array(errormapdf.index.to_list()))  # value for every tile
        errormap = np.reshape(errormap, (ztiles, ytiles, xtiles))  # back to 3d arrays
    except ValueError:
        logger.warning(f'Not much we can interpolate with here. {nn_coords=}')
        errormap = np.zeros((ztiles, ytiles, xtiles))  # back to 3d arrays, zero for every tile
    errormap = resize(errormap, (ztiles, vol.shape[1], vol.shape[2]),  order=1, mode='edge')  # linear interp XY
    errormap = resize(errormap, vol.shape,  order=0, mode='edge')   # nearest neighbor for z
    # errormap = resize(errormap, volume_shape, order=0, mode='constant')  # to show tiles
    imwrite(Path(f"{model_pred.with_suffix('')}_{postfix}_p2v_error.tif"), errormap.astype(np.float32), compression='deflate', dtype=np.float32)

    cluster_colors = np.split(
        np.array(sns.color_palette(clusters3d_colormap, n_colors=(max_isoplanatic_clusters * ztiles)))*255,
        ztiles,
    )   # list of colors for each z tiles

    clusters3d_colormap = []
    for cc in cluster_colors:  # for each z tile's colors
        clusters3d_colormap.extend([zero_confident_color, *cc])  # append the same zero color (e.g. yellow) at the front
    clusters3d_colormap.extend([unconfident_color])  # append the unconfident color (e.g. white) to the end
    clusters3d_colormap = np.array(clusters3d_colormap)  # yellow, blue, orange,...  yellow, ...  white

    predictions, stdevs, corrections = cluster_tiles(
        predictions=predictions,
        stdevs=stdevs,
        where_unconfident=where_unconfident,
        dm_calibration=dm_calibration,
        dm_state=dm_state,
        savepath=model_pred.with_suffix(''),
        dm_damping_scalar=dm_damping_scalar,
        wavelength=wavelength,
        aggregation_rule=aggregation_rule,
        max_isoplanatic_clusters=max_isoplanatic_clusters,
        optimize_max_isoplanatic_clusters=optimize_max_isoplanatic_clusters,
        plot=plot,
        postfix=postfix,
        minimum_number_of_tiles_per_cluster=np.maximum(np.minimum(number_of_nonzero_tiles * 0.09, 3).astype(int), 1), # 3 or less tiles
    )

    for z in range(ztiles):
        # create a mask to get the indices for each z tile and set the mask for the rest of the tiles to False
        zmask = all_zeros_tiles.mask(all_zeros_tiles.index.get_level_values(0) != z).fillna(False)

        predictions.loc[zmask, 'cluster'] = z * (max_isoplanatic_clusters + 1)
        stdevs.loc[zmask, 'cluster'] = z * (max_isoplanatic_clusters + 1)

        predictions.loc[zmask, 'cluster'] = z * (max_isoplanatic_clusters + 1)
        stdevs.loc[zmask, 'cluster'] = z * (max_isoplanatic_clusters + 1)

    # assign unconfident cluster id to last one
    predictions.loc[unconfident_tiles, 'cluster'] = len(clusters3d_colormap) - 1
    stdevs.loc[unconfident_tiles, 'cluster'] = len(clusters3d_colormap) - 1

    predictions.to_csv(f"{model_pred.with_suffix('')}_{postfix}_clusters.csv") # e.g. clusterids: [0,1,2,3, 4,5,6,7, 8] 8 is unconfident

    clusters_rgb = np.full((ztiles, *vol.shape[1:]), len(clusters3d_colormap)-1, dtype=np.float32)
    clusters3d_heatmap = np.full_like(vol, len(clusters3d_colormap)-1, dtype=np.float32)
    wavefront_heatmap = np.zeros((ztiles, *vol.shape[1:]), dtype=np.float32)
    expected_wavefront_heatmap = np.zeros((ztiles, *vol.shape[1:]), dtype=np.float32)
    psf_heatmap = np.zeros((ztiles, *vol.shape[1:]), dtype=np.float32)
    expected_psf_heatmap = np.zeros((ztiles, *vol.shape[1:]), dtype=np.float32)

    zw, yw, xw = predictions_settings['window_size']
    logger.info(f"volume_size = {vol.shape}")
    logger.info(f"window_size = {zw, yw, xw}")
    logger.info(f"      tiles = {ztiles, ytiles, xtiles}")

    for i, (z, y, x) in enumerate(itertools.product(range(ztiles), range(ytiles), range(xtiles))):
        c = predictions.loc[(z, y, x), 'cluster']
        if not np.isnan(c):
            clusters_rgb[z, y*yw:(y*yw)+yw, x*xw:(x*xw)+xw] = np.full((yw, xw), int(c))    # cluster group id

            if c == len(clusters3d_colormap)-1:     # last code (e.g. 8) = unconfident gray
                wavefront_heatmap[z, y*yw:(y*yw)+yw, x*xw:(x*xw)+xw] = np.zeros((yw, xw))
                expected_wavefront_heatmap[z, y*yw:(y*yw)+yw, x*xw:(x*xw)+xw] = np.zeros((yw, xw))

                psf_heatmap[z, y*yw:(y*yw)+yw, x*xw:(x*xw)+xw] = np.zeros((yw, xw))
                expected_psf_heatmap[z, y*yw:(y*yw)+yw, x*xw:(x*xw)+xw] = np.zeros((yw, xw))
            else:   # gets a color
                w = wavefronts[(z, y, x)]

                if c != 0:
                    expected_w = Wavefront(
                        w.amplitudes_ansi - corrections[f"z{z}_c{int(c)}"].values,
                        lam_detection=wavelength
                    )
                else:
                    expected_w = w


                abberated_psf = samplepsfgen.single_psf(w)
                abberated_psf *= np.sum(samplepsfgen.ipsf)/np.sum(abberated_psf)
                expected_psf = samplepsfgen.single_psf(expected_w)
                expected_psf *= np.sum(samplepsfgen.ipsf)/np.sum(abberated_psf)

                wavefront_heatmap[
                    z, y*yw:(y*yw)+yw, x*xw:(x*xw)+xw
                ] = np.nan_to_num(w.wave(xw), nan=0)

                expected_wavefront_heatmap[
                    z, y*yw:(y*yw)+yw, x*xw:(x*xw)+xw
                ] = np.nan_to_num(expected_w.wave(xw), nan=0)

                psf_heatmap[
                    z, y*yw:(y*yw)+yw, x*xw:(x*xw)+xw
                ] = np.max(abberated_psf, axis=0)

                expected_psf_heatmap[
                    z, y*yw:(y*yw)+yw, x*xw:(x*xw)+xw
                ] = np.max(expected_psf, axis=0)

            clusters3d_heatmap[z*zw:(z*zw)+zw, y*yw:(y*yw)+yw, x*xw:(x*xw)+xw] = np.full((zw, yw, xw), int(c)) # filled with cluster id 0,1,2,3, 4,5,6,7, 8] 8 is unconfident, color gets assigned later

    imwrite(f"{model_pred.with_suffix('')}_{postfix}_wavefronts.tif", wavefront_heatmap.astype(np.float32), compression='deflate', dtype=np.float32)
    imwrite(f"{model_pred.with_suffix('')}_{postfix}_wavefronts_expected.tif", expected_wavefront_heatmap.astype(np.float32), compression='deflate', dtype=np.float32)
    imwrite(f"{model_pred.with_suffix('')}_{postfix}_psfs.tif", psf_heatmap.astype(np.float32), compression='deflate', dtype=np.float32)
    imwrite(f"{model_pred.with_suffix('')}_{postfix}_psfs_expected.tif", expected_psf_heatmap.astype(np.float32), compression='deflate', dtype=np.float32)

    color_clusters(
        vol,
        clusters3d_heatmap,
        savepath=f"{model_pred.with_suffix('')}_{postfix}_clusters.tif",
        xw=xw,
        yw=yw,
        colormap=clusters3d_colormap,
    )

    for name, heatmap in zip(
        ('wavefronts', 'wavefronts_expected', 'psfs', 'psfs_expected'),
        (wavefront_heatmap, expected_wavefront_heatmap, psf_heatmap, expected_psf_heatmap),
    ):
        color_clusters(
            heatmap,
            clusters_rgb,
            savepath=f"{model_pred.with_suffix('')}_{postfix}_clusters_{name}.tif",
            xw=xw,
            yw=yw,
            colormap=clusters3d_colormap,
        )

    # reconstruct_wavefront_error_landscape(
    #     wavefronts=wavefronts,
    #     xtiles=xtiles,
    #     ytiles=ytiles,
    #     ztiles=ztiles,
    #     image=vol,
    #     save_path=Path(f"{model_pred.with_suffix('')}_{postfix}_error_landscape.tif"),
    #     window_size=predictions_settings['window_size'],
    #     lateral_voxel_size=lateral_voxel_size,
    #     axial_voxel_size=axial_voxel_size,
    #     wavelength=wavelength,
    #     na=.9,
    #     tile_p2v=predictions['p2v'].values,
    # )

    # vis.plot_volume(
    #     vol=terrain3d,
    #     results=coefficients,
    #     window_size=predictions_settings['window_size'],
    #     dxy=lateral_voxel_size,
    #     dz=axial_voxel_size,
    #     save_path=f"{model_pred.with_suffix('')}_{postfix}_projections.svg",
    # )

    # vis.plot_isoplanatic_patchs(
    #     results=isoplanatic_patchs,
    #     clusters=clusters,
    #     save_path=f"{model_pred.with_suffix('')}_{postfix}_isoplanatic_patchs.svg"
    # )

    logger.info(f'Done. Waiting for plots to write for {model_pred.with_suffix("")}')
    pool.close()    # close the pool
    pool.join()     # wait for all tasks to complete

    non_zero_tiles = ~(unconfident_tiles | zero_confident_tiles | all_zeros_tiles)
    with Path(f"{model_pred.with_suffix('')}_{postfix}_settings.json").open('w') as f:
        json = dict(
            model=predictions_settings['model'],
            model_pred=str(model_pred),
            dm_calibration=str(dm_calibration),
            dm_state=list(dm_state),
            majority_threshold=float(majority_threshold),
            min_percentile=int(min_percentile),
            max_percentile=int(max_percentile),
            prediction_threshold=float(prediction_threshold),
            aggregation_rule=str(aggregation_rule),
            dm_damping_scalar=float(dm_damping_scalar),
            max_isoplanatic_clusters=int(max_isoplanatic_clusters),
            optimize_max_isoplanatic_clusters=bool(optimize_max_isoplanatic_clusters),
            ignore_tile=list(ignore_tile) if ignore_tile is not None else None,
            window_size=list(window_size),
            wavelength=float(wavelength),
            psf_type=samplepsfgen.psf_type,
            sample_voxel_size=[axial_voxel_size, lateral_voxel_size, lateral_voxel_size],
            ztiles=int(ztiles),
            ytiles=int(ytiles),
            xtiles=int(xtiles),
            input_shape=list(vol.shape),
            total_confident_zero_tiles=int(zero_confident_tiles.sum()),
            total_unconfident_tiles=int(unconfident_tiles.sum()),
            total_all_zeros_tiles=int(all_zeros_tiles.sum()),
            total_non_zero_tiles=int(non_zero_tiles.sum()),
            confident_zero_tiles=zero_confident_tiles.loc[zero_confident_tiles].index.to_list(),
            unconfident_tiles=unconfident_tiles.loc[unconfident_tiles].index.to_list(),
            all_zeros_tiles=all_zeros_tiles.loc[all_zeros_tiles].index.to_list(),
            non_zero_tiles=non_zero_tiles.loc[non_zero_tiles].index.to_list(),
        )
        ujson.dump(
            json,
            f,
            indent=4,
            sort_keys=False,
            ensure_ascii=False,
            escape_forward_slashes=False
        )

    return predictions


@profile
def create_consensus_map(
    org_cluster_map: pd.DataFrame,
    correction_scans: Union[list, np.ndarray],
    stack_preds: Union[list, np.ndarray],
    stack_stdevs: Union[list, np.ndarray],
    zernikes_on_mirror: pd.DataFrame,
    zernike_indices: np.ndarray,
    window_size: tuple,
    ztiles: int,
    ytiles: int,
    xtiles: int,
    new_zernikes_path: Path,
    new_stdevs_path: Path,
    consensus_stacks_path: Path, # .csv of optimized_stack_id
):
    """
        1. Build a consensus isoplanatic map of the wavefront aberrations for each tile.
        2. each tile's consensus is built from the predictions from the 4 stacks.
        3. arguments are won by the stack that was optimized for that tile. previous consensus map gives the cluster
        tile masks aka the location of the optimized tiles
        (e.g. the yellow mask for first, brown mask for next, dark green for next, light green for last)

            3b. if the optimized stack did not have an answer, use before image,
            if that doesn't have an answer, leave gray.
            3c. if the tile was gray in the before image, only let the other three vote on that tile
            if that tile neighbors one their "optimized" tiles.
            Then remaining argument winners are based on std dev, then snr.
            For the "gray in before image case" only, take each cluster tile mask
            (e.g. light green), dilate by one tile, then mask the predictions of that stack (e.g. "after_three"),
            before letting everyone vote.
            This will prevent a scan that was optimal from the upper left corner
            from voting on something in the bottom right corner.

            3d. If the tile is gray in all stacks, it stays gray in the consensus map
    """
    zw, yw, xw = window_size

    # for loop on every tile
    # make data frame with 15 rows, will add column for each tile as we go
    optimized_wavefronts = pd.DataFrame([], index=zernike_indices)
    consensus_predictions = pd.DataFrame([], index=zernike_indices)
    consensus_stdevs = pd.DataFrame([], index=zernike_indices)
    consensus_stacks = pd.DataFrame([], index=[0])
    optimized_volume = np.zeros_like(correction_scans[0])
    volume_used = np.zeros((ztiles, *optimized_volume.shape[1:]))

    unconfident_cluster_id = ztiles * len(correction_scans)
    org_cluster_array = np.reshape((org_cluster_map['cluster']).to_numpy(),
                                   [ztiles, ytiles, xtiles])  # 3D np array of cluster ids
    num_of_stacks = len(correction_scans)

    # get max estimated std per stack for each tile
    error = pd.concat([df[zernike_indices].max(axis=1) for df in stack_stdevs], axis=1)

    # becomes a binary mask of what tiles in the stacks can be used to argue for tiles that were gray before.
    org_cluster_arrays = np.array([org_cluster_array] * num_of_stacks)

    # 3x3 structuring element with connectivity 2
    struct2 = generate_binary_structure(2, 2)
    for stack in range(num_of_stacks):
        for z in range(ztiles):
            org_cluster_arrays[stack, z] = binary_dilation(
                (org_cluster_arrays[stack, z] - (z * num_of_stacks)) == stack,
                structure=struct2)  # does cluster id belong to this stack? Dilate in 2D z slab
        error[stack] = error[stack].mask(~np.reshape(org_cluster_arrays[stack], error.shape[0]).astype(bool))

    # pick best stack id based on the lowest std, and assign nan to tiles with no std predictions
    votes = error.replace(0, np.nan).idxmin(skipna=True, axis=1)

    # replace nans with -1 and covert to integers
    votes = votes.fillna(-1).astype(int)

    for i, (z, y, x) in enumerate(itertools.product(range(ztiles), range(ytiles), range(xtiles))):
        optimized_cluster_id = org_cluster_map.loc[(z, y, x), 'cluster'].astype(int)  # cluster group id

        # last code (e.g. 8) = unconfident gray. was gray in the "before" stack
        if optimized_cluster_id == unconfident_cluster_id:

            optimized_stack_id = votes.loc[z, y, x]

            # nobody has an optimized tile nor neighboring an optimized tile to this one. Leave unconfident gray.
            if optimized_stack_id == -1:
                # arbitrarily using first stack
                optimized_stack_id = 0
                current_zernikes = zernikes_on_mirror[f'z{z}_c{z * len(correction_scans)}'].values

            else:
                # figure out the cluster_id from stack_id
                optimized_cluster_id = optimized_stack_id + (z * len(correction_scans))
                current_zernikes = zernikes_on_mirror[f'z{z}_c{optimized_cluster_id}']
                newtile = Wavefront(stack_preds[optimized_stack_id].loc[(z, y, x)][zernike_indices].values)
                logger.info(
                    f'Before tile ({z:2}, {y:2}, {x:2}) was gray color now assigned wavefront from stack index {optimized_stack_id} ({newtile.peak2valley():.2f} p2v)'
                )

        else:  # before has a color, we took an optimized stack for this tile
            optimized_stack_id = optimized_cluster_id - (z * len(correction_scans))
            cluster_result_from_optimized_stack = stack_preds[optimized_stack_id].loc[(z, y, x), 'cluster'].astype(int)

            if cluster_result_from_optimized_stack == unconfident_cluster_id:  # optimized stack was gray
                # the optimized stack was expected to have a prediction here.
                # It doesn't.  So use result from the first stack
                # (which is most similar to our previous time point which made the prediction).
                # arbitrarily using first stack
                optimized_stack_id = 0
                current_zernikes = zernikes_on_mirror[f'z{z}_c{z * len(correction_scans)}'].values
            else:  # optimized stack has a confident prediction
                current_zernikes = zernikes_on_mirror[f'z{z}_c{optimized_cluster_id}']

        optimized_zernikes = stack_preds[optimized_stack_id].loc[(z, y, x)][zernike_indices].values

        consensus_tile = optimized_zernikes + current_zernikes  # current_zernikes came from before's clusters, that we used to take the optimized stack.  optimized_zernikes = new predictions on the optimized stack.
        consensus_stdev = stack_stdevs[optimized_stack_id].loc[(z, y, x)][zernike_indices].values

        optimized_volume[
            z*zw:(z*zw)+zw,
            y*yw:(y*yw)+yw,
            x*xw:(x*xw)+xw
        ] = correction_scans[optimized_stack_id][
            z*zw:(z*zw)+zw,
            y*yw:(y*yw)+yw,
            x*xw:(x*xw)+xw
        ]
        volume_used[z, y*yw:(y*yw)+yw, x*xw:(x*xw)+xw] = np.full((yw, xw), optimized_stack_id)

        # assign predicted modes to the consensus row (building a new column there at the same time)
        consensus_predictions[f'z{z}-y{y}-x{x}'] = consensus_tile
        consensus_stdevs[f'z{z}-y{y}-x{x}'] = consensus_stdev
        consensus_stacks[f'z{z}-y{y}-x{x}'] = optimized_stack_id
        optimized_wavefronts[f'z{z}-y{y}-x{x}'] = optimized_zernikes

    tile_names = consensus_predictions.columns.values

    optimized_wavefronts['mean'] = optimized_wavefronts[tile_names].mean(axis=1)
    optimized_wavefronts['median'] = optimized_wavefronts[tile_names].median(axis=1)
    optimized_wavefronts['min'] = optimized_wavefronts[tile_names].min(axis=1)
    optimized_wavefronts['max'] = optimized_wavefronts[tile_names].max(axis=1)
    optimized_wavefronts['std'] = optimized_wavefronts[tile_names].std(axis=1)
    optimized_wavefronts.index.name = 'ansi'
    optimized_wavefronts.to_csv(str(new_zernikes_path).replace('combined', 'optimized'))

    consensus_predictions['mean'] = consensus_predictions[tile_names].mean(axis=1)
    consensus_predictions['median'] = consensus_predictions[tile_names].median(axis=1)
    consensus_predictions['min'] = consensus_predictions[tile_names].min(axis=1)
    consensus_predictions['max'] = consensus_predictions[tile_names].max(axis=1)
    consensus_predictions['std'] = consensus_predictions[tile_names].std(axis=1)
    consensus_predictions.index.name = 'ansi'
    consensus_predictions.to_csv(new_zernikes_path)

    consensus_stdevs['mean'] = consensus_stdevs[tile_names].mean(axis=1)
    consensus_stdevs['median'] = consensus_stdevs[tile_names].median(axis=1)
    consensus_stdevs['min'] = consensus_stdevs[tile_names].min(axis=1)
    consensus_stdevs['max'] = consensus_stdevs[tile_names].max(axis=1)
    consensus_stdevs['std'] = consensus_stdevs[tile_names].std(axis=1)
    consensus_stdevs.index.name = 'ansi'
    consensus_stdevs.to_csv(new_stdevs_path)
    consensus_stdevs.to_csv(str(new_stdevs_path).replace('combined', 'optimized'))

    consensus_stacks.to_csv(consensus_stacks_path)
    return optimized_volume, volume_used

@profile
def combine_tiles(
    corrected_actuators_csv: Path,
    corrections: list,
    postfix: str = 'combined',
    consensus_postfix: str = 'consensus'
):
    """
    Combine tiles from several DM patterns based on cluster IDs
    Args:
        corrected_actuators_csv: either _tiles_predictions_aggregated_corrected_actuators.csv (0th iteration)
                                     or _corrected_cluster_actuators.csv (Nth iteration)

        corrections: a list of tuples (clusterid, path to _tiles_predictions_aggregated_p2v_error.tif for each scan taken with the given DM pattern)


        Build a consensus isoplanatic map of the wavefront aberrations for each tile.
        we need to be able to deduce native wavefront from a scan that had a DM applied
        (for now assume the DM was perfect and gave us the wavefront we asked for,
            if the clusters stay spatially the same, then this should successfully iterate the individual scans even
            if the DM isn't perfect).

        we select the best scan of each tile (based upon '_aggregated_p2v_error.tif'),
        assign that to 'indices' and (dealing with z_slabs, aka convert to cluster id) assign to 'tile_indices'
    """

    acts_on_mirror = pd.read_csv(
        corrected_actuators_csv,
        index_col=0,
        header=0
    )  # 'z0_c0 z0_c1	z0_c2	z0_c3	z1_c4	z1_c5	z1_c6	z1_c7

    zernikes_on_mirror = pd.read_csv(
        str(corrected_actuators_csv).replace('corrected_actuators.csv', 'zernike_coefficients.csv'),
        index_col=0,
        header=0
    )  # 'z0_c0     z0_c1	z0_c2	z0_c3	z1_c4	z1_c5	z1_c6	z1_c7

    org_cluster_map = pd.read_csv(
        str(corrected_actuators_csv).replace('corrected_actuators.csv', 'clusters.csv'),
        index_col=['z', 'y', 'x'],  # z, y, x are going to be the MultiIndex
        header=0
    )   # cluster ids, e.g. 0,1,2,3, 4,5,6,7, 8 is unconfident

    output_base_path = str(corrections[0]).replace('_tiles_predictions_aggregated_p2v_error.tif', '')

    with open(str(corrected_actuators_csv).replace('corrected_actuators.csv', 'settings.json')) as f:
        predictions_settings = ujson.load(f)

    ztiles = predictions_settings['ztiles']
    ytiles = predictions_settings['ytiles']
    xtiles = predictions_settings['xtiles']
    dm_calibration = predictions_settings['dm_calibration']
    model_path = Path(predictions_settings['model'])
    if not model_path.exists():
        # if cluster wrote this path, and windows is now running it.
        opticalaberrations_path = Path(__file__).parent.parent
        model_relative_path = model_path.parts[model_path.parts.index("opticalaberrations")+1:]
        model_path = opticalaberrations_path.joinpath(*model_relative_path)
        if not model_path.exists():
            # if windows wrote this path, and cluster is now running it.
            model_path = Path(re.sub(r".*/opticalaberrations", str(Path(__file__).parent.parent), predictions_settings['model']))

    psfgen = backend.load_metadata(model_path)

    n_modes = psfgen.n_modes
    zernike_indices = np.arange(n_modes)

    # regex needs four backslashes to indicate one
    dm_calibration = re.sub(pattern="\\\\", repl='/', string=dm_calibration)
    if Path(dm_calibration).is_file():
        pass
    else:
        dm_calibration = Path(__file__).parent / dm_calibration  # for some reason we are not in the src folder already
        if Path(dm_calibration).is_file():
            pass
        else:
            dm_calibration = Path(__file__).parent.parent / "calibration" / dm_calibration.parent.name / dm_calibration.name # for some reason we are not in the src folder already

    logger.info(f'dm_calibration file is {Path(dm_calibration).resolve()}')

    stack_preds = []  # build a list of prediction dataframes for each stack.
    stack_stdevs = []  # build a list of standard deviations dataframes for each stack.
    correction_scans = []
    correction_scans_b = []
    correction_scan_paths = []

    for t, path in tqdm(enumerate(corrections), desc='Loading corrections', file=sys.stdout, unit=' image files'):
        correction_base_path = str(path).replace('_tiles_predictions_aggregated_p2v_error.tif', '')
        correction_scan_paths.append(Path(f'{correction_base_path}.tif'))
        correction_scans.append(backend.load_sample(f'{correction_base_path}.tif'))
        try:
            correction_scans_b.append(backend.load_sample(utils.convert_path_to_other_cam(Path(f'{correction_base_path}.tif'))))
        except Exception:
            pass
        stack_preds.append(
            pd.read_csv(
                f'{correction_base_path}_tiles_predictions_aggregated_clusters.csv',
                index_col=['z', 'y', 'x'],  # z, y, x are going to be the MultiIndex
                header=0,
            )
        )   # cluster ids, e.g. 0,1,2,3, 4,5,6,7, 8 is unconfident
        stack_stdevs.append(utils.create_multiindex_tile_dataframe(f'{correction_base_path}_tiles_stdevs.csv'))

    # reverse corrections to get the base DM for the before stack
    if isinstance(predictions_settings['dm_state'], str):
        dm_state = utils.zernikies_to_actuators(
            -1 * zernikes_on_mirror[f'z0_c0'].values,
            dm_calibration=dm_calibration,
            dm_state=acts_on_mirror[f'z0_c0'].values,
        )
    else:
        dm_state = predictions_settings['dm_state']

    consensus_stacks_path = Path(f"{output_base_path}_{postfix}_tiles_predictions_stacks.csv")
    new_zernikes_path = Path(f"{output_base_path}_{postfix}_tiles_predictions.csv")
    new_stdevs_path = Path(f"{output_base_path}_{postfix}_tiles_stdevs.csv")

    # optimized_volume for Cam B
    if len(correction_scans_b) > 0:
        optimized_volume_b, volume_used = create_consensus_map(
            org_cluster_map=org_cluster_map,
            correction_scans=correction_scans_b,
            stack_preds=stack_preds,
            stack_stdevs=stack_stdevs,
            zernikes_on_mirror=zernikes_on_mirror,
            zernike_indices=zernike_indices,
            window_size=predictions_settings['window_size'],
            ztiles=ztiles,
            ytiles=ytiles,
            xtiles=xtiles,
            new_zernikes_path=new_zernikes_path,
            new_stdevs_path=new_stdevs_path,
            consensus_stacks_path=consensus_stacks_path,
        )
        output_base_path_b = utils.convert_path_to_other_cam(correction_scan_paths[0])
        imwrite(f"{output_base_path_b.with_suffix('')}_optimized.tif", optimized_volume_b.astype(np.float32), compression='deflate', dtype=np.float32)
        logger.info(f"{output_base_path_b.with_suffix('')}_optimized.tif")

    # optimized_volume for Cam A
    optimized_volume, volume_used = create_consensus_map(
        org_cluster_map=org_cluster_map,
        correction_scans=correction_scans,
        stack_preds=stack_preds,
        stack_stdevs=stack_stdevs,
        zernikes_on_mirror=zernikes_on_mirror,
        zernike_indices=zernike_indices,
        window_size=predictions_settings['window_size'],
        ztiles=ztiles,
        ytiles=ytiles,
        xtiles=xtiles,
        new_zernikes_path=new_zernikes_path,
        new_stdevs_path=new_stdevs_path,
        consensus_stacks_path=consensus_stacks_path,
    )
    imwrite(f"{output_base_path}_{postfix}_volume_used.tif", volume_used.astype(np.uint16), compression='deflate')
    imwrite(f"{output_base_path}_optimized.tif", optimized_volume.astype(np.float32), compression='deflate', dtype=np.float32)
    logger.info(f"{output_base_path}_optimized.tif")

    # aggregate consensus maps
    imwrite(f"{output_base_path}_{postfix}.tif", correction_scans[0].astype(np.float32), compression='deflate', dtype=np.float32)
    logger.info(f"{output_base_path}_{postfix}.tif")
    with Path(f"{output_base_path}_{postfix}_tiles_predictions_settings.json").open('w') as f:
        ujson.dump(
            predictions_settings,
            f,
            indent=4,
            sort_keys=False,
            ensure_ascii=False,
            escape_forward_slashes=False
        )

    print(f"\nAggregating {postfix} maps...\n")
    aggregate_predictions(
            model_pred=Path(f"{output_base_path}_{postfix}_tiles_predictions.csv"),
            dm_calibration=dm_calibration,
            dm_state=dm_state,
            majority_threshold=predictions_settings['majority_threshold'],
            min_percentile=predictions_settings['min_percentile'],
            max_percentile=predictions_settings['max_percentile'],
            prediction_threshold=predictions_settings['prediction_threshold'],
            aggregation_rule=predictions_settings['aggregation_rule'],
            max_isoplanatic_clusters=predictions_settings['max_isoplanatic_clusters'],
            ignore_tile=predictions_settings['ignore_tile'],
            postfix=consensus_postfix
    )

    # aggregate optimized maps
    with Path(f"{output_base_path}_optimized_tiles_predictions_settings.json").open('w') as f:
        ujson.dump(
            predictions_settings,
            f,
            indent=4,
            sort_keys=False,
            ensure_ascii=False,
            escape_forward_slashes=False
        )

    print("\nAggregating optimized maps...\n")
    aggregate_predictions(
            model_pred=Path(f"{output_base_path}_optimized_tiles_predictions.csv"),
            dm_calibration=dm_calibration,
            dm_state=dm_state,
            majority_threshold=predictions_settings['majority_threshold'],
            min_percentile=predictions_settings['min_percentile'],
            max_percentile=predictions_settings['max_percentile'],
            prediction_threshold=predictions_settings['prediction_threshold'],
            aggregation_rule=predictions_settings['aggregation_rule'],
            max_isoplanatic_clusters=predictions_settings['max_isoplanatic_clusters'],
            ignore_tile=predictions_settings['ignore_tile'],
            postfix=consensus_postfix
    )

    # used in LabVIEW
    new_acts_path = Path(f"{output_base_path}_{postfix}_tiles_predictions_{consensus_postfix}_corrected_actuators.csv")

    if not new_acts_path.exists():
        raise Exception(f'New actuators were not written to {new_acts_path}')

    logger.info(f"Org actuators: {corrected_actuators_csv}")
    logger.info(f"New actuators: {new_acts_path}")
    logger.info(f"New predictions: {new_zernikes_path}")
    logger.info(f"Columns: {acts_on_mirror.columns.values}")


@profile
def phase_retrieval(
    img: Path,
    num_modes: int,
    dm_calibration: Any,
    dm_state: Any,
    axial_voxel_size: float,
    lateral_voxel_size: float,
    wavelength: float = .510,
    dm_damping_scalar: float = 1,
    plot: bool = False,
    num_iterations: int = 150,
    ignore_modes: list = (0, 1, 2, 4),
    prediction_threshold: float = 0.0,
    use_pyotf_zernikes: bool = False,
    plot_otf_diagnosis: bool = True,
    RW_path: Path = Path(__file__).parent.parent / "calibration" / "PSF_RW_515em_128_128_101_100nmSteps_97nmXY.tif",
):

    try:
        import pyotf.pyotf.phaseretrieval as pr
        from pyotf.pyotf.utils import prep_data_for_PR, psqrt
        from pyotf.pyotf.zernike import osa2degrees
        from pyotf.pyotf.otf import HanserPSF, SheppardPSF
    except ImportError as e:
        logger.error(e)
        return -1

    dm_state = None if (dm_state is None or str(dm_state) == 'None') else dm_state

    data = np.int_(imread(img))
    crop_shape = [round_to_odd(dim_len - .1) for dim_len in data.shape]
    data = resize_with_crop_or_pad(data, crop_shape)    # make each dimension an odd number of voxels
    logger.info(f'Cropping from {data.shape} to {crop_shape}')

    psfgen = SyntheticPSF(
        psf_type='widefield',
        psf_shape=data.shape,
        n_modes=num_modes,
        lam_detection=wavelength,
        x_voxel_size=lateral_voxel_size,
        y_voxel_size=lateral_voxel_size,
        z_voxel_size=axial_voxel_size
    )

    psf = data / np.nanmax(data)
    otf = utils.fft(psf)
    otf = remove_interference_pattern(
        psf=psf,
        otf=otf,
        plot=f"{img.with_suffix('')}_phase_retrieval" if plot else None,
        max_num_peaks=1,
        windowing=False,
    )
    data = np.int_(utils.ifft(otf) * np.nanmax(data))

    params = dict(
        wl=psfgen.lam_detection,
        na=psfgen.na_detection,
        ni=psfgen.refractive_index,
        res=lateral_voxel_size,
        zres=axial_voxel_size,
    )   # all in microns

    logger.info("Starting phase retrieval iterations")
    data_prepped = prep_data_for_PR(np.flip(data, axis=0), multiplier=1.15)
    logger.info(f"Subtracted background of: {np.max(data) - np.max(data_prepped):0.2f} counts")

    try:
        data_prepped = cp.asarray(data_prepped)  # use GPU. Comment this line to use CPU.
    except Exception:
        logger.warning(f"No CUDA-capable device is detected. 'image' will be type {type(data_prepped)}")
    pr_result = pr.retrieve_phase(
        data_prepped,
        params,
        max_iters=num_iterations,
        pupil_tol=1e-5,
        mse_tol=0,
        phase_only=False
    )
    pupil = pr_result.phase / (2 * np.pi)  # convert radians to waves
    pupil[pupil != 0.] -= np.mean(pupil[pupil != 0.])   # remove a piston term by subtracting the mean of the pupil
    pr_result.phase = utils.waves2microns(pupil, wavelength=psfgen.lam_detection)  # convert waves to um before fitting.
    pr_result.fit_to_zernikes(num_modes-1, mapping=osa2degrees)  # pyotf's zernikes now in um rms
    pr_result.phase = pupil  # phase is now again in waves

    pupil[pupil == 0.] = np.nan # put NaN's outside of pupil
    pupil_path = Path(f"{img.with_suffix('')}_phase_retrieval_wavefront.tif")
    imwrite(pupil_path, cp.asnumpy(pupil))

    threshold = utils.waves2microns(prediction_threshold, wavelength=psfgen.lam_detection)
    ignore_modes = list(map(int, ignore_modes))

    if use_pyotf_zernikes:
        # use pyotf definition of zernikes and fit using them. I suspect m=0 modes have opposite sign to our definition.
        pred = np.zeros(num_modes)
        pred[1:] = cp.asnumpy(pr_result.zd_result.pcoefs)
        pred[ignore_modes] = 0.
        pred[np.abs(pred) <= threshold] = 0.
        pred = Wavefront(pred, modes=num_modes, order='ansi', lam_detection=wavelength)
    else:
        # use our definition of zernikes and fit using them
        pred = Wavefront(pupil_path, modes=num_modes, order='ansi', lam_detection=wavelength)

    # finding the error in the coeffs is difficult.  This makes the error bars zero.
    pred_std = Wavefront(np.zeros(num_modes), modes=num_modes, order='ansi', lam_detection=wavelength)

    coefficients = [
        {'n': z.n, 'm': z.m, 'amplitude': a}
        for z, a in pred.zernikes.items()
    ]

    coefficients = pd.DataFrame(coefficients, columns=['n', 'm', 'amplitude'])
    coefficients.index.name = 'ansi'
    coefficients.to_csv(f"{img.with_suffix('')}_phase_retrieval_zernike_coefficients.csv")

    if dm_calibration is not None:
        estimate_and_save_new_dm(
            savepath=Path(f"{img.with_suffix('')}_phase_retrieval_corrected_actuators.csv"),
            coefficients=coefficients['amplitude'].values,
            dm_calibration=dm_calibration,
            dm_state=dm_state,
            dm_damping_scalar=dm_damping_scalar
        )

    psf = psfgen.single_psf(pred, normed=True, )
    data_prepped = cp.asnumpy(data_prepped)
    pupil_mag = cp.asnumpy(pr_result.mag)
    imwrite(f"{img.with_suffix('')}_phase_retrieval_psf.tif", psf.astype(np.float32), compression='deflate', dtype=np.float32)
    imwrite(f"{img.with_suffix('')}_phase_retrieval_pupil_mag.tif", pupil_mag.astype(np.float32), compression='deflate', dtype=np.float32)
    imwrite(f"{img.with_suffix('')}_phase_retrieval_pupil_mag_kxx.tif", cp.asnumpy(pr_result.kxx).astype(np.float32), compression='deflate', dtype=np.float32)
    imwrite(f"{img.with_suffix('')}_phase_retrieval_pupil_mag_kyy.tif", cp.asnumpy(pr_result.kyy).astype(np.float32), compression='deflate', dtype=np.float32)

    if plot_otf_diagnosis:
        logger.info(f'Plotting OTF Diagnosis...')

        # common parameters for pyotf to generate PSFs
        kwargs = dict(
            wl=wavelength,
            na=psfgen.na_detection,
            ni=psfgen.refractive_index,
            res=lateral_voxel_size,
            size=data_prepped.shape[-1],
            zres=axial_voxel_size,
            zsize=data_prepped.shape[0],
            vec_corr="none",    # we will overwrite the pupil magnitude: Set to 'none' to match retrieve_phase()
            condition="none",   # we will overwrite the pupil magnitude: Set to 'none' to match retrieve_phase()
        )

        pupil_field = np.fft.ifftshift(pupil_mag.astype(complex))
        model = HanserPSF(**kwargs)
        model.apply_pupil(pupil_field)
        hanser_pupil = np.squeeze(model.PSFi)

        RW = imread(RW_path)
        RW = resize_with_crop_or_pad(RW, crop_shape)

        model_result = cp.asnumpy(pr_result.model.PSFi)     # direct from PR. Has pupil magnitude *and* phase.

        vis.otf_diagnosis(
            psfs=[data_prepped, hanser_pupil, RW],
            labels=["Experimental", "FT(Experimental Pupil)", "RW theory"],
            save_path=img.with_suffix(''),
            lateral_voxel_size=lateral_voxel_size,
            axial_voxel_size=axial_voxel_size,
            na_detection=psfgen.na_detection,
            lam_detection=psfgen.lam_detection,
            refractive_index=psfgen.refractive_index,
        )

    if plot:
        vis.diagnosis(
            pred=pred,
            pred_std=pred_std,
            save_path=Path(f"{img.with_suffix('')}_phase_retrieval_diagnosis"),
        )

        fig, axes = pr_result.plot()
        axes[0].set_title("Phase in waves")
        vis.savesvg(fig, Path(f"{img.with_suffix('')}_phase_retrieval_convergence.svg"))
        logger.info(f'Files saved to : {img.parent.resolve()}')

    return coefficients


def silence(enabled, stdout=None):
    if enabled:
        stdout = os.dup(1)  # silence
        silent = os.open(os.devnull, os.O_WRONLY)
        os.dup2(silent, 1)
    elif stdout is not None:
        os.dup2(stdout, 1)

    return stdout


def overlap_tile(volume_shape, tile_shape, border, target, tile_index):
    """
    Used to process tiles, where each tile is processed using an extra border.

    This function returns either:
     - The slice into the source array, which to pass to processing,
     - The slice to extract from the processed array, which can be copied into the dst.
     - The slice in the dst to write the data to.

    Args:
        volume_shape: tuple of the volume shape
        tile_shape: tuple of the tile (aka the window) shape
        border: int for extra voxels to pass to processing
        target: either 'src', 'extract', or 'dst'
        tile_index: the tuple of indices

    Returns:
        slice object that can get the view from the np.array.

    """
    tile_shape = np.array(tile_shape)
    n_dims = len(volume_shape)
    # zw, yw, xw = tile_shape
    # z, y, x = tile_index
    # ranges = (
    #         slice(max(z * zw - border, 0), min((z * zw) + zw + border, volume_shape[0])),
    #         slice(max(y * yw - border, 0), min((y * yw) + yw + border, volume_shape[1])),
    #         slice(max(x * xw - border, 0), min((x * xw) + xw + border, volume_shape[2])),
    #       )

    dst_beg = np.multiply(tile_shape, tile_index)
    dst_end = np.multiply(tile_shape, tile_index) + tile_shape
    dst_end = np.minimum(dst_end, volume_shape)     # don't go past the end of the volume
    tile_length = dst_end - dst_beg     # nominally this is "tile_size" if we didn't run past the end of volume

    src_beg = np.maximum(dst_beg - border, 0)
    src_end = np.minimum(dst_end + border, volume_shape)

    ext_beg = dst_beg - src_beg         # nominally this is 'border' if the overlap didn't run out of bounds
    ext_end = ext_beg + tile_length

    if target == 'src':
        ranges = tuple([slice(src_beg[d], src_end[d]) for d in range(n_dims)])

    elif target == 'extract':
        ranges = tuple([slice(ext_beg[d], ext_end[d]) for d in range(n_dims)])

    else: # target == 'dst'
        ranges = tuple([slice(dst_beg[d], dst_end[d]) for d in range(n_dims)])

    return ranges


def f_to_minimize(defocus: float,
                  w: Wavefront,
                  corrected_psf: np.ndarray,
                  samplepsfgen: SyntheticPSF) -> float:
    """

    Args:
        defocus: amount of lightsheet defocus in microns
        w: wavefront aberration
        corrected_psf: corrected empirical psf to match (3D array)
        samplepsfgen: PSF generator (has voxel sizes, wavelength, etc..) which will make the defocus'ed 3D PSF.

    Returns:
        max correlation amount between 'corrected_psf' and 'defocused psf'

    """
    if isinstance(defocus, np.ndarray):
        defocus = defocus.item()
    kernel = samplepsfgen.single_psf(w, normed=True, lls_defocus_offset=defocus)
    # kernel /= np.max(kernel)
    return np.max(correlate(corrected_psf, kernel, mode='same'))


@profile
def decon(
    model_pred: Path,       # predictions  _tiles_predictions.csv
    iters: int = 10,
    prediction_threshold: float = 0.25,  # peak to valley in waves. you already have this diffraction limited data
    plot: bool = False,
    ignore_tile: Any = None,
    decon_tile: bool = False,   # Decon each tile individually if True, otherwise decon whole volume and extract tiles.
    preloaded: Preloadedmodelclass = None,
    only_use_ideal_psf: bool = False,    # Don't use psf from predictions.
    task: str = 'decon',    # 'decon' or 'cocoa'
):
    if only_use_ideal_psf:
        savepath = Path(f"{model_pred.with_suffix('')}_ideal_{task}.tif")
    else:
        savepath = Path(f"{model_pred.with_suffix('')}_{task}.tif")

    logger.info(f"Decon image will be saved to : \n{savepath.resolve()}")

    pd.options.display.width = 200
    pd.options.display.max_columns = 20

    with open(str(model_pred).replace('.csv', '_settings.json')) as f:
        predictions_settings = ujson.load(f)

    wavelength = predictions_settings['wavelength']
    axial_voxel_size = predictions_settings['sample_voxel_size'][0]
    lateral_voxel_size = predictions_settings['sample_voxel_size'][2]
    window_size = predictions_settings['window_size']

    psf_voxel_size = np.array([0.03, 0.03, 0.03])
    tile_fov = np.array(window_size) * (axial_voxel_size, lateral_voxel_size, lateral_voxel_size)
    psf_shape = np.full(shape=3, fill_value=round_to_even(np.min(tile_fov / psf_voxel_size) * 0.5), dtype=np.int32)
    print(f"  {axial_voxel_size=:0.03f}\n"
          f"{lateral_voxel_size=:0.03f}\n"
          f"         psf_shape={psf_shape}\n"
          f"    psf_voxel_size={psf_voxel_size}um\n"
          f"          tile_fov={tile_fov}um\n")
    samplepsfgen = SyntheticPSF(
        psf_type=predictions_settings['psf_type'],
        psf_shape=psf_shape,
        lam_detection=wavelength,
        x_voxel_size=psf_voxel_size[0],
        y_voxel_size=psf_voxel_size[1],
        z_voxel_size=psf_voxel_size[2],
    )

    # tile id is the column header, rows are the predictions
    predictions, wavefronts = utils.create_multiindex_tile_dataframe(model_pred, return_wavefronts=True, describe=True)
    stdevs = utils.create_multiindex_tile_dataframe(str(model_pred).replace('_predictions.csv', '_stdevs.csv'))
    if only_use_ideal_psf:
        predictions *= 0
        stdevs *= 0

    try:
        assert predictions_settings['ignore_modes']
    except KeyError:
        predictions_settings['ignore_modes'] = [0, 1, 2, 4]

    unconfident_tiles, zero_confident_tiles, all_zeros_tiles = utils.get_tile_confidence(
        predictions=predictions,
        stdevs=stdevs,
        prediction_threshold=prediction_threshold,
        ignore_tile=ignore_tile,
        ignore_modes=predictions_settings['ignore_modes'],
        verbose=True
    )

    where_unconfident = stdevs == 0
    where_unconfident[predictions_settings['ignore_modes']] = False

    ztiles = predictions.index.get_level_values('z').unique().shape[0]
    ytiles = predictions.index.get_level_values('y').unique().shape[0]
    xtiles = predictions.index.get_level_values('x').unique().shape[0]

    vol = backend.load_sample(str(model_pred).replace('_tiles_predictions.csv', '.tif'))

    if vol.shape != tuple(predictions_settings['input_shape']):
        logger.error(f"vol.shape {vol.shape} != json's input_shape {tuple(predictions_settings['input_shape'])}")

    decon_vol = vol.copy()
    est_vol   = vol.copy()
    imwrite(savepath, decon_vol.astype(np.float32), compression='deflate', dtype=np.float32)

    psfs = np.zeros(
        (ztiles*samplepsfgen.psf_shape[0], ytiles*samplepsfgen.psf_shape[1], xtiles*samplepsfgen.psf_shape[2]),
        dtype=np.float32
    )

    zw, yw, xw = window_size
    kzw, kyw, kxw = samplepsfgen.psf_shape

    logger.info(f"volume_size = {vol.shape}")
    logger.info(f"window_size = {zw, yw, xw}")
    logger.info(f"      tiles = {ztiles, ytiles, xtiles}")

    border = 32     # cudadecon likes sizes to be powers of 2, otherwise it may return a different size than input.

    tile_slice = partial(
        overlap_tile,
        volume_shape=vol.shape,
        tile_shape=window_size,
        border=border
    )

    if task == 'decon':
        reconstruct_decon = partial(
            cuda_decon,
            dzdata=axial_voxel_size,
            dxdata=lateral_voxel_size,
            dzpsf=samplepsfgen.z_voxel_size,
            dxpsf=samplepsfgen.x_voxel_size,
            n_iters=iters,
            skewed_decon=True,
            deskew=0,
            na=samplepsfgen.na_detection,
            background=100,
            wavelength=samplepsfgen.lam_detection * 1000,    # wavelength in nm
            nimm=samplepsfgen.refractive_index,
            cleanup_otf=False,
            dup_rev_z=True,
            )
    elif task == 'cocoa':
        from experimental_benchmarks import predict_cocoa
        reconstruct_cocoa = partial(
            predict_cocoa,
            plot=False,
            axial_voxel_size=axial_voxel_size,
            lateral_voxel_size=lateral_voxel_size,
            na_detection=samplepsfgen.na_detection,
            lam_detection=samplepsfgen.lam_detection,
            refractive_index=samplepsfgen.refractive_index,
            psf_type=samplepsfgen.psf_type,
            decon=False,
        )
    else:
        logger.error(f'Invalid task given: {task=}')
        reconstruct = None

    if decon_tile:
        # decon the tiles independently
        for i, (z, y, x) in tqdm(
            enumerate(itertools.product(range(ztiles), range(ytiles), range(xtiles))),
            total=ztiles*ytiles*xtiles,
            desc=f'{task} tiles',
            unit=f'tiles',
            position=0
        ):
            w = Wavefront(predictions.loc[z, y, x].values, lam_detection=wavelength)

            kernel = samplepsfgen.single_psf(w, normed=False)
            kernel /= np.max(kernel)

            tile = vol[tile_slice(target='src', tile_index=(z, y, x))]

            if task == 'cocoa':
                out_y, reconstructed, out_k_m = reconstruct_cocoa(tile)

                est_vol[tile_slice(target='dst', tile_index=(z, y, x))
                ] = out_y[tile_slice(target='extract', tile_index=(z, y, x))]
                imwrite(f"{savepath.with_suffix('')}_estimated.tif", est_vol, compression='deflate', dtype=np.float32)

                kernel = out_k_m
            elif task == 'decon':
                # stdout = silence(task == 'decon')
                reconstructed = reconstruct_decon(tile, psf=kernel)
                # silence(False, stdout=stdout)
            else:
                logger.error(f"Task of '{task}' is unknown")
                return

            decon_vol[tile_slice(target='dst', tile_index=(z, y, x))
            ] = reconstructed[tile_slice(target='extract', tile_index=(z, y, x))]

            psfs[   z * kzw:(z * kzw) + kzw,
                    y * kyw:(y * kyw) + kyw,
                    x * kxw:(x * kxw) + kxw] = kernel[0:kzw, 0:kyw, 0:kxw]

            imwrite(f"{model_pred.with_suffix('')}_{task}_psfs.tif", psfs.astype(np.float32), resolution=(xw, yw), compression='deflate', dtype=np.float32)
            imwrite(savepath, decon_vol, compression='deflate', dtype=np.float32)
    else:
        # identify all the unique PSFs that we need to deconvolve with
        predictions['psf_id'] = predictions.groupby(predictions.columns.values.tolist(), sort=False).grouper.group_info[0]
        groups = predictions.groupby('psf_id')

        # for each psf_id, deconvolve the entire volume
        for psf_id in tqdm(predictions['psf_id'].unique(), desc=f'Do {task} entire vol with each psf, {iters} RL iterations', unit='vols to decon', position=0):
            df = groups.get_group(psf_id).drop(columns=['p2v', 'psf_id'])

            zernikes = df.values[0]  # all rows in this group should be equal. Take the first one as the wavefront.
            w = Wavefront(zernikes, lam_detection=wavelength)
            z, y, x = df.index[0]

            if task == 'cocoa':
                deconv = reconstruct_cocoa(vol)

            elif task == 'decon':
                defocus = 0
                if np.count_nonzero(zernikes) > 0:
                    corrected_psf_path = Path(
                        str(model_pred / f'z{z}-y{y}-x{x}_corrected_psf.tif').replace("_predictions.csv", ""))
                    corrected_psf = np.zeros_like(imread(corrected_psf_path))

                    for index, zernikes in df.iterrows():
                        z, y, x = index
                        corrected_psf_path = Path(str(model_pred / f'z{z}-y{y}-x{x}_corrected_psf.tif').replace("_predictions.csv", ""))
                        corrected_psf += imread(corrected_psf_path)

                    corrected_psf /= np.max(corrected_psf)
                    res = minimize(
                        f_to_minimize,
                        x0=0,
                        args=(w, corrected_psf, samplepsfgen),
                        tol=samplepsfgen.z_voxel_size,
                        bounds=[(-0.6, 0.6)],
                        method='Nelder-Mead',
                        options={
                            'disp': False,
                            'initial_simplex': [[-samplepsfgen.z_voxel_size], [samplepsfgen.z_voxel_size]],   # need 1st step > one z, overrides x0
                            }
                        )
                    defocus = res.x[0]
                    # defocus_steps = np.linspace(-2, 2, 41)
                    # correlations = np.zeros_like(defocus_steps)
                    #
                    # for i, defocus in enumerate(defocus_steps):
                    #     kernel = samplepsfgen.single_psf(w, normed=False, lls_defocus_offset=defocus)
                    #     kernel /= np.max(kernel)
                    #     correlations[i] = np.max(correlate(corrected_psf, kernel, mode='same'))
                    #
                    # defocus = defocus_steps[np.argmax(correlations)]
                print(f'\t Defocus for ({z:2d}, {y:2d}, {x:2d}) is {defocus: 0.2f} um, p2V is {w.peak2valley(na=samplepsfgen.na_detection):0.02f}')
                kernel = samplepsfgen.single_psf(w, normed=False, lls_defocus_offset=defocus)
                kernel /= np.max(kernel)

                stdout = silence(task == 'decon')
                deconv = reconstruct_decon(vol, psf=kernel)
                silence(False, stdout=stdout)

            else:
                logger.error(f"Task of '{task}' is unknown")
                return

            for index, zernikes in df.iterrows():
                z, y, x = index
                decon_vol[
                    z * zw :(z * zw) + zw,
                    y * yw :(y * yw) + yw,
                    x * xw :(x * xw) + xw,
                ] = deconv[
                    z * zw :(z * zw) + zw,
                    y * yw :(y * yw) + yw,
                    x * xw :(x * xw) + xw,
                ]
                psfs[
                    z * kzw:(z * kzw) + kzw,
                    y * kyw:(y * kyw) + kyw,
                    x * kxw:(x * kxw) + kxw] = kernel[0:kzw, 0:kyw, 0:kxw]

            imwrite(savepath, decon_vol.astype(np.float32), resolution=(xw, yw))
            imwrite(
                f"{model_pred.with_suffix('')}_{task}_psfs.tif", psfs.astype(np.float32),
                resolution=(psf_shape[1], psf_shape[2]),
                compression='deflate',
                dtype=np.float32
            )

    imwrite(savepath, decon_vol.astype(np.float32), compression='deflate', dtype=np.float32)
    logger.info(f"Decon image saved to : \n{savepath.resolve()}")



    with Path(f"{model_pred.with_suffix('')}_{task}_settings.json").open('w') as f:
        json = dict(
            model=predictions_settings['model'],
            model_pred=str(model_pred),
            iters=int(iters),
            prediction_threshold=float(prediction_threshold),
            ignore_tile=list(ignore_tile) if ignore_tile is not None else None,
            window_size=list(window_size),
            wavelength=float(wavelength),
            psf_type=samplepsfgen.psf_type,
            sample_voxel_size=[axial_voxel_size, lateral_voxel_size, lateral_voxel_size],
            ztiles=int(ztiles),
            ytiles=int(ytiles),
            xtiles=int(xtiles),
            input_shape=list(vol.shape),
            total_confident_zero_tiles=int(zero_confident_tiles.sum()),
            total_unconfident_tiles=int(unconfident_tiles.sum()),
            total_all_zeros_tiles=int(all_zeros_tiles.sum()),
            confident_zero_tiles=zero_confident_tiles.loc[zero_confident_tiles].index.to_list(),
            unconfident_tiles=unconfident_tiles.loc[unconfident_tiles].index.to_list(),
            all_zeros_tiles=all_zeros_tiles.loc[all_zeros_tiles].index.to_list(),
        )
        ujson.dump(
            json,
            f,
            indent=4,
            sort_keys=False,
            ensure_ascii=False,
            escape_forward_slashes=False
        )
    return savepath

