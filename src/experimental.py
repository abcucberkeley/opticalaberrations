import matplotlib
matplotlib.use('Agg')

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
from tifffile import imread, imwrite
from line_profiler_pycharm import profile

import multiprocessing as mp
from sklearn.cluster import KMeans
from skimage.transform import resize
from sklearn.metrics import silhouette_samples, silhouette_score
from joblib import Parallel, delayed
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import shift

import utils
import vis
import backend

from synthetic import SyntheticPSF
from wavefront import Wavefront
from preloaded import Preloadedmodelclass
from embeddings import remove_interference_pattern
from preprocessing import prep_sample, optimal_rolling_strides, find_roi, get_tiles

import logging
logger = logging.getLogger('')

try:
    import cupy as cp
except ImportError as e:
    logging.warning(f"Cupy not supported on your system: {e}")


@profile
def reloadmodel_if_needed(
    modelpath: Path,
    preloaded: Optional[Preloadedmodelclass] = None,
    ideal_empirical_psf: Union[Path, np.ndarray] = None,
    ideal_empirical_psf_voxel_size: Any = None,
    n_modes: Optional[int] = None,
    psf_type: Optional[np.ndarray] = None
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
        preloaded.modelpsfgen.update_ideal_psf_with_empirical(
            ideal_empirical_psf=ideal_empirical_psf,
            voxel_size=ideal_empirical_psf_voxel_size,
            remove_background=True,
            normalize=True,
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
    edge_filter: bool = False,
    filter_mask_dilation: bool = True,
    plot: bool = False,
    fov_is_small: bool = True,
    preloaded: Preloadedmodelclass = None,
    ideal_empirical_psf: Any = None,
    digital_rotations: Optional[int] = None
):

    model, modelpsfgen = reloadmodel_if_needed(
        modelpath=model,
        preloaded=preloaded,
        ideal_empirical_psf=ideal_empirical_psf,
        ideal_empirical_psf_voxel_size=(axial_voxel_size, lateral_voxel_size, lateral_voxel_size)
    )

    sample = backend.load_sample(file)
    psnr = prep_sample(
        sample,
        return_psnr=True,
        remove_background=True,
        normalize=False,
        edge_filter=False,
        filter_mask_dilation=False,
    )
    logger.info(f"Sample: {sample.shape}")

    samplepsfgen = SyntheticPSF(
        psf_type=modelpsfgen.psf_type,
        psf_shape=sample.shape,
        n_modes=model.output_shape[1],
        lam_detection=wavelength,
        x_voxel_size=lateral_voxel_size,
        y_voxel_size=lateral_voxel_size,
        z_voxel_size=axial_voxel_size
    )

    sample = prep_sample(
        sample,
        sample_voxel_size=samplepsfgen.voxel_size,
        remove_background=remove_background,
        normalize=normalize,
        edge_filter=edge_filter,
        filter_mask_dilation=filter_mask_dilation,
        read_noise_bias=read_noise_bias,
        plot=file.with_suffix('') if plot else None,
    )

    return backend.preprocess(
        sample,
        modelpsfgen=modelpsfgen,
        samplepsfgen=samplepsfgen,
        freq_strength_threshold=freq_strength_threshold,
        remove_background=True,
        normalize=True,
        edge_filter=False,
        fov_is_small=fov_is_small,
        digital_rotations=digital_rotations,
        plot=file.with_suffix('') if plot else None,
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
    lateral_voxel_size: float = .108,
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
    imwrite(save_path, terrain3d, photometric='rgb')

    return terrain3d


@profile
def predict(
    rois: np.ndarray,
    outdir: Path,
    model: tf.keras.Model,
    modelpsfgen: SyntheticPSF,
    samplepsfgen: SyntheticPSF,
    dm_calibration: Any,
    dm_state: Any,
    wavelength: float = .510,
    ignore_modes: list = (0, 1, 2, 4),
    prediction_threshold: float = 0.,
    freq_strength_threshold: float = .01,
    confidence_threshold: float = .02,
    batch_size: int = 1,
    digital_rotations: Optional[int] = 361,
    rolling_strides: Optional[tuple] = None,
    fov_is_small: bool = True,
    plot: bool = True,
    plot_rotations: bool = False,
    cpu_workers: int = -1,
):
    no_phase = True if model.input_shape[1] == 3 else False

    generate_fourier_embeddings = partial(
        utils.multiprocess,
        func=partial(
            backend.preprocess,
            modelpsfgen=modelpsfgen,
            samplepsfgen=samplepsfgen,
            freq_strength_threshold=freq_strength_threshold,
            digital_rotations=digital_rotations,
            plot=plot,
            no_phase=no_phase,
            remove_background=True,
            normalize=True,
            edge_filter=False,
            filter_mask_dilation=True,
            fov_is_small=fov_is_small,
            rolling_strides=rolling_strides,
        ),
        desc='Generate Fourier embeddings',
        cores=cpu_workers
    )

    inputs = tf.data.Dataset.from_tensor_slices(np.vectorize(str)(rois))
    inputs = inputs.batch(batch_size).map(
        lambda x: tf.py_function(
            generate_fourier_embeddings,
            inp=[x],
            Tout=tf.float32,
        ),
    ).unbatch()

    preds, std = backend.predict_dataset(
        model,
        inputs=inputs,
        psfgen=modelpsfgen,
        batch_size=batch_size,
        ignore_modes=ignore_modes,
        threshold=prediction_threshold,
        save_path=[f.with_suffix('') for f in rois],
        plot_rotations=plot_rotations,
        digital_rotations=digital_rotations,
        confidence_threshold=confidence_threshold,
        desc=f"ROIs [{rois.shape[0]}] x [{digital_rotations}] Rotations",
    )

    tile_names = [f.with_suffix('').name for f in rois]
    predictions = pd.DataFrame(preds.T, columns=tile_names)
    predictions['mean'] = predictions[tile_names].mean(axis=1)
    predictions['median'] = predictions[tile_names].median(axis=1)
    predictions['min'] = predictions[tile_names].min(axis=1)
    predictions['max'] = predictions[tile_names].max(axis=1)
    predictions['std'] = predictions[tile_names].std(axis=1)
    predictions.index.name = 'ansi'
    predictions.to_csv(f"{outdir}_predictions.csv")

    stdevs = pd.DataFrame(std.T, columns=tile_names)
    stdevs['mean'] = stdevs[tile_names].mean(axis=1)
    stdevs['median'] = stdevs[tile_names].median(axis=1)
    stdevs['min'] = stdevs[tile_names].min(axis=1)
    stdevs['max'] = stdevs[tile_names].max(axis=1)
    stdevs['std'] = stdevs[tile_names].std(axis=1)
    stdevs.index.name = 'ansi'
    stdevs.to_csv(f"{outdir}_stdevs.csv")

    if dm_calibration is not None:
        actuators = {}

        for t in tile_names:
            actuators[t] = utils.zernikies_to_actuators(
                predictions[t].values,
                dm_calibration=dm_calibration,
                dm_state=dm_state,
            )

        actuators = pd.DataFrame.from_dict(actuators)
        actuators.index.name = 'actuators'
        actuators.to_csv(f"{outdir}_predictions_corrected_actuators.csv")

    return predictions


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
    verbose: bool = False,
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
    cpu_workers: int = -1
):
    lls_defocus = 0.
    dm_state = None if (dm_state is None or str(dm_state) == 'None') else dm_state

    preloadedmodel, premodelpsfgen = reloadmodel_if_needed(
        modelpath=model,
        preloaded=preloaded,
        ideal_empirical_psf=ideal_empirical_psf,
        ideal_empirical_psf_voxel_size=(axial_voxel_size, lateral_voxel_size, lateral_voxel_size)
    )
    no_phase = True if preloadedmodel.input_shape[1] == 3 else False

    logger.info(f"Loading file: {img.name}")
    sample = backend.load_sample(img)
    psnr = prep_sample(
        sample,
        return_psnr=True,
        remove_background=True,
        normalize=False,
        edge_filter=False,
        filter_mask_dilation=False,
    )
    logger.info(f"Sample: {sample.shape}")

    samplepsfgen = SyntheticPSF(
        psf_type=premodelpsfgen.psf_type,
        psf_shape=sample.shape,
        n_modes=preloadedmodel.output_shape[1],
        lam_detection=wavelength,
        x_voxel_size=lateral_voxel_size,
        y_voxel_size=lateral_voxel_size,
        z_voxel_size=axial_voxel_size
    )

    embeddings = backend.preprocess(
        sample,
        modelpsfgen=premodelpsfgen,
        samplepsfgen=samplepsfgen,
        digital_rotations=digital_rotations,
        remove_background=True,
        normalize=True,
        edge_filter=False,
        filter_mask_dilation=True,
        fov_is_small=True,
        plot=Path(f"{img.with_suffix('')}_sample_predictions") if plot else None,
    )

    if no_phase:
        p, std, pchange = backend.dual_stage_prediction(
            preloadedmodel,
            inputs=embeddings,
            threshold=prediction_threshold,
            sign_threshold=sign_threshold,
            n_samples=num_predictions,
            verbose=verbose,
            gen=samplepsfgen,
            modelgen=premodelpsfgen,
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
            psfgen=premodelpsfgen,
            no_phase=False,
            verbose=verbose,
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
    imwrite(f"{img.with_suffix('')}_sample_predictions_psf.tif", psf)
    imwrite(f"{img.with_suffix('')}_sample_predictions_wavefront.tif", p.wave(), dtype=np.float32)

    with Path(f"{img.with_suffix('')}_sample_predictions_settings.json").open('w') as f:
        json = dict(
            path=str(img),
            model=str(model),
            input_shape=list(sample.shape),
            sample_voxel_size=list([axial_voxel_size, lateral_voxel_size, lateral_voxel_size]),
            model_voxel_size=list(premodelpsfgen.voxel_size),
            psf_fov=list(premodelpsfgen.psf_fov),
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
            psnr=psnr
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
    verbose: bool = False,
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
    cpu_workers: int = -1
):
    lls_defocus = 0.
    dm_state = None if (dm_state is None or str(dm_state) == 'None') else dm_state
    sample_voxel_size = (axial_voxel_size, lateral_voxel_size, lateral_voxel_size)

    preloadedmodel, premodelpsfgen = reloadmodel_if_needed(
        modelpath=model,
        preloaded=preloaded,
        ideal_empirical_psf=ideal_empirical_psf,
        ideal_empirical_psf_voxel_size=sample_voxel_size
    )
    no_phase = True if preloadedmodel.input_shape[1] == 3 else False

    sample = backend.load_sample(img)
    psnr = prep_sample(
        sample,
        return_psnr=True,
        remove_background=True,
        normalize=False,
        edge_filter=False,
        filter_mask_dilation=False,
    )
    logger.info(f"Sample: {sample.shape}")

    samplepsfgen = SyntheticPSF(
        psf_type=premodelpsfgen.psf_type,
        psf_shape=sample.shape,
        n_modes=preloadedmodel.output_shape[1],
        lam_detection=wavelength,
        x_voxel_size=lateral_voxel_size,
        y_voxel_size=lateral_voxel_size,
        z_voxel_size=axial_voxel_size
    )

    embeddings = backend.preprocess(
        sample,
        modelpsfgen=premodelpsfgen,
        samplepsfgen=samplepsfgen,
        digital_rotations=digital_rotations,
        no_phase=no_phase,
        freq_strength_threshold=freq_strength_threshold,
        remove_background=True,
        normalize=True,
        edge_filter=False,
        fov_is_small=False,
        rolling_strides=optimal_rolling_strides(premodelpsfgen.psf_fov, sample_voxel_size, sample.shape),
        plot=Path(f"{img.with_suffix('')}_large_fov_predictions") if plot else None,
    )

    res = backend.predict_rotation(
        preloadedmodel,
        inputs=embeddings,
        psfgen=premodelpsfgen,
        no_phase=False,
        verbose=verbose,
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
    imwrite(f"{img.with_suffix('')}_large_fov_predictions_psf.tif", psf)
    imwrite(f"{img.with_suffix('')}_large_fov_predictions_wavefront.tif", p.wave(), dtype=np.float32)

    with Path(f"{img.with_suffix('')}_large_fov_predictions_settings.json").open('w') as f:
        json = dict(
            path=str(img),
            model=str(model),
            input_shape=list(sample.shape),
            sample_voxel_size=list([axial_voxel_size, lateral_voxel_size, lateral_voxel_size]),
            model_voxel_size=list(premodelpsfgen.voxel_size),
            psf_fov=list(premodelpsfgen.psf_fov),
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
            psnr=psnr,
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
    cpu_workers: int = -1
):

    preloadedmodel, premodelpsfgen = reloadmodel_if_needed(
        modelpath=model,
        preloaded=preloaded,
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
        psf_type=premodelpsfgen.psf_type,
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
            model_voxel_size=list(premodelpsfgen.voxel_size),
            psf_fov=list(premodelpsfgen.psf_fov),
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
        )

        ujson.dump(
            json,
            f,
            indent=4,
            sort_keys=False,
            ensure_ascii=False,
            escape_forward_slashes=False
        )

    predict(
        rois=rois,
        outdir=outdir,
        model=preloadedmodel,
        modelpsfgen=premodelpsfgen,
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
    rois, ztiles, nrows, ncols = get_tiles(
        sample,
        savepath=outdir,
        strides=window_size,
        window_size=window_size,
        save_files=save_files,
    )

    prep = partial(prep_sample, return_psnr=True)
    snrs = utils.multiprocess(func=prep, jobs=rois, desc=f'PNSR, {rois.shape[0]} rois per tile.')
    snrs = np.reshape(snrs, (ztiles, nrows, ncols))
    snrs = resize(snrs, (snrs.shape[0], sample.shape[1], sample.shape[2]), order=1, mode='edge')
    snrs = resize(snrs, sample.shape, order=0, mode='edge')
    imwrite(Path(f"{img.with_suffix('')}_snrs.tif"), snrs.astype(np.float32), dtype=np.float32)


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
    plot: bool = True,
    plot_rotations: bool = False,
    prev: Any = None,
    estimate_sign_with_decon: bool = False,
    ignore_modes: list = (0, 1, 2, 4),
    preloaded: Preloadedmodelclass = None,
    ideal_empirical_psf: Any = None,
    digital_rotations: Optional[int] = 361,
    cpu_workers: int = -1,
    shifting: tuple = (0, 0, 0)
):

    preloadedmodel, premodelpsfgen = reloadmodel_if_needed(
        modelpath=model,
        preloaded=preloaded,
        ideal_empirical_psf=ideal_empirical_psf,
        ideal_empirical_psf_voxel_size=(axial_voxel_size, lateral_voxel_size, lateral_voxel_size)
    )

    logger.info(f"Loading file: {img.name}")
    sample = backend.load_sample(img)
    logger.info(f"Sample: {sample.shape}")
    psnr = prep_sample(
        sample,
        return_psnr=True,
        remove_background=True,
        normalize=False,
        edge_filter=False,
        filter_mask_dilation=False,
    )

    if any(np.array(shifting) != 0):
        sample = shift(sample, shift=(-1*shifting[0], -1*shifting[1], -1*shifting[2]))
        img = Path(f"{img.with_suffix('')}_shifted_z{shifting[0]}_y{shifting[1]}_x{shifting[2]}.tif")
        imwrite(img, sample.astype(np.float32))

    outdir = Path(f"{img.with_suffix('')}_tiles")
    outdir.mkdir(exist_ok=True, parents=True)

    # obtain each tile and save to .tif.
    rois, ztiles, nrows, ncols = get_tiles(
        sample,
        savepath=outdir,
        strides=window_size,
        window_size=window_size,
    )

    samplepsfgen = SyntheticPSF(
        psf_type=premodelpsfgen.psf_type,
        psf_shape=window_size,
        n_modes=preloadedmodel.output_shape[1],
        lam_detection=wavelength,
        x_voxel_size=lateral_voxel_size,
        y_voxel_size=lateral_voxel_size,
        z_voxel_size=axial_voxel_size
    )

    with Path(f"{img.with_suffix('')}_tiles_predictions_settings.json").open('w') as f:
        json = dict(
            path=str(img),
            model=str(model),
            input_shape=list(sample.shape),
            sample_voxel_size=list(samplepsfgen.voxel_size),
            model_voxel_size=list(premodelpsfgen.voxel_size),
            psf_fov=list(premodelpsfgen.psf_fov),
            window_size=list(window_size),
            wavelength=float(wavelength),
            prediction_threshold=float(0),
            freq_strength_threshold=float(freq_strength_threshold),
            prev=str(prev),
            ignore_modes=list(ignore_modes),
            ideal_empirical_psf=str(ideal_empirical_psf),
            number_of_tiles=int(len(rois)),
            ztiles=int(ztiles),
            ytiles=int(nrows),
            xtiles=int(ncols),
            psnr=psnr,
        )

        ujson.dump(
            json,
            f,
            indent=4,
            sort_keys=False,
            ensure_ascii=False,
            escape_forward_slashes=False
        )

    predictions = predict(
        rois=rois,
        outdir=outdir,
        model=preloadedmodel,
        modelpsfgen=premodelpsfgen,
        samplepsfgen=samplepsfgen,
        dm_calibration=dm_calibration,
        dm_state=dm_state,
        prediction_threshold=0,
        confidence_threshold=confidence_threshold,
        batch_size=batch_size,
        wavelength=wavelength,
        ignore_modes=ignore_modes,
        freq_strength_threshold=freq_strength_threshold,
        fov_is_small=True if all(np.array(samplepsfgen.psf_fov) <= np.array(premodelpsfgen.psf_fov)) else False,
        plot=plot,
        plot_rotations=plot_rotations,
        digital_rotations=digital_rotations,
        rolling_strides=optimal_rolling_strides(premodelpsfgen.psf_fov, samplepsfgen.voxel_size, window_size),
        cpu_workers=cpu_workers,
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


@profile
def aggregate_predictions(
    model_pred: Path,
    dm_calibration: Path,
    dm_state: Any,
    majority_threshold: float = .5,
    min_percentile: int = 1,
    max_percentile: int = 99,
    prediction_threshold: float = 0.25, # peak to valley in waves. you already have this diffraction limited data
    aggregation_rule: str = 'mean',
    dm_damping_scalar: float = 1,
    max_isoplanatic_clusters: int = 3,
    optimize_max_isoplanatic_clusters: bool = False,
    plot: bool = False,
    ignore_tile: Any = None,
    clusters3d_colormap: str = 'tab10',
    zero_confident_color: tuple = (255, 255, 0),
    unconfident_color: tuple = (255, 255, 255),
    preloaded: Preloadedmodelclass = None,
):
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
        psf_type=Path(__file__).parent.parent.resolve() / 'lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat',
        psf_shape=window_size,
        lam_detection=wavelength,
        x_voxel_size=lateral_voxel_size,
        y_voxel_size=lateral_voxel_size,
        z_voxel_size=axial_voxel_size
    )

    predict_snr_map(
        Path(str(model_pred).replace('_tiles_predictions.csv', '.tif')),
        window_size=window_size
    )

    # tile id is the column header, rows are the predictions
    predictions, wavefronts = utils.create_multiindex_tile_dataframe(model_pred, return_wavefronts=True, describe=True)
    stdevs = utils.create_multiindex_tile_dataframe(str(model_pred).replace('_predictions.csv', '_stdevs.csv'))

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
    n_modes = np.sum([1 for col in predictions if str(col).isdigit()])

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
    errormap = resize(errormap, (ztiles, vol.shape[1], vol.shape[2]),  order=1, mode='edge') # linear interp XY
    errormap = resize(errormap, vol.shape,  order=0, mode='edge')   # nearest neighbor for z
    # errormap = resize(errormap, volume_shape, order=0, mode='constant')  # to show tiles
    imwrite(Path(f"{model_pred.with_suffix('')}_aggregated_p2v_error.tif"), errormap.astype(np.float32))

    # create a new column for cluster ids.
    predictions['cluster'] = np.nan
    stdevs['cluster'] = np.nan

    valid_predictions = predictions.loc[~(unconfident_tiles | zero_confident_tiles | all_zeros_tiles)]
    valid_predictions = valid_predictions.groupby('z')

    valid_stdevs = stdevs.loc[~(unconfident_tiles | zero_confident_tiles | all_zeros_tiles)]
    valid_stdevs = valid_stdevs.groupby('z')

    cluster_colors = np.split(
        np.array(sns.color_palette(clusters3d_colormap, n_colors=(max_isoplanatic_clusters * ztiles)))*255,
        ztiles,
    )   # list of colors for each z tiles

    clusters3d_colormap = []
    for cc in cluster_colors: # for each z tile's colors
        clusters3d_colormap.extend([zero_confident_color, *cc])  # append the same zero color (e.g. yellow) at the front
    clusters3d_colormap.extend([unconfident_color])  # append the unconfident color (e.g. white) to the end
    clusters3d_colormap = np.array(clusters3d_colormap)  # yellow, blue, orange,...  yellow, ...  white

    coefficients, actuators = {}, {}
    for z in valid_predictions.groups.keys():   # basically loop through all ztiles, unless no valid predictions exist
        ztile_preds = valid_predictions.get_group(z)
        ztile_preds.drop(columns=['cluster', 'p2v'], errors='ignore', inplace=True)

        ztile_stds = valid_stdevs.get_group(z)
        ztile_stds.drop(columns=['cluster', 'p2v'], errors='ignore', inplace=True)

        if optimize_max_isoplanatic_clusters:
            logger.info('KMeans calculating...')
            ks = np.arange(2, max_isoplanatic_clusters+1)
            ans = Parallel(n_jobs=-1, verbose=0)(delayed(kmeans_clustering)(ztile_preds.values, k) for k in ks)
            results = pd.DataFrame(ans, index=ks, columns=['silhouette'])
            max_silhouette = results['silhouette'].idxmax()
            max_isoplanatic_clusters = max_silhouette

        # weight zernike coefficients by their mth order for clustering
        features = ztile_preds.copy()
        for mode, twin in Wavefront(np.zeros(features.shape[1])).twins.items():
            if twin is not None:
                features[mode.index_ansi] /= abs(mode.m - 1)
                features[twin.index_ansi] /= twin.m + 1
            else:  # spherical modes
                features[mode.index_ansi] /= mode.m + 1

        n_clusters = min(max_isoplanatic_clusters, len(features))
        kmeans = KMeans(init="k-means++", n_clusters=n_clusters)
        kmeans.fit(features)

        ztile_preds['cluster'] = kmeans.predict(features) + 1
        ztile_preds['cluster'] += z * (max_isoplanatic_clusters + 1)

        # assign KMeans cluster ids to full dataframes (untouched ones, remain NaN)
        predictions.loc[ztile_preds.index, 'cluster'] = ztile_preds['cluster']
        stdevs.loc[ztile_preds.index, 'cluster'] = ztile_preds['cluster']

        predictions.loc[all_zeros_tiles, 'cluster'] = 0         # z * (max_isoplanatic_clusters + 1)
        stdevs.loc[all_zeros_tiles, 'cluster'] = 0              # z * (max_isoplanatic_clusters + 1)

        predictions.loc[zero_confident_tiles, 'cluster'] = 0    # z * (max_isoplanatic_clusters + 1)
        stdevs.loc[zero_confident_tiles, 'cluster'] = 0         # z * (max_isoplanatic_clusters + 1)

        predictions.loc[unconfident_tiles, 'cluster'] = len(clusters3d_colormap) - 1
        stdevs.loc[unconfident_tiles, 'cluster'] = len(clusters3d_colormap) - 1

        clusters = ztile_preds.groupby('cluster')
        for k in range(n_clusters+1):
            c = k + z * (max_isoplanatic_clusters + 1)

            if k == 0:    # "before" volume
                pred = np.zeros(n_modes)   # "before" will not have a wavefront update here.
                pred_std = np.zeros(n_modes)
            else:         # "after" volumes

                g = clusters.get_group(c).index

                # come up with a pred for this cluster based on user's choice of metric ("mean", "median", ...)
                if aggregation_rule == 'centers':
                    pred = kmeans.cluster_centers_[k-1]
                    pred_std = ztile_stds.loc[g].mask(where_unconfident).agg('mean', axis=0)
                else:
                    pred = ztile_preds.loc[g].mask(where_unconfident).drop(columns='cluster').agg(aggregation_rule, axis=0)
                    pred_std = ztile_stds.loc[g].mask(where_unconfident).agg(aggregation_rule, axis=0)

            cluster = f'z{z}_c{c}'

            pred = Wavefront(
                np.nan_to_num(pred, nan=0, posinf=0, neginf=0),
                order='ansi',
                lam_detection=wavelength
            )
            imwrite(
                Path(f"{model_pred.with_suffix('')}_aggregated_{cluster}_wavefront.tif"), pred.wave().astype(np.float32)
            )

            pred_std = Wavefront(
                np.nan_to_num(pred_std, nan=0, posinf=0, neginf=0),
                order='ansi',
                lam_detection=wavelength
            )

            if plot:
                task = partial(
                    vis.diagnosis,
                    pred=pred,
                    pred_std=pred_std,
                    save_path=Path(f"{model_pred.with_suffix('')}_aggregated_{cluster}_diagnosis"),
                )
                pool.apply_async(task)

            coefficients[cluster] = pred.amplitudes

            actuators[cluster] = utils.zernikies_to_actuators(
                pred.amplitudes,
                dm_calibration=dm_calibration,
                dm_state=dm_state,
                scalar=dm_damping_scalar
            )

    coefficients = pd.DataFrame.from_dict(coefficients)
    coefficients.index.name = 'ansi'
    coefficients.to_csv(f"{model_pred.with_suffix('')}_aggregated_zernike_coefficients.csv")

    actuators = pd.DataFrame.from_dict(actuators)
    actuators.index.name = 'actuators'
    actuators.to_csv(f"{model_pred.with_suffix('')}_aggregated_corrected_actuators.csv")

    predictions.to_csv(f"{model_pred.with_suffix('')}_aggregated_clusters.csv")

    clusters3d_heatmap = np.full_like(vol, len(clusters3d_colormap)-1, dtype=np.float32)
    wavefront_heatmap = np.zeros((ztiles, *vol.shape[1:]), dtype=np.float32)
    psf_heatmap = np.zeros((ztiles, *vol.shape[1:]), dtype=np.float32)
    clusters_rgb = np.full((ztiles, *vol.shape[1:]), len(clusters3d_colormap)-1, dtype=np.float32)

    zw, yw, xw = predictions_settings['window_size']
    logger.info(f"volume_size = {vol.shape}")
    logger.info(f"window_size = {zw, yw, xw}")
    logger.info(f"      tiles = {ztiles, ytiles, xtiles}")

    for i, (z, y, x) in enumerate(itertools.product(range(ztiles), range(ytiles), range(xtiles))):
        c = predictions.loc[(z, y, x), 'cluster']

        clusters_rgb[z, y*yw:(y*yw)+yw, x*xw:(x*xw)+xw] = np.full((yw, xw), int(c))    # cluster group id

        if c == len(clusters3d_colormap)-1:
            wavefront_heatmap[z, y*yw:(y*yw)+yw, x*xw:(x*xw)+xw] = np.zeros((yw, xw))
            psf_heatmap[z, y*yw:(y*yw)+yw, x*xw:(x*xw)+xw] = np.zeros((yw, xw))
        else:
            wavefront_heatmap[z, y*yw:(y*yw)+yw, x*xw:(x*xw)+xw] = np.nan_to_num(wavefronts[(z, y, x)].wave(xw), nan=0)
            psf_heatmap[z, y*yw:(y*yw)+yw, x*xw:(x*xw)+xw] = np.max(samplepsfgen.single_psf(wavefronts[(z, y, x)]), axis=0)

        clusters3d_heatmap[z*zw:(z*zw)+zw, y*yw:(y*yw)+yw, x*xw:(x*xw)+xw] = np.full((zw, yw, xw), int(c))

    imwrite(f"{model_pred.with_suffix('')}_aggregated_wavefronts.tif", wavefront_heatmap.astype(np.float32))
    imwrite(f"{model_pred.with_suffix('')}_aggregated_psfs.tif", psf_heatmap.astype(np.float32))

    scaled_wavefront_heatmap = (wavefront_heatmap - np.nanpercentile(wavefront_heatmap, 1)) / \
        (np.nanpercentile(wavefront_heatmap, 99) - np.nanpercentile(wavefront_heatmap, 1))
    scaled_wavefront_heatmap = np.clip(scaled_wavefront_heatmap, a_min=0, a_max=1)
    wavefront_rgb = clusters3d_colormap[clusters_rgb.astype(np.ubyte)] * scaled_wavefront_heatmap[..., np.newaxis]
    imwrite(
        f"{model_pred.with_suffix('')}_aggregated_clusters_wavefronts.tif",
        wavefront_rgb.astype(np.ubyte),
        photometric='rgb'
    )

    scaled_psf_heatmap = (psf_heatmap - np.nanpercentile(psf_heatmap, 1)) / \
        (np.nanpercentile(psf_heatmap, 99) - np.nanpercentile(psf_heatmap, 1))
    scaled_psf_heatmap = np.clip(scaled_psf_heatmap, a_min=0, a_max=1)
    psfs_rgb = clusters3d_colormap[clusters_rgb.astype(np.ubyte)] * scaled_psf_heatmap[..., np.newaxis]
    psfs_rgb = psfs_rgb
    imwrite(
        f"{model_pred.with_suffix('')}_aggregated_clusters_psfs.tif",
        psfs_rgb.astype(np.ubyte),
        photometric='rgb'
    )

    clusters3d = clusters3d_colormap[clusters3d_heatmap.astype(np.ubyte)] * vol[..., np.newaxis]
    clusters3d = clusters3d.astype(np.ubyte)
    imwrite(
        f"{model_pred.with_suffix('')}_aggregated_clusters.tif",
        clusters3d,
        photometric='rgb',
        imagej=True,
        resolution=(xw, yw),
    )

    reconstruct_wavefront_error_landscape(
        wavefronts=wavefronts,
        xtiles=xtiles,
        ytiles=ytiles,
        ztiles=ztiles,
        image=vol,
        save_path=Path(f"{model_pred.with_suffix('')}_aggregated_error_landscape.tif"),
        window_size=predictions_settings['window_size'],
        lateral_voxel_size=lateral_voxel_size,
        axial_voxel_size=axial_voxel_size,
        wavelength=wavelength,
        na=.9,
        tile_p2v=predictions['p2v'].values,
    )

    # vis.plot_volume(
    #     vol=terrain3d,
    #     results=coefficients,
    #     window_size=predictions_settings['window_size'],
    #     dxy=lateral_voxel_size,
    #     dz=axial_voxel_size,
    #     save_path=f"{model_pred.with_suffix('')}_aggregated_projections.svg",
    # )

    # vis.plot_isoplanatic_patchs(
    #     results=isoplanatic_patchs,
    #     clusters=clusters,
    #     save_path=f"{model_pred.with_suffix('')}_aggregated_isoplanatic_patchs.svg"
    # )

    logger.info(f'Done. Waiting for plots to write for {model_pred.with_suffix("")}')
    pool.close()    # close the pool
    pool.join()     # wait for all tasks to complete

    non_zero_tiles = ~(unconfident_tiles | zero_confident_tiles | all_zeros_tiles)
    with Path(f"{model_pred.with_suffix('')}_aggregate_settings.json").open('w') as f:
        json = dict(
            model_pred=str(model_pred),
            dm_calibration=str(dm_calibration),
            dm_state=str(dm_state),
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
            axial_voxel_size=float(axial_voxel_size),
            lateral_voxel_size=float(lateral_voxel_size),
            ztiles=int(ztiles),
            ytiles=int(ytiles),
            xtiles=int(xtiles),
            volume_size=list(vol.shape),

            total_confident_zero_tiles=int(zero_confident_tiles.sum()),
            total_unconfident_tiles=int(unconfident_tiles.sum()),
            total_all_zeros_tiles=int(all_zeros_tiles.sum()),
            total_non_zero_tiles=int(non_zero_tiles.sum()),

            confident_zero_tiles=zero_confident_tiles.loc[zero_confident_tiles].index.to_list(),
            unconfident_tiles=unconfident_tiles.loc[unconfident_tiles].index.to_list(),
            all_zeros_tiles=all_zeros_tiles.loc[all_zeros_tiles].index.to_list(),
            non_zero_tiles=non_zero_tiles.loc[non_zero_tiles].index.to_list(),

            zero_confident_color=list(zero_confident_color),
            unconfident_color=list(unconfident_color),
            clusters3d_colormap=clusters3d_colormap.tolist(),
        )
        ujson.dump(
            json,
            f,
            indent=4,
            sort_keys=False,
            ensure_ascii=False,
            escape_forward_slashes=False
        )

    return coefficients


@profile
def combine_tiles(
    corrected_actuators_csv: Path,
    corrections: list,
    prediction_threshold: float = 0.25,
    aggregation_rule: str = 'median',
    dm_calibration: Path = Path('../calibration/aang/28_mode_calibration.csv')
):
    """
    Combine tiles from several DM patterns based on cluster IDs
    Args:
        corrected_actuators_csv: either _tiles_predictions_aggregated_corrected_actuators.csv (0th iteration)
                                     or _corrected_cluster_actuators.csv (Nth iteration)
        corrections: a list of tuples (clusterid, path to .tif scan taken with the given DM pattern)

    """
    original_acts = pd.read_csv(
        corrected_actuators_csv,
        index_col=0,
        header=0
    )

    acts_suffix = "_combined_corrected_actuators.csv"
    base_path = str(corrected_actuators_csv).replace('_tiles_predictions_aggregated_corrected_actuators.csv', '')
    base_path = base_path.replace(acts_suffix, '') # remove this if it exists
    base_path = Path(base_path.replace('_corrected_cluster_actuators.csv', '')) # also remove this if it exists

    output_base_path = str(corrections[0]).replace('_tiles_predictions_aggregated_p2v_error.tif', '')

    with open(f"{base_path}_tiles_predictions_settings.json") as f:
        predictions_settings = ujson.load(f)

    image_shape = tuple(predictions_settings['input_shape'])

    correction_scans = np.zeros((len(corrections), *image_shape))
    error_maps = np.zeros((len(corrections), *image_shape))            # series of 3d p2v maps aka a 4d array
    snr_scans = np.zeros((len(corrections), *image_shape))            # series of 3d p2v maps aka a 4d array

    for t, path in enumerate(corrections):
        error_maps[t] = backend.load_sample(path)
        correction_scans[t] = backend.load_sample(str(path).replace('_tiles_predictions_aggregated_p2v_error.tif', '.tif'))
        snr_scans[t] = backend.load_sample(str(path).replace('_tiles_predictions_aggregated_p2v_error.tif', '_snrs.tif'))

    # indices = np.argmin(error_maps, axis=0)  # locate the correction with the lowest error for every voxel (3D array)
    indices = np.argmax(snr_scans, axis=0)  # locate the correction with the highest snr for every voxel (3D array)
    z, y, x = np.indices(indices.shape)
    combined_errormap = error_maps[indices, z, y, x]    # retrieve the best p2v
    combined_snrmap = snr_scans[indices, z, y, x]       # retrieve the best snr
    combined = correction_scans[indices, z, y, x]       # retrieve the best data

    imwrite(f"{output_base_path}_volume_used.tif", indices.astype(np.uint16))
    imwrite(f"{output_base_path}_combined.tif", combined.astype(np.float32))
    imwrite(f"{output_base_path}_combined_error.tif", combined_errormap.astype(np.float32))
    imwrite(f"{output_base_path}_combined_snr.tif", combined_snrmap.astype(np.float32))

    tile_ids = resize(
        indices,
        (
            predictions_settings['ztiles'],
            predictions_settings['ytiles'],
            predictions_settings['xtiles'],
        ),
        order=0,
        mode='edge',
        anti_aliasing=False,
        preserve_range=True,
    ).astype(np.float)
    z_indices, y_indices, x_indices = np.indices(tile_ids.shape)
    tile_ids += np.nanmax(tile_ids) * z_indices

    coefficients, actuators = {}, {}
    for i, path in enumerate(corrections):  # skip the before
        model_pred = str(path).replace('_tiles_predictions_aggregated_p2v_error.tif', '_tiles_predictions.csv')

        predictions = utils.create_multiindex_tile_dataframe(model_pred)
        stdevs = utils.create_multiindex_tile_dataframe(model_pred.replace('_predictions.csv', '_stdevs.csv'))

        unconfident_tiles, zero_confident_tiles, all_zeros_tiles = utils.get_tile_confidence(
            predictions=predictions,
            stdevs=stdevs,
            prediction_threshold=prediction_threshold,
            ignore_modes=predictions_settings['ignore_modes'],
            verbose=False
        )
        for z_tile_index in range(predictions_settings['ztiles']):
            clusterid = i + (len(corrections) * z_tile_index)
            dm_state = original_acts[f'z{z_tile_index}_c{clusterid}'].values

            winners = pd.MultiIndex.from_arrays(np.where(tile_ids == clusterid), names=('z', 'y', 'x'))
            pred = predictions.loc[winners].loc[~unconfident_tiles]
            pred_shape = pred.shape
            pred = pred.drop(columns='p2v').agg(aggregation_rule, axis=0)

            pred = Wavefront(
                np.nan_to_num(pred, nan=0, posinf=0, neginf=0),
                order='ansi',
                lam_detection=predictions_settings['wavelength']
            )

            coefficients[f'z{z_tile_index}_c{clusterid}'] = pred.amplitudes
            logger.info(f'z{z_tile_index}_c{clusterid} wavefront change (p2V) = {pred.peak2valley(na=0.9):4.3f} waves.'
                        f' Using {pred_shape[0]:3d} tiles from {Path(model_pred).name}')

            actuators[f'z{z_tile_index}_c{clusterid}'] = utils.zernikies_to_actuators(
                pred.amplitudes,
                dm_calibration=dm_calibration,
                dm_state=dm_state,
            )

    coefficients = pd.DataFrame.from_dict(coefficients)
    coefficients.index.name = 'ansi'
    coefficients.sort_index(axis=1, inplace=True)
    new_zcoeff_path = f"{output_base_path}_combined_zernike_coefficients.csv"
    coefficients.to_csv(new_zcoeff_path)

    actuators = pd.DataFrame.from_dict(actuators)
    actuators.index.name = 'actuators'
    actuators.sort_index(axis=1, inplace=True)

    acts_path = f"{output_base_path}{acts_suffix}" # used in Labview
    actuators.to_csv(acts_path)
    logger.info(f"Org actuators: {corrected_actuators_csv}")
    logger.info(f"New actuators: {acts_path}")
    logger.info(f"New predictions: {new_zcoeff_path}")
    logger.info(f"Columns: {actuators.columns.values}")


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
    use_pyotf_zernikes: bool = False
):

    try:
        import pyotf.pyotf.phaseretrieval as pr
        from pyotf.pyotf.utils import prep_data_for_PR
        from pyotf.pyotf.zernike import osa2degrees
    except ImportError as e:
        logger.error(e)
        return -1

    dm_state = None if (dm_state is None or str(dm_state) == 'None') else dm_state

    data = np.int_(imread(img))

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
    otf = psfgen.fft(psf)
    otf = remove_interference_pattern(
        psf=psf,
        otf=otf,
        plot=f"{img.with_suffix('')}_phase_retrieval" if plot else None,
        max_num_peaks=1,
        windowing=False,
    )
    data = np.int_(psfgen.ifft(otf) * np.nanmax(data))

    params = dict(
        wl=psfgen.lam_detection,
        na=psfgen.na_detection,
        ni=psfgen.refractive_index,
        res=lateral_voxel_size,
        zres=axial_voxel_size,
    )

    logger.info("Starting phase retrieval iterations")
    data_prepped = prep_data_for_PR(np.flip(data, axis=0), multiplier=1.1)
    data_prepped = cp.asarray(data_prepped)  # use GPU. Comment this line to use CPU.
    pr_result = pr.retrieve_phase(
        data_prepped,
        params,
        max_iters=num_iterations,
        pupil_tol=1e-5,
        mse_tol=0,
        phase_only=True
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

    psf = psfgen.single_psf(pred, normed=True)
    imwrite(f"{img.with_suffix('')}_phase_retrieval_psf.tif", psf)

    if plot:
        vis.diagnosis(
            pred=pred,
            pred_std=pred_std,
            save_path=Path(f"{img.with_suffix('')}_phase_retrieval_diagnosis"),
        )

        fig, axes = pr_result.plot()
        axes[0].set_title("Phase in waves")
        vis.savesvg(fig, Path(f"{img.with_suffix('')}_phase_retrieval_convergence.svg"))

    return coefficients
