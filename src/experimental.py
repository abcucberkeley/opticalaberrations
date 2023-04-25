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
from tqdm import trange
from line_profiler_pycharm import profile
from numpy.lib.stride_tricks import sliding_window_view
import multiprocessing as mp
from sklearn.cluster import KMeans
from skimage.transform import resize
from sklearn.metrics import silhouette_samples, silhouette_score
from joblib import Parallel, delayed

import utils
import vis
import backend
import preprocessing

from synthetic import SyntheticPSF
from wavefront import Wavefront
from data_utils import get_image
from preloaded import Preloadedmodelclass
from embeddings import remove_interference_pattern, fourier_embeddings, rolling_fourier_embeddings
from preprocessing import round_to_even

import logging
logger = logging.getLogger('')

try:
    import cupy as cp
except ImportError as e:
    logging.warning(f"Cupy not supported on your system: {e}")


@profile
def load_sample(data: Union[tf.Tensor, Path, str, np.ndarray]):
    if isinstance(data, np.ndarray):
        img = data
    elif isinstance(data, bytes):
        img = Path(str(data, "utf-8"))
    elif isinstance(data, tf.Tensor):
        path = Path(str(data.numpy(), "utf-8"))
        img = get_image(path).astype(float)
    else:
        path = Path(str(data))
        img = get_image(path).astype(float)
    return np.squeeze(img)


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


def preprocess(
    file: Union[tf.Tensor, Path, str],
    modelpsfgen: SyntheticPSF,
    samplepsfgen: SyntheticPSF,
    freq_strength_threshold: float = .01,
    digital_rotations: Optional[int] = 361,
    remove_background: bool = True,
    read_noise_bias: float = 5,
    normalize: bool = True,
    edge_filter: bool = True,
    filter_mask_dilation: bool = True,
    plot: Any = None,
    no_phase: bool = False,
    match_model_fov: bool = True
):
    if isinstance(file, tf.Tensor):
        file = Path(str(file.numpy(), "utf-8"))

    if isinstance(plot, bool) and plot:
        plot = file.with_suffix('')

    sample = load_sample(file)

    if match_model_fov:
        sample = preprocessing.prep_sample(
            sample,
            model_fov=modelpsfgen.psf_fov,
            sample_voxel_size=samplepsfgen.voxel_size,
            remove_background=remove_background,
            normalize=normalize,
            edge_filter=edge_filter,
            filter_mask_dilation=filter_mask_dilation,
            read_noise_bias=read_noise_bias,
            plot=plot if plot else None
        )

        return fourier_embeddings(
            sample,
            iotf=modelpsfgen.iotf,
            plot=plot if plot else None,
            no_phase=no_phase,
            remove_interference=True,
            embedding_option=modelpsfgen.embedding_option,
            freq_strength_threshold=freq_strength_threshold,
            digital_rotations=digital_rotations,
            poi_shape=modelpsfgen.psf_shape[1:]
        )
    else:
        window_size = (
            round_to_even(modelpsfgen.psf_fov[0] / samplepsfgen.voxel_size[0]),
            round_to_even(modelpsfgen.psf_fov[1] / samplepsfgen.voxel_size[1]),
            round_to_even(modelpsfgen.psf_fov[2] / samplepsfgen.voxel_size[2]),
        )

        rois = sliding_window_view(
            sample,
            window_shape=window_size
        )[::window_size[0], ::window_size[1], ::window_size[2]]
        ztiles, nrows, ncols = rois.shape[:3]
        rois = np.reshape(rois, (-1, *window_size))

        prep = partial(
            preprocessing.prep_sample,
            sample_voxel_size=samplepsfgen.voxel_size,
            remove_background=remove_background,
            normalize=normalize,
            edge_filter=edge_filter,
            filter_mask_dilation=filter_mask_dilation,
            read_noise_bias=read_noise_bias,
        )

        rois = utils.multiprocess(
            func=prep,
            jobs=rois,
            desc='Preprocessing'
        )

        return rolling_fourier_embeddings(
            rois,
            iotf=modelpsfgen.iotf,
            plot=plot,
            no_phase=no_phase,
            remove_interference=True,
            embedding_option=modelpsfgen.embedding_option,
            freq_strength_threshold=freq_strength_threshold,
            digital_rotations=digital_rotations,
            poi_shape=modelpsfgen.psf_shape[1:],
            nrows=nrows,
            ncols=ncols,
            ztiles=ztiles
        )


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
    match_model_fov: bool = True,
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

    sample = load_sample(file)
    psnr = preprocessing.prep_sample(
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
        snr=psnr,
        psf_shape=sample.shape,
        n_modes=model.output_shape[1],
        lam_detection=wavelength,
        x_voxel_size=lateral_voxel_size,
        y_voxel_size=lateral_voxel_size,
        z_voxel_size=axial_voxel_size
    )

    sample = preprocessing.prep_sample(
        sample,
        sample_voxel_size=samplepsfgen.voxel_size,
        remove_background=remove_background,
        normalize=normalize,
        edge_filter=edge_filter,
        filter_mask_dilation=filter_mask_dilation,
        read_noise_bias=read_noise_bias,
        plot=file.with_suffix('') if plot else None,
    )

    return preprocess(
        sample,
        modelpsfgen=modelpsfgen,
        samplepsfgen=samplepsfgen,
        freq_strength_threshold=freq_strength_threshold,
        remove_background=True,
        normalize=True,
        edge_filter=False,
        match_model_fov=match_model_fov,
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
    isoplanatic_patch_colormap: Union[Path, str] = Path.joinpath(Path(__file__).parent, '../CETperceptual/CET-C2.csv'),
    na: float = 1.0,
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
        volume_shape: Number of pixels in full volume
        na: Numerical aperature limit which to use for calculating p2v error
        wavelength:

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
    tile_p2v = np.full((xtiles * ytiles * ztiles), np.nan)

    matrix_row = 0  # pointer to where we are writing
    for i, tile_coords in enumerate(itertools.product(range(ztiles), range(ytiles), range(xtiles))):
        neighbours = list(get_neighbors(tile_coords))
        tile_wavefront = wavefronts[tile_coords]
        if np.isnan(tile_p2v[i]):
            tile_p2v[i] = tile_wavefront.peak2valley(na=na)

        for j, neighbour_coords in enumerate(neighbours):  # ordered as (z, y, x) neighbours
            try:
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
                    slopes[matrix_row] = p2v / h[j]
                    A[matrix_row, np.ravel_multi_index(neighbour_coords, (ztiles, ytiles, xtiles))] = 1 / h[j]
                    A[matrix_row, i] = -1 / h[j]
                    matrix_row += 1

            except KeyError:
                pass    # e.g. if neighbor is beyond the border or that tile was dropped

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
    terrain3d = resize(terrain3d, image.shape)
    # terrain3d = resize(terrain3d, volume_shape, order=0, mode='constant')  # to show tiles

    isoplanatic_patch_colormap = pd.read_csv(
        isoplanatic_patch_colormap.resolve(),
        header=None,
        index_col=None,
        dtype=np.ubyte
    ).values

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
    confidence_threshold: float = .0099,
    batch_size: int = 1,
    digital_rotations: Optional[int] = 361,
    plot: bool = True,
    plot_rotations: bool = False,
    ztiles: int = 1,
    nrows: int = 1,
    ncols: int = 1,
    cpu_workers: int = -1
):
    no_phase = True if model.input_shape[1] == 3 else False

    generate_fourier_embeddings = partial(
        utils.multiprocess,
        func=partial(
            preprocess,
            modelpsfgen=modelpsfgen,
            samplepsfgen=samplepsfgen,
            freq_strength_threshold=freq_strength_threshold,
            digital_rotations=digital_rotations,
            plot=plot,
            no_phase=no_phase,
            remove_background=True,
            normalize=True,
            edge_filter=False,
            filter_mask_dilation=True
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
    confidence_threshold: float = .0099,
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
    sample = load_sample(img)
    psnr = preprocessing.prep_sample(
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
        snr=psnr,
        psf_shape=sample.shape,
        n_modes=preloadedmodel.output_shape[1],
        lam_detection=wavelength,
        x_voxel_size=lateral_voxel_size,
        y_voxel_size=lateral_voxel_size,
        z_voxel_size=axial_voxel_size
    )

    embeddings = preprocess(
        sample,
        modelpsfgen=premodelpsfgen,
        samplepsfgen=samplepsfgen,
        digital_rotations=digital_rotations,
        remove_background=True,
        normalize=True,
        edge_filter=False,
        filter_mask_dilation=True,
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

    psf = samplepsfgen.single_psf(phi=p, normed=True, noise=False)
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
    confidence_threshold: float = .0099,
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

    sample = load_sample(img)
    psnr = preprocessing.prep_sample(
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
        snr=psnr,
        psf_shape=sample.shape,
        n_modes=preloadedmodel.output_shape[1],
        lam_detection=wavelength,
        x_voxel_size=lateral_voxel_size,
        y_voxel_size=lateral_voxel_size,
        z_voxel_size=axial_voxel_size
    )

    embeddings = preprocess(
        sample,
        modelpsfgen=premodelpsfgen,
        samplepsfgen=samplepsfgen,
        digital_rotations=digital_rotations,
        no_phase=no_phase,
        freq_strength_threshold=freq_strength_threshold,
        remove_background=True,
        normalize=True,
        edge_filter=False,
        match_model_fov=False,
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

    psf = samplepsfgen.single_psf(phi=p, normed=True, noise=False)
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
    sample = np.squeeze(get_image(img).astype(float))
    logger.info(f"Sample: {sample.shape}")

    rois, ztiles, nrows, ncols = preprocessing.find_roi(
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
        snr=100,
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
    prediction_threshold: float = 0.,
    freq_strength_threshold: float = .01,
    confidence_threshold: float = .0099,
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
):

    preloadedmodel, premodelpsfgen = reloadmodel_if_needed(
        modelpath=model,
        preloaded=preloaded,
        ideal_empirical_psf=ideal_empirical_psf,
        ideal_empirical_psf_voxel_size=(axial_voxel_size, lateral_voxel_size, lateral_voxel_size)
    )

    logger.info(f"Loading file: {img.name}")
    sample = load_sample(img)
    psnr = preprocessing.prep_sample(
        sample,
        return_psnr=True,
        remove_background=True,
        normalize=False,
        edge_filter=False,
        filter_mask_dilation=False,
    )
    logger.info(f"Sample: {sample.shape}")

    outdir = Path(f"{img.with_suffix('')}_tiles")
    outdir.mkdir(exist_ok=True, parents=True)

    rois, ztiles, nrows, ncols = preprocessing.get_tiles(
        sample,
        savepath=outdir,
        strides=window_size,
        window_size=window_size,
    )

    samplepsfgen = SyntheticPSF(
        psf_type=premodelpsfgen.psf_type,
        snr=psnr,
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
            prediction_threshold=float(prediction_threshold),
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
        prediction_threshold=prediction_threshold,
        confidence_threshold=confidence_threshold,
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
    prediction_threshold: float = 0.,
    confidence_threshold: float = .0099,
    aggregation_rule: str = 'mean',
    dm_damping_scalar: float = 1,
    max_isoplanatic_clusters: int = 3,
    optimize_max_isoplanatic_clusters: bool = False,
    plot: bool = False,
    ignore_tile: Any = None,
    preloaded: Preloadedmodelclass = None,
):

    vol = load_sample(str(model_pred).replace('_tiles_predictions.csv', '.tif'))
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


    # tile id is the column header, rows are the predictions
    predictions = pd.read_csv(
        model_pred,
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
    print(f'prediction stats \n{predictions.describe()}')

    stdevs = pd.read_csv(
        str(model_pred).replace('_predictions.csv', '_stdevs.csv'),
        index_col=0,
        header=0,
        usecols=lambda col: col == 'ansi' or col.startswith('z')
    ).T
    stdevs.index = pd.MultiIndex.from_tuples(stdevs.index.str.split('-').to_list())
    stdevs.index = pd.MultiIndex.from_arrays([
        stdevs.index.get_level_values(0).str.lstrip('z').astype(np.int),
        stdevs.index.get_level_values(1).str.lstrip('y').astype(np.int),
        stdevs.index.get_level_values(2).str.lstrip('x').astype(np.int),
    ], names=('z', 'y', 'x'))
    # print(f'std dev stats \n{stdevs.describe()}')

    ztiles = predictions.index.get_level_values('z').unique().shape[0]
    ytiles = predictions.index.get_level_values('y').unique().shape[0]
    xtiles = predictions.index.get_level_values('x').unique().shape[0]

    if ignore_tile is not None:
        for cc in ignore_tile:
            z, y, x = [int(s) for s in cc if s.isdigit()]
            predictions.loc[(z, y, x)] = np.nan
            stdevs.loc[(z, y, x)] = np.nan

    all_zeros = predictions == 0    # will label tiles that are any mix of (confident zero and unconfident).

    # filter out unconfident predictions (std deviation is too large)
    where_unconfident = stdevs == 0
    where_unconfident[predictions_settings['ignore_modes']] = False

    # filter out small predictions from KMeans cluster analysis, but keep these as an additional group
    prediction_threshold = utils.waves2microns(prediction_threshold, wavelength=wavelength)
    where_zero_confident = (predictions.abs() <= prediction_threshold) & ~where_unconfident
    where_zero_confident[predictions_settings['ignore_modes']] = True

    where_unconfident[predictions_settings['ignore_modes']] = True  # ignore these modes during agg
    all_zeros = all_zeros.agg('all', axis=1)  # 1D (one value for each tile)
    where_unconfident = where_unconfident.agg('all', axis=1)  # 1D (one value for each tile)
    where_zero_confident = where_zero_confident.agg('all', axis=1)  # 1D (one value for each tile)

    print(f'Number of confident zero tiles {where_zero_confident.sum():4} out of {where_zero_confident.count()}')
    print(f'Number of unconfident tiles    {where_unconfident.sum():4} out of {where_unconfident.count()}')
    print(f'Number of all zeros tiles      {all_zeros.sum():4} out of {all_zeros.count()}')
    print(f'Number of non-zero tiles       {(~(where_unconfident | where_zero_confident | all_zeros)).sum():4} out of {all_zeros.count()}')

    coefficients, actuators = {}, {}

    # create a new column for cluster ids.
    predictions['cluster'] = np.nan
    stdevs['cluster'] = np.nan

    valid_predictions = predictions.loc[~(where_unconfident | where_zero_confident | all_zeros)]
    valid_predictions = valid_predictions.groupby('z')

    valid_stdevs = stdevs.loc[~(where_unconfident | where_zero_confident | all_zeros)]
    valid_stdevs = valid_stdevs.groupby('z')

    clusters3d_colormap = sns.color_palette("tab10", n_colors=(max_isoplanatic_clusters * ztiles))
    clusters3d_colormap.append((1, 1, 0))  # yellow color for no aberration
    clusters3d_colormap.append((1, 1, 1))  # white color for unconfident (no data)
    clusters3d_colormap = np.array(clusters3d_colormap)*255

    for z in trange(ztiles, desc='Aggregating Z tiles', total=ztiles):
        ztile_preds = valid_predictions.get_group(z)
        ztile_preds.drop(columns='cluster', errors='ignore', inplace=True)

        ztile_stds = valid_stdevs.get_group(z)
        ztile_stds.drop(columns='cluster', errors='ignore', inplace=True)

        if optimize_max_isoplanatic_clusters:
            logger.info('KMeans calculating...')
            ks = np.arange(2, max_isoplanatic_clusters+1)
            ans = Parallel(n_jobs=-1, verbose=0)(delayed(kmeans_clustering)(ztile_preds.values, k) for k in ks)
            results = pd.DataFrame(ans, index=ks, columns=['silhouette'])
            max_silhouette = results['silhouette'].idxmax()
            max_isoplanatic_clusters = max_silhouette

        ztile_preds['cluster'] = KMeans(init="k-means++", n_clusters=max_isoplanatic_clusters).fit_predict(ztile_preds)
        ztile_preds['cluster'] += z * max_isoplanatic_clusters

        # assign KMeans cluster ids to full dataframes (untouched ones, remain NaN)
        predictions.loc[ztile_preds.index, 'cluster'] = ztile_preds['cluster']
        stdevs.loc[ztile_preds.index, 'cluster'] = ztile_preds['cluster']

        clusters = ztile_preds.groupby('cluster')
        for c in range(max_isoplanatic_clusters):
            c += z * max_isoplanatic_clusters

            g = clusters.get_group(c).index  # get all tiles that belong to cluster "c"
            # come up with a pred for this cluster based on user's choice of metric ("mean", "median", ...)
            pred = ztile_preds.loc[g].drop(columns='cluster').agg(aggregation_rule, axis=0)     # mean ignoring NaNs
            pred_std = ztile_stds.loc[g].agg(aggregation_rule, axis=0)

            pred = Wavefront(
                np.nan_to_num(pred, nan=0, posinf=0, neginf=0),
                order='ansi',
                lam_detection=wavelength
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
                    save_path=Path(f"{model_pred.with_suffix('')}_aggregated_diagnosis_z{z}_c{c}"),
                )
                pool.apply_async(task)

            coefficients[f'z{z}_c{c}'] = pred.amplitudes

            actuators[f'z{z}_c{c}'] = utils.zernikies_to_actuators(
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

    wavefronts = {}
    predictions.loc[all_zeros, 'cluster'] = len(clusters3d_colormap) - 2
    predictions.loc[where_zero_confident, 'cluster'] = len(clusters3d_colormap) - 2
    predictions.loc[where_unconfident, 'cluster'] = len(clusters3d_colormap) - 1

    for index, zernikes in predictions.drop(columns='cluster').iterrows():
        wavefronts[index] = Wavefront(
            np.nan_to_num(zernikes.values, nan=0),
            lam_detection=wavelength,
        )

    clusters3d_heatmap = np.ones_like(vol, dtype=np.float32) * len(clusters3d_colormap) - 1
    wavefront_heatmap = np.zeros((ztiles, *vol.shape[1:]), dtype=np.float32)
    wavefront_rgb = np.ones((ztiles, *vol.shape[1:]), dtype=np.float32) * len(clusters3d_colormap) - 1

    zw, yw, xw = predictions_settings['window_size']
    print(f"volume_size = {vol.shape}\n"
          f"window_size = {predictions_settings['window_size']}\n"
          f"      tiles = {ztiles, ytiles, xtiles}")
    for i, (z, y, x) in enumerate(itertools.product(range(ztiles), range(ytiles), range(xtiles))):
        c = predictions.loc[(z, y, x), 'cluster']

        wavefront_rgb[z, y*yw:(y*yw)+yw, x*xw:(x*xw)+xw] = np.ones((yw, xw)) * int(c)
        wavefront_heatmap[z, y*yw:(y*yw)+yw, x*xw:(x*xw)+xw] = np.nan_to_num(wavefronts[(z, y, x)].wave(xw), nan=0)
        clusters3d_heatmap[z*zw:(z*zw)+zw, y*yw:(y*yw)+yw, x*xw:(x*xw)+xw] = np.ones((zw, yw, xw)) * int(c)

    imwrite(f"{model_pred.with_suffix('')}_aggregated_wavefronts.tif", wavefront_heatmap, dtype=np.float32)

    scaled_wavefront_heatmap = (wavefront_heatmap - np.nanmin(wavefront_heatmap)) / (np.nanmax(wavefront_heatmap) * 2)
    wavefront_rgb = clusters3d_colormap[wavefront_rgb.astype(np.ubyte)] * scaled_wavefront_heatmap[..., np.newaxis]
    wavefront_rgb = wavefront_rgb.astype(np.ubyte)
    imwrite(f"{model_pred.with_suffix('')}_aggregated_clusters.tif", wavefront_rgb, photometric='rgb')

    clusters3d = clusters3d_colormap[clusters3d_heatmap.astype(np.ubyte)] * vol[..., np.newaxis]
    clusters3d = clusters3d.astype(np.ubyte)
    imwrite(f"{model_pred.with_suffix('')}_aggregated_isoplanatic_patchs.tif", clusters3d, photometric='rgb')

    reconstruct_wavefront_error_landscape(
        wavefronts=wavefronts,
        xtiles=xtiles,
        ytiles=ytiles,
        ztiles=ztiles,
        image=vol,
        save_path=Path(f"{model_pred.with_suffix('')}_aggregated_error.tif"),
        window_size=predictions_settings['window_size'],
        lateral_voxel_size=lateral_voxel_size,
        axial_voxel_size=axial_voxel_size,
        wavelength=wavelength,
        na=.9,
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

    logger.info('Done. Waiting for plots to write.')
    pool.close()    # close the pool
    pool.join()     # wait for all tasks to complete

    return coefficients


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
        snr=100,
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
    pr_result.phase = utils.waves2microns(pupil, wavelength=psfgen.lam_detection)  # convert waves to microns before fitting.
    pr_result.fit_to_zernikes(num_modes-1, mapping=osa2degrees)  # pyotf's zernikes now in um rms
    pr_result.phase = pupil  # phase is now again in waves

    pupil[pupil == 0.] = np.nan # put NaN's outside of pupil
    pupil_path = Path(f"{img.with_suffix('')}_phase_retrieval_wavefront.tif")
    imwrite(pupil_path, cp.asnumpy(pupil))

    threshold = utils.waves2microns(prediction_threshold, wavelength=psfgen.lam_detection)
    ignore_modes = list(map(int, ignore_modes))

    if use_pyotf_zernikes:
        # use pyotf definition of zernikes and fit using them.  I suspect m=0 modes have opposite sign to our definition.
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

    psf = psfgen.single_psf(pred, normed=True, noise=False)
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
