import matplotlib
matplotlib.use('Agg')

import re
from functools import partial
import ujson

import matplotlib.pyplot as plt
plt.set_loglevel('error')

from pathlib import Path
import tensorflow as tf

from typing import Any, Union, Optional, Generator
import numpy as np
import pandas as pd
from tifffile import imread, imsave
from tqdm import trange
from line_profiler_pycharm import profile
from numpy.lib.stride_tricks import sliding_window_view

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

def weighted_avg_and_std(values, weights, axis):
    """
    Return the weighted average and standard deviation.

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights, axis=axis)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights, axis=axis)
    return (average, np.sqrt(variance))


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
    digital_rotations: Optional[np.ndarray] = np.arange(0, 360 + 1, 1).astype(int),
    remove_background: bool = True,
    read_noise_bias: float = 5,
    normalize: bool = True,
    edge_filter: bool = True,
    filter_mask_dilation: bool = True,
    plot: Any = None,
    no_phase: bool = False,
    match_model_fov: bool = True,
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
            plot=plot
        )

        return fourier_embeddings(
            sample,
            iotf=modelpsfgen.iotf,
            plot=plot,
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
    digital_rotations: Optional[Union[Generator, list, np.ndarray]] = None
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
    digital_rotations: np.ndarray = np.arange(0, 360+1, 1).astype(int),
    plot: Optional[Union[bool, Path, str]] = None,
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
            filter_mask_dilation=True,
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

    ps, std = [], []
    for tile, file in zip(inputs.as_numpy_iterator(), rois):
        res = backend.predict_rotation(
            model,
            inputs=tile,
            psfgen=modelpsfgen,
            no_phase=no_phase,
            batch_size=batch_size,
            ignore_modes=ignore_modes,
            threshold=prediction_threshold,
            confidence_threshold=confidence_threshold,
            freq_strength_threshold=freq_strength_threshold,
            plot=file.with_suffix('') if plot else None,
            plot_rotations=file.with_suffix('') if plot_rotations else None,
            digital_rotations=digital_rotations,
            desc=f'Predicting ROIs in ({outdir.name})',
        )

        try:
            p, s = res
            lls_defocus = 0.
        except ValueError:
            p, s, lls_defocus = res

        if plot:
            vis.diagnosis(
                pred=Wavefront(p, lam_detection=wavelength),
                pred_std=Wavefront(s, lam_detection=wavelength),
                save_path=Path(f"{file.with_suffix('')}_diagnosis"),
                lls_defocus=lls_defocus
            )

        ps.append(p)
        std.append(s)

    ps, std = np.concatenate(ps), np.concatenate(std)

    tile_names = [f.with_suffix('').name for f in rois]
    predictions = pd.DataFrame(ps.T, columns=tile_names)
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

    if plot:
        vis.wavefronts(
            predictions=predictions,
            nrows=nrows,
            ncols=ncols,
            ztiles=ztiles,
            wavelength=wavelength,
            save_path=Path(f"{outdir.with_suffix('')}_predictions_wavefronts"),
        )

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
    digital_rotations: np.ndarray = np.arange(0, 360 + 1, 1).astype(int),
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
    imsave(f"{img.with_suffix('')}_sample_predictions_psf.tif", psf)

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
    digital_rotations: np.ndarray = np.arange(0, 360 + 1, 1).astype(int),
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
        cpu_workers=cpu_workers
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
    imsave(f"{img.with_suffix('')}_large_fov_predictions_psf.tif", psf)

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
    digital_rotations: np.ndarray = np.arange(0, 360 + 1, 1).astype(int),
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
    digital_rotations: np.ndarray = np.arange(0, 360 + 1, 1).astype(int),
    cpu_workers: int = -1
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
        psf_shape=sample.shape,
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
        cpu_workers=cpu_workers
    )

    if plot:
        vis.tiles(
            data=sample,
            strides=window_size,
            window_size=window_size,
            save_path=Path(f"{outdir.with_suffix('')}_predictions_mips"),
        )

    return predictions


@profile
def aggregate_predictions(
    model: Path,
    model_pred: Path,
    dm_calibration: Path,
    dm_state: Any,
    wavelength: float = .605,
    axial_voxel_size: float = .1,
    lateral_voxel_size: float = .108,
    majority_threshold: float = .5,
    min_percentile: int = 1,
    max_percentile: int = 99,
    prediction_threshold: float = 0.,
    confidence_threshold: float = .0099,
    final_prediction: str = 'mean',
    dm_damping_scalar: float = 1,
    plot: bool = False,
    ignore_tile: Any = None,
    preloaded: Preloadedmodelclass = None
):
    def calc_length(s):
        return int(re.sub(r'[a-z]+', '', s)) + 1

    if final_prediction == 'mean':
        final_prediction = np.nanmean
    elif final_prediction == 'median':
        final_prediction = np.nanmedian
    elif final_prediction == 'min':
        final_prediction = np.nanmin
    elif final_prediction == 'max':
        final_prediction = np.nanmax
    else:
        logger.error(f'Unknown function: {final_prediction}')
        return -1

    preloadedmodel, premodelpsfgen = reloadmodel_if_needed(
        modelpath=model,
        preloaded=preloaded,
        ideal_empirical_psf_voxel_size=(axial_voxel_size, lateral_voxel_size, lateral_voxel_size)
    )

    samplepsfgen = SyntheticPSF(
        psf_type=premodelpsfgen.psf_type,
        n_modes=preloadedmodel.output_shape[1],
        lam_detection=wavelength,
        x_voxel_size=lateral_voxel_size,
        y_voxel_size=lateral_voxel_size,
        z_voxel_size=axial_voxel_size
    )

    predictions = pd.read_csv(
        model_pred,
        index_col=0,
        header=0,
        usecols=lambda col: col == 'ansi' or col.startswith('z')
    )

    stdevs = pd.read_csv(
        str(model_pred).replace('_predictions.csv', '_stdevs.csv'),
        index_col=0,
        header=0,
        usecols=lambda col: col == 'ansi' or col.startswith('z')
    )

    pcols = predictions.columns[pd.Series(predictions.columns).str.startswith('z')]
    ztiles, nrows, ncols = map(lambda x: calc_length(x), [c.split('-') for c in pcols][-1])

    coefficients, actuators = {}, {}
    for z in trange(ztiles, desc='Aggregating Z tiles', total=ztiles):
        tiles = predictions.columns[pd.Series(predictions.columns).str.startswith(f'z{z}')]

        if ignore_tile is not None:
            for cc in ignore_tile:
                if cc in tiles:
                    predictions.loc[:, cc] = np.zeros_like(predictions.index)
                else:
                    logger.warning(f"`{cc}` was not found!")

        # filter out small predictions
        prediction_threshold = utils.waves2microns(prediction_threshold, wavelength=wavelength)
        stdevs[np.abs(predictions) < prediction_threshold] = np.inf

        # filter out unconfident predictions
        stdevs[stdevs > confidence_threshold] = np.inf

        # get tile votes per mode
        votes = predictions[tiles].values   # votes is a 2D array
        votes[votes != 0] = 1   # all predictions.values (e.g. each mode of each XY tile) that are non-zero get one vote

        # sum votes per mode
        total_votes = np.sum(votes, axis=1)  # 1D array

        # weighted by 1/stddev**2
        # weights = 1 / np.square(stdevs[tiles])

        # weighted by votes
        weights = votes

        mean_prediction, mean_stdev = weighted_avg_and_std(predictions[tiles].values, weights=weights, axis=1)
        mean_prediction = np.nan_to_num(mean_prediction) # if no votes, then set prediction to zero
        mean_stdev = np.nan_to_num(mean_stdev)  # if no votes, then set prediction to zero


        pred = Wavefront(
            mean_prediction,
            order='ansi',
            lam_detection=wavelength
        )

        pred_std = Wavefront(
            mean_stdev,
            order='ansi',
            lam_detection=wavelength
        )

        coefficients[f'z{z}'] = pred.amplitudes

        actuators[f'z{z}'] = utils.zernikies_to_actuators(
            pred.amplitudes,
            dm_calibration=dm_calibration,
            dm_state=dm_state,
            scalar=dm_damping_scalar
        )

        psf = samplepsfgen.single_psf(phi=pred, normed=True, noise=False)
        imsave(f"{model_pred.with_suffix('')}_aggregated_psf_z{z}.tif", psf)

        if plot:
            vis.diagnosis(
                pred=pred,
                pred_std=pred_std,
                save_path=Path(f"{model_pred.with_suffix('')}_aggregated_diagnosis_z{z}"),
            )

    coefficients = pd.DataFrame.from_dict(coefficients)
    coefficients.index.name = 'ansi'
    coefficients.to_csv(f"{model_pred.with_suffix('')}_aggregated_zernike_coefficients.csv")

    actuators = pd.DataFrame.from_dict(actuators)
    actuators.index.name = 'actuators'
    actuators.to_csv(f"{model_pred.with_suffix('')}_aggregated_corrected_actuators.csv")

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
    imsave(pupil_path, cp.asnumpy(pupil))

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
    imsave(f"{img.with_suffix('')}_phase_retrieval_psf.tif", psf)

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
