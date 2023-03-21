import matplotlib
matplotlib.use('Agg')

import re
import json
from functools import partial
import fnmatch
import os
import ujson

import matplotlib.pyplot as plt
plt.set_loglevel('error')

from pathlib import Path
from subprocess import call
import multiprocessing as mp
import tensorflow as tf

from typing import Any, Union, Optional, Generator
import numpy as np
import pandas as pd
from tifffile import imread, imsave
import seaborn as sns
from tqdm import tqdm
from line_profiler_pycharm import profile
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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


@profile
def reloadmodel_if_needed(
    modelpath: Path,
    preloaded: Optional[Preloadedmodelclass] = None,
    ideal_empirical_psf: Union[Path, np.ndarray] = None,
    ideal_empirical_psf_voxel_size: Any = None
):
    if preloaded is None:
        logger.info("Loading new model, because model didn't exist")
        preloaded = Preloadedmodelclass(modelpath, ideal_empirical_psf, ideal_empirical_psf_voxel_size)

    if ideal_empirical_psf is None and preloaded.ideal_empirical_psf is not None:
        logger.info("Loading new model, because ideal_empirical_psf has been removed")
        preloaded = Preloadedmodelclass(modelpath)

    elif preloaded.ideal_empirical_psf != ideal_empirical_psf:
        logger.info(f"Updating ideal psf with empirical, because {chr(10)} {preloaded.ideal_empirical_psf} of type {type(preloaded.ideal_empirical_psf)} has been changed to {chr(10)} {ideal_empirical_psf} of type {type(ideal_empirical_psf)}")
        preloaded.modelpsfgen.update_ideal_psf_with_empirical(
            ideal_empirical_psf=ideal_empirical_psf,
            voxel_size=ideal_empirical_psf_voxel_size,
            remove_background=True,
            normalize=True,
        )

    return preloaded.model, preloaded.modelpsfgen


@profile
def zernikies_to_actuators(
        coefficients: np.array,
        dm_calibration: Path,
        dm_state: np.array,
        scalar: float = 1
) -> np.ndarray:
    dm_calibration = pd.read_csv(dm_calibration, header=None).values

    if dm_calibration.shape[-1] > coefficients.size:
        # if we have <55 coefficients, crop the calibration matrix columns
        dm_calibration = dm_calibration[:, :coefficients.size]
    else:
        # if we have >55 coefficients, crop the coefficients array
        coefficients = coefficients[:dm_calibration.shape[-1]]

    offset = np.dot(dm_calibration, coefficients)
    return dm_state - (offset * scalar)


@profile
def load_dm(dm_state: Any) -> np.ndarray:
    if isinstance(dm_state, np.ndarray):
        assert len(dm_state) == 69
    elif dm_state is None or str(dm_state) == 'None':
        dm_state = np.zeros(69)
    else:
        dm_state = pd.read_csv(dm_state, header=None).values[:, 0]
    return dm_state


@profile
def estimate_and_save_new_dm(
    savepath: Path,
    coefficients: np.array,
    dm_calibration: Path,
    dm_state: np.array,
    dm_damping_scalar: float = 1
):
    dm_state = load_dm(dm_state)
    dm = pd.DataFrame(zernikies_to_actuators(
        coefficients,
        dm_calibration=dm_calibration,
        dm_state=dm_state,
        scalar=dm_damping_scalar
    ))
    dm.to_csv(savepath, index=False, header=False)
    return dm


@profile
def percentile_filter(data: np.ndarray, min_pct: int = 5, max_pct: int = 95) -> np.ndarray:
    minval, maxval = np.percentile(data, [min_pct, max_pct])
    return (data < minval) | (data > maxval)


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
def deskew(
    img: Path,
    axial_voxel_size: float,
    lateral_voxel_size: float,
    flipz: bool = False,
    skew_angle: float = 32.45,
):
    matlab = 'matlab '
    matlab += f' -wait'
    matlab += f' -nodisplay'
    matlab += f' -nosplash'
    matlab += f' -nodesktop'
    matlab += f' -r '

    flags = f"{lateral_voxel_size},{axial_voxel_size},"
    flags += f"'SkewAngle',{skew_angle},"
    flags += f"'Reverse',true,"
    flags += f"'Rotate',false,"
    flags += f"'flipZstack',{str(flipz).lower()},"
    flags += f"'Save16bit',true,"
    flags += f"'save3DStack',true,"
    flags += f"'DSRCombined',false"

    deskew = f"XR_deskewRotateFrame('{img}',{flags})"
    repo = Path(__file__).parent.parent.absolute()
    llsm = f"addpath(genpath('{repo}/LLSM3DTools/'))"
    job = f"{matlab} \"{llsm}; {deskew}; exit;\""

    print(job)
    call(job, shell=True)


@profile
def detect_rois(
    img: Path,
    axial_voxel_size: float,
    lateral_voxel_size: float,
    psf: str = 'none',
    skew_angle: float = 32.45,
    sigma_xy: float = 1.1,
    sigma_z: float = 1.1,
):
    psf = None if psf == '' else psf

    matlab = 'matlab '
    matlab += f' -wait'
    matlab += f' -nodisplay'
    matlab += f' -nosplash'
    matlab += f' -nodesktop'
    matlab += f' -r '

    det = f"TA_PointDetection('{img}','{psf}',{lateral_voxel_size},{axial_voxel_size},{skew_angle},{sigma_xy},{sigma_z})"
    repo = Path(__file__).parent.parent.absolute()
    llsm = f"addpath(genpath('{repo}/LLSM3DTools/'))"
    job = f"{matlab} \"{llsm}; {det}; exit;\""

    print(job)
    call(job, shell=True)


@profile
def decon(img: Path, psf: Path, iters: int = 10, plot: bool = False):
    matlab = 'matlab '
    matlab += f' -wait'
    matlab += f' -nodisplay'
    matlab += f' -nosplash'
    matlab += f' -nodesktop'
    matlab += f' -r '

    save_path = Path(f"{str(psf.with_suffix('')).replace('_psf', '')}_decon.tif")
    det = f"TA_Decon('{img}','{psf}',{iters}, '{save_path}')"
    repo = Path(__file__).parent.parent.absolute()
    llsm = f"addpath(genpath('{repo}/LLSM3DTools/'))"
    job = f"{matlab} \"{llsm}; {det}; exit;\""

    print(job)
    call(job, shell=True)

    if plot:
        original_image = imread(img).astype(float)
        original_image /= np.max(original_image)

        corrected_image = imread(save_path).astype(float)
        corrected_image /= np.max(corrected_image)

        vis.prediction(
            original_image=original_image,
            corrected_image=corrected_image,
            save_path=f"{save_path.with_suffix('')}_correction"
        )


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
        preloaded,
        model,
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
    wavelength: float = .510,
    ignore_modes: list = (0, 1, 2, 4),
    prediction_threshold: float = 0.,
    freq_strength_threshold: float = .01,
    batch_size: int = 1,
    digital_rotations: np.ndarray = np.arange(0, 360+1, 1).astype(int),
    plot: Optional[np.ndarray] = None,
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
            plot=None,
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
            threshold=prediction_threshold,
            ignore_modes=ignore_modes,
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

    predictions = pd.DataFrame(ps.T, columns=[f.with_suffix('').name for f in rois])
    pcols = predictions.columns[pd.Series(predictions.columns).str.startswith('z')]
    predictions['mean'] = predictions[pcols].mean(axis=1)
    predictions['median'] = predictions[pcols].median(axis=1)
    predictions['min'] = predictions[pcols].min(axis=1)
    predictions['max'] = predictions[pcols].max(axis=1)
    predictions['std'] = predictions[pcols].std(axis=1)
    predictions.index.name = 'ansi'
    predictions.to_csv(f"{outdir}_predictions.csv")

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
        preloaded,
        model,
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
        preloaded,
        model,
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
        digital_rotations=digital_rotations,
        plot=Path(f"{img.with_suffix('')}_large_fov_predictions") if plot else None,
        plot_rotations=Path(f"{img.with_suffix('')}_large_fov_predictions") if plot_rotations else None,
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
        preloaded,
        model,
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
    axial_voxel_size: float,
    lateral_voxel_size: float,
    wavelength: float = .605,
    num_predictions: int = 1,
    batch_size: int = 1,
    window_size: tuple = (64, 64, 64),
    prediction_threshold: float = 0.,
    freq_strength_threshold: float = .01,
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
        preloaded,
        model,
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
    min_percentile: int = 10,
    max_percentile: int = 90,
    prediction_threshold: float = 0.,
    final_prediction: str = 'mean',
    dm_damping_scalar: float = 1,
    plot: bool = False,
    ignore_tile: Any = None,
    preloaded: Preloadedmodelclass = None
):
    def calc_length(s):
        return int(re.sub(r'[a-z]+', '', s)) + 1

    modelpsfgen = backend.load_metadata(model) if preloaded is None else preloaded.modelpsfgen

    predictions = pd.read_csv(
        model_pred,
        index_col=0,
        header=0,
        usecols=lambda col: col == 'ansi' or col.startswith('z')
    )
    original_pcols = predictions.columns

    if ignore_tile is not None:
        for tile in ignore_tile:
            col = tile
            if col in predictions.columns:
                predictions.loc[:, col] = np.zeros_like(predictions.index)
            else:
                logger.warning(f"`{tile}` was not found!")

    # filter out small predictions
    prediction_threshold = utils.waves2microns(prediction_threshold, wavelength=wavelength)
    predictions[np.abs(predictions) < prediction_threshold] = 0.

    # drop null predictions
    predictions = predictions.loc[:, (predictions != 0).any(axis=0)]
    pcols = predictions.columns[pd.Series(predictions.columns).str.startswith('z')]

    p_modes = predictions[pcols].values
    p_modes[p_modes != 0] = 1
    p_modes = np.sum(p_modes, axis=1)
    predictions['votes'] = p_modes
    det_modes = predictions[predictions['votes'] > p_modes.max() * majority_threshold][pcols]

    if det_modes.shape[0] > 0:
        if plot:
            fig, axes = plt.subplots(nrows=det_modes.shape[0], figsize=(8, 11))

        for i in tqdm(range(det_modes.shape[0]), desc='Detecting outliers', total=det_modes.shape[0]):
            preds = det_modes.iloc[i]

            if plot:
                min_amp = det_modes.values.flatten().min()
                max_amp = det_modes.values.flatten().max()

                ax = axes[i] if det_modes.shape[0] > 1 else axes
                ax = sns.violinplot(
                    preds,
                    linewidth=1,
                    alpha=.75,
                    ax=ax,
                    color="lightgrey",
                    notch=False,
                    showcaps=False,
                    flierprops={"marker": "."},
                    boxprops={"facecolor": "lightgrey"},
                )

            outliers = preds[percentile_filter(preds.values, min_pct=min_percentile, max_pct=max_percentile)]

            if not outliers.empty and preds.shape[0] > 2:
                preds.drop(outliers.index.values, inplace=True)

            sign = preds[preds != 0]
            sign = 1 if sign[sign < 0].shape[0] <= sign[sign > 0].shape[0] else -1

            predictions.loc[det_modes.index[i], 'mean'] = np.nanmean(preds.abs()) * sign
            predictions.loc[det_modes.index[i], 'median'] = np.nanmedian(preds.abs()) * sign

            predictions.loc[det_modes.index[i], 'min'] = np.nanmin(preds)
            predictions.loc[det_modes.index[i], 'max'] = np.nanmax(preds)
            predictions.loc[det_modes.index[i], 'std'] = np.nanstd(preds)

            if plot:
                ax.plot(
                    predictions.loc[det_modes.index[i], final_prediction], np.zeros_like(1),
                    'o', clip_on=False, color='C0', label='Prediction', zorder=3
                )
                ax.plot(
                    outliers, np.zeros_like(outliers),
                    'x', clip_on=False, color='C3', label='Outliers', zorder=3
                )

                ax.spines.right.set_visible(False)
                ax.spines.left.set_visible(False)
                ax.spines.top.set_visible(False)
                ax.set_yticks([])
                ax.set_xlim(min_amp - .05, max_amp + .05)
                ax.set_ylabel(f'{det_modes.index[i]}')
                ax.set_xlabel('')

        if plot:
            if det_modes.shape[0] > 1:
                axes[0].legend(ncol=2, frameon=False)
                axes[-1].set_xlabel(f'Zernike coefficients ($\mu$m)')
            else:
                axes.legend(ncol=2, frameon=False)
                axes.set_xlabel(f'Zernike coefficients ($\mu$m)')

            vis.savesvg(fig, f"{model_pred.with_suffix('')}_aggregated.svg")
    else:
        logger.warning(f"No modes detected with the current configs")
        predictions['mean'] = predictions[pcols].mean(axis=1)
        predictions['median'] = predictions[pcols].median(axis=1)
        predictions['min'] = predictions[pcols].min(axis=1)
        predictions['max'] = predictions[pcols].max(axis=1)
        predictions['std'] = predictions[pcols].std(axis=1)

    for c in original_pcols:
        if c not in predictions.columns:
            predictions[c] = np.zeros_like(predictions.index)

    predictions.fillna(0, inplace=True)
    predictions.index.name = 'ansi'
    predictions.to_csv(f"{model_pred.with_suffix('')}_aggregated.csv")

    estimate_and_save_new_dm(
        savepath=Path(f"{model_pred.with_suffix('')}_aggregated_corrected_actuators.csv"),
        coefficients=predictions[final_prediction].values,
        dm_calibration=dm_calibration,
        dm_state=dm_state,
        dm_damping_scalar=dm_damping_scalar
    )

    p = Wavefront(predictions[final_prediction].values, order='ansi', lam_detection=wavelength)
    pred_std = Wavefront(predictions['std'].values, order='ansi', lam_detection=wavelength)

    coefficients = [
        {'n': z.n, 'm': z.m, 'amplitude': a}
        for z, a in p.zernikes.items()
    ]
    coefficients = pd.DataFrame(coefficients, columns=['n', 'm', 'amplitude'])
    coefficients.index.name = 'ansi'
    coefficients.to_csv(f"{model_pred.with_suffix('')}_aggregated_zernike_coefficients.csv")

    psfgen = SyntheticPSF(
        psf_type=modelpsfgen.psf_type,
        snr=100,
        psf_shape=modelpsfgen.psf_shape,
        n_modes=predictions[final_prediction].shape[0],
        lam_detection=wavelength,
        x_voxel_size=lateral_voxel_size,
        y_voxel_size=lateral_voxel_size,
        z_voxel_size=axial_voxel_size,
    )

    psf = psfgen.single_psf(phi=p, normed=True, noise=False)
    imsave(f"{model_pred.with_suffix('')}_aggregated_psf.tif", psf)

    ztiles, nrows, ncols = [c.split('-') for c in pcols][-1]

    if plot:
        vis.wavefronts(
            ztiles=calc_length(ztiles),
            nrows=calc_length(nrows),
            ncols=calc_length(ncols),
            predictions=predictions,
            wavelength=wavelength,
            save_path=Path(f"{model_pred.with_suffix('')}_aggregated_wavefronts"),
        )

        vis.diagnosis(
            pred=p,
            pred_std=pred_std,
            save_path=Path(f"{model_pred.with_suffix('')}_aggregated_diagnosis"),
        )

    return coefficients


def plot_dm_actuators(
    dm_path: Path,
    flat_path: Path,
    save_path: Path,
    pred_path: Any = None
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

    with open(dm_path) as f:
        offsets = json.load(f)

    offsets = np.array(offsets["ALPAO_Offsets"])

    if flat_path.suffix == '.json':
        with open(flat_path) as f:
            flat_offsets = json.load(f)

        flat_offsets = np.array(flat_offsets["ALPAO_Offsets"])
    else:
        flat_offsets = pd.read_csv(flat_path, header=None).iloc[:, 0].values

    if pred_path is not None:
        pred_offsets = pd.read_csv(pred_path, header=None).iloc[:, 0].values
    else:
        pred_offsets = None

    mask = np.ones((9, 9))
    dm = np.zeros((9, 9))

    for x, y in [
        (0, 0),
        (1, 0), (0, 1),
        (7, 0), (0, 7),
        (8, 0), (0, 8),
        (8, 1), (1, 8),
        (7, 8), (8, 7),
        (8, 8)
    ]:
        mask[x, y] = 0

    dm[mask.astype(bool)] = offsets - flat_offsets

    fig = plt.figure(figsize=(11, 4))
    gs = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(gs[0, -1])
    ax2 = fig.add_subplot(gs[0, :-1])

    m = ax1.imshow(dm.T, cmap='Spectral_r', vmin=-1*dm.max(), vmax=dm.max())
    ax1.set_xticks(range(9))
    ax1.set_yticks(range(9))
    cax = inset_axes(ax1, width="10%", height="100%", loc='center right', borderpad=-3)
    cb = plt.colorbar(m, cax=cax)
    cax.yaxis.set_label_position("right")

    ax2.plot(flat_offsets, '--', label='Flat')
    ax2.plot(offsets, label='DM')

    if pred_offsets is not None:
        ax2.plot(pred_offsets, ':', label='Predictions')

    ax2.grid(True, which="both", axis='both', lw=.1, ls='--', zorder=0)
    ax2.legend(frameon=False, ncol=2, loc='upper right')
    vis.savesvg(fig, save_path)


def eval_mode(
    input_path: Path,
    prediction_path: Path,
    gt_path: Path,
    model_path: Path,
    flat_path: Any = None,
    normalize: bool = True,
    remove_background: bool = True,
    postfix: str = '',
    gt_postfix: str = '',
    plot: bool = True,
):
    logger.info(f"Pred: {prediction_path.name}")
    logger.info(f"GT: {gt_path.name}")

    save_postfix = 'pr' if postfix.startswith('phase_retrieval') else 'ml'
    save_path = Path(f'{prediction_path.parent}/{prediction_path.stem}_{save_postfix}_eval')

    with open(str(prediction_path).replace('_zernike_coefficients.csv', '_settings.json')) as f:
        predictions_settings = ujson.load(f)

    noisy_img = np.squeeze(get_image(input_path).astype(float))
    maxcounts = np.max(noisy_img)
    psnr = predictions_settings['psnr']
    gen = backend.load_metadata(
        model_path,
        snr=psnr,
        psf_shape=noisy_img.shape,
        psf_type='widefield' if save_postfix == 'pr' else None,
        z_voxel_size=.1 if save_postfix == 'pr' else None,
    )

    if prediction_path is None:
        predict_sample(
            img=input_path,
            model=model_path,
            wavelength=gen.lam_detection,
            axial_voxel_size=gen.z_voxel_size,
            lateral_voxel_size=gen.x_voxel_size,
            prev=None,
            dm_state=None,
            dm_calibration=None,
            prediction_threshold=0.,
        )
        prediction_path = Path(
            f"{str(gt_path.with_suffix('')).replace(f'_{gt_postfix}', '')}_{postfix}"
        )

    try:
        p = pd.read_csv(prediction_path, header=0)['amplitude'].values
    except KeyError:
        p = pd.read_csv(prediction_path, header=None).iloc[:, 0].values

    if gt_path.suffix == '.tif':
        y = gt_path
    elif gt_path.suffix == '.csv':
        try:
            y = pd.read_csv(gt_path, header=0)['amplitude'].values[:len(p)]
        except KeyError:
            y = pd.read_csv(gt_path, header=None).iloc[:, -1].values[:len(p)]
    else:
        y = np.zeros_like(p)

    p_wave = Wavefront(p, lam_detection=gen.lam_detection, modes=len(p))
    y_wave = Wavefront(y, lam_detection=gen.lam_detection, modes=len(p))
    diff = Wavefront(y-p, lam_detection=gen.lam_detection, modes=len(p))

    if flat_path is not None:
        rfilter = f"{str(gt_path.name).replace(gt_postfix, '')}"
        dm_path = Path(str(list(input_path.parent.glob(f"{rfilter}*JSONsettings.json"))[0]))
        dm_wavefront = Path(gt_path.parent/f"{rfilter}_dm_wavefront.svg")

        plot_dm_actuators(
            dm_path=dm_path,
            flat_path=flat_path,
            save_path=dm_wavefront
        )

    if plot:
        prep = partial(
            preprocessing.prep_sample,
            normalize=normalize,
            remove_background=remove_background,
            model_fov=gen.psf_fov,
            plot=input_path.with_suffix(''),
        )

        noisy_img = prep(noisy_img, sample_voxel_size=predictions_settings['sample_voxel_size'])
        psfgen = backend.load_metadata(
            model_path,
            snr=psnr,
            psf_shape=(64, 64, 64),
            psf_type='widefield' if save_postfix == 'pr' else None,
            z_voxel_size=.1 if save_postfix == 'pr' else None,
        )
        p_psf = psfgen.single_psf(p_wave, normed=True, noise=False)
        gt_psf = psfgen.single_psf(y_wave, normed=True, noise=False)
        corrected_psf = psfgen.single_psf(diff, normed=True, noise=False)

        plt.style.use("default")
        vis.diagnostic_assessment(
            psf=noisy_img,
            gt_psf=gt_psf,
            predicted_psf=p_psf,
            corrected_psf=corrected_psf,
            psnr=psnr,
            maxcounts=maxcounts,
            y=y_wave,
            pred=p_wave,
            save_path=save_path,
            display=False,
            dxy=gen.x_voxel_size,
            dz=gen.z_voxel_size,
            transform_to_align_to_DM=True,
        )

    residuals = [
        {
            'n': z.n,
            'm': z.m,
            'prediction': p_wave.zernikes[z],
            'ground_truth': y_wave.zernikes[z],
            'residuals': diff.zernikes[z],
            'psnr': psnr,
        }
        for z in p_wave.zernikes.keys()
    ]

    residuals = pd.DataFrame(residuals, columns=['n', 'm', 'prediction', 'ground_truth', 'residuals', 'psnr'])
    residuals.index.name = 'ansi'
    residuals.to_csv(f'{save_path}_residuals.csv')


def process_eval_file(file: Path, nas=(1.0, .95, .85)):
    results = {}
    iteration_labels = [
        'before',
        'after0',
        'after1',
        'after2',
        'after3',
        'after4',
        'after5',
    ]

    state = file.stem.split('_')[0]
    modes = ':'.join(s.lstrip('z') if s.startswith('z') else '' for s in file.stem.split('_')).split(':')
    modes = [m for m in modes if m.isdigit()]
    res = pd.read_csv(file)

    p = Wavefront(res['prediction'].values, modes=res.shape[0])
    y = Wavefront(res['ground_truth'].values, modes=res.shape[0])
    diff = Wavefront(res['residuals'].values, modes=res.shape[0])
    file = Path(file)
    eval_file = Path(str(file).replace('_residuals.csv', '.svg'))

    for i, na in enumerate(nas):
        results[i] = {
            'modes': '-'.join(str(e) for e in modes),
            'state': state,
            'iteration_index': iteration_labels.index(state),
            'num_model_modes': p.modes,
            'eval_file': str(eval_file),
            'na': na,
            'p2v_residual': diff.peak2valley(na=na),
            'p2v_gt': y.peak2valley(na=na),
            'p2v_pred': p.peak2valley(na=na),
            f'mode_1': modes[0],
            f'mode_2': None if len(modes) < 2 else modes[1],
            'psnr': np.mean(res['psnr']),
        }

        if len(modes) > 1 and modes[0] != modes[1]:
            results[i+len(nas)] = {
                'modes': '-'.join(str(e) for e in modes[::-1]),
                'state': state,
                'iteration_index': iteration_labels.index(state),
                'num_model_modes': p.modes,
                'eval_file': str(eval_file),
                'na': na,
                'p2v_residual': diff.peak2valley(na=na),
                'p2v_gt': y.peak2valley(na=na),
                'p2v_pred': p.peak2valley(na=na),
                f'mode_1': None if len(modes) < 2 else modes[1],
                f'mode_2': modes[0],
                'psnr': np.mean(res['psnr']),
            }   # if we have mixed modes, duplicate for the opposite combination (e.g. 12,13 copied to -> 13,12)

    return results


def plot_eval_dataset(
    model,
    datadir: Path,
    postfix: str = 'predictions_zernike_coefficients_ml_eval_residuals.csv',
):
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.autolimit_mode': 'round_numbers'
    })

    savepath = Path(f'{datadir}/p2v_eval')

    # Permanently changes the pandas settings
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 600)
    pd.set_option('display.max_colwidth', 20)

    results = utils.multiprocess(
        func=process_eval_file,
        jobs=sorted(datadir.rglob(f'*{postfix}'), key=os.path.getctime),  # sort by creation time
        desc=f'Collecting *{postfix} results'
    )

    if results == []:
        logger.error(f'Did not find results in {datadir}\\*{postfix}    Please reurun without --precomputed flag.')
        return

    df = pd.DataFrame([v for d in results for k, v in d.items()])
    df.sort_values(by=['modes', 'iteration_index', 'na'], ascending=[True, True, False], inplace=True)
    df['model'] = str(model)

    df.to_csv(f'{savepath}.csv')
    logger.info(f'{savepath}.csv')

    for col, label in zip(
            ["p2v_gt", "p2v_residual", "psnr"],
            [r"Remaining aberration (P-V $\lambda$)", "PR-Model (P-V $\lambda$)", "PSNR"]
    ):
        fig = plt.figure(figsize=(11, 8))
        g = sns.relplot(
            data=df,
            x="iteration_index",
            y=col,
            hue="na",
            col="mode_1",
            col_wrap=4,
            kind="line",
            height=3,
            aspect=1.,
            palette='tab10',
            ci='sd',
            # units="modes",
            # estimator=None,
            facet_kws=dict(sharex=True),
        )

        (
            g.map(plt.axhline, y=.5, color="red", dashes=(2, 1), zorder=3)
            .map(plt.grid, which="both", axis='both', lw=.25, ls='--', zorder=0, color='lightgrey')
            .set_axis_labels("Iteration", label)
            .set_titles("Mode: {col_name}")
            .set(xlim=(0, max(df['iteration_index'])))
            .set(ylim=(0, np.ceil(np.max(df['psnr']))) if col == 'psnr' else (0, 5))
            .tight_layout(w_pad=0)
        )

        leg = g._legend
        leg.set_bbox_to_anchor([.95, .93])
        leg.set_title('NA')
        #g.fig.suptitle(f"{df['num_model_modes'].unique()} mode Model")

        plt.subplots_adjust(top=0.95, right=0.95, wspace=.1, hspace=.2)
        plt.savefig(f'{savepath}_{col}.png', dpi=300, bbox_inches='tight', pad_inches=.25)
        logger.info(f'{savepath}_{col}.png')


@profile
def eval_dataset(
    datadir: Path,
    flat: Any = None,
    postfix: str = 'predictions_zernike_coefficients.csv',
    gt_postfix: str = 'phase_retrieval_zernike_coefficients.csv',
    plot_evals: bool = True,
    precomputed: bool = False
):
    # get model from .json file
    with open(list(Path(datadir / 'MLResults').glob('*_settings.json'))[0]) as f:
        predictions_settings = ujson.load(f)
        model = Path(predictions_settings['model'])

        if not model.exists():
            logger.error(f'Model not found {model}')
            return

    if not precomputed:
        pool = mp.Pool(processes=mp.cpu_count())
        MLresultsdir = Path(datadir / 'MLResults')
        MLresults_list = list(MLresultsdir.glob('**/*'))    # only get directory list once for speed

        evaluate = partial(
            eval_mode,
            model_path=model,
            flat_path=flat,
            postfix=postfix,
            gt_postfix=gt_postfix,
            plot=plot_evals,
        )

        logger.info('Beginning evaluations')
        for file in sorted(datadir.glob('*_lightsheet_ansi_z*.tif'), key=os.path.getctime):  # sort by creation time
            if 'CamB' in str(file) or 'pupil' in str(file) or 'autoexpos' in str(file):
                continue

            state = file.stem.split('_')[0]
            modes = ':'.join(s.lstrip('z') if s.startswith('z') else '' for s in file.stem.split('_')).split(':')
            modes = [m for m in modes if m]
            logger.info(f"ansi_z{modes}")

            if len(modes) > 1:
                prefix = f"ansi_"
                for m in modes:
                    prefix += f"z{m}*"
            else:
                mode = modes[0]
                prefix = f"ansi_z{mode}*"

            gt_path = None
            for gtfile in MLresults_list:
                if fnmatch.fnmatch(gtfile.name, f'{state}_widefield_{prefix}_{gt_postfix}'):
                    gt_path = gtfile
                    continue
            if gt_path is None:
                logger.warning(f'GT not found for: {state}_widefield_{prefix}_*.tif')

            prediction_path = None
            for predfile in MLresults_list:
                if fnmatch.fnmatch(predfile.name, f'{state}_lightsheet_{prefix}_{postfix}'):
                    prediction_path = predfile
                    continue
            if prediction_path is None: logger.warning(f'Prediction not found for: {file.name}')

            task = partial(
                evaluate,
                input_path=file,
                prediction_path=prediction_path,
                gt_path=gt_path
            )
            _ = pool.apply_async(task)  # issue task

        pool.close()    # close the pool
        pool.join()     # wait for all tasks to complete

    plot_eval_dataset(model, datadir)


@profile
def eval_dm(
    datadir: Path,
    num_modes: int = 15,
    gt_postfix: str = 'pr_pupil_waves.tif',
    # gt_postfix: str = 'ground_truth_zernike_coefficients.csv',
    postfix: str = 'sample_predictions_zernike_coefficients.csv'
):

    data = np.identity(num_modes)
    for file in sorted(datadir.glob('*_lightsheet_ansi_z*.tif')):
        if 'CamB' in str(file):
            continue

        state = file.stem.split('_')[0]
        modes = ':'.join(s.lstrip('z') if s.startswith('z') else '' for s in file.stem.split('_')).split(':')
        modes = [m for m in modes if m]
        logger.info(modes)
        logger.info(f"Input: {file.name[:75]}....tif")

        if len(modes) > 1:
            prefix = f"ansi_"
            for m in modes:
                prefix += f"z{m}*"
        else:
            mode = modes[0]
            prefix = f"ansi_z{mode}*"

        logger.info(f"Looking for: {prefix}")

        try:
            gt_path = list(datadir.rglob(f'{state}_widefield_{prefix}_{gt_postfix}'))[0]
            logger.info(f"GT: {gt_path.name}")
        except IndexError:
            logger.warning(f'GT not found for: {file.name}')
            continue

        try:
            prediction_path = list(datadir.rglob(f'{state}_lightsheet_{prefix}_{postfix}'))[0]
            logger.info(f"Pred: {prediction_path.name}")
        except IndexError:
            logger.warning(f'Prediction not found for: {file.name}')
            prediction_path = None

        try:
            p = pd.read_csv(prediction_path, header=0)['amplitude'].values
        except KeyError:
            p = pd.read_csv(prediction_path, header=None).iloc[:, 0].values

        try:
            y = pd.read_csv(gt_path, header=0)['amplitude'].values[:len(p)]
        except KeyError:
            y = pd.read_csv(gt_path, header=0).iloc[:, -1].values[:len(p)]

        magnitude = y[np.argmax(np.abs(y))]

        for i in range(p.shape[0]):
            data[i, int(mode)] = p[i] / magnitude   # normalize by the magnitude of the mode we put on the mirror

    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.heatmap(
        data, ax=ax, annot=True, fmt=".2f", vmin=-1, vmax=1,
        cmap='coolwarm', square=True, cbar_kws={'label': 'Ratio of ML prediction to GT', 'shrink': .8}
    )
    ax.set(ylabel="ML saw these modes", xlabel="DM applied this mode",)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    for t in ax.texts:
        if abs(float(t.get_text())) >= 0.01:
            t.set_text(t.get_text())
        else:
            t.set_text("")

    ax.set_title(f'DM magnitude = {magnitude} um RMS {chr(10)} {datadir.parts[-2]}')

    output_file = Path(f'{datadir}/../{datadir.parts[-1]}_dm_matrix')
    df.to_csv(f"{output_file}.csv")
    plt.savefig(f"{output_file}.png", bbox_inches='tight', pad_inches=.25)
    logger.info(f"Saved result to: {output_file}")


def calibrate_dm(datadir, dm_calibration):
    dataframes = []
    for file in sorted(datadir.glob('*dm_matrix.csv')):
        df = pd.read_csv(file, header=0, index_col=0)
        dataframes.append(df)

    df = pd.concat(dataframes)
    avg = df.groupby(df.index).mean()
    dm = pd.read_csv(dm_calibration, header=None)

    scalers = np.identity(dm.shape[1])
    scalers[np.diag_indices_from(avg)] /= np.diag(avg)
    calibration = np.dot(dm, scalers)
    calibration = pd.DataFrame(calibration)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.heatmap(avg, ax=ax, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap='coolwarm', square=True,
                     cbar_kws={'label': 'Ratio of ML prediction to GT', 'shrink': .8})
    ax.set(ylabel="ML saw these modes", xlabel="DM applied this mode", )
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    for t in ax.texts:
        if abs(float(t.get_text())) >= 0.01:
            t.set_text(t.get_text())
        else:
            t.set_text("")

    output_file = Path(f"{datadir}/calibration")
    plt.savefig(f"{output_file}.png", bbox_inches='tight', pad_inches=.25)

    dm = calibration / dm
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.heatmap(dm, ax=ax, vmin=0, vmax=2, cmap='coolwarm', square=True, cbar_kws={'shrink': .8})
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set(ylabel="Actuators", xlabel="Zernike modes", )
    plt.savefig(f"{output_file}_diff.png", bbox_inches='tight', pad_inches=.25)

    calibration.to_csv(f"{output_file}.csv", header=False, index=False)
    logger.info(f"Saved result to: {output_file}")


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
