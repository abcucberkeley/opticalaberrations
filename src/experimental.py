import matplotlib
matplotlib.use('Agg')

import re
import json
import time
from functools import partial
from pathlib import Path
from subprocess import call
import multiprocessing as mp
import tensorflow as tf

from typing import Any, Sequence, Union
import numpy as np
from scipy import stats as st
import pandas as pd
from tifffile import imread, imsave
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from line_profiler_pycharm import profile
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pyotf.phaseretrieval as pr
from pyotf.utils import prep_data_for_PR
from pyotf.zernike import osa2degrees

import utils
import vis
import preprocessing
from synthetic import SyntheticPSF
from wavefront import Wavefront
from data_utils import get_image
from preloaded import Preloadedmodelclass
from backend import load_metadata, dual_stage_prediction, predict_rotation

import logging
logger = logging.getLogger('')


@profile
def reloadmodel_if_needed(
    preloaded: Preloadedmodelclass,
    modelpath: Path,
    ideal_empirical_psf: Any = None,
    ideal_empirical_psf_voxel_size: Any = None
):
    if preloaded is None:
        logger.info("Loading new model, because model didn't exist")
        preloaded = Preloadedmodelclass(modelpath, ideal_empirical_psf, ideal_empirical_psf_voxel_size)

    if ideal_empirical_psf is None and preloaded.ideal_empirical_psf is not None:
        logger.info("Loading new model, because ideal_empirical_psf has been removed")
        preloaded = Preloadedmodelclass(modelpath)


    elif preloaded.ideal_empirical_psf != ideal_empirical_psf:
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
        dm_calibration = dm_calibration[:, :coefficients.size]  # if we have <55 coefficients, crop the calibration matrix columns
    else:
        coefficients = coefficients[:dm_calibration.shape[-1]]  # if we have >55 coefficients, crop the coefficients array

    #coefficients = np.expand_dims(coefficients, axis=-1)
    #offset = np.dot(dm_calibration, coefficients)[:, 0]
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


@profile
def load_sample(
    data: Path,
    model_voxel_size: tuple,
    sample_voxel_size: tuple,
    remove_background: bool = True,
    normalize: bool = True,
):
    try:
        if isinstance(data, np.ndarray):
            img = data
        elif isinstance(data, tf.Tensor):
            path = Path(str(data.numpy(), "utf-8"))
            img = get_image(path).astype(float)
        else:
            path = Path(str(data))
            img = get_image(path).astype(float)

        img = preprocessing.prep_sample(
            np.squeeze(img),
            model_voxel_size=model_voxel_size,
            sample_voxel_size=sample_voxel_size,
            remove_background=remove_background,
            normalize=normalize
        )

        return img

    except Exception as e:
        logger.warning(e)


@profile
def predict(
    rois: list,
    data: Path,
    model: Path,
    axial_voxel_size: float,
    lateral_voxel_size: float,
    wavelength: float = .605,
    prev: Any = None,
    estimate_sign_with_decon: bool = False,
    ignore_modes: list = (0, 1, 2, 4),
    prediction_threshold: float = 0.,
    freq_strength_threshold: float = .01,
    sign_threshold: float = .9,
    num_predictions: int = 1,
    batch_size: int = 1,
    plot: bool = True,
    ztiles: int = 1,
    nrows: int = 1,
    ncols: int = 1,
    preloaded: Preloadedmodelclass = None,
    ideal_empirical_psf: Any = None,
):
    def summarize_predictions(data):
        columns = []
        for z in range(ztiles):
            for y in range(nrows):
                for x in range(ncols):
                    columns.append(f"p-z{z}-y{y}-x{x}")

        df = pd.DataFrame(data.T, columns=columns)
        pcols = df.columns[pd.Series(df.columns).str.startswith('p')]

        df['mean'] = df[pcols].mean(axis=1)
        df['median'] = df[pcols].median(axis=1)
        df['min'] = df[pcols].min(axis=1)
        df['max'] = df[pcols].max(axis=1)
        df['std'] = df[pcols].std(axis=1)

        df.index.name = 'ansi'
        return df

    model, modelpsfgen = reloadmodel_if_needed(
        preloaded,
        model,
        ideal_empirical_psf=ideal_empirical_psf,
        ideal_empirical_psf_voxel_size=(axial_voxel_size, lateral_voxel_size, lateral_voxel_size)
    )
    
    psfgen = SyntheticPSF(
        psf_type=modelpsfgen.psf_type,
        snr=100,
        psf_shape=modelpsfgen.psf_shape,
        n_modes=model.output_shape[1],
        lam_detection=wavelength,
        x_voxel_size=lateral_voxel_size,
        y_voxel_size=lateral_voxel_size,
        z_voxel_size=axial_voxel_size,
    )

    load = partial(
        load_sample,
        model_voxel_size=modelpsfgen.voxel_size,
        sample_voxel_size=psfgen.voxel_size,
        remove_background=True,
        normalize=True,
    )

    rois = np.array(utils.multiprocess(load, rois, desc='Processing ROIs'))
    logger.info(rois.shape)

    no_phase = True if model.input_shape[1] == 3 else False

    if no_phase:
        preds, stds, pchange = dual_stage_prediction(
            model,
            batch_size=batch_size,
            inputs=rois[..., np.newaxis],
            threshold=prediction_threshold,
            sign_threshold=sign_threshold,
            n_samples=num_predictions,
            verbose=True,
            gen=psfgen,
            modelgen=modelpsfgen,
            prev_pred=prev,
            estimate_sign_with_decon=estimate_sign_with_decon,
            ignore_modes=ignore_modes,
            freq_strength_threshold=freq_strength_threshold,
            plot=Path(f"{data.with_suffix('')}_predictions") if plot else None,
        )
    else:
        preds, stds = predict_rotation(
            model,
            batch_size=batch_size,
            inputs=rois[..., np.newaxis],
            threshold=prediction_threshold,
            sign_threshold=sign_threshold,
            verbose=True,
            gen=psfgen,
            modelgen=modelpsfgen,
            prev_pred=prev,
            estimate_sign_with_decon=estimate_sign_with_decon,
            ignore_modes=ignore_modes,
            freq_strength_threshold=freq_strength_threshold,
            plot=Path(f"{data.with_suffix('')}_predictions") if plot else None,
        )
        pchange = None

    predictions = summarize_predictions(preds)
    predictions.to_csv(f"{data}_predictions.csv")

    if pchange is not None:
        pchanges = summarize_predictions(pchange)
        pchanges.to_csv(f"{data}_predictions_percent_changes.csv")

    if plot:
        vis.wavefronts(
            scale='max',
            predictions=predictions,
            nrows=nrows,
            ncols=ncols,
            ztiles=ztiles,
            wavelength=wavelength,
            save_path=Path(f"{data.with_suffix('')}_predictions_wavefronts"),
        )


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
    num_predictions: int = 1,
    batch_size: int = 1,
    prev: Any = None,
    estimate_sign_with_decon: bool = False,
    ignore_modes: list = (0, 1, 2, 4),
    preloaded: Preloadedmodelclass = None,
    ideal_empirical_psf: Any = None,
):
    dm_state = None if (dm_state is None or str(dm_state) == 'None') else dm_state

    model, modelpsfgen = reloadmodel_if_needed(
        preloaded,
        model,
        ideal_empirical_psf=ideal_empirical_psf,
        ideal_empirical_psf_voxel_size=(axial_voxel_size, lateral_voxel_size, lateral_voxel_size)
    )

    psfgen = SyntheticPSF(
        psf_type=modelpsfgen.psf_type,
        snr=100,
        psf_shape=modelpsfgen.psf_shape,
        n_modes=model.output_shape[1],
        lam_detection=wavelength,
        x_voxel_size=lateral_voxel_size,
        y_voxel_size=lateral_voxel_size,
        z_voxel_size=axial_voxel_size
    )

    inputs = load_sample(
        img,
        model_voxel_size=modelpsfgen.voxel_size,
        sample_voxel_size=psfgen.voxel_size,
        remove_background=True,
        normalize=True,
    )

    inputs = np.expand_dims(inputs, axis=0)
    no_phase = True if model.input_shape[1] == 3 else False

    if no_phase:
        p, std, pchange = dual_stage_prediction(
            model,
            inputs=inputs,
            threshold=prediction_threshold,
            sign_threshold=sign_threshold,
            n_samples=num_predictions,
            verbose=verbose,
            gen=psfgen,
            modelgen=modelpsfgen,
            batch_size=batch_size,
            prev_pred=prev,
            estimate_sign_with_decon=estimate_sign_with_decon,
            ignore_modes=ignore_modes,
            freq_strength_threshold=freq_strength_threshold,
            plot=Path(f"{img.with_suffix('')}_sample_predictions") if plot else None,
        )
    else:
        p, std = predict_rotation(
            model,
            inputs=inputs,
            psfgen=modelpsfgen,
            no_phase=False,
            verbose=verbose,
            batch_size=batch_size,
            threshold=prediction_threshold,
            ignore_modes=ignore_modes,
            freq_strength_threshold=freq_strength_threshold,
            plot=Path(f"{img.with_suffix('')}_sample_predictions") if plot else None,
        )

    p = Wavefront(p, order='ansi', lam_detection=wavelength)
    std = Wavefront(std, order='ansi', lam_detection=wavelength)

    coefficients = [
        {'n': z.n, 'm': z.m, 'amplitude': a}
        for z, a in p.zernikes.items()
    ]
    coefficients = pd.DataFrame(coefficients, columns=['n', 'm', 'amplitude'])
    coefficients.index.name = 'ansi'
    coefficients.to_csv(f"{img.with_suffix('')}_sample_predictions_zernike_coefficients.csv")

    if dm_calibration is not None:
        dm_state = load_dm(dm_state)
        estimate_and_save_new_dm(
            savepath=Path(f"{img.with_suffix('')}_sample_predictions_corrected_actuators.csv"),
            coefficients=coefficients['amplitude'].values,
            dm_calibration=dm_calibration,
            dm_state=dm_state,
            dm_damping_scalar=dm_damping_scalar
        )

    psf = psfgen.single_psf(phi=p, normed=True, noise=False)
    imsave(f"{img.with_suffix('')}_sample_predictions_psf.tif", psf)

    if plot:
        vis.diagnosis(
            pred=p,
            pred_std=std,
            save_path=Path(f"{img.with_suffix('')}_sample_predictions_diagnosis"),
        )


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
    window_size: int = 64,
    num_rois: int = 10,
    min_intensity: int = 200,
    prediction_threshold: float = 0.,
    freq_strength_threshold: float = .01,
    sign_threshold: float = .9,
    minimum_distance: float = 1.,
    plot: bool = False,
    prev: Any = None,
    estimate_sign_with_decon: bool = False,
    ignore_modes: list = (0, 1, 2, 4),
    preloaded: Preloadedmodelclass = None,
    ideal_empirical_psf: Any = None,
):
    sample = imread(img).astype(float)
    esnr = np.sqrt(sample.max()).astype(int)

    mode = st.mode(sample[sample < np.quantile(sample, .99)], axis=None).mode[0]
    sample -= mode
    sample[sample < 0] = 0
    sample = sample / np.nanmax(sample)

    outdir = Path(f"{img.with_suffix('')}_rois")
    logger.info(f"Sample: {sample.shape}")

    rois = preprocessing.find_roi(
        sample,
        # savepath=outdir,
        peaks=pois,
        window_size=tuple(3*[window_size]),
        plot=f"{outdir}_predictions" if plot else None,
        num_peaks=num_rois,
        min_dist=minimum_distance,
        max_dist=None,
        max_neighbor=20,
        min_intensity=min_intensity,
        voxel_size=(axial_voxel_size, lateral_voxel_size, lateral_voxel_size),
    )

    ncols = int(np.ceil(len(rois)/5))
    nrows = int(np.ceil(len(rois)/ncols))

    predict(
        rois=rois,
        data=outdir,
        model=model,
        axial_voxel_size=axial_voxel_size,
        lateral_voxel_size=lateral_voxel_size,
        prediction_threshold=prediction_threshold,
        sign_threshold=sign_threshold,
        num_predictions=num_predictions,
        batch_size=batch_size,
        wavelength=wavelength,
        prev=prev,
        ztiles=1,
        nrows=ncols,
        ncols=nrows,
        estimate_sign_with_decon=estimate_sign_with_decon,
        ignore_modes=ignore_modes,
        freq_strength_threshold=freq_strength_threshold,
        preloaded=preloaded,
        ideal_empirical_psf=ideal_empirical_psf
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
    window_size: int = 64,
    prediction_threshold: float = 0.,
    freq_strength_threshold: float = .01,
    sign_threshold: float = .9,
    plot: bool = True,
    prev: Any = None,
    estimate_sign_with_decon: bool = False,
    ignore_modes: list = (0, 1, 2, 4),    
    preloaded: Preloadedmodelclass = None,
    ideal_empirical_psf: Any = None,
):

    preloadedmodel, premodelpsfgen = reloadmodel_if_needed(
        preloaded,
        model,
        ideal_empirical_psf=ideal_empirical_psf,
        ideal_empirical_psf_voxel_size=(axial_voxel_size, lateral_voxel_size, lateral_voxel_size)
    )

    sample = load_sample(
        img,
        model_voxel_size=premodelpsfgen.voxel_size,
        sample_voxel_size=(axial_voxel_size, lateral_voxel_size, lateral_voxel_size),
        remove_background=True,
        normalize=True,
    )
    outdir = Path(f"{img.with_suffix('')}_tiles")
    logger.info(f"Sample: {sample.shape}")

    windows, ztiles, nrows, ncols = preprocessing.get_tiles(
        sample,
        # savepath=outdir,
        strides=window_size,
        window_size=tuple(3*[window_size]),
    )

    predict(
        rois=windows,
        data=outdir,
        model=model,
        axial_voxel_size=premodelpsfgen.z_voxel_size,
        lateral_voxel_size=premodelpsfgen.x_voxel_size,
        prediction_threshold=prediction_threshold,
        sign_threshold=sign_threshold,
        num_predictions=num_predictions,
        batch_size=batch_size,
        wavelength=wavelength,
        prev=prev,
        plot=plot,
        ztiles=ztiles,
        nrows=nrows,
        ncols=ncols,
        estimate_sign_with_decon=estimate_sign_with_decon,
        ignore_modes=ignore_modes,
        freq_strength_threshold=freq_strength_threshold,
        preloaded=preloaded,
        ideal_empirical_psf=ideal_empirical_psf
    )

    if plot:
        vis.tiles(
            data=sample,
            strides=window_size,
            window_size=window_size,
            save_path=Path(f"{outdir.with_suffix('')}_predictions_mips"),
        )


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

    modelpsfgen = load_metadata(model) if preloaded is None else preloaded.modelpsfgen

    predictions = pd.read_csv(
        model_pred,
        index_col=0,
        header=0,
        usecols=lambda col: col == 'ansi' or col.startswith('p')
    )
    original_pcols = predictions.columns

    if ignore_tile is not None:
        for tile in ignore_tile:
            col = f"p-{tile}"
            if col in predictions.columns:
                predictions.loc[:, col] = np.zeros_like(predictions.index)
            else:
                logger.warning(f"`{tile}` was not found!")

    # filter out small predictions
    prediction_threshold = utils.waves2microns(prediction_threshold, wavelength=wavelength)
    predictions[np.abs(predictions) < prediction_threshold] = 0.

    # drop null predictions
    predictions = predictions.loc[:, (predictions != 0).any(axis=0)]
    pcols = predictions.columns[pd.Series(predictions.columns).str.startswith('p')]

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

            plt.tight_layout()
            plt.savefig(f"{model_pred.with_suffix('')}_aggregated.svg", bbox_inches='tight', dpi=300, pad_inches=.25)
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

    dm_state = load_dm(dm_state)
    dm = estimate_and_save_new_dm(
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

    _, ztiles, nrows, ncols = [c.split('-') for c in pcols][-1]

    if plot:
        vis.wavefronts(
            ztiles=calc_length(ztiles),
            nrows=calc_length(nrows),
            ncols=calc_length(ncols),
            scale='max',
            predictions=predictions,
            wavelength=wavelength,
            save_path=Path(f"{model_pred.with_suffix('')}_aggregated_wavefronts"),
        )

        vis.diagnosis(
            pred=p,
            pred_std=pred_std,
            save_path=Path(f"{model_pred.with_suffix('')}_aggregated_diagnosis"),
        )


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
        f.close()

    offsets = np.array(offsets["ALPAO_Offsets"])

    if flat_path.suffix == '.json':
        with open(flat_path) as f:
            flat_offsets = json.load(f)
            f.close()
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
    plt.savefig(save_path)


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
):
    save_postfix = 'pr' if postfix.startswith('pr') else 'ml'

    noisy_img = np.squeeze(get_image(input_path).astype(float))
    maxcounts = np.max(noisy_img)
    psnr = np.sqrt(maxcounts)
    gen = load_metadata(
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
            f"{str(gt_path.with_suffix('')).replace(f'_{gt_postfix}', '')}_sample_predictions_zernike_coefficients.csv"
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
    diff = y_wave - p_wave

    prep = partial(
        preprocessing.prep_sample,
        normalize=normalize,
        remove_background=remove_background,
        sample_voxel_size=gen.voxel_size,
        model_voxel_size=gen.voxel_size,
    )

    noisy_img = prep(noisy_img)
    p_psf = prep(gen.single_psf(p_wave, normed=False, noise=True))
    gt_psf = prep(gen.single_psf(y_wave, normed=False, noise=True))
    corrected_psf = prep(gen.single_psf(diff, normed=False, noise=True))

    if flat_path is not None:
        rfilter = f"{str(gt_path.name).replace(gt_postfix, '')}"
        dm_path = Path(str(list(input_path.parent.glob(f"{rfilter}*JSONsettings.json"))[0]))
        dm_wavefront = Path(gt_path.parent/f"{rfilter}_dm_wavefront.svg")

        plot_dm_actuators(
            dm_path=dm_path,
            flat_path=flat_path,
            save_path=dm_wavefront
        )

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
        save_path=Path(f'{prediction_path.parent}/{prediction_path.stem}_{save_postfix}_eval'),
        display=False,
        dxy=gen.x_voxel_size,
        dz=gen.z_voxel_size
    )

    plt.style.use("dark_background")
    vis.diagnostic_assessment(
        psf=noisy_img,
        gt_psf=gt_psf,
        predicted_psf=p_psf,
        corrected_psf=corrected_psf,
        psnr=psnr,
        maxcounts=maxcounts,
        y=y_wave,
        pred=p_wave,
        save_path=Path(f'{prediction_path.parent}/{prediction_path.stem}_{save_postfix}_eval_db'),
        display=False,
        dxy=gen.x_voxel_size,
        dz=gen.z_voxel_size
    )


@profile
def eval_dataset(
    model: Path,
    datadir: Path,
    flat: Any = None,
    postfix: str = 'sample_predictions_zernike_coefficients.csv',
    gt_postfix: str = 'pr_pupil_waves.tif',
    # gt_postfix: str = 'ground_truth_zernike_coefficients.csv',
):
    func = partial(
        eval_mode,
        model_path=model,
        flat_path=flat,
        postfix=postfix,
        gt_postfix=gt_postfix,
    )

    jobs = []
    for file in sorted(datadir.glob('*_lightsheet_ansi_z*.tif')):
        if 'CamB' in str(file) or 'pupil' in str(file):
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

        worker = partial(func, prediction_path=prediction_path, gt_path=gt_path)
        p = mp.Process(target=worker, args=(file,))
        p.start()
        jobs.append(p)
        logger.info(f'-'*50)

        while len(jobs) >= 10:
            for p in jobs:
                if not p.is_alive():
                    jobs.remove(p)
            time.sleep(10)


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
    dm_state = None if (dm_state is None or str(dm_state) == 'None') else dm_state

    data = np.int_(imread(img))
    data_prepped = prep_data_for_PR(data, multiplier=1.1)

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

    params = dict(
        wl=psfgen.lam_detection,
        na=psfgen.na_detection,
        ni=psfgen.refractive_index,
        res=lateral_voxel_size,
        zres=axial_voxel_size,
    )

    pr_result = pr.retrieve_phase(
        data_prepped,
        params,
        max_iters=num_iterations,
        pupil_tol=1e-5,
        mse_tol=0,
        phase_only=True
    )
    pupil = pr_result.phase / (2 * np.pi)  # convert radians to waves
    pupil[pupil != 0.] -= np.mean(pupil[pupil != 0.])
    pr_result.phase = utils.waves2microns(pupil, wavelength=psfgen.lam_detection)  # convert waves to microns before fitting.
    pr_result.fit_to_zernikes(num_modes-1, mapping=osa2degrees)  # zernikes now in um rms
    pr_result.phase = pupil  # phase is now again in waves

    pupil[pupil == 0.] = np.nan
    pupil_path = Path(f"{img.with_suffix('')}_phase_retrieval_wavefront.tif")
    imsave(pupil_path, pupil)

    threshold = utils.waves2microns(prediction_threshold, wavelength=psfgen.lam_detection)
    ignore_modes = list(map(int, ignore_modes))

    if use_pyotf_zernikes:
        pred = np.zeros(num_modes)
        pred[1:] = pr_result.zd_result.pcoefs
        pred[ignore_modes] = 0.
        pred[np.abs(pred) <= threshold] = 0.
        pred = Wavefront(pred, modes=num_modes, order='ansi', lam_detection=wavelength)
    else:
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
        dm_state = load_dm(dm_state)
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
        plt.savefig(Path(f"{img.with_suffix('')}_phase_retrieval_convergence.svg"), bbox_inches='tight', pad_inches=.25)
