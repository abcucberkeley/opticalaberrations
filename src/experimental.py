import matplotlib
matplotlib.use('Agg')

import time
from functools import partial
from pathlib import Path
from subprocess import call
import multiprocessing as mp
import tensorflow as tf
from tensorflow import config as tfc
from typing import Any, Sequence, Union
import numpy as np
from scipy import stats as st
import pandas as pd
from tifffile import imread, imsave
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import utils
import vis
import backend
import preprocessing
from synthetic import SyntheticPSF
from wavefront import Wavefront
from data_utils import get_image

import logging
logger = logging.getLogger('')


def zernikies_to_actuators(
        coefficients: np.array,
        dm_pattern: Path,
        dm_state: np.array,
        scalar: float = 1
) -> np.ndarray:
    dm_pattern = pd.read_csv(dm_pattern, header=None).values

    if dm_pattern.shape[-1] > coefficients.size:
        dm_pattern = dm_pattern[:, :coefficients.size]
    else:
        coefficients = coefficients[:dm_pattern.shape[-1]]

    coefficients = np.expand_dims(coefficients, axis=-1)
    offset = np.dot(dm_pattern, coefficients)[:, 0]
    return dm_state + (offset * scalar)


def matlab_phase_retrieval(psf: Path, dx=.15, dz=.6, wavelength=.605, n_modes=60) -> list:
    try:
        import matlab.engine
        matlab = matlab.engine.start_matlab()
        matlab.addpath(matlab.genpath('phase_retrieval'), nargout=0)

        zcoffs = matlab.PhaseRetrieval(str(psf), dx, dz, wavelength)
        zcoffs = np.array(zcoffs._data).flatten()[:n_modes]
        zcoffs = utils.waves2microns(zcoffs, wavelength=wavelength)
        return list(zcoffs)

    except Exception:
        logger.error(
            'Matlab-python engine is not installed! See'
            'https://www.mathworks.com/help/matlab/matlab_external/call-user-script-and-function-from-python.html'
        )


def percentile_filter(data: np.ndarray, min_pct: int = 5, max_pct: int = 95) -> np.ndarray:
    minval, maxval = np.percentile(data, [min_pct, max_pct])
    return (data < minval) | (data > maxval)


def deskew(
    img: Path,
    axial_voxel_size: float,
    lateral_voxel_size: float,
    flipz: bool = False,
    skew_angle: float = 32.45,
):
    matlab = 'matlab '
    matlab += f' -nodisplay'
    matlab += f' -nosplash'
    matlab += f' -nodesktop'
    matlab += f' -nojvm -r '

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
    call([job], shell=True)


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
    matlab += f' -nodisplay'
    matlab += f' -nosplash'
    matlab += f' -nodesktop'
    matlab += f' -nojvm -r '

    det = f"TA_PointDetection('{img}','{psf}',{lateral_voxel_size},{axial_voxel_size},{skew_angle},{sigma_xy},{sigma_z})"
    repo = Path(__file__).parent.parent.absolute()
    llsm = f"addpath(genpath('{repo}/LLSM3DTools/'))"
    job = f"{matlab} \"{llsm}; {det}; exit;\""

    print(job)
    call([job], shell=True)


def decon(img: Path, psf: Path, iters: int = 10):
    matlab = 'matlab '
    matlab += f' -nodisplay'
    matlab += f' -nosplash'
    matlab += f' -nodesktop'
    matlab += f' -nojvm -r '

    det = f"TA_Decon('{img}','{psf}',{iters}, '{psf.with_suffix('')}_decon.tif')"
    repo = Path(__file__).parent.parent.absolute()
    llsm = f"addpath(genpath('{repo}/LLSM3DTools/'))"
    job = f"{matlab} \"{llsm}; {det}; exit;\""

    print(job)
    call([job], shell=True)

    original_image = imread(img).astype(float)
    original_image /= np.max(original_image)

    corrected_image = imread(f"{psf.with_suffix('')}_decon.tif").astype(float)
    corrected_image /= np.max(corrected_image)

    vis.prediction(
        original_image=original_image,
        corrected_image=corrected_image,
        save_path=f"{psf.with_suffix('')}_decon"
    )


def load_sample(
    path: Path,
    model_voxel_size: tuple,
    sample_voxel_size: tuple,
    remove_background: bool = True,
    normalize: bool = True,
    debug: bool = False
):
    try:
        if isinstance(path, tf.Tensor):
            path = Path(str(path.numpy(), "utf-8"))
        else:
            path = Path(str(path))

        img = get_image(path).astype(float)

        if remove_background:
            mode = st.mode(img[img < np.quantile(img, .99)], axis=None).mode[0]
            img -= mode
            img[img < 0] = 0

        if normalize:
            img /= np.nanmax(img)

        img = preprocessing.prep_sample(
            np.squeeze(img),
            model_voxel_size=model_voxel_size,
            sample_voxel_size=sample_voxel_size,
            debug=Path(f"{path.with_suffix('')}_preprocessing") if debug else None
        )

        return img

    except Exception as e:
        logger.warning(e)


def predict(
    data: Path,
    model: Path,
    axial_voxel_size: float,
    model_axial_voxel_size: float,
    lateral_voxel_size: float,
    model_lateral_voxel_size: float,
    wavelength: float = .605,
    psf_type: str = 'widefield',
    mosaic: bool = True,
    prev: Any = None,
    prediction_threshold: float = 0.,
    sign_threshold: float = .4,
    num_predictions: int = 1,
    batch_size: int = 1,
    plot: bool = True,
    zplanes: int = 1,
    nrows: int = 1,
    ncols: int = 1,
):
    model_voxel_size = (model_axial_voxel_size, model_lateral_voxel_size, model_lateral_voxel_size)
    sample_voxel_size = (axial_voxel_size, lateral_voxel_size, lateral_voxel_size)

    physical_devices = tfc.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tfc.experimental.set_memory_growth(gpu_instance, True)

    model = backend.load(model, mosaic=mosaic)

    psfgen = SyntheticPSF(
        dtype=psf_type,
        psf_shape=(64, 64, 64),
        n_modes=model.output_shape[1],
        lam_detection=wavelength,
        x_voxel_size=model_lateral_voxel_size,
        y_voxel_size=model_lateral_voxel_size,
        z_voxel_size=model_axial_voxel_size,
    )

    load = partial(
        load_sample,
        model_voxel_size=model_voxel_size,
        sample_voxel_size=sample_voxel_size,
        remove_background=False,
        normalize=False
    )
    rois = np.array(utils.multiprocess(load, list(data.glob(r'roi_[0-9][0-9].tif')), desc='Loading ROIs'))
    logger.info(rois.shape)

    preds, stds = backend.booststrap_predict_sign(
        model,
        batch_size=batch_size,
        inputs=rois[..., np.newaxis],
        threshold=prediction_threshold,
        sign_threshold=sign_threshold,
        n_samples=num_predictions,
        verbose=True,
        gen=psfgen,
        prev_pred=prev,
        plot=None,
    )

    predictions = pd.DataFrame(preds.T, columns=[f"p{k}" for k in range(preds.shape[0])])
    pcols = predictions.columns[pd.Series(predictions.columns).str.startswith('p')]

    predictions['mean'] = predictions[pcols].mean(axis=1)
    predictions['median'] = predictions[pcols].median(axis=1)
    predictions['min'] = predictions[pcols].min(axis=1)
    predictions['max'] = predictions[pcols].max(axis=1)
    predictions['std'] = predictions[pcols].std(axis=1)

    predictions.index.name = 'ansi'
    predictions.to_csv(f"{data}_predictions.csv")

    if plot:
        vis.wavefronts(
            scale='max',
            predictions=predictions,
            nrows=nrows,
            ncols=ncols,
            wavelength=wavelength,
            save_path=Path(f"{data.with_suffix('')}_predictions_wavefronts"),
        )


def predict_sample(
    img: Path,
    model: Path,
    dm_pattern: Path,
    dm_state: Any,
    axial_voxel_size: float,
    model_axial_voxel_size: float,
    lateral_voxel_size: float,
    model_lateral_voxel_size: float,
    wavelength: float = .605,
    scalar: float = 1,
    prediction_threshold: float = 0.0,
    sign_threshold: float = .4,
    verbose: bool = False,
    plot: bool = False,
    psf_type: str = 'widefield',
    num_predictions: int = 1,
    batch_size: int = 1,
    mosaic: bool = True,
    prev: Any = None
):
    dm_state = None if eval(str(dm_state)) is None else dm_state

    physical_devices = tfc.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tfc.experimental.set_memory_growth(gpu_instance, True)

    model = backend.load(model, mosaic=mosaic)

    modelpsfgen = SyntheticPSF(
        dtype=psf_type,
        psf_shape=(64, 64, 64),
        n_modes=model.output_shape[1],
        lam_detection=wavelength,
        x_voxel_size=model_lateral_voxel_size,
        y_voxel_size=model_lateral_voxel_size,
        z_voxel_size=model_axial_voxel_size,
    )

    psfgen = SyntheticPSF(
        dtype=psf_type,
        snr=1000,
        psf_shape=(64, 64, 64),
        n_modes=model.output_shape[1],
        lam_detection=wavelength,
        x_voxel_size=lateral_voxel_size,
        y_voxel_size=lateral_voxel_size,
        z_voxel_size=axial_voxel_size,
    )

    inputs = load_sample(
        img,
        model_voxel_size=(model_axial_voxel_size, model_lateral_voxel_size, model_lateral_voxel_size),
        sample_voxel_size=(axial_voxel_size, lateral_voxel_size, lateral_voxel_size),
        remove_background=True,
        normalize=True,
    )

    inputs = np.expand_dims(inputs, axis=0)

    p, std = backend.booststrap_predict_sign(
        model,
        inputs=inputs,
        threshold=prediction_threshold,
        sign_threshold=sign_threshold,
        n_samples=num_predictions,
        verbose=verbose,
        gen=modelpsfgen,
        prev_pred=prev,
        batch_size=batch_size,
        plot=Path(f"{img.with_suffix('')}_predictions_embeddings") if plot else None,
    )

    dm_state = np.zeros(69) if dm_state is None else pd.read_csv(dm_state, header=None).values[:, 0]
    dm = pd.DataFrame(zernikies_to_actuators(p, dm_pattern=dm_pattern, dm_state=dm_state, scalar=scalar))
    dm.to_csv(f"{img.with_suffix('')}_predictions_corrected_actuators.csv", index=False, header=False)

    p = Wavefront(p, order='ansi', lam_detection=wavelength)
    std = Wavefront(std, order='ansi', lam_detection=wavelength)

    if verbose:
        logger.info('Prediction')
        logger.info(p.zernikes)

    coffs = [
        {'n': z.n, 'm': z.m, 'amplitude': a}
        for z, a in p.zernikes.items()
    ]
    coffs = pd.DataFrame(coffs, columns=['n', 'm', 'amplitude'])
    coffs.index.name = 'ansi'
    coffs.to_csv(f"{img.with_suffix('')}_predictions_zernike_coffs.csv")

    pupil_displacement = np.array(p.wave(size=100), dtype='float32')
    imsave(f"{img.with_suffix('')}_predictions_pupil_displacement.tif", pupil_displacement)

    psf = psfgen.single_psf(phi=p, normed=True, noise=False)
    imsave(f"{img.with_suffix('')}_predictions_psf.tif", psf)

    if plot:
        vis.diagnosis(
            pred=p,
            pred_std=std,
            dm_before=dm_state,
            dm_after=dm.values[:, 0],
            wavelength=wavelength,
            save_path=Path(f"{img.with_suffix('')}_predictions_diagnosis"),
        )


def predict_dataset(
        dataset: Path,
        model: Path,
        dm_pattern: Path,
        dm_state: Any,
        axial_voxel_size: float,
        model_axial_voxel_size: float,
        lateral_voxel_size: float,
        model_lateral_voxel_size: float,
        wavelength: float = .605,
        scalar: float = 1,
        prediction_threshold: float = 0.0,
        verbose: bool = False,
        plot: bool = False,
        psf_type: str = 'widefield',
        n_modes: int = 60,
        num_predictions: int = 1,
        batch_size: int = 1,
        mosaic: bool = False,
        prev: Any = None
):
    func = partial(
        predict_sample,
        model=model,
        dm_pattern=dm_pattern,
        axial_voxel_size=axial_voxel_size,
        model_axial_voxel_size=model_axial_voxel_size,
        lateral_voxel_size=lateral_voxel_size,
        model_lateral_voxel_size=model_lateral_voxel_size,
        wavelength=wavelength,
        scalar=scalar,
        threshold=prediction_threshold,
        verbose=verbose,
        plot=plot,
        psf_type=psf_type,
        n_modes=n_modes,
        mosaic=mosaic,
        num_predictions=num_predictions,
        batch_size=batch_size,
        prev=prev,
    )

    jobs = []
    for file in dataset.rglob('*.tif'):
        if '_' not in file.stem:
            worker = partial(func, dm_state=f"{file.parent}/DM{file.stem}.csv")
            p = mp.Process(target=worker, args=(file,))
            p.start()
            jobs.append(p)
            print(f"Evaluating: {file}")

            while len(jobs) >= 6:
                for p in jobs:
                    if not p.is_alive():
                        jobs.remove(p)
                time.sleep(10)


def predict_rois(
    img: Path,
    model: Path,
    peaks: Any,
    axial_voxel_size: float,
    model_axial_voxel_size: float,
    lateral_voxel_size: float,
    model_lateral_voxel_size: float,
    wavelength: float = .605,
    psf_type: str = 'widefield',
    num_predictions: int = 1,
    batch_size: int = 1,
    prev: Any = None,
    window_size: int = 64,
    num_rois: int = 10,
    min_intensity: int = 200,
    prediction_threshold: float = 0.,
    sign_threshold: float = .4,
    minimum_distance: float = 1.,
    plot: bool = False,
):
    sample = imread(img).astype(float)
    esnr = np.sqrt(sample.max()).astype(int)

    mode = st.mode(sample[sample < np.quantile(sample, .99)], axis=None).mode[0]
    sample -= mode
    sample[sample < 0] = 0
    sample = sample / np.nanmax(sample)

    outdir = Path(f"{img.with_suffix('')}_rois")
    logger.info(f"Sample: {sample.shape}")

    preprocessing.find_roi(
        sample,
        savepath=outdir,
        peaks=peaks,
        window_size=tuple(3*[window_size]),
        plot=f"{outdir}_predictions" if plot else None,
        num_peaks=num_rois,
        min_dist=minimum_distance,
        max_dist=None,
        max_neighbor=10,
        min_intensity=min_intensity,
        voxel_size=(axial_voxel_size, lateral_voxel_size, lateral_voxel_size),
    )

    ncols = num_rois // 4
    nrows = num_rois // ncols

    predict(
        data=outdir,
        model=model,
        axial_voxel_size=axial_voxel_size,
        model_axial_voxel_size=model_axial_voxel_size,
        lateral_voxel_size=lateral_voxel_size,
        model_lateral_voxel_size=model_lateral_voxel_size,
        prediction_threshold=prediction_threshold,
        sign_threshold=sign_threshold,
        num_predictions=num_predictions,
        batch_size=batch_size,
        wavelength=wavelength,
        psf_type=psf_type,
        mosaic=True,
        prev=prev,
        zplanes=1,
        nrows=ncols,
        ncols=nrows
    )


def predict_tiles(
    img: Path,
    model: Path,
    axial_voxel_size: float,
    model_axial_voxel_size: float,
    lateral_voxel_size: float,
    model_lateral_voxel_size: float,
    wavelength: float = .605,
    psf_type: str = 'widefield',
    num_predictions: int = 1,
    batch_size: int = 1,
    prev: Any = None,
    window_size: int = 64,
    prediction_threshold: float = 0.,
    sign_threshold: float = .4,
    plot: bool = True
):

    sample = load_sample(
        img,
        model_voxel_size=(model_axial_voxel_size, model_lateral_voxel_size, model_lateral_voxel_size),
        sample_voxel_size=(axial_voxel_size, lateral_voxel_size, lateral_voxel_size),
        remove_background=True,
        normalize=True,
    )
    outdir = Path(f"{img.with_suffix('')}_tiles")
    logger.info(f"Sample: {sample.shape}")

    zplanes, nrows, ncols = preprocessing.get_tiles(
        sample,
        savepath=outdir,
        strides=window_size,
        window_size=tuple(3*[window_size]),
    )

    predict(
        data=outdir,
        model=model,
        axial_voxel_size=model_axial_voxel_size,
        model_axial_voxel_size=model_axial_voxel_size,
        lateral_voxel_size=model_lateral_voxel_size,
        model_lateral_voxel_size=model_lateral_voxel_size,
        prediction_threshold=prediction_threshold,
        sign_threshold=sign_threshold,
        num_predictions=num_predictions,
        batch_size=batch_size,
        wavelength=wavelength,
        psf_type=psf_type,
        mosaic=True,
        prev=prev,
        plot=plot,
        zplanes=zplanes,
        nrows=nrows,
        ncols=ncols
    )

    if plot:
        vis.tiles(
            data=np.max(sample, axis=0),
            strides=64,
            window_size=(64, 64),
            save_path=Path(f"{outdir.with_suffix('')}_predictions_mips"),
        )


def aggregate_predictions(
    data: Path,
    model_pred: Path,
    dm_pattern: Path,
    dm_state: Any,
    wavelength: float = .605,
    axial_voxel_size: float = .1,
    lateral_voxel_size: float = .108,
    psf_type: str = 'widefield',
    majority_threshold: float = .5,
    min_percentile: int = 10,
    max_percentile: int = 90,
    prediction_threshold: float = 0.,
    final_prediction: str = 'mean',
    scalar: float = 1,
    plot: bool = False,
    window_size: int = 64,
):
    data = imread(data).astype(float)
    ncols = data.shape[-1] // window_size
    nrows = data.shape[-2] // window_size

    predictions = pd.read_csv(
        model_pred,
        index_col=0,
        header=0,
        usecols=lambda col: col == 'ansi' or col.startswith('p')
    )
    original_pcols = predictions.columns

    # filter out diffraction small predictions
    prediction_threshold = utils.waves2microns(prediction_threshold, wavelength=wavelength)
    predictions[predictions < prediction_threshold] = 0.

    # drop null predictions
    predictions = predictions.loc[:, (predictions != 0).any(axis=0)]
    pcols = predictions.columns[pd.Series(predictions.columns).str.startswith('p')]

    p_modes = predictions[pcols].values
    p_modes[p_modes > 0] = 1
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

            predictions.loc[det_modes.index[i], 'mean'] = np.nanmean(preds)
            predictions.loc[det_modes.index[i], 'median'] = np.nanmedian(preds)
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
                axes[-1].set_xlabel(f'Amplitudes ($\mu m$)')
            else:
                axes.legend(ncol=2, frameon=False)
                axes.set_xlabel(f'Amplitudes ($\mu m$)')

            plt.tight_layout()
            plt.savefig(f"{model_pred.with_suffix('')}_aggregated.png", bbox_inches='tight', dpi=300,
                        pad_inches=.25)
    else:
        logger.warning(f"No modes detected with the current configs")

        for c in original_pcols:
            predictions[c] = np.zeros_like(predictions.index)

        predictions['mean'] = predictions[pcols].mean(axis=1)
        predictions['median'] = predictions[pcols].median(axis=1)
        predictions['min'] = predictions[pcols].min(axis=1)
        predictions['max'] = predictions[pcols].max(axis=1)
        predictions['std'] = predictions[pcols].std(axis=1)

    predictions.fillna(0, inplace=True)
    predictions.index.name = 'ansi'
    predictions.to_csv(f"{model_pred.with_suffix('')}_aggregated.csv")

    dm_state = np.zeros(69) if eval(str(dm_state)) is None else pd.read_csv(dm_state, header=None).values[:, 0]
    dm = pd.DataFrame(zernikies_to_actuators(
        predictions[final_prediction].values, dm_pattern=dm_pattern, dm_state=dm_state, scalar=scalar
    ))
    dm.to_csv(f"{model_pred.with_suffix('')}_aggregated_corrected_actuators.csv", index=False, header=False)

    p = Wavefront(predictions[final_prediction].values, order='ansi', lam_detection=wavelength)
    pred_std = Wavefront(predictions['std'].values, order='ansi', lam_detection=wavelength)

    coffs = [
        {'n': z.n, 'm': z.m, 'amplitude': a}
        for z, a in p.zernikes.items()
    ]
    coffs = pd.DataFrame(coffs, columns=['n', 'm', 'amplitude'])
    coffs.index.name = 'ansi'
    coffs.to_csv(f"{model_pred.with_suffix('')}_aggregated_zernike_coffs.csv")

    pupil_displacement = np.array(p.wave(size=100), dtype='float32')
    imsave(f"{model_pred.with_suffix('')}_aggregated_pupil_displacement.tif", pupil_displacement)

    psfgen = SyntheticPSF(
        dtype=psf_type,
        snr=1000,
        psf_shape=(64, 64, 64),
        n_modes=predictions[final_prediction].shape[0],
        lam_detection=wavelength,
        x_voxel_size=lateral_voxel_size,
        y_voxel_size=lateral_voxel_size,
        z_voxel_size=axial_voxel_size,
    )

    psf = psfgen.single_psf(phi=p, normed=True, noise=False)
    imsave(f"{model_pred.with_suffix('')}_aggregated_psf.tif", psf)

    if plot:
        vis.wavefronts(
            scale='max',
            predictions=predictions,
            ncols=ncols,
            nrows=nrows,
            wavelength=wavelength,
            save_path=Path(f"{model_pred.with_suffix('')}_aggregated_wavefronts"),
        )

        vis.diagnosis(
            pred=p,
            pred_std=pred_std,
            dm_before=dm_state,
            dm_after=dm.values[:, 0],
            wavelength=wavelength,
            save_path=Path(f"{model_pred.with_suffix('')}_aggregated_diagnosis"),
        )
