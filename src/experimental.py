import matplotlib
from tqdm import tqdm

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

import utils
import vis
import backend
import preprocessing
from synthetic import SyntheticPSF
from wavefront import Wavefront
from data_utils import get_image, load_dataset

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


def pct_outliers(data: np.ndarray, threshold: int = 95) -> np.ndarray:
    diff = (100 - threshold) / 2.0
    minval, maxval = np.percentile(data, [diff, 100 - diff])
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


def points_detection(
    img: Path,
    axial_voxel_size: float,
    lateral_voxel_size: float,
    psf: Any = None,
    skew_angle: float = 32.45,
    sigma_xy: float = 1.1,
    sigma_z: float = 1.1,
):
    matlab = 'matlab '
    matlab += f' -nodisplay'
    matlab += f' -nosplash'
    matlab += f' -nodesktop'
    matlab += f' -nojvm -r '

    if psf is not None:
        det = f"TA_PointDetection('{img}','{psf}',{lateral_voxel_size},{axial_voxel_size},{skew_angle},{sigma_xy},{sigma_z})"
    else:
        det = f"TA_PointDetection('{img}','none',{lateral_voxel_size},{axial_voxel_size},{skew_angle},{sigma_xy},{sigma_z})"

    repo = Path(__file__).parent.parent.absolute()
    llsm = f"addpath(genpath('{repo}/LLSM3DTools/'))"
    job = f"{matlab} \"{llsm}; {det}; exit;\""

    print(job)
    call([job], shell=True)


def get_roi(path, crop_shape, model_voxel_size, sample_voxel_size):
    try:
        if isinstance(path, tf.Tensor):
            path = Path(str(path.numpy(), "utf-8"))
        else:
            path = Path(str(path))

        img = get_image(path)

        img = preprocessing.prep_sample(
            np.squeeze(img),
            crop_shape=crop_shape,
            model_voxel_size=model_voxel_size,
            sample_voxel_size=sample_voxel_size,
            debug=f"{path.parent}/{path.stem}_preprocessing"
        )

        return img

    except Exception as e:
        logger.warning(e)


def predict(
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
        threshold: float = 0.0,
        verbose: bool = False,
        plot: bool = False,
        psf_type: str = 'widefield',
        n_modes: int = 60,
        mosaic: bool = True,
        prev: Any = None
):
    physical_devices = tfc.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tfc.experimental.set_memory_growth(gpu_instance, True)

    inputs = imread(img).astype(int)
    esnr = np.sqrt(inputs.max()).round(0).astype(int)

    psfgen = SyntheticPSF(
        dtype=psf_type,
        order='ansi',
        snr=esnr,
        n_modes=n_modes,
        gamma=.75,
        bimodal=True,
        lam_detection=wavelength,
        psf_shape=(64, 64, 64),
        x_voxel_size=model_lateral_voxel_size,
        y_voxel_size=model_lateral_voxel_size,
        z_voxel_size=model_axial_voxel_size,
        batch_size=1,
        max_jitter=0,
        cpu_workers=-1,
    )

    inputs = preprocessing.prep_sample(
        inputs,
        crop_shape=(64, 64, 64),
        model_voxel_size=(model_axial_voxel_size, model_lateral_voxel_size, model_lateral_voxel_size),
        sample_voxel_size=(axial_voxel_size, lateral_voxel_size, lateral_voxel_size),
        debug=Path(f'{img.parent / img.stem}_preprocessing')
    )

    inputs = np.expand_dims(inputs, axis=0)

    model = backend.load(model, mosaic=mosaic)

    p, std = backend.booststrap_predict_sign(
        model,
        inputs=inputs,
        threshold=threshold,
        verbose=verbose,
        gen=psfgen,
        prev_pred=prev,
        plot=Path(f'{img.parent / img.stem}') if plot else None,
    )

    dm_state = np.zeros(69) if dm_state is None else pd.read_csv(dm_state, header=None).values[:, 0]
    dm = zernikies_to_actuators(p, dm_pattern=dm_pattern, dm_state=dm_state, scalar=scalar)
    dm = pd.DataFrame(dm)
    dm.to_csv(f"{img.parent / img.stem}_corrected_actuators.csv")

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
    coffs.to_csv(f"{img.parent / img.stem}_zernike_coffs.csv")

    pupil_displacement = np.array(p.wave(size=100), dtype='float32')
    imsave(f"{img.parent / img.stem}_pred_pupil_displacement.tif", pupil_displacement)

    if plot:
        vis.prediction(
            psf=np.squeeze(inputs),
            pred=p,
            pred_std=std,
            dm_before=dm_state,
            dm_after=dm.values[:, 0],
            wavelength=wavelength,
            save_path=Path(f'{img.parent / img.stem}_pred'),
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
        threshold: float = 0.0,
        verbose: bool = False,
        plot: bool = False,
        psf_type: str = 'widefield',
        n_modes: int = 60,
        mosaic: bool = False,
        prev: Any = None
):
    func = partial(
        predict,
        model=model,
        dm_pattern=dm_pattern,
        axial_voxel_size=axial_voxel_size,
        model_axial_voxel_size=model_axial_voxel_size,
        lateral_voxel_size=lateral_voxel_size,
        model_lateral_voxel_size=model_lateral_voxel_size,
        wavelength=wavelength,
        scalar=scalar,
        threshold=threshold,
        verbose=verbose,
        plot=plot,
        psf_type=psf_type,
        n_modes=n_modes,
        mosaic=mosaic,
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
    dm_pattern: Path,
    dm_state: Any,
    peaks: Any,
    axial_voxel_size: float,
    model_axial_voxel_size: float,
    lateral_voxel_size: float,
    model_lateral_voxel_size: float,
    wavelength: float = .605,
    plot: bool = False,
    psf_type: str = 'widefield',
    mosaic: bool = True,
    prev: Any = None,
    window_size: int = 64,
    num_rois: int = 10,
    min_intensity: int = 200,
    threshold: float = 1e-2,
    sign_threshold: float = .4,
    outliers_threshold: int = 75,
    scalar: float = 1,
):
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    physical_devices = tfc.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tfc.experimental.set_memory_growth(gpu_instance, True)

    sample = imread(img).astype(int)
    esnr = np.sqrt(sample.max()).round(0).astype(int)
    model_voxel_size = (model_axial_voxel_size, model_lateral_voxel_size, model_lateral_voxel_size)
    sample_voxel_size = (axial_voxel_size, lateral_voxel_size, lateral_voxel_size)

    mode = int(st.mode(sample[sample < np.quantile(sample, .99)], axis=None).mode[0])
    sample -= mode
    sample[sample < 0] = 0
    sample = sample / np.nanmax(sample)

    model = backend.load(model, mosaic=mosaic)

    psfgen = SyntheticPSF(
        dtype=psf_type,
        order='ansi',
        snr=esnr,
        n_modes=model.output_shape[1],
        gamma=.75,
        bimodal=True,
        lam_detection=wavelength,
        psf_shape=(64, 64, 64),
        x_voxel_size=model_lateral_voxel_size,
        y_voxel_size=model_lateral_voxel_size,
        z_voxel_size=model_axial_voxel_size,
        batch_size=1,
        max_jitter=0,
        cpu_workers=-1,
    )

    outdir = Path(f'{img.parent / img.stem}_rois')
    dm_state = np.zeros(69) if dm_state is None else pd.read_csv(dm_state, header=None).values[:, 0]
    logger.info(f"Sample: {sample.shape}")

    preprocessing.find_roi(
        sample,
        savepath=outdir,
        peaks=peaks,
        window_size=tuple(3*[window_size]),
        plot=Path(f'{img.parent/img.stem}'),
        num_peaks=num_rois,
        min_dist=1,
        max_dist=None,
        max_neighbor=10,
        min_intensity=min_intensity,
        voxel_size=sample_voxel_size,
    )

    load_sample = partial(
        get_roi,
        crop_shape=(64, 64, 64),
        model_voxel_size=model_voxel_size,
        sample_voxel_size=sample_voxel_size,
    )
    rois = np.array(utils.multiprocess(load_sample, list(outdir.glob(r'roi_[0-9][0-9].tif')), desc='Loading ROIs'))
    logger.info(rois.shape)

    predictions, stds = backend.booststrap_predict_sign(
        model,
        inputs=rois[..., np.newaxis],
        threshold=threshold,
        sign_threshold=sign_threshold,
        n_samples=5,
        verbose=True,
        gen=psfgen,
        prev_pred=prev,
        plot=None,
        desc='Predicting'
    )

    # drop null predictions
    predictions = predictions[~np.all(predictions == 0, axis=1)]

    p_modes = predictions.copy()
    p_modes[p_modes > 0] = 1
    p_modes = np.sum(p_modes, axis=0)

    predictions = pd.DataFrame(predictions.T, columns=[f"p{k}" for k in range(predictions.shape[0])])
    predictions['votes'] = p_modes
    predictions['mean'] = np.zeros_like(p_modes)
    predictions['std'] = np.zeros_like(p_modes)

    pcols = predictions.columns[pd.Series(predictions.columns).str.startswith('p')]
    det_modes = predictions[predictions['votes'] >= p_modes.max()//2][pcols]
    min_amp = det_modes.values.flatten().min()
    max_amp = det_modes.values.flatten().max()

    if plot:
        fig, axes = plt.subplots(nrows=det_modes.shape[0], figsize=(8, 11))

    for i in tqdm(range(det_modes.shape[0]), desc='Detecting outliers', total=det_modes.shape[0]):
        preds = det_modes.iloc[i]
        outliers = preds[pct_outliers(preds.values, threshold=outliers_threshold)]

        if not outliers.empty and preds.shape[0] > 2:
            preds.drop(outliers.index.values, inplace=True)

        p = np.mean(preds)
        std = np.std(preds)

        predictions.loc[det_modes.index[i], 'mean'] = p
        predictions.loc[det_modes.index[i], 'std'] = std

        if plot:
            ax = axes[i] if det_modes.shape[0] > 1 else axes
            ax = sns.boxplot(
                preds,
                orient='h',
                ax=ax,
                color="lightgrey",
                notch=False,
                showcaps=False,
                flierprops={"marker": "."},
                boxprops={"facecolor": "lightgrey"},
            )
            ax.plot(p, np.zeros_like(1), 'o', clip_on=False, color='C0', label='Prediction', zorder=3)
            ax.plot(outliers, np.zeros_like(outliers), 'x', clip_on=False, color='C3', label='Outliers', zorder=3)

            ax.spines.right.set_visible(False)
            ax.spines.left.set_visible(False)
            ax.spines.top.set_visible(False)
            ax.set_yticks([])
            ax.set_xlim(min_amp-.05, max_amp+.05)
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
        plt.savefig(f"{outdir}_predictions.png", bbox_inches='tight', dpi=300, pad_inches=.25)

    predictions.fillna(0, inplace=True)
    predictions.index.name = 'ansi'
    predictions.to_csv(f"{outdir}_predictions.csv")
    print(predictions)

    dm = pd.DataFrame(zernikies_to_actuators(
        predictions['mean'].values, dm_pattern=dm_pattern, dm_state=dm_state, scalar=scalar
    ))
    dm.to_csv(f"{outdir}_corrected_actuators.csv", index=False, header=False)

    p = Wavefront(predictions['mean'].values, order='ansi', lam_detection=wavelength)
    pred_std = Wavefront(predictions['std'].values, order='ansi', lam_detection=wavelength)

    coffs = [
        {'n': z.n, 'm': z.m, 'amplitude': a}
        for z, a in p.zernikes.items()
    ]
    coffs = pd.DataFrame(coffs, columns=['n', 'm', 'amplitude'])
    coffs.index.name = 'ansi'
    coffs.to_csv(f"{outdir}_zernike_coffs.csv")

    pupil_displacement = np.array(p.wave(size=100), dtype='float32')
    imsave(f"{outdir}_pred_pupil_displacement.tif", pupil_displacement)

    if plot:
        vis.prediction(
            psf=np.squeeze(sample),
            pred=p,
            pred_std=pred_std,
            dm_before=dm_state,
            dm_after=dm.values[:, 0],
            wavelength=wavelength,
            save_path=Path(f'{outdir}_pred'),
        )


def predict_tiles(
    img: Path,
    model: Path,
    dm_pattern: Path,
    dm_state: Any,
    axial_voxel_size: float,
    model_axial_voxel_size: float,
    lateral_voxel_size: float,
    model_lateral_voxel_size: float,
    wavelength: float = .605,
    plot: bool = False,
    psf_type: str = 'widefield',
    prev: Any = None,
    window_size: int = 64,
    threshold: float = 1e-2,
    sign_threshold: float = .4,
    outliers_threshold: int = 75,
    scalar: float = 1,
):
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    physical_devices = tfc.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tfc.experimental.set_memory_growth(gpu_instance, True)

    sample = imread(img).astype(int)
    esnr = np.sqrt(sample.max()).round(0).astype(int)
    model_voxel_size = (model_axial_voxel_size, model_lateral_voxel_size, model_lateral_voxel_size)
    sample_voxel_size = (axial_voxel_size, lateral_voxel_size, lateral_voxel_size)

    mode = int(st.mode(sample[sample < np.quantile(sample, .99)], axis=None).mode[0])
    sample -= mode
    sample[sample < 0] = 0
    sample = sample / np.nanmax(sample)

    model = backend.load(model, mosaic=True)

    psfgen = SyntheticPSF(
        dtype=psf_type,
        order='ansi',
        snr=esnr,
        n_modes=model.output_shape[1],
        gamma=.75,
        bimodal=True,
        lam_detection=wavelength,
        psf_shape=(64, 64, 64),
        x_voxel_size=model_lateral_voxel_size,
        y_voxel_size=model_lateral_voxel_size,
        z_voxel_size=model_axial_voxel_size,
        batch_size=1,
        max_jitter=0,
        cpu_workers=-1,
    )

    outdir = Path(f'{img.parent / img.stem}_tiles')
    dm_state = np.zeros(69) if dm_state is None else pd.read_csv(dm_state, header=None).values[:, 0]
    logger.info(f"Sample: {sample.shape}")

    preprocessing.get_tiles(
        sample,
        savepath=outdir,
        strides=window_size,
        window_size=tuple(3*[window_size]),
    )

    load_sample = partial(
        get_roi,
        crop_shape=(64, 64, 64),
        model_voxel_size=model_voxel_size,
        sample_voxel_size=sample_voxel_size,
    )
    rois = np.array(utils.multiprocess(load_sample, list(outdir.glob(r'roi_[0-9][0-9].tif')), desc='Loading ROIs'))
    logger.info(rois.shape)

    predictions, stds = backend.booststrap_predict_sign(
        model,
        inputs=rois[..., np.newaxis],
        threshold=threshold,
        sign_threshold=sign_threshold,
        n_samples=5,
        verbose=True,
        gen=psfgen,
        prev_pred=prev,
        plot=None,
        desc='Predicting'
    )

    # drop null predictions
    predictions = predictions[~np.all(predictions == 0, axis=1)]

    p_modes = predictions.copy()
    p_modes[p_modes > 0] = 1
    p_modes = np.sum(p_modes, axis=0)

    predictions = pd.DataFrame(predictions.T, columns=[f"p{k}" for k in range(predictions.shape[0])])
    predictions['votes'] = p_modes
    predictions['mean'] = np.zeros_like(p_modes)
    predictions['std'] = np.zeros_like(p_modes)

    pcols = predictions.columns[pd.Series(predictions.columns).str.startswith('p')]
    det_modes = predictions[predictions['votes'] >= p_modes.max()//2][pcols]
    min_amp = det_modes.values.flatten().min()
    max_amp = det_modes.values.flatten().max()

    if plot:
        fig, axes = plt.subplots(nrows=det_modes.shape[0], figsize=(8, 11))

    for i in tqdm(range(det_modes.shape[0]), desc='Detecting outliers', total=det_modes.shape[0]):
        preds = det_modes.iloc[i]
        outliers = preds[pct_outliers(preds.values, threshold=outliers_threshold)]

        if not outliers.empty and preds.shape[0] > 2:
            preds.drop(outliers.index.values, inplace=True)

        p = np.mean(preds)
        std = np.std(preds)

        predictions.loc[det_modes.index[i], 'mean'] = p
        predictions.loc[det_modes.index[i], 'std'] = std

        if plot:
            ax = axes[i] if det_modes.shape[0] > 1 else axes
            ax = sns.boxplot(
                preds,
                orient='h',
                ax=ax,
                color="lightgrey",
                notch=False,
                showcaps=False,
                flierprops={"marker": "."},
                boxprops={"facecolor": "lightgrey"},
            )
            ax.plot(p, np.zeros_like(1), 'o', clip_on=False, color='C0', label='Prediction', zorder=3)
            ax.plot(outliers, np.zeros_like(outliers), 'x', clip_on=False, color='C3', label='Outliers', zorder=3)

            ax.spines.right.set_visible(False)
            ax.spines.left.set_visible(False)
            ax.spines.top.set_visible(False)
            ax.set_yticks([])
            ax.set_xlim(min_amp-.05, max_amp+.05)
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
        plt.savefig(f"{outdir}_predictions.png", bbox_inches='tight', dpi=300, pad_inches=.25)

    predictions.fillna(0, inplace=True)
    predictions.index.name = 'ansi'
    predictions.to_csv(f"{outdir}_predictions.csv")
    print(predictions)

    dm = pd.DataFrame(zernikies_to_actuators(
        predictions['mean'].values, dm_pattern=dm_pattern, dm_state=dm_state, scalar=scalar
    ))
    dm.to_csv(f"{outdir}_corrected_actuators.csv", index=False, header=False)

    p = Wavefront(predictions['mean'].values, order='ansi', lam_detection=wavelength)
    pred_std = Wavefront(predictions['std'].values, order='ansi', lam_detection=wavelength)

    coffs = [
        {'n': z.n, 'm': z.m, 'amplitude': a}
        for z, a in p.zernikes.items()
    ]
    coffs = pd.DataFrame(coffs, columns=['n', 'm', 'amplitude'])
    coffs.index.name = 'ansi'
    coffs.to_csv(f"{outdir}_zernike_coffs.csv")

    pupil_displacement = np.array(p.wave(size=100), dtype='float32')
    imsave(f"{outdir}_pred_pupil_displacement.tif", pupil_displacement)

    if plot:
        vis.prediction(
            psf=np.squeeze(sample),
            pred=p,
            pred_std=pred_std,
            dm_before=dm_state,
            dm_after=dm.values[:, 0],
            wavelength=wavelength,
            save_path=Path(f'{outdir}_pred'),
        )
