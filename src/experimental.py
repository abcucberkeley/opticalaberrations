import matplotlib
matplotlib.use('Agg')

import time
from functools import partial
from pathlib import Path
from subprocess import call
import multiprocessing as mp
from tensorflow import config as tfc
from typing import Any, Sequence, Union
import numpy as np
from scipy import stats as st
import pandas as pd
import zarr
import h5py
from tifffile import imread, imsave
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import matplotlib.patches as patches


import utils
import vis
import backend
import preprocessing
from synthetic import SyntheticPSF
from wavefront import Wavefront

import logging
logger = logging.getLogger('')


def zernikies_to_actuators(coefficients: np.array, dm_pattern: Path, dm_state: np.array, scalar: float = 1):
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
    repo = "addpath(genpath('~/fiona/ABCcode/XR_Repository'))"
    job = f"{matlab} \"{repo}; {deskew}; exit;\""

    print(job)
    call([job], shell=True)


def points_detection(
    img: Path,
    psf: Path,
    axial_voxel_size: float,
    lateral_voxel_size: float,
    skew_angle: float = 32.45,
):
    matlab = 'matlab '
    matlab += f' -nodisplay'
    matlab += f' -nosplash'
    matlab += f' -nodesktop'
    matlab += f' -nojvm -r '

    det = f"TA_PointDetection('{img}','{psf}',{lateral_voxel_size},{axial_voxel_size},{skew_angle})"
    repo = "addpath(genpath('~/fiona/ABCcode/XR_Repository'))"
    job = f"{matlab} \"{repo}; {det}; exit;\""

    print(job)
    call([job], shell=True)


def predict_rois(
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
        input_shape=(64, 64, 64),
        model_voxel_size=(model_axial_voxel_size, model_lateral_voxel_size, model_lateral_voxel_size),
        sample_voxel_size=(axial_voxel_size, lateral_voxel_size, lateral_voxel_size),
        debug=Path(f'{img.parent/img.stem}_preprocessing')
    )

    inputs = np.expand_dims(inputs, axis=0)

    model = backend.load(model, mosaic=mosaic)

    p = backend.booststrap_predict_sign(
        model,
        inputs=inputs,
        batch_size=1,
        threshold=threshold,
        verbose=verbose,
        gen=psfgen,
        prev_pred=prev,
        plot=Path(f'{img.parent/img.stem}') if plot else None,
    )

    dm_state = np.zeros(69) if dm_state is None else pd.read_csv(dm_state, header=None).values[:, 0]
    dm = zernikies_to_actuators(p, dm_pattern=dm_pattern, dm_state=dm_state, scalar=scalar)
    dm = pd.DataFrame(dm)
    dm.to_csv(f"{img.parent/img.stem}_corrected_actuators.csv", index=False, header=False)

    p = Wavefront(p, order='ansi', lam_detection=wavelength)
    if verbose:
        logger.info('Prediction')
        logger.info(p.zernikes)

    coffs = [
        {'n': z.n, 'm': z.m, 'amplitude': a}
        for z, a in p.zernikes.items()
    ]
    coffs = pd.DataFrame(coffs, columns=['n', 'm', 'amplitude'])
    coffs.index.name = 'ansi'
    coffs.to_csv(f"{img.parent/img.stem}_zernike_coffs.csv")

    pupil_displacement = np.array(p.wave(size=100), dtype='float32')
    imsave(f"{img.parent/img.stem}_pred_pupil_displacement.tif", pupil_displacement)

    if plot:
        vis.prediction(
            psf=np.squeeze(inputs),
            pred=p,
            dm_before=dm_state,
            dm_after=dm.values[:, 0],
            wavelength=wavelength,
            save_path=Path(f'{img.parent/img.stem}_pred'),
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
            worker = partial(func, dm_state=dm_state)
            p = mp.Process(target=worker, args=(file,))
            p.start()
            jobs.append(p)
            print(f"Evaluating: {file}")

            while len(jobs) >= 6:
                for p in jobs:
                    if not p.is_alive():
                        jobs.remove(p)
                time.sleep(10)
