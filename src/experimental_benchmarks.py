
import matplotlib
matplotlib.use('Agg')

import logging
import sys
import subprocess
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
plt.set_loglevel('error')

import time
import argparse
from tqdm import tqdm
from tifffile import imwrite
import numpy as np
import pandas as pd
import tensorflow as tf
from functools import partial
from line_profiler_pycharm import profile

from csbdeep.utils.tf import limit_gpu_memory

limit_gpu_memory(allow_growth=True, fraction=None, total_memory=None)
from csbdeep.models import CARE

import utils
import preprocessing
import backend
import eval
import vis

from wavefront import Wavefront
from synthetic import SyntheticPSF, PsfGenerator3D
from scipy import stats as st

try:
    import cupy as cp
except ImportError as e:
    logging.warning(f"Cupy not supported on your system: {e}")

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
tf.get_logger().setLevel(logging.ERROR)


@profile
def download_phasenet(phasenet_path: Path = Path('phasenet_repo')):
    if not phasenet_path.exists():
        subprocess.run(f"git clone https://github.com/mpicbg-csbd/phasenet.git phasenet_repo", shell=True)

    from csbdeep.utils import download_and_extract_zip_file

    download_and_extract_zip_file(
        url='https://github.com/mpicbg-csbd/phasenet/releases/download/0.1.0/model.zip',
        targetdir=f'{phasenet_path}/models/',
        verbose=1,
    )

    try:
        from phasenet_repo.phasenet.model import PhaseNet
    except ImportError as e:
        raise e


@profile
def download_cocoa(cocoa_path: Path = Path('cocoa_repo')):

    if not cocoa_path.exists():
        subprocess.run(f"git clone https://github.com/iksungk/CoCoA.git cocoa_repo", shell=True)

    try:
        from cocoa_repo.misc.models import LinearNet
    except ImportError as e:
        raise e


@profile
def predict_phasenet(
    inputs: Path,
    plot: bool = False,
    phasenet: Any = None,
    phasenetgen: Optional[SyntheticPSF] = None,
    phasenet_path: Path = Path('phasenet_repo')
):

    download_phasenet(phasenet_path)
    from csbdeep.utils import normalize

    if phasenet is None:
        from phasenet_repo.phasenet.model import PhaseNet

        phasenet = PhaseNet(
            config=None,
            name='16_05_2020_11_48_14_berkeley_50planes',
            basedir=f'{phasenet_path}/models/'
        )

    if phasenetgen is None:
        phasenetgen = SyntheticPSF(
            psf_type='widefield',
            lls_excitation_profile=None,
            psf_shape=(64, 64, 64),
            n_modes=15,
            lam_detection=.510,
            x_voxel_size=.086,
            y_voxel_size=.086,
            z_voxel_size=.1,
            na_detection=1.1,
            refractive_index=1.33,
        )

    psf = backend.load_sample(inputs)
    psf = preprocessing.resize_image(psf, crop_shape=(50, 50, 50))
    
    mode = st.mode(psf, axis=None).mode
    mode = int(mode[0]) if isinstance(mode, (list, tuple, np.ndarray)) else int(mode)
    
    psf -= mode
    psf = np.expand_dims(normalize(psf), axis=-1)
    
    p = list(phasenet.predict(psf))
    
    wavefront = Wavefront(
        amplitudes=[0, 0, 0, 0] + p,
        lam_detection=phasenetgen.lam_detection,
        modes=phasenetgen.n_modes,
        order='ansi',
        rotate=False,
    )

    coefficients = [
        {'n': z.n, 'm': z.m, 'amplitude': a}
        for z, a in wavefront.zernikes.items()
    ]
    df = pd.DataFrame(coefficients, columns=['n', 'm', 'amplitude'])
    df.index.name = 'ansi'
    df.to_csv(f"{inputs.with_suffix('')}_phasenet_zernike_coefficients.csv")

    if plot:
        vis.diagnosis(
            pred=wavefront,
            pred_std=Wavefront(np.zeros_like(wavefront.amplitudes_ansi)),
            save_path=Path(f"{inputs.with_suffix('')}_phasenet_diagnosis"),
        )

    return wavefront.amplitudes_ansi


@profile
def predict_phaseretrieval(
    inputs: Path,
    plot: bool = False,
    psfgen: Optional[SyntheticPSF] = None,
    num_iterations: int = 200,
    use_pyotf_zernikes: bool = False,
    ignore_modes: list = (0, 1, 2, 4),
    prediction_threshold: float = 0.0,
):
    try:
        import pyotf.pyotf.phaseretrieval as pr
        from pyotf.pyotf.utils import prep_data_for_PR, psqrt
        from pyotf.pyotf.zernike import osa2degrees
        from pyotf.pyotf.otf import HanserPSF, SheppardPSF
    except ImportError as e:
        logger.error(e)
        return -1
    
    if psfgen is None:
        psfgen = SyntheticPSF(
            psf_type='widefield',
            lls_excitation_profile=None,
            psf_shape=(64, 64, 64),
            n_modes=15,
            lam_detection=.510,
            x_voxel_size=.097,
            y_voxel_size=.097,
            z_voxel_size=.1,
        )
    
    data = backend.load_sample(inputs)
    crop_shape = [utils.round_to_odd(dim_len - .1) for dim_len in data.shape]
    logger.info(f'Cropping from {data.shape} to {crop_shape}')
    data = preprocessing.resize_image(data, crop_shape)  # make each dimension an odd number of voxels
    
    params = dict(
        wl=psfgen.lam_detection,
        na=psfgen.na_detection,
        ni=psfgen.refractive_index,
        res=psfgen.x_voxel_size,
        zres=psfgen.z_voxel_size,
    )  # all in microns
    
    logger.info("Starting phase retrieval iterations")
    data_prepped = prep_data_for_PR(np.flip(data.astype(int), axis=0), multiplier=1.15)
    logger.info(f"Subtracted background of: {np.max(data) - np.max(data_prepped):0.2f} counts")
    
    try:
        data_prepped = cp.asarray(data_prepped)  # use GPU. Comment this line to use CPU.
    except Exception:
        logger.warning(f"No CUDA-capable device is detected. 'image' will be type {type(data_prepped)}")
    
    pr_result = pr.retrieve_phase(
        data_prepped,
        params,
        max_iters=num_iterations,
        pupil_tol=1e-6,
        mse_tol=1e-6,
        phase_only=False
    )
    pupil = pr_result.phase / (2 * np.pi)  # convert radians to waves
    pupil[pupil != 0.] -= np.mean(pupil[pupil != 0.])  # remove a piston term by subtracting the mean of the pupil
    pr_result.phase = utils.waves2microns(pupil, wavelength=psfgen.lam_detection)  # convert waves to um before fitting.
    pr_result.fit_to_zernikes(psfgen.n_modes - 1, mapping=osa2degrees)  # pyotf's zernikes now in um rms
    pr_result.phase = pupil  # phase is now again in waves
    
    pupil[pupil == 0.] = np.nan  # put NaN's outside of pupil
    pupil_path = Path(f"{inputs.with_suffix('')}_phase_retrieval_wavefront.tif")
    imwrite(pupil_path, cp.asnumpy(pupil))
    
    threshold = utils.waves2microns(prediction_threshold, wavelength=psfgen.lam_detection)
    ignore_modes = list(map(int, ignore_modes))
    
    if use_pyotf_zernikes:
        # use pyotf definition of zernikes and fit using them. I suspect m=0 modes have opposite sign to our definition.
        pred = np.zeros(psfgen.n_modes)
        pred[1:] = cp.asnumpy(pr_result.zd_result.pcoefs)
        pred[ignore_modes] = 0.
        pred[np.abs(pred) <= threshold] = 0.
        pred = Wavefront(pred, modes=psfgen.n_modes, order='ansi', lam_detection=psfgen.lam_detection)
    else:
        # use our definition of zernikes and fit using them
        pred = Wavefront(pupil_path, modes=psfgen.n_modes, order='ansi', lam_detection=psfgen.lam_detection)
    
    coefficients = [
        {'n': z.n, 'm': z.m, 'amplitude': a}
        for z, a in pred.zernikes.items()
    ]
    coefficients = pd.DataFrame(coefficients, columns=['n', 'm', 'amplitude'])
    coefficients.index.name = 'ansi'
    coefficients.to_csv(f"{inputs.with_suffix('')}_phase_retrieval_zernike_coefficients.csv")
    
    if plot:
        vis.diagnosis(
            pred=pred,
            pred_std=Wavefront(np.zeros_like(pred.amplitudes_ansi)),
            save_path=Path(f"{inputs.with_suffix('')}_phase_retrieval_diagnosis"),
        )
    
    return pred.amplitudes_ansi


@profile
def phasenet_heatmap(
    inputs: Path,
    outdir: Path,
    iter_num: int = 1,
    distribution: str = '/',
    batch_size: int = 128,
    samplelimit: Any = None,
    na: float = 1.0,
    eval_sign: str = 'signed',
    agg: str = 'median',
    modes: int = 15,
    num_beads: Optional[int] = 1,
    simulate_psf_only: bool = False,
    phasenet_path: Path = Path('phasenet_repo'),
    denoiser: Optional[Path] = None,
    denoiser_window_size: tuple = (32, 64, 64),
):
    download_phasenet(phasenet_path)
    from phasenet_repo.phasenet.model import PhaseNet
    
    if outdir == Path('../benchmarks'):
        savepath = outdir / 'phasenet' / eval_sign
        
        if num_beads is None:
            savepath = savepath / 'psf'
        else:
            savepath = savepath / f'beads-{num_beads}'
        
        if distribution != '/':
            savepath = Path(f'{savepath}/{distribution}_na_{str(na).replace("0.", "p")}')
        else:
            savepath = Path(f'{savepath}/na_{str(na).replace("0.", "p")}')
    else:
        savepath = outdir
    
    savepath.mkdir(parents=True, exist_ok=True)
    
    phasenet = PhaseNet(None, name='16_05_2020_11_48_14_berkeley_50planes', basedir=f'{phasenet_path}/models/')
    
    if denoiser is not None:
        if isinstance(denoiser, Path):
            logger.info(f"Loading denoiser model: {denoiser}")
            denoiser = CARE(config=None, name=denoiser.name, basedir=denoiser.parent)
            logger.info(f"{denoiser.name} loaded")
            
    phasenetgen = SyntheticPSF(
        psf_type='widefield',
        lls_excitation_profile=None,
        psf_shape=(64, 64, 64),
        n_modes=modes,
        lam_detection=.510,
        x_voxel_size=.086,
        y_voxel_size=.086,
        z_voxel_size=.1,
        na_detection=1.1,
        refractive_index=1.33,
    )

    if inputs.suffix == '.csv':
        results = pd.read_csv(inputs, header=0, index_col=0)

    elif Path(f'{savepath}_predictions.csv').exists():
        # continue from previous results, ignoring criteria
        results = pd.read_csv(f'{savepath}_predictions.csv', header=0, index_col=0)

    else:
        # on first call, setup the dataframe with the 0th iteration stuff
        results = eval.collect_data(
            datapath=inputs,
            model_output_shape=15,
            model_input_shape=phasenetgen.psf_shape,
            samplelimit=samplelimit,
            distribution=distribution,
            photons_range=None,
            npoints_range=(1, num_beads) if num_beads is not None else None,
            psf_type=phasenetgen.psf_type,
            lam_detection=phasenetgen.lam_detection,
        )
    
    if 'object_gaussian_sigma' not in results.columns:
        results['object_gaussian_sigma'] = 0.
    
    if iter_num != results['iter_num'].values.max():
        prediction_cols = [col for col in results.columns if col.endswith('_prediction')]
        ground_truth_cols = [col for col in results.columns if col.endswith('_ground_truth')]
        residual_cols = [col for col in results.columns if col.endswith('_residual')]
        previous = results[results['iter_num'] == iter_num - 1]   # previous iteration = iter_num - 1

        # create realspace images for the current iteration
        paths = utils.multiprocess(
            func=partial(
                eval.generate_sample,
                iter_number=iter_num,
                savedir=savepath.resolve(),
                data=previous,
                psfgen=phasenetgen,
                no_phase=False,
                digital_rotations=None,
                simulate_psf_only=simulate_psf_only,
                denoiser=denoiser,
                denoiser_window_size=denoiser_window_size
            ),
            jobs=previous['id'].values,
            desc=f'Generate samples ({savepath.resolve()})',
            unit=' sample',
            cores=-1
        )

        current = previous.copy()
        current['iter_num'] = iter_num
        current['file'] = paths
        current['file_windows'] = [utils.convert_to_windows_file_string(f) for f in paths]

        current[ground_truth_cols] = previous[residual_cols]
        current[prediction_cols] = np.array([
            predict_phasenet(p, phasenet=phasenet, phasenetgen=phasenetgen)
            for p in paths
        ])

        if eval_sign == 'positive_only':
            current[ground_truth_cols] = current[ground_truth_cols].abs()
            current[prediction_cols] = current[prediction_cols].abs()

        current[residual_cols] = current[ground_truth_cols].values - current[prediction_cols].values

        # compute residuals for each sample
        current['residuals'] = current.apply(
            lambda row: Wavefront(row[residual_cols].values, lam_detection=phasenetgen.lam_detection).peak2valley(na=na),
            axis=1
        )

        current['residuals_umRMS'] = current.apply(
            lambda row: np.linalg.norm(row[residual_cols].values),
            axis=1
        )

        results = pd.concat([results, current], ignore_index=True, sort=False)

        if savepath is not None:
            try:
                results.to_csv(f'{savepath}_predictions.csv')
            except PermissionError:
                savepath = f'{savepath}_x'
                results.to_csv(f'{savepath}_predictions.csv')
            logger.info(f'Saved: {savepath.resolve()}_predictions.csv')
    
    if 'aberration_umRMS' not in results.columns.values:
        results['aberration_umRMS'] = np.nan
        for idx in results.id.values:
            results.loc[results.id == idx, 'aberration_umRMS'] = results[results.id == idx].iloc[0]['residuals_umRMS']

    df = results[results['iter_num'] == iter_num]
    df['photoelectrons'] = utils.photons2electrons(df['photons'], quantum_efficiency=.82)

    for x in ['photons', 'photoelectrons', 'counts', 'counts_p100', 'counts_p99']:

        if x == 'photons':
            label = f'Integrated photons'
            lims = (0, 5 * 10 ** 5)
            pbins = np.arange(lims[0], lims[-1] + 1e4, 5e4)
        elif x == 'photoelectrons':
            label = f'Integrated photoelectrons'
            lims = (0, 5 * 10 ** 5)
            pbins = np.arange(lims[0], lims[-1] + 1e4, 5e4)
        elif x == 'counts':
            label = f'Integrated counts'
            lims = (2.6e7, 3e7)
            pbins = np.arange(lims[0], lims[-1]+2e5, 1e5)
        elif x == 'counts_p100':
            label = f'Max counts (camera background offset = 100)'
            lims = (100, 2000)
            pbins = np.arange(lims[0], lims[-1]+400, 200)
        else:
            label = f'99th percentile of counts (camera background offset = 100)'
            lims = (100, 300)
            pbins = np.arange(lims[0], lims[-1]+50, 25)

        df['pbins'] = pd.cut(df[x], pbins, labels=pbins[1:], include_lowest=True)
        
        for agg in ['mean', 'median']:
            bins = np.arange(0, 2.55, .05).round(2)
            df['ibins'] = pd.cut(
                df['aberration_umRMS'].apply(partial(utils.microns2waves, wavelength=phasenetgen.lam_detection)),
                bins,
                labels=bins[1:],
                include_lowest=True
            )
            rms_dataframe = pd.pivot_table(df, values='residuals_umRMS', index='ibins', columns='pbins', aggfunc=agg)
            rms_dataframe = rms_dataframe.applymap(partial(utils.microns2waves, wavelength=phasenetgen.lam_detection))
            rms_dataframe.insert(0, 0, rms_dataframe.index.values.astype(df['residuals'].dtype))
            
            eval.plot_heatmap_rms(
                rms_dataframe,
                histograms=df if x == 'photons' else None,
                wavelength=phasenetgen.lam_detection,
                savepath=Path(f"{savepath}_iter_{iter_num}_{x}_{agg}"),
                label=label,
                lims=lims,
                agg=agg,
                sci=True,
            )
            
            coverage = pd.pivot_table(df, values='residuals_umRMS', index='ibins', columns='pbins', aggfunc='count')
            eval.plot_coverage(
                coverage,
                wavelength=phasenetgen.lam_detection,
                savepath=Path(f"{savepath}_iter_{iter_num}_{x}_coverage"),
                label=label,
                lims=lims,
                sci=True,
                p2v=False
            )
            
            bins = np.arange(0, 10.25, .25).round(2)
            df['ibins'] = pd.cut(
                df['aberration'],
                bins,
                labels=bins[1:],
                include_lowest=True
            )
            dataframe = pd.pivot_table(df, values='residuals', index='ibins', columns='pbins', aggfunc=agg)
            dataframe.insert(0, 0, dataframe.index.values.astype(df['residuals'].dtype))
            
            dataframe.to_csv(f'{savepath}_{x}_{agg}.csv')
            logger.info(f'Saved: {savepath.resolve()}_{x}_{agg}.csv')
            
            eval.plot_heatmap_p2v(
                dataframe,
                histograms=df if x == 'photons' else None,
                wavelength=phasenetgen.lam_detection,
                savepath=Path(f"{savepath}_iter_{iter_num}_{x}_{agg}"),
                label=label,
                hist_col='residuals',
                sci=True,
                lims=lims,
                agg=agg
            )
            
            coverage = pd.pivot_table(df, values='residuals_umRMS', index='ibins', columns='pbins', aggfunc='count')
            eval.plot_coverage(
                coverage,
                wavelength=phasenetgen.lam_detection,
                savepath=Path(f"{savepath}_iter_{iter_num}_{x}_coverage"),
                label=label,
                lims=lims,
                sci=True,
                p2v=True
            )
    return savepath


@profile
def phaseretrieval_heatmap(
    inputs: Path,
    outdir: Path,
    iter_num: int = 1,
    distribution: str = '/',
    batch_size: int = 128,
    samplelimit: Any = None,
    na: float = 1.0,
    eval_sign: str = 'signed',
    agg: str = 'median',
    modes: int = 15,
    num_beads: Optional[int] = 1,
    simulate_psf_only: bool = False,
    denoiser: Optional[Path] = None,
    denoiser_window_size: tuple = (32, 64, 64),
):
    try:
        import pyotf.pyotf.phaseretrieval as pr
        from pyotf.pyotf.utils import prep_data_for_PR, psqrt
        from pyotf.pyotf.zernike import osa2degrees
        from pyotf.pyotf.otf import HanserPSF, SheppardPSF
    except ImportError as e:
        logger.error(e)
        return -1
    
    if outdir == Path('../benchmarks'):
        savepath = outdir / 'phaseretrieval' / eval_sign
        
        if num_beads is None:
            savepath = savepath / 'psf'
        else:
            savepath = savepath / f'beads-{num_beads}'
        
        if distribution != '/':
            savepath = Path(f'{savepath}/{distribution}_na_{str(na).replace("0.", "p")}')
        else:
            savepath = Path(f'{savepath}/na_{str(na).replace("0.", "p")}')
    else:
        savepath = outdir
    
    savepath.mkdir(parents=True, exist_ok=True)
    
    psfgen = SyntheticPSF(
        psf_type='widefield',
        lls_excitation_profile=None,
        psf_shape=(64, 64, 64),
        n_modes=modes,
        lam_detection=.510,
        x_voxel_size=.097,
        y_voxel_size=.097,
        z_voxel_size=.1,
    )
    
    if inputs.suffix == '.csv':
        results = pd.read_csv(inputs, header=0, index_col=0)
    
    elif Path(f'{savepath}_predictions.csv').exists():
        # continue from previous results, ignoring criteria
        results = pd.read_csv(f'{savepath}_predictions.csv', header=0, index_col=0)
    
    else:
        # on first call, setup the dataframe with the 0th iteration stuff
        results = eval.collect_data(
            datapath=inputs,
            model_output_shape=15,
            model_input_shape=psfgen.psf_shape,
            samplelimit=samplelimit,
            distribution=distribution,
            photons_range=None,
            npoints_range=(1, num_beads) if num_beads is not None else None,
            psf_type=psfgen.psf_type,
            lam_detection=psfgen.lam_detection,
        )
    
    if denoiser is not None:
        if isinstance(denoiser, Path):
            logger.info(f"Loading denoiser model: {denoiser}")
            denoiser = CARE(config=None, name=denoiser.name, basedir=denoiser.parent)
            logger.info(f"{denoiser.name} loaded")
    
    if 'object_gaussian_sigma' not in results.columns:
        results['object_gaussian_sigma'] = 0.
    
    if iter_num != results['iter_num'].values.max():
        prediction_cols = [col for col in results.columns if col.endswith('_prediction')]
        ground_truth_cols = [col for col in results.columns if col.endswith('_ground_truth')]
        residual_cols = [col for col in results.columns if col.endswith('_residual')]
        previous = results[results['iter_num'] == iter_num - 1]  # previous iteration = iter_num - 1
        
        # create realspace images for the current iteration
        paths = utils.multiprocess(
            func=partial(
                eval.generate_sample,
                iter_number=iter_num,
                savedir=savepath.resolve(),
                data=previous,
                psfgen=psfgen,
                no_phase=False,
                digital_rotations=None,
                simulate_psf_only=simulate_psf_only,
                denoiser=denoiser,
                denoiser_window_size=denoiser_window_size
            ),
            jobs=previous['id'].values,
            desc=f'Generate samples ({savepath.resolve()})',
            unit=' sample',
            cores=-1
        )
        
        current = previous.copy()
        current['iter_num'] = iter_num
        current['file'] = paths
        current['file_windows'] = [utils.convert_to_windows_file_string(f) for f in paths]
        
        current[ground_truth_cols] = previous[residual_cols]
        current[prediction_cols] = np.array([
            predict_phaseretrieval(p, psfgen=psfgen)
            for p in paths
        ])
        
        if eval_sign == 'positive_only':
            current[ground_truth_cols] = current[ground_truth_cols].abs()
            current[prediction_cols] = current[prediction_cols].abs()
        
        current[residual_cols] = current[ground_truth_cols].values - current[prediction_cols].values
        
        # compute residuals for each sample
        current['residuals'] = current.apply(
            lambda row: Wavefront(row[residual_cols].values, lam_detection=psfgen.lam_detection).peak2valley(
                na=na),
            axis=1
        )
        
        current['residuals_umRMS'] = current.apply(
            lambda row: np.linalg.norm(row[residual_cols].values),
            axis=1
        )
        
        results = pd.concat([results, current], ignore_index=True, sort=False)
        
        if savepath is not None:
            try:
                results.to_csv(f'{savepath}_predictions.csv')
            except PermissionError:
                savepath = f'{savepath}_x'
                results.to_csv(f'{savepath}_predictions.csv')
            logger.info(f'Saved: {savepath.resolve()}_predictions.csv')
    
    if 'aberration_umRMS' not in results.columns.values:
        results['aberration_umRMS'] = np.nan
        for idx in results.id.values:
            results.loc[results.id == idx, 'aberration_umRMS'] = results[results.id == idx].iloc[0]['residuals_umRMS']
    
    df = results[results['iter_num'] == iter_num]
    df['photoelectrons'] = utils.photons2electrons(df['photons'], quantum_efficiency=.82)
    
    for x in ['photons', 'photoelectrons', 'counts', 'counts_p100', 'counts_p99']:
        
        if x == 'photons':
            label = f'Integrated photons'
            lims = (0, 5 * 10 ** 5)
            pbins = np.arange(lims[0], lims[-1] + 1e4, 5e4)
        elif x == 'photoelectrons':
            label = f'Integrated photoelectrons'
            lims = (0, 5 * 10 ** 5)
            pbins = np.arange(lims[0], lims[-1] + 1e4, 5e4)
        elif x == 'counts':
            label = f'Integrated counts'
            lims = (2.6e7, 3e7)
            pbins = np.arange(lims[0], lims[-1] + 2e5, 1e5)
        elif x == 'counts_p100':
            label = f'Max counts (camera background offset = 100)'
            lims = (100, 2000)
            pbins = np.arange(lims[0], lims[-1] + 400, 200)
        else:
            label = f'99th percentile of counts (camera background offset = 100)'
            lims = (100, 300)
            pbins = np.arange(lims[0], lims[-1] + 50, 25)
        
        df['pbins'] = pd.cut(df[x], pbins, labels=pbins[1:], include_lowest=True)
        
        for agg in ['mean', 'median']:
            bins = np.arange(0, 2.55, .05).round(2)
            df['ibins'] = pd.cut(
                df['aberration_umRMS'].apply(partial(utils.microns2waves, wavelength=psfgen.lam_detection)),
                bins,
                labels=bins[1:],
                include_lowest=True
            )
            rms_dataframe = pd.pivot_table(df, values='residuals_umRMS', index='ibins', columns='pbins', aggfunc=agg)
            rms_dataframe = rms_dataframe.applymap(partial(utils.microns2waves, wavelength=psfgen.lam_detection))
            rms_dataframe.insert(0, 0, rms_dataframe.index.values.astype(df['residuals'].dtype))
            
            eval.plot_heatmap_rms(
                rms_dataframe,
                histograms=df if x == 'photons' else None,
                wavelength=psfgen.lam_detection,
                savepath=Path(f"{savepath}_iter_{iter_num}_{x}_{agg}"),
                label=label,
                lims=lims,
                agg=agg,
                sci=True,
            )
            
            coverage = pd.pivot_table(df, values='residuals_umRMS', index='ibins', columns='pbins', aggfunc='count')
            eval.plot_coverage(
                coverage,
                wavelength=psfgen.lam_detection,
                savepath=Path(f"{savepath}_iter_{iter_num}_{x}_coverage"),
                label=label,
                lims=lims,
                sci=True,
                p2v=False
            )
            
            bins = np.arange(0, 10.25, .25).round(2)
            df['ibins'] = pd.cut(
                df['aberration'],
                bins,
                labels=bins[1:],
                include_lowest=True
            )
            dataframe = pd.pivot_table(df, values='residuals', index='ibins', columns='pbins', aggfunc=agg)
            dataframe.insert(0, 0, dataframe.index.values.astype(df['residuals'].dtype))
            
            dataframe.to_csv(f'{savepath}_{x}_{agg}.csv')
            logger.info(f'Saved: {savepath.resolve()}_{x}_{agg}.csv')
            
            eval.plot_heatmap_p2v(
                dataframe,
                histograms=df if x == 'photons' else None,
                wavelength=psfgen.lam_detection,
                savepath=Path(f"{savepath}_iter_{iter_num}_{x}_{agg}"),
                label=label,
                hist_col='residuals',
                sci=True,
                lims=lims,
                agg=agg
            )
            
            coverage = pd.pivot_table(df, values='residuals_umRMS', index='ibins', columns='pbins', aggfunc='count')
            eval.plot_coverage(
                coverage,
                wavelength=psfgen.lam_detection,
                savepath=Path(f"{savepath}_iter_{iter_num}_{x}_coverage"),
                label=label,
                lims=lims,
                sci=True,
                p2v=True
            )
    return savepath


@profile
def predict_cocoa(
    inputs: Union[Path, str, np.array],
    plot: bool = False,
    axial_voxel_size: float = .086,
    lateral_voxel_size: float = .2,
    na_detection: float = 1.1,
    lam_detection: float = .510,
    refractive_index: float = 1.33,
    decon_iters: int = 30,
    psf_type: str = 'widefield',
    cocoa_path: Path = Path('cocoa_repo'),
    decon: bool = True,
):
    if isinstance(inputs, np.ndarray):
        savepath = None
    else:
        savepath = inputs.with_suffix('')
    download_cocoa(cocoa_path)

    import os
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    import torch
    import torch.nn as nn
    from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

    dtype = torch.cuda.FloatTensor
    torch.backends.cudnn.benchmark = True

    from cocoa_repo.misc import models as cocoa_models
    from cocoa_repo.misc import utils as cocoa_utils
    from cocoa_repo.misc import losses as cocoa_losses
    from cocoa_repo.misc import psf_torch as cocoa_psf

    parser = argparse.ArgumentParser(description="CoCoA")
    parser.add_argument('--padding', type=int, default=24)
    parser.add_argument('--normalized', type=bool, default=True)

    parser.add_argument('--encoding_option', type=str, default='radial')  # 'cartesian', 'radial'
    parser.add_argument('--radial_encoding_angle', type=float, default=3,
                        help='Typically, 3 ~ 7.5. Smaller values indicates the ability to represent fine features.')
    parser.add_argument('--radial_encoding_depth', type=int, default=7,
                        help='If too large, stripe artifacts. If too small, oversmoothened features. Typically, 6 or 7.')  # 7, 8 (jiggling artifacts)

    parser.add_argument('--nerf_num_layers', type=int, default=6)
    parser.add_argument('--nerf_num_filters', type=int,
                        default=128)  # 32 (not enough), 64, 128 / at least y_.shape[0]/2? Helps to reduce artifacts fitted to aberrated features and noise.
    parser.add_argument('--nerf_skips', type=list,
                        default=[2, 4, 6])  # [2,4,6], [2,4,6,8]: good, [2,4], [4], [4, 8]: insufficient.
    parser.add_argument('--nerf_beta', type=float, default=1.0)  # 1.0 or None (sigmoid)
    parser.add_argument('--nerf_max_val', type=float, default=40.0)

    parser.add_argument('--pretraining', type=bool, default=True)  # True, False
    parser.add_argument('--pretraining_num_iter', type=int, default=500)  # 2500
    parser.add_argument('--pretraining_lr', type=float, default=1e-2)
    parser.add_argument('--pretraining_measurement_scalar', type=float, default=3.5)  # 3.5
    parser.add_argument('--training_num_iter', type=int, default=1000)
    parser.add_argument('--training_lr_obj', type=float, default=5e-3)
    parser.add_argument('--training_lr_ker', type=float, default=1e-2)  # 1e-2
    parser.add_argument('--kernel_max_val', type=float, default=1e-2)
    parser.add_argument('--kernel_order_up_to', type=int, default=4)  # True, False

    parser.add_argument('--ssim_weight', type=float, default=1.0)
    parser.add_argument('--tv_z', type=float, default=1e-9,
                        help='larger tv_z helps for denser samples.')
    parser.add_argument('--tv_z_normalize', type=bool, default=False)
    parser.add_argument('--rsd_reg_weight', type=float, default=5e-4,  # 5e-4 ~ 2.5e-3 with radial.
                        help='Helps to retrieve aberrations correctly. Too large, skeletionize the image.')

    parser.add_argument('--lr_schedule', type=str, default='cosine')  # 'multi_step', 'cosine'

    args = parser.parse_args(args=[])

    img = backend.load_sample(inputs)

    psfgen = PsfGenerator3D(
        psf_shape=img.shape,
        units=(axial_voxel_size, lateral_voxel_size, lateral_voxel_size),
        lam_detection=lam_detection,
        n=refractive_index,
        na_detection=na_detection,
        psf_type=psf_type,
    )

    if psfgen.lls_excitation_profile is not None:
        lls_excitation_profile = psfgen.lls_excitation_profile.copy()
        lls_excitation_profile /= np.max(lls_excitation_profile)
        # if psf_type is not 'widefield' or 'confocal'
        lls_excitation_profile = torch.from_numpy(lls_excitation_profile).type(dtype).cuda(0).view(
            psfgen.lls_excitation_profile.shape[0],
            psfgen.lls_excitation_profile.shape[1],
            psfgen.lls_excitation_profile.shape[2]
        )

    y_max = np.max(img)
    y_min = np.min(img)
    y_ = (img - y_min) / (y_max - y_min)

    y = torch.from_numpy(y_.copy()).type(dtype).cuda(0).view(y_.shape[0], y_.shape[1], y_.shape[2])

    INPUT_HEIGHT = y_.shape[1]
    INPUT_WIDTH = y_.shape[2]
    INPUT_DEPTH = y_.shape[0]

    psf = cocoa_psf.PsfGenerator3D(
        psf_shape=(y_.shape[0], y_.shape[1], y_.shape[2]),
        units=(axial_voxel_size, lateral_voxel_size, lateral_voxel_size),
        na_detection=na_detection,
        lam_detection=lam_detection,
        n=refractive_index
    )

    coordinates = cocoa_models.input_coord_2d(INPUT_WIDTH, INPUT_HEIGHT).cuda(0)

    if args.encoding_option == 'cartesian':
        # print('Cartesian encoding')
        embed_func = cocoa_models.Embedding(
            args.cartesian_encoding_dim,
            args.cartesian_encoding_depth
        ).cuda(0)
        coordinates = embed_func(coordinates).cuda(0)

    elif args.encoding_option == 'radial':
        # print('Radial encoding')
        # sometime causes unwanted astigmatism, but works better with dense samples.
        coordinates = cocoa_models.radial_encoding(
            coordinates,
            args.radial_encoding_angle,
            args.radial_encoding_depth
        ).cuda(0)

    net_obj = cocoa_models.NeRF(
        D=args.nerf_num_layers,
        W=args.nerf_num_filters,
        skips=args.nerf_skips,
        in_channels=coordinates.shape[-1],
        out_channels=INPUT_DEPTH
    ).cuda(0)

    if args.pretraining:  # (aka just training because there's no self-supervision here)
        t_start = time.time()

        optimizer = torch.optim.Adam([{'params': net_obj.parameters(), 'lr': args.pretraining_lr}],
                                     betas=(0.9, 0.999), eps=1e-8)
        if args.lr_schedule == 'multi_step':
            scheduler = MultiStepLR(optimizer, milestones=[1000, 1500, 2000], gamma=0.5)
        elif args.lr_schedule == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, args.pretraining_num_iter, args.pretraining_lr / 25)

        loss_list = np.empty(shape=(1 + args.pretraining_num_iter,))
        loss_list[:] = np.NaN

        for step in tqdm(range(args.pretraining_num_iter), desc=f"Pretraining (without PSF), {args.pretraining_num_iter} steps", unit='steps', leave=False, delay=5, mininterval=2):
            out_x = net_obj(coordinates)

            if args.nerf_beta is None:
                out_x = args.nerf_max_val * nn.Sigmoid()(out_x)
            else:
                out_x = nn.Softplus(beta=args.nerf_beta)(out_x)

            out_x_m = out_x.view(y_.shape[1], y_.shape[2], y_.shape[0]).permute(2, 0, 1)
            loss = cocoa_losses.ssim_loss(out_x_m, args.pretraining_measurement_scalar * y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_list[step] = loss.item()

        t_end = time.time()
        # logger.info(f'Initialization: {(t_end - t_start):.2f} elapsed seconds.')

        # torch.save(net_obj.state_dict(), net_obj_save_path_pretrained)
        # print('Pre-trained model saved.')

    # kernel with simple coefficients
    net_ker = cocoa_models.optimal_kernel(
        max_val=args.kernel_max_val,
        order_up_to=args.kernel_order_up_to,
        piston_tip_tilt=False
    )  # 5e-2

    # Dan thinks: The net_ker.parameters are given to Adam to optimize (along with the weights of the model) to
    # minimize loss. The net_ker.parameters() gets the named parameters (which is just _k which is a list of zernikes).
    # This list then gets made into a wavefront to generate an aberrated PSF which is convolved with the object (out_x),
    # to calculate the loss.

    optimizer = torch.optim.Adam([{'params': net_obj.parameters(), 'lr': args.training_lr_obj},  # 1e-3
                                  {'params': net_ker.parameters(), 'lr': args.training_lr_ker}],  # 4e-3
                                 betas=(0.9, 0.999), eps=1e-8)

    scheduler = CosineAnnealingLR(optimizer, args.training_num_iter, args.training_lr_ker / 25)

    loss_list = np.empty(shape=(1 + args.training_num_iter,))
    loss_list[:] = np.NaN

    wfe_list = np.empty(shape=(1 + args.training_num_iter,))
    wfe_list[:] = np.NaN

    lr_obj_list = np.empty(shape=(1 + args.training_num_iter,))
    lr_obj_list[:] = np.NaN

    lr_ker_list = np.empty(shape=(1 + args.training_num_iter,))
    lr_ker_list[:] = np.NaN

    t_start = time.time()

    for step in tqdm(range(args.training_num_iter), desc=f"Training (with PSF), {args.training_num_iter} steps", unit='steps', leave=False, delay=5, mininterval=2):
        out_x = net_obj(coordinates)

        if args.nerf_beta is None:
            out_x = args.nerf_max_val * nn.Sigmoid()(out_x)
        else:
            out_x = nn.Softplus(beta=args.nerf_beta)(out_x)
            out_x = torch.minimum(torch.full_like(out_x, args.nerf_max_val), out_x)  # 30.0

        out_x_m = out_x.view(y_.shape[1], y_.shape[2], y_.shape[0]).permute(2, 0, 1)

        wf = net_ker.k

        if psf_type == 'widefield':
            out_k_m = psf.incoherent_psf(wf, normalized=args.normalized) / y_.shape[0]

        else:
            out_k_m = psf.incoherent_psf(wf, normalized=args.normalized) / y_.shape[0] * lls_excitation_profile

        k_vis = psf.masked_phase_array(wf, normalized=args.normalized)
        out_y = cocoa_utils.fft_convolve(out_x_m, out_k_m, mode='fftn')

        loss = args.ssim_weight * cocoa_losses.ssim_loss(out_y, y)
        loss += cocoa_utils.single_mode_control(wf, 1, -0.0, 0.0)  # quite crucial for suppressing unwanted defocus.
        loss += args.tv_z * cocoa_losses.tv_1d(out_x_m, axis='z', normalize=args.tv_z_normalize)
        loss += args.rsd_reg_weight * torch.reciprocal(torch.std(out_x_m) / torch.mean(out_x_m))  # 4e-3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.lr_schedule == 'cosine':
            scheduler.step()

        elif args.lr_schedule == 'multi_step':
            if step == 500 - 1:
                optimizer.param_groups[0]['lr'] = args.training_lr_obj / 10

            if step == 750 - 1:
                optimizer.param_groups[1]['lr'] = args.training_lr_ker / 10
                optimizer.param_groups[0]['lr'] = args.training_lr_obj / 100

        loss_list[step] = loss.item()
        wfe_list[step] = cocoa_utils.torch_to_np(lam_detection * 1e3 * torch.sqrt(torch.sum(torch.square(wf))))  # wave -> nm RMS
        lr_obj_list[step] = optimizer.param_groups[0]['lr']
        lr_ker_list[step] = optimizer.param_groups[1]['lr']

    t_end = time.time()
    # logger.info(f'Training: {(t_end - t_start):.2f} elapsed seconds.')

    y = cocoa_utils.torch_to_np(y)
    out_k_m = cocoa_utils.torch_to_np(out_k_m)
    reconstructed = cocoa_utils.torch_to_np(out_x_m) * np.sum(out_k_m) * (y_max - y_min) + y_min  # undo normalization: (img - y_min) / (y_max - y_min) = out_x_m
    out_y = cocoa_utils.torch_to_np(out_y) * (y_max - y_min) + y_min   # undo normalization: (img - y_min) / (y_max - y_min) = out_y
    zernikes = cocoa_utils.torch_to_np(wf)
    predicted_wavefront = np.fft.fftshift(k_vis.detach().cpu().numpy())

    wavefront = Wavefront(
        amplitudes=[0, 0, 0] + list(zernikes),
        lam_detection=lam_detection,
        modes=15,
        order='ansi',
        rotate=False,
    )

    coefficients = [
        {'n': z.n, 'm': z.m, 'amplitude': a}
        for z, a in wavefront.zernikes.items()
    ]
    df = pd.DataFrame(coefficients, columns=['n', 'm', 'amplitude'])
    df.index.name = 'ansi'

    if savepath is not None:
        df.to_csv(f"{savepath}_cocoa_zernike_coefficients.csv")

        imwrite(f"{savepath}_cocoa_predicted_psf.tif", out_k_m, compression='deflate', dtype=np.float32)
        imwrite(f"{savepath}_cocoa_wavefront.tif", wavefront.wave(), compression='deflate', dtype=np.float32)
        imwrite(f"{savepath}_cocoa_predicted_wavefront.tif", predicted_wavefront, compression='deflate', dtype=np.float32)
        imwrite(f"{savepath}_cocoa_estimated.tif", out_y, compression='deflate', dtype=np.float32)
        imwrite(f"{savepath}_cocoa_reconstructed.tif", reconstructed, compression='deflate', dtype=np.float32)

        logger.info(f"Raw data: \t{Path(f'{savepath}.tif').resolve()}")
        logger.info(f"Saved prediction of the raw data to: \t{Path(f'{savepath}_cocoa_estimated.tif').resolve()}")
        logger.info(f"Saved prediction of the object to:   \t{Path(f'{savepath}_cocoa_reconstructed.tif').resolve()}")

        predicted_psf = psfgen.incoherent_psf(phi=wavefront)
        predicted_psf /= predicted_psf.sum()
        imwrite(f"{savepath}_cocoa_psf.tif", predicted_psf, compression='deflate', dtype=np.float32)
        if decon:
            out_decon = utils.fft_decon(kernel=predicted_psf, sample=img, iters=decon_iters)
            imwrite(f"{savepath}_cocoa_deconvolved.tif", out_decon, compression='deflate', dtype=np.float32)
            logger.info(f"Saved deconvolved (w/ cocoa PSF) to: \t{Path(f'{savepath}_cocoa_deconvolved.tif').resolve()}")
    
        if plot:
            plt.rcParams.update({
                'font.size': 10,
                'axes.titlesize': 12,
                'axes.labelsize': 12,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'axes.autolimit_mode': 'round_numbers'
            })

            fig = plt.figure()
            mat = plt.imshow(predicted_wavefront)
            cbar = plt.colorbar(mat)
            vis.savesvg(fig, Path(f"{savepath}_cocoa_predicted_wavefront.svg"))

            fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 16))

            vis.plot_mip(
                vol=img/img.max(),
                xy=axes[0, 0],
                xz=axes[0, 1],
                yz=axes[0, 2],
                dxy=lateral_voxel_size,
                dz=axial_voxel_size,
                label=r'Input (MIP) [$\gamma$=.5]'
            )

            vis.plot_mip(
                vol=out_y/out_y.max(),
                xy=axes[1, 0],
                xz=axes[1, 1],
                yz=axes[1, 2],
                dxy=lateral_voxel_size,
                dz=axial_voxel_size,
                label=r'Estimated (MIP) [$\gamma$=.5]'
            )

            vis.plot_mip(
                vol=reconstructed/reconstructed.max(),
                xy=axes[2, 0],
                xz=axes[2, 1],
                yz=axes[2, 2],
                dxy=lateral_voxel_size,
                dz=axial_voxel_size,
                label=r'Reconstructed (MIP) [$\gamma$=.5]'
            )

            vis.plot_mip(
                vol=out_decon/out_decon.max(),
                xy=axes[-1, 0],
                xz=axes[-1, 1],
                yz=axes[-1, 2],
                dxy=lateral_voxel_size,
                dz=axial_voxel_size,
                label=r'Deconvolved (MIP) [$\gamma$=.5]'
            )

            vis.savesvg(fig, Path(f"{savepath}_cocoa_mips.svg"))

            vis.diagnosis(
                pred=wavefront,
                pred_std=Wavefront(np.zeros_like(wavefront.amplitudes_ansi)),
                save_path=Path(f"{savepath}_cocoa_diagnosis"),
            )
            logger.info(f'Figure: \t{Path(f"{savepath}_cocoa_mips.svg").resolve()}')

    return out_y, reconstructed, out_k_m