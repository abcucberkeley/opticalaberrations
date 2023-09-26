
import matplotlib
matplotlib.use('Agg')

import logging
import sys
import subprocess
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
plt.set_loglevel('error')

import swifter
import numpy as np
import pandas as pd
import tensorflow as tf
from functools import partial
from line_profiler_pycharm import profile

import utils
import backend
import eval
import vis

from wavefront import Wavefront
from synthetic import SyntheticPSF

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
            order='ansi',
            distribution='mixed',
            mode_weights='pyramid',
        )

    psf = backend.load_sample(inputs)
    psf = utils.resize_with_crop_or_pad(psf, crop_shape=(50, 50, 50))
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
            pred_std=wavefront,
            save_path=Path(f"{inputs.with_suffix('')}_phasenet_predictions_diagnosis"),
        )

    return wavefront.amplitudes_ansi


@profile
def phasenet_heatmap(
    inputs: Path,
    iter_num: int = 1,
    distribution: str = '/',
    batch_size: int = 128,
    samplelimit: Any = None,
    na: float = 1.0,
    eval_sign: str = 'signed',
    agg: str = 'median',
    modes: int = 15,
    no_beads: bool = True,
    phasenet_path: Path = Path('phasenet_repo')
):
    download_phasenet(phasenet_path)
    from phasenet_repo.phasenet.model import PhaseNet

    if no_beads:
        savepath = phasenet_path.with_suffix('') / eval_sign / 'psf'
    else:
        savepath = phasenet_path.with_suffix('') / eval_sign / 'bead'

    savepath.mkdir(parents=True, exist_ok=True)

    if distribution != '/':
        savepath = Path(f'{savepath}/{distribution}_na_{str(na).replace("0.", "p")}')
    else:
        savepath = Path(f'{savepath}/na_{str(na).replace("0.", "p")}')

    phasenet = PhaseNet(None, name='16_05_2020_11_48_14_berkeley_50planes', basedir=f'{phasenet_path}/models/')

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
        order='ansi',
        distribution='mixed',
        mode_weights='pyramid',
    )

    if iter_num == 1:
        # on first call, setup the dataframe with the 0th iteration stuff
        results = eval.collect_data(
            datapath=inputs,
            model=15,
            samplelimit=samplelimit,
            distribution=distribution,
            photons_range=None,
            npoints_range=(1, 1),
            psf_type=phasenetgen.psf_type,
            lam_detection=phasenetgen.lam_detection
        )
    else:
        # read previous results, ignoring criteria
        results = pd.read_csv(f'{savepath}_predictions.csv', header=0, index_col=0)

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
            no_beads=no_beads
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

    results = results.append(current, ignore_index=True)

    if savepath is not None:
        try:
            results.to_csv(f'{savepath}_predictions.csv')
        except PermissionError:
            savepath = f'{savepath}_x'
            results.to_csv(f'{savepath}_predictions.csv')
        logger.info(f'Saved: {savepath.resolve()}_predictions.csv')

    df = results[results['iter_num'] == iter_num]
    df['photoelectrons'] = utils.photons2electrons(df['photons'], quantum_efficiency=.82)

    for x in ['photons', 'photoelectrons', 'counts', 'counts_p100', 'counts_p99']:

        if x == 'photons' or x == 'photoelectrons':
            label = f'Integrated photoelectrons'
            lims = (0, 10**6)
            pbins = np.arange(lims[0], lims[-1]+10e4, 5e4)
        elif x == 'counts':
            label = f'Integrated counts'
            lims = (4e6, 7.5e6)
            pbins = np.arange(lims[0], lims[-1]+2e5, 1e5)
        elif x == 'counts_p100':
            label = f'Max counts'
            lims = (0, 5000)
            pbins = np.arange(lims[0], lims[-1]+400, 200)
        else:
            label = f'99th percentile of counts'
            lims = (0, 300)
            pbins = np.arange(lims[0], lims[-1]+50, 25)

        df['pbins'] = pd.cut(df[x], pbins, labels=pbins[1:], include_lowest=True)
        bins = np.arange(0, 10.25, .25).round(2)
        df['ibins'] = pd.cut(
            df['aberration'],
            bins,
            labels=bins[1:],
            include_lowest=True
        )

        dataframe = pd.pivot_table(df, values='residuals', index='ibins', columns='pbins', aggfunc=agg)
        dataframe.insert(0, 0, dataframe.index.values)

        try:
            dataframe = dataframe.sort_index().interpolate()
        except ValueError:
            pass

        dataframe.to_csv(f'{savepath}_{x}.csv')
        logger.info(f'Saved: {savepath.resolve()}_{x}.csv')

        eval.plot_heatmap_p2v(
            dataframe,
            histograms=df if x == 'photons' else None,
            wavelength=phasenetgen.lam_detection,
            savepath=Path(f"{savepath}_iter_{iter_num}_{x}"),
            label=label,
            lims=lims,
            agg=agg
        )

    return savepath


@profile
def predict_cocoa(
    inputs: Path,
    plot: bool = False,
    iter_num: int = 1,
    cocoa_path: Path = Path('cocoa_repo')
):
    download_cocoa(cocoa_path)
