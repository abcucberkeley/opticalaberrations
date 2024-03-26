import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.set_loglevel('error')

import json
from functools import partial
import fnmatch
import os
import ujson
from pathlib import Path
import multiprocessing as mp
import seaborn as sns
from typing import Any
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

import utils
import vis
import backend
import preprocessing
from wavefront import Wavefront
from embeddings import fft
from synthetic import SyntheticPSF
from experimental import phase_retrieval

import logging
logger = logging.getLogger('')


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

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("left", size="5%", pad=0.1)
    cb = plt.colorbar(m, cax=cax)
    cax.yaxis.set_label_position("right")

    ax2.plot(flat_offsets, '--', label='Flat')
    ax2.plot(offsets, label='DM')

    if pred_offsets is not None:
        ax2.plot(pred_offsets, ':', label='Predictions')

    ax2.grid(True, which="both", axis='both', lw=.1, ls='--', zorder=0)
    ax2.legend(frameon=False, ncol=2, loc='upper right')
    vis.savesvg(fig, save_path)


def eval_dm(
    datadir: Path,
    num_modes: int = 15,
    # gt_postfix: str = 'pr_pupil_waves.tif',
    gt_postfix: str = 'ground_truth_zernike_coefficients.csv',
    postfix: str = 'sample_predictions_zernike_coefficients.csv'
):
    """
    Creates a matrix of the ML predicted amplitudes vs the DM amplitudes.  This shows what scalar to multiply each of
    the modes in the dm_calibration, to produce a dm_calibration that will work for this ML model.

    The ML model has a learned using a different definition of each Zernike mode than the ALPAO provided dm_calibration
    matrix.  Both sign and amplitude need to be adjusted. We take empirical data on beads using applied aberrations to
    the DM. These series of measurements (one at each mode) yield the scaling factor.

    Args:
        datadir: where to find  the data files ('*_lightsheet_ansi_z*.tif'),
                                the ground truth ('*_widefield_z6_pr_pupil_waves.tif'),
                                the ML predictions

        num_modes: number of modes
        gt_postfix: suffix for phase retrieval waves.tif or phase retrieval CSV
        postfix: suffix for ML coefficients CSV

    Returns:

    Hard coded for CamA (skips 'CamB')

    """

    data = np.identity(num_modes)
    logger.info(f'{datadir=}')
    for file in sorted(datadir.glob('before*_lightsheet_ansi_z*.tif')):
        if 'CamB' in str(file):
            continue

        state = file.stem.split('_')[0]
        modes = ':'.join(s.lstrip('z') if s.startswith('z') else '' for s in file.stem.split('_')).split(':')
        modes = [m for m in modes if m]
        logger.info(modes)

        if len(modes) > 1:
            prefix = f"ansi_"
            for m in modes:
                prefix += f"z{m}*"
            if modes[0] == modes[1]:
                mode = int(modes[0])
            else:   # mixed mode. Don't use this one.
                continue
        else:
            mode = int(modes[0])
            prefix = f"ansi_z{modes[0]}*"

        if mode >= num_modes:
            continue

        logger.info(f"Input: {file.name[:75]}....tif              Looking for: {prefix}")
        try:
            gt_path = list(datadir.rglob(f'{state}_widefield_{prefix}_{gt_postfix}'))[0]
            logger.info(f"GrTrth: {gt_path.name}")
        except IndexError:
            logger.warning(f'GT not found for: {file.name}')
            continue

        try:
            prediction_path = list(datadir.rglob(f'{state}_lightsheet_{prefix}_{postfix}'))[0]
            logger.info(f"Predt: {prediction_path.name}")
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
            data[i, mode] = p[i] / magnitude   # normalize by the magnitude of the mode we put on the mirror

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

    print(' ')
    output_file = Path(f'{datadir}/../{datadir.parts[-1]}_dm_matrix')
    df.to_csv(f"{output_file}.csv")
    plt.savefig(f"{output_file}.png", bbox_inches='tight', pad_inches=.25)
    logger.info(f"Saved result to: {output_file.resolve()}.png  .csv")


def calibrate_dm(datadir: Path, dm_calibration: Path):
    """
    Converts DM_vs_prediction matrix to a dm_calibration spreadsheet (e.g. zernike-to-actuators CSV)
    (Run this after eval_dm.)


    Takes the heat maps from dm_matrix.csv from datadir (generated by eval_dm).
        If multiple (could be multiple if heat maps was taken for different DM amplitudes), computes the average
    Uses the diagonal terms to scale the dm_calibration used when collecting the data.

    Args:
        datadir: where dm_matrix are found
        dm_calibration: dm_calibration that was used to collect the data (most likely the ALPAO provided)

    Outputs a png file that has averaged all the input csv's:
        15_mode_calibration.png
    Outputs a CSV file with all 55 modes:
        15_mode_calibration.csv
    Outputs a PNG file showing resulant scalar factors for each of 55 modes:
        15_mode_calibration_scalar_factors.png

    """
    dataframes = []
    logger.info(f'{datadir=}')
    for file in sorted(datadir.glob('*dm_matrix.csv')):
        df = pd.read_csv(file, header=0, index_col=0)
        logger.info(f'Reading {file}, found {df.shape} matrix')
        dataframes.append(df)

    df = pd.concat(dataframes)
    avg = df.groupby(df.index).mean()
    np.set_printoptions(linewidth=np.inf)
    logger.info(f'Diagonal terms of final matrix are: {np.round(np.diag(avg), 2)}')
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
    ax = sns.heatmap(dm, ax=ax, vmin=0, vmax=2, cmap='coolwarm', square=True,
                     cbar_kws={'label': 'Scalar factor (new calib / original)', 'shrink': .8})
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set(ylabel="Actuators", xlabel="Zernike modes", )
    plt.savefig(f"{output_file}_scalar_factors.png", bbox_inches='tight', pad_inches=.25)

    calibration.to_csv(f"{output_file}.csv", header=False, index=False)
    logger.info(f"Saved results to: {output_file.resolve()}.csv  .png  _scalar_factors.png")


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
    gt_unit: str = 'um',
    plot: bool = True,
):
    logger.info(f"Pred: {prediction_path.name}")
    logger.info(f"GT: {gt_path.name}")

    if postfix.startswith('DSH'):
        save_postfix = 'sh'
    elif postfix.startswith('phase_retrieval'):
        save_postfix = 'pr'
    else:
        save_postfix = 'ml'

    save_path = Path(f'{prediction_path.parent}/{prediction_path.stem}_{save_postfix}_eval')

    with open(str(prediction_path).replace('_zernike_coefficients.csv', '_settings.json')) as f:
        predictions_settings = ujson.load(f)

    noisy_img = backend.load_sample(input_path)
    maxcounts = np.max(noisy_img)
    gen = backend.load_metadata(
        model_path,
        psf_shape=noisy_img.shape,
        psf_type='widefield' if save_postfix == 'pr' else None,
        z_voxel_size=.1 if save_postfix == 'pr' else None,
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
    y_wave = Wavefront(y, lam_detection=gen.lam_detection, modes=len(p), unit=gt_unit)
    diff = Wavefront(y_wave.amplitudes-p_wave.amplitudes, lam_detection=gen.lam_detection, modes=len(p))

    if flat_path is not None:
        rfilter = str(input_path.name).split("_Cam", 1)[0] # rfilter = f"{str(input_path.name).replace(gt_postfix, '')}"
        dm_path = Path(str(list(input_path.parent.glob(f"{rfilter}*JSONsettings.json"))[0]))
        dm_wavefront = Path(gt_path.parent/f"{rfilter}_dm_wavefront.svg")

        plot_dm_actuators(
            dm_path=dm_path,
            flat_path=flat_path,
            save_path=dm_wavefront
        )

    residuals = [
        {
            'n': z.n,
            'm': z.m,
            'prediction': p_wave.zernikes[z],
            'ground_truth': y_wave.zernikes[z],
            'residuals': diff.zernikes[z],
        }
        for z in p_wave.zernikes.keys()
    ]

    residuals = pd.DataFrame(residuals, columns=['n', 'm', 'prediction', 'ground_truth', 'residuals'])
    residuals.index.name = 'ansi'
    residuals.to_csv(f'{save_path}_residuals.csv')

    if plot:
        noisy_img = preprocessing.prep_sample(
            noisy_img,
            normalize=normalize,
            remove_background=remove_background,
            windowing=False,
            sample_voxel_size=predictions_settings['sample_voxel_size'],
            na_mask=gen.na_mask
        )

        psfgen = backend.load_metadata(
            model_path,
            psf_shape=(64, 64, 64),
            psf_type='widefield' if save_postfix == 'pr' else None,
            z_voxel_size=.1 if save_postfix == 'pr' else None,
        )
        p_psf = psfgen.single_psf(p_wave, normed=True)
        gt_psf = psfgen.single_psf(y_wave, normed=True)
        corrected_psf = psfgen.single_psf(diff, normed=True)

        plt.style.use("default")
        vis.diagnostic_assessment(
            psf=noisy_img,
            gt_psf=gt_psf,
            predicted_psf=p_psf,
            corrected_psf=corrected_psf,
            maxcounts=maxcounts,
            y=y_wave,
            pred=p_wave,
            save_path=save_path,
            display=False,
            dxy=gen.x_voxel_size,
            dz=gen.z_voxel_size,
            transform_to_align_to_DM=True,
            photons=np.NaN,
        )



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
            # 'psnr': np.mean(res['psnr']),
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
                # 'psnr': np.mean(res['psnr']),
            }   # if we have mixed modes, duplicate for the opposite combination (e.g. 12,13 copied to -> 13,12)

    return results


def eval_dataset(
    datadir: Path,
    flat: Any = None,
    postfix: str = 'predictions_zernike_coefficients.csv',
    gt_postfix: str = 'phase_retrieval_zernike_coefficients.csv',
    plot_evals: bool = True,
    precomputed: bool = False,
    rerun_calc: bool = True,
):
    results = {}
    savepath = Path(f'{datadir}/beads_evaluation')

    if rerun_calc or not Path(f'{savepath}.csv').exists:

        # get model from .json file
        with open(list(Path(datadir / 'MLResults').glob('*_settings.json'))[0]) as f:
            predictions_settings = ujson.load(f)
            model = Path(predictions_settings['model'])

            if not model.exists():
                filename = str(model).split('\\')[-1]
                model = Path(f"../pretrained_models/{filename}")

            mp.set_start_method('spawn', force=True)
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

                pr_path = None
                for gtfile in sorted(datadir.glob('*_widefield_ansi_z*.tif')):
                    if fnmatch.fnmatch(gtfile.name, f'{state}_widefield_{prefix}_*.tif'):
                        pr_path = gtfile
                        break

                if gt_path is None:
                    logger.warning(f'GT not found for: {file.name}.tif')

                prediction_path = None
                for predfile in MLresults_list:
                    if fnmatch.fnmatch(predfile.name, f'{state}_lightsheet_{prefix}_{postfix}'):
                        prediction_path = predfile
                        continue
                if prediction_path is None: logger.warning(f'Prediction not found for: {file.name}')

                ml_img = backend.load_sample(file)
                # ml_img -= 100
                # ml_img = preprocessing.prep_sample(
                #     ml_img,
                #     normalize=True,
                #     remove_background=False,
                #     windowing=False,
                #     sample_voxel_size=predictions_settings['sample_voxel_size'],
                #     na_mask=gen.na_mask
                # )

                pr_img = backend.load_sample(pr_path)
                # pr_img -= 100
                # pr_img = preprocessing.prep_sample(
                #     pr_img,
                #     normalize=True,
                #     remove_background=True,
                #     windowing=False,
                #     sample_voxel_size=predictions_settings['sample_voxel_size']
                # )

                if prediction_path is not None:
                    p = pd.read_csv(prediction_path)
                    ml_wavefront = Wavefront(
                        p.amplitude.values,
                        modes=p.shape[0],
                        lam_detection=predictions_settings['wavelength']
                    )
                else:
                    ml_wavefront = None

                if gt_path is not None:
                    y = pd.read_csv(gt_path)
                    gt_wavefront = Wavefront(
                        y.amplitude.values[:p.shape[0]],
                        modes=p.shape[0],
                        lam_detection=predictions_settings['wavelength']
                    )
                else:
                    gt_wavefront = None

                if gt_wavefront is not None and ml_wavefront is not None:
                    diff_wavefront = Wavefront(
                        gt_wavefront - ml_wavefront,
                        modes=p.shape[0],
                        lam_detection=predictions_settings['wavelength']
                    )

                    results[(state, '-'.join(modes))] = dict(
                        ml_img=ml_img,
                        ml_wavefront=ml_wavefront,
                        gt_img=pr_img,
                        gt_wavefront=gt_wavefront,
                        diff_wavefront=diff_wavefront,
                        residuals=f'{prediction_path.parent}/{prediction_path.stem}_ml_eval_residuals.csv',
                    )

                if not precomputed:
                    logger.info(f"ansi_z{modes}")
                    logger.info(file.stem)
                    logger.info(pr_path.stem)

                    task = partial(
                        evaluate,
                        input_path=file,
                        prediction_path=prediction_path,
                        gt_path=gt_path
                    )
                    _ = pool.apply_async(task)  # issue task

            if not precomputed:
                children = mp.active_children()
                logger.info(f"Awaiting {len(children)} 'eval_mode' tasks to finish.")
            pool.close()    # close the pool
            pool.join()     # wait for all tasks to complete

        postfix = 'predictions_zernike_coefficients_ml_eval_residuals.csv'
        residuals = utils.multiprocess(
            func=process_eval_file,
            jobs=sorted(datadir.rglob(f'*{postfix}'), key=os.path.getctime),  # sort by creation time
            desc=f'Collecting *{postfix} results',
            unit='_eval_files'
        )

        if isinstance(residuals, bool) or len(residuals) == 0:
            raise Exception(f'Did not find eval_mode results in: {Path(datadir / f"*{postfix}").resolve()} \t '
                            f'Please rerun without --precomputed flag to compute _residuals.csv.')

        residuals = pd.DataFrame([v for d in residuals for k, v in d.items()])
        residuals.sort_values(by=['modes', 'iteration_index', 'na'], ascending=[True, True, False], inplace=True)
        print(residuals)


        residuals.to_csv(f'{savepath}.csv')
        np.save(f'{savepath}_results.npy', results)

    else:
        # skip calc. Reload results and just replot
        residuals = pd.read_csv(f'{savepath}.csv')
        print(residuals)
        results = np.load(f'{savepath}_results.npy', allow_pickle='TRUE').item()

    logger.info(f'{savepath}.csv')
    vis.plot_beads_dataset(results, residuals, savepath=savepath)


def eval_ao_dataset(
    datadir: Path,
    flat: Any = None,
    postfix: str = 'predictions_zernike_coefficients.csv',
    gt_postfix: str = 'DSH1_DSH_Wvfrt*',
    gt_unit: str = 'nm',
    plot_evals: bool = True,
    precomputed: bool = False,
    compare_ao_iterations: bool = True,
):
    mldir = Path(datadir/'MLResults')
    ml_results = sorted(mldir.glob('**/*'), key=os.path.getctime)
    sh_results = sorted(Path(datadir/'DSH1/DSH_Wavefront_TIF').glob('**/*.tif'), key=os.path.getctime)
    results = {}

    # get model from .json file
    with open(list(mldir.glob('*_settings.json'))[0]) as f:
        predictions_settings = ujson.load(f)
        model = Path(predictions_settings['model'])

        if not model.exists():
            filename = str(model).split('\\')[-1]
            model = Path(f"../pretrained_models/{filename}")

        logger.info(model)

    logger.info('Beginning evaluations')
    for file in sorted(datadir.glob('MLAO*.tif'), key=os.path.getctime):  # sort by creation time
        if 'CamB' in str(file) or 'pupil' in str(file) or 'autoexpos' in str(file) or 'background' in str(file):
            continue

        method = file.stem.split('_')[0]
        iter_num = int(file.stem.split('_')[1][-1])

        prediction_path = None
        for predfile in ml_results:
            if fnmatch.fnmatch(predfile.name, f'{method}*round{iter_num}*{postfix}'):
                prediction_path = predfile
                break

        if prediction_path is None:
            logger.warning(f'Prediction not found for: {file.name}')
            break

        if prediction_path is not None:
            p = pd.read_csv(prediction_path)
            ml_wavefront = Wavefront(
                p.amplitude.values,
                modes=p.shape[0],
                lam_detection=predictions_settings['wavelength']
            )
        else:
            ml_wavefront = None

        try:
            sh_path = sh_results[iter_num]
            gt_wavefront = Wavefront(
                sh_path,
                modes=p.shape[0],
                lam_detection=predictions_settings['wavelength'],
                unit='nm'
            )
        except IndexError:
            logger.warning(f'GT not found for: {file.name}')
            gt_wavefront = None

        ml_img = backend.load_sample(file)
        ml_img -= 100
        # ml_img = preprocessing.prep_sample(
        #     ml_img,
        #     normalize=True,
        #     remove_background=False,
        #     windowing=False,
        #     sample_voxel_size=predictions_settings['sample_voxel_size']
        # )

        results[iter_num] = dict(
            ml_img=ml_img,
            ml_img_fft=np.abs(fft(ml_img)),
            ml_wavefront=ml_wavefront,
            gt_wavefront=gt_wavefront,
        )

        # if iter_num == 4:
        #     break

    noao = sorted(datadir.glob('NoAO*CamA*.tif'))[-1]
    noao_img = backend.load_sample(noao)
    noao_img -= 100
    # noao_img = preprocessing.prep_sample(
    #     noao_img,
    #     normalize=True,
    #     remove_background=False,
    #     windowing=False,
    #     sample_voxel_size=predictions_settings['sample_voxel_size']
    # )

    prediction_path = sorted(mldir.glob(f'NoAO*{postfix}'))[-1]
    p = pd.read_csv(prediction_path)
    ml_wavefront = Wavefront(
        p.amplitude.values,
        modes=p.shape[0],
        lam_detection=predictions_settings['wavelength']
    )

    sh_path = sh_results[0]
    gt_wavefront = Wavefront(
        sh_path,
        modes=p.shape[0],
        lam_detection=predictions_settings['wavelength'],
        unit='nm'
    )

    results[0] = dict(
        ml_img=noao_img,
        ml_img_fft=np.abs(fft(noao_img)),
        ml_wavefront=ml_wavefront,
        gt_wavefront=gt_wavefront,
    )

    results['noao_img'] = noao_img

    ml_img = backend.load_sample(sorted(datadir.glob('MLAO*CamA*.tif'))[-1])
    ml_img -= 100

    # results['ml_img'] = preprocessing.prep_sample(
    #     ml_img,
    #     normalize=True,
    #     remove_background=False,
    #     windowing=False,
    #     sample_voxel_size=predictions_settings['sample_voxel_size']
    # )

    gt_img = backend.load_sample(sorted(datadir.glob('SHAO*CamA*.tif'))[-1])
    gt_img -= 100
    # results['gt_img'] = preprocessing.prep_sample(
    #     gt_img,
    #     normalize=True,
    #     remove_background=False,
    #     windowing=False,
    #     sample_voxel_size=predictions_settings['sample_voxel_size']
    # )

    samplepsfgen = SyntheticPSF(
        psf_type='../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat',
        psf_shape=noao_img.shape,
        n_modes=15,
        lam_detection=predictions_settings['wavelength'],
        x_voxel_size=predictions_settings['sample_voxel_size'][2],
        y_voxel_size=predictions_settings['sample_voxel_size'][1],
        z_voxel_size=predictions_settings['sample_voxel_size'][0]
    )

    ipsf = samplepsfgen.ipsf
    results['iotf'] = np.abs(fft(ipsf))
    results['na_mask'] = samplepsfgen.na_mask

    vis.compare_ao_iterations(
        results=results,
        num_iters=iter_num,
        save_path=datadir/'iterative_evaluation',
        dxy=predictions_settings['sample_voxel_size'][1],
        dz=predictions_settings['sample_voxel_size'][0],
    )


def plot_dataset_mips(datadir: Path):
    mldir = Path(datadir/'MLResults')

    # get model from .json file
    with open(list(mldir.glob('*_settings.json'))[0]) as f:
        predictions_settings = ujson.load(f)
        model = Path(predictions_settings['model'])

        if not model.exists():
            filename = str(model).split('\\')[-1]
            model = Path(f"../pretrained_models/{filename}")

        logger.info(model)

    logger.info('Beginning evaluations')
    for file in sorted(datadir.glob('MLAO*.tif'), key=os.path.getctime):  # sort by creation time
        if 'CamB' in str(file) or 'pupil' in str(file) or 'autoexpos' in str(file):
            continue

        if file.stem.split('_')[1].startswith('round'):
            iter_num = int(file.stem.split('_')[1][-1])
        else:
            iter_num = file.stem.split('_')[3]

        noao_path = None
        for ifile in sorted(datadir.glob('NoAO*.tif')):
            if fnmatch.fnmatch(ifile.name, f'NoAO*.tif'):
                noao_path = ifile
                break

        if noao_path is None:
            logger.warning(f'NoAO not found for: {file.name}')
            continue

        sh_path = None
        for gtfile in sorted(datadir.glob('SHAO*.tif')):
            if fnmatch.fnmatch(gtfile.name, f'SHAO_Scan_Iter_{str(iter_num-1).zfill(4)}*.tif'):
                sh_path = gtfile
                break
            elif fnmatch.fnmatch(gtfile.name, f'SHAO_Scan_Iter_{iter_num}*.tif'):
                sh_path = gtfile
                break
            elif fnmatch.fnmatch(gtfile.name, f'SHAO_Scan*.tif'):
                sh_path = gtfile
                break

        if sh_path is None:
            logger.warning(f'SH not found for: {file.name}')
            continue

        prediction_path = None
        for predfile in sorted(datadir.glob('MLAO*.tif')):
            if fnmatch.fnmatch(predfile.name, f'MLAO_Scan_Iter_{iter_num}*.tif'):
                prediction_path = predfile
                break
            elif fnmatch.fnmatch(predfile.name, f'MLAO_round{iter_num}_Scan*.tif'):
                prediction_path = predfile
                break

        if prediction_path is None:
            logger.warning(f'Prediction not found for: {file.name}')
            continue

        noao_img = backend.load_sample(noao_path)
        # noao_img = preprocessing.prep_sample(
        #     noao_img,
        #     normalize=True,
        #     remove_background=True,
        #     windowing=False,
        #     sample_voxel_size=predictions_settings['sample_voxel_size']
        # )

        ml_img = backend.load_sample(prediction_path)
        # ml_img = preprocessing.prep_sample(
        #     ml_img,
        #     normalize=True,
        #     remove_background=True,
        #     windowing=False,
        #     sample_voxel_size=predictions_settings['sample_voxel_size']
        # )

        gt_img = backend.load_sample(sh_path)
        # gt_img = preprocessing.prep_sample(
        #     gt_img,
        #     normalize=True,
        #     remove_background=True,
        #     windowing=False,
        #     sample_voxel_size=predictions_settings['sample_voxel_size']
        # )

        vis.compare_mips(
            results=dict(
                noao_img=noao_img,
                ml_img=ml_img,
                gt_img=gt_img,
                residuals=f'{prediction_path.parent}/{prediction_path.stem}_ml_eval_residuals.csv',
            ),
            save_path=datadir / f'mips_evaluation_iter_{iter_num}',
            dxy=predictions_settings['sample_voxel_size'][1],
            dz=predictions_settings['sample_voxel_size'][0],
            transform_to_align_to_DM=True
        )


def eval_bleaching_rate(datadir: Path):
    results = {}

    def get_stats(path: Path, key: int, background: int = 100):
        img = backend.load_sample(path)
        img -= background  # remove background offset

        quality = preprocessing.prep_sample(
            img,
            return_psnr=True,
            remove_background=True,
            plot=None,
            normalize=False,
            remove_background_noise_method='dog'
        )

        imin = np.nanmin(img)
        imax = np.nanmax(img)
        isum = np.nansum(img)
        imean = np.nanmean(img)
        istd = np.nanstd(img)

        ip1 = np.nanpercentile(img, 1)
        ip5 = np.nanpercentile(img, 5)
        ip15 = np.nanpercentile(img, 15)
        ip25 = np.nanpercentile(img, 25)
        ip50 = np.nanpercentile(img, 50)
        ip75 = np.nanpercentile(img, 75)
        ip85 = np.nanpercentile(img, 85)
        ip95 = np.nanpercentile(img, 95)
        ip99 = np.nanpercentile(img, 99)

        psnr = (imax - ip50) / np.sqrt((imax - ip50))

        results[key] = dict(
            ipath=str(path),
            iquality=quality,
            ipsnr=psnr,
            isum=isum,
            imean=imean,
            istd=istd,
            imin=imin,
            ip1=ip1,
            ip5=ip5,
            ip15=ip15,
            ip25=ip25,
            ip50=ip50,
            ip75=ip75,
            ip85=ip85,
            ip95=ip95,
            ip99=ip99,
            imax=imax,
        )

    pool = mp.Pool(processes=mp.cpu_count())

    logger.info('Beginning evaluations')
    for file in sorted(datadir.glob('Imaging_Scan*.tif'), key=os.path.getctime):  # sort by creation time
        if 'CamB' in str(file) or 'pupil' in str(file) or 'autoexpos' in str(file):
            continue

        iter_num = int(file.stem.split('_')[3])
        pool.apply_async(get_stats(file, key=iter_num))

    pool.close()  # close the pool
    pool.join()  # wait for all tasks to complete

    df = pd.DataFrame.from_dict(results, orient='index')
    df.index.name = 'iter_num'
    df.to_csv(datadir/'bleaching_rate.csv')


def plot_bleaching_rate(datadir: Path):
    results = []

    ml_counter, sh_counter = 0, 0
    logger.info('Beginning evaluations')
    for file in sorted(datadir.rglob('*bleaching_rate.csv'), key=os.path.getctime):  # sort by creation time
        df = pd.read_csv(file, header=0)
        df['method'] = 'ML' if 'ML' in str(file) else 'SH'
        df['signal'] = df['isum'] / df['isum'].iloc[0]
        df['max_signal'] = df['ip99'] / df['ip99'].iloc[0]

        if 'ML' in str(file):
            df['method'] = 'OpticalNet'
            ml_counter += 1
            df['exp'] = ml_counter
        else:
            df['method'] = 'Shack–Hartmann'
            sh_counter += 1
            df['exp'] = sh_counter

        results.append(df)

    results = pd.concat(results, ignore_index=True)
    results.to_csv(datadir/'bleaching_rate_evaluations.csv')
    results['iter_num'] += 1

    df = results.groupby(['iter_num', 'method'])['signal']
    means = df.mean()
    diff = means.loc[:, 'OpticalNet'] - means.loc[:, 'Shack–Hartmann']


    fig = plt.figure(figsize=(5, 8))
    sns.set_theme(style="ticks")
    g = sns.relplot(
        data=results,
        x="iter_num",
        y="signal",
        hue="method",
        kind="line",
        height=5,
        aspect=1.,
    )

    g.map(plt.axhline, y=means.loc[39, 'Shack–Hartmann'], color='C1', dashes=(2, 1), zorder=3)
    g.map(plt.axhline, y=means.loc[39, 'OpticalNet'], color='C0', dashes=(2, 1), zorder=3)

    (
        g.map(plt.grid, which="both", axis='both', lw=.25, ls='--', zorder=0, color='lightgrey')
        .set_axis_labels("Iteration", "Signal")
        .set(xlim=(1, max(results['iter_num'])), ylim=(.7, 1))
        .tight_layout(w_pad=0)
    )

    leg = g._legend
    leg.set_title('Method')
    leg.set_bbox_to_anchor((.9, .8))
    vis.savesvg(fig, f'{datadir}/bleaching_rate_signal.svg')

    fig = plt.figure(figsize=(5, 8))
    sns.set_theme(style="ticks")
    g = sns.relplot(
        data=results,
        x="iter_num",
        y="signal",
        hue="exp",
        kind="line",
        height=5,
        aspect=1.,
        col="method",
        col_wrap=2,
        palette='tab10',
        facet_kws=dict(sharex=True),
    )
    g.map(plt.axhline, y=means.loc[39, 'Shack–Hartmann'], color='C1', dashes=(2, 1), zorder=3)
    g.map(plt.axhline, y=means.loc[39, 'OpticalNet'], color='C0', dashes=(2, 1), zorder=3)

    (
        g.map(plt.grid, which="both", axis='both', lw=.25, ls='--', zorder=0, color='lightgrey')
        .set_axis_labels("Iteration", "Signal")
        .set(xlim=(1, max(results['iter_num'])), ylim=(.7, 1))
        .tight_layout(w_pad=0)
    )

    leg = g._legend
    leg.set_title('Experiment')
    leg.set_bbox_to_anchor((.9, .8))
    vis.savesvg(fig, f'{datadir}/bleaching_rate_evaluations.svg')

    fig = plt.figure(figsize=(5, 8))
    sns.set_theme(style="ticks")
    g = sns.relplot(
        data=results,
        x="iter_num",
        y="isum",
        hue="exp",
        kind="line",
        height=5,
        aspect=1.,
        col="method",
        col_wrap=2,
        palette='tab10',
        facet_kws=dict(sharex=True),
    )

    (
        g.map(plt.grid, which="both", axis='both', lw=.25, ls='--', zorder=0, color='lightgrey')
        .set_axis_labels("Iteration", "Integration")
        .set(xlim=(1, max(results['iter_num'])),)
        .tight_layout(w_pad=0)
    )

    leg = g._legend
    leg.set_title('Experiment')
    leg.set_bbox_to_anchor((.9, .8))
    vis.savesvg(fig, f'{datadir}/bleaching_rate_integration.svg')

    fig = plt.figure(figsize=(5, 8))
    sns.set_theme(style="ticks")
    g = sns.relplot(
        data=results,
        x="iter_num",
        y="ip99",
        hue="exp",
        kind="line",
        height=5,
        aspect=1.,
        col="method",
        col_wrap=2,
        palette='tab10',
        facet_kws=dict(sharex=True),
    )

    (
        g.map(plt.grid, which="both", axis='both', lw=.25, ls='--', zorder=0, color='lightgrey')
        .set_axis_labels("Iteration", r"Max signal ($99^{th}$ percentile)")
        .set(xlim=(1, max(results['iter_num'])))
        .tight_layout(w_pad=0)
    )

    leg = g._legend
    leg.set_title('Experiment')
    leg.set_bbox_to_anchor((.9, .8))
    vis.savesvg(fig, f'{datadir}/bleaching_rate_max.svg')



def eval_cell_dataset(
    datadir: Path,
    flat: Any = None,
    postfix: str = 'predictions_aggregated_zernike_coefficients.csv',
    gt_postfix: str = 'phase_retrieval_zernike_coefficients.csv',
    precomputed: bool = False,
):
    results = {}
    savepath = Path(f'{datadir}/cells_evaluation')

    if precomputed or not Path(f'{savepath}_results.npy').exists():

        # get model from .json file
        with open(sorted(Path(datadir / 'rotated').glob('*_predictions_settings.json'))[0]) as f:
            predictions_settings = ujson.load(f)
            model = Path(predictions_settings['model'])

            if not model.exists():
                filename = str(model).split('\\')[-1]
                model = Path(f"../pretrained_models/{filename}")

            mp.set_start_method('spawn', force=True)
            pool = mp.Pool(processes=mp.cpu_count())
            resultsdir = Path(datadir / 'rotated')
            results_list = sorted(resultsdir.glob('**/*'))    # only get directory list once for speed
            wf_list = sorted(datadir.glob('*widefield*'))
            

            logger.info('Beginning evaluations')
            for cam_a_file in sorted(datadir.glob('*Scan_Iter*CamA*.tif'), key=os.path.getctime):  # sort by creation time
                if 'background' in str(cam_a_file) or 'pupil' in str(cam_a_file) or 'autoexpos' in str(cam_a_file):
                    continue

                iter_number = cam_a_file.name[10:14]

                gt_path = None
                for gtfile in wf_list:
                    if fnmatch.fnmatch(gtfile.name, f'*{iter_number}*CamA*{iter_number}t.tif'):
                        gt_path = gtfile
                        continue
                if gt_path is None: logger.warning(f'GT not found for: {cam_a_file.name}')

                prediction_path = None
                for predfile in results_list:
                    if fnmatch.fnmatch(predfile.name, f'*{iter_number}*CamA*_{postfix}'):
                        prediction_path = predfile
                        continue
                if prediction_path is None: logger.warning(f'Prediction not found for: {cam_a_file.name}')

                cam_a_ml_img = backend.load_sample(cam_a_file)
                cam_a_ml_img = preprocessing.prep_sample(
                    cam_a_ml_img,
                    normalize=True,
                    remove_background=True,
                    windowing=False,
                    sample_voxel_size=predictions_settings['sample_voxel_size'],
                    remove_background_noise_method='dog',
                    min_psnr=0
                )
                
                cam_b_ml_img = backend.load_sample(str(cam_a_file).replace('CamA', 'CamB'))
                cam_b_ml_img = preprocessing.prep_sample(
                    cam_b_ml_img,
                    normalize=True,
                    remove_background=True,
                    windowing=False,
                    sample_voxel_size=predictions_settings['sample_voxel_size'],
                    remove_background_noise_method='dog',
                    min_psnr=0
                )
                
                ml_img = np.stack([cam_b_ml_img, cam_a_ml_img, np.zeros_like(cam_b_ml_img)], axis=-1)
                
                pr_img = backend.load_sample(gt_path)
                pr_img = preprocessing.prep_sample(
                    pr_img,
                    normalize=True,
                    remove_background=True,
                    windowing=False,
                    sample_voxel_size=predictions_settings['sample_voxel_size'],
                    remove_background_noise_method='dog',
                    min_psnr=0
                )

                if prediction_path is not None:
                    p = pd.read_csv(prediction_path)
                    ml_wavefront = Wavefront(
                        p['z0_c0'].values,
                        modes=p.shape[0],
                        lam_detection=predictions_settings['wavelength']
                    )
                else:
                    ml_wavefront = None

                if gt_path is not None:
                    pr_prediction_path = Path(f"{gt_path.with_suffix('')}_{gt_postfix}")
                    
                    if pr_prediction_path.exists():
                        y = pd.read_csv(pr_prediction_path)
                    else:
                        y = phase_retrieval(
                            img=gt_path,
                            num_modes=15,
                            dm_calibration=None,
                            dm_state=None,
                            lateral_voxel_size=predictions_settings['sample_voxel_size'][-1],
                            axial_voxel_size=.1,
                        )
                        
                    gt_wavefront = Wavefront(
                        y.amplitude.values[:p.shape[0]],
                        modes=p.shape[0],
                        lam_detection=predictions_settings['wavelength']
                    )
                else:
                    gt_wavefront = None

                if gt_wavefront is not None and ml_wavefront is not None:
                    
                    diff_wavefront = Wavefront(
                        gt_wavefront - ml_wavefront,
                        modes=p.shape[0],
                        lam_detection=predictions_settings['wavelength']
                    )
                    
                    results[(iter_number, cam_a_file.parent.name)] = dict(
                        ml_img=ml_img,
                        ml_wavefront=ml_wavefront,
                        gt_img=pr_img,
                        gt_wavefront=gt_wavefront,
                        diff_wavefront=diff_wavefront,
                        residuals=f'{prediction_path.parent}/{prediction_path.stem}_ml_eval_residuals.csv',
                    )

            if not precomputed:
                children = mp.active_children()
                logger.info(f"Awaiting {len(children)} 'eval_mode' tasks to finish.")
                
            pool.close()    # close the pool
            pool.join()     # wait for all tasks to complete

        np.save(f'{savepath}_results.npy', results)

    else:
        # skip calc. Reload results and just replot
        results = np.load(f'{savepath}_results.npy', allow_pickle='TRUE').item()

    logger.info(f'{savepath}_results.npy')
    vis.plot_cell_dataset(
        results,
        savepath=savepath,
        list_of_files=[datadir.name],
        transform_to_align_to_DM=False
    )
    