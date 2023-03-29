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
from line_profiler_pycharm import profile
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import utils
import vis
import backend
import preprocessing
from wavefront import Wavefront

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

    noisy_img = utils.load_sample(input_path)
    maxcounts = np.max(noisy_img)
    psnr = predictions_settings['psnr']
    gen = backend.load_metadata(
        model_path,
        snr=psnr,
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

    if plot:
        noisy_img = preprocessing.prep_sample(
            noisy_img,
            normalize=normalize,
            remove_background=remove_background,
            windowing=False,
            sample_voxel_size=predictions_settings['sample_voxel_size']
        )

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
    precomputed: bool = False,
    compare_iterations: bool = False,
):
    results = {}

    # get model from .json file
    with open(list(Path(datadir / 'MLResults').glob('*_settings.json'))[0]) as f:
        predictions_settings = ujson.load(f)
        model = Path(predictions_settings['model'])

        if not model.exists():
            filename = str(model).split('\\')[-1]
            model = Path(f"../pretrained_models/lattice_yumb_x108um_y108um_z200um/{filename}")

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

            task = partial(
                evaluate,
                input_path=file,
                prediction_path=prediction_path,
                gt_path=gt_path
            )
            _ = pool.apply_async(task)  # issue task

            if compare_iterations:
                logger.info(file.stem)
                ml_img = preprocessing.prep_sample(
                    utils.load_sample(file),
                    normalize=True,
                    remove_background=True,
                    windowing=False,
                    sample_voxel_size=predictions_settings['sample_voxel_size']
                )

                logger.info(pr_path.stem)
                pr_img = preprocessing.prep_sample(
                    utils.load_sample(pr_path),
                    normalize=True,
                    remove_background=True,
                    windowing=False,
                    sample_voxel_size=predictions_settings['sample_voxel_size']
                )

                results[state] = dict(
                    ml_img=ml_img,
                    gt_img=pr_img,
                    residuals=f'{prediction_path.parent}/{prediction_path.stem}_ml_eval_residuals.csv',
                )

        pool.close()    # close the pool
        pool.join()     # wait for all tasks to complete

    plot_eval_dataset(model, datadir)

    if compare_iterations:
        vis.compare_iterations(
            results=results,
            save_path=datadir/'iterative_evaluation',
            dxy=predictions_settings['sample_voxel_size'][1],
            dz=predictions_settings['sample_voxel_size'][0],
            transform_to_align_to_DM=True
        )


@profile
def plot_dataset_mips(datadir: Path):
    mldir = Path(datadir/'MLResults')

    # get model from .json file
    with open(list(mldir.glob('*_settings.json'))[0]) as f:
        predictions_settings = ujson.load(f)
        model = Path(predictions_settings['model'])

        if not model.exists():
            filename = str(model).split('\\')[-1]
            model = Path(f"../pretrained_models/lattice_yumb_x108um_y108um_z200um/{filename}")

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

        noao_img = preprocessing.prep_sample(
            utils.load_sample(noao_path),
            normalize=True,
            remove_background=True,
            windowing=False,
            sample_voxel_size=predictions_settings['sample_voxel_size']
        )

        ml_img = preprocessing.prep_sample(
            utils.load_sample(prediction_path),
            normalize=True,
            remove_background=True,
            windowing=False,
            sample_voxel_size=predictions_settings['sample_voxel_size']
        )

        gt_img = preprocessing.prep_sample(
            utils.load_sample(sh_path),
            normalize=True,
            remove_background=True,
            windowing=False,
            sample_voxel_size=predictions_settings['sample_voxel_size']
        )

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


@profile
def eval_ao_dataset(
    datadir: Path,
    flat: Any = None,
    postfix: str = 'predictions_zernike_coefficients.csv',
    gt_postfix: str = 'DSH1_DSH_Wvfrt*',
    gt_unit: str = 'nm',
    plot_evals: bool = True,
    precomputed: bool = False,
    compare_iterations: bool = True,
):
    mldir = Path(datadir/'MLResults')
    ml_results = list(mldir.glob('**/*'))
    sh_results = list(Path(datadir/'DSH1/DSH_Wavefront_TIF').glob('**/*'))
    results = {}

    # get model from .json file
    with open(list(mldir.glob('*_settings.json'))[0]) as f:
        predictions_settings = ujson.load(f)
        model = Path(predictions_settings['model'])

        if not model.exists():
            filename = str(model).split('\\')[-1]
            model = Path(f"../pretrained_models/lattice_yumb_x108um_y108um_z200um/{filename}")

        logger.info(model)

    if not precomputed:
        pool = mp.Pool(processes=mp.cpu_count())

        evaluate = partial(
            eval_mode,
            model_path=model,
            flat_path=flat,
            postfix=postfix,
            gt_postfix=gt_postfix,
            plot=plot_evals,
        )

        logger.info('Beginning evaluations')
        for file in sorted(datadir.glob('MLAO_Scan*.tif'), key=os.path.getctime):  # sort by creation time
            if 'CamB' in str(file) or 'pupil' in str(file) or 'autoexpos' in str(file):
                continue

            method = file.stem.split('_')[0]
            iter_num = file.stem.split('_')[3]

            gt_path = None
            for gtfile in sh_results:
                if fnmatch.fnmatch(gtfile.name, f'{gt_postfix}*{iter_num}*.tif'):
                    gt_path = gtfile
                    break

            if gt_path is None:
                logger.warning(f'GT not found for: {file.name}.tif')

            sh_path = None
            for gtfile in sorted(datadir.glob('SHAO_Scan*.tif')):
                if fnmatch.fnmatch(gtfile.name, f'SHAO_Scan_Iter_{iter_num}*.tif'):
                    sh_path = gtfile
                    break

            if gt_path is None:
                logger.warning(f'GT not found for: {file.name}.tif')

            prediction_path = None
            for predfile in ml_results:
                if fnmatch.fnmatch(predfile.name, f'{method}_Scan_Iter_{iter_num}*{postfix}'):
                    prediction_path = predfile
                    break

            if prediction_path is None:
                logger.warning(f'Prediction not found for: {file.name}')

            task = partial(
                evaluate,
                input_path=file,
                prediction_path=prediction_path,
                gt_path=gt_path,
                gt_unit=gt_unit
            )
            _ = pool.apply_async(task)  # issue task

            if compare_iterations:
                ml_img = preprocessing.prep_sample(
                    utils.load_sample(file),
                    normalize=True,
                    remove_background=True,
                    windowing=False,
                    sample_voxel_size=predictions_settings['sample_voxel_size']
                )

                gt_img = preprocessing.prep_sample(
                    utils.load_sample(sh_path),
                    normalize=True,
                    remove_background=True,
                    windowing=False,
                    sample_voxel_size=predictions_settings['sample_voxel_size']
                )

                results[iter_num] = dict(
                    ml_img=ml_img,
                    gt_img=gt_img,
                    residuals=f'{prediction_path.parent}/{prediction_path.stem}_ml_eval_residuals.csv',
                )

        pool.close()    # close the pool
        pool.join()     # wait for all tasks to complete

    if compare_iterations:
        vis.compare_iterations(
            results=results,
            save_path=datadir/'iterative_evaluation',
            dxy=predictions_settings['sample_voxel_size'][1],
            dz=predictions_settings['sample_voxel_size'][0],
            transform_to_align_to_DM=True
        )


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
