import logging
import sys
from functools import partial
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from tqdm import tqdm
from tifffile import imread
import tensorflow as tf
from scipy import signal
from skimage import transform, filters
from tifffile import imsave

import utils
import vis
import data_utils
import backend

from synthetic import SyntheticPSF
from wavefront import Wavefront
from zernike import Zernike

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
tf.get_logger().setLevel(logging.ERROR)


def iterative_eval(
        model: tf.keras.Model,
        inputs: np.array,
        outputs: np.array,
        batch_size: int,
        psfargs: dict,
        iterations: int = 15,
        desc: str = '',
        tolerance: float = 1e-2,
        patience: int = 5,
):
    predictions = {}
    corrections = {}
    gen = SyntheticPSF(**psfargs)

    errors = pd.DataFrame.from_dict({
        'niter': np.zeros(outputs.shape[0], dtype=int),
        'residuals': utils.peak2peak(outputs)
    })
    means = None

    i = 1
    converged = False
    while not converged:
        corrections[i] = np.squeeze(inputs[0], axis=-1)
        predictions[i], stdev = backend.bootstrap_predict(
            model,
            inputs,
            psfgen=gen,
            batch_size=batch_size,
            n_samples=1,
            desc=desc
        )

        y_pred = pd.DataFrame([utils.peak_aberration(k) for k in predictions[i]], columns=['residuals'])
        y_true = pd.DataFrame([utils.peak_aberration(k) for k in outputs], columns=['residuals'])

        err = np.abs(y_true - y_pred)
        err['niter'] = i
        errors = pd.concat([errors, err], ignore_index=True)
        means = errors.groupby(['niter']).mean().reset_index()

        # check if converged
        if (i >= 1) and (
            i >= iterations or means['residuals'].iloc[-1] + tolerance > means['residuals'].iloc[-2] or np.allclose(
                means['residuals'].tail(patience).values, means['residuals'].iloc[-1], rtol=tolerance, atol=tolerance
        )):
            converged = True

        # setup next iter
        res = outputs - predictions[i]
        g = partial(
            gen.single_psf,
            zplanes=0,
            normed=True,
            noise=True,
            augmentation=True,
            meta=False
        )
        inputs = np.expand_dims(np.stack(gen.batch(g, res), axis=0), -1)
        outputs = res
        i += 1

    # psf_cmap = 'Spectral_r'
    # fig, axes = plt.subplots(len(predictions), 4, figsize=(8, 11))
    # for i, vol in corrections.items():
    #     axes[i - 1, 0].bar(range(len(predictions[i][0])), height=predictions[i][0], color='dimgrey')
    #     m = axes[i - 1, 1].imshow(np.max(vol, axis=0) if vol.shape[0] > 3 else vol[0], cmap=psf_cmap)
    #     axes[i - 1, 2].imshow(np.max(vol, axis=1) if vol.shape[0] > 3 else vol[1], cmap=psf_cmap)
    #     axes[i - 1, 3].imshow(np.max(vol, axis=2).T if vol.shape[0] > 3 else vol[2], cmap=psf_cmap)
    #     cax = inset_axes(axes[i - 1, 3], width="10%", height="100%", loc='center right', borderpad=-2)
    #     cb = plt.colorbar(m, cax=cax)
    #     cax.yaxis.set_label_position("right")
    # plt.show()

    return means


def evaluate_psnrs(
        model: Path,
        wavelength: float,
        x_voxel_size: float,
        y_voxel_size: float,
        z_voxel_size: float,
        n_samples: int,
        batch_size: int,
        cpu_workers: int,
        plot=True,
        metric='peak2peak',
):
    n_batches = n_samples // batch_size
    save_path = model / "eval"
    save_path.mkdir(exist_ok=True, parents=True)
    model = backend.load(model)
    model.summary()

    error = {}
    peak2peak = []
    scales = sorted(set([int(t) for t in np.logspace(1, 3, num=20)]))
    logger.info(f"PSNRs: {scales}")

    for snr in scales:
        psfargs = dict(
            lam_detection=wavelength,
            amplitude_ranges=(-.2, .2),
            psf_shape=model.layers[0].input_shape[0][1:-1],
            x_voxel_size=x_voxel_size,
            y_voxel_size=y_voxel_size,
            z_voxel_size=z_voxel_size,
            batch_size=batch_size,
            snr=snr,
            max_jitter=1,
            cpu_workers=cpu_workers,
        )

        z = Zernike(0)
        cols = [f"(n={z.ansi_to_nm(j)[0]}, m={z.ansi_to_nm(j)[1]})" for j in range(model.layers[-1].output_shape[-1])]

        y_pred = pd.DataFrame([], columns=cols)
        y_true = pd.DataFrame([], columns=cols)

        target_psnr = None
        gen = SyntheticPSF(**psfargs)
        for i, (psfs, ys, psnrs, zplanes, maxcounts) in zip(range(n_batches), gen.generator(debug=True)):
            psnr_pct = np.ceil(np.nanquantile(psnrs, .75))

            if target_psnr is None:
                target_psnr = int(psnr_pct)
            else:
                target_psnr = int(np.mean([psnr_pct, target_psnr]))

            preds, stdev = backend.bootstrap_predict(
                model,
                psfs,
                psfgen=gen,
                batch_size=batch_size,
                desc=f"Predictions for PSNR({int(target_psnr)})"
            )

            if plot:
                dir = save_path / f"psnr_{target_psnr}"
                dir.mkdir(exist_ok=True, parents=True)
                paths = [f"{dir}/{(i * batch_size) + n}" for n in range(batch_size)]
                utils.multiprocess(
                    partial(utils.eval, psfargs=psfargs),
                    list(zip(psfs, ys, preds, paths, psnrs, zplanes, maxcounts)),
                    desc=f"Plotting PSNR({int(target_psnr)})"
                )

            y_pred = y_pred.append(pd.DataFrame(utils.peak2peak(preds), columns=['sample']), ignore_index=True)
            y_true = y_true.append(pd.DataFrame(utils.peak2peak(ys), columns=['sample']), ignore_index=True)
            peak2peak.extend(list(y_true['sample'].values))

        df = np.abs(y_true - y_pred)
        df = pd.DataFrame(df, columns=['sample'])
        error[target_psnr] = df['sample']
        df.to_csv(f"{save_path}/psnr_{target_psnr}.csv")

    error = pd.DataFrame.from_dict(error)
    error = error.reindex(sorted(error.columns), axis=1)
    logger.info(error)
    vis.plot_residuals(
        error,
        wavelength=wavelength,
        nsamples=n_samples,
        save_path=f"{save_path}/psnr_{metric}",
        label=r'Peak signal-to-noise ratio'
    )

    plt.figure(figsize=(6, 6))
    plt.hist(peak2peak, bins=100)
    plt.grid()
    plt.xlabel(
        'Peak-to-peak aberration $|P_{95} - P_{5}|$'
        rf'($\lambda = {int(wavelength*1000)}~nm$)'
    )
    plt.ylabel(rf'Number of samples')
    plt.savefig(f'{save_path}/dist_peak2peak.png', dpi=300, bbox_inches='tight', pad_inches=.25)


def evaluate(
        model: Path,
        target: str,
        wavelength: float,
        x_voxel_size: float,
        y_voxel_size: float,
        z_voxel_size: float,
        batch_size: int,
        cpu_workers: int,
        n_samples: int = 100,
        dist: str = 'mixed',
        metric='peak2peak',
        plot=False,
        dominant_modes=None
):
    means, stds = {}, {}

    if plot:
        save_path = model / "eval"
        save_path.mkdir(exist_ok=True, parents=True)

        for p in sorted(save_path.glob(f'{target}*.csv')):
            df = pd.read_csv(p, header=0, index_col=0)
            bits = p.stem.split('_')
            try:
                var, snr = round(float(bits[2]), 3), int(bits[4])
            except ValueError:
                var, snr = round(float(bits[3]), 3), int(bits[5])

            logger.info(f"PSNR: {snr}, {target}: {var}")

            if means.get(snr) is None:
                means[snr] = {var: df[metric].mean()}
                stds[snr] = {var: df[metric].std()}
            else:
                means[snr].update({var: df[metric].mean()})
                stds[snr].update({var: df[metric].std()})

    else:
        n_batches = n_samples // batch_size
        save_path = model / "eval"
        save_path.mkdir(exist_ok=True, parents=True)
        model = backend.load(model)
        model.summary()
        psnrs = sorted(set([int(t) for t in np.linspace(1, 100, num=10).round(0)]))

        pconfigs = dict(
            amplitude_ranges=[
                dict(
                    snr=p,
                    amplitude_ranges=a,
                    distribution=dist,
                    lam_detection=wavelength,
                    batch_size=batch_size,
                    x_voxel_size=x_voxel_size,
                    y_voxel_size=y_voxel_size,
                    z_voxel_size=z_voxel_size,
                    max_jitter=0,
                    cpu_workers=cpu_workers,
                    n_modes=model.output_shape[-1],
                    psf_shape=tuple(3 * [model.input_shape[-2]]),
                )
                for p in psnrs for a in np.linspace(0.01, .3, num=7).round(3)
            ],
            max_jitter=[
                dict(
                    snr=p,
                    max_jitter=j,
                    distribution=dist,
                    lam_detection=wavelength,
                    batch_size=batch_size,
                    x_voxel_size=x_voxel_size,
                    y_voxel_size=y_voxel_size,
                    z_voxel_size=z_voxel_size,
                    amplitude_ranges=(-.3, .3),
                    cpu_workers=cpu_workers,
                    n_modes=model.output_shape[-1],
                    psf_shape=tuple(3 * [model.input_shape[-2]]),
                )
                for p in psnrs for j in np.linspace(0, 2, num=7).round(2)
            ],
            z_voxel_size=[
                dict(
                    snr=p,
                    distribution=dist,
                    lam_detection=wavelength,
                    batch_size=batch_size,
                    x_voxel_size=x_voxel_size,
                    y_voxel_size=y_voxel_size,
                    z_voxel_size=s,
                    amplitude_ranges=(-.3, .3),
                    max_jitter=1,
                    cpu_workers=cpu_workers,
                    n_modes=model.output_shape[-1],
                    psf_shape=tuple(3 * [model.input_shape[-2]]),
                )
                for p in psnrs for s in np.linspace(.1, 1, num=7).round(2)
            ],
        )

        for psfargs in pconfigs[target]:
            logger.info(psfargs)

            y_pred = pd.DataFrame([], columns=['sample'])
            y_true = pd.DataFrame([], columns=['sample'])

            gen = SyntheticPSF(**psfargs)
            for i, (inputs, ys, snrs, zplanes, maxcounts) in zip(range(n_batches), gen.generator(debug=True, otf=False)):

                if dominant_modes is not None:
                    # normalized contribution
                    df = pd.DataFrame(np.abs(ys))
                    df = df.div(df.sum(axis=1), axis=0)

                    # dominant modes
                    dmodes = (df[df > .05]).count(axis=1)
                    dmodes = dmodes >= dominant_modes
                    toi = df[dmodes].index

                    inputs = inputs[toi]
                    ys = ys[toi]

                if model.name.lower() != 'phasenet':
                    inputs = np.stack(utils.multiprocess(
                        func=partial(gen.embedding, ishape=model.input_shape[-2]),
                        jobs=inputs,
                        desc='Preprocessing',
                        cores=cpu_workers
                    ),
                        axis=0
                    )

                    # vol = np.squeeze(inputs[0], axis=-1)
                    # fig, axes = plt.subplots(1, 3, figsize=(6, 6))
                    # for i in range(3):
                    #     m = axes[i].imshow(vol[i], cmap='Spectral_r', vmin=0, vmax=1)
                    # plt.show()

                preds, stdev = backend.bootstrap_predict(
                    model,
                    inputs,
                    psfgen=gen,
                    batch_size=batch_size,
                    n_samples=1,
                    desc=f"Predictions for ({(psfargs[target], psfargs['snr'])})"
                )

                y_pred = y_pred.append(pd.DataFrame(utils.peak2peak(preds), columns=['sample']), ignore_index=True)
                y_true = y_true.append(pd.DataFrame(utils.peak2peak(ys), columns=['sample']), ignore_index=True)

            error = np.abs(y_true - y_pred)
            error = pd.DataFrame(error, columns=['sample'])
            error.to_csv(f"{save_path}/{target}_{psfargs[target]}_snr_{psfargs['snr']}.csv")

            if target == 'amplitude_ranges':
                bins = np.arange(0, y_true['sample'].max() + .25, .25)
                df = pd.DataFrame(zip(y_true['sample'], error['sample']), columns=['aberration', 'error'])
                df['bins'] = pd.cut(df['aberration'], bins, labels=bins[1:], include_lowest=True)

                if means.get(psfargs['snr']) is None:
                    means[psfargs['snr']] = df
                else:
                    means[psfargs['snr']] = means[psfargs['snr']].append(df, ignore_index=True)

            else:
                if means.get(psfargs['snr']) is None:
                    means[psfargs['snr']] = {psfargs[target]: error['sample'].mean()}
                else:
                    means[psfargs['snr']].update({psfargs[target]: error['sample'].mean()})

    if target == 'amplitude_ranges':
        for k, df in means.items():
            means[k] = df.groupby('bins').mean()
            means[k] = means[k]['error'].to_dict()

    means = pd.DataFrame.from_dict(means)
    means = means.reindex(sorted(means.columns), axis=1)
    means = means.sort_index().interpolate()

    logger.info(means)
    vis.plot_eval(
        means,
        wavelength=wavelength,
        nsamples=n_samples,
        save_path=f"{save_path}/{target}_{dist}_{metric}_dmodes{dominant_modes}",
        label=target
    )


def compare_models(
        modelsdir: Path,
        wavelength: float,
        x_voxel_size: float,
        y_voxel_size: float,
        z_voxel_size: float,
        psf_shape: tuple,
        n_samples: int,
        batch_size: int,
        cpu_workers: int,
        metric='peak2peak',
        iterations: int = 10,
):
    n_batches = n_samples // batch_size
    models = [p for p in modelsdir.iterdir() if p.is_dir()]
    errors = [{m.stem: {} for m in models} for _ in range(iterations)]
    peak2peak = []
    scales = sorted(set([int(t) for t in np.logspace(1, 2, num=3)]))
    logger.info(f"Models: {models}")
    logger.info(f"PSNRs: {scales}")

    modes = backend.load(models[0]).layers[-1].output_shape[-1]
    eval_distribution = 'mixed'

    for snr in scales:
        psfargs = dict(
            n_modes=modes,
            distribution=eval_distribution,
            lam_detection=wavelength,
            amplitude_ranges=(-.3, .3),
            psf_shape=psf_shape,
            x_voxel_size=x_voxel_size,
            y_voxel_size=y_voxel_size,
            z_voxel_size=z_voxel_size,
            batch_size=batch_size,
            snr=snr,
            max_jitter=1,
            cpu_workers=cpu_workers,
        )

        target_psnr = None
        for _, (psfs, ys, psnrs, zplanes, maxcounts) in zip(range(n_batches),
                                                            SyntheticPSF(**psfargs).generator(debug=True)):
            psnr_pct = np.ceil(np.nanquantile(psnrs, .75))

            if target_psnr is None:
                target_psnr = int(psnr_pct)
            else:
                target_psnr = int(np.mean([psnr_pct, target_psnr]))

            for m in models:
                model = backend.load(m)

                preds = iterative_eval(
                    model,
                    inputs=psfs,
                    outputs=ys,
                    batch_size=batch_size,
                    psfargs=psfargs,
                    desc=f"{m.stem}, PSNR({int(target_psnr)})",
                    iterations=iterations,
                )

                for i in range(iterations):
                    ps = preds[0] if i == 0 else np.sum([preds[k] for k in range(i + 1)], axis=0)
                    y_pred = pd.DataFrame(utils.peak2peak(ps), columns=['sample'])
                    y_true = pd.DataFrame(utils.peak2peak(ys), columns=['sample'])

                    # drop aberrations below diffraction limit
                    idx = y_true.index[y_true['sample'] >= .5]
                    y_true = y_true.loc[idx]
                    y_pred = y_pred.loc[idx]

                    peak2peak.extend(list(y_true['sample'].values))

                    if errors[i][m.stem].get(target_psnr) is not None:
                        errors[i][m.stem][target_psnr] = np.mean([
                            errors[i][m.stem][target_psnr], np.nanmean(np.abs(y_true - y_pred))
                        ])
                    else:
                        errors[i][m.stem][target_psnr] = np.nanmean(np.abs(y_true - y_pred))

    for i in range(iterations):
        error = pd.DataFrame(errors[i])
        error = error.reindex(sorted(error.columns), axis=1)
        logger.info(error)
        vis.plot_models(
            error,
            wavelength=wavelength,
            nsamples=n_samples,
            save_path=f"{modelsdir}/{eval_distribution}_psnr_{metric}_iter{i + 1}",
            label=r'Peak signal-to-noise ratio'
        )

    plt.figure(figsize=(6, 6))
    plt.hist(peak2peak, bins=100)
    plt.grid()
    plt.xlabel(
        'Peak-to-peak aberration $|P_{95} - P_{5}|$'
        rf'($\lambda = {int(wavelength*1000)}~nm$)'
    )
    plt.ylabel(rf'Number of samples')
    plt.savefig(f'{modelsdir}/{eval_distribution}_peak2peak.png', dpi=300, bbox_inches='tight', pad_inches=.25)


def compare_models_and_modes(
        modelsdir: Path,
        wavelength: float,
        x_voxel_size: float,
        y_voxel_size: float,
        z_voxel_size: float,
        psf_shape: tuple,
        n_samples: int,
        batch_size: int,
        cpu_workers: int,
        iterations: int = 5,
        psnr: int = 30
):
    n_batches = n_samples // batch_size
    models = [p for p in modelsdir.iterdir() if p.is_dir()]
    errors = [{m.stem: {} for m in models} for _ in range(iterations)]
    logger.info(f"Models: {models}")

    modes = backend.load(models[0]).layers[-1].output_shape[-1]
    eval_distribution = 'mixed'

    psfargs = dict(
        n_modes=modes,
        distribution=eval_distribution,
        lam_detection=wavelength,
        amplitude_ranges=.3,
        psf_shape=psf_shape,
        x_voxel_size=x_voxel_size,
        y_voxel_size=y_voxel_size,
        z_voxel_size=z_voxel_size,
        batch_size=batch_size,
        snr=psnr,
        max_jitter=1,
        cpu_workers=cpu_workers,
    )

    for _, (psfs, ys, psnrs, zplanes, maxcounts) in zip(
            range(n_batches), SyntheticPSF(**psfargs).generator(debug=True)
    ):

        for m in models:
            model = backend.load(m)

            preds = iterative_eval(
                model,
                inputs=psfs,
                outputs=ys,
                batch_size=batch_size,
                psfargs=psfargs,
                desc=f"{m.stem}, PSNR({int(psnr)})",
                iterations=iterations,
            )

            for i in range(iterations):
                ps = preds[0] if i == 0 else np.sum([preds[k] for k in range(i + 1)], axis=0)

                y_pred = pd.DataFrame(utils.microns2waves(ps, wavelength),
                                      columns=[f'Z{i}' for i in range(1, modes + 1)])
                y_true = pd.DataFrame(utils.microns2waves(ys, wavelength),
                                      columns=[f'Z{i}' for i in range(1, modes + 1)])
                residuals = np.abs(y_true - y_pred)
                residuals['model'] = m.stem
                errors[i][m.stem] = residuals

                if errors[i].get(m.stem) is not None:
                    errors[i][m.stem] = pd.concat([errors[i][m.stem], residuals], ignore_index=True)
                else:
                    errors[i][m.stem] = residuals

    for i in range(iterations):
        res = pd.concat(errors[i], ignore_index=True)
        logger.info(res)
        vis.plot_residuals_per_mode(
            res,
            wavelength=wavelength,
            nsamples=n_samples,
            save_path=f"{modelsdir}/{eval_distribution}_modes_iter{i + 1}",
        )


def synthatic_convergence(
        modelsdir: Path,
        wavelength: float,
        x_voxel_size: float,
        y_voxel_size: float,
        z_voxel_size: float,
        psf_shape: tuple,
        n_samples: int,
        batch_size: int,
        cpu_workers: int,
        psnr: int = 30,
        tolerance: float = 1e-2,
        patience: int = 5,
        max_iters: int = 15,
        eval_distribution: str = 'powerlaw',
):
    n_batches = n_samples // batch_size
    models = [p for p in modelsdir.iterdir() if p.is_dir()]
    logger.info(f"Models: {models}")
    modes = backend.load(models[0]).layers[-1].output_shape[-1]
    errors = pd.DataFrame([], columns=['niter', 'model', 'residuals'])
    avgs = None

    psfargs = dict(
        n_modes=modes,
        distribution=eval_distribution,
        lam_detection=wavelength,
        amplitude_ranges=(0, .3),
        psf_shape=psf_shape,
        x_voxel_size=x_voxel_size,
        y_voxel_size=y_voxel_size,
        z_voxel_size=z_voxel_size,
        batch_size=batch_size,
        snr=psnr,
        max_jitter=1,
        cpu_workers=cpu_workers,
    )
    gen = SyntheticPSF(**psfargs)

    for _, (psfs, ys, psnrs, zplanes, maxcounts) in zip(range(n_batches), gen.generator(debug=True)):

        for m in models:
            model = backend.load(m)

            niter = 0
            inputs = psfs
            outputs = ys
            predictions = {}
            converged = False

            while not converged:
                if model.name.lower() != 'phasenet':
                    inputs = np.stack(utils.multiprocess(
                        func=partial(gen.embedding, ishape=model.input_shape[-2]),
                        jobs=inputs,
                        desc='Preprocessing',
                        cores=cpu_workers
                    ),
                        axis=0
                    )

                predictions[niter], stdev = backend.bootstrap_predict(
                    model,
                    inputs,
                    psfgen=gen,
                    batch_size=batch_size,
                    desc=f"iter {niter} - {m.stem}, PSNR({int(psnr)})"
                )

                # compute error
                ps = predictions[0] if niter == 0 else np.sum([predictions[k] for k in range(niter + 1)], axis=0)
                y_pred = pd.DataFrame(utils.peak2peak(ps), columns=['residuals'])
                y_true = pd.DataFrame(utils.peak2peak(ys), columns=['residuals'])

                # drop aberrations below diffraction limit
                idx = y_true.index[y_true['residuals'] >= .25]
                y_true = y_true.loc[idx]
                y_pred = y_pred.loc[idx]

                err = np.abs(y_true - y_pred)
                err['model'] = m.stem
                err['niter'] = niter
                current_mean = err['residuals'].mean()

                errors = pd.concat([errors, err], ignore_index=True)
                avgs = errors.groupby(['niter', 'model']).mean().reset_index()

                # check if converged
                if niter >= max_iters \
                        or current_mean <= .05 \
                        or avgs[avgs['model'] == m.stem].shape[0] > 1 \
                        and np.allclose(avgs[avgs['model'] == m.stem]['residuals'].tail(patience).values,
                                        current_mean, rtol=tolerance, atol=tolerance):
                    converged = True

                # setup next iter
                res = outputs - predictions[niter]
                g = partial(
                    gen.single_psf,
                    zplanes=0,
                    normed=True,
                    noise=True,
                )
                inputs = np.expand_dims(np.stack(gen.batch(g, res), axis=0), -1)
                outputs = res
                niter += 1

    logger.info(avgs)
    errors.to_csv(f"{modelsdir}/{eval_distribution}_iters_residuals.csv", index=False)
    vis.plot_convergence(
        avgs,
        wavelength=wavelength,
        nsamples=n_samples,
        save_path=f"{modelsdir}/{eval_distribution}_iters",
    )


def convergence(
        modelsdir: Path,
        datadir: Path,
        wavelength: float,
        n_samples: int,
        batch_size: int,
        cpu_workers: int,
        psf_shape: tuple,
        psnr: int = 50,
        amplitude: int = 2,
        max_iters: int = 10,
        x_voxel_size: float = .15,
        y_voxel_size: float = .15,
        z_voxel_size: float = .6,
):
    classes = sorted([
        c for c in Path(datadir).rglob('*/')
        if c.is_dir()
           and len(list(c.glob('*.tif'))) > 0
           and f'psnr{psnr - 9}-{psnr}' in str(c)
           and f'p{amplitude - 1}9-p{amplitude}' in str(c)
    ])
    models = [p for p in modelsdir.iterdir() if p.is_dir()]
    logger.info(f"Models: {models}")
    modes = backend.load(models[0]).layers[-1].output_shape[-1]
    errors = pd.DataFrame([], columns=['niter', 'model', 'residuals'])

    psfargs = dict(
        n_modes=modes,
        lam_detection=wavelength,
        amplitude_ranges=(0, .3),
        psf_shape=psf_shape,
        x_voxel_size=x_voxel_size,
        y_voxel_size=y_voxel_size,
        z_voxel_size=z_voxel_size,
        batch_size=batch_size,
        snr=(psnr - 9, psnr),
        max_jitter=0,
        cpu_workers=cpu_workers,
    )

    for m in tqdm(models, desc='Evaluating', unit='model'):
        model = backend.load(m)

        for c in tqdm(classes, unit='class'):
            val = data_utils.load_dataset(c, samplelimit=n_samples)
            val = val.map(lambda x: tf.py_function(data_utils.get_sample, [x], [tf.float32, tf.float32]))

            for inputs, ys in val.batch(n_samples):
                predictions = iterative_eval(
                    model,
                    inputs=inputs,
                    outputs=ys.numpy(),
                    batch_size=n_samples,
                    iterations=max_iters,
                    psfargs=psfargs,
                    desc=f"Predictions for [{m.stem}] ({c})"
                )
                y_pred = pd.DataFrame(predictions, columns=['niter', 'residuals'])
                y_pred['model'] = m.stem
                y_pred['class'] = c
                errors = pd.concat([errors, y_pred], ignore_index=True)
                logger.info(errors)

    avgs = errors.groupby(['niter', 'model']).mean().reset_index()
    logger.info(avgs)
    errors.to_csv(f"{modelsdir}/iters_residuals_psnr{psnr - 9}-{psnr}_p{amplitude - 1}9-p{amplitude}.csv", index=False)
    vis.plot_convergence(
        avgs,
        wavelength=wavelength,
        nsamples=n_samples * len(classes),
        psnr=f'{psnr - 9}-{psnr}',
        save_path=f"{modelsdir}/iters_psnr{psnr - 9}-{psnr}_p{amplitude - 1}9-p{amplitude}",
    )


def eval_mode(phi, model, psfargs):
    gen = SyntheticPSF(**psfargs)
    model = backend.load(model)
    input_shape = model.layers[0].output_shape[0][1:-1]

    w = Wavefront(phi, order='ansi')
    abr = 0 if np.count_nonzero(phi) == 0 else round(utils.peak_aberration(phi))

    if input_shape[0] == 3:
        inputs = gen.single_otf(
            w, zplanes=0, normed=True, noise=True, na_mask=True, ratio=True, augmentation=True
        )
    else:
        inputs = gen.single_psf(w, zplanes=0, normed=True, noise=True, augmentation=True)

    # fig, axes = plt.subplots(1, 3)
    # img = inputs
    # m = axes[0].imshow(np.max(img, axis=0), cmap='Spectral_r', vmin=0, vmax=1)
    # axes[1].imshow(np.max(img, axis=1), cmap='Spectral_r', vmin=0, vmax=1)
    # axes[2].imshow(np.max(img, axis=2).T, cmap='Spectral_r', vmin=0, vmax=1)
    # cax = inset_axes(axes[2], width="10%", height="100%", loc='center right', borderpad=-3)
    # cb = plt.colorbar(m, cax=cax)
    # cax.yaxis.set_label_position("right")
    # plt.show()

    inputs = np.expand_dims(np.stack(inputs, axis=0), 0)
    inputs = np.expand_dims(np.stack(inputs, axis=0), -1)

    pred, stdev = backend.bootstrap_predict(
        model,
        inputs,
        psfgen=gen,
        batch_size=1,
        n_samples=1,
        desc=f"P2P({abr}), PSNR({int(psfargs['snr'])})"
    )

    phi = utils.peak_aberration(phi)
    pred = utils.peak_aberration(pred)
    residuals = np.abs(phi - pred)
    return residuals


def evaluate_modes(
        model,
        wavelength=.605,
        n_modes=60,
        psf_shape=64,
        psnr=30,
        x_voxel_size=.15,
        y_voxel_size=.15,
        z_voxel_size=.6,
):
    gen = dict(
        amplitude_ranges=(-1, 1),
        n_modes=n_modes,
        lam_detection=wavelength,
        psf_shape=tuple(3 * [psf_shape]),
        x_voxel_size=x_voxel_size,
        y_voxel_size=y_voxel_size,
        z_voxel_size=z_voxel_size,
        snr=psnr,
        max_jitter=0,
        cpu_workers=-1,
    )

    residuals = {}
    waves = np.arange(0, .5, step=.05)

    for i in range(5, n_modes):
        residuals[i] = {}
        jobs = np.zeros((len(waves), n_modes))
        jobs[:, i] = waves

        res = utils.multiprocess(
            partial(eval_mode, model=model, psfargs=gen),
            jobs=list(jobs),
            cores=1
        )

        residuals[i] = {round(utils.peak_aberration(jobs[k, :])): res[k] for k in range(len(waves))}
        df = pd.DataFrame.from_dict(residuals[i], orient="index")
        logger.info(df)

        vis.plot_mode(
            f'{model}/res_mode_{i}.png',
            df,
            mode_index=i,
            n_modes=n_modes,
            wavelength=wavelength
        )


def eval_bin(datapath, modelpath, samplelimit, na):
    model = backend.load(modelpath)
    gen = SyntheticPSF(
        n_modes=60,
        lam_detection=.605,
        amplitude_ranges=(-.25, .25),
        psf_shape=(64, 64, 64),
        x_voxel_size=.15,
        y_voxel_size=.15,
        z_voxel_size=.6,
        batch_size=100,
        snr=100,
        max_jitter=0,
        cpu_workers=-1,
    )

    val = data_utils.load_dataset(datapath, samplelimit=samplelimit)
    val = val.map(lambda x: tf.py_function(data_utils.get_sample, [x], [tf.float32, tf.float32]))

    y_pred = pd.DataFrame([], columns=['sample'])
    y_true = pd.DataFrame([], columns=['sample'])

    for inputs, ys in val.batch(100):
        preds, stdev = backend.bootstrap_predict(
            model,
            inputs,
            psfgen=gen,
            batch_size=100,
            n_samples=1,
            desc=f"Predictions for ({datapath})"
        )

        p = pd.DataFrame([utils.peak_aberration(i, na=na) for i in preds], columns=['sample'])
        y_pred = y_pred.append(p, ignore_index=True)

        y = pd.DataFrame([utils.peak_aberration(i, na=na) for i in ys.numpy()], columns=['sample'])
        y['snr'] = int(np.mean(list(map(int, datapath.parent.stem.lstrip('psnr_').split('-')))))
        y_true = y_true.append(y, ignore_index=True)

    return (y_pred, y_true)


def evalheatmap(
    modelpath: Path,
    datadir: Path,
    distribution: str = '/',
    samplelimit: Any = None,
    max_amplitude: float = .25,
    na: float = 1.0,
):
    savepath = modelpath / 'evalheatmaps'
    savepath.mkdir(parents=True, exist_ok=True)
    if distribution != '/':
        savepath = f'{savepath}/{distribution}_na_{str(na).replace("0.", "p")}'
    else:
        savepath = f'{savepath}/na_{str(na).replace("0.", "p")}'

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'xtick.major.pad': 10
    })

    classes = sorted([
        c for c in Path(datadir).rglob('*/')
        if c.is_dir()
           and len(list(c.glob('*.tif'))) > 0
           and distribution in str(c)
           and float(str(c.name.split('-')[-1].replace('p', '.'))) <= max_amplitude
    ])

    job = partial(eval_bin, modelpath=modelpath, samplelimit=samplelimit, na=na)
    preds, ys = zip(*utils.multiprocess(job, classes))
    y_true = pd.DataFrame([], columns=['sample']).append(ys, ignore_index=True)
    y_pred = pd.DataFrame([], columns=['sample']).append(preds, ignore_index=True)

    error = np.abs(y_true - y_pred)
    error = pd.DataFrame(error, columns=['sample'])

    bins = np.arange(0, 10.25, .25)
    df = pd.DataFrame(zip(y_true['sample'], error['sample'], y_true['snr']), columns=['aberration', 'error', 'snr'])
    df['bins'] = pd.cut(df['aberration'], bins, labels=bins[1:], include_lowest=True)

    means = pd.pivot_table(df, values='error', index='bins', columns='snr', aggfunc=np.mean)
    means = means.sort_index().interpolate()

    logger.info(means)
    means.to_csv(f'{savepath}.csv')

    fig, ax = plt.subplots(figsize=(8, 6))
    levels = [
        0, .05, .1, .15, .2, .25, .3, .35, .4, .45,
        .5, .6, .7, .8, .9,
        1, 1.25, 1.5, 1.75, 2., 2.5,
        3., 4., 5.,
    ]

    vmin, vmax, vcenter, step = levels[0], levels[-1], .25, .05
    highcmap = plt.get_cmap('magma_r', 256)
    lowcmap = plt.get_cmap('GnBu_r', 256)
    low = np.linspace(0, 1 - step, int(abs(vcenter - vmin) / step))
    high = np.linspace(0, 1 + step, int(abs(vcenter - vmax) / step))
    cmap = np.vstack((lowcmap(low), [1, 1, 1, 1], highcmap(high)))
    cmap = mcolors.ListedColormap(cmap)

    contours = ax.contourf(
        means.columns.values,
        means.index.values,
        means.values,
        cmap=cmap,
        levels=levels,
        extend='max',
        linewidths=2,
        linestyles='dashed',
    )
    ax.patch.set(hatch='/', edgecolor='lightgrey', lw=.01)

    cax = fig.add_axes([1.01, 0.08, 0.03, 0.87])
    cbar = plt.colorbar(
        contours,
        cax=cax,
        fraction=0.046,
        pad=0.04,
        extend='both',
        spacing='proportional',
        format=FormatStrFormatter("%.2f"),
        ticks=[0, .15, .3, .5, .75, 1., 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5],
    )

    cbar.ax.set_ylabel(rf'Average peak-to-peak residuals ($\lambda = 605~nm$)')
    cbar.ax.set_title(r'$\lambda$')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('left')

    ax.set_xlabel(f'Peak signal-to-noise ratio')
    ax.set_xlim(0, 100)
    ax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

    ax.set_ylabel(
        'Average Peak-to-peak aberration $|P_{95} - P_{5}|$'
        rf'($\lambda = 605~nm$)'
    )
    ax.set_yticks(np.arange(0, 11, .5), minor=True)
    ax.set_yticks(np.arange(0, 11, 1))
    ax.set_ylim(.25, 10)

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()

    plt.savefig(f'{savepath}.pdf', bbox_inches='tight', pad_inches=.25)
    plt.savefig(f'{savepath}.png', dpi=300, bbox_inches='tight', pad_inches=.25)
    return fig


def iter_eval_bin(datapath, modelpath, niter, psnr, samples, na):
    model = backend.load(modelpath)
    gen = SyntheticPSF(
        n_modes=60,
        lam_detection=.605,
        amplitude_ranges=(-.25, .25),
        psf_shape=(64, 64, 64),
        x_voxel_size=.15,
        y_voxel_size=.15,
        z_voxel_size=.6,
        batch_size=100,
        snr=psnr,
        max_jitter=0,
        cpu_workers=-1,
    )

    val = data_utils.load_dataset(datapath, samplelimit=samples)
    val = val.map(lambda x: tf.py_function(data_utils.get_sample, [x], [tf.float32, tf.float32]))
    val = np.array(list(val.take(-1)))

    inputs = np.array([i.numpy() for i in val[:, 0]])
    ys = np.array([i.numpy() for i in val[:, 1]])

    y_pred = pd.DataFrame.from_dict({
        'id': np.arange(inputs.shape[0], dtype=int),
        'niter': np.zeros(samples, dtype=int),
        'residuals': np.zeros(samples, dtype=float)
    })

    y_true = pd.DataFrame.from_dict({
        'id': np.arange(inputs.shape[0], dtype=int),
        'niter': np.zeros(samples, dtype=int),
        'residuals': [utils.peak_aberration(i, na=na) for i in ys]
    })

    for k in range(1, niter+1):
        preds, stdev = backend.bootstrap_predict(
            model,
            inputs,
            psfgen=gen,
            batch_size=samples,
            n_samples=1,
            desc=f"Predictions for ({datapath})"
        )

        p = pd.DataFrame([utils.peak_aberration(i, na=na) for i in preds], columns=['residuals'])
        p['niter'] = k
        p['id'] = np.arange(inputs.shape[0], dtype=int)
        y_pred = y_pred.append(p, ignore_index=True)

        y = pd.DataFrame([utils.peak_aberration(i, na=na) for i in ys], columns=['residuals'])
        y['niter'] = k
        y['id'] = np.arange(inputs.shape[0], dtype=int)
        y_true = y_true.append(y, ignore_index=True)

        # setup next iter
        res = ys - preds
        if model.name == 'PhaseNet':
            g = partial(
                gen.single_psf,
                zplanes=0,
                normed=True,
                noise=True,
                augmentation=True,
                meta=False
            )
        else:
            g = partial(
                gen.single_otf,
                zplanes=0,
                normed=True,
                noise=True,
                augmentation=True,
                meta=False,
                na_mask=True,
                ratio=True,
                padsize=None
            )

        inputs = np.expand_dims(np.stack(gen.batch(g, res), axis=0), -1)
        ys = res

    return (y_pred, y_true)


def iter_eval_bin_with_reference(datapath, modelpath, reference, niter, psnr, samples, na):
    reference = imread(reference).astype(np.float)
    reference /= np.max(reference)
    reference = np.expand_dims(reference, axis=-1)

    model = backend.load(modelpath)
    gen = SyntheticPSF(
        n_modes=60,
        lam_detection=.605,
        amplitude_ranges=(-.25, .25),
        psf_shape=(64, 64, 64),
        x_voxel_size=.15,
        y_voxel_size=.15,
        z_voxel_size=.6,
        batch_size=100,
        snr=psnr,
        max_jitter=0,
        cpu_workers=-1,
    )

    val = data_utils.load_dataset(datapath, samplelimit=samples)
    val = val.map(lambda x: tf.py_function(data_utils.get_sample, [x], [tf.float32, tf.float32]))
    val = np.array(list(val.take(-1)))

    inputs = np.array([i.numpy() for i in val[:, 0]])
    ys = np.array([i.numpy() for i in val[:, 1]])

    y_pred = pd.DataFrame.from_dict({
        'id': np.arange(inputs.shape[0], dtype=int),
        'niter': np.zeros(samples, dtype=int),
        'residuals': np.zeros(samples, dtype=float)
    })

    y_true = pd.DataFrame.from_dict({
        'id': np.arange(inputs.shape[0], dtype=int),
        'niter': np.zeros(samples, dtype=int),
        'residuals': [utils.peak_aberration(i, na=na) for i in ys]
    })

    for k in range(1, niter+1):
        inputs = np.array([
            transform.resize(
                signal.fftconvolve(i, reference, mode='same'),
                output_shape=(64, 64, 64),
                order=3,
                anti_aliasing=True
            ) for i in inputs
        ])

        preds, stdev = backend.bootstrap_predict(
            model,
            inputs,
            psfgen=gen,
            batch_size=samples,
            n_samples=1,
            desc=f"Predictions for ({datapath})"
        )

        p = pd.DataFrame([utils.peak_aberration(i, na=na) for i in preds], columns=['residuals'])

        p['niter'] = k
        p['id'] = np.arange(inputs.shape[0], dtype=int)
        y_pred = y_pred.append(p, ignore_index=True)

        y = pd.DataFrame([utils.peak_aberration(i, na=na) for i in ys], columns=['residuals'])
        y['niter'] = k
        y['id'] = np.arange(inputs.shape[0], dtype=int)
        y_true = y_true.append(y, ignore_index=True)

        # setup next iter
        res = ys - preds
        g = partial(
            gen.single_psf,
            zplanes=0,
            normed=True,
            noise=True,
            augmentation=True,
            meta=False
        )

        inputs = np.expand_dims(np.stack(gen.batch(g, res), axis=0), -1)
        ys = res

    return (y_pred, y_true)


def iterheatmap(
    modelpath: Path,
    datadir: Path,
    reference: Path = None,
    psnr: tuple = (91, 100),
    niter: int = 10,
    distribution: str = '/',
    samplelimit: Any = None,
    max_amplitude: float = .25,
    na: float = 1.0,
):
    if reference is None:
        savepath = modelpath / 'iterheatmap'
    else:
        savepath = modelpath / f'{reference.stem}_iterheatmap'

    savepath.mkdir(parents=True, exist_ok=True)
    if distribution != '/':
        savepath = f'{savepath}/{distribution}_na_{str(na).replace("0.", "p")}'
    else:
        savepath = f'{savepath}/na_{str(na).replace("0.", "p")}'

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'xtick.major.pad': 10
    })

    classes = sorted([
        c for c in Path(datadir).rglob('*/')
        if c.is_dir()
           and len(list(c.glob('*.tif'))) > 0
           and f'psnr_{psnr[0]}-{psnr[1]}' in str(c)
           and distribution in str(c)
           and float(str(c.name.split('-')[-1].replace('p', '.'))) <= max_amplitude
    ])

    if reference is None:
        job = partial(
            iter_eval_bin,
            modelpath=modelpath,
            niter=niter,
            psnr=psnr,
            samples=samplelimit,
            na=na
        )
    else:
        job = partial(
            iter_eval_bin_with_reference,
            modelpath=modelpath,
            reference=reference,
            niter=niter,
            psnr=psnr,
            samples=samplelimit,
            na=na
        )

    preds, ys = zip(*utils.multiprocess(job, classes))
    y_true = pd.DataFrame([], columns=['niter', 'residuals']).append(ys, ignore_index=True)
    y_pred = pd.DataFrame([], columns=['niter', 'residuals']).append(preds, ignore_index=True)

    error = np.abs(y_true['residuals'] - y_pred['residuals'])
    error = pd.DataFrame(error, columns=['residuals'])

    df = pd.DataFrame(
        zip(y_true['id'], y_true['residuals'], error['residuals'], y_true['niter']),
        columns=['id', 'aberration', 'error', 'niter'],
    )

    means = pd.pivot_table(
        df[df['niter'] == 0], values='error', index='aberration', columns='niter', aggfunc=np.mean
    )
    for i in range(1, niter+1):
        means[i] = pd.pivot_table(
            df[df['niter'] == i],
            values='error', index=means.index, columns='niter', aggfunc=np.mean
        )

    bins = np.arange(0, 11, .25)
    means.index = pd.cut(means.index, bins, labels=bins[1:], include_lowest=True)
    means.index.name = 'bins'
    means = means.groupby("bins").agg("mean")
    means.loc[0] = pd.Series({cc: 0 for cc in means.columns})
    means = means.sort_index().interpolate()

    logger.info(means)
    means.to_csv(f'{savepath}.csv')

    fig, ax = plt.subplots(figsize=(8, 6))
    levels = [
        0, .05, .1, .15, .2, .25, .3, .35, .4, .45,
        .5, .6, .7, .8, .9,
        1, 1.25, 1.5, 1.75, 2., 2.5,
        3., 4., 5.,
    ]

    vmin, vmax, vcenter, step = levels[0], levels[-1], .25, .05
    highcmap = plt.get_cmap('magma_r', 256)
    lowcmap = plt.get_cmap('GnBu_r', 256)
    low = np.linspace(0, 1 - step, int(abs(vcenter - vmin) / step))
    high = np.linspace(0, 1 + step, int(abs(vcenter - vmax) / step))
    cmap = np.vstack((lowcmap(low), [1, 1, 1, 1], highcmap(high)))
    cmap = mcolors.ListedColormap(cmap)

    contours = ax.contourf(
        means.columns.values,
        means.index.values,
        means.values,
        cmap=cmap,
        levels=levels,
        extend='max',
        linewidths=2,
        linestyles='dashed',
    )
    ax.patch.set(hatch='/', edgecolor='lightgrey', lw=.01)

    cax = fig.add_axes([1.01, 0.08, 0.03, 0.87])
    cbar = plt.colorbar(
        contours,
        cax=cax,
        fraction=0.046,
        pad=0.04,
        extend='both',
        spacing='proportional',
        format=FormatStrFormatter("%.2f"),
        ticks=[0, .15, .3, .5, .75, 1., 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5],
    )

    cbar.ax.set_ylabel(rf'Average peak-to-peak residuals ($\lambda = 605~nm$)')
    cbar.ax.set_title(r'$\lambda$')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('left')

    ax.set_xlabel(f'Number of iterations')
    ax.set_xlim(0, niter)
    ax.set_xticks(range(niter+1))
    ax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

    ax.set_ylabel(
        'Average Peak-to-peak aberration $|P_{95} - P_{5}|$ '
        rf'($\lambda = 605~nm$)'
    )

    ax.set_yticks(np.arange(0, 11, .5), minor=True)
    ax.set_yticks(np.arange(0, 11, 1))
    ax.set_ylim(.25, 10)

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()

    plt.savefig(f'{savepath}.pdf', bbox_inches='tight', pad_inches=.25)
    plt.savefig(f'{savepath}.png', dpi=300, bbox_inches='tight', pad_inches=.25)
    return fig


def evalsample(
    model_path: Path,
    kernel_path: Path = None,
    reference_path: Path = None,
    psnr: tuple = (90, 100),
    niter: int = 1,
    na: float = 1.0,
    reference_voxel_size: tuple = (.002, .002, .002),
    rolling_embedding: bool = False
):

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'xtick.major.pad': 10
    })

    modelgen = SyntheticPSF(
        n_modes=60,
        lam_detection=.605,
        psf_shape=(64, 64, 64),
        x_voxel_size=.15,
        y_voxel_size=.15,
        z_voxel_size=.6,
        snr=psnr,
        max_jitter=0,
    )

    savepath = model_path / f'{reference_path.stem}'
    savepath.mkdir(parents=True, exist_ok=True)

    model = backend.load(model_path)

    reference = imread(reference_path).astype(np.float)
    reference /= np.max(reference)
    logger.info(f"Reference: {reference.shape}")

    ys = np.zeros(60)
    ys[10] = .1

    if kernel_path is not None:
        kernel = imread(kernel_path)
    else:
        gen = SyntheticPSF(
            n_modes=60,
            lam_detection=.605,
            psf_shape=reference.shape,
            z_voxel_size=reference_voxel_size[0],
            y_voxel_size=reference_voxel_size[1],
            x_voxel_size=reference_voxel_size[2],
            snr=psnr,
            max_jitter=0,
        )

        kernel = gen.single_psf(
            phi=Wavefront(ys),
            zplanes=0,
            normed=True,
            noise=False,
            augmentation=False,
            meta=False
        )
    imsave(savepath/f'kernel.tif', kernel)
    logger.info(f"Kernel: {kernel.shape}")

    y_pred = pd.DataFrame.from_dict({'niter': [0], 'residuals': [0]})
    y_true = pd.DataFrame.from_dict({'niter': [0], 'residuals': [utils.peak_aberration(ys, na=na)]})

    for k in range(1, niter+1):
        conv = signal.convolve(reference, kernel, mode='full')
        width = [(i//2) for i in reference.shape]
        center = [(i//2)+1 for i in conv.shape]
        # center = np.unravel_index(np.argmax(conv, axis=None), conv.shape)
        conv = conv[
             center[0]-width[0]:center[0]+width[0],
             center[1]-width[1]:center[1]+width[1],
             center[2]-width[2]:center[2]+width[2],
        ]
        imsave(savepath/f'convolved_iter_{k}.tif', conv)

        preds, stdev = backend.bootstrap_predict(
            model,
            np.expand_dims(conv[np.newaxis, :], axis=-1),
            psfgen=modelgen,
            resize=reference_voxel_size,
            rolling_embedding=rolling_embedding,
            batch_size=1,
            n_samples=1,
            desc=f'Iter[{k}]',
            plot=savepath/f'embeddings_convolved_iter_{k}',
        )

        p_wave = Wavefront(preds)
        y_wave = Wavefront(ys.flatten())
        diff_wave = Wavefront(ys - preds)

        p_psf = modelgen.single_psf(p_wave, zplanes=0)
        gt_psf = modelgen.single_psf(y_wave, zplanes=0)
        corrected_psf = modelgen.single_psf(diff_wave, zplanes=0)
        imsave(savepath / f'corrected_psf_iter_{k}.tif', corrected_psf)

        vis.diagnostic_assessment(
            psf=conv,
            gt_psf=gt_psf,
            predicted_psf=p_psf,
            corrected_psf=corrected_psf,
            psnr=psnr,
            maxcounts=psnr,
            y=y_wave,
            pred=p_wave,
            save_path=savepath/f'iter_{k}',
        )

        p = pd.DataFrame({'residuals': [utils.peak_aberration(preds, na=na)], 'niter': k})
        y_pred = y_pred.append(p, ignore_index=True)

        y = pd.DataFrame({'residuals': [utils.peak_aberration(ys, na=na)], 'niter': k})
        y_true = y_true.append(y, ignore_index=True)

        # setup next iter
        if niter > 1:
            res = ys - preds
            kernel = gen.single_psf(
                phi=Wavefront(res),
                zplanes=0,
                normed=True,
                noise=False,
                augmentation=False,
                meta=False
            )
            ys = res

    error = np.abs(y_true['residuals'] - y_pred['residuals'])
    error = pd.DataFrame(error, columns=['residuals'])

    df = pd.DataFrame(
        zip(y_true['residuals'], error['residuals'], y_true['niter']),
        columns=['aberration', 'error', 'niter'],
    )

    means = pd.pivot_table(
        df[df['niter'] == 0], values='error', index='aberration', columns='niter', aggfunc=np.mean
    )
    for i in range(1, niter+1):
        means[i] = pd.pivot_table(
            df[df['niter'] == i],
            values='error', index=means.index, columns='niter', aggfunc=np.mean
        )

    bins = np.arange(0, 11, .25)
    means.index = pd.cut(means.index, bins, labels=bins[1:], include_lowest=True)
    means.index.name = 'bins'
    means = means.groupby("bins").agg("mean")
    means.loc[0] = pd.Series({cc: 0 for cc in means.columns})
    means = means.sort_index().interpolate()

    logger.info(means)
    means.to_csv(savepath/f'results_na_{str(na).replace("0.", "p")}.csv')

    fig, ax = plt.subplots(figsize=(8, 6))
    levels = [
        0, .05, .1, .15, .2, .25, .3, .35, .4, .45,
        .5, .6, .7, .8, .9,
        1, 1.25, 1.5, 1.75, 2., 2.5,
        3., 4., 5.,
    ]

    vmin, vmax, vcenter, step = levels[0], levels[-1], .25, .05
    highcmap = plt.get_cmap('magma_r', 256)
    lowcmap = plt.get_cmap('GnBu_r', 256)
    low = np.linspace(0, 1 - step, int(abs(vcenter - vmin) / step))
    high = np.linspace(0, 1 + step, int(abs(vcenter - vmax) / step))
    cmap = np.vstack((lowcmap(low), [1, 1, 1, 1], highcmap(high)))
    cmap = mcolors.ListedColormap(cmap)

    contours = ax.contourf(
        means.columns.values,
        means.index.values,
        means.values,
        cmap=cmap,
        levels=levels,
        extend='max',
        linewidths=2,
        linestyles='dashed',
    )
    ax.patch.set(hatch='/', edgecolor='lightgrey', lw=.01)

    cax = fig.add_axes([1.01, 0.08, 0.03, 0.87])
    cbar = plt.colorbar(
        contours,
        cax=cax,
        fraction=0.046,
        pad=0.04,
        extend='both',
        spacing='proportional',
        format=FormatStrFormatter("%.2f"),
        ticks=[0, .15, .3, .5, .75, 1., 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5],
    )

    cbar.ax.set_ylabel(rf'Average peak-to-peak residuals ($\lambda = 605~nm$)')
    cbar.ax.set_title(r'$\lambda$')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('left')

    ax.set_xlabel(f'Number of iterations')
    ax.set_xlim(0, niter)
    ax.set_xticks(range(niter+1))
    ax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

    ax.set_ylabel(
        'Average Peak-to-peak aberration $|P_{95} - P_{5}|$ '
        rf'($\lambda = 605~nm$)'
    )

    ax.set_yticks(np.arange(0, 11, .5), minor=True)
    ax.set_yticks(np.arange(0, 11, 1))
    ax.set_ylim(.25, 10)

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()

    plt.savefig(savepath/f'iterheatmap_na_{str(na).replace("0.", "p")}.pdf', bbox_inches='tight', pad_inches=.25)
    plt.savefig(savepath/f'iterheatmap_na_{str(na).replace("0.", "p")}.png', dpi=300, bbox_inches='tight', pad_inches=.25)
    return fig
