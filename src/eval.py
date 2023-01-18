import matplotlib
matplotlib.use('Agg')

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
from tqdm import tqdm, trange
import tensorflow as tf
from preprocessing import resize_with_crop_or_pad
from line_profiler_pycharm import profile
from tqdm import tqdm
from tifffile import imsave

import utils
import data_utils
import backend
import vis
import multipoint_dataset

from wavefront import Wavefront

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
tf.get_logger().setLevel(logging.ERROR)


@profile
def simulate_beads(psf, gen, snr, object_size=0, num_objs=1, beads=None, noise=None):
    if beads is None:
        beads = multipoint_dataset.beads(
            gen=gen,
            object_size=object_size,
            num_objs=num_objs
        )

    img = utils.fftconvolution(sample=beads, kernel=psf)
    img *= snr ** 2

    if noise is None:
        noise = gen._random_noise(
            image=img,
            mean=gen.mean_background_noise,
            sigma=gen.sigma_background_noise
        )

    noisy_img = noise + img
    noisy_img /= np.max(noisy_img)
    return noisy_img


@profile
def eval_mode(
    phi,
    modelpath,
    na: float = 1.0,
    batch_size: int = 100,
    snr_range: tuple = (21, 30),
    n_samples: int = 10,
    eval_sign: str = 'positive_only'
):
    model = backend.load(modelpath)
    gen = backend.load_metadata(modelpath, psf_shape=3*[model.input_shape[2]])

    p2v = utils.peak2valley(phi, na=na, wavelength=gen.lam_detection)
    df = pd.DataFrame([], columns=['aberration', 'prediction', 'residuals', 'object_size'])

    w = Wavefront(phi, lam_detection=gen.lam_detection)
    kernel = gen.single_psf(
        phi=w,
        normed=True,
        noise=False,
        meta=False,
    )

    k = np.where(phi > 0)[0]
    for isize in tqdm([0, 1, 2, 3, 4, 5], desc=f"Evaluate different sizes [mode #{k}]"):
        psnr = gen._randuniform(snr_range)
        inputs = np.array([
            simulate_beads(
                psf=kernel,
                gen=gen,
                object_size=isize,
                num_objs=1,
                snr=psnr,
            )
            for i in range(n_samples)
        ])
        ys = np.array([phi for i in inputs])

        residuals, ys, preds = backend.evaluate(
            model=model,
            inputs=inputs,
            gen=gen,
            ys=ys,
            psnr=psnr,
            batch_size=batch_size,
            eval_sign=eval_sign
        )

        p = pd.DataFrame([p2v for i in inputs], columns=['aberration'])
        p['prediction'] = [utils.peak2valley(i, na=na, wavelength=gen.lam_detection) for i in preds]
        p['residuals'] = [utils.peak2valley(i, na=na, wavelength=gen.lam_detection) for i in residuals]
        p['object_size'] = 1 if isize == 0 else isize * 2
        df = df.append(p, ignore_index=True)

    return df


@profile
def evaluate_modes(model: Path, eval_sign: str = 'positive_only'):
    outdir = model.with_suffix('') / 'evalmodes'
    outdir.mkdir(parents=True, exist_ok=True)
    modelspecs = backend.load_metadata(model)

    waves = np.arange(1e-5, .75, step=.05)
    aberrations = np.zeros((len(waves), modelspecs.n_modes))

    for i in trange(5, modelspecs.n_modes):
        savepath = outdir / f"m{i}"

        classes = aberrations.copy()
        classes[:, i] = waves

        job = partial(eval_mode, modelpath=model, eval_sign=eval_sign)
        preds = utils.multiprocess(job, list(classes), cores=-1)
        df = pd.DataFrame([]).append(preds, ignore_index=True)

        bins = np.arange(0, 10.25, .25)
        df['bins'] = pd.cut(df['aberration'], bins, labels=bins[1:], include_lowest=True)

        means = pd.pivot_table(df, values='residuals', index='bins', columns='object_size', aggfunc=np.mean)
        means = means.sort_index().interpolate()
        logger.info(means)

        fig = plt.figure(figsize=(8, 8))
        gs = fig.add_gridspec(4, 4)
        ax_xy = fig.add_subplot(gs[0, 0])
        ax_xz = fig.add_subplot(gs[0, 1])
        ax_yz = fig.add_subplot(gs[0, 2])
        ax_wavevfront = fig.add_subplot(gs[0, -1])
        ax = fig.add_subplot(gs[1:, :])

        levels = [
            0, .05, .1, .15, .2, .25, .3, .35, .4, .45,
            .5, .6, .7, .8, .9,
            1, 1.25, 1.5, 1.75, 2., 2.5,
            3., 4., 5.,
        ]

        vmin, vmax, vcenter, step = levels[0], levels[-1], .5, .05
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

        cax = fig.add_axes([1.01, 0.08, 0.03, 0.7])
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

        cbar.ax.set_ylabel(rf'Average peak-to-valley residuals ($\lambda = 510~nm$)')
        cbar.ax.set_title(r'$\lambda$')
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.yaxis.set_label_position('left')

        ax.set_xlabel(f'Diameter of the simulated object (pixels)')
        ax.set_xlim(1, 10)
        ax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

        ax.set_ylabel(
            'Average peak-to-valley aberration'
            rf' ($\lambda = 510~nm$)'
        )
        ax.set_yticks(np.arange(0, 6, .5), minor=True)
        ax.set_yticks(np.arange(0, 6, 1))
        ax.set_ylim(.25, 5)

        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        phi = np.zeros_like(classes[-1, :])
        phi[i] = .2
        gen = backend.load_metadata(model, psf_shape=(64, 64, 64))
        w = Wavefront(phi, lam_detection=gen.lam_detection)
        kernel = gen.single_psf(
            phi=w,
            normed=True,
            noise=False,
            meta=False,
        )
        ax_xy.imshow(np.max(kernel, axis=0)**.5, vmin=0, vmax=1, cmap='hot')
        ax_xz.imshow(np.max(kernel, axis=1)**.5, vmin=0, vmax=1, cmap='hot')
        ax_yz.imshow(np.max(kernel, axis=2)**.5, vmin=0, vmax=1, cmap='hot')
        ax_wavevfront.imshow(w.wave(size=100), vmin=-1, vmax=1, cmap='Spectral_r')

        for a, t in zip([ax_xy, ax_xz, ax_yz, ax_wavevfront], ['XY', 'XZ', 'YZ', 'Wavefront']):
            a.axis('off')
            a.set_title(t)

        plt.tight_layout()

        plt.savefig(f'{savepath}.pdf', bbox_inches='tight', pad_inches=.25)
        plt.savefig(f'{savepath}.png', dpi=300, bbox_inches='tight', pad_inches=.25)


@profile
def eval_bin(
    datapath,
    modelpath,
    samplelimit: int = 1,
    na: float = 1.0,
    input_coverage: float = 1.0,
    no_phase: bool = False,
    batch_size: int = 100,
    snr_range: Any = None,
    distribution: str = '',
    eval_sign: str = 'positive_only'
):
    model = backend.load(modelpath)
    gen = backend.load_metadata(
        modelpath,
        snr=1000,
        signed=True,
        rotate=True,
        batch_size=batch_size,
        psf_shape=3*[model.input_shape[2]]
    )

    val = data_utils.collect_dataset(
        datapath,
        modes=gen.n_modes,
        samplelimit=samplelimit,
        distribution=distribution,
        no_phase=no_phase,
        metadata=True,
        snr_range=snr_range
    )

    df = pd.DataFrame([], columns=['aberration', 'prediction', 'residuals', 'snr', 'neighbors', 'dist'])

    for inputs, ys, snr, p2v, npoints in val.batch(batch_size):
        if input_coverage != 1.:
            inputs = resize_with_crop_or_pad(inputs, crop_shape=[int(s*input_coverage) for s in gen.psf_shape])

        residuals, ys, preds = backend.evaluate(
            model=model,
            inputs=inputs,
            gen=gen,
            ys=ys,
            psnr=snr,
            batch_size=batch_size,
            eval_sign=eval_sign
        )

        p = pd.DataFrame([utils.peak2valley(i, na=na, wavelength=gen.lam_detection) for i in ys], columns=['aberration'])
        p['prediction'] = [utils.peak2valley(i, na=na, wavelength=gen.lam_detection) for i in preds]
        p['residuals'] = [utils.peak2valley(i, na=na, wavelength=gen.lam_detection) for i in residuals]
        p['snr'] = snr.numpy()
        p['neighbors'] = npoints
        p['dist'] = [
            utils.mean_min_distance(np.squeeze(i), voxel_size=gen.voxel_size) if objs > 1 else 0.
            for i, objs in zip(inputs, npoints)
        ]
        df = df.append(p, ignore_index=True)

    bins = np.arange(0, 10.25, .25)
    distbins = np.arange(0, 5.5, .5)
    df['bins'] = pd.cut(df['aberration'], bins, labels=bins[1:], include_lowest=True)
    df['dist'] = pd.cut(df['dist'], distbins, labels=distbins[:-1], include_lowest=True)
    return df


@profile
def plot_heatmap(means, wavelength, savepath, label=f'Peak signal-to-noise ratio', lims=(0, 100)):
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'xtick.major.pad': 10
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    levels = [
        0, .05, .1, .15, .2, .25, .3, .35, .4, .45,
        .5, .6, .7, .8, .9,
        1, 1.25, 1.5, 1.75, 2., 2.5,
        3., 4., 5.,
    ]

    vmin, vmax, vcenter, step = levels[0], levels[-1], .5, .05
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

    cbar.ax.set_ylabel(rf'Average peak-to-valley residuals ($\lambda = {int(wavelength * 1000)}~nm$)')
    cbar.ax.set_title(r'$\lambda$')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('left')

    ax.set_xlabel(label)
    ax.set_xlim(lims)
    ax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

    ax.set_ylabel(
        'Average peak-to-valley aberration'
        rf'($\lambda = {int(wavelength * 1000)}~nm$)'
    )
    ax.set_yticks(np.arange(0, 6, .5), minor=True)
    ax.set_yticks(np.arange(0, 6, 1))
    ax.set_ylim(.25, 5)

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()

    plt.savefig(f'{savepath}.pdf', bbox_inches='tight', pad_inches=.25)
    plt.savefig(f'{savepath}.png', dpi=300, bbox_inches='tight', pad_inches=.25)
    return fig


@profile
def snrheatmap(
    modelpath: Path,
    datadir: Path,
    distribution: str = '/',
    samplelimit: Any = None,
    na: float = 1.0,
    input_coverage: float = 1.0,
    no_phase: bool = False,
    batch_size: int = 100,
    eval_sign: str = 'positive_only'
):
    modelspecs = backend.load_metadata(modelpath)
    savepath = modelpath.with_suffix('') / f'snrheatmaps_{input_coverage}'
    savepath.mkdir(parents=True, exist_ok=True)

    if distribution != '/':
        savepath = Path(f'{savepath}/{distribution}_na_{str(na).replace("0.", "p")}')
    else:
        savepath = Path(f'{savepath}/na_{str(na).replace("0.", "p")}')

    df = eval_bin(
        datadir,
        modelpath=modelpath,
        samplelimit=samplelimit,
        na=na,
        input_coverage=input_coverage,
        no_phase=no_phase,
        batch_size=batch_size,
        snr_range=(0, 100),
        eval_sign=eval_sign
    )

    means = pd.pivot_table(df, values='residuals', index='bins', columns='snr', aggfunc=np.mean)
    means = means.sort_index().interpolate()

    logger.info(means)
    means.to_csv(f'{savepath}.csv')

    plot_heatmap(
        means,
        wavelength=modelspecs.lam_detection,
        savepath=savepath,
        label=f'Peak signal-to-noise ratio',
        lims=(0, 100)
    )


@profile
def densityheatmap(
    modelpath: Path,
    datadir: Path,
    distribution: str = '/',
    na: float = 1.0,
    no_phase: bool = False,
    samplelimit: Any = None,
    input_coverage: float = 1.0,
    batch_size: int = 100,
    snr_range: tuple = (21, 30),
    eval_sign: str = 'positive_only'
):
    modelspecs = backend.load_metadata(modelpath)
    df = eval_bin(
        datadir,
        modelpath=modelpath,
        samplelimit=samplelimit,
        distribution=distribution,
        na=na,
        snr_range=snr_range,
        input_coverage=input_coverage,
        no_phase=no_phase,
        batch_size=batch_size,
        eval_sign=eval_sign
    )

    for savedir, col, label, lims in zip(
        ['densityheatmaps', 'distanceheatmaps'],
        ['neighbors', 'dist'],
        ['Number of objects', 'Average distance to nearest neighbor (microns)'],
        [(1, 30), (0, 4)]
    ):
        savepath = modelpath.with_suffix('') / f'{savedir}_{input_coverage}'
        savepath.mkdir(parents=True, exist_ok=True)

        if distribution != '/':
            savepath = Path(f'{savepath}/{distribution}_na_{str(na).replace("0.", "p")}')
        else:
            savepath = Path(f'{savepath}/na_{str(na).replace("0.", "p")}')

        means = pd.pivot_table(df, values='residuals', index='bins', columns=col, aggfunc=np.mean)
        means = means.sort_index().interpolate()
        means.to_csv(f'{savepath}.csv')

        plot_heatmap(
            means,
            wavelength=modelspecs.lam_detection,
            savepath=savepath,
            label=label,
            lims=lims
        )


@profile
def iter_eval_bin_with_reference(
    datapath,
    modelpath,
    niter: int = 5,
    samplelimit: int = 1,
    na: float = 1.0,
    input_coverage: float = 1.0,
    no_phase: bool = False,
    batch_size: int = 100,
    snr_range: tuple = (21, 30),
    eval_sign: str = 'positive_only'
):
    model = backend.load(modelpath)
    gen = backend.load_metadata(
        modelpath,
        snr=1000,
        signed=True,
        rotate=True,
        batch_size=batch_size,
        psf_shape=3*[model.input_shape[2]]
    )

    val = data_utils.collect_dataset(
        datapath,
        modes=gen.n_modes,
        samplelimit=samplelimit,
        no_phase=no_phase,
        snr_range=snr_range
    )
    val = np.array(list(val.take(-1)))
    inputs = np.array([i.numpy() for i in val[:, 0]])
    ys = np.array([i.numpy() for i in val[:, 1]])

    # ys = np.zeros_like(ys)
    # ys[:, 3] = .1

    p2v = [utils.peak2valley(i, na=na, wavelength=gen.lam_detection) for i in ys]
    residuals = pd.DataFrame.from_dict({
        'id': np.arange(ys.shape[0], dtype=int),
        'niter': np.zeros(ys.shape[0], dtype=int),
        'aberration': p2v,
        'residuals': p2v,
    })

    reference = multipoint_dataset.beads(gen=gen, object_size=0, num_objs=1)
    snr = gen._randuniform(snr_range)
    rand_noise = gen._random_noise(
        image=reference,
        mean=gen.mean_background_noise,
        sigma=gen.sigma_background_noise
    )

    for k in range(1, niter+1):
        for i in range(inputs.shape[0]):
            phi = Wavefront(
                ys[i],
                modes=gen.n_modes,
                signed=True,
                rotate=True,
                mode_weights='pyramid',
                lam_detection=gen.lam_detection,
            )

            psf = gen.single_psf(
                phi=phi,
                normed=True,
                noise=False,
                meta=False,
            )

            img = utils.fftconvolution(sample=reference, kernel=psf)
            img *= snr ** 2

            noisy_img = rand_noise + img
            noisy_img /= np.max(noisy_img)
            inputs[i] = noisy_img[..., np.newaxis]

        if input_coverage != 1.:
            inputs = resize_with_crop_or_pad(inputs, crop_shape=[int(s*input_coverage) for s in gen.psf_shape])

        res, ys, ps = backend.evaluate(
            model=model,
            inputs=inputs,
            gen=gen,
            ys=ys,
            psnr=snr,
            batch_size=batch_size,
            reference=reference,
            eval_sign=eval_sign
        )

        y = pd.DataFrame(np.arange(inputs.shape[0], dtype=int), columns=['id'])
        y['residuals'] = [utils.peak2valley(i, na=na, wavelength=gen.lam_detection) for i in res]
        y['aberration'] = [utils.peak2valley(i, na=na, wavelength=gen.lam_detection) for i in ys]
        y['predictions'] = [utils.peak2valley(i, na=na, wavelength=gen.lam_detection) for i in ps]
        y['niter'] = k
        residuals = residuals.append(y, ignore_index=True)

        # setup next iter
        ys = res

    return residuals


@profile
def iterheatmap(
    modelpath: Path,
    datadir: Path,
    niter: int = 5,
    distribution: str = '/',
    samplelimit: Any = None,
    na: float = 1.0,
    input_coverage: float = 1.0,
    no_phase: bool = False,
    batch_size: int = 100,
    snr_range: tuple = (21, 30),
    eval_sign: str = 'positive_only'
):
    modelspecs = backend.load_metadata(modelpath)
    savepath = modelpath.with_suffix('') / f'iterheatmaps_{input_coverage}'
    savepath.mkdir(parents=True, exist_ok=True)

    if distribution != '/':
        savepath = Path(f'{savepath}/{distribution}_na_{str(na).replace("0.", "p")}')
    else:
        savepath = Path(f'{savepath}/na_{str(na).replace("0.", "p")}')

    df = iter_eval_bin_with_reference(
        datadir,
        niter=niter,
        modelpath=modelpath,
        samplelimit=samplelimit,
        na=na,
        snr_range=snr_range,
        input_coverage=input_coverage,
        no_phase=no_phase,
        batch_size=batch_size,
        eval_sign=eval_sign
    )

    means = pd.pivot_table(
        df[df['niter'] == 0], values='residuals', index='aberration', columns='niter', aggfunc=np.mean
    )
    for i in range(1, niter+1):
        means[i] = pd.pivot_table(
            df[df['niter'] == i],
            values='residuals', index=means.index, columns='niter', aggfunc=np.mean
        )

    bins = np.arange(0, 11, .25)
    means.index = pd.cut(means.index, bins, labels=bins[1:], include_lowest=True)
    means.index.name = 'bins'
    means = means.groupby("bins").agg("mean")
    means.loc[0] = pd.Series({cc: 0 for cc in means.columns})
    means = means.sort_index().interpolate()

    logger.info(means)
    means.to_csv(f'{savepath}.csv')

    plot_heatmap(
        means,
        wavelength=modelspecs.lam_detection,
        savepath=savepath,
        label=f'Number of iterations',
        lims=(0, niter)
    )


@profile
def random_samples(
    model: Path,
    psnr: int = 30,
    eval_sign: str = 'positive_only'
):
    m = backend.load(model)
    m.summary()

    for dist in ['single', 'bimodal', 'multinomial', 'powerlaw', 'dirichlet']:
        for amplitude_range in [(.05, .1), (.1, .2), (.2, .3)]:
            gen = backend.load_metadata(
                model,
                snr=1000,
                batch_size=1,
                amplitude_ranges=amplitude_range,
                distribution=dist,
                signed=False if eval_sign == 'positive_only' else True,
                rotate=True,
                mode_weights='pyramid',
                psf_shape=(64, 64, 64),
                mean_background_noise=0,
            )
            for s in range(1):
                for num_objs in tqdm([1, 2, 5, 10]):
                    reference = multipoint_dataset.beads(
                        gen=gen,
                        object_size=0,
                        num_objs=num_objs
                    )

                    phi = Wavefront(
                        amplitude_range,
                        modes=gen.n_modes,
                        distribution=dist,
                        signed=False if eval_sign == 'positive_only' else True,
                        rotate=True,
                        mode_weights='pyramid',
                        lam_detection=gen.lam_detection,
                    )

                    # aberrated PSF without noise
                    psf, y, snr, maxcounts = gen.single_psf(
                        phi=phi,
                        normed=True,
                        noise=False,
                        meta=True,
                    )

                    img = utils.fftconvolution(sample=reference, kernel=psf)
                    img *= psnr ** 2

                    rand_noise = gen._random_noise(
                        image=img,
                        mean=0,
                        sigma=gen.sigma_background_noise
                    )
                    noisy_img = rand_noise + img
                    maxcounts = np.max(noisy_img)
                    noisy_img /= maxcounts

                    save_path = Path(
                        f"{model.with_suffix('')}/samples/{dist}/um-{amplitude_range[-1]}/num_objs-{num_objs}"
                    )
                    save_path.mkdir(exist_ok=True, parents=True)

                    residuals, y, p = backend.evaluate(
                        model=m,
                        inputs=noisy_img[np.newaxis, :, :, :, np.newaxis],
                        reference=reference,
                        gen=gen,
                        ys=y,
                        psnr=psnr,
                        batch_size=1,
                        plot=save_path / f'{s}',
                        eval_sign=eval_sign
                    )

                    p_wave = Wavefront(p, lam_detection=gen.lam_detection)
                    y_wave = Wavefront(y, lam_detection=gen.lam_detection)
                    residuals = Wavefront(residuals, lam_detection=gen.lam_detection)

                    p_psf = gen.single_psf(p_wave, normed=True, noise=True)
                    gt_psf = gen.single_psf(y_wave, normed=True, noise=True)

                    corrected_psf = gen.single_psf(residuals)
                    corrected_noisy_img = utils.fftconvolution(sample=reference, kernel=corrected_psf)
                    corrected_noisy_img *= psnr ** 2
                    corrected_noisy_img = rand_noise + corrected_noisy_img
                    corrected_noisy_img /= np.max(corrected_noisy_img)

                    imsave(save_path / f'psf_{s}.tif', noisy_img)
                    imsave(save_path / f'corrected_psf_{s}.tif', corrected_psf)

                    plt.style.use("default")
                    vis.diagnostic_assessment(
                        psf=noisy_img,
                        gt_psf=gt_psf,
                        predicted_psf=p_psf,
                        corrected_psf=corrected_noisy_img,
                        psnr=psnr,
                        maxcounts=maxcounts,
                        y=y_wave,
                        pred=p_wave,
                        save_path=save_path / f'{s}',
                        display=False
                    )

                    plt.style.use('dark_background')
                    vis.diagnostic_assessment(
                        psf=noisy_img,
                        gt_psf=gt_psf,
                        predicted_psf=p_psf,
                        corrected_psf=corrected_noisy_img,
                        psnr=psnr,
                        maxcounts=maxcounts,
                        y=y_wave,
                        pred=p_wave,
                        save_path=save_path / f'{s}_db',
                        display=False
                    )
