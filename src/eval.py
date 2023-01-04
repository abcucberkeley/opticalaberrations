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
from scipy import stats as st

import utils
import data_utils
import backend

from wavefront import Wavefront

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
tf.get_logger().setLevel(logging.ERROR)


def eval_mode(
    phi,
    modelpath,
    npoints: int = 1,
    n_samples: int = 10,
    psnr: tuple = (21, 30),
    na: float = 1.0,
    batch_size: int = 100
):
    def sim_beads(ker):
        snr = gen._randuniform(psnr)
        ref = backend.beads(
            gen=gen,
            object_size=isize,
            num_objs=npoints
        )

        img = utils.fftconvolution(sample=ref, kernel=ker)
        img *= snr ** 2

        rand_noise = gen._random_noise(
            image=img,
            mean=gen.mean_background_noise,
            sigma=gen.sigma_background_noise
        )
        noisy_img = rand_noise + img
        noisy_img /= np.max(noisy_img)
        return noisy_img

    model = backend.load(modelpath)
    gen = backend.load_metadata(modelpath, psf_shape=3*[model.input_shape[2]])

    p2v = utils.peak2valley(phi, na=na, wavelength=gen.lam_detection)
    y_pred = pd.DataFrame([], columns=['sample'])
    y_true = pd.DataFrame([], columns=['sample'])

    w = Wavefront(phi, lam_detection=gen.lam_detection)
    kernel = gen.single_psf(
        phi=w,
        normed=True,
        noise=False,
        meta=False,
    )

    k = np.where(phi > 0)[0]
    for isize in tqdm([0, 1, 2, 3, 4, 5], desc=f"Evaluate different sizes [mode #{k}]"):
        inputs = np.array([sim_beads(kernel) for i in range(n_samples)])

        preds, stdev = backend.bootstrap_predict(
            model=model,
            inputs=inputs,
            psfgen=gen,
            n_samples=10,
            no_phase=True,
            batch_size=batch_size,
        )

        p = pd.DataFrame([utils.peak2valley(i, na=na, wavelength=gen.lam_detection) for i in preds], columns=['sample'])
        y_pred = y_pred.append(p, ignore_index=True)

        y = pd.DataFrame([p2v for i in preds], columns=['sample'])
        y['object_size'] = 1 if isize == 0 else isize * 2
        y_true = y_true.append(y, ignore_index=True)

    return (y_pred, y_true)


def evaluate_modes(model: Path, n_modes: int = 55):
    outdir = model.with_suffix('') / 'evalmodes'
    outdir.mkdir(parents=True, exist_ok=True)

    waves = np.arange(1e-5, .75, step=.05)
    aberrations = np.zeros((len(waves), n_modes))

    for i in trange(5, n_modes):
        savepath = outdir / f"m{i}"

        classes = aberrations.copy()
        classes[:, i] = waves

        job = partial(eval_mode, modelpath=model, npoints=1, n_samples=5)
        preds, ys = zip(*utils.multiprocess(job, classes, cores=-1))
        y_true = pd.DataFrame([], columns=['sample']).append(ys, ignore_index=True)
        y_pred = pd.DataFrame([], columns=['sample']).append(preds, ignore_index=True)

        error = np.abs(y_true - y_pred)
        error = pd.DataFrame(error, columns=['sample'])

        bins = np.arange(0, 10.25, .25)
        df = pd.DataFrame(
            zip(y_true['sample'], error['sample'], y_true['object_size']),
            columns=['aberration', 'error', 'object_size']
        )
        df['bins'] = pd.cut(df['aberration'], bins, labels=bins[1:], include_lowest=True)

        means = pd.pivot_table(df, values='error', index='bins', columns='object_size', aggfunc=np.mean)
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

        cbar.ax.set_ylabel(rf'Average peak-to-peak residuals ($\lambda = 510~nm$)')
        cbar.ax.set_title(r'$\lambda$')
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.yaxis.set_label_position('left')

        ax.set_xlabel(f'Diameter of the simulated object (pixels)')
        ax.set_xlim(1, 10)
        ax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

        ax.set_ylabel(
            'Average Peak-to-peak aberration'
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
        emb = gen.embedding(psf=kernel, no_phase=True)

        ax_xy.imshow(emb[0], vmin=0, vmax=2, cmap='Spectral_r')
        ax_xz.imshow(emb[1], vmin=0, vmax=2, cmap='Spectral_r')
        ax_yz.imshow(emb[2], vmin=0, vmax=2, cmap='Spectral_r')
        ax_wavevfront.imshow(w.wave(size=100), vmin=-1, vmax=1, cmap='Spectral_r')

        for a, t in zip([ax_xy, ax_xz, ax_yz, ax_wavevfront], ['XY', 'XZ', 'YZ', 'Wavefront']):
            a.axis('off')
            a.set_title(t)

        plt.tight_layout()

        plt.savefig(f'{savepath}.pdf', bbox_inches='tight', pad_inches=.25)
        plt.savefig(f'{savepath}.png', dpi=300, bbox_inches='tight', pad_inches=.25)


def eval_bin(
    datapath,
    modelpath,
    samplelimit: int = 1,
    na: float = 1.0,
    input_coverage: float = 1.0,
    no_phase: bool = False,
    batch_size: int = 100
):
    model = backend.load(modelpath)
    gen = backend.load_metadata(
        modelpath,
        snr=1000,
        bimodal=True,
        rotate=True,
        batch_size=batch_size,
        psf_shape=3*[model.input_shape[2]]
    )

    val = data_utils.collect_dataset(
        datapath,
        samplelimit=samplelimit,
        no_phase=no_phase,
        modes=gen.n_modes,
        embedding=gen.embedding_option,
    )

    y_pred = pd.DataFrame([], columns=['sample'])
    y_true = pd.DataFrame([], columns=['sample'])

    for inputs, ys in val.batch(100):
        if input_coverage != 1.:
            mode = np.mean([np.abs(st.mode(s, axis=None).mode[0]) for s in inputs])
            inputs = resize_with_crop_or_pad(inputs, crop_shape=[int(s*input_coverage) for s in gen.psf_shape])
            inputs = resize_with_crop_or_pad(inputs, crop_shape=gen.psf_shape, constant_values=mode)

        preds = backend.eval_sign(
            model=model,
            inputs=inputs if isinstance(inputs, np.ndarray) else inputs.numpy(),
            gen=gen,
            ys=ys if isinstance(ys, np.ndarray) else ys.numpy(),
            batch_size=batch_size,
            desc=f"Predictions for ({datapath})"
        )

        p = pd.DataFrame([utils.peak2valley(i, na=na, wavelength=gen.lam_detection) for i in preds], columns=['sample'])
        y_pred = y_pred.append(p, ignore_index=True)

        y = pd.DataFrame([utils.peak2valley(i, na=na, wavelength=gen.lam_detection) for i in ys.numpy()], columns=['sample'])
        y['snr'] = int(np.mean(list(
            map(int, str([s for s in datapath.parts if s.startswith('psnr_')][0]).lstrip('psnr_').split('-'))
        )))
        y_true = y_true.append(y, ignore_index=True)

    return (y_pred, y_true)


def evalheatmap(
    modelpath: Path,
    datadir: Path,
    n_modes: int = 55,
    distribution: str = '/',
    samplelimit: Any = None,
    max_amplitude: float = .25,
    na: float = 1.0,
    input_coverage: float = 1.0,
    no_phase: bool = False,
    batch_size: int = 100
):
    gen = backend.load_metadata(
        modelpath,
        snr=1000,
        bimodal=True,
        rotate=True,
        batch_size=batch_size,
    )

    savepath = modelpath / f'evalheatmaps_{input_coverage}'
    savepath.mkdir(parents=True, exist_ok=True)

    if distribution != '/':
        savepath = Path(f'{savepath}/{distribution}_na_{str(na).replace("0.", "p")}')
    else:
        savepath = Path(f'{savepath}/na_{str(na).replace("0.", "p")}')

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
           and gen.embedding_option in str(c)
           and f"z{int(n_modes)}" in str(c)
           and float(str([s for s in c.parts if s.startswith('amp_')][0]).split('-')[-1].replace('p', '.')) <= max_amplitude
    ])
    logger.info(f"BINs: {[len(classes)]}")

    job = partial(
        eval_bin,
        modelpath=modelpath,
        samplelimit=samplelimit,
        na=na,
        input_coverage=input_coverage,
        no_phase=no_phase
    )
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

    cbar.ax.set_ylabel(rf'Average peak-to-peak residuals ($\lambda = {int(gen.lam_detection*1000)}~nm$)')
    cbar.ax.set_title(r'$\lambda$')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('left')

    ax.set_xlabel(f'Peak signal-to-noise ratio')
    ax.set_xlim(0, 100)
    ax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

    ax.set_ylabel(
        'Average Peak-to-peak aberration'
        rf'($\lambda = {int(gen.lam_detection*1000)}~nm$)'
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


def evaldistbin(
    datapath: Path,
    modelpath: Path,
    samplelimit: Any = None,
    na: float = 1.0,
    no_phase: bool = False,
    input_coverage: float = 1.0,
    batch_size: int = 100
):
    model = backend.load(modelpath)
    gen = backend.load_metadata(
        modelpath,
        snr=1000,
        bimodal=True,
        rotate=True,
        batch_size=batch_size,
        psf_shape=3*[model.input_shape[2]]
    )

    val = data_utils.collect_dataset(
        datapath,
        samplelimit=samplelimit,
        no_phase=no_phase,
        modes=gen.n_modes,
        embedding=gen.embedding_option,
    )

    y_true = pd.DataFrame([], columns=['dist', 'sample'])
    y_pred = pd.DataFrame([], columns=['dist', 'sample'])

    for inputs, ys in val.batch(100):
        if input_coverage != 1.:
            mode = np.mean([np.abs(st.mode(s, axis=None).mode[0]) for s in inputs])
            inputs = resize_with_crop_or_pad(inputs, crop_shape=[int(s*input_coverage) for s in gen.psf_shape])
            inputs = resize_with_crop_or_pad(inputs, crop_shape=gen.psf_shape, constant_values=mode)

        preds = backend.eval_sign(
            model=model,
            inputs=inputs if isinstance(inputs, np.ndarray) else inputs.numpy(),
            gen=gen,
            ys=ys if isinstance(ys, np.ndarray) else ys.numpy(),
            batch_size=batch_size,
            desc=f"Predictions for ({datapath})"
        )

        p = pd.DataFrame([utils.peak2valley(i, na=na, wavelength=gen.lam_detection) for i in preds], columns=['sample'])
        y_pred = y_pred.append(p, ignore_index=True)

        y = pd.DataFrame([utils.peak2valley(i, na=na, wavelength=gen.lam_detection) for i in ys.numpy()], columns=['sample'])
        y['dist'] = [
            utils.mean_min_distance(np.squeeze(i), voxel_size=gen.voxel_size)
            for i in inputs
        ]
        y_true = y_true.append(y, ignore_index=True)

    return (y_pred, y_true)


def distheatmap(
    modelpath: Path,
    datadir: Path,
    distribution: str = '/',
    n_modes: int = 55,
    max_amplitude: float = .25,
    na: float = 1.0,
    no_phase: bool = False,
    psnr: tuple = (21, 30),
    num_neighbor: Any = None,
    samplelimit: Any = None,
    input_coverage: float = 1.0,
    batch_size: int = 100
):
    gen = backend.load_metadata(
        modelpath,
        snr=1000,
        bimodal=True,
        rotate=True,
        batch_size=batch_size,
    )

    if num_neighbor is not None:
        savepath = modelpath / f'distheatmaps_neighbor_{num_neighbor}_{input_coverage}'
    else:
        savepath = modelpath / f'distheatmaps_{input_coverage}'

    savepath.mkdir(parents=True, exist_ok=True)

    if distribution != '/':
        savepath = Path(f'{savepath}/{distribution}_na_{str(na).replace("0.", "p")}')
    else:
        savepath = Path(f'{savepath}/na_{str(na).replace("0.", "p")}')

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'xtick.major.pad': 10
    })

    if num_neighbor is not None:
        classes = sorted([
            c for c in Path(datadir).rglob('*/')
            if c.is_dir()
               and len(list(c.glob('*.tif'))) > 0
               and f'psnr_{psnr[0]}-{psnr[1]}' in str(c)
               and distribution in str(c)
               and gen.embedding_option in str(c)
               and f"z{int(n_modes)}" in str(c)
               and f"npoints_{num_neighbor}" in str(c)
               and float(str([s for s in c.parts if s.startswith('amp_')][0]).split('-')[-1].replace('p', '.')) <= max_amplitude
        ])
    else:
        classes = sorted([
            c for c in Path(datadir).rglob('*/')
            if c.is_dir()
               and f'psnr_{psnr[0]}-{psnr[1]}' in str(c)
               and len(list(c.glob('*.tif'))) > 0
               and distribution in str(c)
               and gen.embedding_option in str(c)
               and float(str([s for s in c.parts if s.startswith('amp_')][0]).split('-')[-1].replace('p', '.')) <= max_amplitude
        ])
    logger.info(f"BINs: {[len(classes)]}")

    job = partial(
        evaldistbin,
        modelpath=modelpath,
        samplelimit=samplelimit,
        na=na,
        no_phase=no_phase,
        input_coverage=input_coverage,
    )
    preds, ys = zip(*utils.multiprocess(job, classes))
    y_true = pd.DataFrame([], columns=['sample']).append(ys, ignore_index=True)
    y_pred = pd.DataFrame([], columns=['sample']).append(preds, ignore_index=True)

    error = np.abs(y_true - y_pred)
    error = pd.DataFrame(error, columns=['sample'])

    bins = np.arange(0, 10.25, .25)
    df = pd.DataFrame(zip(y_true['sample'], error['sample'], y_true['dist']), columns=['aberration', 'error', 'dist'])
    df['bins'] = pd.cut(df['aberration'], bins, labels=bins[1:], include_lowest=True)

    means = pd.pivot_table(df, values='error', index='bins', columns='dist', aggfunc=np.mean)
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

    cbar.ax.set_ylabel(rf'Average peak-to-peak residuals ($\lambda = {int(gen.lam_detection*1000)}~nm$)')
    cbar.ax.set_title(r'$\lambda$')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('left')

    ax.set_xlabel(rf'Average distance to nearest neighbor (microns)')
    ax.set_xlim(0, 4)
    ax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

    ax.set_ylabel(
        'Average Peak-to-peak aberration'
        rf'($\lambda = {int(gen.lam_detection*1000)}~nm$)'
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


def evaldensitybin(
    datapath: Path,
    modelpath: Path,
    samplelimit: Any = None,
    na: float = 1.0,
    no_phase: bool = False,
    input_coverage: float = 1.0,
    batch_size: int = 100
):
    model = backend.load(modelpath)
    gen = backend.load_metadata(
        modelpath,
        snr=1000,
        bimodal=True,
        rotate=True,
        batch_size=batch_size,
        psf_shape=3*[model.input_shape[2]]
    )

    val = data_utils.collect_dataset(
        datapath,
        samplelimit=samplelimit,
        no_phase=no_phase,
        modes=gen.n_modes,
        embedding=gen.embedding_option,
    )

    y_true = pd.DataFrame([], columns=['neighbors', 'dist', 'sample'])
    y_pred = pd.DataFrame([], columns=['neighbors', 'dist', 'sample'])

    for inputs, ys in val.batch(100):
        if input_coverage != 1.:
            mode = np.mean([np.abs(st.mode(s, axis=None).mode[0]) for s in inputs])
            inputs = resize_with_crop_or_pad(inputs, crop_shape=[int(s*input_coverage) for s in gen.psf_shape])
            inputs = resize_with_crop_or_pad(inputs, crop_shape=gen.psf_shape, constant_values=mode)

        preds = backend.eval_sign(
            model=model,
            inputs=inputs if isinstance(inputs, np.ndarray) else inputs.numpy(),
            gen=gen,
            ys=ys if isinstance(ys, np.ndarray) else ys.numpy(),
            batch_size=batch_size,
            desc=f"Predictions for ({datapath})"
        )

        p = pd.DataFrame([utils.peak2valley(i, na=na, wavelength=gen.lam_detection) for i in preds], columns=['sample'])
        y_pred = y_pred.append(p, ignore_index=True)

        y = pd.DataFrame([utils.peak2valley(i, na=na, wavelength=gen.lam_detection) for i in ys.numpy()], columns=['sample'])
        y['dist'] = [
            utils.mean_min_distance(np.squeeze(i), voxel_size=gen.voxel_size)
            for i in inputs
        ]
        y['neighbors'] = int(str([s for s in datapath.parts if s.startswith('npoints_')][0]).lstrip('npoints_'))
        y_true = y_true.append(y, ignore_index=True)

    return (y_pred, y_true)


def densityheatmap(
    modelpath: Path,
    datadir: Path,
    n_modes: int = 55,
    distribution: str = '/',
    max_amplitude: float = .25,
    na: float = 1.0,
    no_phase: bool = False,
    psnr: tuple = (21, 30),
    samplelimit: Any = None,
    input_coverage: float = 1.0,
    batch_size: int = 100
):
    gen = backend.load_metadata(
        modelpath,
        snr=1000,
        bimodal=True,
        rotate=True,
        batch_size=batch_size,
    )

    savepath = modelpath / f'densityheatmaps_{input_coverage}'
    savepath.mkdir(parents=True, exist_ok=True)

    if distribution != '/':
        savepath = Path(f'{savepath}/{distribution}_na_{str(na).replace("0.", "p")}')
    else:
        savepath = Path(f'{savepath}/na_{str(na).replace("0.", "p")}')

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
           and gen.embedding_option in str(c)
           and f"z{int(n_modes)}" in str(c)
           and float(str([s for s in c.parts if s.startswith('amp_')][0]).split('-')[-1].replace('p', '.')) <= max_amplitude
    ])
    logger.info(f"BINs: {[len(classes)]}")

    job = partial(
        evaldensitybin,
        modelpath=modelpath,
        samplelimit=samplelimit,
        na=na,
        no_phase=no_phase,
        input_coverage=input_coverage,
    )
    preds, ys = zip(*utils.multiprocess(job, classes))
    y_true = pd.DataFrame([], columns=['sample']).append(ys, ignore_index=True)
    y_pred = pd.DataFrame([], columns=['sample']).append(preds, ignore_index=True)

    error = np.abs(y_true - y_pred)
    error = pd.DataFrame(error, columns=['sample'])

    bins = np.arange(0, 10.25, .25)
    df = pd.DataFrame(
        zip(y_true['sample'], error['sample'], y_true['neighbors'], y_true['dist']),
        columns=['aberration', 'error', 'neighbors', 'dist']
    )
    df['bins'] = pd.cut(df['aberration'], bins, labels=bins[1:], include_lowest=True)

    means = pd.pivot_table(df, values='error', index='bins', columns='neighbors', aggfunc=np.mean)
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

    cbar.ax.set_ylabel(rf'Average peak-to-peak residuals ($\lambda = {int(gen.lam_detection*1000)}~nm$)')
    cbar.ax.set_title(r'$\lambda$')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('left')

    ax.set_xlabel(rf'Number of points')
    ax.set_xticks(np.arange(0, 35, 5))
    ax.set_xlim(1, 30)
    ax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

    ax.set_ylabel(
        'Average Peak-to-peak aberration'
        rf'($\lambda = {int(gen.lam_detection*1000)}~nm$)'
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


def iter_eval_bin_with_reference(
    datapath,
    modelpath,
    psnr: tuple = (21, 30),
    niter: int = 5,
    samplelimit: int = 1,
    na: float = 1.0,
    input_coverage: float = 1.0,
    no_phase: bool = False,
    batch_size: int = 100
):

    model = backend.load(modelpath)
    gen = backend.load_metadata(
        modelpath,
        snr=1000,
        bimodal=True,
        rotate=True,
        batch_size=batch_size,
        psf_shape=3*[model.input_shape[2]]
    )

    val = data_utils.collect_dataset(
        datapath,
        samplelimit=samplelimit,
        no_phase=no_phase,
        modes=gen.n_modes,
        embedding=gen.embedding_option,
    )
    val = np.array(list(val.take(-1)))

    inputs = np.array([i.numpy() for i in val[:, 0]])
    ys = np.array([i.numpy() for i in val[:, 1]])

    y_pred = pd.DataFrame.from_dict({
        'id': np.arange(ys.shape[0], dtype=int),
        'niter': np.zeros(ys.shape[0], dtype=int),
        'residuals': np.zeros(ys.shape[0], dtype=int)
    })

    y_true = pd.DataFrame.from_dict({
        'id': np.arange(ys.shape[0], dtype=int),
        'niter': np.zeros(ys.shape[0], dtype=int),
        'residuals': [utils.peak2valley(i, na=na, wavelength=gen.lam_detection) for i in ys]
    })

    reference = backend.beads(
        gen=gen,
        object_size=0,
        num_objs=np.random.randint(1, 10)
    )
    snr = gen._randuniform(psnr)

    for k in range(1, niter+1):
        for i in range(inputs.shape[0]):
            kernel = gen.single_psf(
                phi=ys[i],
                normed=True,
                noise=False,
                meta=False,
            )
            img = utils.fftconvolution(sample=reference, kernel=kernel)
            img *= snr ** 2

            rand_noise = gen._random_noise(
                image=img,
                mean=gen.mean_background_noise,
                sigma=gen.sigma_background_noise
            )
            noisy_img = rand_noise + img
            noisy_img /= np.max(noisy_img)
            inputs[i] = noisy_img[..., np.newaxis]

        if input_coverage != 1.:
            mode = np.mean([np.abs(st.mode(s, axis=None).mode[0]) for s in inputs])
            inputs = resize_with_crop_or_pad(inputs, crop_shape=[int(s*input_coverage) for s in gen.psf_shape])
            inputs = resize_with_crop_or_pad(inputs, crop_shape=gen.psf_shape, constant_values=mode)

        preds = backend.eval_sign(
            model=model,
            inputs=inputs,
            gen=gen,
            ys=ys,
            batch_size=batch_size,
            reference=reference,
            desc=f"Predictions for ({datapath})"
        )

        p = pd.DataFrame([utils.peak2valley(i, na=na, wavelength=gen.lam_detection) for i in preds], columns=['residuals'])
        p['niter'] = k
        p['id'] = np.arange(inputs.shape[0], dtype=int)
        y_pred = y_pred.append(p, ignore_index=True)

        y = pd.DataFrame([utils.peak2valley(i, na=na, wavelength=gen.lam_detection) for i in ys], columns=['residuals'])
        y['niter'] = k
        y['id'] = np.arange(inputs.shape[0], dtype=int)
        y_true = y_true.append(y, ignore_index=True)

        # setup next iter
        res = ys - preds
        ys = res

    return (y_pred, y_true)


def iterheatmap(
    modelpath: Path,
    datadir: Path,
    psnr: tuple = (21, 30),
    niter: int = 5,
    n_modes: int = 55,
    distribution: str = '/',
    samplelimit: Any = None,
    max_amplitude: float = .25,
    na: float = 1.0,
    input_coverage: float = 1.0,
    no_phase: bool = False,
    batch_size: int = 100
):
    gen = backend.load_metadata(
        modelpath,
        snr=1000,
        bimodal=True,
        rotate=True,
        batch_size=batch_size,
    )

    savepath = modelpath / f'iterheatmap_{input_coverage}'
    savepath.mkdir(parents=True, exist_ok=True)

    if distribution != '/':
        savepath = Path(f'{savepath}/{distribution}_na_{str(na).replace("0.", "p")}')
    else:
        savepath = Path(f'{savepath}/na_{str(na).replace("0.", "p")}')

    classes = sorted([
        c for c in Path(datadir).rglob('*/')
        if c.is_dir()
           and len(list(c.glob('*.tif'))) > 0
           and f'psnr_{psnr[0]}-{psnr[1]}' in str(c)
           and distribution in str(c)
           and gen.embedding_option in str(c)
           and f"z{int(n_modes)}" in str(c)
           and float(str([s for s in c.parts if s.startswith('amp_')][0]).split('-')[-1].replace('p', '.')) <= max_amplitude
    ])

    job = partial(
        iter_eval_bin_with_reference,
        modelpath=modelpath,
        niter=niter,
        psnr=psnr,
        samplelimit=samplelimit,
        na=na,
        no_phase=no_phase,
        input_coverage=input_coverage
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

    cbar.ax.set_ylabel(rf'Average peak-to-peak residuals ($\lambda = {int(gen.lam_detection*1000)}~nm$)')
    cbar.ax.set_title(r'$\lambda$')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('left')

    ax.set_xlabel(f'Number of iterations')
    ax.set_xlim(0, niter)
    ax.set_xticks(range(niter+1))
    ax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

    ax.set_ylabel(
        'Average Peak-to-peak aberration '
        rf'($\lambda = {int(gen.lam_detection*1000)}~nm$)'
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

