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
from tifffile import imread
import tensorflow as tf
from preprocessing import resize_with_crop_or_pad
from scipy import stats as st
import raster_geometry as rg

import utils
import data_utils
import backend

from synthetic import SyntheticPSF
from wavefront import Wavefront

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
tf.get_logger().setLevel(logging.ERROR)


def beads(
    gen: SyntheticPSF,
    kernel: np.ndarray,
    psnr: tuple = (21, 30),
    object_size: float = 0,
    num_objs: int = 1,
    radius: float = .45,
):
    snr = gen._randuniform(psnr)
    reference = np.zeros(gen.psf_shape)

    for i in range(num_objs):
        if object_size > 0:
            reference += rg.sphere(
                shape=gen.psf_shape,
                radius=object_size,
                position=np.random.uniform(low=.2, high=.8, size=3)
            ).astype(np.float) * np.random.random()
        else:
            reference[
                np.random.randint(int(gen.psf_shape[0] * (.5 - radius)), int(gen.psf_shape[0] * (.5 + radius))),
                np.random.randint(int(gen.psf_shape[1] * (.5 - radius)), int(gen.psf_shape[1] * (.5 + radius))),
                np.random.randint(int(gen.psf_shape[2] * (.5 - radius)), int(gen.psf_shape[2] * (.5 + radius)))
            ] += np.random.random()

    reference /= np.max(reference)
    img = utils.fftconvolution(reference, kernel)
    snr = gen._randuniform(snr)
    img *= snr ** 2

    rand_noise = gen._random_noise(
        image=img,
        mean=gen.mean_background_noise,
        sigma=gen.sigma_background_noise
    )
    noisy_img = rand_noise + img
    noisy_img /= np.max(noisy_img)

    return noisy_img


def eval_mode(
    phi,
    model,
    npoints: int = 1,
    n_samples: int = 10,
    psnr: tuple = (21, 30),
    na: float = 1.0
):
    p2p = utils.peak_aberration(phi, na=na)
    y_pred = pd.DataFrame([], columns=['sample'])
    y_true = pd.DataFrame([], columns=['sample'])

    gen = backend.load_metadata(model, psf_shape=(64, 64, 64))
    model = backend.load(model)
    w = Wavefront(phi, lam_detection=gen.lam_detection)
    kernel = gen.single_psf(
        phi=w,
        zplanes=0,
        normed=True,
        noise=False,
        augmentation=False,
        meta=False,
    )

    k = np.where(phi > 0)[0]
    for isize in tqdm([0, 1, 2, 3, 4, 5], desc=f"Evaluate different sizes [mode #{k}]"):
        inputs = np.array([
            beads(gen=gen, kernel=kernel, psnr=psnr, object_size=isize, num_objs=npoints)
            for i in range(n_samples)
        ])

        preds, stdev = backend.bootstrap_predict(
            model=model,
            inputs=inputs,
            psfgen=gen,
            n_samples=10,
            no_phase=True,
            batch_size=100,
        )

        p = pd.DataFrame([utils.peak_aberration(i, na=na) for i in preds], columns=['sample'])
        y_pred = y_pred.append(p, ignore_index=True)

        y = pd.DataFrame([p2p for i in preds], columns=['sample'])
        y['object_size'] = isize
        y_true = y_true.append(y, ignore_index=True)

    return (y_pred, y_true)


def evaluate_modes(model: Path, n_modes: int = 55):
    outdir = model.with_suffix('') / 'modes'
    outdir.mkdir(parents=True, exist_ok=True)

    waves = np.arange(1e-5, .25, step=.01)
    aberrations = np.zeros((len(waves), n_modes))

    for i in trange(5, n_modes):
        savepath = outdir / f"m{i}"

        classes = aberrations.copy()
        classes[:, i] = waves

        job = partial(eval_mode, model=model, npoints=1, n_samples=10)
        preds, ys = zip(*utils.multiprocess(job, list(classes), cores=-1))
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

        cbar.ax.set_ylabel(rf'Average peak-to-peak residuals ($\lambda = 510~nm$)')
        cbar.ax.set_title(r'$\lambda$')
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.yaxis.set_label_position('left')

        ax.set_xlabel(f'Radius of the simulated object (pixels)')
        ax.set_xlim(0, 5)
        ax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

        ax.set_ylabel(
            'Average Peak-to-peak aberration'
            rf'($\lambda = 510~nm$)'
        )
        ax.set_yticks(np.arange(0, 6, .5), minor=True)
        ax.set_yticks(np.arange(0, 6, 1))
        ax.set_ylim(.25, 5)

        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.tight_layout()

        plt.savefig(f'{savepath}.pdf', bbox_inches='tight', pad_inches=.25)
        plt.savefig(f'{savepath}.png', dpi=300, bbox_inches='tight', pad_inches=.25)


def eval_bin(
    datapath,
    modelpath,
    samplelimit,
    na,
    psf_type,
    x_voxel_size,
    y_voxel_size,
    z_voxel_size,
    wavelength,
    input_coverage,
    no_phase,
    modes,
):
    model = backend.load(modelpath)
    gen = SyntheticPSF(
        n_modes=modes,
        amplitude_ranges=(-.25, .25),
        psf_shape=(64, 64, 64),
        psf_type=psf_type,
        lam_detection=wavelength,
        x_voxel_size=x_voxel_size,
        y_voxel_size=y_voxel_size,
        z_voxel_size=z_voxel_size,
        batch_size=100,
        snr=100,
        max_jitter=0,
        cpu_workers=-1,
    )

    val = data_utils.load_dataset(datapath, samplelimit=samplelimit)
    func = partial(data_utils.get_sample, no_phase=no_phase)
    val = val.map(lambda x: tf.py_function(func, [x], [tf.float32, tf.float32]))

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
            batch_size=100,
            desc=f"Predictions for ({datapath})"
        )

        p = pd.DataFrame([utils.peak_aberration(i, na=na) for i in preds], columns=['sample'])
        y_pred = y_pred.append(p, ignore_index=True)

        y = pd.DataFrame([utils.peak_aberration(i, na=na) for i in ys.numpy()], columns=['sample'])
        y['snr'] = int(np.mean(list(
            map(int, str([s for s in datapath.parts if s.startswith('psnr_')][0]).lstrip('psnr_').split('-'))
        )))
        y_true = y_true.append(y, ignore_index=True)

    return (y_pred, y_true)


def evalheatmap(
    modelpath: Path,
    datadir: Path,
    distribution: str = '/',
    samplelimit: Any = None,
    max_amplitude: float = .25,
    modes: int = 55,
    na: float = 1.0,
    psf_type: str = 'widefield',
    x_voxel_size: float = .15,
    y_voxel_size: float = .15,
    z_voxel_size: float = .6,
    wavelength: float = .510,
    input_coverage: float = 1.0,
    no_phase: bool = False,
):
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
           and float(str([s for s in c.parts if s.startswith('amp_')][0]).split('-')[-1].replace('p', '.')) <= max_amplitude
    ])
    logger.info(f"BINs: {[len(classes)]}")

    job = partial(
        eval_bin,
        modelpath=modelpath,
        samplelimit=samplelimit,
        na=na,
        psf_type=psf_type,
        x_voxel_size=x_voxel_size,
        y_voxel_size=y_voxel_size,
        z_voxel_size=z_voxel_size,
        wavelength=wavelength,
        modes=modes,
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

    cbar.ax.set_ylabel(rf'Average peak-to-peak residuals ($\lambda = {int(wavelength*1000)}~nm$)')
    cbar.ax.set_title(r'$\lambda$')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('left')

    ax.set_xlabel(f'Peak signal-to-noise ratio')
    ax.set_xlim(0, 100)
    ax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

    ax.set_ylabel(
        'Average Peak-to-peak aberration'
        rf'($\lambda = {int(wavelength*1000)}~nm$)'
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


def iter_eval_bin(
    datapath,
    modelpath,
    niter,
    psnr,
    samples,
    na,
    psf_type,
    x_voxel_size,
    y_voxel_size,
    z_voxel_size,
    wavelength,
    input_coverage,
    no_phase,
    modes
):
    model = backend.load(modelpath)
    gen = SyntheticPSF(
        n_modes=modes,
        amplitude_ranges=(-.25, .25),
        psf_shape=(64, 64, 64),
        psf_type=psf_type,
        lam_detection=wavelength,
        x_voxel_size=x_voxel_size,
        y_voxel_size=y_voxel_size,
        z_voxel_size=z_voxel_size,
        batch_size=100,
        snr=psnr,
        max_jitter=0,
        cpu_workers=-1,
    )

    val = data_utils.load_dataset(datapath, samplelimit=samples)
    func = partial(data_utils.get_sample, no_phase=no_phase)
    val = val.map(lambda x: tf.py_function(func, [x], [tf.float32, tf.float32]))
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
        if input_coverage != 1.:
            mode = np.mean([np.abs(st.mode(s, axis=None).mode[0]) for s in inputs])
            inputs = resize_with_crop_or_pad(inputs, crop_shape=[int(s*input_coverage) for s in gen.psf_shape])
            inputs = resize_with_crop_or_pad(inputs, crop_shape=gen.psf_shape, constant_values=mode)

        preds = backend.eval_sign(
            model=model,
            inputs=inputs,
            gen=gen,
            ys=ys,
            batch_size=samples,
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


def iter_eval_bin_with_reference(
    datapath,
    modelpath,
    reference: Any = 'random',
    psnr: tuple = (21, 30),
    niter: int = 5,
    samples: int = 1,
    na: float = 1.0,
    psf_type: str = 'widefield',
    x_voxel_size: float = .15,
    y_voxel_size: float = .15,
    z_voxel_size: float = .6,
    wavelength: float = .510,
    num_neighbor: int = 5,
    radius: float = .45,
    modes: int = 55,
    input_coverage: float = 1.0,
    no_phase: bool = False,
):
    model = backend.load(modelpath)
    gen = SyntheticPSF(
        n_modes=modes,
        amplitude_ranges=(-.25, .25),
        psf_shape=(64, 64, 64),
        psf_type=psf_type,
        lam_detection=wavelength,
        x_voxel_size=x_voxel_size,
        y_voxel_size=y_voxel_size,
        z_voxel_size=z_voxel_size,
        batch_size=100,
        snr=psnr,
        max_jitter=0,
        cpu_workers=-1,
    )
    if num_neighbor is None:
        num_neighbor = np.random.randint(low=1, high=6)

    if reference == 'random':
        snr = gen._randuniform(psnr)
        reference = np.zeros(gen.psf_shape)
        for i in range(num_neighbor):
            reference[
                np.random.randint(int(gen.psf_shape[0] * (.5 - radius)), int(gen.psf_shape[0] * (.5 + radius))),
                np.random.randint(int(gen.psf_shape[1] * (.5 - radius)), int(gen.psf_shape[1] * (.5 + radius))),
                np.random.randint(int(gen.psf_shape[2] * (.5 - radius)), int(gen.psf_shape[2] * (.5 + radius)))
            ] = snr ** 2
        reference *= snr ** 2

        rand_noise = gen._random_noise(
            image=reference,
            mean=gen.mean_background_noise,
            sigma=gen.sigma_background_noise
        )
        reference += rand_noise
        reference /= np.max(reference)
        reference = reference[..., np.newaxis]

    elif isinstance(reference, Path):
        reference = imread(reference).astype(np.float)
        reference /= np.max(reference)
        reference = np.expand_dims(reference, axis=-1)

    val = data_utils.load_dataset(datapath, samplelimit=samples)
    func = partial(data_utils.get_sample, no_phase=no_phase)
    val = val.map(lambda x: tf.py_function(func, [x], [tf.float32, tf.float32]))
    val = np.array(list(val.take(-1)))

    inputs = np.array([i.numpy() for i in val[:, 0]])
    ys = np.array([i.numpy() for i in val[:, 1]])

    for i in range(inputs.shape[0]):
        kernel = gen.single_psf(
            phi=Wavefront(ys[i], lam_detection=wavelength),
            zplanes=0,
            normed=True,
            noise=False,
            augmentation=False,
            meta=False,
        )
        inputs[i] = kernel[..., np.newaxis]

    inputs = utils.fftconvolution(reference, inputs)

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
        if input_coverage != 1.:
            mode = np.mean([np.abs(st.mode(s, axis=None).mode[0]) for s in inputs])
            inputs = resize_with_crop_or_pad(inputs, crop_shape=[int(s*input_coverage) for s in gen.psf_shape])
            inputs = resize_with_crop_or_pad(inputs, crop_shape=gen.psf_shape, constant_values=mode)

        preds = backend.eval_sign(
            model=model,
            inputs=inputs,
            gen=gen,
            ys=ys,
            batch_size=samples,
            reference=reference,
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
            noise=False,
            augmentation=False,
            meta=False
        )

        inputs = np.expand_dims(np.stack(gen.batch(g, res), axis=0), -1)
        inputs = utils.fftconvolution(reference, inputs)
        ys = res

    return (y_pred, y_true)


def iterheatmap(
    modelpath: Path,
    datadir: Path,
    reference: Path = None,
    psnr: tuple = (21, 30),
    niter: int = 5,
    distribution: str = '/',
    samplelimit: Any = None,
    max_amplitude: float = .25,
    modes: int = 55,
    na: float = 1.0,
    psf_type: str = 'widefield',
    x_voxel_size: float = .15,
    y_voxel_size: float = .15,
    z_voxel_size: float = .6,
    wavelength: float = .510,
    input_coverage: float = 1.0,
    no_phase: bool = False,
):
    if reference is None or reference == 'unknown':
        savepath = modelpath / f'iterheatmaps_{input_coverage}'
    else:
        reference = Path(reference)
        savepath = modelpath / f'{reference.stem}_iterheatmaps_{input_coverage}'

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
           and float(str([s for s in c.parts if s.startswith('amp_')][0]).split('-')[-1].replace('p', '.')) <= max_amplitude
    ])
    logger.info(f"BINs: {[len(classes)]}")

    if reference is None:
        job = partial(
            iter_eval_bin,
            modelpath=modelpath,
            niter=niter,
            psnr=psnr,
            samples=samplelimit,
            na=na,
            psf_type=psf_type,
            x_voxel_size=x_voxel_size,
            y_voxel_size=y_voxel_size,
            z_voxel_size=z_voxel_size,
            wavelength=wavelength,
            input_coverage=input_coverage,
            modes=modes,
            no_phase=no_phase
        )
    else:
        job = partial(
            iter_eval_bin_with_reference,
            modelpath=modelpath,
            reference=reference,
            niter=niter,
            psnr=psnr,
            samples=samplelimit,
            na=na,
            modes=modes,
            psf_type=psf_type,
            x_voxel_size=x_voxel_size,
            y_voxel_size=y_voxel_size,
            z_voxel_size=z_voxel_size,
            wavelength=wavelength,
            no_phase=no_phase
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

    cbar.ax.set_ylabel(rf'Average peak-to-peak residuals ($\lambda = {int(wavelength*1000)}~nm$)')
    cbar.ax.set_title(r'$\lambda$')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('left')

    ax.set_xlabel(f'Number of iterations')
    ax.set_xlim(0, niter)
    ax.set_xticks(range(niter+1))
    ax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

    ax.set_ylabel(
        'Average Peak-to-peak aberration '
        rf'($\lambda = {int(wavelength*1000)}~nm$)'
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


def eval_roi(
    rois: np.array,
    modelpath: Path,
    psnr: tuple = (21, 30),
    na: float = 1.0,
    psf_type: str = 'widefield',
    x_voxel_size: float = .15,
    y_voxel_size: float = .15,
    z_voxel_size: float = .6,
    wavelength: float = .510,
    avg_dist: int = 10
):
    model = backend.load(modelpath)
    gen = SyntheticPSF(
        n_modes=55,
        psf_type=psf_type,
        lam_detection=wavelength,
        psf_shape=[64, 64, 64],
        z_voxel_size=x_voxel_size,
        y_voxel_size=y_voxel_size,
        x_voxel_size=z_voxel_size,
        batch_size=100,
        snr=psnr,
        max_jitter=0,
        cpu_workers=-1,
    )

    y_pred = pd.DataFrame([], columns=['sample'])

    for w in tqdm(range(rois.shape[0]), total=rois.shape[0], desc=f"ROIs[d{avg_dist}]"):
        reference = rois[w]
        inputs = reference / np.nanpercentile(reference, 99.99)
        inputs[inputs > 1] = 1

        preds, stdev = backend.bootstrap_predict(
            model,
            inputs[np.newaxis, :, :, :, np.newaxis],
            psfgen=gen,
            batch_size=1,
            n_samples=1,
        )

        p = pd.DataFrame([utils.peak_aberration(i, na=na) for i in preds], columns=['sample'])
        y_pred = y_pred.append(p, ignore_index=True)

    return y_pred


def evaldistbin(
    datapath: Path,
    modelpath: Path,
    samplelimit: Any = None,
    psnr: tuple = (21, 30),
    na: float = 1.0,
    modes: int = 55,
    psf_type: str = 'widefield',
    x_voxel_size: float = .15,
    y_voxel_size: float = .15,
    z_voxel_size: float = .6,
    wavelength: float = .510,
    no_phase: bool = False,
    input_coverage: float = 1.0
):

    model = backend.load(modelpath)
    gen = SyntheticPSF(
        n_modes=modes,
        amplitude_ranges=(-.25, .25),
        psf_shape=(64, 64, 64),
        psf_type=psf_type,
        lam_detection=wavelength,
        x_voxel_size=x_voxel_size,
        y_voxel_size=y_voxel_size,
        z_voxel_size=z_voxel_size,
        batch_size=100,
        snr=psnr,
        max_jitter=0,
        cpu_workers=-1,
    )

    val = data_utils.load_dataset(datapath, samplelimit=samplelimit)
    func = partial(data_utils.get_sample, no_phase=no_phase)
    val = val.map(lambda x: tf.py_function(func, [x], [tf.float32, tf.float32]))

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
            batch_size=100,
            desc=f"Predictions for ({datapath})"
        )

        p = pd.DataFrame([utils.peak_aberration(i, na=na) for i in preds], columns=['sample'])
        y_pred = y_pred.append(p, ignore_index=True)

        y = pd.DataFrame([utils.peak_aberration(i, na=na) for i in ys.numpy()], columns=['sample'])
        y['dist'] = [
            utils.mean_min_distance(np.squeeze(i), voxel_size=(z_voxel_size, y_voxel_size, x_voxel_size))
            for i in inputs
        ]
        y_true = y_true.append(y, ignore_index=True)

    return (y_pred, y_true)


def distheatmap(
    modelpath: Path,
    datadir: Path,
    distribution: str = '/',
    max_amplitude: float = .25,
    na: float = 1.0,
    modes: int = 55,
    psf_type: str = 'widefield',
    x_voxel_size: float = .15,
    y_voxel_size: float = .15,
    z_voxel_size: float = .6,
    wavelength: float = .510,
    no_phase: bool = False,
    psnr: tuple = (21, 30),
    num_neighbor: Any = None,
    samplelimit: Any = None,
    input_coverage: float = 1.0,
):
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
               and float(str([s for s in c.parts if s.startswith('amp_')][0]).split('-')[-1].replace('p', '.')) <= max_amplitude
        ])
    logger.info(f"BINs: {[len(classes)]}")

    job = partial(
        evaldistbin,
        modelpath=modelpath,
        samplelimit=samplelimit,
        na=na,
        psnr=psnr,
        psf_type=psf_type,
        x_voxel_size=x_voxel_size,
        y_voxel_size=y_voxel_size,
        z_voxel_size=z_voxel_size,
        modes=modes,
        wavelength=wavelength,
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

    cbar.ax.set_ylabel(rf'Average peak-to-peak residuals ($\lambda = {int(wavelength*1000)}~nm$)')
    cbar.ax.set_title(r'$\lambda$')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('left')

    ax.set_xlabel(rf'Average distance to nearest neighbor (microns)')
    ax.set_xlim(0, 4)
    ax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

    ax.set_ylabel(
        'Average Peak-to-peak aberration'
        rf'($\lambda = {int(wavelength*1000)}~nm$)'
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
    psnr: tuple = (21, 30),
    na: float = 1.0,
    modes: int = 55,
    psf_type: str = 'widefield',
    x_voxel_size: float = .15,
    y_voxel_size: float = .15,
    z_voxel_size: float = .6,
    wavelength: float = .510,
    no_phase: bool = False,
    input_coverage: float = 1.0
):

    model = backend.load(modelpath)
    gen = SyntheticPSF(
        n_modes=modes,
        amplitude_ranges=(-.25, .25),
        psf_shape=(64, 64, 64),
        psf_type=psf_type,
        lam_detection=wavelength,
        x_voxel_size=x_voxel_size,
        y_voxel_size=y_voxel_size,
        z_voxel_size=z_voxel_size,
        batch_size=100,
        snr=psnr,
        max_jitter=0,
        cpu_workers=-1,
    )

    val = data_utils.load_dataset(datapath, samplelimit=samplelimit)
    func = partial(data_utils.get_sample, no_phase=no_phase)
    val = val.map(lambda x: tf.py_function(func, [x], [tf.float32, tf.float32]))

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
            batch_size=100,
            desc=f"Predictions for ({datapath})"
        )

        p = pd.DataFrame([utils.peak_aberration(i, na=na) for i in preds], columns=['sample'])
        y_pred = y_pred.append(p, ignore_index=True)

        y = pd.DataFrame([utils.peak_aberration(i, na=na) for i in ys.numpy()], columns=['sample'])
        y['dist'] = [
            utils.mean_min_distance(np.squeeze(i), voxel_size=(z_voxel_size, y_voxel_size, x_voxel_size))
            for i in inputs
        ]
        y['neighbors'] = int(str([s for s in datapath.parts if s.startswith('npoints_')][0]).lstrip('npoints_'))
        y_true = y_true.append(y, ignore_index=True)

    return (y_pred, y_true)


def densityheatmap(
    modelpath: Path,
    datadir: Path,
    distribution: str = '/',
    max_amplitude: float = .25,
    na: float = 1.0,
    modes: int = 55,
    psf_type: str = 'widefield',
    x_voxel_size: float = .15,
    y_voxel_size: float = .15,
    z_voxel_size: float = .6,
    wavelength: float = .510,
    no_phase: bool = False,
    psnr: tuple = (21, 30),
    samplelimit: Any = None,
    input_coverage: float = 1.0,
):
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
           and float(str([s for s in c.parts if s.startswith('amp_')][0]).split('-')[-1].replace('p', '.')) <= max_amplitude
    ])
    logger.info(f"BINs: {[len(classes)]}")

    job = partial(
        evaldensitybin,
        modelpath=modelpath,
        samplelimit=samplelimit,
        na=na,
        psnr=psnr,
        psf_type=psf_type,
        x_voxel_size=x_voxel_size,
        y_voxel_size=y_voxel_size,
        z_voxel_size=z_voxel_size,
        modes=modes,
        wavelength=wavelength,
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

    cbar.ax.set_ylabel(rf'Average peak-to-peak residuals ($\lambda = {int(wavelength*1000)}~nm$)')
    cbar.ax.set_title(r'$\lambda$')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('left')

    ax.set_xlabel(rf'Number of points')
    ax.set_xticks(np.arange(0, 35, 5))
    ax.set_xlim(1, 30)
    ax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

    ax.set_ylabel(
        'Average Peak-to-peak aberration'
        rf'($\lambda = {int(wavelength*1000)}~nm$)'
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


def evalpoints(
    modelpath: Path,
    datadir: Path,
    psnr: tuple = (21, 30),
    niter: int = 5,
    distribution: str = '/',
    samplelimit: Any = None,
    max_amplitude: float = .25,
    modes: int = 55,
    na: float = 1.0,
    psf_type: str = 'widefield',
    x_voxel_size: float = .15,
    y_voxel_size: float = .15,
    z_voxel_size: float = .6,
    wavelength: float = .510,
    input_coverage: float = 1.0,
    no_phase: bool = False,
):
    savepath = modelpath / f'evalpoints_{input_coverage}'
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
           and float(str([s for s in c.parts if s.startswith('amp_')][0]).split('-')[-1].replace('p', '.')) <= max_amplitude
    ])

    job = partial(
        iter_eval_bin_with_reference,
        modelpath=modelpath,
        reference='random',
        niter=niter,
        psnr=psnr,
        samples=samplelimit,
        na=na,
        modes=modes,
        psf_type=psf_type,
        x_voxel_size=x_voxel_size,
        y_voxel_size=y_voxel_size,
        z_voxel_size=z_voxel_size,
        wavelength=wavelength,
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

    cbar.ax.set_ylabel(rf'Average peak-to-peak residuals ($\lambda = {int(wavelength*1000)}~nm$)')
    cbar.ax.set_title(r'$\lambda$')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('left')

    ax.set_xlabel(f'Number of iterations')
    ax.set_xlim(0, niter)
    ax.set_xticks(range(niter+1))
    ax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

    ax.set_ylabel(
        'Average Peak-to-peak aberration '
        rf'($\lambda = {int(wavelength*1000)}~nm$)'
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

