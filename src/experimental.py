from functools import partial
from pathlib import Path
from typing import Any

import ujson
import numpy as np
import pandas as pd
from tensorflow import config as tfc

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mtick
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as mcolors
from tifffile import imsave

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


def phase_retrieval(psf: Path, dx=.15, dz=.6, wavelength=.605, n_modes=60) -> list:
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


def create_dataset(data_path: Path):
    """
    Creates a JSON file mapping .tif scans to GTs

    Args:
        data_path: path to raw data directory
        noisy: a flag to add Gaussian noise to GT

    """
    files = sorted(data_path.rglob('*.tif'))
    zcoffs = utils.multiprocess(phase_retrieval, files, desc='Phase retrieval', cores=-1)
    data = dict(zip(files, zcoffs))

    with open(data_path / 'dataset.json', 'w') as fp:
        ujson.dump(data, fp, indent=4)


def correct(
    k: tuple,
    outdir: Path,
    dm_pattern: Path,
    input_shape: int,
    x_voxel_size: float,
    y_voxel_size: float,
    z_voxel_size: float,
):
    y, pred, path = k

    psf = preprocessing.prep_psf(path, image_size=input_shape)
    diff = y - pred

    dm = zernikies_to_actuators(pred, dm_pattern=dm_pattern)
    df = pd.DataFrame(dm)
    df.to_csv(f"{outdir}/{Path(path).stem}_correction_dmpattern.csv", index=False, header=False)

    y = Wavefront(y)
    pred = Wavefront(pred)
    diff = Wavefront(diff)

    psfargs = dict(
        amplitude_ranges=(-.2, .2),
        psf_shape=psf.shape,
        x_voxel_size=x_voxel_size,
        y_voxel_size=y_voxel_size,
        z_voxel_size=z_voxel_size,
        snr=100,
        max_jitter=0,
        batch_size=1
    )

    gen = SyntheticPSF(**psfargs)

    predicted_psf = gen.single_psf(pred, zplanes=0)
    corrected_psf = gen.single_psf(diff, zplanes=0)
    gt_psf = gen.single_psf(y, zplanes=0)

    vis.diagnostic_assessment(
        psf=psf,
        gt_psf=gt_psf,
        predicted_psf=predicted_psf,
        corrected_psf=corrected_psf,
        psnr=100,
        maxcounts=10000,
        y=y,
        pred=pred,
        save_path=Path(f"{outdir}/{Path(path).stem}"),
        display=False
    )


def matlab_comparison(
    args: tuple,
    wavelength: float = .605,
    psf_cmap: str = 'hot',
    wave_cmap: str = 'Spectral_r',
    gamma: float = .5,
):
    def plot_wavefront(iax, w, levels, label=''):
        mat = iax.contourf(
            w,
            levels=levels,
            cmap=wave_cmap,
            vmin=np.min(levels),
            vmax=np.max(levels),
            extend='both'
        )

        divider = make_axes_locatable(iax)
        top = divider.append_axes("top", size='30%', pad=0.2)
        top.hist(w.flatten(), bins=w.shape[0], color='grey')

        top.set_title(label)
        top.set_yticks([])
        top.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        top.spines['right'].set_visible(False)
        top.spines['top'].set_visible(False)
        top.spines['left'].set_visible(False)
        return mat

    def plot_psf_slice(xy, zy, zx, vol, label='', ylim=(None, None), xlim=(None, None), zlim=(None, None)):

        vol = vol ** gamma
        vol = np.nan_to_num(vol)

        mid_plane = vol.shape[0] // 2
        m = xy.imshow(vol[mid_plane, :, :], cmap=psf_cmap)
        zx.imshow(vol[:, mid_plane, :], cmap=psf_cmap)
        zy.imshow(vol[:, :, mid_plane].T, cmap=psf_cmap)

        xy.set_ylabel(label)

        xy.set_xlim(*xlim)
        xy.set_ylim(*ylim)

        zx.set_xlim(*xlim)
        zx.set_ylim(*zlim)

        zy.set_xlim(*ylim)
        zy.set_ylim(*zlim)

        return m

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    psfargs, psf, y_true, y_pred, save_path, psnr, zplanes = args
    y_matlab = np.array(phase_retrieval(Path(save_path).with_suffix('.tif')))

    if psf.ndim == 5:
        psf = np.squeeze(psf, axis=0)
        psf = np.squeeze(psf, axis=-1)
    elif psf.ndim == 4:
        psf = np.squeeze(psf, axis=-1)

    if not np.isscalar(psnr):
        psnr = psnr[0]

    y_true = Wavefront(y_true)
    y_pred = Wavefront(y_pred)
    y_matlab = Wavefront(y_matlab)

    mat_diff = Wavefront(y_true - y_matlab)
    net_diff = Wavefront(y_true - y_pred)

    step = .25
    y_wave = y_true.wave(size=100)

    vmax = round(np.max([
        np.abs(round(np.nanquantile(y_wave, .1), 2)),
        np.abs(round(np.nanquantile(y_wave, .9), 2))
    ])*4)/4
    vmax = .25 if vmax == 0.0 else vmax

    highcmap = plt.get_cmap('magma_r', 256)
    middlemap = plt.get_cmap('gist_gray', 256)
    lowcmap = plt.get_cmap('gist_earth_r', 256)

    ll = np.arange(-vmax, -.25+step, step)
    mm = [-.15, 0, .15]
    hh = np.arange(.25, vmax+step, step)
    mticks = np.concatenate((ll, mm, hh))

    levels = np.vstack((
        lowcmap(.66*ll/ll.min()),
        middlemap([.85, .95, 1, .95, .85]),
        highcmap(.66*hh/hh.max())
    ))
    wave_cmap = mcolors.ListedColormap(levels)

    fig = plt.figure(figsize=(10, 18))
    gs = fig.add_gridspec(21, 4)

    psfs = {
        'Input': [],
        'Ground truth': [],
        'Predicted': [],
        'Matlab': [],
        'Matlab_PH': [],
        'Net_PH': [],
    }
    for i, k in zip(range(0, 21, 3), psfs):
        psfs[k] = [fig.add_subplot(gs[i:i+3, j]) for j in range(3)]

    #cax = fig.add_subplot(gs[0:3, 3])
    ax_zcoff = fig.add_subplot(gs[-3:, :])

    for i, (k, axes), phi in zip(
        range(0, 24, 3),
        psfs.items(),
        [psf, y_true, y_pred, y_matlab, mat_diff, net_diff]
    ):
        if k == 'Input':
            plot_psf_slice(*axes, psf, label=k)
        else:
            p, y, psnr, zs = SyntheticPSF(
                amplitude_ranges=phi,
                psf_shape=psfargs['psf_shape'],
                x_voxel_size=psfargs['x_voxel_size'],
                y_voxel_size=psfargs['y_voxel_size'],
                z_voxel_size=psfargs['z_voxel_size'],
                snr=psfargs['snr'],
                max_jitter=psfargs['max_jitter']
            ).single_psf(
                phi, meta=True, zplanes=0, normed=True, noise=False, augmentation=False
            )

            plot_psf_slice(*axes, p, label=k)

            if k == 'Ground truth':
                y_wave = phi.wave(size=100)

                wax = fig.add_subplot(gs[i:i+3, 3])
                m = plot_wavefront(wax, y_wave, label='', levels=mticks)
                cbar = fig.colorbar(
                    m,
                    cax=fig.add_axes([.95, 0.25, 0.02, .5]),
                    fraction=0.046,
                    pad=0.04,
                    extend='both',
                    format=FormatStrFormatter("%.2g"),
                    # spacing='proportional',
                )
                cbar.ax.set_title(r'$\lambda$')
                cbar.ax.set_ylabel(f'$\lambda = {wavelength}~\mu m$')
                cbar.ax.yaxis.set_ticks_position('right')
                cbar.ax.yaxis.set_label_position('left')

            else:
                wax = fig.add_subplot(gs[i:i+3, 3])
                plot_wavefront(wax, phi.wave(size=100), label='', levels=mticks)

            wax.axis('off')

    psfs['Input'][0].set_title(f"PSNR: {psnr:.2f}")
    psfs['Input'][2].set_title(f"$\gamma$: {gamma:.2f}")
    psfs['Net_PH'][0].set_title('XY')
    psfs['Net_PH'][1].set_title('ZY')
    psfs['Net_PH'][2].set_title('ZX')

    # ax_zcoff.set_title('Zernike modes')
    ax_zcoff.plot(y_true.amplitudes_ansi_waves, '-o', color='C3', label='Ground truth')
    ax_zcoff.plot(y_pred.amplitudes_ansi_waves, '-o', color='C0', label='Predictions')
    ax_zcoff.plot(y_matlab.amplitudes_ansi_waves, '-o', color='C1', label='Matlab')

    ax_zcoff.legend(frameon=False)
    ax_zcoff.set_xticks(range(len(y_pred.amplitudes_ansi_waves)))
    ax_zcoff.set_ylabel(f'Amplitudes\n($\lambda = {wavelength}~\mu m$)')
    ax_zcoff.spines['top'].set_visible(False)

    net_error = 100 * np.abs(y_true.amplitudes_ansi_waves - y_pred.amplitudes_ansi_waves) / np.abs(y_true.amplitudes_ansi_waves)
    mat_error = 100 * np.abs(y_true.amplitudes_ansi_waves - y_matlab.amplitudes_ansi_waves) / np.abs(y_true.amplitudes_ansi_waves)

    ax_error = ax_zcoff.twinx()
    ax_error.bar(range(len(y_pred.zernikes)), net_error, color='C0', alpha=.2)
    ax_error.bar(range(len(y_pred.zernikes)), mat_error, color='C1', alpha=.2)

    ax_error.set_ylabel(f'MAPE = {np.mean(net_error[np.isfinite(net_error)]):.2f}%', color='darkgrey')
    ax_error.tick_params(axis='y', labelcolor='darkgrey')
    ax_error.set_ylim(0, 100)

    ax_error.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax_error.grid(True, which="both", axis='both', lw=1, ls='--', zorder=0)
    ax_error.spines['top'].set_visible(False)
    xticks = [f"$\\alpha$={z.index_ansi}\n$j$={z.index_noll}\n$n$={z.n}\n$m$={z.m}" for z in y_pred.zernikes]
    ax_error.set_xticklabels(xticks)

    #plt.subplots_adjust(top=0.95, right=0.95, wspace=.33)
    plt.tight_layout()
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight', pad_inches=.25)
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight', pad_inches=.25)


def predict(
    model: Path,
    img: Path,
    dm_pattern: Path,
    dm_state: Any,
    axial_voxel_size: float,
    lateral_voxel_size: float,
    wavelength: float = .605,
    scalar: float = 1,
    threshold: float = 0.0,
    verbose: bool = False,
    plot: bool = False
):
    physical_devices = tfc.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tfc.experimental.set_memory_growth(gpu_instance, True)

    model = backend.load(model, mosaic=True)

    psf = preprocessing.prep_psf(
        img,
        input_shape=model.layers[0].input_shape[0][1:-1],
        voxel_size=(axial_voxel_size, lateral_voxel_size, lateral_voxel_size)
    )

    psfgen = SyntheticPSF(
        snr=30,
        n_modes=60,
        lam_detection=wavelength,
        psf_shape=psf.shape,
        x_voxel_size=lateral_voxel_size,
        y_voxel_size=lateral_voxel_size,
        z_voxel_size=axial_voxel_size,
        batch_size=1,
        max_jitter=0,
        cpu_workers=-1,
    )

    psf_input = np.expand_dims(psf, axis=0)

    p, std = backend.bootstrap_predict(
        model,
        inputs=psf_input,
        batch_size=1,
        n_samples=1,
        threshold=threshold,
        verbose=False,
        psfgen=psfgen,
        plot=Path(f'{img.parent/img.stem}') if plot else None
    )
    dm_state = np.zeros(69) if dm_state is None else pd.read_csv(dm_state, header=None).values[:, 0]
    dm = zernikies_to_actuators(p, dm_pattern=dm_pattern, dm_state=dm_state, scalar=scalar)
    dm = pd.DataFrame(dm)
    dm.to_csv(f"{img.parent/img.stem}_corrected_actuators.csv", index=False, header=False)

    p = Wavefront(p, order='ansi')
    if verbose:
        logger.info('Prediction')
        logger.info(p.zernikes)

    coffs = [
        {'n': z.n, 'm': z.m, 'amplitude': utils.microns2waves(a, wavelength=wavelength)}
        for z, a in p.zernikes.items()
    ]
    coffs = pd.DataFrame(coffs, columns=['n', 'm', 'amplitude'])
    coffs.index.name = 'ansi'
    coffs.to_csv(f"{img.parent/img.stem}_zernike_coffs.csv")

    pupil_displacement = np.array(p.wave(size=100), dtype='float32')
    imsave(f"{img.parent/img.stem}_pred_pupil_displacement.tif", pupil_displacement)

    if plot:
        psfgen.single_otf(
            p.amplitudes,
            zplanes=0,
            normed=True,
            noise=True,
            na_mask=True,
            ratio=True,
            augmentation=True,
            meta=True,
            plot=Path(f'{img.parent/img.stem}_diagnosis'),
        )

        vis.prediction(
            psf=psf,
            pred=p,
            dm_before=dm_state,
            dm_after=dm.values[:, 0],
            save_path=Path(f'{img.parent/img.stem}_pred'),
        )


def predict_dataset(
    dataset: Path,
    model: Path,
    dm_pattern: Path,
    batch_size: int,
    x_voxel_size: float,
    y_voxel_size: float,
    z_voxel_size: float,
    wavelength: float,
):
    model = backend.load(model)
    model.summary()
    input_shape = model.layers[0].input_shape[0][1]
    voxel_size = (z_voxel_size, y_voxel_size, x_voxel_size)

    df = pd.read_json(dataset, orient='index')
    psfs = np.array(
        utils.multiprocess(
            partial(
                preprocessing.prep_psf,
                image_size=input_shape,
                voxel_size=voxel_size
            ),
            df.index.values,
            desc="Preprocessing PSFs")
    )
    predictions, stdev = backend.bootstrap_predict(model, psfs, batch_size=batch_size)
    df = df.iloc[:, :predictions.shape[-1]]

    outdir = Path(f"{dataset.parent}/predictions")
    outdir.mkdir(exist_ok=True, parents=True)

    utils.multiprocess(
        partial(
            correct,
            outdir=outdir,
            dm_pattern=dm_pattern,
            input_shape=input_shape,
            x_voxel_size=x_voxel_size,
            y_voxel_size=y_voxel_size,
            z_voxel_size=z_voxel_size,
        ),
        list(zip(df.values, predictions, df.index.values)),
        desc="Correcting PSFs"
    )

    res = pd.DataFrame.from_dict({
        'psf': df.index.values,
        'mae': utils.mae(df.values, predictions),
        'mse': utils.mse(df.values, predictions),
        'rmse': utils.rmse(df.values, predictions),
    })
    logger.info(res)
    logger.info(f"MAE: {res.mae.mean()}")
    logger.info(f"MSE: {res.mse.mean()}")
    res.to_csv(f"{outdir}/results.csv")


def compare(
    dataset: Path,
    model: Path,
    dm_pattern: Path,
    batch_size: int,
    x_voxel_size: float,
    y_voxel_size: float,
    z_voxel_size: float,
    wavelength: float,
):
    model = backend.load(model)
    model.summary()
    input_shape = model.layers[0].input_shape[0][1]
    voxel_size = (z_voxel_size, y_voxel_size, x_voxel_size)

    df = pd.read_json(dataset, orient='index')
    psfs = np.array(
        utils.multiprocess(
            partial(
                preprocessing.prep_psf,
                image_size=input_shape,
                voxel_size=voxel_size
            ),
            df.index.values,
            desc="Preprocessing PSFs")
    )
    predictions, stdev = backend.bootstrap_predict(model, psfs, batch_size=batch_size)
    df = df.iloc[:, :predictions.shape[-1]]

    outdir = Path(f"{dataset.parent}/predictions")
    outdir.mkdir(exist_ok=True, parents=True)

    utils.multiprocess(
        partial(
            correct,
            outdir=outdir,
            dm_pattern=dm_pattern,
            input_shape=input_shape,
            x_voxel_size=x_voxel_size,
            y_voxel_size=y_voxel_size,
            z_voxel_size=z_voxel_size,
        ),
        list(zip(df.values, predictions, df.index.values)),
        desc="Correcting PSFs"
    )

    res = pd.DataFrame.from_dict({
        'psf': df.index.values,
        'mae': utils.mae(df.values, predictions),
        'mse': utils.mse(df.values, predictions),
        'rmse': utils.rmse(df.values, predictions),
    })
    logger.info(res)
    logger.info(f"MAE: {res.mae.mean()}")
    logger.info(f"MSE: {res.mse.mean()}")
    res.to_csv(f"{outdir}/results.csv")
