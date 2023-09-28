import itertools
import matplotlib
matplotlib.use('Agg')

from multiprocessing import Pool

import logging
import sys
from functools import partial
from pathlib import Path
from typing import Any, Optional
from matplotlib.ticker import FormatStrFormatter, PercentFormatter
import matplotlib.colors as mcolors
import seaborn as sns

import matplotlib.pyplot as plt
plt.set_loglevel('error')

import numpy as np
import pandas as pd
import tensorflow as tf
from line_profiler_pycharm import profile
from tqdm import tqdm
from tifffile import imwrite

import utils
import data_utils
import backend
import vis
import multipoint_dataset

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
def simulate_beads(
    psf,
    psf_type,
    beads=None,
    photons=100000,
    maxcounts=None,
    object_size=0,
    num_objs=1,
    noise=True,
    fill_radius=.4,
    fast=False,
):

    if beads is None:
        beads = multipoint_dataset.beads(
            image_shape=psf.shape,
            photons=photons if maxcounts is None else 1,
            object_size=object_size,
            num_objs=num_objs,
            fill_radius=fill_radius
        )

    if maxcounts is not None:
        psf /= psf.max()
        psf *= maxcounts
    else:
        if psf_type == 'widefield':  # normalize PSF by the total energy in the focal plane
            focal_plane_index = [(w // 2) - 1 for w in psf.shape]
            psf /= np.sum(psf[focal_plane_index[0], focal_plane_index[1], focal_plane_index[2]])
        else:
            psf /= np.sum(psf)

    if fast:
        original_shape = beads.shape
        inputs = np.zeros_like(beads)
        bead_indices = np.vstack(np.nonzero(beads)).transpose()
        shift_amount = bead_indices - np.array(beads.shape)//2

        pad_amount = np.max(np.abs(shift_amount), axis=None)
        inputs = np.pad(inputs, pad_width=pad_amount)
        psf = np.pad(psf, pad_width=pad_amount)
        beads = np.pad(beads, pad_width=pad_amount)

        bead_indices = np.vstack(np.nonzero(beads)).transpose()
        shift_amount = bead_indices - np.array(beads.shape) // 2
        for idx, shift in zip(bead_indices, shift_amount):
            inputs += np.roll(psf*beads[tuple(idx)], shift, axis=(0,1,2))

        inputs = inputs[
               pad_amount:pad_amount + original_shape[0],
               pad_amount:pad_amount + original_shape[1],
               pad_amount:pad_amount + original_shape[2],
               ]

    else:
        inputs = utils.fftconvolution(sample=beads, kernel=psf)  # takes 1 second.

    if noise:
        inputs = utils.add_noise(inputs)
    else:  # convert image to counts
        inputs = utils.photons2electrons(inputs)
        inputs = utils.electrons2counts(inputs)

    return inputs


def generate_sample(
    image_id: int,
    data: pd.DataFrame,
    psfgen: SyntheticPSF,
    iter_number: Optional[int] = None,
    savedir: Optional[Path] = None,
    no_phase: bool = False,
    digital_rotations: Optional[int] = None,
    no_beads: bool = False
):
    hashtable = data[data['id'] == image_id].iloc[0].to_dict()
    f = Path(str(hashtable['file']))
    beads = Path(str(hashtable['beads']))

    if savedir is not None:
        outdir = savedir / '/'.join(beads.parent.parts[-4:]) / f'iter_{iter_number}'
        outdir.mkdir(exist_ok=True, parents=True)
        savepath = outdir / f.name

    if savepath.exists():
        return savepath
    else:
        ys = [hashtable[cc] for cc in data.columns[data.columns.str.endswith('_residual')]]
        ref = np.squeeze(data_utils.get_image(beads))

        wavefront = Wavefront(
            ys,
            modes=psfgen.n_modes,
            lam_detection=psfgen.lam_detection,
        )

        psf = psfgen.single_psf(
            phi=wavefront,
            normed=True,
            meta=False,
        )

        if no_beads:
            noisy_img = simulate_beads(
                psf=psf,
                psf_type=psfgen.psf_type,
                beads=None,
                fill_radius=0,
                object_size=0,
                photons=hashtable['photons']
            )
        else:
            noisy_img = simulate_beads(
                psf=psf,
                psf_type=psfgen.psf_type,
                beads=ref,
                photons=hashtable['photons']
            )

        if savedir is not None:
            imwrite(savepath, noisy_img.astype(np.float32), dtype=np.float32)
            return savepath
        else:
            emb = backend.preprocess(
                noisy_img,
                modelpsfgen=psfgen,
                digital_rotations=digital_rotations,
                no_phase=no_phase,
                remove_background=True,
                normalize=True,
                # plot=True
            )
            return emb


def collect_data(
    datapath,
    model,
    samplelimit: int = 1,
    distribution: str = '/',
    no_phase: bool = False,
    photons_range: Optional[tuple] = None,
    npoints_range: Optional[tuple] = None,
    psf_type: Optional[str] = None,
    lam_detection: Optional[float] = None,
    default_wavelength: Optional[float] = .510,
):

    if isinstance(model, tf.keras.Model):
        predicted_modes = model.output_shape[-1]
    elif isinstance(model, int):
        predicted_modes = model

    metadata = data_utils.collect_dataset(
        datapath,
        metadata=True,
        modes=predicted_modes,
        samplelimit=samplelimit,
        distribution=distribution,
        no_phase=no_phase,
        photons_range=photons_range,
        npoints_range=npoints_range,
        suffix_to_avoid="_sample_predictions_psf.tif"
    )  # metadata is a list of arrays

    # This runs multiple samples (aka images) at a time.
    # ys is a 2D array, rows are each sample, columns give aberration in zernike coefficients
    metadata = np.array(list(metadata.take(-1)))
    ys = np.zeros((metadata.shape[0], predicted_modes))
    counts_percentiles = np.zeros((metadata.shape[0], 100))

    results = {
        # image number where the voxel locations of the beads are given in 'file'. Constant over iterations.
        'id': np.arange(metadata.shape[0], dtype=int),
        'iter_num': np.zeros(metadata.shape[0], dtype=int),          # iteration index.
        'aberration': np.zeros(metadata.shape[0], dtype=float),      # initial p2v aberration. Constant over iterations.
        'residuals': np.zeros(metadata.shape[0], dtype=float),       # remaining p2v aberration after ML correction.
        'residuals_umRMS': np.zeros(metadata.shape[0], dtype=float), # remaining umRMS aberration after ML correction.
        'photons': np.zeros(metadata.shape[0], dtype=int),           # integrated photons
        'counts': np.zeros(metadata.shape[0], dtype=int),            # integrated counts
        'counts_mode': np.zeros(metadata.shape[0], dtype=int),       # counts mode
        'distance': np.zeros(metadata.shape[0], dtype=float),        # average distance to nearst bead
        'neighbors': np.zeros(metadata.shape[0], dtype=int),         # number of beads
        'file': np.empty(metadata.shape[0], dtype=Path),             # path to realspace images
        'file_windows': np.empty(metadata.shape[0], dtype=Path),     # stupid windows path
        'beads': np.zeros(metadata.shape[0], dtype=Path),
        'psf_type': np.full(metadata.shape[0], dtype=str, fill_value=psf_type),
        'wavelength': np.full(metadata.shape[0], dtype=float, fill_value=lam_detection),
        # path to binary image file filled with zeros except at location of beads
    }

    # see `data_utils.get_sample` to check order of objects returned
    for i in range(metadata.shape[0]):

        # rescale zernike amplitudes to maintain the same peak2valley for different PSFs
        ys[i] = lam_detection / default_wavelength * metadata[i, 0].numpy()[:predicted_modes]
        results['residuals_umRMS'][i] = np.linalg.norm(ys[i])

        results['photons'][i] = metadata[i, 1].numpy()
        results['counts'][i] = metadata[i, 2].numpy()
        results['counts_mode'][i] = metadata[i, 3].numpy()

        counts_percentiles[i] = metadata[i, 4].numpy()

        results['aberration'][i] = metadata[i, 5].numpy()
        results['residuals'][i] = results['aberration'][i]

        results['neighbors'][i] = metadata[i, 7].numpy()
        results['distance'][i] = metadata[i, 8].numpy()

        f = Path(str(metadata[i, -1].numpy(), "utf-8"))
        results['file'][i] = f
        results['file_windows'][i] = utils.convert_to_windows_file_string(f)
        results['beads'][i] = f.with_name(f'{f.stem}_gt' + f.suffix)

    # 'results' is a df to be written out as the _predictions.csv.
    # 'results' holds the information from every iteration.
    # Initialize it first with the zeroth iteration.
    results = pd.DataFrame.from_dict(results)

    for z in range(ys.shape[-1]):
        results[f'z{z}_ground_truth'] = ys[:, z]
        results[f'z{z}_prediction'] = np.zeros_like(ys[:, z])
        results[f'z{z}_residual'] = ys[:, z]

    for p in range(100):
        results[f'counts_p{p+1}'] = counts_percentiles[:, p]

    return results


@profile
def iter_evaluate(
    datapath,
    modelpath,
    iter_num: int = 5,
    samplelimit: int = 1,
    na: float = 1.0,
    distribution: str = '/',
    threshold: float = 0.,
    no_phase: bool = False,
    batch_size: int = 128,
    photons_range: Optional[tuple] = None,
    npoints_range: Optional[tuple] = None,
    eval_sign: str = 'signed',
    digital_rotations: bool = False,
    rotations: Optional[int] = 361,
    savepath: Any = None,
    plot: Any = None,
    plot_rotations: bool = False,
    psf_type: Optional[str] = None,
    lam_detection: Optional[float] = .510
):
    """
    Gathers the set of .tif files that meet the input criteria.
    Predicts on all of those for (iter_num) iterations.
    Results go into dataframe called "results"
    Saves "results" dataframe to _predictions.csv file

    Returns:
        "results" dataframe
    """

    model = backend.load(modelpath)

    gen = backend.load_metadata(
        modelpath,
        signed=True,
        rotate=False,
        batch_size=batch_size,
        psf_shape=3 * [model.input_shape[2]],
        psf_type=psf_type,
        lam_detection=lam_detection,
    )

    if iter_num == 1:
        # on first call, setup the dataframe with the 0th iteration stuff
        results = collect_data(
            datapath=datapath,
            model=model,
            samplelimit=samplelimit,
            distribution=distribution,
            no_phase=no_phase,
            photons_range=photons_range,
            npoints_range=npoints_range,
            psf_type=gen.psf_type,
            lam_detection=gen.lam_detection
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
            generate_sample,
            iter_number=iter_num,
            savedir=savepath.resolve(),
            data=previous,
            psfgen=gen,
            no_phase=no_phase,
            digital_rotations=rotations if digital_rotations else None,
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

    predictions = backend.predict_files(
        paths=paths,
        outdir=savepath/f'iter_{iter_num}',
        model=model,
        modelpsfgen=gen,
        samplepsfgen=None,
        dm_calibration=None,
        dm_state=None,
        batch_size=batch_size,
        fov_is_small=True,
        plot=plot,
        plot_rotations=plot_rotations,
        digital_rotations=rotations if digital_rotations else None,
        cpu_workers=-1,
    ).T
    current[prediction_cols] = predictions.values[:paths.shape[0]]  # drop (mean, median, min, max, and std)
    current[ground_truth_cols] = previous[residual_cols]

    if eval_sign == 'positive_only':
        current[prediction_cols] = current[prediction_cols].abs()
        current[ground_truth_cols] = current[ground_truth_cols].abs()

    current[residual_cols] = current[ground_truth_cols].values - current[prediction_cols].values

    # compute residuals for each sample
    current['residuals'] = current.apply(
        lambda row: Wavefront(row[residual_cols].values, lam_detection=gen.lam_detection).peak2valley(na=na),
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

    return results


@profile
def plot_heatmap_p2v(
    dataframe,
    wavelength,
    savepath: Path,
    label='Integrated photoelectrons',
    lims=(0, 100),
    ax=None,
    cax=None,
    agg='mean',
    histograms: Optional[pd.DataFrame] = None
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

    if ax is None:

        if histograms is not None:
            fig = plt.figure(figsize=(12, 8))
            gs = fig.add_gridspec(3, 4)
            ax = fig.add_subplot(gs[:, 1:])
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0])
            ax3 = fig.add_subplot(gs[2, 0])
        else:
            fig, ax = plt.subplots(figsize=(8, 6))

    if cax is None:
        cax = fig.add_axes([1.03, 0.08, 0.03, 0.87])

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
        dataframe.columns.values,
        dataframe.index.values,
        dataframe.values,
        cmap=cmap,
        levels=levels,
        extend='max',
        linewidths=2,
        linestyles='dashed',
    )
    ax.patch.set(hatch='/', edgecolor='lightgrey', lw=.01)

    if histograms is not None:
        if label == 'Integrated photoelectrons':
            x = histograms[
                (histograms.pbins <= 1e5) &
                (histograms.ibins >= 1.5) & (histograms.ibins <= 2.5)
            ]
            ax1 = sns.histplot(
                ax=ax1,
                data=x,
                x="residuals",
                stat='percent',
                kde=True,
                bins=25,
                color='dimgrey'
            )
            ax1.axvline(np.median(x['residuals']), c='C0', ls='-', lw=2, label='Median')
            ax1.axvline(np.mean(x['residuals']), c='C1', ls='--', lw=2, label='Mean')
            ax1.set_ylim(0, 30)
            ax1.set_xlim(0, 3)
            ax1.set_xlabel('Residuals')
            ax1.set_ylabel('')
            ax1.text(
                .9, .8, 'I',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=20,
                color='k',
                transform=ax1.transAxes
            )

            ax.add_patch(
                plt.Rectangle((0, .3), .1, .2, ec="k", fc="none", transform=ax.transAxes)
            )
            ax.text(
                .05, .4, 'I',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=20,
                color='k',
                transform=ax.transAxes
            )

            x = histograms[
                (histograms.pbins >= 2e5) & (histograms.pbins <= 3e5) &
                (histograms.ibins >= 1.5) & (histograms.ibins <= 2.5)
            ]
            ax2 = sns.histplot(
                ax=ax2,
                data=x,
                x="residuals",
                stat='percent',
                kde=True,
                bins=25,
                color='dimgrey'
            )
            ax2.axvline(np.median(x['residuals']), c='C0', ls='-', lw=2)
            ax2.axvline(np.mean(x['residuals']), c='C1', ls='--', lw=2)
            ax2.set_ylim(0, 30)
            ax2.set_xlim(0, 3)
            ax2.set_xlabel('Residuals')
            ax2.set_ylabel('')
            ax2.text(
                .9, .8, 'II',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=20,
                color='k',
                transform=ax2.transAxes
            )

            ax.add_patch(
                plt.Rectangle((.2, .3), .1, .2, ec="k", fc="none", transform=ax.transAxes)
            )
            ax.text(
                .25, .4, 'II',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=20,
                color='k',
                transform=ax.transAxes
            )

            x = histograms[
                (histograms.pbins >= 5e5) & (histograms.pbins <= 6e5) &
                (histograms.ibins >= 1.5) & (histograms.ibins <= 2.5)
            ]
            ax3 = sns.histplot(
                ax=ax3,
                data=x,
                x="residuals",
                stat='percent',
                kde=True,
                bins=25,
                color='dimgrey'
            )
            ax3.axvline(np.median(x['residuals']), c='C0', ls='-', lw=2)
            ax3.axvline(np.mean(x['residuals']), c='C1', ls='--', lw=2)
            ax3.set_ylim(0, 30)
            ax3.set_xlim(0, 3)
            ax3.set_xlabel('Residuals')
            ax3.set_ylabel('')
            ax3.text(
                .9, .8, 'III',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=20,
                color='k',
                transform=ax3.transAxes
            )

            ax.add_patch(
                plt.Rectangle((.5, .3), .1, .2, ec="k", fc="none", transform=ax.transAxes)
            )

            ax.text(
                .55, .4, 'III',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=20,
                color='k',
                transform=ax.transAxes
            )

            ax1.legend(frameon=False, ncol=1, loc='upper left')
            ax1.yaxis.set_major_formatter(PercentFormatter(decimals=0))
            ax1.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)
            ax2.yaxis.set_major_formatter(PercentFormatter(decimals=0))
            ax2.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)
            ax3.yaxis.set_major_formatter(PercentFormatter(decimals=0))
            ax3.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

        elif label == f'Number of iterations':

            x = histograms[
                (histograms.iter_num == 1) &
                (histograms.aberration >= 2.5) & (histograms.aberration <= 3.5)
            ]
            ax1 = sns.histplot(
                ax=ax1,
                data=x,
                x="residuals",
                stat='percent',
                kde=True,
                bins=25,
                color='dimgrey'
            )

            ax1.axvline(np.median(x['residuals']), c='C0', ls='-', lw=2)
            ax1.axvline(np.mean(x['residuals']), c='C1', ls='--', lw=2)
            ax1.axvline(np.median(x['residuals']), c='C0', ls='-', lw=2, label='Median')
            ax1.axvline(np.mean(x['residuals']), c='C1', ls='--', lw=2, label='Mean')
            ax1.set_ylim(0, 80)
            ax1.set_xlim(0, 5)
            ax1.set_xlabel('Residuals')
            ax1.set_ylabel('')
            ax1.text(
                .9, .8, 'I',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=20,
                color='k',
                transform=ax1.transAxes
            )

            ax.add_patch(
                plt.Rectangle((.05, .5), .1, .2, ec="k", fc="none", transform=ax.transAxes)
            )
            ax.text(
                .1, .6, 'I',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=20,
                color='k',
                transform=ax.transAxes
            )

            x = histograms[
                (histograms.iter_num == 2) &
                (histograms.aberration >= 2.5) & (histograms.aberration <= 3.5)
            ]
            ax2 = sns.histplot(
                ax=ax2,
                data=x,
                x="residuals",
                stat='percent',
                kde=True,
                bins=25,
                color='dimgrey'
            )
            ax2.axvline(np.median(x['residuals']), c='C0', ls='-', lw=2)
            ax2.axvline(np.mean(x['residuals']), c='C1', ls='--', lw=2)
            ax2.set_ylim(0, 80)
            ax2.set_xlim(0, 5)
            ax2.set_xlabel('Residuals')
            ax2.set_ylabel('')
            ax2.text(
                .9, .8, 'II',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=20,
                color='k',
                transform=ax2.transAxes
            )

            ax.add_patch(
                plt.Rectangle((.15, .5), .1, .2, ec="k", fc="none", transform=ax.transAxes)
            )
            ax.text(
                .2, .6, 'II',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=20,
                color='k',
                transform=ax.transAxes
            )

            x = histograms[
                (histograms.iter_num == 5) &
                (histograms.aberration >= 2.5) & (histograms.aberration <= 3.5)
            ]
            ax3 = sns.histplot(
                ax=ax3,
                data=x,
                x="residuals",
                stat='percent',
                kde=True,
                bins=25,
                color='dimgrey'
            )
            ax3.axvline(np.median(x['residuals']), c='C0', ls='-', lw=2, label='Median')
            ax3.axvline(np.mean(x['residuals']), c='C1', ls='--', lw=2, label='Mean')
            ax3.set_ylim(0, 80)
            ax3.set_xlim(0, 5)
            ax3.set_xlabel('Residuals')
            ax3.set_ylabel('')
            ax3.text(
                .9, .8, 'III',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=20,
                color='k',
                transform=ax3.transAxes
            )

            ax.add_patch(
                plt.Rectangle((.45, .5), .1, .2, ec="k", fc="none", transform=ax.transAxes)
            )
            ax.text(
                .5, .6, 'III',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=20,
                color='k',
                transform=ax.transAxes
            )

            ax1.legend(frameon=False, ncol=1, loc='upper center')
            ax1.yaxis.set_major_formatter(PercentFormatter(decimals=0))
            ax1.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)
            ax2.yaxis.set_major_formatter(PercentFormatter(decimals=0))
            ax2.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)
            ax3.yaxis.set_major_formatter(PercentFormatter(decimals=0))
            ax3.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

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

    cbar.ax.set_ylabel(rf'Residuals ({agg} peak-to-valley, $\lambda = {int(wavelength*1000)}~nm$)')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('left')

    if label == 'Integrated photons per object':
        ax.set_xticks(np.arange(0, 1e6+1e5, 1e5), minor=False)
        ax.set_xticks(np.arange(0, 1e6+10e4, 5e4), minor=True)
    elif label == 'Number of iterations':
        ax.set_xticks(np.arange(0, dataframe.columns.values.max()+1, 1), minor=False)

    ax.set_xlabel(label)
    ax.set_xlim(lims)

    ax.set_ylabel(rf'Initial aberration ({agg} peak-to-valley, $\lambda = {int(wavelength*1000)}~nm$)')
    ax.set_yticks(np.arange(0, 6, .5), minor=True)
    ax.set_yticks(np.arange(0, 6, 1))
    ax.set_ylim(0, 5)

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)
    plt.tight_layout()

    savepath = Path(f'{savepath}_p2v')
    plt.savefig(f'{savepath}.pdf', bbox_inches='tight', pad_inches=.25)
    plt.savefig(f'{savepath}.png', dpi=300, bbox_inches='tight', pad_inches=.25)
    plt.savefig(f'{savepath}.svg', dpi=300, bbox_inches='tight', pad_inches=.25)

    logger.info(f'Saved: {savepath.resolve()}.png  .pdf  .svg')
    return ax


@profile
def plot_heatmap_umRMS(
    dataframe,
    wavelength,
    savepath:Path,
    label='Integrated photons per object',
    lims=(0, 100),
    ax=None,
    cax=None,
    agg='mean',
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

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if cax is None:
        cax = fig.add_axes([1.01, 0.08, 0.03, 0.87])

    levels = np.array([
        0, .05, .1, .15, .2, .25, .3, .35, .4, .45,
        .5, .6, .7, .8, .9,
        1, 1.25, 1.5, 1.75, 2., 2.5,
        3., 4., 5.,
    ])

    umRMS_per_p2v_factor = round(np.percentile(dataframe.to_numpy(), 98) / np.max(levels), 2)
    levels *= umRMS_per_p2v_factor
    vmin, vmax, vcenter, step = levels[0], levels[-1], levels[10], levels[1] - levels[0]
    highcmap = plt.get_cmap('magma_r', 256)
    lowcmap = plt.get_cmap('GnBu_r', 256)
    low = np.linspace(0, 1 - step, int(abs(vcenter - vmin) / step))
    high = np.linspace(0, 1 + step, int(abs(vcenter - vmax) / step))
    cmap = np.vstack((lowcmap(low), [1, 1, 1, 1], highcmap(high)))
    cmap = mcolors.ListedColormap(cmap)

    contours = ax.contourf(
        dataframe.columns.values,
        dataframe.index.values,
        dataframe.values,
        cmap=cmap,
        levels=levels,
        extend='max',
        linewidths=2,
        linestyles='dashed',
    )

    ax.patch.set(hatch='/', edgecolor='lightgrey', lw=.01)

    cbar = plt.colorbar(
        contours,
        cax=cax,
        fraction=0.046,
        pad=0.04,
        extend='both',
        spacing='proportional',
        format=FormatStrFormatter("%.2f"),
        ticks=np.array([0, .15, .3, .5, .75, 1., 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5]) * umRMS_per_p2v_factor,
    )
    cbar.ax.set_ylabel(fr'Residuals ({agg} $\mu$mRMS)')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('left')

    if label == 'Integrated photons per object':
        ax.set_xticks(np.arange(0, 1e6+1e5, 1e5), minor=False)
        ax.set_xticks(np.arange(0, 1e6+10e4, 5e4), minor=True)
    elif label == 'Number of iterations':
        ax.set_xticks(np.arange(0, dataframe.columns.values.max() + 1, 1), minor=False)

    ax.set_xlabel(label)
    ax.set_xlim(lims)

    ax.set_ylabel(fr'Initial aberration ({agg} $\mu$mRMS)')
    ax.set_yticks(np.arange(0, 6, .5) * umRMS_per_p2v_factor, minor=True)
    ax.set_yticks(np.arange(0, 6, 1) * umRMS_per_p2v_factor)
    ax.set_ylim(0, levels[-1])

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)
    plt.tight_layout()

    savepath = Path(f'{savepath}_umRMS')
    plt.savefig(f'{savepath}.pdf', bbox_inches='tight', pad_inches=.25)
    plt.savefig(f'{savepath}.png', dpi=300, bbox_inches='tight', pad_inches=.25)
    plt.savefig(f'{savepath}.svg', dpi=300, bbox_inches='tight', pad_inches=.25)

    logger.info(f'Saved: {savepath.resolve()}.png  .pdf  .svg')
    return ax


@profile
def snrheatmap(
    modelpath: Path,
    datadir: Path,
    iter_num: int = 1,
    distribution: str = '/',
    samplelimit: Any = None,
    na: float = 1.0,
    batch_size: int = 100,
    eval_sign: str = 'signed',
    digital_rotations: bool = False,
    plot: Any = None,
    plot_rotations: bool = False,
    agg: str = 'median',
    psf_type: Optional[str] = None,
    lam_detection: Optional[float] = .510,
):
    modelspecs = backend.load_metadata(modelpath)
    savepath = modelpath.with_suffix('') / eval_sign / f'snrheatmaps'

    if psf_type is not None:
        savepath = Path(f"{savepath}/mode-{str(psf_type).replace('../lattice/', '').split('_')[0]}")

    savepath.mkdir(parents=True, exist_ok=True)

    if distribution != '/':
        savepath = Path(f'{savepath}/{distribution}_na_{str(na).replace("0.", "p")}')
    else:
        savepath = Path(f'{savepath}/na_{str(na).replace("0.", "p")}')

    if datadir.suffix == '.csv':
        df = pd.read_csv(datadir, header=0, index_col=0)
    else:
        df = iter_evaluate(
            iter_num=iter_num,
            modelpath=modelpath,
            datapath=datadir,
            savepath=savepath,
            samplelimit=samplelimit,
            na=na,
            batch_size=batch_size,
            photons_range=None,
            npoints_range=(1, 1),
            eval_sign=eval_sign,
            digital_rotations=digital_rotations,
            plot=plot,
            plot_rotations=plot_rotations,
            psf_type=psf_type,
            lam_detection=lam_detection
        )

    df = df[df['iter_num'] == iter_num]
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

        plot_heatmap_p2v(
            dataframe,
            histograms=df if x == 'photons' else None,
            wavelength=modelspecs.lam_detection,
            savepath=Path(f"{savepath}_iter_{iter_num}_{x}"),
            label=label,
            lims=lims,
            agg=agg
        )

    return savepath


@profile
def densityheatmap(
    modelpath: Path,
    datadir: Path,
    iter_num: int = 1,
    distribution: str = '/',
    na: float = 1.0,
    samplelimit: Any = None,
    batch_size: int = 100,
    eval_sign: str = 'signed',
    digital_rotations: bool = False,
    plot: Any = None,
    plot_rotations: bool = False,
    agg: str = 'median',
    photons_range: Optional[tuple] = None,
    psf_type: Optional[str] = None,
    lam_detection: Optional[float] = .510,
):
    modelspecs = backend.load_metadata(modelpath)

    savepath = modelpath.with_suffix('') / eval_sign / f'densityheatmaps'

    if psf_type is not None:
        savepath = Path(f"{savepath}/mode-{str(psf_type).replace('../lattice/', '').split('_')[0]}")

    savepath.mkdir(parents=True, exist_ok=True)

    if distribution != '/':
        savepath = Path(f'{savepath}/{distribution}_na_{str(na).replace("0.", "p")}')
    else:
        savepath = Path(f'{savepath}/na_{str(na).replace("0.", "p")}')

    if datadir.suffix == '.csv':
        df = pd.read_csv(datadir, header=0, index_col=0)
    else:
        df = iter_evaluate(
            iter_num=iter_num,
            modelpath=modelpath,
            datapath=datadir,
            savepath=savepath,
            samplelimit=samplelimit,
            distribution=distribution,
            na=na,
            photons_range=photons_range,
            npoints_range=None,
            batch_size=batch_size,
            eval_sign=eval_sign,
            digital_rotations=digital_rotations,
            plot=plot,
            plot_rotations=plot_rotations,
            psf_type=psf_type,
            lam_detection=lam_detection
        )

    df = df[df['iter_num'] == iter_num]

    bins = np.arange(0, 10.25, .25).round(2)
    df['ibins'] = pd.cut(
        df['aberration'],
        bins,
        labels=bins[1:],
        include_lowest=True
    )

    for col, label, lims in zip(
        ['neighbors', 'distance'],
        ['Number of objects', 'Average distance to nearest neighbor (microns)'],
        [(1, 150), (0, 2)]
    ):
        dataframe = pd.pivot_table(df, values='residuals', index='ibins', columns=col, aggfunc=agg)
        dataframe = dataframe.sort_index().interpolate()

        dataframe.to_csv(f'{savepath}.csv')
        logger.info(f'Saved: {savepath.resolve()}.csv')

        plot_heatmap_p2v(
            dataframe,
            wavelength=modelspecs.lam_detection,
            savepath=Path(f'{savepath}_iter_{iter_num}_{col}'),
            label=label,
            lims=lims,
            agg=agg
        )

    return savepath


@profile
def iterheatmap(
    modelpath: Path,
    datadir: Path,  # folder or _predictions.csv file
    iter_num: int = 5,
    distribution: str = '/',
    samplelimit: Any = None,
    na: float = 1.0,
    no_phase: bool = False,
    batch_size: int = 1024,
    eval_sign: str = 'signed',
    digital_rotations: bool = False,
    photons_range: Optional[tuple] = None,
    plot: Any = None,
    plot_rotations: bool = False,
    agg: str = 'median',
    psf_type: Optional[str] = None,
    lam_detection: Optional[float] = .510,
):
    modelspecs = backend.load_metadata(modelpath)
    savepath = modelpath.with_suffix('') / eval_sign / f'iterheatmaps'

    if psf_type is not None:
        savepath = Path(f"{savepath}/mode-{str(psf_type).replace('../lattice/', '').split('_')[0]}")

    savepath.mkdir(parents=True, exist_ok=True)

    if distribution != '/':
        savepath = Path(f'{savepath}/{distribution}_na_{str(na).replace("0.", "p")}')
    else:
        savepath = Path(f'{savepath}/na_{str(na).replace("0.", "p")}')

    logger.info(f'Save path = {savepath.resolve()}')
    if datadir.suffix == '.csv':
        df = pd.read_csv(datadir, header=0, index_col=0)  # read previous results, ignoring criteria
        logger.info(f'Using "{datadir}"')
        logger.info(f'Found {len(df.id.unique())} samples, {df.iter_num.max()} iterations.')
    else:
        # make new inferences and obtain new results
        df = iter_evaluate(
            datadir,
            savepath=savepath,
            iter_num=iter_num,
            modelpath=modelpath,
            samplelimit=samplelimit,
            na=na,
            photons_range=photons_range,
            npoints_range=(1, 1),
            no_phase=no_phase,
            batch_size=batch_size,
            eval_sign=eval_sign,
            digital_rotations=digital_rotations,
            plot=plot,
            plot_rotations=plot_rotations,
            psf_type=psf_type,
            lam_detection=lam_detection
        )

    max_iter = df['iter_num'].max()
    for value in ('residuals', 'residuals_umRMS'):
        dataframe = pd.pivot_table(df[df['iter_num'] == 0], values=value, index='id', columns='iter_num')
        for i in range(1, max_iter+1):
            dataframe[i] = pd.pivot_table(df[df['iter_num'] == i], values=value, index='id', columns='iter_num')

        bins = np.linspace(0, np.nanmax(df[value]), num=25)
        dataframe.index = pd.cut(dataframe[0], bins, labels=bins[1:], include_lowest=True)
        dataframe.index.name = 'bins'
        dataframe = dataframe.groupby("bins").agg(agg)
        dataframe.loc[0] = pd.Series({cc: 0 for cc in dataframe.columns})
        dataframe = dataframe.sort_index().interpolate()
        dataframe.to_csv(f'{savepath}.csv')
        logger.info(f'Saved: {savepath.resolve()}.csv')

        if value == 'residuals':
            plot_heatmap_p2v(
                dataframe,
                histograms=df,
                wavelength=modelspecs.lam_detection,
                savepath=savepath,
                label=f'Number of iterations',
                lims=(0, max_iter),
                agg=agg,
            )
        elif value == 'residuals_umRMS':
            plot_heatmap_umRMS(
                dataframe,
                wavelength=modelspecs.lam_detection,
                savepath=savepath,
                label=f'Number of iterations',
                lims=(0, max_iter),
                agg=agg,
            )
        else:
            raise Exception(f"We don't have code for this case: {value}")

    return savepath


@profile
def random_samples(
    model: Path,
    photons: int = 3e5,
    batch_size: int = 512,
    eval_sign: str = 'signed',
    digital_rotations: bool = False,
):
    m = backend.load(model)
    m.summary()
    no_phase = True if m.input_shape[1] == 3 else False

    pool = Pool(processes=4)  # plotting = 2*calculation time, so shouldn't need more than 2-4 processes to keep up.

    for dist in ['single', 'bimodal', 'multinomial', 'powerlaw', 'dirichlet']:
        for amplitude_range in [(.05, .1), (.1, .2), (.2, .3)]:
            gen = backend.load_metadata(
                model,
                amplitude_ranges=amplitude_range,
                distribution=dist,
                signed=False if eval_sign == 'positive_only' else True,
                rotate=True,
                mode_weights='pyramid',
                psf_shape=(64, 64, 64),
            )
            for s in range(10):
                for num_objs in tqdm([1, 2, 5, 25, 50, 100, 150], file=sys.stdout):
                    reference = multipoint_dataset.beads(
                        photons=photons,
                        image_shape=gen.psf_shape,
                        object_size=0,
                        num_objs=num_objs,
                        fill_radius=.3 if num_objs > 1 else 0
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
                    psf, y, y_lls_defocus = gen.single_psf(
                        phi=phi,
                        normed=True,
                        meta=True,
                        lls_defocus_offset=(0, 0)
                    )

                    noisy_img = simulate_beads(psf, psf_type=gen.psf_type, beads=reference, noise=True)
                    maxcounts = np.max(noisy_img)
                    noisy_img /= maxcounts

                    save_path = Path(
                        f"{model.with_suffix('')}/{eval_sign}/samples/{dist}/um-{amplitude_range[-1]}/num_objs-{num_objs:02d}"
                    )
                    save_path.mkdir(exist_ok=True, parents=True)

                    embeddings = backend.preprocess(
                        noisy_img,
                        modelpsfgen=gen,
                        digital_rotations=361 if digital_rotations else None,
                        remove_background=True,
                        normalize=True,
                        plot=save_path / f'{s}',
                    )

                    if digital_rotations:
                        res = backend.predict_rotation(
                            m,
                            embeddings,
                            save_path=save_path / f'{s}',
                            psfgen=gen,
                            no_phase=no_phase,
                            batch_size=batch_size,
                            plot=save_path / f'{s}',
                            plot_rotations=save_path / f'{s}',
                        )
                    else:
                        res = backend.bootstrap_predict(
                            m,
                            embeddings,
                            psfgen=gen,
                            no_phase=no_phase,
                            batch_size=batch_size,
                            plot=save_path / f'{s}',
                        )

                    try:
                        p, std = res
                        p_lls_defocus = None
                    except ValueError:
                        p, std, p_lls_defocus = res

                    if eval_sign == 'positive_only':
                        y = np.abs(y)
                        if len(p.shape) > 1:
                            p = np.abs(p)[:, :y.shape[-1]]
                        else:
                            p = np.abs(p)[np.newaxis, :y.shape[-1]]
                    else:
                        if len(p.shape) > 1:
                            p = p[:, :y.shape[-1]]
                        else:
                            p = p[np.newaxis, :y.shape[-1]]

                    residuals = y - p

                    p_wave = Wavefront(p, lam_detection=gen.lam_detection)
                    y_wave = Wavefront(y, lam_detection=gen.lam_detection)
                    residuals = Wavefront(residuals, lam_detection=gen.lam_detection)

                    p_psf = gen.single_psf(p_wave, normed=True)
                    gt_psf = gen.single_psf(y_wave, normed=True)

                    corrected_psf = gen.single_psf(residuals)
                    corrected_noisy_img = simulate_beads(corrected_psf, psf_type=gen.psf_type, beads=reference, noise=True)
                    corrected_noisy_img /= np.max(corrected_noisy_img)

                    imwrite(save_path / f'psf_{s}.tif', noisy_img)
                    imwrite(save_path / f'corrected_psf_{s}.tif', corrected_psf)

                    task = partial(
                        vis.diagnostic_assessment,
                        psf=noisy_img,
                        gt_psf=gt_psf,
                        predicted_psf=p_psf,
                        corrected_psf=corrected_noisy_img,
                        photons=photons,
                        maxcounts=maxcounts,
                        y=y_wave,
                        pred=p_wave,
                        y_lls_defocus=y_lls_defocus,
                        p_lls_defocus=p_lls_defocus,
                        save_path=save_path / f'{s}',
                        display=False,
                        pltstyle='default'
                    )
                    _ = pool.apply_async(task)  # issue task

    pool.close()    # close the pool
    pool.join()     # wait for all tasks to complete

    return save_path


def plot_templates(model: Path, num_objs: Optional[int] = 1):
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.autolimit_mode': 'round_numbers'
    })

    num_objs = 1 if num_objs is None else num_objs

    outdir = model.with_suffix('') / 'evalmodes' / 'templates' / f'num_objs_{num_objs}'
    outdir.mkdir(parents=True, exist_ok=True)
    modelspecs = backend.load_metadata(model)

    photons = np.arange(0, 1e6+5e4, 5e4)
    photons[0] = 1e4
    waves = np.arange(1e-5, .55, step=.05).round(2)

    aberrations = np.zeros((len(waves), modelspecs.n_modes))
    gen = backend.load_metadata(model, psf_shape=(64, 64, 64))

    # plot templates
    for i in range(3, modelspecs.n_modes):
        if i == 4:
            continue

        savepath = outdir / f"m{i}"

        fig, axes = plt.subplots(nrows=len(waves), ncols=len(photons), figsize=(14, 10))

        for t, a in enumerate(waves[::-1]):
            for j, ph in enumerate(photons):
                phi = np.zeros_like(aberrations[0])
                phi[i] = a

                w = Wavefront(phi, lam_detection=gen.lam_detection)
                kernel = gen.single_psf(phi=w, meta=False)

                img = simulate_beads(
                    psf=kernel,
                    psf_type=gen.psf_type,
                    object_size=0,
                    photons=ph,
                    # maxcounts=ph,
                    noise=True,
                    fill_radius=0
                )

                axes[t, j].imshow(np.max(img, axis=0) ** .5, cmap='hot')
                axes[t, j].axis('off')
                axes[t, j].set_title(
                    f"{int(np.max(img) / 1e3)}$\\times 10^3$" if np.max(img) > 1e4 else int(np.max(img)),
                    # f"{int(np.sum(img)/1e6)}$\\times 10^6$" if np.sum(img) > 1e6 else int(np.sum(img)),
                    fontsize=8,
                    pad=1
                )

        plt.subplots_adjust(top=.9, bottom=.1, left=.1, right=.9, hspace=.15, wspace=.15)
        plt.savefig(f'{savepath}_templateheatmap.pdf', bbox_inches='tight', pad_inches=.25)
        plt.savefig(f'{savepath}_templateheatmap.png', dpi=300, bbox_inches='tight', pad_inches=.25)
        plt.savefig(f'{savepath}_templateheatmap.svg', dpi=300, bbox_inches='tight', pad_inches=.25)


def create_samples(
    wavefronts: list,
    photons: list,
    gen: SyntheticPSF,
    savepath: Any = None,
    num_objs: int = 1,
):
    kernels = [gen.single_psf(phi=w, normed=True) for w in wavefronts]

    if Path(f"{savepath}_inputs.npy").exists():
        inputs = np.load(f"{savepath}_inputs.npy")
    else:
        inputs = np.stack([
                simulate_beads(
                    psf=kernels[k],
                    psf_type=gen.psf_type,
                    object_size=0,
                    num_objs=num_objs,
                    photons=ph,
                    # maxcounts=ph,
                    noise=True,
                    fill_radius=0 if num_objs == 1 else .35
                )
            for k, ph in tqdm(
                itertools.product(range(len(kernels)), photons),
                desc='Generating samples',
                total=len(kernels) * len(photons),
                file=sys.stdout
            )
        ], axis=0)[..., np.newaxis]
        np.save(f"{savepath}_inputs", inputs)
    return inputs


@profile
def eval_object(
    wavefronts,
    modelpath,
    photons: list,
    na: float = 1.0,
    batch_size: int = 512,
    num_objs: int = 1,
    eval_sign: str = 'signed',
    savepath: Any = None,
    psf_type: Any = None,
    digital_rotations: Optional[int] = 361
):
    model = backend.load(modelpath)
    gen = backend.load_metadata(
        modelpath,
        psf_type=psf_type,
        psf_shape=3*[model.input_shape[2]],
        rotate=False
    )

    if not isinstance(wavefronts[0], Wavefront):
        wavefronts = [Wavefront(w, lam_detection=gen.lam_detection, rotate=False) for w in wavefronts]

    p2v = [w.peak2valley(na=na) for w in wavefronts]

    inputs = create_samples(
        wavefronts=wavefronts,
        photons=photons,
        gen=gen,
        savepath=savepath,
        num_objs=num_objs,
    )

    if Path(f"{savepath}_embeddings.npy").exists():
        embeddings = np.load(f"{savepath}_embeddings.npy")
    else:
        embeddings = np.stack([
            backend.preprocess(
                i,
                modelpsfgen=gen,
                digital_rotations=digital_rotations,
                remove_background=True,
                normalize=True,
            )
            for i in tqdm(inputs, desc='Generating fourier embeddings', total=inputs.shape[0], file=sys.stdout)
        ], axis=0)
        np.save(f"{savepath}_embeddings", embeddings)

    ys = np.stack([w.amplitudes for w, ph in itertools.product(wavefronts, photons)])

    embeddings = tf.data.Dataset.from_tensor_slices(embeddings)

    if Path(f"{savepath}_predictions.npy").exists():
        preds = np.load(f"{savepath}_predictions.npy")
    else:
        res = backend.predict_dataset(
            model,
            inputs=embeddings,
            psfgen=gen,
            batch_size=batch_size,
            save_path=[f"{savepath}_{a}_{ph}" for a, ph in itertools.product(p2v, photons)],
            digital_rotations=digital_rotations,
            # plot_rotations=True
        )

        try:
            preds, stdev = res
        except ValueError:
            preds, stdev, lls_defocus = res

        np.save(f"{savepath}_predictions", preds)

    if eval_sign == 'positive_only':
        ys = np.abs(ys)
        preds = np.abs(preds)[:, :ys.shape[-1]]

    residuals = ys - preds

    p = pd.DataFrame([p for p, ph in itertools.product(p2v, photons)], columns=['aberration'])
    p['prediction'] = [Wavefront(i, lam_detection=gen.lam_detection).peak2valley(na=na) for i in preds]
    p['residuals'] = [Wavefront(i, lam_detection=gen.lam_detection).peak2valley(na=na) for i in residuals]
    p['photons'] = np.concatenate([photons for i in itertools.product(wavefronts)])
    p['counts'] = [np.sum(i) for i in inputs]

    for percentile in range(1, 101):
        p[f'counts_p{percentile}'] = [np.percentile(i, percentile) for i in inputs]

    if savepath is not None:
        p.to_csv(f'{savepath}_predictions_num_objs_{num_objs}.csv')

    return p


@profile
def evaluate_modes(
    model: Path,
    eval_sign: str = 'signed',
    batch_size: int = 512,
    num_objs: Optional[int] = 1,
    digital_rotations: bool = True,
    agg: str = 'median'
):
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.autolimit_mode': 'round_numbers'
    })

    num_objs = 1 if num_objs is None else num_objs

    outdir = model.with_suffix('') / eval_sign / 'evalmodes' / f'num_objs_{num_objs}'
    outdir.mkdir(parents=True, exist_ok=True)
    modelspecs = backend.load_metadata(model)

    photons = np.arange(0, 1e6+5e4, 5e4)
    photons[0] = 1e4
    waves = np.arange(1e-5, .55, step=.05).round(2)
    aberrations = np.zeros((len(waves), modelspecs.n_modes))
    gen = backend.load_metadata(model, psf_shape=(64, 64, 64))

    # plot_templates(model=model, num_objs=1)

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

    for i in range(3, modelspecs.n_modes):
        if i == 4:
            continue

        savepath = outdir / f"m{i}"

        classes = aberrations.copy()
        classes[:, i] = waves

        path = Path(f'{savepath}_predictions_num_objs_{num_objs}.csv')
        if path.exists():
            df = pd.read_csv(path, index_col=0, header=0)
        else:
            df = eval_object(
                phi=classes,
                num_objs=num_objs,
                photons=photons,
                modelpath=model,
                batch_size=batch_size,
                eval_sign=eval_sign,
                savepath=savepath,
                digital_rotations=361 if digital_rotations else None
            )

        df['photoelectrons'] = utils.photons2electrons(df['photons'], quantum_efficiency=.82)

        x, y = 'photons', 'aberration'
        xstep, ystep = 5e4, .25

        ybins = np.arange(0, df[y].max()+ystep, ystep)
        df['ybins'] = pd.cut(df[y], ybins, labels=ybins[1:], include_lowest=True)

        if x == 'photons':
            xbins = df['photons'].values
            df['xbins'] = xbins
        else:
            xbins = np.arange(0, df[x].max()+xstep, xstep)
            df['xbins'] = pd.cut(df[x], xbins, labels=xbins[1:], include_lowest=True)

        dataframe = pd.pivot_table(df, values='residuals', index='ybins', columns='xbins', aggfunc=agg)
        dataframe = dataframe.sort_index().interpolate()

        fig = plt.figure(figsize=(8, 8))
        gs = fig.add_gridspec(4, 4)
        ax_xy = fig.add_subplot(gs[0, 0])
        ax_xz = fig.add_subplot(gs[0, 1])
        ax_yz = fig.add_subplot(gs[0, 2])
        ax_wavevfront = fig.add_subplot(gs[0, -1])

        axt = fig.add_subplot(gs[1:, :])

        contours = axt.contourf(
            dataframe.columns,
            dataframe.index.values,
            dataframe.values,
            cmap=cmap,
            levels=levels,
            extend='max',
            linewidths=2,
            linestyles='dashed',
        )
        axt.patch.set(hatch='/', edgecolor='lightgrey', lw=.01)

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

        cbar.ax.set_ylabel(rf'Residuals ({agg} peak-to-valley, $\lambda = {int(modelspecs.lam_detection*1000)}~nm$)')
        cbar.ax.set_title(r'$\lambda$')
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.yaxis.set_label_position('left')

        axt.set_ylabel(rf'Initial aberration ({agg} peak-to-valley, $\lambda = {int(modelspecs.lam_detection*1000)}~nm$)')
        axt.set_yticks(np.arange(0, 6, .5), minor=True)
        axt.set_yticks(np.arange(0, 6, 1))
        axt.set_ylim(ybins[0], ybins[-1])

        # axt.set_xscale('log')
        axt.set_xlim(xbins[0], xbins[-1])
        axt.set_xlabel(f'{x}')

        axt.spines['right'].set_visible(False)
        axt.spines['left'].set_visible(False)
        axt.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

        phi = np.zeros_like(classes[-1, :])
        phi[i] = .2
        w = Wavefront(phi, lam_detection=gen.lam_detection)
        kernel = gen.single_psf(
            phi=w,
            normed=True,
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
        plt.savefig(f'{savepath}.svg', dpi=300, bbox_inches='tight', pad_inches=.25)

    return savepath


@profile
def eval_modalities(
    model: Path,
    photons: int = 3e5,
    lam_detection: float = .510,
    batch_size: int = 512,
    eval_sign: str = 'signed',
    digital_rotations: bool = False,
    num_objs: int = 1,
    psf_shape: tuple = (128, 128, 128),  # needs to be large enough for 2photon
    modalities: tuple = (
        '../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat',
        '../lattice/ACHex_NAexc0p40_NAsigma0p075_annulus0p6-0p2_crop0p1_FWHM52p0.mat',
        '../lattice/Gaussian_NAexc0p21_NAsigma0p21_annulus0p4-0p2_crop0p1_FWHM51p0.mat',
        '../lattice/MBHex_NAexc0p43_annulus0p47_0p40_crop0p08_FWHM48p0.mat',
        '../lattice/MBSq_NAexc0p30_annulus0p375-0p225_FWHM48p5.mat',
        '../lattice/Sinc_by_lateral_SW_NAexc0p32_NAsigma5p0_annulus0p4-0p2_realSLM_FWHM51p5.mat',
        '../lattice/v2Hex_NAexc0p50_NAsigma0p075_annulus0p60-0p40_FWHM53p0.mat',
        '../lattice/v2HexRect_NAexc0p50_NAsigma0p15_annulus0p60-0p40_FWHM_56p0.mat',
        'widefield',
        'confocal',
        '2photon',
    )
):
    m = backend.load(model)
    m.summary()
    no_phase = True if m.input_shape[1] == 3 else False

    reference = multipoint_dataset.beads(
        photons=photons,
        image_shape=psf_shape,
        object_size=0,
        num_objs=num_objs,
        fill_radius=.3 if num_objs > 1 else 0
    )

    lam_2photon = .920

    modalities_generators = [
        backend.load_metadata(
            model,
            signed=False if eval_sign == 'positive_only' else True,
            rotate=True,
            mode_weights='pyramid',
            psf_type=psf_type,
            lam_detection=lam_2photon if psf_type == '2photon' else lam_detection
        )
        for psf_type in modalities
    ]

    for dist in ['single']:
        for amp in [0, .1]:
            for z in range(3, 15):
                if z == 4:
                    continue

                if amp == 0 and z > 3:
                    break

                if dist == 'single':
                    amplitudes = np.zeros(15)
                    amplitudes[z] = amp
                else:
                    amplitudes = np.repeat(amp, 2)

                for i, gen in enumerate(modalities_generators):
                    mode = modalities[i].replace('../lattice/', '').split('_')[0]

                    phi = Wavefront(
                        # boost um RMS aberration amplitudes for '2photon', so we create equivalent p2v aberrations
                        lam_2photon/lam_detection * amplitudes if modalities_generators[i].psf_type == '2photon' else amplitudes,
                        modes=modalities_generators[i].n_modes,
                        distribution=dist,
                        signed=False if eval_sign == 'positive_only' else True,
                        rotate=True,
                        mode_weights='pyramid',
                        lam_detection=modalities_generators[i].lam_detection,
                    )

                    # aberrated PSF without noise
                    psf, y, y_lls_defocus = gen.single_psf(
                        phi=phi,
                        normed=True,
                        meta=True,
                        lls_defocus_offset=None
                    )

                    noisy_img = simulate_beads(psf, psf_type=gen.psf_type, beads=reference, noise=True)
                    noisy_img -= 100
                    maxcounts = np.max(noisy_img)
                    noisy_img /= maxcounts

                    save_path = Path(
                        f"{model.with_suffix('')}/"
                        f"{eval_sign}/"
                        f"modalities/"
                        f"{dist}/"
                        f"um-{amp}/"
                        f"mode-{mode}"
                    )
                    save_path.mkdir(exist_ok=True, parents=True)

                    embeddings = backend.preprocess(
                        noisy_img,
                        modelpsfgen=gen,
                        digital_rotations=361 if digital_rotations else None,
                        remove_background=False,
                        normalize=True,
                        plot=save_path / f'{z}',
                    )
                    imwrite(save_path / f'{z}_embeddings.tif', embeddings.astype(np.float32), imagej=True)

                    if digital_rotations:
                        res = backend.predict_rotation(
                            m,
                            embeddings,
                            save_path=save_path / f'{z}',
                            psfgen=gen,
                            no_phase=no_phase,
                            batch_size=batch_size,
                            plot=save_path / f'{z}',
                            plot_rotations=save_path / f'{z}',
                        )
                    else:
                        res = backend.bootstrap_predict(
                            m,
                            embeddings,
                            psfgen=gen,
                            no_phase=no_phase,
                            batch_size=batch_size,
                            plot=save_path / f'{z}',
                        )

                    try:
                        p, std = res
                        p_lls_defocus = None
                    except ValueError:
                        p, std, p_lls_defocus = res

                    if eval_sign == 'positive_only':
                        y = np.abs(y)
                        if len(p.shape) > 1:
                            p = np.abs(p)[:, :y.shape[-1]]
                        else:
                            p = np.abs(p)[np.newaxis, :y.shape[-1]]
                    else:
                        if len(p.shape) > 1:
                            p = p[:, :y.shape[-1]]
                        else:
                            p = p[np.newaxis, :y.shape[-1]]

                    residuals = y - p

                    p_wave = Wavefront(p, lam_detection=gen.lam_detection)
                    y_wave = Wavefront(y, lam_detection=gen.lam_detection)
                    residuals = Wavefront(residuals, lam_detection=gen.lam_detection)

                    p_psf = gen.single_psf(p_wave, normed=True)
                    gt_psf = gen.single_psf(y_wave, normed=True)

                    corrected_psf = gen.single_psf(residuals)
                    corrected_noisy_img = simulate_beads(corrected_psf, psf_type=gen.psf_type, beads=reference, noise=True)
                    corrected_noisy_img /= np.max(corrected_noisy_img)

                    if amp == 0:
                        imwrite(save_path / f'{z}_na_mask.tif', gen.na_mask().astype(np.float32))

                    imwrite(save_path / f'{z}_input.tif', noisy_img.astype(np.float32))

                    imwrite(save_path / f'{z}_pred_psf.tif', p_psf.astype(np.float32))
                    imwrite(save_path / f'{z}_pred_wavefront.tif', p_wave.wave().astype(np.float32))

                    imwrite(save_path / f'{z}_gt_psf.tif', gt_psf.astype(np.float32))
                    imwrite(save_path / f'{z}_gt_wavefront.tif', y_wave.wave().astype(np.float32))

                    imwrite(save_path / f'{z}_corrected_psf.tif', corrected_psf.astype(np.float32))
                    imwrite(save_path / f'{z}_corrected_wavefront.tif', residuals.wave().astype(np.float32))

                    processed_input = backend.prep_sample(
                        noisy_img,
                        model_fov=gen.psf_fov,  # this is what we will crop to
                        sample_voxel_size=gen.voxel_size,
                        remove_background=True,
                        normalize=True,
                    )

                    processed_corrected = backend.prep_sample(
                        corrected_noisy_img,
                        model_fov=gen.psf_fov,  # this is what we will crop to
                        sample_voxel_size=gen.voxel_size,
                        remove_background=True,
                        normalize=True,
                    )

                    vis.diagnostic_assessment(
                        psf=processed_input,
                        gt_psf=gt_psf,
                        predicted_psf=p_psf,
                        corrected_psf=processed_corrected,
                        photons=photons,
                        maxcounts=maxcounts,
                        y=y_wave,
                        pred=p_wave,
                        y_lls_defocus=y_lls_defocus,
                        p_lls_defocus=p_lls_defocus,
                        save_path=save_path / f'{z}_psf',
                        display=False,
                        pltstyle='default'
                    )

                    vis.diagnostic_assessment(
                        psf=processed_input,
                        gt_psf=gt_psf,
                        predicted_psf=p_psf,
                        corrected_psf=processed_corrected,
                        photons=photons,
                        maxcounts=maxcounts,
                        y=y_wave,
                        pred=p_wave,
                        y_lls_defocus=y_lls_defocus,
                        p_lls_defocus=p_lls_defocus,
                        display_otf=True,
                        save_path=save_path / f'{z}_otf',
                        display=False,
                        pltstyle='default'
                    )

    return save_path
