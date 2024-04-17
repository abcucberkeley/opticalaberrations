import itertools

import matplotlib

matplotlib.use('Agg')
import tempfile
import shutil

from multiprocessing import Pool

import ujson
import logging
import sys
from functools import partial
from pathlib import Path
from typing import Any, Optional, Union
from matplotlib.ticker import FormatStrFormatter, PercentFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as mcolors
import seaborn as sns

import matplotlib.pyplot as plt
plt.set_loglevel('error')

import numpy as np
import pandas as pd
import tensorflow as tf
from line_profiler_pycharm import profile
from tqdm import tqdm, trange
from tifffile import imwrite
from astropy import convolution
from csbdeep.utils.tf import limit_gpu_memory

limit_gpu_memory(allow_growth=True, fraction=None, total_memory=None)
from csbdeep.models import CARE

import utils
import data_utils
import profile_utils
import backend
import vis
import multipoint_dataset
from preprocessing import denoise_image
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
    fill_radius=.66,
    fast=False,
    scale_by_maxcounts=None
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
        if psf_type == 'widefield':
            psf /= psf.max()
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

    if psf_type == 'widefield':  # scale widefield PSF by maxcounts of the GT PSF
        inputs /= np.max(inputs)
        inputs *= utils.electrons2photons(utils.counts2electrons(scale_by_maxcounts))

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
    plot: bool = False,
    simulate_psf_only: bool = False,
    preprocess: bool = False,
    file_format: str = 'tif',
    denoiser: Optional[Union[Path, CARE]] = None,
    denoiser_window_size: tuple = (32, 64, 64),
):
    hashtable = data[data['id'] == image_id].iloc[0].to_dict()
    f = Path(str(hashtable['file']))
    beads = Path(str(hashtable['beads']))

    if savedir is not None:
        outdir = savedir / '/'.join(beads.parent.parts[-4:]) / f'iter_{iter_number}'
        outdir.mkdir(exist_ok=True, parents=True)
        savepath = outdir / f"{f.with_suffix('').name}.{file_format}"

    ys = [hashtable[cc] for cc in data.columns[data.columns.str.endswith('_residual')]]
    ref = np.squeeze(data_utils.get_image(beads))
    
    if hashtable['object_gaussian_sigma'] > 0:
        kernel = utils.gaussian_kernel(kernlen=[21, 21, 21], std=hashtable['object_gaussian_sigma'])
        
        # convolve template with the input image
        ref = convolution.convolve_fft(
            ref,
            kernel,
            allow_huge=True,
            boundary='fill',
            nan_treatment='fill',
            fill_value=0,
            normalize_kernel=np.sum
        )

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

    if simulate_psf_only:
        noisy_img = simulate_beads(
            psf=psf,
            psf_type=psfgen.psf_type,
            beads=None,
            fill_radius=0,
            object_size=0,
            photons=hashtable['photons'],
            scale_by_maxcounts=hashtable['counts_p100'] if psfgen.psf_type == 'widefield' else None
        )
    else:
        noisy_img = simulate_beads(
            psf=psf,
            psf_type=psfgen.psf_type,
            beads=ref,
            photons=hashtable['photons'],
            scale_by_maxcounts=hashtable['counts_p100'] if psfgen.psf_type == 'widefield' else None
        )
    
    if savedir is not None:
        init_path = Path(f"{savepath.with_suffix('')}_not_processed")
        imwrite(init_path.with_suffix('.tif'), noisy_img.astype(np.float32), compression='deflate', dtype=np.float32)
    
    if preprocess:
        noisy_img = backend.preprocess(
            noisy_img,
            modelpsfgen=psfgen,
            digital_rotations=digital_rotations,
            no_phase=no_phase,
            remove_background=True,
            normalize=True,
            min_psnr=0,
            plot=savepath.with_suffix('') if plot else None,
            denoiser=denoiser,
            denoiser_window_size=denoiser_window_size,
        )
    
    if denoiser is not None and not preprocess:
        noisy_img = denoise_image(
            image=noisy_img,
            denoiser=denoiser,
            denoiser_window_size=denoiser_window_size,
        )
        
    if savedir is not None:
        if file_format == 'tif':
            imwrite(
                savepath,
                data=noisy_img.astype(np.float32),
                compression='deflate',
                dtype=np.float32,
                imagej=True,
            )
        else:
            np.savez_compressed(
                Path(f"{savepath.with_suffix('')}"), noisy_img.astype(np.float32),
                allow_pickle=True
            )
        return savepath
    
    else:
        return noisy_img


@profile
def eval_template(shape, psf_type, lam_detection):
    return {
        # image number where the voxel locations of the beads are given in 'file'. Constant over iterations.
        'id': np.arange(shape[0], dtype=int),
        'iter_num': np.zeros(shape[0], dtype=int),          # iteration index.
        'aberration': np.zeros(shape[0], dtype=float),      # initial p2v aberration. Constant over iterations.
        'aberration_umRMS': np.zeros(shape[0], dtype=float),  # initial rms aberration. Constant over iterations.
        'residuals': np.zeros(shape[0], dtype=float),       # remaining p2v aberration after ML correction.
        'residuals_umRMS': np.zeros(shape[0], dtype=float), # remaining umRMS aberration after ML correction.
        'confidence': np.zeros(shape[0], dtype=float),      # model's confidence for the primary mode (waves)
        'confidence_sum': np.zeros(shape[0], dtype=float),  # model's confidence for the all modes (waves)
        'confidence_umRMS': np.zeros(shape[0], dtype=float),# model's confidence for the all modes (umRMS)
        'photons': np.zeros(shape[0], dtype=int),           # integrated photons
        'counts': np.zeros(shape[0], dtype=int),            # integrated counts
        'counts_mode': np.zeros(shape[0], dtype=int),       # counts mode
        'distance': np.zeros(shape[0], dtype=float),        # average distance to nearst bead
        'neighbors': np.zeros(shape[0], dtype=int),         # number of beads
        'file': np.empty(shape[0], dtype=Path),             # path to realspace images
        'file_windows': np.empty(shape[0], dtype=Path),     # stupid windows path
        'beads': np.zeros(shape[0], dtype=Path),
        'psf_type': np.full(shape[0], dtype=str, fill_value=psf_type),
        'wavelength': np.full(shape[0], dtype=float, fill_value=lam_detection),
        # path to binary image file filled with zeros except at location of beads
    }


def collect_data(
    datapath,
    model_output_shape: int,
    model_input_shape: tuple,
    samplelimit: int = 1,
    distribution: str = '/',
    no_phase: bool = False,
    photons_range: Optional[tuple] = None,
    npoints_range: Optional[tuple] = None,
    psf_type: Optional[str] = None,
    lam_detection: Optional[float] = None,
    default_wavelength: Optional[float] = .510,
    filename_pattern: str = r"*[!_gt|!_realspace|!_noisefree|!_predictions_psf|!_corrected_psf|!_reconstructed_psf].tif"
):

    dataset = data_utils.collect_dataset(
        datapath,
        metadata=True,
        modes=model_output_shape,
        samplelimit=samplelimit,
        distribution=distribution,
        no_phase=no_phase,
        photons_range=photons_range,
        npoints_range=npoints_range,
        filename_pattern=filename_pattern,
        model_input_shape=model_input_shape
    )

    # This runs multiple samples (aka images) at a time.
    # ys is a 2D array, rows are each sample, columns give aberration in zernike coefficients
    dataset = np.array(list(dataset.take(-1)), dtype=object)
    ys = np.zeros((dataset.shape[0], model_output_shape))
    counts_percentiles = np.zeros((dataset.shape[0], 100))
    results = eval_template(shape=dataset.shape, psf_type=psf_type, lam_detection=lam_detection)

    # see `data_utils.get_sample` to check order of objects returned
    for i in range(dataset.shape[0]):

        # rescale zernike amplitudes to maintain the same peak2valley for different PSFs
        ys[i] = lam_detection / default_wavelength * dataset[i, 0].numpy()[:model_output_shape]
        results['residuals_umRMS'][i] = np.linalg.norm(ys[i])

        results['photons'][i] = dataset[i, 1].numpy()
        results['counts'][i] = dataset[i, 2].numpy()
        results['counts_mode'][i] = dataset[i, 3].numpy()

        counts_percentiles[i] = dataset[i, 4].numpy()

        results['aberration_umRMS'][i] = np.linalg.norm(ys[i])
        results['aberration'][i] = dataset[i, 5].numpy()
        results['residuals'][i] = results['aberration'][i]

        results['neighbors'][i] = dataset[i, 7].numpy()
        results['distance'][i] = dataset[i, 8].numpy()

        f = Path(str(dataset[i, -1].numpy(), "utf-8"))
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
        results[f'z{z}_confidence'] = np.zeros_like(ys[:, z])
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
    plot: bool = False,
    plot_rotations: bool = False,
    psf_type: Optional[str] = None,
    lam_detection: Optional[float] = .510,
    filename_pattern: str = r"*[!_gt|!_realspace|!_noisefree|!_predictions_psf|!_corrected_psf|!_reconstructed_psf].tif",
    preprocess: bool = False,
    skip_remove_background: bool = False,
    simulate_psf_only: bool = False,
    use_theoretical_widefield_simulator: bool = False,
    denoiser: Optional[Union[Path, CARE]] = None,
    denoiser_window_size: tuple = (32, 64, 64),
    simulate_samples: bool = False,
    estimated_object_gaussian_sigma: float = 0,
    randomize_object_gaussian_sigma: Optional[np.ndarray | list | tuple] = None,
):
    """
    Gathers the set of .tif files that meet the input criteria.
    Predicts on all of those for (iter_num) iterations.
    Results go into dataframe called "results"
    Saves "results" dataframe to _predictions.csv file

    Returns:
        "results" dataframe
    """
    model = backend.load(modelpath, model_arch=modelpath.name.split('-')[0])

    gen = backend.load_metadata(
        modelpath,
        signed=True,
        rotate=False,
        batch_size=batch_size,
        psf_shape=3 * [model.input_shape[2]],
        psf_type=psf_type,
        lam_detection=lam_detection,
        skip_remove_background_ideal_psf=skip_remove_background,
        use_theoretical_widefield_simulator=use_theoretical_widefield_simulator,
    )
    
    if Path(f'{savepath}_predictions.csv').exists():
        # continue from previous results, ignoring criteria
        results = pd.read_csv(f'{savepath}_predictions.csv', header=0, index_col=0)

        if iter_num == results['iter_num'].values.max():
            return results  # already computed

    else:
        # on first call, setup the dataframe with the 0th iteration stuff
        results = collect_data(
            datapath=datapath,
            model_output_shape=model.output_shape[-1],
            model_input_shape=model.input_shape[1:],
            samplelimit=samplelimit,
            distribution=distribution,
            no_phase=no_phase,
            photons_range=photons_range,
            npoints_range=npoints_range,
            psf_type=gen.psf_type,
            lam_detection=gen.lam_detection,
            filename_pattern=filename_pattern
        )
    
    if 'object_gaussian_sigma' not in results.columns:
        results['object_gaussian_sigma'] = 0.
    
    if randomize_object_gaussian_sigma is not None:
        results['object_gaussian_sigma'] = np.random.choice(randomize_object_gaussian_sigma, size=results.shape[0])
    
    prediction_cols = [col for col in results.columns if col.endswith('_prediction')]
    confidence_cols = [col for col in results.columns if col.endswith('_confidence')]
    ground_truth_cols = [col for col in results.columns if col.endswith('_ground_truth')]
    residual_cols = [col for col in results.columns if col.endswith('_residual')]
    previous = results[results['iter_num'] == iter_num - 1]   # previous iteration = iter_num - 1
    
    if denoiser is not None:
        logger.info(f"Loading denoiser model: {denoiser}")
        denoiser = CARE(config=None, name=denoiser.name, basedir=denoiser.parent)
        logger.info(f"{denoiser.name} loaded")
    
    print(previous)
    # create realspace images for the current iteration
    if simulate_samples:
        paths = utils.multiprocess(
            func=partial(
                generate_sample,
                iter_number=iter_num,
                savedir=savepath.resolve(),
                data=previous,
                psfgen=gen,
                no_phase=no_phase,
                digital_rotations=rotations if digital_rotations else None,
                preprocess=preprocess,
                plot=plot,
                denoiser=denoiser,
                denoiser_window_size=denoiser_window_size,
                simulate_psf_only=simulate_psf_only,
            ),
            jobs=previous['id'].values,
            desc=f'Create samples ({savepath.resolve()})',
            unit=' sample',
            cores=-1
        )
    else:
        paths = previous.file.values
    
    current = previous.copy()
    current['iter_num'] = iter_num
    current['file'] = paths
    current['file_windows'] = [utils.convert_to_windows_file_string(f) for f in paths]

    predictions, stdevs = backend.predict_files(
        paths=np.hstack(paths) if digital_rotations else paths,
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
        min_psnr=0,
        skip_prep_sample=False,
        preprocessed=preprocess,
        remove_background=False if skip_remove_background else True,
        estimated_object_gaussian_sigma=estimated_object_gaussian_sigma,
        save_processed_tif_file=True
    )
    current[prediction_cols] = predictions.T.values[:paths.shape[0]]  # drop (mean, median, min, max, and std)
    current[confidence_cols] = stdevs.T.values[:paths.shape[0]]  # drop (mean, median, min, max, and std)
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

    primary_modes = current[prediction_cols].idxmax(axis=1).replace(r'_prediction', r'_confidence', regex=True)
    current['confidence'] = [
        utils.microns2waves(current.loc[i, primary_modes[i]], wavelength=gen.lam_detection)
        for i in primary_modes.index.values
    ]

    current['confidence_sum'] = current.apply(
        lambda row: utils.microns2waves(np.sum(row[confidence_cols].values), wavelength=gen.lam_detection),
        axis=1
    )

    current['confidence_umRMS'] = current.apply(
        lambda row: np.linalg.norm(row[confidence_cols].values),
        axis=1
    )
    print(current)

    results = pd.concat([results, current], ignore_index=True, sort=False)

    if savepath is not None:
        try:
            results.to_csv(f'{savepath}_predictions.csv')
        except PermissionError:
            savepath = f'{savepath}_x'
            results.to_csv(f'{savepath}_predictions.csv')
        logger.info(f'Saved: {savepath.resolve()}_predictions.csv')

    return results


@profile
def plot_coverage(
    dataframe,
    wavelength,
    savepath:Path,
    label='Integrated photons per object',
    lims=(0, 100),
    ax=None,
    cax=None,
    sci=False,
    p2v=False,
    cmap='viridis'
):
    try:
        dataframe = dataframe.sort_index().interpolate()
    except ValueError:
        pass

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

    contours = ax.contourf(
        dataframe.columns.values,
        dataframe.index.values,
        dataframe.values,
        cmap=cmap,
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
    )
    cbar.ax.set_ylabel(fr'Number of test samples')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('left')

    if label == 'Integrated photons' or label == 'Integrated photoelectrons':
        ax.set_xticks(np.arange(lims[0], lims[1]+5e4, 5e4), minor=False)
        ax.set_xticks(np.arange(lims[0], lims[1]+2.5e4, 2.5e4), minor=True)
    elif label == 'Number of iterations':
        ax.set_xticks(np.arange(0, dataframe.columns.values.max()+1, 1), minor=False)

    if sci:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)

    ax.set_xlabel(label)
    ax.set_xlim(lims)

    if p2v:
        ax.set_ylabel(rf'Initial aberration (peak-to-valley, $\lambda = {int(wavelength * 1000)}~nm$)')
        ax.set_yticks(np.arange(0, 6, .5), minor=True)
        ax.set_yticks(np.arange(0, 6, 1))
        ax.set_ylim(0, 5)
    else:
        ax.set_ylabel(rf'Initial aberration ($\lambda$ RMS, $\lambda = {int(wavelength * 1000)}~nm$)')
        ax.set_yticks(np.arange(0, 1.1, .05), minor=True)
        ax.set_yticks(np.arange(0, 1.1, .1))
        ax.set_ylim(0, 1)

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)
    plt.tight_layout()

    if p2v:
        savepath = Path(f'{savepath}_p2v')
    else:
        savepath = Path(f'{savepath}_umRMS')

    plt.savefig(f'{savepath}.pdf', bbox_inches='tight', pad_inches=.25)
    plt.savefig(f'{savepath}.png', dpi=300, bbox_inches='tight', pad_inches=.25)
    plt.savefig(f'{savepath}.svg', dpi=300, bbox_inches='tight', pad_inches=.25)

    logger.info(f'Saved: {savepath.resolve()}.png  .pdf  .svg')
    return ax


@profile
def plot_heatmap_p2v(
    dataframe,
    wavelength,
    savepath: Path,
    label='Integrated photoelectrons',
    color_label='Residuals',
    hist_col='confidence',
    lims=(0, 100),
    ax=None,
    cax=None,
    agg='mean',
    sci=False,
    histograms: Optional[pd.DataFrame] = None,
    kde_color='grey',
    cdf_color='k',
    hist_color='lightgrey',
):
    try:
        dataframe = dataframe.sort_index().interpolate()
    except ValueError:
        pass

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

    if color_label == 'Residuals':
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
        cbar.ax.set_ylabel(rf'Residuals ({agg} peak-to-valley, $\lambda = {int(wavelength * 1000)}~nm$)')
    else:
        if hist_col == 'confidence':
            ticks = np.arange(0, .11, step=.01)
        else:
            ticks = np.arange(0, .275, step=.025)

        contours = ax.contourf(
            dataframe.columns.values,
            dataframe.index.values,
            dataframe.values,
            cmap='nipy_spectral',
            levels=ticks,
            extend='max',
            linewidths=2,
            linestyles='dashed',
        )
        cbar = plt.colorbar(
            contours,
            cax=cax,
            fraction=0.046,
            pad=0.04,
            extend='both',
            spacing='proportional',
            format=FormatStrFormatter("%.2f"),
            ticks=ticks,
        )

        if hist_col == 'confidence':
            cbar.ax.set_ylabel(rf'Standard deviation: ({agg} $\hat{{\sigma}}$, $\lambda = {int(wavelength * 1000)}~nm$)')
        else:
            cbar.ax.set_ylabel(rf'Standard deviation: ({agg} $\sum{{\sigma_i}}$, $\lambda = {int(wavelength * 1000)}~nm$)')

    ax.patch.set(hatch='/', edgecolor='lightgrey', lw=.01)
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('left')

    if histograms is not None:
        if label == 'Integrated photons' or label == 'Integrated photoelectrons':
            x = histograms[
                (histograms.pbins <= 1e5) &
                (histograms.ibins >= 1.5) & (histograms.ibins <= 2.5)
            ]

            if color_label == 'Residuals':
                xmax = 3
                binwidth = .25
                bins = np.arange(0, xmax + binwidth, binwidth)
                xticks = np.arange(0, xmax+.5, .5)
            else:
                if hist_col == 'confidence':
                    xmax = .15
                    binwidth = .01
                    bins = np.arange(0, xmax + binwidth, binwidth)
                    xticks = np.arange(0, xmax+.05, .05)
                else:
                    xmax = .3
                    binwidth = .025
                    bins = np.arange(0, xmax + binwidth, binwidth)
                    xticks = np.arange(0, xmax+.05, .05)

            try:
                ax1t = ax1.twinx()
                ax1t = sns.histplot(
                    ax=ax1t,
                    data=x,
                    x=hist_col,
                    stat='percent',
                    kde=True,
                    bins=bins,
                    color=hist_color,
                    element="step",
                )
                ax1t.lines[0].set_color(kde_color)
                ax1t.tick_params(axis='y', labelcolor=kde_color, color=kde_color)
                ax1t.set_ylabel('KDE', color=kde_color)
                ax1t.set_ylim(0, 30)

                ax1 = sns.histplot(
                    ax=ax1,
                    data=x,
                    x=hist_col,
                    stat='proportion',
                    color=cdf_color,
                    bins=bins,
                    element="poly",
                    fill=False,
                    cumulative=True,
                )

                ax1.tick_params(axis='y', labelcolor=cdf_color, color=cdf_color)
                ax1.set_ylabel('CDF', color=cdf_color)
                ax1.set_ylim(0, 1)
                ax1.set_yticks(np.arange(0, 1.2, .2))
                ax1.axvline(np.median(x[hist_col]), c='C0', ls='--', lw=2, label='Median', zorder=3)
                ax1.axvline(np.mean(x[hist_col]), c='C1', ls=':', lw=2, label='Mean', zorder=3)
                ax1.set_xlim(0, xmax)
                ax1.set_xticks(xticks)
                ax1.set_xlabel(color_label)
                ax1.text(
                    .9, .8, 'I',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=20,
                    color='k',
                    transform=ax1.transAxes
                )
            except IndexError:
                pass

            ax.add_patch(
                plt.Rectangle((0, .3), .2, .2, ec="k", fc="none", transform=ax.transAxes)
            )
            ax.text(
                .1, .4, 'I',
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

            try:
                ax2t = ax2.twinx()
                ax2t = sns.histplot(
                    ax=ax2t,
                    data=x,
                    x=hist_col,
                    stat='percent',
                    kde=True,
                    bins=bins,
                    color=hist_color,
                    element="step",
                )
                ax2t.lines[0].set_color(kde_color)
                ax2t.tick_params(axis='y', labelcolor=kde_color, color=kde_color)
                ax2t.set_ylabel('KDE', color=kde_color)
                ax2t.set_ylim(0, 30)

                ax2 = sns.histplot(
                    ax=ax2,
                    data=x,
                    x=hist_col,
                    stat='proportion',
                    color=cdf_color,
                    bins=bins,
                    element="poly",
                    fill=False,
                    cumulative=True,
                )
                ax2.tick_params(axis='y', labelcolor=cdf_color, color=cdf_color)
                ax2.set_ylabel('CDF', color=cdf_color)
                ax2.set_ylim(0, 1)
                ax2.set_yticks(np.arange(0, 1.2, .2))
                ax2.axvline(np.median(x[hist_col]), c='C0', ls='--', lw=2, zorder=3)
                ax2.axvline(np.mean(x[hist_col]), c='C1', ls=':', lw=2, zorder=3)
                ax2.set_xlim(0, xmax)
                ax2.set_xticks(xticks)
                ax2.set_xlabel(color_label)
                ax2.text(
                    .9, .8, 'II',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=20,
                    color='k',
                    transform=ax2.transAxes
                )
            except IndexError:
                pass

            ax.add_patch(
                plt.Rectangle((.4, .3), .2, .2, ec="k", fc="none", transform=ax.transAxes)
            )
            ax.text(
                .5, .4, 'II',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=20,
                color='k',
                transform=ax.transAxes
            )

            x = histograms[
                (histograms.pbins >= 4e5) & (histograms.pbins <= 5e5) &
                (histograms.ibins >= 1.5) & (histograms.ibins <= 2.5)
            ]

            try:
                ax3t = ax3.twinx()
                ax3t = sns.histplot(
                    ax=ax3t,
                    data=x,
                    x=hist_col,
                    stat='percent',
                    kde=True,
                    bins=bins,
                    color=hist_color,
                    element="step",
                )
                ax3t.lines[0].set_color(kde_color)
                ax3t.tick_params(axis='y', labelcolor=kde_color, color=kde_color)
                ax3t.set_ylabel('KDE', color=kde_color)
                ax3t.set_ylim(0, 30)

                ax3 = sns.histplot(
                    ax=ax3,
                    data=x,
                    x=hist_col,
                    stat='proportion',
                    color='k',
                    bins=bins,
                    element="poly",
                    fill=False,
                    cumulative=True,
                    zorder=3
                )
                ax3.tick_params(axis='y', labelcolor=cdf_color, color=cdf_color)
                ax3.set_ylabel('CDF', color=cdf_color)
                ax3.set_ylim(0, 1)
                ax3.set_yticks(np.arange(0, 1.2, .2))
                ax3.axvline(np.median(x[hist_col]), c='C0', ls='--', lw=2, zorder=3)
                ax3.axvline(np.mean(x[hist_col]), c='C1', ls=':', lw=2, zorder=3)
                ax3.set_xlim(0, xmax)
                ax3.set_xticks(xticks)
                ax3.set_xlabel(color_label)
                ax3.text(
                    .9, .8, 'III',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=20,
                    color='k',
                    transform=ax3.transAxes
                )
            except IndexError:
                pass

            ax.add_patch(
                plt.Rectangle((.8, .3), .2, .2, ec="k", fc="none", transform=ax.transAxes)
            )

            ax.text(
                .9, .4, 'III',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=20,
                color='k',
                transform=ax.transAxes
            )

            if color_label == 'Residuals':
                ax1.legend(frameon=False, ncol=1, loc='upper left')
            else:
                ax1.legend(frameon=False, ncol=1, loc='center right')

            ax1.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)
            ax2.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)
            ax3.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

            ax1.xaxis.set_major_formatter(FormatStrFormatter('%g'))
            ax2.xaxis.set_major_formatter(FormatStrFormatter('%g'))
            ax3.xaxis.set_major_formatter(FormatStrFormatter('%g'))

            ax1t.yaxis.set_major_formatter(PercentFormatter(decimals=0))
            ax2t.yaxis.set_major_formatter(PercentFormatter(decimals=0))
            ax3t.yaxis.set_major_formatter(PercentFormatter(decimals=0))

            ax1.set_zorder(ax1t.get_zorder()+1)
            ax1.patch.set_visible(False)
            ax2.set_zorder(ax2t.get_zorder()+1)
            ax2.patch.set_visible(False)
            ax3.set_zorder(ax3t.get_zorder()+1)
            ax3.patch.set_visible(False)

        elif label == f'Number of iterations':

            x = histograms[
                (histograms.iter_num == 1) &
                (histograms.ibins >= 2.5) & (histograms.ibins <= 3.5)
            ]
            ax1 = sns.histplot(
                ax=ax1,
                data=x,
                x=hist_col,
                stat='percent',
                kde=True,
                bins=25,
                color='dimgrey'
            )


            ax1.axvline(np.median(x[hist_col]), c='C0', ls='-', lw=2)
            ax1.axvline(np.mean(x[hist_col]), c='C1', ls='--', lw=2)
            ax1.axvline(np.median(x[hist_col]), c='C0', ls='-', lw=2, label='Median')
            ax1.axvline(np.mean(x[hist_col]), c='C1', ls='--', lw=2, label='Mean')
            ax1.set_ylim(0, 80)
            ax1.set_xlim(0, 5)
            ax1.set_xlabel(color_label)
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
                (histograms.ibins >= 2.5) & (histograms.ibins <= 3.5)
            ]
            ax2 = sns.histplot(
                ax=ax2,
                data=x,
                x=hist_col,
                stat='percent',
                kde=True,
                bins=25,
                color='dimgrey'
            )
            ax2.axvline(np.median(x[hist_col]), c='C0', ls='-', lw=2)
            ax2.axvline(np.mean(x[hist_col]), c='C1', ls='--', lw=2)
            ax2.set_ylim(0, 80)
            ax2.set_xlim(0, 5)
            ax2.set_xlabel(color_label)
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
                (histograms.ibins >= 2.5) & (histograms.ibins <= 3.5)
            ]
            ax3 = sns.histplot(
                ax=ax3,
                data=x,
                x=hist_col,
                stat='percent',
                kde=True,
                bins=25,
                color='dimgrey'
            )
            ax3.axvline(np.median(x[hist_col]), c='C0', ls='-', lw=2, label='Median')
            ax3.axvline(np.mean(x[hist_col]), c='C1', ls='--', lw=2, label='Mean')
            ax3.set_ylim(0, 80)
            ax3.set_xlim(0, 5)
            ax3.set_xlabel(color_label)
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

    if label == 'Integrated photons' or label == 'Integrated photoelectrons':
        ax.set_xticks(np.arange(lims[0], lims[1]+5e4, 5e4), minor=False)
        ax.set_xticks(np.arange(lims[0], lims[1]+2.5e4, 2.5e4), minor=True)
    elif label == 'Number of iterations':
        ax.set_xticks(np.arange(0, dataframe.columns.values.max()+1, 1), minor=False)

    if sci:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)

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
def plot_heatmap_rms(
    dataframe,
    wavelength,
    savepath:Path,
    label='Integrated photoelectrons',
    color_label='Residuals',
    hist_col='confidence',
    lims=(0, 100),
    ax=None,
    cax=None,
    agg='mean',
    sci=False,
    histograms: Optional[pd.DataFrame] = None,
    kde_color='grey',
    cdf_color='k',
    hist_color='lightgrey',
):
    try:
        dataframe = dataframe.sort_index().interpolate()
    except ValueError:
        pass

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
        cax = fig.add_axes([1.01, 0.08, 0.03, 0.87])

    levels = np.array([
        0, .025, .05, .075, .1, .125, .15, .175, .2, .225,
        .25, .3, .35, .4, .45,
        .5, .55, .6, .65, .7,
        .75, .8, .9, 1,
    ])

    vmin, vmax, vcenter, step = levels[0], levels[-1], .25, levels[1] - levels[0]
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

    cbar = plt.colorbar(
        contours,
        cax=cax,
        fraction=0.046,
        pad=0.04,
        extend='both',
        spacing='proportional',
        format=FormatStrFormatter("%.2f"),
        # ticks=np.arange(vmin, vmax+.05, .05),
        ticks=[
            0, .05, .1, .15, .2, .25, .3, .35, .4, .45,
            .5, .6, .7, .8, .9, 1,
        ],
    )
    cbar.ax.set_ylabel(fr'Residuals ({agg} $\lambda$ RMS)')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('left')
    ax.patch.set(hatch='/', edgecolor='lightgrey', lw=.01)

    if histograms is not None:
        if label == 'Integrated photons' or label == 'Integrated photoelectrons':
            x = histograms[
                (histograms.pbins <= 1e5) &
                (histograms.ibins >= .4) & (histograms.ibins <= .6)
            ]

            if color_label == 'Residuals':
                xmax = .3
                binwidth = .02
                bins = np.arange(0, xmax + binwidth, binwidth)
                xticks = np.arange(0, xmax+.05, .05)
            else:
                if hist_col == 'confidence':
                    xmax = .15
                    binwidth = .01
                    bins = np.arange(0, xmax + binwidth, binwidth)
                    xticks = np.arange(0, xmax+.05, .05)
                else:
                    xmax = .3
                    binwidth = .025
                    bins = np.arange(0, xmax + binwidth, binwidth)
                    xticks = np.arange(0, xmax+.05, .05)

            try:
                ax1t = ax1.twinx()
                ax1t = sns.histplot(
                    ax=ax1t,
                    data=x,
                    x=hist_col,
                    stat='percent',
                    kde=True,
                    bins=bins,
                    color=hist_color,
                    element="step",
                )
                ax1t.lines[0].set_color(kde_color)
                ax1t.tick_params(axis='y', labelcolor=kde_color, color=kde_color)
                ax1t.set_ylabel('KDE', color=kde_color)
                ax1t.set_ylim(0, 50)

                ax1 = sns.histplot(
                    ax=ax1,
                    data=x,
                    x=hist_col,
                    stat='proportion',
                    color=cdf_color,
                    bins=bins,
                    element="poly",
                    fill=False,
                    cumulative=True,
                )

                ax1.tick_params(axis='y', labelcolor=cdf_color, color=cdf_color)
                ax1.set_ylabel('CDF', color=cdf_color)
                ax1.set_ylim(0, 1)
                ax1.set_yticks(np.arange(0, 1.2, .2))
                ax1.axvline(np.median(x[hist_col]), c='C0', ls='--', lw=2, label='Median', zorder=3)
                ax1.axvline(np.mean(x[hist_col]), c='C1', ls=':', lw=2, label='Mean', zorder=3)
                ax1.set_xlim(0, xmax)
                ax1.set_xticks(xticks)
                ax1.set_xlabel(color_label)
                ax1.text(
                    .9, .8, 'I',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=20,
                    color='k',
                    transform=ax1.transAxes
                )
            except IndexError:
                pass

            ax.add_patch(
                plt.Rectangle((0, .4), .2, .2, ec="k", fc="none", transform=ax.transAxes)
            )
            ax.text(
                .1, .5, 'I',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=20,
                color='k',
                transform=ax.transAxes
            )

            x = histograms[
                (histograms.pbins >= 2e5) & (histograms.pbins <= 3e5) &
                (histograms.ibins >= .4) & (histograms.ibins <= .6)
            ]

            try:
                ax2t = ax2.twinx()
                ax2t = sns.histplot(
                    ax=ax2t,
                    data=x,
                    x=hist_col,
                    stat='percent',
                    kde=True,
                    bins=bins,
                    color=hist_color,
                    element="step",
                )
                ax2t.lines[0].set_color(kde_color)
                ax2t.tick_params(axis='y', labelcolor=kde_color, color=kde_color)
                ax2t.set_ylabel('KDE', color=kde_color)
                ax2t.set_ylim(0, 50)

                ax2 = sns.histplot(
                    ax=ax2,
                    data=x,
                    x=hist_col,
                    stat='proportion',
                    color=cdf_color,
                    bins=bins,
                    element="poly",
                    fill=False,
                    cumulative=True,
                )
                ax2.tick_params(axis='y', labelcolor=cdf_color, color=cdf_color)
                ax2.set_ylabel('CDF', color=cdf_color)
                ax2.set_ylim(0, 1)
                ax2.set_yticks(np.arange(0, 1.2, .2))
                ax2.axvline(np.median(x[hist_col]), c='C0', ls='--', lw=2, zorder=3)
                ax2.axvline(np.mean(x[hist_col]), c='C1', ls=':', lw=2, zorder=3)
                ax2.set_xlim(0, xmax)
                ax2.set_xticks(xticks)
                ax2.set_xlabel(color_label)
                ax2.text(
                    .9, .8, 'II',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=20,
                    color='k',
                    transform=ax2.transAxes
                )
            except IndexError:
                pass

            ax.add_patch(
                plt.Rectangle((.4, .4), .2, .2, ec="k", fc="none", transform=ax.transAxes)
            )
            ax.text(
                .5, .5, 'II',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=20,
                color='k',
                transform=ax.transAxes
            )

            x = histograms[
                (histograms.pbins >= 4e5) & (histograms.pbins <= 5e5) &
                (histograms.ibins >= .4) & (histograms.ibins <= .6)
            ]

            try:
                ax3t = ax3.twinx()
                ax3t = sns.histplot(
                    ax=ax3t,
                    data=x,
                    x=hist_col,
                    stat='percent',
                    kde=True,
                    bins=bins,
                    color=hist_color,
                    element="step",
                )
                ax3t.lines[0].set_color(kde_color)
                ax3t.tick_params(axis='y', labelcolor=kde_color, color=kde_color)
                ax3t.set_ylabel('KDE', color=kde_color)
                ax3t.set_ylim(0, 50)

                ax3 = sns.histplot(
                    ax=ax3,
                    data=x,
                    x=hist_col,
                    stat='proportion',
                    color='k',
                    bins=bins,
                    element="poly",
                    fill=False,
                    cumulative=True,
                    zorder=3
                )
                ax3.tick_params(axis='y', labelcolor=cdf_color, color=cdf_color)
                ax3.set_ylabel('CDF', color=cdf_color)
                ax3.set_ylim(0, 1)
                ax3.set_yticks(np.arange(0, 1.2, .2))
                ax3.axvline(np.median(x[hist_col]), c='C0', ls='--', lw=2, zorder=3)
                ax3.axvline(np.mean(x[hist_col]), c='C1', ls=':', lw=2, zorder=3)
                ax3.set_xlim(0, xmax)
                ax3.set_xticks(xticks)
                ax3.set_xlabel(color_label)
                ax3.text(
                    .9, .8, 'III',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=20,
                    color='k',
                    transform=ax3.transAxes
                )
            except IndexError:
                pass

            ax.add_patch(
                plt.Rectangle((.8, .4), .2, .2, ec="k", fc="none", transform=ax.transAxes)
            )

            ax.text(
                .9, .5, 'III',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=20,
                color='k',
                transform=ax.transAxes
            )

            if color_label == 'Residuals':
                ax1.legend(frameon=False, ncol=1, loc='upper left')
            else:
                ax1.legend(frameon=False, ncol=1, loc='center right')

            ax1.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)
            ax2.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)
            ax3.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

            ax1.xaxis.set_major_formatter(FormatStrFormatter('%g'))
            ax2.xaxis.set_major_formatter(FormatStrFormatter('%g'))
            ax3.xaxis.set_major_formatter(FormatStrFormatter('%g'))

            ax1t.yaxis.set_major_formatter(PercentFormatter(decimals=0))
            ax2t.yaxis.set_major_formatter(PercentFormatter(decimals=0))
            ax3t.yaxis.set_major_formatter(PercentFormatter(decimals=0))

            ax1.set_zorder(ax1t.get_zorder()+1)
            ax1.patch.set_visible(False)
            ax2.set_zorder(ax2t.get_zorder()+1)
            ax2.patch.set_visible(False)
            ax3.set_zorder(ax3t.get_zorder()+1)
            ax3.patch.set_visible(False)

        elif label == f'Number of iterations':

            x = histograms[
                (histograms.iter_num == 1) &
                (histograms.ibins >= 2.5) & (histograms.ibins <= 3.5)
            ]
            ax1 = sns.histplot(
                ax=ax1,
                data=x,
                x=hist_col,
                stat='percent',
                kde=True,
                bins=25,
                color='dimgrey'
            )

            ax1.axvline(np.median(x[hist_col]), c='C0', ls='-', lw=2)
            ax1.axvline(np.mean(x[hist_col]), c='C1', ls='--', lw=2)
            ax1.axvline(np.median(x[hist_col]), c='C0', ls='-', lw=2, label='Median')
            ax1.axvline(np.mean(x[hist_col]), c='C1', ls='--', lw=2, label='Mean')
            ax1.set_ylim(0, 80)
            ax1.set_xlim(0, 5)
            ax1.set_xlabel(color_label)
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
                (histograms.ibins >= 2.5) & (histograms.ibins <= 3.5)
            ]
            ax2 = sns.histplot(
                ax=ax2,
                data=x,
                x=hist_col,
                stat='percent',
                kde=True,
                bins=25,
                color='dimgrey'
            )
            ax2.axvline(np.median(x[hist_col]), c='C0', ls='-', lw=2)
            ax2.axvline(np.mean(x[hist_col]), c='C1', ls='--', lw=2)
            ax2.set_ylim(0, 80)
            ax2.set_xlim(0, 5)
            ax2.set_xlabel(color_label)
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
                (histograms.ibins >= 2.5) & (histograms.ibins <= 3.5)
            ]
            ax3 = sns.histplot(
                ax=ax3,
                data=x,
                x=hist_col,
                stat='percent',
                kde=True,
                bins=25,
                color='dimgrey'
            )
            ax3.axvline(np.median(x[hist_col]), c='C0', ls='-', lw=2, label='Median')
            ax3.axvline(np.mean(x[hist_col]), c='C1', ls='--', lw=2, label='Mean')
            ax3.set_ylim(0, 80)
            ax3.set_xlim(0, 5)
            ax3.set_xlabel(color_label)
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

    if label == 'Integrated photons' or label == 'Integrated photoelectrons':
        ax.set_xticks(np.arange(lims[0], lims[1]+5e4, 5e4), minor=False)
        ax.set_xticks(np.arange(lims[0], lims[1]+2.5e4, 2.5e4), minor=True)
    elif label == 'Number of iterations':
        ax.set_xticks(np.arange(0, dataframe.columns.values.max()+1, 1), minor=False)

    if sci:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)

    ax.set_xlabel(label)
    ax.set_xlim(lims)

    ax.set_ylabel(rf'Initial aberration ({agg} $\lambda$ RMS, $\lambda = {int(wavelength * 1000)}~nm$)')
    ax.set_yticks(np.arange(0, 2, .05), minor=True)
    ax.set_yticks(np.arange(0, 2, .1))
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
    outdir: Path,
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
    num_beads: Optional[int] = None,
    simulate_psf_only: bool = False,
    lam_detection: Optional[float] = .510,
    skip_remove_background: bool = False,
    use_theoretical_widefield_simulator: bool = False,
    denoiser: Optional[Path] = None,
    denoiser_window_size: tuple = (32, 64, 64),
    simulate_samples: bool = False,
    estimated_object_gaussian_sigma: float = 0,
):
    modelspecs = backend.load_metadata(modelpath)

    if outdir == Path('../evaluations'):
        savepath = outdir / modelpath.with_suffix('').name / eval_sign / f'snrheatmaps'

        if psf_type is not None:
            savepath = Path(f"{savepath}/mode-{str(psf_type).replace('../lattice/', '').split('_')[0]}")
        
        if simulate_psf_only:
            savepath = savepath / 'psf'
        else:
            if num_beads is not None:
                savepath = savepath / f'beads-{num_beads}'
            else:
                savepath = savepath / 'beads'

        if distribution != '/':
            savepath = Path(f'{savepath}/{distribution}_na_{str(na).replace("0.", "p")}')
        else:
            savepath = Path(f'{savepath}/na_{str(na).replace("0.", "p")}')
    else:
        savepath = outdir

    savepath.mkdir(parents=True, exist_ok=True)

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
            npoints_range=(1, num_beads) if num_beads is not None else None,
            eval_sign=eval_sign,
            digital_rotations=digital_rotations,
            plot=plot,
            plot_rotations=plot_rotations,
            psf_type=psf_type,
            lam_detection=lam_detection,
            skip_remove_background=skip_remove_background,
            use_theoretical_widefield_simulator=use_theoretical_widefield_simulator,
            simulate_psf_only=simulate_psf_only,
            denoiser=denoiser,
            denoiser_window_size=denoiser_window_size,
            simulate_samples=simulate_samples,
            estimated_object_gaussian_sigma=estimated_object_gaussian_sigma
        )

    if 'aberration_umRMS' not in df.columns.values:
        df['aberration_umRMS'] = np.nan
        for idx in df.id.values:
            df.loc[df.id == idx, 'aberration_umRMS'] = df[df.id == idx].iloc[0]['residuals_umRMS']

    backup = df.copy()
    df = backup[backup['iter_num'] == iter_num]
    df['photoelectrons'] = utils.photons2electrons(df['photons'], quantum_efficiency=.82)

    for x in ['photons', 'photoelectrons', 'counts', 'counts_p100', 'counts_p99']:

        if x == 'photons':
            label = f'Integrated photons'
            lims = (0, 5*10**5)
            pbins = np.arange(lims[0], lims[-1]+1e4, 5e4)
        elif x == 'photoelectrons':
            label = f'Integrated photoelectrons'
            lims = (0, 5*10**5)
            pbins = np.arange(lims[0], lims[-1]+1e4, 5e4)
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
                df['aberration_umRMS'].apply(partial(utils.microns2waves, wavelength=modelspecs.lam_detection)),
                bins,
                labels=bins[1:],
                include_lowest=True
            )
            rms_dataframe = pd.pivot_table(df, values='residuals_umRMS', index='ibins', columns='pbins', aggfunc=agg)
            rms_dataframe = rms_dataframe.applymap(partial(utils.microns2waves, wavelength=modelspecs.lam_detection))
            rms_dataframe.insert(0, 0, rms_dataframe.index.values.astype(df['residuals'].dtype))

            plot_heatmap_rms(
                rms_dataframe,
                histograms=df if x == 'photons' else None,
                wavelength=modelspecs.lam_detection,
                savepath=Path(f"{savepath}_iter_{iter_num}_{x}_{agg}"),
                label=label,
                lims=lims,
                agg=agg,
                sci=True,
            )

            coverage = pd.pivot_table(df, values='residuals_umRMS', index='ibins', columns='pbins', aggfunc='count')
            plot_coverage(
                coverage,
                wavelength=modelspecs.lam_detection,
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

            plot_heatmap_p2v(
                dataframe,
                histograms=df if x == 'photons' else None,
                wavelength=modelspecs.lam_detection,
                savepath=Path(f"{savepath}_iter_{iter_num}_{x}_{agg}"),
                label=label,
                hist_col='residuals',
                sci=True,
                lims=lims,
                agg=agg
            )

            coverage = pd.pivot_table(df, values='residuals_umRMS', index='ibins', columns='pbins', aggfunc='count')
            plot_coverage(
                coverage,
                wavelength=modelspecs.lam_detection,
                savepath=Path(f"{savepath}_iter_{iter_num}_{x}_coverage"),
                label=label,
                lims=lims,
                sci=True,
                p2v=True
            )

            try:
                for c in ['confidence', 'confidence_sum']:
                    dataframe = pd.pivot_table(df, values=c, index='ibins', columns='pbins', aggfunc=agg)
                    dataframe.insert(0, 0, dataframe.index.values.astype(df[c].dtype))

                    # replace unconfident predictions with max std
                    dataframe.replace(0, dataframe.max(), inplace=True)
                    dataframe.to_csv(f'{savepath}_{x}_{c}_{agg}.csv')
                    logger.info(f'Saved: {savepath.resolve()}_{x}_{c}_{agg}.csv')

                    plot_heatmap_p2v(
                        dataframe,
                        histograms=df if x == 'photons' else None,
                        wavelength=modelspecs.lam_detection,
                        savepath=Path(f"{savepath}_iter_{iter_num}_{x}_{c}_{agg}"),
                        label=label,
                        color_label='Standard deviation',
                        sci=True,
                        hist_col=c,
                        lims=lims,
                        agg=agg
                    )
            except Exception:
                pass

    return savepath


@profile
def densityheatmap(
    modelpath: Path,
    datadir: Path,
    outdir: Path,
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
    num_beads: Optional[int] = None,
    photons_range: Optional[tuple] = None,
    psf_type: Optional[str] = None,
    lam_detection: Optional[float] = .510,
    skip_remove_background: bool = False,
    use_theoretical_widefield_simulator: bool = False,
    denoiser: Optional[Path] = None,
    denoiser_window_size: tuple = (32, 64, 64),
    simulate_samples: bool = False,
    estimated_object_gaussian_sigma: float = 0,
):
    modelspecs = backend.load_metadata(modelpath)

    if outdir == Path('../evaluations'):
        savepath = outdir / modelpath.with_suffix('').name / eval_sign / f'densityheatmaps'

        if psf_type is not None:
            savepath = Path(f"{savepath}/mode-{str(psf_type).replace('../lattice/', '').split('_')[0]}")

        if num_beads is not None:
            savepath = savepath / f'beads-{num_beads}'

        if distribution != '/':
            savepath = Path(f'{savepath}/{distribution}_na_{str(na).replace("0.", "p")}')
        else:
            savepath = Path(f'{savepath}/na_{str(na).replace("0.", "p")}')
    else:
        savepath = outdir

    savepath.mkdir(parents=True, exist_ok=True)

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
            npoints_range=(1, num_beads) if num_beads is not None else None,
            batch_size=batch_size,
            eval_sign=eval_sign,
            digital_rotations=digital_rotations,
            plot=plot,
            plot_rotations=plot_rotations,
            psf_type=psf_type,
            lam_detection=lam_detection,
            skip_remove_background=skip_remove_background,
            use_theoretical_widefield_simulator=use_theoretical_widefield_simulator,
            denoiser=denoiser,
            denoiser_window_size=denoiser_window_size,
            simulate_samples=simulate_samples,
            estimated_object_gaussian_sigma=estimated_object_gaussian_sigma
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
        dataframe.insert(0, 0, dataframe.index.values.astype(df['residuals'].dtype))
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
def objectsizeheatmap(
    modelpath: Path,
    datadir: Path,
    outdir: Path,
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
    num_beads: Optional[int] = None,
    photons_range: Optional[tuple] = None,
    psf_type: Optional[str] = None,
    lam_detection: Optional[float] = .510,
    skip_remove_background: bool = False,
    use_theoretical_widefield_simulator: bool = False,
    denoiser: Optional[Path] = None,
    denoiser_window_size: tuple = (32, 64, 64),
    simulate_samples: bool = False,
    estimated_object_gaussian_sigma: float = 0,
    object_gaussian_sigma_range: tuple = (0, 2)
):
    modelspecs = backend.load_metadata(modelpath)
    
    if outdir == Path('../evaluations'):
        savepath = outdir / modelpath.with_suffix('').name / eval_sign / f'objectsizeheatmaps'
        
        if psf_type is not None:
            savepath = Path(f"{savepath}/mode-{str(psf_type).replace('../lattice/', '').split('_')[0]}")
        
        if num_beads is not None:
            savepath = savepath / f'beads-{num_beads}'
        
        if distribution != '/':
            savepath = Path(f'{savepath}/{distribution}_na_{str(na).replace("0.", "p")}')
        else:
            savepath = Path(f'{savepath}/na_{str(na).replace("0.", "p")}')
    else:
        savepath = outdir
    
    savepath.mkdir(parents=True, exist_ok=True)
    
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
            npoints_range=(1, num_beads) if num_beads is not None else None,
            batch_size=batch_size,
            eval_sign=eval_sign,
            digital_rotations=digital_rotations,
            plot=plot,
            plot_rotations=plot_rotations,
            psf_type=psf_type,
            lam_detection=lam_detection,
            skip_remove_background=skip_remove_background,
            use_theoretical_widefield_simulator=use_theoretical_widefield_simulator,
            denoiser=denoiser,
            denoiser_window_size=denoiser_window_size,
            simulate_samples=simulate_samples,
            estimated_object_gaussian_sigma=estimated_object_gaussian_sigma,
            randomize_object_gaussian_sigma=np.arange(
                object_gaussian_sigma_range[0], object_gaussian_sigma_range[1]+.1, step=.1
            )
        )
    
    df = df[df['iter_num'] == iter_num]
    
    bins = np.arange(0, 10.25, .25).round(2)
    df['ibins'] = pd.cut(
        df['aberration'],
        bins,
        labels=bins[1:],
        include_lowest=True
    )
    
    dataframe = pd.pivot_table(df, values='residuals', index='ibins', columns='object_gaussian_sigma', aggfunc=agg)
    dataframe.insert(0, 0, dataframe.index.values.astype(df['residuals'].dtype))
    dataframe.to_csv(f'{savepath}.csv')
    logger.info(f'Saved: {savepath.resolve()}.csv')
    
    plot_heatmap_p2v(
        dataframe,
        wavelength=modelspecs.lam_detection,
        savepath=Path(f'{savepath}_iter_{iter_num}_gaussian_sigma'),
        label=r'Gaussian $\sigma$',
        lims=object_gaussian_sigma_range,
        agg=agg
    )
    
    return savepath


@profile
def iterheatmap(
    modelpath: Path,
    datadir: Path,  # folder or _predictions.csv file
    outdir: Path,
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
    skip_remove_background: bool = False,
    use_theoretical_widefield_simulator: bool = False,
    denoiser: Optional[Path] = None,
    denoiser_window_size: tuple = (32, 64, 64),
    simulate_samples: bool = False,
    estimated_object_gaussian_sigma: float = 0,
):
    modelspecs = backend.load_metadata(modelpath)

    if outdir == Path('../evaluations'):
        savepath = outdir / modelpath.with_suffix('').name / eval_sign / f'iterheatmaps'

        if psf_type is not None:
            savepath = Path(f"{savepath}/mode-{str(psf_type).replace('../lattice/', '').split('_')[0]}")

        if distribution != '/':
            savepath = Path(f'{savepath}/{distribution}_na_{str(na).replace("0.", "p")}')
        else:
            savepath = Path(f'{savepath}/na_{str(na).replace("0.", "p")}')
    else:
        savepath = outdir

    savepath.mkdir(parents=True, exist_ok=True)

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
            lam_detection=lam_detection,
            skip_remove_background=skip_remove_background,
            use_theoretical_widefield_simulator=use_theoretical_widefield_simulator,
            denoiser=denoiser,
            denoiser_window_size=denoiser_window_size,
            simulate_samples=simulate_samples,
            estimated_object_gaussian_sigma=estimated_object_gaussian_sigma
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
            plot_heatmap_rms(
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
    estimated_object_gaussian_sigma: float = 0,
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
                        fill_radius=.66 if num_objs > 1 else 0
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
                        min_psnr=0,
                        estimated_object_gaussian_sigma=estimated_object_gaussian_sigma
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

                    imwrite(save_path / f'psf_{s}.tif', noisy_img, compression='deflate', dtype=np.float32)
                    imwrite(save_path / f'corrected_psf_{s}.tif', corrected_psf, compression='deflate', dtype=np.float32)

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


def plot_templates(model: Path):
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.autolimit_mode': 'round_numbers'
    })

    outdir = model.with_suffix('') / 'templates'
    outdir.mkdir(parents=True, exist_ok=True)
    modelspecs = backend.load_metadata(model)

    photon_step = 10e3
    photons = np.arange(photon_step, 1e5+photon_step, photon_step).astype(int)
    waves = np.arange(0, .55, step=.05).round(2)

    aberrations = np.zeros((len(waves), modelspecs.n_modes))
    gen = backend.load_metadata(model, psf_shape=(64, 64, 64))

    # plot templates
    for i in trange(3, modelspecs.n_modes):
        if i == 4:
            continue

        savepath = outdir / f"m{i}"

        fig, axes = plt.subplots(nrows=len(waves), ncols=len(photons), figsize=(12, 14))

        for t, a in tqdm(
            enumerate(waves[::-1]),
            desc=f'Simulating aberrations for [{len(waves)}] AMP bins & [{len(photons)}] PH bins',
            total=len(waves)
        ):
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
                    fill_radius=0,
                )
                imwrite(f'{savepath}_ph{ph}_a{str(a).replace("0.", "p")}.tif', img.astype(np.float32))
                img -= 100
                img[img < 0] = 0
                # img = imread(f'{savepath}_ph{ph}_a{str(a).replace("0.", "p")}.tif').astype(np.float32)

                axes[t, j].imshow(np.max(img, axis=0) ** .5, cmap='hot')
                axes[t, j].set_xticks([])
                axes[t, j].set_yticks([])

                axes[t, j].set_title(
                    f"{int(np.max(img) / 1e3)}$\\times 10^3$" if np.max(img) > 1e4 else int(np.max(img)),
                    # f"{int(np.sum(img)/1e6)}$\\times 10^6$" if np.sum(img) > 1e6 else int(np.sum(img)),
                    pad=1
                )

                if j == 0:
                    p2v = np.round(w.peak2valley(), 1)
                    axes[t, j].set_ylabel(f'{p2v:.1f}$\lambda$ (p2v)\n{a:.2f}$\mu$m (rms)')

                # if j == len(photons) - 1:
                #     tax = axes[t, j].twinx()
                #     tax.set_yticks([])
                #     tax.set_xticks([])
                #     tax.set_ylabel(f'{a:.2f}$\mu$m\n')

                if t == len(waves) - 1:
                    axes[t, j].set_xlabel(f"{int(ph / 1e3)}$\\times 10^3$\nphotons")

        plt.subplots_adjust(top=.9, bottom=.1, left=.1, right=.9, hspace=.15, wspace=.15)
        plt.savefig(f'{savepath}_templateheatmap.pdf', bbox_inches='tight', pad_inches=.25)
        plt.savefig(f'{savepath}_templateheatmap.png', dpi=300, bbox_inches='tight', pad_inches=.25)
        plt.savefig(f'{savepath}_templateheatmap.svg', dpi=300, bbox_inches='tight', pad_inches=.25)
        logger.info(f'Saved: {savepath}_templateheatmap.png  .pdf  .svg')


def create_samples(
    wavefronts: list,
    photons: list,
    gen: SyntheticPSF,
    savepath: Any = None,
    num_objs: int = 1,
    object_size: int = 0,
):
    kernels = [gen.single_psf(phi=w, normed=True) for w in wavefronts]
    # amps = [np.max(w.amplitudes) for w in wavefronts]

    if Path(f"{savepath}_inputs.npy").exists():
        inputs = np.load(f"{savepath}_inputs.npy")
    else:
        inputs = np.stack([
                simulate_beads(
                    psf=kernels[k],
                    psf_type=gen.psf_type,
                    object_size=object_size,
                    num_objs=num_objs,
                    photons=ph,
                    # maxcounts=ph,
                    noise=True,
                    fill_radius=0 if num_objs == 1 else .66
                )
            # imread(f'{savepath}_ph{ph}_a{str(amps[k]).replace("0.", "p")}.tif').astype(np.float32)
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
    digital_rotations: Optional[int] = 361,
    object_size: int = 0,
    denoiser: Optional[CARE] = None,
    denoiser_window_size: tuple = (32, 64, 64),
    estimated_object_gaussian_sigma: float = 0,
):
    model = backend.load(modelpath)
    gen = backend.load_metadata(
        modelpath,
        psf_type=psf_type,
        psf_shape=3*[model.input_shape[2]],
        rotate=False
    )
    samplepsfgen = backend.load_metadata(
        modelpath,
        psf_type=psf_type,
        psf_shape=3*[model.input_shape[2]],
        rotate=False,
        x_voxel_size=.097,
        y_voxel_size=.097
    )

    if not isinstance(wavefronts[0], Wavefront):
        wavefronts = [Wavefront(w, lam_detection=gen.lam_detection, rotate=False) for w in wavefronts]

    p2v = [w.peak2valley(na=na) for w in wavefronts]

    inputs = create_samples(
        wavefronts=wavefronts,
        photons=photons,
        gen=samplepsfgen,
        savepath=savepath,
        num_objs=num_objs,
        object_size=object_size,
    )

    if Path(f"{savepath}_embeddings.npy").exists():
        embeddings = np.load(f"{savepath}_embeddings.npy")
    else:
        embeddings = np.stack([
            backend.preprocess(
                inputs[i],
                modelpsfgen=gen,
                samplepsfgen=samplepsfgen,
                digital_rotations=digital_rotations,
                remove_background=True,
                normalize=True,
                min_psnr=0,
                plot=f"{savepath}_{a}_{ph}",
                denoiser=denoiser,
                denoiser_window_size=denoiser_window_size,
                estimated_object_gaussian_sigma=estimated_object_gaussian_sigma
            )
            for i, (a, ph) in enumerate(itertools.product(p2v, photons))
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
            plot_rotations=True
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
    outdir: Path,
    eval_sign: str = 'signed',
    batch_size: int = 512,
    num_objs: Optional[int] = 1,
    digital_rotations: bool = True,
    agg: str = 'median',
    denoiser: Optional[Path] = None,
    denoiser_window_size: tuple = (32, 64, 64),
    estimated_object_gaussian_sigma: float = 0,
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
    
    outdir = outdir / model.with_suffix('').name / eval_sign / 'evalmodes' / f'test'
    outdir.mkdir(parents=True, exist_ok=True)
    modelspecs = backend.load_metadata(model)
    
    if denoiser is not None:
        logger.info(f"Loading denoiser model: {denoiser}")
        denoiser = CARE(config=None, name=denoiser.name, basedir=denoiser.parent)
    else:
        denoiser = None

    photon_step = 10e3
    photons = np.arange(1, 1e5 + photon_step, photon_step).astype(int)
    waves = np.arange(0, .35, step=.05).round(2)
    aberrations = np.zeros((len(waves), modelspecs.n_modes))
    gen = backend.load_metadata(model, psf_shape=(64, 64, 64))

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
                wavefronts=classes,
                modelpath=model,
                num_objs=num_objs,
                photons=photons,
                batch_size=batch_size,
                eval_sign=eval_sign,
                savepath=savepath,
                digital_rotations=361 if digital_rotations else None,
                denoiser=denoiser,
                denoiser_window_size=denoiser_window_size,
                estimated_object_gaussian_sigma=estimated_object_gaussian_sigma
            )

        df['photoelectrons'] = utils.photons2electrons(df['photons'], quantum_efficiency=.82)

        x, y = 'photons', 'aberration'
        xstep, ystep = 1e4, .25

        ybins = np.arange(0, df[y].max()+ystep, ystep)
        df['ybins'] = pd.cut(df[y], ybins, labels=ybins[1:], include_lowest=True)

        if x == 'photons':
            xbins = df['photons'].values
            df['xbins'] = xbins
        else:
            xbins = np.arange(0, df[x].max()+xstep, xstep)
            df['xbins'] = pd.cut(df[x], xbins, labels=xbins[1:], include_lowest=True)

        dataframe = pd.pivot_table(df, values='residuals', index='ybins', columns='xbins', aggfunc=agg)
        dataframe.insert(0, 0, dataframe.index.values.astype(df['residuals'].dtype))
        dataframe.drop(columns=[1], inplace=True)

        try:
            dataframe = dataframe.sort_index().interpolate()
        except ValueError:
            pass

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
        axt.set_xlim(0, xbins[-1])
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
    estimated_object_gaussian_sigma: float = 0,
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
        fill_radius=.66 if num_objs > 1 else 0
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
                        min_psnr=0,
                        estimated_object_gaussian_sigma=estimated_object_gaussian_sigma
                    )
                    imwrite(save_path / f'{z}_embeddings.tif', embeddings.astype(np.float32), imagej=True, compression='deflate', dtype=np.float32)

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
                        imwrite(save_path / f'{z}_na_mask.tif', gen.na_mask.astype(np.float32), compression='deflate', dtype=np.float32)

                    imwrite(save_path / f'{z}_input.tif', noisy_img.astype(np.float32), compression='deflate', dtype=np.float32)

                    imwrite(save_path / f'{z}_pred_psf.tif', p_psf.astype(np.float32), compression='deflate', dtype=np.float32)
                    imwrite(save_path / f'{z}_pred_wavefront.tif', p_wave.wave().astype(np.float32), compression='deflate', dtype=np.float32)

                    imwrite(save_path / f'{z}_gt_psf.tif', gt_psf.astype(np.float32), compression='deflate', dtype=np.float32)
                    imwrite(save_path / f'{z}_gt_wavefront.tif', y_wave.wave().astype(np.float32), compression='deflate', dtype=np.float32)

                    imwrite(save_path / f'{z}_corrected_psf.tif', corrected_psf.astype(np.float32), compression='deflate', dtype=np.float32)
                    imwrite(save_path / f'{z}_corrected_wavefront.tif', residuals.wave().astype(np.float32), compression='deflate', dtype=np.float32)

                    processed_input = backend.prep_sample(
                        noisy_img,
                        model_fov=gen.psf_fov,  # this is what we will crop to
                        sample_voxel_size=gen.voxel_size,
                        remove_background=True,
                        normalize=True,
                        min_psnr=0,
                        na_mask=gen.na_mask
                    )

                    processed_corrected = backend.prep_sample(
                        corrected_noisy_img,
                        model_fov=gen.psf_fov,  # this is what we will crop to
                        sample_voxel_size=gen.voxel_size,
                        remove_background=True,
                        normalize=True,
                        min_psnr=0,
                        na_mask=gen.na_mask
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


@profile
def eval_confidence(
    model: Path,
    batch_size: int = 512,
    eval_sign: str = 'signed',
    dist: str = 'single',
    digital_rotations: bool = False,
    estimated_object_gaussian_sigma: float = 0,
):
    pool = Pool(processes=4)  # plotting = 2*calculation time, so shouldn't need more than 2-4 processes to keep up.

    models = list(model.glob('*.h5'))

    for photons in [5e4, 1e5, 2e5]:
        photons = int(photons)
        for amplitude_range in [(.1, .11), (.2, .22)]:
            gen = backend.load_metadata(
                models[0],
                amplitude_ranges=amplitude_range,
                distribution=dist,
                signed=False if eval_sign == 'positive_only' else True,
                rotate=True,
                mode_weights='pyramid',
                psf_shape=(64, 64, 64),
            )
            for s in range(5):
                for num_objs in [1, 3, 5]:
                    reference = multipoint_dataset.beads(
                        photons=photons,
                        image_shape=gen.psf_shape,
                        object_size=0,
                        num_objs=num_objs,
                        fill_radius=.66 if num_objs > 1 else 0
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

                    for trained_model in tqdm(models, file=sys.stdout):
                        logger.info(trained_model)
                        m = backend.load(trained_model)
                        no_phase = True if m.input_shape[1] == 3 else False

                        save_path = Path(
                            f"{model.with_suffix('')}/{eval_sign}/confidence/ph-{photons}/um-{amplitude_range[-1]}/num_objs-{num_objs:02d}/{trained_model.name.strip('.h5')}"
                        )
                        save_path.mkdir(exist_ok=True, parents=True)

                        embeddings = backend.preprocess(
                            noisy_img,
                            modelpsfgen=gen,
                            digital_rotations=361 if digital_rotations else None,
                            remove_background=True,
                            normalize=True,
                            plot=save_path / f'{s}',
                            min_psnr=0,
                            estimated_object_gaussian_sigma=estimated_object_gaussian_sigma
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

                        imwrite(save_path / f'psf_{s}.tif', noisy_img, compression='deflate', dtype=np.float32)
                        imwrite(save_path / f'corrected_psf_{s}.tif', corrected_psf, compression='deflate', dtype=np.float32)

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


@profile
def confidence_heatmap(
    modelpath: Path,
    datadir: Path,
    outdir: Path,
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
    skip_remove_background: bool = False,
    use_theoretical_widefield_simulator: bool = False,
    denoiser: Optional[Path] = None,
    denoiser_window_size: tuple = (32, 64, 64),
    simulate_samples: bool = False,
    estimated_object_gaussian_sigma: float = 0,
):
    modelspecs = backend.load_metadata(modelpath)

    if outdir == Path('../evaluations'):
        savepath = outdir / modelpath.with_suffix('').name / eval_sign / f'confidence'

        if psf_type is not None:
            savepath = Path(f"{savepath}/mode-{str(psf_type).replace('../lattice/', '').split('_')[0]}")

        if distribution != '/':
            savepath = Path(f'{savepath}/{distribution}_na_{str(na).replace("0.", "p")}')
        else:
            savepath = Path(f'{savepath}/na_{str(na).replace("0.", "p")}')
    else:
        savepath = outdir

    savepath.mkdir(parents=True, exist_ok=True)

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
            lam_detection=lam_detection,
            skip_remove_background=skip_remove_background,
            use_theoretical_widefield_simulator=use_theoretical_widefield_simulator,
            denoiser=denoiser,
            denoiser_window_size=denoiser_window_size,
            simulate_samples=simulate_samples,
            estimated_object_gaussian_sigma=estimated_object_gaussian_sigma
        )

    df = df[df['iter_num'] == iter_num]
    df['photoelectrons'] = utils.photons2electrons(df['photons'], quantum_efficiency=.82)

    for x in ['photons', 'photoelectrons', 'counts', 'counts_p100', 'counts_p99']:

        if x == 'photons':
            label = f'Integrated photons'
            lims = (0, 10**6)
            pbins = np.arange(lims[0], lims[-1]+10e4, 5e4)
        elif x == 'photoelectrons':
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

        for c in ['confidence', 'confidence_sum']:
            dataframe = pd.pivot_table(df, values=c, index='ibins', columns='pbins', aggfunc=agg)
            dataframe.insert(0, 0, dataframe.index.values.astype(df[c].dtype))
            dataframe.to_csv(f'{savepath}_{x}_{c}.csv')
            logger.info(f'Saved: {savepath.resolve()}_{x}_{c}.csv')

            plot_heatmap_p2v(
                dataframe,
                histograms=df if x == 'photons' else None,
                wavelength=modelspecs.lam_detection,
                savepath=Path(f"{savepath}_iter_{iter_num}_{x}_{c}"),
                label=label,
                color_label='Confidence',
                hist_col=c,
                lims=lims,
                agg='mean'
            )

    return savepath



def residuals_histogram(
    csv_path: Path = Path(r"C:\Users\milkied10\Desktop\na_1.0_predictions.csv"),
    amp_range: tuple = (0.18, 0.22),
    photon_min: float = 50000,
    total_ab_max: float = 3.5,
):
    """
    Make some histogram plots to show if the model overshoots or undershoots and under which conditions.

    Args:
        csv_path:
        amp_range:
        photon_min:
        total_ab_max:

    Returns:

    """
    predictions = pd.read_csv(csv_path)
    save_path = Path(f'{csv_path.with_suffix("")}_histogram.png')
    fig, axes = plt.subplots(8, 2, figsize=(16, 28))
    axes[0, 0].set_title(f'Samples with gt amp of {amp_range}, >{photon_min//1000}k photons')

    predictions = predictions[predictions['photons'] > photon_min]   # keep rows that have photons above threshold
    predictions = predictions[
        (predictions['aberration'] < total_ab_max) &
        (predictions['aberration'] < total_ab_max)
    ]  # keep rows that have photons above threshold

    predictions['astig_gt'] = np.sqrt(predictions["z3_ground_truth"] * predictions["z3_ground_truth"] +
                                      predictions["z5_ground_truth"] * predictions["z5_ground_truth"])

    predictions['astig_pred'] = np.sqrt(predictions["z3_prediction"] * predictions["z3_prediction"] +
                                        predictions["z5_prediction"] * predictions["z5_prediction"])

    predictions['astig_res'] = predictions['astig_gt'] - predictions['astig_pred']

    pax = axes[0,0]
    gt_col = 'astig_gt'
    col = 'astig_res'
    data = predictions[(predictions[gt_col] > amp_range[0]) &
                       (predictions[gt_col] < amp_range[1])
                       ]

    binrange = (-0.205, 0.2)
    sns.histplot(
        (data[col], data[gt_col]),
        ax=pax,
        binwidth=0.01,
        kde=False,
        log_scale=(False, False),
        binrange=binrange,
        multiple="layer",
        stat="percent",
    )
    pax.set_xlabel('astig_res')
    pax.set_ylabel(f'Samples ({len(data[col])} total samples)')
    logger.info(f"{col}, \t{len(data[col])} total samples")

    for z in range(3, 15):
        if z == 4:
            pass
        else:
            pax = axes[z // 2, z % 2]
            gt_col = f"z{z}_ground_truth"
            col = f"z{z}_residual"
            data = predictions[(predictions[gt_col] > amp_range[0]) &
                               (predictions[gt_col] < amp_range[1])
            ]
            sns.histplot(
                (data[col], data[gt_col]),
                ax=pax,
                binwidth=0.01,
                kde=False,
                log_scale=(False, False),
                binrange=binrange,
                multiple="layer",
                stat="percent",
            )
            pax.set_xlabel(f"z{z}_residual")
            pax.set_ylabel(f'Percentage ({len(data[col])} total samples)')
            logger.info(f"{col}, \t{len(data[col])} total samples")

        with tempfile.NamedTemporaryFile(suffix=f'_{col}.png', delete=False) as fp:
            # use a temporary file for savefig, so a real-time viewing of the image file doesn't cause errors
            fp.close()
            plt.savefig(fp.name, dpi=300, bbox_inches='tight', pad_inches=.25)
            shutil.move(fp.name, save_path)

    logger.info(f"Saved: {save_path}")


@profile
def evaluate_object_sizes(
    model: Path,
    outdir: Path,
    eval_sign: str = 'signed',
    batch_size: int = 256,
    num_objs: Optional[int] = 1,
    digital_rotations: bool = True,
    agg: str = 'median',
    na: float = 1.0,
    photons: int = 40000,
    override: bool = False,
    denoiser: Optional[Path] = None,
    denoiser_window_size: tuple = (32, 64, 64),
    amp: float = .15,
    estimated_object_gaussian_sigma: float = 0,
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

    digital_rotations = 361 if digital_rotations else None

    m = backend.load(model)
    modelgen = backend.load_metadata(
        model,
        psf_shape=3*[m.input_shape[2]],
        rotate=False,
    )
    samplegen = backend.load_metadata(
        model,
        psf_shape=(96, 96, 96),
        rotate=False,
        # psf_type='widefield',
        x_voxel_size=.097,
        y_voxel_size=.097,
        z_voxel_size=.2,
        use_theoretical_widefield_simulator=True,
    )
    w = Wavefront(np.zeros(15))
    
    outdir = outdir / model.with_suffix('').name / eval_sign / 'evalobjects' / f'num_objs_{num_objs}'

    for i, (mode, twin) in enumerate(w.twins.items()):
        if mode.index_ansi == 4: continue

        zernikes = np.zeros(15)
        zernikes[mode.index_ansi] = amp

        if np.all(zernikes == 0):
            savepath = outdir / f"z0"
        else:
            savepath = outdir / f"z{mode.index_ansi}"

        savepath.mkdir(parents=True, exist_ok=True)
        savepath = savepath / f"ph{photons}"
        
        sizes = np.arange(0, 6.5, .25).round(2)
        wavefront = Wavefront(zernikes, lam_detection=samplegen.lam_detection, rotate=False)
        psf = samplegen.single_psf(phi=wavefront, normed=True)
        psf /= np.sum(psf)

        if not override and Path(f"{savepath}_inputs.npy").exists():
            inputs = np.load(f"{savepath}_inputs.npy")
        else:
            inputs = np.zeros((len(sizes), *psf.shape))

            for i, w in enumerate(sizes):
                if w > 0:
                    inputs[i] = utils.add_noise(utils.fftconvolution(
                        sample=psf,
                        kernel=utils.gaussian_kernel(kernlen=(21, 21, 21), std=utils.fwhm2sigma(w)) * photons
                    ))
                else:
                    inputs[i] = utils.add_noise(psf * photons)

                imwrite(f"{savepath}_{w}.tif", inputs[i].astype(np.float32), compression='deflate', dtype=np.float32)

            inputs = np.stack(inputs, axis=0)[..., np.newaxis]
            np.save(f"{savepath}_inputs", inputs)

        if not override and Path(f"{savepath}_embeddings.npy").exists():
            embeddings = np.load(f"{savepath}_embeddings.npy")
        else:
            embeddings = np.stack([
                backend.preprocess(
                    i,
                    modelpsfgen=modelgen,
                    samplepsfgen=samplegen,
                    digital_rotations=digital_rotations,
                    remove_background=True,
                    normalize=True,
                    min_psnr=0,
                    plot=Path(f"{savepath}_{sizes[w]}"),
                    denoiser=denoiser,
                    denoiser_window_size=denoiser_window_size,
                    estimated_object_gaussian_sigma=estimated_object_gaussian_sigma
                )
                for w, i in enumerate(tqdm(
                    inputs, desc='Generating fourier embeddings', total=inputs.shape[0], file=sys.stdout
                ))
            ], axis=0)
            np.save(f"{savepath}_embeddings", embeddings)

        ys = np.stack([wavefront.amplitudes for w in sizes])

        embeddings = tf.data.Dataset.from_tensor_slices(embeddings)

        if not override and Path(f"{savepath}_predictions.npy").exists():
            preds = np.load(f"{savepath}_predictions.npy")
        else:
            res = backend.predict_dataset(
                model,
                inputs=embeddings,
                psfgen=modelgen,
                batch_size=batch_size,
                save_path=[f"{savepath}_{w}" for w in sizes],
                digital_rotations=digital_rotations,
                plot_rotations=True
            )

            try:
                preds, stdev = res
            except ValueError:
                preds, stdev, lls_defocus = res

            np.save(f"{savepath}_predictions", preds)

        residuals = ys - preds
        sizes = [s * samplegen.x_voxel_size * 1000 for s in sizes]
        df = pd.DataFrame([w for w in sizes], columns=['size'])
        df['prediction'] = [Wavefront(i, lam_detection=modelgen.lam_detection).peak2valley(na=na) for i in preds]
        df['residuals'] = [Wavefront(i, lam_detection=modelgen.lam_detection).peak2valley(na=na) for i in residuals]
        df['moi'] = ys[:, mode.index_ansi] - preds[:, mode.index_ansi]
        df['counts'] = [np.sum(i) for i in inputs]
        df.to_csv(f'{savepath}.csv')
        print(df)

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.regplot(
            data=df,
            x="size",
            y="moi",
            scatter=True,
            truncate=False,
            order=2,
            color=".2",
            ax=ax
        )
        
        ax.set_yticks(np.arange(-1 * amp, amp + .025, .025))
        ax.set_ylim(-1 * amp, amp)
        ax.set_xlim(0, sizes[-1])
        ax.axhline(y=0, color='r')
        ax.set_ylabel(r'Residuals ($y - \hat{y}$) $\mu$m RMS')
        # ax.set_ylabel(rf'Residuals ($\lambda = {int(gen.lam_detection * 1000)}~nm$)')
        ax.set_xlabel(r'Gaussian kernel full width at half maximum (FWHM) $w$ [nm]')
        ax.grid(True, which="both", axis='y', lw=1, ls='--', zorder=0, alpha=.5)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)

        secax = ax.secondary_xaxis('top', functions=(utils.fwhm2sigma, utils.sigma2fwhm))
        secax.set_xlabel(r'Gaussian kernel ($\sigma = w / 2 \sqrt{2 \ln{2}}$) $\sigma$ [nm]')

        plt.savefig(f'{savepath}.pdf', bbox_inches='tight', pad_inches=.25)
        plt.savefig(f'{savepath}.png', dpi=300, bbox_inches='tight', pad_inches=.25)
        plt.savefig(f'{savepath}.svg', dpi=300, bbox_inches='tight', pad_inches=.25)

    return savepath


@profile
def evaluate_uniform_background(
    model: Path,
    eval_sign: str = 'signed',
    batch_size: int = 256,
    num_objs: Optional[int] = 10,
    digital_rotations: bool = True,
    agg: str = 'median',
    na: float = 1.0,
    photons: int = 20000,
    override: bool = False,
    amplutide: float = .0,
    estimated_object_gaussian_sigma: float = 0,
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

    digital_rotations = 361 if digital_rotations else None

    m = backend.load(model)
    modelgen = backend.load_metadata(
        model,
        psf_shape=3*[m.input_shape[2]],
        rotate=False,
    )
    samplegen = backend.load_metadata(
        model,
        psf_shape=(64, 64, 64),
        rotate=False,
        x_voxel_size=.097,
        y_voxel_size=.097,
        z_voxel_size=.2,
    )
    w = Wavefront(np.zeros(15))

    outdir = model.with_suffix('') / eval_sign / 'background'

    for i, (mode, twin) in enumerate(w.twins.items()):
        if mode.index_ansi == 4: continue

        zernikes = np.zeros(15)
        zernikes[mode.index_ansi] = amplutide

        if np.all(zernikes == 0):
            savepath = outdir / f"z0"
        else:
            savepath = outdir / f"z{mode.index_ansi}"

        savepath.mkdir(parents=True, exist_ok=True)
        savepath = savepath / f"ph{photons}"

        backgrounds = np.arange(0, 2200, 200)
        wavefront = Wavefront(zernikes, lam_detection=samplegen.lam_detection, rotate=False)
        psf = samplegen.single_psf(phi=wavefront, normed=True)
        psf /= np.sum(psf)

        ref = multipoint_dataset.beads(
            image_shape=modelgen.psf_shape,
            photons=photons,
            object_size=0,
            num_objs=10,
            fill_radius=.66,
            uniform_background=0
        )

        if not override and Path(f"{savepath}_inputs.npy").exists():
            inputs = np.load(f"{savepath}_inputs.npy")
        else:
            inputs = np.zeros((len(backgrounds), *psf.shape))

            for i, w in enumerate(backgrounds):
                sample = ref.copy()
                sample[ref == 0] = w

                inputs[i] = utils.add_noise(utils.fftconvolution(
                    sample=sample,
                    kernel=psf
                ))

                imwrite(f"{savepath}_{w}.tif", inputs[i].astype(np.float32), compression='deflate', dtype=np.float32)

            inputs = np.stack(inputs, axis=0)[..., np.newaxis]
            np.save(f"{savepath}_inputs", inputs)

        if not override and Path(f"{savepath}_embeddings.npy").exists():
            embeddings = np.load(f"{savepath}_embeddings.npy")
        else:
            embeddings = np.stack([
                backend.preprocess(
                    i,
                    modelpsfgen=modelgen,
                    samplepsfgen=samplegen,
                    digital_rotations=digital_rotations,
                    remove_background=True,
                    normalize=True,
                    min_psnr=0,
                    plot=Path(f"{savepath}_{backgrounds[w]}"),
                    estimated_object_gaussian_sigma=estimated_object_gaussian_sigma
                )
                for w, i in enumerate(tqdm(
                    inputs, desc='Generating fourier embeddings', total=inputs.shape[0], file=sys.stdout
                ))
            ], axis=0)
            np.save(f"{savepath}_embeddings", embeddings)

        ys = np.stack([wavefront.amplitudes for w in backgrounds])

        embeddings = tf.data.Dataset.from_tensor_slices(embeddings)

        if not override and Path(f"{savepath}_predictions.npy").exists():
            preds = np.load(f"{savepath}_predictions.npy")
        else:
            res = backend.predict_dataset(
                model,
                inputs=embeddings,
                psfgen=modelgen,
                batch_size=batch_size,
                save_path=[f"{savepath}_{w}" for w in backgrounds],
                digital_rotations=digital_rotations,
                plot_rotations=True
            )

            try:
                preds, stdev = res
            except ValueError:
                preds, stdev, lls_defocus = res

            np.save(f"{savepath}_predictions", preds)

        residuals = ys - preds

        df = pd.DataFrame([w for w in backgrounds], columns=['backgrounds'])
        df['prediction'] = [Wavefront(i, lam_detection=modelgen.lam_detection).peak2valley(na=na) for i in preds]
        df['residuals'] = [Wavefront(i, lam_detection=modelgen.lam_detection).peak2valley(na=na) for i in residuals]
        df['moi'] = ys[:, mode.index_ansi] - preds[:, mode.index_ansi]
        df['counts'] = [np.sum(i) for i in inputs]
        df.to_csv(f'{savepath}.csv')
        print(df)

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.regplot(
            data=df,
            x="backgrounds",
            y="moi",
            scatter=True,
            truncate=False,
            order=2,
            color=".2",
            ax=ax
        )

        ax.set_yticks(np.arange(-.1, .11, .01))
        ax.set_xlim(0, backgrounds[-1])
        ax.set_ylim(-.1, .1)
        ax.axhline(y=0, color='r')
        ax.set_ylabel(r'Residuals ($y - \hat{y}$)')
        ax.set_xlabel(r'Uniform background value (photons)')
        ax.grid(True, which="both", axis='y', lw=1, ls='--', zorder=0, alpha=.5)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)

        plt.savefig(f'{savepath}.pdf', bbox_inches='tight', pad_inches=.25)
        plt.savefig(f'{savepath}.png', dpi=300, bbox_inches='tight', pad_inches=.25)
        plt.savefig(f'{savepath}.svg', dpi=300, bbox_inches='tight', pad_inches=.25)

    return savepath



@profile
def compare_models(
    models_codenames: list,
    predictions_paths: list,
    iter_num: int = 1,
    photon_range: tuple = (5e4, 1e5),
    aberration_range: tuple = (1, 2),
    outdir: Path = Path('benchmark'),
    wavelength: float = .510
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

    outdir.mkdir(parents=True, exist_ok=True)

    dataframes = []
    for codename, file in zip(models_codenames, predictions_paths):
        df = pd.read_csv(file, header=0, index_col=0)
        df = df[df['iter_num'] == iter_num]
        df['model'] = codename
        dataframes.append(df)

    df = pd.concat(dataframes)

    for c in ['residuals', 'confidence_sum', 'confidence', ]:
        test = df[
            (df.photons >= photon_range[0]) &
            (df.photons <= photon_range[1]) &
            (df.aberration >= aberration_range[0]) &
            (df.aberration <= aberration_range[1])
        ]

        if c == 'residuals':
            xmax = .5  #aberration_range[1]
            binwidth = .25
            bins = np.arange(0, xmax + binwidth, binwidth)
            label = '\n'.join([
                rf'Residuals (Peak-to-valley, $\lambda = {int(wavelength * 1000)}~nm$)',
                rf'Initial aberration [{aberration_range[0]}$\lambda$, {aberration_range[1]}$\lambda$] simulated with [{int(photon_range[0]//1000):d}k, {int(photon_range[1]//1000):d}k] integrated photons'
            ])
            outliers = test[test[c] < xmax]
            unconfident = outliers.groupby('model')['id'].count() / test.groupby('model')['id'].count()

        elif c == 'confidence':
            xmax = .02
            binwidth = .01
            bins = np.arange(0, xmax + binwidth, binwidth)
            label = '\n'.join([
                rf'Estimated error for the primary mode of aberration ($\hat{{\sigma}}$, $\lambda = {int(wavelength * 1000)}~nm$)',
                rf'Initial aberration [{aberration_range[0]}$\lambda$, {aberration_range[1]}$\lambda$] simulated with [{int(photon_range[0] // 1000):d}k, {int(photon_range[1] // 1000):d}k] integrated photons'
            ])
            # outliers = test[test[c] == 0]
            test[c].replace(0, test[c].max(), inplace=True)
            outliers = test[test[c] < xmax]
            unconfident = outliers.groupby('model')['id'].count() / test.groupby('model')['id'].count()

        else:
            xmax = .1
            binwidth = .01
            bins = np.arange(0, xmax + binwidth, binwidth)
            label = '\n'.join([
                rf'Estimated error for all modes of aberration ($\sum{{\sigma_i}}$, $\lambda = {int(wavelength * 1000)}~nm$)',
                rf'Initial aberration [{aberration_range[0]}$\lambda$, {aberration_range[1]}$\lambda$] simulated with [{int(photon_range[0]//1000):d}k, {int(photon_range[1]//1000):d}k] integrated photons'
            ])
            # outliers = test[test[c] == 0]
            test[c].replace(0, test[c].max(), inplace=True)
            outliers = test[test[c] < xmax]
            unconfident = outliers.groupby('model')['id'].count() / test.groupby('model')['id'].count()


        fig, ax = plt.subplots(figsize=(8, 6))
        histax = inset_axes(
            ax,
            width="100%",
            height="100%",
            bbox_to_anchor=(.4, .1, .3, .3),
            bbox_transform=ax.transAxes,
            loc='lower center'
        )

        g = sns.histplot(
            ax=ax,
            data=test,
            x=c,
            hue='model',
            # bins=bins,
            common_norm=False,
            common_bins=True,
            element="poly",
            stat='proportion',
            fill=False,
            cumulative=True,
        )

        # sns.violinplot(
        #     ax=histax,
        #     data=outliers,
        #     x='model',
        #     y=c,
        #     common_norm=False,
        #     common_bins=True,
        #     palette="tab10",
        # )

        sns.barplot(
            x=unconfident[models_codenames].index,
            y=unconfident[models_codenames].values,
            ax=histax,
            palette="tab10"
        )
        
        ax.set_xlabel(label)
        ax.set_ylabel('CDF')
        ax.set_xlim(0, None)
        ax.set_ylim(None, 1)
        ax.set_yticks(np.arange(0, 1.1, .1))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)
        
        hist_step = .05 if c == 'residuals' else .02
        histax.set_ylim(unconfident.min()-.01, unconfident.max())
        histax.set_yticks(np.arange(unconfident.min()-hist_step, unconfident.max()+hist_step, hist_step).round(2))
        histax.set_xlabel(f'')
        histax.set_ylabel('')
        histax.set_xticks([])
        histax.spines['top'].set_visible(False)
        histax.spines['right'].set_visible(False)
        histax.spines['left'].set_visible(False)
        histax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)
        
        axins = ax.inset_axes(
            [.3, .075, .45, .4],
            xlim=(xmax, xmax),
            ylim=(unconfident.min()-.01, unconfident.max()+.05),
        )
        ax.indicate_inset_zoom(axins, edgecolor="k")
        axins.set_xticks([])
        axins.set_yticks([])


        sns.move_legend(g, title='Training samples', frameon=False, ncol=1, loc='lower right')
        plt.tight_layout()

        savepath = Path(f'{outdir}/compare_{c}')
        plt.savefig(f'{savepath}.pdf', bbox_inches='tight', pad_inches=.25)
        plt.savefig(f'{savepath}.png', dpi=300, bbox_inches='tight', pad_inches=.25)
        plt.savefig(f'{savepath}.svg', dpi=300, bbox_inches='tight', pad_inches=.25)


@profile
def profile_models(
    models_codenames: list,
    predictions_paths: list,
    outdir: Path = Path('benchmark'),
    batch_size: int = 1024
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
    
    models = {
        'otfnet': 'Baseline',
        'vit-P8-R8-H8': '*ViT-8/8',
        'vit-P16-R8-H8': '*ViT-8/16',
        'vit-P32-R8-H8': '*ViT-8/32',
        'prototype-P8-R8-H8': '*ViT-8/8',
        'prototype-P16-R8-H8': '*ViT-8/16',
        'prototype-P32-R8-H8': '*ViT-8/32',
        'opticalnet-P32321616-R1111-H8888': 'Ours-4/32-16',
        'opticalnet-P32321616-R2222-H8888': 'Ours-8/32-16',
        'opticalnet-P32321616-R4444-H8888': 'Ours-16/32-16',
        'opticalnet-P32321616-R6666-H8888': 'Ours-24/32-16',
        'opticalnet-P321688-R2222-H8888': 'Ours-8/32-16-8-8',
        'opticalnet-P3216168-R2222-H8888': 'Ours-8/32-16-16-8',
    }
    
    outdir.mkdir(parents=True, exist_ok=True)
    dataframes = []
    for codename, modeldir in zip(models_codenames, predictions_paths):
        savepath = Path(f'{outdir}/{codename}.csv')
        if savepath.exists():
            logger.info(f"Loading {savepath}")
            df = pd.read_csv(savepath, header=0, index_col=0)
        else:

            try:
                modeldir = Path(modeldir)
                log = sorted(modeldir.rglob(r"*train/*tfevents*"))[0]
                
                df = profile_utils.load_tf_logs(log)
                model_config = ujson.loads(df.keras[0])

                best = df['epoch_loss'].idxmin()
                
                candidates = sorted(modeldir.rglob(rf"*keras/*epoch{best}.h5"))
                if len(candidates) == 0:
                    model = sorted(modeldir.rglob(rf"*keras/*.h5"))[-1]
                else:
                    model = candidates[0]
                
                model = backend.load(model, model_arch=model_config['config']['name'])
                
                df = df[[
                    'step',
                    'epoch_learning_rate',
                    'epoch_loss',
                    'epoch_mae',
                    'epoch_mse',
                    'epoch_root_mean_squared_error',
                    'epoch_weight_decay',
                    'wall_time'
                ]]
                df["wall_time"] = df.wall_time.apply(pd.Series)[0]
                df["wall_clock"] = pd.to_datetime(df.wall_time, unit="s")
                
                transformers_blocks = profile_utils.count_transformer_blocks(model=model)
                for k, v in transformers_blocks.items(): df[f'transformers_{k}'] = v
                
                df["training"] = (df.wall_clock - df.wall_clock[0]) / np.timedelta64(1, "h")
                df['transformers'] = sum(transformers_blocks.values())
                df['memory'] = profile_utils.measure_memory_usage(model=model, batch_size=batch_size)
                df['gflops'] = profile_utils.measure_gflops(model)
                df['throughput'] = profile_utils.measure_throughput(model, number_of_samples=10*1024, batch_size=batch_size)
                df['params'] = model.count_params()
                df['model'] = codename
                
                logger.info(f"Saving {savepath}")
                df.to_csv(f'{outdir}/{codename}.csv')
            
            except Exception:
                continue
        
        training_batch_size = 4096
        dataset_size = 2000000
        training_steps_per_epoch = dataset_size // training_batch_size
        df['training_gflops'] = training_steps_per_epoch * training_batch_size * df['step'] * df['gflops'] * 3
        # where the factor of 3 roughly approximates the backwards pass as being twice as compute-heavy as the forward pass
        df["training_gflops"] =  df["training_gflops"] * 1e-9
        df["gflops"] =  df["gflops"]
        df["params"] =  df["params"] * 1e-6
        
        df['cat'] = 'Baseline'
        df.loc[df.model.str.match(r'opticalnet'), 'cat'] = 'Ours'
        df.loc[df.model.str.match(r'prototype'), 'cat'] = '*ViT'
        df.model = df.model.replace(models)
        
        dataframes.append(df)
        
    df = pd.concat(dataframes).reset_index(drop=True)
    steps = [1] + list(range(49, 500, 50))
    fig, ax = plt.subplots(figsize=(8, 8))
    for cc, colormap, cmarker in zip(['*ViT', 'Ours', 'Baseline'], ['Oranges', 'Blues', 'Greys_r'], ['C1', 'C0', 'k']):
        data = df[df.step.isin(steps)][df.cat == cc]
        g = sns.lineplot(
            data=data,
            x="training_gflops",
            y="epoch_mse",
            hue="model",
            hue_order=[m if m in data.model.unique() else None for m in models.values()],
            # size="gflops",
            palette=colormap,
            dashes=False,
            marker="o",
            # sizes=sizes,
            ax=ax
        )

        best = df[df.cat == cc]['epoch_mse'].astype(np.float32).idxmin()
        data = df.iloc[best].to_frame().T
        g = sns.scatterplot(
            data=data,
            x="training_gflops",
            y="epoch_mse",
            c=cmarker,
            ax=ax
        )

        ax.text(
            data["training_gflops"] + 0.1, data["epoch_mse"], cc,
            horizontalalignment='left', size='medium', color=cmarker, weight='semibold'
        )

    ax.grid(True, which="major", axis='both', lw=.5, ls='--', zorder=0)
    ax.grid(True, which="minor", axis='both', lw=.25, ls='--', zorder=0)
    ax.set_xlabel('Training compute (Billions of GFLOPs)')
    ax.set_ylabel('MSE')
    # ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0, 20)
    ax.set_ylim(10 ** -3, 1)
    ax.legend(loc='lower left', ncol=1, title="", frameon=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    savepath = Path(f'{outdir}/compute')
    plt.savefig(f'{savepath}.pdf', bbox_inches='tight', pad_inches=.25)
    plt.savefig(f'{savepath}.png', dpi=300, bbox_inches='tight', pad_inches=.25)
    plt.savefig(f'{savepath}.svg', dpi=300, bbox_inches='tight', pad_inches=.25)

    df = pd.DataFrame()
    for d in dataframes:
        best = d['epoch_mse'].idxmin()
        df = df.append(d.iloc[best].to_frame().T, ignore_index=True)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    for cc, colormap in zip(['*ViT', 'Ours', 'Baseline'], ['C1', 'C0', 'k']):
        best = df[df.cat == cc]['epoch_mse'].astype(np.float32).idxmin()
        data = df.iloc[best].to_frame().T
        g = sns.scatterplot(
            data=data,
            x="training_gflops",
            y="epoch_mse",
            c=colormap,
            ax=ax
        )

        ax.text(
            data["training_gflops"] + 0.1, data["epoch_mse"], cc,
            horizontalalignment='left', size='medium', color=colormap, weight='semibold'
        )

    ax.grid(True, which="major", axis='both', lw=.5, ls='--', zorder=0)
    ax.grid(True, which="minor", axis='both', lw=.25, ls='--', zorder=0)
    ax.set_xlabel('Training compute (Billions of GFLOPs)')
    ax.set_ylabel('MSE')
    ax.set_yscale('log')
    ax.set_xlim(0, 11)
    ax.set_ylim(10 ** -3, 1)
    ax.legend(loc='lower left', ncol=1, title="", frameon=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    savepath = Path(f'{outdir}/best')
    plt.savefig(f'{savepath}.pdf', bbox_inches='tight', pad_inches=.25)
    plt.savefig(f'{savepath}.png', dpi=300, bbox_inches='tight', pad_inches=.25)
    plt.savefig(f'{savepath}.svg', dpi=300, bbox_inches='tight', pad_inches=.25)
    
    coi = [
        'epoch_mse',
        'training',
	    'training_gflops',
        'memory',
        'throughput',
        'gflops',
        'params',
    ]
    titles = [
        'MSE',
        f'Training Time (H) \n8xH100s [BS=4096]',
	    'Training GFLOPs (billions)',
        f'Memory (GB) \n[BS={batch_size}]',
        f'Throughput (P/S) \nRTX8000 [BS={batch_size}]',
        'GFLOPs',
        'Parameters (millions)',
    ]

    g = sns.PairGrid(df.sort_values('epoch_mse'), x_vars=coi, y_vars=["model"], hue="model", height=8, aspect=.4, palette='muted')
    g.map(sns.stripplot, size=10, orient="h", jitter=False, palette="tab10", linewidth=1, edgecolor="w")
    g.set(xlabel="", ylabel="")

    for ax, cc, title in zip(g.axes.flat, coi, titles):

        if cc == 'epoch_mse':
            ax.set_xscale('log')

        ax.set(title=title)
        ax.xaxis.grid(True)
        ax.yaxis.grid(True)
    
    plt.tight_layout()
    
    savepath = Path(f'{outdir}/profiles')
    plt.savefig(f'{savepath}.pdf', bbox_inches='tight', pad_inches=.25)
    plt.savefig(f'{savepath}.png', dpi=300, bbox_inches='tight', pad_inches=.25)
    plt.savefig(f'{savepath}.svg', dpi=300, bbox_inches='tight', pad_inches=.25)
