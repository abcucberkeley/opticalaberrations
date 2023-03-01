import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.set_loglevel('error')

import logging
import sys
from pathlib import Path
from typing import Any, Sequence, Union
import numpy as np
from scipy import stats as st
import pandas as pd
import seaborn as sns
import zarr
import h5py
import scipy.io
from tqdm.contrib import itertools
from tifffile import imread, imsave
from scipy.spatial import KDTree
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.patches as patches
from line_profiler_pycharm import profile
from skimage.morphology import ball
from skimage.morphology import dilation
from canny import CannyEdgeDetector3D

from vis import plot_mip

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def round_to_even(n):
    answer = round(n)
    if not answer % 2:
        return int(answer)
    if abs(answer + 1 - n) < abs(answer - 1 - n):
        return int(answer + 1)
    else:
        return int(answer - 1)


def round_to_odd(n):
    answer = round(n)
    if answer % 2:
        return int(answer)
    if abs(answer + 1 - n) < abs(answer - 1 - n):
        return int(answer + 1)
    else:
        return int(answer - 1)


@profile
def resize_with_crop_or_pad(img: np.array, crop_shape: Sequence, **kwargs):
    """Crops or pads array.  Output will have dimensions "crop_shape". No interpolation. Padding type
    can be customized with **kwargs, like "reflect" to get mirror pad.

    Args:
        img (np.array): N-dim array
        crop_shape (Sequence): desired output dimensions
        **kwargs: arguments to pass to np.pad

    Returns:
        N-dim array with desired output shape
    """
    rank = len(crop_shape)
    psf_shape = img.shape[1:-1] if len(img.shape) == 5 else img.shape
    index = [[0, psf_shape[d]] for d in range(rank)]
    pad = [[0, 0] for _ in range(rank)]
    slicer = [slice(None)] * rank

    for i in range(rank):
        if psf_shape[i] < crop_shape[i]:
            pad[i][0] = (crop_shape[i] - psf_shape[i]) // 2
            pad[i][1] = crop_shape[i] - psf_shape[i] - pad[i][0]
        else:
            index[i][0] = int(np.floor((psf_shape[i] - crop_shape[i]) / 2.))
            index[i][1] = index[i][0] + crop_shape[i]

        slicer[i] = slice(index[i][0], index[i][1])

    if len(img.shape) == 5:
        if img.shape[0] != 1:
            return np.array([np.pad(s[slicer], pad, **kwargs) for s in np.squeeze(img)])[..., np.newaxis]
        else:
            return np.pad(np.squeeze(img)[slicer], pad, **kwargs)[np.newaxis, ..., np.newaxis]
    else:
        return np.pad(img[tuple(slicer)], pad, **kwargs)


def remove_background_noise(image, read_noise_bias: float = 5):
    """ A simple function to remove background noise from a given image """
    mode = int(st.mode(image, axis=None).mode[0])
    image -= mode + read_noise_bias
    image[image < 0] = 0
    return image


@profile
def prep_sample(
    sample: np.array,
    sample_voxel_size: tuple,
    model_fov: tuple,
    debug: Any = None,
    remove_background: bool = True,
    normalize: bool = True,
    edge_filter: bool = False,
    filter_mask_dilation: bool = True
):
    """ Input 3D array (or series of 3D arrays) is preprocessed in this order:

        -Background subtraction
        -Normalization to 0-1
        -Crop (or zero pad) to model field of view

    Args:
        sample (np.array): Input 3D array (or series of 3D arrays)
        sample_voxel_size (tuple): 
        model_fov (tuple):
        debug (Any, optional): plot or save .svg's. Defaults to None.
        remove_background (bool, optional): Defaults to True.
        normalize (bool, optional): Defaults to True.
        edge_filter (bool, optional): Defaults to True.

    Returns:
        _type_: 3D array (or series of 3D arrays)
    """
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    })

    sample = np.nan_to_num(sample, nan=0, posinf=0, neginf=0)

    if debug is not None:
        debug = Path(debug)
        if debug.is_dir(): debug.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, ncols=4, figsize=(12, 5))

        plot_mip(
            vol=sample,
            xy=axes[0, 0],
            xz=axes[0, 1],
            yz=axes[0, 2],
            dxy=sample_voxel_size[-1],
            dz=sample_voxel_size[0],
            label='Input (MIP) [$\gamma$=0.5]'
        )

        sns.histplot(
            sample.flatten(),
            bins=200,
            ax=axes[0, -1],
            label='Input',
            color='C0',
        )
        axes[0, -1].yaxis.tick_right()
        axes[0, -1].yaxis.set_label_position("right")
        axes[0, -1].grid(True, which="both", axis='y', lw=1, ls='--', zorder=0)
        axes[0, -1].legend(frameon=False, loc='upper right', ncol=1)
        axes[0, -1].set_yscale('symlog')
        axes[0, -1].set_xlim(0, None)

    if remove_background:
        sample = remove_background_noise(sample)

        if debug is not None:
            sns.histplot(
                sample.flatten(),
                bins=200,
                ax=axes[0, -1],
                label='No background',
                color='C1',
                alpha=.5
            )
            axes[0, -1].yaxis.tick_right()
            axes[0, -1].yaxis.set_label_position("right")
            axes[0, -1].grid(True, which="both", axis='y', lw=1, ls='--', zorder=0)
            axes[0, -1].legend(frameon=False, loc='upper right', ncol=1)
            axes[0, -1].set_yscale('symlog')
            axes[0, -1].set_xlim(0, None)

    if normalize:
        sample /= np.nanmax(sample)

    if edge_filter:
        mask = CannyEdgeDetector3D(
            sample,
            sigma=.5,
            lowthresholdratio=0.3,
            highthresholdratio=0.2,
            weak_voxel=1.5*(st.mode(sample, axis=None).mode[0] + 1e-6),
            strong_voxel=np.nanmax(sample)
        ).detect().astype(np.float)

        if filter_mask_dilation:
            mask = dilation(mask, ball(2)).astype(float)

        sample *= mask

    # match the sample's FOV to the iPSF FOV. This will make equal pixel spacing in the OTFs.
    number_of_desired_sample_pixels = (
        round_to_even(model_fov[0] / sample_voxel_size[0]),
        round_to_even(model_fov[1] / sample_voxel_size[1]),
        round_to_even(model_fov[2] / sample_voxel_size[2]),
    )

    if not all(s1 == s2 for s1, s2 in zip(number_of_desired_sample_pixels, sample.shape)):
        sample = resize_with_crop_or_pad(
            sample,
            crop_shape=number_of_desired_sample_pixels
        )

    if debug is not None:
        plot_mip(
            vol=sample,
            xy=axes[1, 0],
            xz=axes[1, 1],
            yz=axes[1, 2],
            dxy=sample_voxel_size[-1],
            dz=sample_voxel_size[0],
            label='Processed (MIP) [$\gamma$=0.5]'
        )

        sns.histplot(
            sample.flatten(),
            bins=200,
            ax=axes[1, -1],
            color='dimgrey',
        )
        axes[1, -1].yaxis.tick_right()
        axes[1, -1].yaxis.set_label_position("right")
        axes[1, -1].grid(True, which="both", axis='y', lw=1, ls='--', zorder=0)
        axes[1, -1].set_yscale('symlog')
        axes[1, -1].set_xlim(0, None)

        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.3, wspace=0.15)
        plt.savefig(f'{debug}_rescaling.svg', dpi=300, bbox_inches='tight', pad_inches=.25)

    return sample


@profile
def find_roi(
    path: Union[Path, np.array],
    savepath: Path,
    window_size: tuple = (64, 64, 64),
    plot: Any = None,
    num_rois: Any = None,
    min_dist: Any = 1,
    max_dist: Any = None,
    min_intensity: Any = 100,
    pois: Any = None,
    max_neighbor: int = 5,
    voxel_size: tuple = (.200, .108, .108),
    timestamp: int = 17
):
    ztiles = 1
    ncols = int(np.ceil(num_rois / 5))
    nrows = int(np.ceil(num_rois / ncols))
    savepath.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    if isinstance(path, (np.ndarray, np.generic)):
        dataset = path
    elif path.suffix == '.tif':
        dataset = imread(path).astype(np.float)
    elif path.suffix == '.zarr':
        dataset = zarr.open_array(str(path), mode='r', order='F')
    else:
        logger.error(f"Unknown file format: {path.name}")
        return

    if isinstance(pois, str) or isinstance(pois, Path):
        try:
            with h5py.File(pois, 'r') as file:
                file = file.get('frameInfo')
                pois = pd.DataFrame(
                    np.hstack((file['x'], file['y'], file['z'], file['A'], file['c'], file['isPSF'])),
                    columns=['x', 'y', 'z', 'A', 'c', 'isPSF']
                ).round(0).astype(int)
        except OSError:
            file = scipy.io.loadmat(pois)
            file = file.get('frameInfo')
            pois = pd.DataFrame(
                np.vstack((
                    file['x'][0][timestamp+1][0],
                    file['y'][0][timestamp+1][0],
                    file['z'][0][timestamp+1][0],
                    file['A'][0][timestamp+1][0],
                    file['c'][0][timestamp+1][0],
                    file['isPSF'][0][timestamp+1][0],
                )).T,
                columns=['x', 'y', 'z', 'A', 'c', 'isPSF']
            ).round(0).astype(int)

        # index by zero like every other good language (stupid, matlab!)
        pois[['z', 'y', 'x']] -= 1

    pois = pois[pois['isPSF'] == 1]
    points = pois[['z', 'y', 'x']].values
    scaled_peaks = np.zeros_like(points)
    scaled_peaks[:, 0] = points[:, 0] * voxel_size[0]
    scaled_peaks[:, 1] = points[:, 1] * voxel_size[1]
    scaled_peaks[:, 2] = points[:, 2] * voxel_size[2]

    kd = KDTree(scaled_peaks)
    dist, idx = kd.query(scaled_peaks, k=11, workers=-1)
    for n in range(1, 11):
        if n == 1:
            pois[f'dist'] = dist[:, n]
        else:
            pois[f'dist_{n}'] = dist[:, n]

    # filter out points too close to the edge
    lzedge = pois['z'] >= window_size[0]//4
    hzedge = pois['z'] <= dataset.shape[0] - window_size[0]//4
    lyedge = pois['y'] >= window_size[1]//4
    hyedge = pois['y'] <= dataset.shape[1] - window_size[1]//4
    lxedge = pois['x'] >= window_size[2]//4
    hxedge = pois['x'] <= dataset.shape[2] - window_size[2]//4
    pois = pois[lzedge & hzedge & lyedge & hyedge & lxedge & hxedge]

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        sns.scatterplot(ax=axes[0], x=pois['dist'], y=pois['A'], s=5, color="C0")
        sns.kdeplot(ax=axes[0], x=pois['dist'], y=pois['A'], levels=5, color="grey", linewidths=1)
        axes[0].set_ylabel('Intensity')
        axes[0].set_xlabel('Distance (microns)')
        axes[0].set_yscale('log')
        axes[0].set_ylim(10 ** 0, None)
        axes[0].set_xlim(0, None)
        axes[0].grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

        x = np.sort(pois['dist'])
        y = np.arange(len(x)) / float(len(x))
        axes[1].plot(x, y, color='dimgrey')
        axes[1].set_xlabel('Distance (microns)')
        axes[1].set_ylabel('CDF')
        axes[1].set_xlim(0, None)
        axes[1].grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

        sns.histplot(ax=axes[2], data=pois, x="dist", kde=True)
        axes[2].set_xlabel('Distance')
        axes[2].set_xlim(0, None)
        axes[2].grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

        plt.tight_layout()
        plt.savefig(f'{plot}_detected_points.svg', bbox_inches='tight', dpi=300, pad_inches=.25)

    if min_dist is not None:
        pois = pois[pois['dist'] >= min_dist]

    if max_dist is not None:
        pois = pois[pois['dist'] <= max_dist]

    if min_intensity is not None:
        pois = pois[pois['A'] >= min_intensity]

    neighbors = pois.columns[pois.columns.str.startswith('dist')].tolist()
    min_dist = np.min(window_size)*np.min(voxel_size)
    pois['neighbors'] = pois[pois[neighbors] <= min_dist].count(axis=1)
    pois.sort_values(by=['neighbors', 'dist', 'A'], ascending=[True, False, False], inplace=True)
    pois = pois[pois['neighbors'] <= max_neighbor]

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        sns.scatterplot(ax=axes[0], x=pois['dist'], y=pois['A'], s=5, color="C0")
        sns.kdeplot(ax=axes[0], x=pois['dist'], y=pois['A'], levels=5, color="grey", linewidths=1)
        axes[0].set_ylabel('Intensity')
        axes[0].set_xlabel('Distance')
        axes[0].set_xlim(0, None)
        axes[0].grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

        x = np.sort(pois['dist'])
        y = np.arange(len(x)) / float(len(x))
        axes[1].plot(x, y, color='dimgrey')
        axes[1].set_xlabel('Distance')
        axes[1].set_ylabel('CDF')
        axes[1].set_xlim(0, None)
        axes[1].grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

        sns.histplot(ax=axes[2], data=pois, x="dist", kde=True)
        axes[2].set_xlabel('Distance')
        axes[2].set_xlim(0, None)
        axes[2].grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

        plt.tight_layout()
        plt.savefig(f'{plot}_selected_points.svg', bbox_inches='tight', dpi=300, pad_inches=.25)

    pois = pois.head(num_rois)
    pois.to_csv(f"{plot}_stats.csv")

    pois = pois[['z', 'y', 'x']].values[:num_rois]
    widths = [w // 2 for w in window_size]

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=False, sharex=False)
        for ax in range(2):
            axes[ax].imshow(
                np.nanmax(dataset, axis=ax),
                aspect='equal',
                cmap='Greys_r',
            )

            for p in range(pois.shape[0]):
                if ax == 0:
                    axes[ax].plot(pois[p, 2], pois[p, 1], marker='.', ls='', color=f'C{p}')
                    axes[ax].add_patch(patches.Rectangle(
                        xy=(pois[p, 2] - window_size[2] // 2, pois[p, 1] - window_size[1] // 2),
                        width=window_size[1],
                        height=window_size[2],
                        fill=None,
                        color=f'C{p}',
                        alpha=1
                    ))
                    axes[ax].set_title('XY')
                elif ax == 1:
                    axes[ax].plot(pois[p, 2], pois[p, 0], marker='.', ls='', color=f'C{p}')
                    axes[ax].add_patch(patches.Rectangle(
                        xy=(pois[p, 2] - window_size[2] // 2, pois[p, 0] - window_size[0] // 2),
                        width=window_size[1],
                        height=window_size[2],
                        fill=None,
                        color=f'C{p}',
                        alpha=1
                    ))
                    axes[ax].set_title('XZ')

        plt.tight_layout()
        plt.savefig(f'{plot}_mips.svg', bbox_inches='tight', dpi=300, pad_inches=.25)

    rois = []
    for p, (z, y, x) in enumerate(itertools.product(
        range(ztiles), range(nrows), range(ncols),
        desc=f"Locating tiles: {[pois.shape[0]]}")
    ):
        start = [
            pois[p, s] - widths[s] if pois[p, s] >= widths[s] else 0
            for s in range(3)
        ]
        end = [
            pois[p, s] + widths[s] if pois[p, s] + widths[s] < dataset.shape[s] else dataset.shape[s]
            for s in range(3)
        ]
        r = dataset[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

        if r.size != 0:
            tile = f"z{0}-y{y}-x{x}"
            imsave(savepath / f"{tile}.tif", r)
            rois.append(savepath / f"{tile}.tif")

    return np.array(rois), ztiles, nrows, ncols


@profile
def get_tiles(
    path: Union[Path, np.array],
    savepath: Path,
    window_size: tuple = (64, 64, 64),
    strides: int = 64,
):
    savepath.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    if isinstance(path, (np.ndarray, np.generic)):
        dataset = path
    elif path.suffix == '.tif':
        dataset = imread(path).astype(np.float)
    elif path.suffix == '.zarr':
        dataset = zarr.open_array(str(path), mode='r', order='F')
    else:
        logger.error(f"Unknown file format: {path.name}")
        return

    windows = sliding_window_view(dataset, window_shape=window_size)[::strides, ::strides, ::strides]
    ztiles, nrows, ncols = windows.shape[:3]
    windows = np.reshape(windows, (-1, *window_size))

    rois = []
    for i, (z, y, x) in enumerate(itertools.product(
        range(ztiles), range(nrows), range(ncols),
        desc=f"Locating tiles: {[windows.shape[0]]}")
    ):
        tile = f"z{0}-y{y}-x{x}"
        imsave(savepath / f"{tile}.tif", windows[i])
        rois.append(savepath / f"{tile}.tif")

    return np.array(rois), ztiles, nrows, ncols
