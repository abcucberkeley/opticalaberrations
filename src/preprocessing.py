import matplotlib
matplotlib.use('Agg')

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
import matplotlib.colors as mcolors
from matplotlib import gridspec
from tifffile import imread, imsave
from skimage import transform
from scipy.spatial import KDTree
from numpy.lib.stride_tricks import sliding_window_view
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def resize_with_crop_or_pad(psf: np.array, crop_shape: Sequence, **kwargs):
    rank = len(crop_shape)
    psf_shape = psf.shape[1:-1] if len(psf.shape) == 5 else psf.shape
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

    if len(psf.shape) == 5:
        if psf.shape[0] != 1:
            return np.array([np.pad(s[slicer], pad, **kwargs) for s in np.squeeze(psf)])[..., np.newaxis]
        else:
            return np.pad(np.squeeze(psf)[slicer], pad, **kwargs)[np.newaxis, ..., np.newaxis]
    else:
        return np.pad(psf[slicer], pad, **kwargs)


def resize(
    vol,
    voxel_size: Sequence,
    crop_shape: Any,
    sample_voxel_size: Sequence = (.1, .1, .1),
    debug: Any = None
):
    def plot(cls, img):
        if img.shape[0] == 6:
            vmin, vmax, vcenter, step = 0, 2, 1, .1
            highcmap = plt.get_cmap('YlOrRd', 256)
            lowcmap = plt.get_cmap('YlGnBu_r', 256)
            low = np.linspace(0, 1 - step, int(abs(vcenter - vmin) / step))
            high = np.linspace(0, 1 + step, int(abs(vcenter - vmax) / step))
            cmap = np.vstack((lowcmap(low), [1, 1, 1, 1], highcmap(high)))
            cmap = mcolors.ListedColormap(cmap)

            for i in range(3):
                inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=cls[i], wspace=0.1, hspace=0.1)
                ax = fig.add_subplot(inner[0])
                m = ax.imshow(img[i], cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel(r'$\alpha = |\tau| / |\hat{\tau}|$')
                ax = fig.add_subplot(inner[1])
                ax.imshow(img[i + 3], cmap='coolwarm', vmin=vmin, vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel(r'$\varphi = \angle \tau$')
                cls[i].axis('off')
        else:
            m = cls[0].imshow(np.max(img, axis=0)**.5, cmap='hot', vmin=0, vmax=1)
            cls[1].imshow(np.max(img, axis=1)**.5, cmap='hot', vmin=0, vmax=1)
            cls[2].imshow(np.max(img, axis=2)**.5, cmap='hot', vmin=0, vmax=1)
            # m = cls[0].imshow(img[img.shape[0] // 2, :, :]**.5, cmap='hot', vmin=0, vmax=1)
            # cls[1].imshow(img[:, img.shape[1] // 2, :]**.5, cmap='hot', vmin=0, vmax=1)
            # cls[2].imshow(img[:, :, img.shape[2] // 2]**.5, cmap='hot', vmin=0, vmax=1)

        cax = inset_axes(cls[2], width="10%", height="100%", loc='center right', borderpad=-2)
        cb = plt.colorbar(m, cax=cax)
        cax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    resampled_vol = transform.rescale(
        vol,
        (
            sample_voxel_size[0] / voxel_size[0],
            sample_voxel_size[1] / voxel_size[1],
            sample_voxel_size[2] / voxel_size[2],
        ),
        order=3,
        anti_aliasing=True,
    )
    if crop_shape is not None:
        mode = np.abs(st.mode(resampled_vol, axis=None).mode[0])
        resized_vol = resize_with_crop_or_pad(resampled_vol, crop_shape=crop_shape, constant_values=mode)
    else:
        resized_vol = resampled_vol

    if debug is not None:
        debug = Path(debug)
        if debug.is_dir():
            debug.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(3, 3, figsize=(11, 11))

        axes[0, 1].set_title(f"{str(vol.shape)} @ {sample_voxel_size}")
        axes[0, 0].set_ylabel('Input (MIP)')
        plot(axes[0, :], vol)

        axes[1, 1].set_title(f"{str(resampled_vol.shape)} @ {voxel_size}")
        axes[1, 0].set_ylabel('Resampled (MIP)')
        plot(axes[1, :], resampled_vol)
        imsave(f'{debug}_resampled_psf.tif', resampled_vol)

        axes[2, 1].set_title(str(resized_vol.shape))
        axes[2, 0].set_ylabel('Resized (MIP)')
        plot(axes[2, :], resized_vol)
        imsave(f'{debug}_resized_psf.tif', resized_vol)

        if debug == True:
            plt.show()
        else:
            plt.savefig(f'{debug}_rescaling.png', dpi=300, bbox_inches='tight', pad_inches=.25)

    return resized_vol


def prep_sample(
    sample: np.array,
    crop_shape: Any,
    sample_voxel_size: tuple,
    model_voxel_size: tuple,
    debug: Any = None,
    remove_background: bool = True,
):
    if len(np.squeeze(sample).shape) == 4:
        samples = []
        for i in range(sample.shape[0]):
            s = sample[i]
            mode = int(st.mode(s[s < np.quantile(s, .99)], axis=None).mode[0])

            if remove_background:
                s -= mode
                s[s < 0] = 0

            s /= np.nanmax(s)

            if not all(s1 == s2 for s1, s2 in zip(sample_voxel_size, model_voxel_size)):
                s = resize(
                    s,
                    sample_voxel_size=sample_voxel_size,
                    voxel_size=model_voxel_size,
                    crop_shape=crop_shape,
                    debug=debug/f"{i}_preprocessing" if debug is not None else None
                )
            s = s.transpose(0, 2, 1)
            samples.append(s)

        return np.array(samples)

    else:
        mode = int(st.mode(sample[sample < np.quantile(sample, .99)], axis=None).mode[0])

        if remove_background:
            sample -= mode
            sample[sample < 0] = 0

        sample = sample / np.nanmax(sample)

        if not all(s1 == s2 for s1, s2 in zip(sample_voxel_size, model_voxel_size)):
            sample = resize(
                sample,
                sample_voxel_size=sample_voxel_size,
                voxel_size=model_voxel_size,
                crop_shape=crop_shape,
                debug=debug
            )

        sample = sample.transpose(0, 2, 1)
        return sample


def find_roi(
    path: Union[Path, np.array],
    savepath: Path,
    window_size: tuple = (64, 64, 64),
    plot: Any = None,
    num_peaks: Any = None,
    min_dist: Any = 1,
    max_dist: Any = None,
    min_intensity: Any = 100,
    peaks: Any = None,
    max_neighbor: int = 5,
    voxel_size: tuple = (.200, .108, .108),
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
        logger.info(f"Sample: {dataset.shape}")
    elif path.suffix == '.zarr':
        dataset = zarr.open_array(str(path), mode='r', order='F')
        logger.info(f"Sample: {dataset.shape}")
    else:
        logger.error(f"Unknown file format: {path.name}")
        return

    if isinstance(peaks, str) or isinstance(peaks, Path):
        with h5py.File(peaks, 'r') as file:
            file = file.get('frameInfo')
            peaks = pd.DataFrame(
                np.hstack((file['x'], file['y'], file['z'], file['A'])),
                columns=['x', 'y', 'z', 'A']
            ).round(0).astype(int)

    points = peaks[['z', 'y', 'x']].values
    scaled_peaks = np.zeros_like(points)
    scaled_peaks[:, 0] = points[:, 0] * voxel_size[0]
    scaled_peaks[:, 1] = points[:, 1] * voxel_size[1]
    scaled_peaks[:, 2] = points[:, 2] * voxel_size[2]

    kd = KDTree(scaled_peaks)
    dist, idx = kd.query(scaled_peaks, k=11, workers=-1)
    for n in range(1, 11):
        if n == 1:
            peaks[f'dist'] = dist[:, n]
        else:
            peaks[f'dist_{n}'] = dist[:, n]

    # filter out points too close to the edge
    lzedge = peaks['z'] >= window_size[0]//4
    hzedge = peaks['z'] <= dataset.shape[0] - window_size[0]//4
    lyedge = peaks['y'] >= window_size[1]//4
    hyedge = peaks['y'] <= dataset.shape[1] - window_size[1]//4
    lxedge = peaks['x'] >= window_size[2]//4
    hxedge = peaks['x'] <= dataset.shape[2] - window_size[2]//4
    peaks = peaks[lzedge & hzedge & lyedge & hyedge & lxedge & hxedge]

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        sns.scatterplot(ax=axes[0], x=peaks['dist'], y=peaks['A'], s=5, color="C0")
        sns.kdeplot(ax=axes[0], x=peaks['dist'], y=peaks['A'], levels=5, color="grey", linewidths=1)
        axes[0].set_ylabel('Intensity')
        axes[0].set_xlabel('Distance (microns)')
        axes[0].set_yscale('log')
        axes[0].set_ylim(10 ** 0, None)
        axes[0].set_xlim(0, None)
        axes[0].grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

        x = np.sort(peaks['dist'])
        y = np.arange(len(x)) / float(len(x))
        axes[1].plot(x, y, color='dimgrey')
        axes[1].set_xlabel('Distance (microns)')
        axes[1].set_ylabel('CDF')
        axes[1].set_xlim(0, None)
        axes[1].grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

        sns.histplot(ax=axes[2], data=peaks, x="dist", kde=True)
        axes[2].set_xlabel('Distance')
        axes[2].set_xlim(0, None)
        axes[2].grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

        plt.tight_layout()
        plt.savefig(f'{plot}_detected_points.png', bbox_inches='tight', dpi=300, pad_inches=.25)

    if min_dist is not None:
        peaks = peaks[peaks['dist'] >= min_dist]

    if max_dist is not None:
        peaks = peaks[peaks['dist'] <= max_dist]

    if min_intensity is not None:
        peaks = peaks[peaks['A'] >= min_intensity]

    neighbors = peaks.columns[peaks.columns.str.startswith('dist')].tolist()
    min_dist = np.min(window_size)*np.min(voxel_size)
    peaks['neighbors'] = peaks[peaks[neighbors] <= min_dist].count(axis=1)
    peaks.sort_values(by=['neighbors', 'dist', 'A'], ascending=[True, False, False], inplace=True)
    peaks = peaks[peaks['neighbors'] <= max_neighbor]

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        sns.scatterplot(ax=axes[0], x=peaks['dist'], y=peaks['A'], s=5, color="C0")
        sns.kdeplot(ax=axes[0], x=peaks['dist'], y=peaks['A'], levels=5, color="grey", linewidths=1)
        axes[0].set_ylabel('Intensity')
        axes[0].set_xlabel('Distance')
        axes[0].set_xlim(0, None)
        axes[0].grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

        x = np.sort(peaks['dist'])
        y = np.arange(len(x)) / float(len(x))
        axes[1].plot(x, y, color='dimgrey')
        axes[1].set_xlabel('Distance')
        axes[1].set_ylabel('CDF')
        axes[1].set_xlim(0, None)
        axes[1].grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

        sns.histplot(ax=axes[2], data=peaks, x="dist", kde=True)
        axes[2].set_xlabel('Distance')
        axes[2].set_xlim(0, None)
        axes[2].grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

        plt.tight_layout()
        plt.savefig(f'{plot}_selected_points.png', bbox_inches='tight', dpi=300, pad_inches=.25)

    peaks = peaks.head(num_peaks)
    peaks.to_csv(f"{plot}_rois.csv")

    logger.info(f"Predicted points of interest")
    print(peaks)

    peaks = peaks[['z', 'y', 'x']].values[:num_peaks]
    widths = [w // 2 for w in window_size]

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False, sharex=False)
        for ax in range(3):
            axes[ax].imshow(
                np.nanmax(dataset, axis=ax),
                aspect='auto',
                cmap='hot'
            )

            for p in range(peaks.shape[0]):
                if ax == 0:
                    axes[ax].plot(peaks[p, 2], peaks[p, 1], marker='.', ls='', color=f'C{p}')
                    axes[ax].add_patch(patches.Rectangle(
                        xy=(peaks[p, 2] - window_size[2] // 2, peaks[p, 1] - window_size[1] // 2),
                        width=window_size[1],
                        height=window_size[2],
                        fill=None,
                        color=f'C{p}',
                        alpha=1
                    ))
                elif ax == 1:
                    axes[ax].plot(peaks[p, 2], peaks[p, 0], marker='.', ls='', color=f'C{p}')
                    axes[ax].add_patch(patches.Rectangle(
                        xy=(peaks[p, 2] - window_size[2] // 2, peaks[p, 0] - window_size[0] // 2),
                        width=window_size[1],
                        height=window_size[2],
                        fill=None,
                        color=f'C{p}',
                        alpha=1
                    ))
                else:
                    axes[ax].plot(peaks[p, 1], peaks[p, 0], marker='.', ls='', color=f'C{p}')
                    axes[ax].add_patch(patches.Rectangle(
                        xy=(peaks[p, 1] - window_size[1] // 2, peaks[p, 0] - window_size[0] // 2),
                        width=window_size[1],
                        height=window_size[2],
                        fill=None,
                        color=f'C{p}',
                        alpha=1
                    ))

        plt.tight_layout()
        plt.savefig(f'{plot}_rois.png', bbox_inches='tight', dpi=300, pad_inches=.25)

    logger.info(f"Locating ROIs: {[peaks.shape[0]]}")
    for p in range(peaks.shape[0]):
        start = [
            peaks[p, s] - widths[s] if peaks[p, s] >= widths[s] else 0
            for s in range(3)
        ]
        end = [
            peaks[p, s] + widths[s] if peaks[p, s] + widths[s] < dataset.shape[s] else dataset.shape[s]
            for s in range(3)
        ]
        r = dataset[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

        if r.size != 0:
            r = resize_with_crop_or_pad(r, crop_shape=window_size)
            imsave(savepath/f"roi_{p:02}.tif", r)


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
        logger.info(f"Sample: {dataset.shape}")
    elif path.suffix == '.zarr':
        dataset = zarr.open_array(str(path), mode='r', order='F')
        logger.info(f"Sample: {dataset.shape}")
    else:
        logger.error(f"Unknown file format: {path.name}")
        return

    logger.info(f"Sample: {[dataset.shape]}")
    windows = sliding_window_view(dataset, window_shape=window_size)[::strides, ::strides, ::strides]
    zplanes, nrows, ncols = windows.shape[:3]
    windows = np.reshape(windows, (-1, *window_size))

    logger.info(f"Locating ROIs: {[windows.shape[0]]}")
    for i, w in enumerate(windows):
        imsave(savepath/f"roi_{i:02}.tif", w)

    return zplanes, nrows, ncols
