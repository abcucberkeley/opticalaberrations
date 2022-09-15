import logging
import sys
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pandas as pd
import zarr
from tqdm import trange
import h5py
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib import gridspec
from tifffile import imread, imsave
from skimage import transform, filters, feature
from scipy.spatial import KDTree
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
    index = [[0, psf.shape[d]] for d in range(rank)]
    pad = [[0, 0] for _ in range(rank)]
    slicer = [slice(None)] * rank

    for i in range(rank):
        if psf.shape[i] < crop_shape[i]:
            pad[i][0] = (crop_shape[i] - psf.shape[i]) // 2
            pad[i][1] = crop_shape[i] - psf.shape[i] - pad[i][0]
        else:
            index[i][0] = int(np.floor((psf.shape[i] - crop_shape[i]) / 2.))
            index[i][1] = index[i][0] + crop_shape[i]

        slicer[i] = slice(index[i][0], index[i][1])
    return np.pad(psf[slicer], pad, **kwargs)


def resize(vol, voxel_size: Sequence, crop_shape: Sequence, sample_voxel_size: Sequence = (.1, .1, .1),
           debug: Any = None):
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
            # m = cls[0].imshow(np.max(img, axis=0), cmap='hot', vmin=0, vmax=1)
            # cls[1].imshow(np.max(img, axis=1), cmap='hot', vmin=0, vmax=1)
            # cls[2].imshow(np.max(img, axis=2).T, cmap='hot', vmin=0, vmax=1)
            m = cls[0].imshow(img[img.shape[0] // 2, :, :], cmap='hot', vmin=0, vmax=1)
            cls[1].imshow(img[:, img.shape[1] // 2, :], cmap='hot', vmin=0, vmax=1)
            cls[2].imshow(img[:, :, img.shape[2] // 2].T, cmap='hot', vmin=0, vmax=1)

        cax = inset_axes(cls[2], width="10%", height="50%", loc='center right', borderpad=-2)
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
    resized_psf = resize_with_crop_or_pad(resampled_vol, crop_shape)

    if debug is not None:
        fig, axes = plt.subplots(3, 3, figsize=(8, 11))

        axes[0, 1].set_title(f"{str(vol.shape)} @ {sample_voxel_size}")
        axes[0, 0].set_ylabel('Input (middle)')
        plot(axes[0, :], vol)

        axes[1, 1].set_title(f"{str(resampled_vol.shape)} @ {voxel_size}")
        axes[1, 0].set_ylabel('Resampled (middle)')
        plot(axes[1, :], resampled_vol)
        imsave(f'{debug}_resampled_psf.tif', resampled_vol)

        axes[2, 1].set_title(str(resized_psf.shape))
        axes[2, 0].set_ylabel('Resized (middle)')
        plot(axes[2, :], resized_psf)
        imsave(f'{debug}_resized_psf.tif', resized_psf)

        if debug == True:
            plt.show()
        else:
            plt.savefig(f'{debug}_rescaling.png', dpi=300, bbox_inches='tight', pad_inches=.25)

    return resized_psf


def prep_psf(path: Path, input_shape: tuple, sample_voxel_size: tuple, model_voxel_size: tuple):
    psf = imread(path)
    psf = psf.transpose(0, 2, 1)
    psf = psf / np.max(psf)
    psf = resize(
        psf,
        sample_voxel_size=sample_voxel_size,
        voxel_size=model_voxel_size,
        crop_shape=tuple(3 * [input_shape[-1]])
    )
    return psf


def find_roi(
        path: Path,
        window_size: tuple = (64, 64, 64),
        plot: Any = True,
        num_peaks: Any = None,
        min_dist: int = 32,
        min_intensity: int = 100,
        peaks_coordinates: Any = None,
        file_order: str = 'c-order'
):
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    fov_size: dict = {'x': 256, 'y': 256, 'z': 256}

    if path.suffix == '.tif':
        dataset = imread(path).astype(np.float)
        logger.info(f"Sample: {dataset.shape}")

    elif path.suffix == '.zarr':
        dataset = zarr.open_array(str(path), mode='r', order='F')
        logger.info(f"Sample: {dataset.shape}")
    else:
        logger.error(f"Unknown file format: {path.name}")
        return

    if peaks_coordinates is not None:
        with h5py.File(peaks_coordinates, 'r') as file:
            file = file.get('frameInfo')
            peaks = pd.DataFrame(
                np.hstack((file['x'], file['y'], file['z'], file['A'])),
                columns=['x', 'y', 'z', 'A']
            ).round(0).astype(int)

            kd = KDTree(peaks[['z', 'y', 'x']].values)
            dist, idx = kd.query(peaks[['z', 'y', 'x']].values, k=2, workers=-1)
            peaks['dist'] = dist[:, 1]
            print(peaks)

            if plot:
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                sns.scatterplot(ax=axes[0], x=peaks['dist'], y=peaks['A'], s=5, color="k")
                sns.kdeplot(ax=axes[0], x=peaks['dist'], y=peaks['A'], levels=5, color="grey", linewidths=1)
                axes[0].set_ylabel('Intensity')
                axes[0].set_xlabel('Distance')
                axes[0].set_yscale('log')
                axes[0].set_ylim(10**0, None)
                axes[0].grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

                x = np.sort(peaks['dist'])
                y = np.arange(len(x)) / float(len(x))
                axes[1].plot(x, y, color='dimgrey')
                axes[1].set_xlabel('Distance')
                axes[1].set_ylabel('CDF')
                axes[1].grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

                sns.histplot(ax=axes[2], data=peaks, x="dist", kde=True)
                axes[2].set_xlabel('Distance')
                axes[2].grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

                plt.tight_layout()
                plt.savefig(plot / f'detected_points.png', bbox_inches='tight', dpi=300, pad_inches=.25)

            peaks = peaks[peaks['dist'] >= min_dist]
            peaks = peaks[peaks['A'] >= min_intensity]
            peaks.sort_values(by=['dist', 'A'], ascending=[False, False], inplace=True)
            logger.info(f"Peaks w/ Min-Dist & PSNR")
            print(peaks)

            if plot:
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                sns.scatterplot(ax=axes[0], x=peaks['dist'], y=peaks['A'], s=5, color="k")
                sns.kdeplot(ax=axes[0], x=peaks['dist'], y=peaks['A'], levels=5, color="grey", linewidths=1)
                axes[0].set_ylabel('Intensity')
                axes[0].set_xlabel('Distance')
                axes[0].set_ylim(0, None)
                axes[0].grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

                x = np.sort(peaks['dist'])
                y = np.arange(len(x)) / float(len(x))
                axes[1].plot(x, y, color='dimgrey')
                axes[1].set_xlabel('Distance')
                axes[1].set_ylabel('CDF')
                axes[1].grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

                sns.histplot(ax=axes[2], data=peaks, x="dist", kde=True)
                axes[2].set_xlabel('Distance')
                axes[2].grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

                plt.tight_layout()
                plt.savefig(plot / f'selected_points.png', bbox_inches='tight', dpi=300, pad_inches=.25)

            peaks = peaks[['z', 'y', 'x']].values[:num_peaks]
    else:
        sliding_windows = np.lib.stride_tricks.sliding_window_view(
            dataset,
            window_shape=(fov_size['y'], fov_size['x'], fov_size['z']),
        )[::fov_size['y'] // 2, ::fov_size['x'] // 2, ::fov_size['z'] // 2]
        sliding_windows = sliding_windows.reshape((-1, fov_size['y'], fov_size['x'], fov_size['z']))

        rois = []
        for i in trange(sliding_windows.shape[0], desc=f'Sliding windows {sliding_windows.shape}'):
            fov = sliding_windows[i].astype(np.float)

            if np.count_nonzero(fov) != 0:
                if file_order != 'c-order':
                    # Convert F-order to C-order
                    fov = np.swapaxes(fov, 0, -1)
                    fov = np.swapaxes(fov, 1, -1)
                    fov /= np.nanmax(fov)

                peaks = feature.peak_local_max(
                    fov,
                    min_distance=min_dist,
                    exclude_border=min_dist,
                    num_peaks=num_peaks,
                    threshold_rel=.5
                )

                for p in range(peaks.shape[0]):
                    rois.append(fov[
                                peaks[p, 0] - window_size[0] // 2:peaks[p, 0] + window_size[0] // 2,
                                peaks[p, 1] - window_size[1] // 2:peaks[p, 1] + window_size[1] // 2,
                                peaks[p, 2] - window_size[2] // 2:peaks[p, 2] + window_size[2] // 2
                                ])

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False, sharex=False)
        for ax in range(3):
            axes[ax].imshow(
                np.nanmax(dataset, axis=ax) ** .5,
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
        plt.savefig(plot / f'rois.png', bbox_inches='tight', dpi=300, pad_inches=.25)

    rois = []
    for p in range(peaks.shape[0]):
        r = dataset[
            peaks[p, 0] - window_size[0] // 2:peaks[p, 0] + window_size[0] // 2,
            peaks[p, 1] - window_size[1] // 2:peaks[p, 1] + window_size[1] // 2,
            peaks[p, 2] - window_size[2] // 2:peaks[p, 2] + window_size[2] // 2
        ]
        r = resize_with_crop_or_pad(r, crop_shape=window_size)
        if r.size != 0:
            rois.append(r)

    return np.array(rois, dtype=np.float)
