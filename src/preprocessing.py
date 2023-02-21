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
import scipy.io
import matplotlib.colors as mcolors
from matplotlib import gridspec
from tifffile import imread, imsave
from skimage import transform
from scipy.spatial import KDTree
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from line_profiler_pycharm import profile
from skimage.filters import difference_of_gaussians, window
from skimage.morphology import ball
from skimage.morphology import erosion, opening, dilation
from canny import CannyEdgeDetector3D


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@profile
def resize_with_crop_or_pad(psf: np.array, crop_shape: Sequence, **kwargs):
    """Crops or pads array.  Output will have dimensions "crop_shape". No interpolation. Padding type
    can be customized with **kwargs, like "reflect" to get mirror pad.

    Args:
        psf (np.array): N-dim array
        crop_shape (Sequence): desired output dimensions
        **kwargs: arguments to pass to np.pad

    Returns:
        N-dim array with desired output shape
    """
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
        return np.pad(psf[tuple(slicer)], pad, **kwargs)


@profile
def resize(
    vol,
    voxel_size: Sequence,
    sample_voxel_size: Sequence = (.1, .1, .1),
    minimum_shape: tuple = (64, 64, 64),
    debug: Any = None
):
    """ Up/down-scales volume to output voxel size using 3rd order interpolation. 
    Output volume is padded if array has fewer voxels than "minimum_shape". 

    Args:
        vol (_type_): 3D volume
        voxel_size (3 element Sequence): Output voxel size
        sample_voxel_size (3 element Sequence, optional): Input voxel size. Defaults to (.1, .1, .1).
        minimum_shape (tuple, optional): Pad array if vol (after resizing) is too small. Defaults to (64, 64, 64).
        debug : "True" to show figure, "not None" will write {debug}_rescaling.svg file. Defaults to None.
    """
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

    mode = np.abs(st.mode(resampled_vol, axis=None).mode[0])
    resized_vol = resize_with_crop_or_pad(
        resampled_vol,
        crop_shape=[s if s >= m else m for s, m in zip(resampled_vol.shape, minimum_shape)],
        constant_values=mode
    )

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
            plt.savefig(f'{debug}_rescaling.svg', dpi=300, bbox_inches='tight', pad_inches=.25)

    return resized_vol


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
    model_voxel_size: tuple,
    debug: Any = None,
    remove_background: bool = True,
    normalize: bool = True,
    edge_filter: bool = False,
    filter_mask_dilation: bool = True
):
    """ Input 3D array (or series of 3D arrays) is preprocessed in this order:

        -Background subtraction
        -Normalization to 0-1
        -Resample to model_voxel_size

    Args:
        sample (np.array): Input 3D array (or series of 3D arrays)
        sample_voxel_size (tuple): 
        model_voxel_size (tuple): 
        debug (Any, optional): plot or save .svg's. Defaults to None.
        remove_background (bool, optional): Defaults to True.
        normalize (bool, optional): Defaults to True.
        edge_filter (bool, optional): Defaults to True.

    Returns:
        _type_: 3D array (or series of 3D arrays)
    """
    sample = np.nan_to_num(sample, nan=0, posinf=0, neginf=0)

    if debug is not None:
        debug = Path(debug)
        if debug.is_dir():
            debug.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 3, figsize=(9, 6))

        axes[0, 1].set_title(f"{str(sample.shape)} @ {sample_voxel_size}")
        axes[0, 0].set_ylabel('Input (MIP)')
        m = axes[0, 0].imshow(np.max(sample, axis=0), cmap='hot')
        axes[0, 1].imshow(np.max(sample, axis=1), cmap='hot')
        axes[0, 2].imshow(np.max(sample, axis=2), cmap='hot')
        cax = inset_axes(axes[0, 2], width="10%", height="100%", loc='center right', borderpad=-2)
        cb = plt.colorbar(m, cax=cax)
        cax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    if remove_background:
        sample = remove_background_noise(sample)

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

    if not all(s1 == s2 for s1, s2 in zip(sample_voxel_size, model_voxel_size)):
        sample = transform.rescale(
            sample,
            (
                sample_voxel_size[0] / model_voxel_size[0],
                sample_voxel_size[1] / model_voxel_size[1],
                sample_voxel_size[2] / model_voxel_size[2],
            ),
            order=3,
            anti_aliasing=True,
        )
        sample = np.nan_to_num(sample, nan=0, posinf=0, neginf=0)

    if debug is not None:
        axes[1, 1].set_title(f"{str(sample.shape)} @ {model_voxel_size}")
        axes[1, 0].set_ylabel('Processed (MIP)')
        m = axes[1, 0].imshow(np.max(sample, axis=0), cmap='hot')
        axes[1, 1].imshow(np.max(sample, axis=1), cmap='hot')
        axes[1, 2].imshow(np.max(sample, axis=2), cmap='hot')
        cax = inset_axes(axes[1, 2], width="10%", height="100%", loc='center right', borderpad=-2)
        cb = plt.colorbar(m, cax=cax)
        cax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        plt.savefig(f'{debug}_rescaling.svg', dpi=300, bbox_inches='tight', pad_inches=.25)

    return sample


@profile
def find_roi(
    path: Union[Path, np.array],
    window_size: tuple = (64, 64, 64),
    plot: Any = None,
    num_rois: Any = None,
    min_dist: Any = 1,
    max_dist: Any = None,
    min_intensity: Any = 100,
    pois: Any = None,
    max_neighbor: int = 5,
    voxel_size: tuple = (.200, .108, .108),
    savepath: Any = None,
    timestamp: int = 17
):

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

    logger.info(f"Predicted points of interest")
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
    logger.info(f"Locating ROIs: {[pois.shape[0]]}")
    for p in range(pois.shape[0]):
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
            rois.append(r)

            if savepath is not None:
                savepath.mkdir(parents=True, exist_ok=True)
                imsave(savepath / f"roi_{p:02}.tif", r)

    return np.array(rois)


@profile
def get_tiles(
    path: Union[Path, np.array],
    window_size: tuple = (64, 64, 64),
    strides: int = 64,
    savepath: Any = None,
):
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

    logger.info(f"Tiling...")

    windows = sliding_window_view(dataset, window_shape=window_size)[::strides, ::strides, ::strides]
    zplanes, nrows, ncols = windows.shape[:3]
    windows = np.reshape(windows, (-1, *window_size))

    if savepath is not None:
        savepath.mkdir(parents=True, exist_ok=True)
        i = 0
        for z in range(zplanes):
            for y in range(nrows):
                for x in range(ncols):
                    imsave(savepath/f"z{z}-y{y}-x{x}.tif", windows[i])
                    i += 1

    return windows, zplanes, nrows, ncols
