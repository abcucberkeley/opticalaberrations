import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.set_loglevel('error')

import logging
import sys
from pathlib import Path
from functools import partial
from typing import Any, Sequence, Union, Optional
import numpy as np
from scipy import stats as st
import pandas as pd
import seaborn as sns
import zarr
import h5py
import scipy.io
from tqdm.contrib import itertools
from tifffile import imread, imwrite
from scipy.spatial import KDTree
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.patches as patches
from line_profiler_pycharm import profile
from skimage.filters import window, difference_of_gaussians
from tifffile import TiffFile

try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter
except ImportError as e:
    logging.warning(f"Cupy not supported on your system: {e}")

from vis import plot_mip, savesvg

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
def measure_noise(a: np.ndarray, axis: Optional[int] = None) -> np.float:
    """ Return estimated noise """
    noise = np.nanstd(a, axis=axis)
    return noise


@profile
def measure_snr(a: np.ndarray, axis: Optional[int] = None) -> int:
    """ Return estimated signal-to-noise ratio or inf if the given image has no noise """

    signal = np.nanmax(a, axis=axis) - np.nanmedian(a, axis=axis)
    noise = measure_noise(a, axis=axis)
    return int(np.round(np.where(noise == 0, 0, signal/noise), 0))


@profile
def resize_with_crop_or_pad(img: np.array, crop_shape: Sequence, mode: str = 'reflect', **kwargs):
    """Crops or pads array.  Output will have dimensions "crop_shape". No interpolation. Padding type
    can be customized with **kwargs, like "reflect" to get mirror pad.

    Args:
        img (np.array): N-dim array
        crop_shape (Sequence): desired output dimensions
        mode: mode to use for padding
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
            padded = np.array([np.pad(s[slicer], pad, mode=mode, **kwargs) for s in np.squeeze(img)])[..., np.newaxis]
        else:
            padded = np.pad(np.squeeze(img)[slicer], pad, mode=mode, **kwargs)[np.newaxis, ..., np.newaxis]
    else:
        padded = np.pad(img[tuple(slicer)], pad, mode=mode, **kwargs)

    pad_width = 1 - np.array(img.shape) / np.array(crop_shape)
    pad_width = np.clip(pad_width*2, 0, 1)

    if all(pad_width == 0):
        return padded
    else:
        window_z = window(('tukey', pad_width[0]), padded.shape[0])**2
        # window_y = window(('tukey', pad_width[1]), padded.shape[1])
        # window_x = window(('tukey', pad_width[2]), padded.shape[2])
        # zv, yv, xv = np.meshgrid(window_z, window_y, window_x, indexing='ij', copy=True)
        if isinstance(padded, cp.ndarray):
            window_z = cp.array(window_z)
        padded *= window_z[..., np.newaxis, np.newaxis]
        return padded


def dog(
    image,
    low_sigma: float,
    high_sigma: Union[float] = None,
    mode: str = 'nearest',
    cval: int = 0,
    truncate: float = 4.0,
    snr_threshold: int = 10,
):
    """
    If image is a cp.ndarray, processing is performed on GPU, and snr_threshold
    is 

    Find features between ``low_sigma`` and ``high_sigma`` in size.
    This function uses the Difference of Gaussians method for applying
    band-pass filters to multi-dimensional arrays. The input array is
    blurred with two Gaussian kernels of differing sigmas to produce two
    intermediate, filtered images. The more-blurred image is then subtracted
    from the less-blurred image. The final output image will therefore have
    had high-frequency components attenuated by the smaller-sigma Gaussian, and
    low frequency components will have been removed due to their presence in
    the more-blurred intermediate.

    Args:
        image: ndarray
            Input array to filter.
        low_sigma: scalar or sequence of scalars
            Standard deviation(s) for the Gaussian kernel with the smaller sigmas
            across all axes. The standard deviations are given for each axis as a
            sequence, or as a single number, in which case the single number is
            used as the standard deviation value for all axes.
        high_sigma: scalar or sequence of scalars, optional (default is None)
            Standard deviation(s) for the Gaussian kernel with the larger sigmas
            across all axes. The standard deviations are given for each axis as a
            sequence, or as a single number, in which case the single number is
            used as the standard deviation value for all axes. If None is given
            (default), sigmas for all axes are calculated as 1.6 * low_sigma.
        mode: {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
            The ``mode`` parameter determines how the array borders are
            handled, where ``cval`` is the value when mode is equal to
            'constant'. Default is 'nearest'.
        cval: scalar, optional
            Value to fill past edges of input if ``mode`` is 'constant'. Default
            is 0.0
        truncate: float, optional (default is 4.0)
            Truncate the filter at this many standard deviations.

    Returns:
        filtered_image : ndarray
    """
    if isinstance(image, cp.ndarray):
        try:    # try on GPU
            spatial_dims = image.ndim

            cp_dtype = cp.float32
            low_sigma = cp.array(low_sigma, dtype=cp_dtype, ndmin=1)
            if high_sigma is None:
                high_sigma = low_sigma * 1.6
            else:
                high_sigma = cp.array(high_sigma, dtype=cp_dtype, ndmin=1)

            if len(low_sigma) != 1 and len(low_sigma) != spatial_dims:
                raise ValueError('low_sigma must have length equal to number of spatial dimensions of input')

            if len(high_sigma) != 1 and len(high_sigma) != spatial_dims:
                raise ValueError('high_sigma must have length equal to number of spatial dimensions of input')

            low_sigma = low_sigma * cp.ones(spatial_dims, dtype=cp_dtype)
            high_sigma = high_sigma * cp.ones(spatial_dims, dtype=cp_dtype)

            if any(high_sigma < low_sigma):
                raise ValueError('high_sigma must be equal to or larger than low_sigma for all axes')

            im1 = cp.empty_like(image, dtype=cp_dtype)  # need to supply gauss filter output with the data type we want
            im2 = cp.empty_like(image, dtype=cp_dtype)  # need to supply gauss filter output with the data type we want
            im1 = gaussian_filter(image, low_sigma, mode=mode, cval=cval, truncate=truncate, output=im1)   # sharper
            im2 = gaussian_filter(image, high_sigma, mode=mode, cval=cval, truncate=truncate, output=im2)  # blurred

            mask = tukey_window(cp.ones_like(image, dtype=cp_dtype))
            mask[mask < .9] = cp.nan
            mask[mask >= .9] = 1

            # if blurred shows little std deviation: this is sparse, will want to more aggressively subtract
            if cp.std(im2[mask == 1]) < 3:
                snr = measure_snr(image*mask)       
                if snr > snr_threshold:             # sparse, yet SNR of original image is good
                    noise = cp.std(image - im2)     # increase the background subtraction by the noise
                    return im1 - (im2 + noise)
                else:                               # sparse, and SNR of original image is poor
                    logger.warning("Dropping sparse image for poor SNR")
                    return np.zeros_like(image)     # return zeros
                
            else:                                               # This is a dense image
                filtered_img = im1 - im2
                if measure_snr(filtered_img) < snr_threshold:   # SNR poor
                    logger.warning("Dropping  dense image for poor SNR")
                    return np.zeros_like(image)                 # return zeros
                else:
                    return filtered_img                         # SNR good. Return filtered image.

        except ImportError:
            return difference_of_gaussians(image, low_sigma=0.7, high_sigma=1.5)
    else:   # try on CPU
        return difference_of_gaussians(image, low_sigma=0.7, high_sigma=1.5)


@profile
def remove_background_noise(
        image,
        read_noise_bias: float = 5,
        method: str = 'difference_of_gaussians',
):
    """ Remove background noise from a given image volume.
        Difference of gaussians (DoG) works best via bandpass (reject past nyquist, reject non-uniform DC/background/scattering). Runs
        on GPU. DoG filter also checks if sparse, returns zeros if image doesn't have signal.

    Args:
        image (np.ndarray or cp.ndarray): 3D image volume
        read_noise_bias (float, optional): When method="mode", empty pixels will still be non-zero due to read noise of camera.
            This value increases the amount subtracted to put empty pixels at zero. Defaults to 5.
        method (str, optional): method for background subtraction. Defaults to 'difference_of_gaussians'.

    Returns:
        _type_: np.array
    """

    try:
        if not isinstance(image, cp.ndarray):
            image = cp.array(image)
    except Exception:
        logger.warning("No CUDA-capable device is detected")

    if method == 'mode':
        mode = int(st.mode(image, axis=None).mode[0])
        image -= mode + read_noise_bias
    else:
        image = dog(image, low_sigma=0.7, high_sigma=1.5)
    image[image < 0] = 0

    return image


def tukey_window(image: np.ndarray, alpha: float = .5):
    """
        To avoid boundary effects (cross in OTF), a tukey window is applied so that the edges of the volume go to zero.
        Args:
            image: input image
            alpha: the fraction of the window inside the cosine tapered region
                1.0 = Hann, 0.0 = rect window
    """

    # Nominally we would use a 3D window, but that would create a spherical mask inscribing the volume bounds.
    # That will cut a lot of data, we can do better by using cylindrical mask, because we don't use
    # an XZ or YZ projection of the FFT.
    w = window(('tukey', alpha), image.shape[1:])  # 2D window (basically an inscribed circle in X, Y)
    if isinstance(image, cp.ndarray):
        w = cp.array(w)
    image *= w[np.newaxis, ...]   # apply 2D window over 3D volume as a cylinder
    return image


@profile
def prep_sample(
    sample: Union[np.array, Path],
    return_psnr: bool = False,
    normalize: bool = True,
    windowing: bool = True,
    sample_voxel_size: tuple = (.2, .108, .108),
    model_fov: Any = None,
    remove_background: bool = True,
    read_noise_bias: float = 5,
    plot: Any = None,
):
    """ Input 3D array (or series of 3D arrays) is preprocessed in this order:
        
        (Converts to 32bit cupy float)
        
        -Background subtraction (via Difference of gaussians)
        -Crop (or mirror pad) to iPSF FOV's size in microns (if model FOV is given)
        -Tukey window
        -Normalization to 0-1
        
        Return 32bit numpy float

    Args:
        sample: Input 3D array (or series of 3D arrays)
        return_psnr: return estimated psnr instead of the image
        sample_voxel_size: voxel size for the given input image (z, y, x)
        model_fov: optional sample range to match what the model was trained on
        plot: plot or save .svg's
        remove_background: subtract background.
        normalize: scale values between 0 and 1.
        read_noise_bias: bias offset for camera noise
        windowing: optional toggle to apply to mask the input with a window to avoid boundary effects
    Returns:
        _type_: 3D array (or series of 3D arrays)
    """
    sample_path = ''
    if isinstance(sample, Path):
        sample_path = sample.name
        with TiffFile(sample) as tif:
            sample = tif.asarray()

        sample = np.expand_dims(sample, axis=-1)

    # convert to 32bit cupy
    if not isinstance(sample, cp.ndarray):
        sample = cp.array(sample, dtype=cp.float32)

    if plot is not None:
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
        })
        plot = Path(plot)
        if plot.is_dir(): plot.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(3, ncols=3, figsize=(10, 10))

        plot_mip(
            vol=cp.asnumpy(sample),
            xy=axes[0, 0],
            xz=axes[0, 1],
            yz=axes[0, 2],
            dxy=sample_voxel_size[-1],
            dz=sample_voxel_size[0],
            label='Input (MIP) [$\gamma$=.5]'
        )

        axes[0, 0].set_title(
            f'${int(sample_voxel_size[0]*1000)}^Z$, '
            f'${int(sample_voxel_size[1]*1000)}^Y$, '
            f'${int(sample_voxel_size[2]*1000)}^X$ (nm)'
        )
        axes[0, 1].set_title(f"PSNR: {measure_snr(sample)}")

    if remove_background:
        sample = remove_background_noise(sample, read_noise_bias=read_noise_bias)
        psnr = measure_snr(sample)
    else:
        psnr = measure_snr(sample)

    if plot is not None:
        axes[1, 1].set_title(f"PSNR: {psnr}")

        plot_mip(
            vol=cp.asnumpy(sample),
            xy=axes[1, 0],
            xz=axes[1, 1],
            yz=axes[1, 2],
            dxy=sample_voxel_size[-1],
            dz=sample_voxel_size[0],
            label='DoG [$\gamma$=.5]'
        )

    if model_fov is not None:
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

    if windowing:
        sample = tukey_window(sample)

    if normalize:  # safe division to not get nans for blank images
        denominator = np.max(sample)
        if denominator != 0:
            sample /= denominator

    if plot is not None:
        plot_mip(
            vol=cp.asnumpy(sample),
            xy=axes[-1, 0],
            xz=axes[-1, 1],
            yz=axes[-1, 2],
            dxy=sample_voxel_size[-1],
            dz=sample_voxel_size[0],
            label='Processed [$\gamma$=.5]'
        )
        savesvg(fig, f'{plot}_preprocessing.svg')

    if return_psnr:
        logger.info(f"PSNR: {psnr:4}   {sample_path}")
        return psnr
    else:
        if isinstance(sample, cp.ndarray):
            return cp.asnumpy(sample).astype(np.float32)
        else:
            return sample.astype(np.float32)


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
        savesvg(fig, f'{plot}_detected_points.svg')

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
        savesvg(fig, f'{plot}_selected_points.svg')


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
        savesvg(fig, f'{plot}_mips.svg')

    rois = []
    ztiles = 1
    ncols = int(np.ceil(len(pois) / 5))
    nrows = int(np.ceil(len(pois) / ncols))

    for p, (z, y, x) in enumerate(itertools.product(
        range(ztiles), range(nrows), range(ncols),
        desc=f"Locating tiles: {[pois.shape[0]]}",
        file=sys.stdout
    )):
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
            imwrite(savepath / f"{tile}.tif", r)
            rois.append(savepath / f"{tile}.tif")

    return np.array(rois), ztiles, nrows, ncols


@profile
def get_tiles(
    path: Union[Path, np.array],
    savepath: Path,
    window_size: tuple = (64, 64, 64),
    strides: Optional[tuple] = None,
    save_files: bool = True,
    save_file_type: str = '.tif',
    prep: Optional[partial] = None,
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

    if strides is None:
        strides = window_size

    windows = sliding_window_view(dataset, window_shape=window_size)[::strides[0], ::strides[1], ::strides[2]]
    ztiles, nrows, ncols = windows.shape[:3]
    windows = np.reshape(windows, (-1, *window_size))

    if prep is not None:
        from utils import multiprocess
        windows = multiprocess(
            jobs=windows,  # [cp.array(x.copy()) for x in windows]  # make copies not views into "dataset"
            func=prep,
            desc="Preprocessing tiles",
            cores=1,  # =1 because "windows" are views in "dataset", so multiprocess would make N copies of "dataset"
            unit='tiles',
        )

    tiles = {}
    for i, (z, y, x) in enumerate(itertools.product(
        range(ztiles), range(nrows), range(ncols),
        desc=f"Locating tiles: {[windows.shape[0]]}",
        bar_format='{l_bar}{bar}{r_bar} {elapsed_s:.1f}s elapsed',
        unit=' tile',
        file=sys.stdout
    )):
        name = f"z{z}-y{y}-x{x}"

        if np.all(windows[i] == 0):
            tiles[name] = dict(
                path=savepath / f"{name}.tif",
                ignored=True,
            )
        else:
            if save_files:
                if save_file_type == '.npy':
                    np.save(savepath / f"{name}.npy", windows[i])
                else:
                    imwrite(savepath / f"{name}.tif", windows[i])


            tiles[name] = dict(
                path=savepath / f"{name}{save_file_type}",
                ignored=False,
            )

    tiles = pd.DataFrame.from_dict(tiles, orient='index')
    return tiles, ztiles, nrows, ncols


def optimal_rolling_strides(model_psf_fov, sample_voxel_size, sample_shape):
    model_window_size = (
        round_to_even(model_psf_fov[0] / sample_voxel_size[0]),
        round_to_even(model_psf_fov[1] / sample_voxel_size[1]),
        round_to_even(model_psf_fov[2] / sample_voxel_size[2]),
    )  # number of sample voxels that make up a model psf.

    model_window_size = np.minimum(model_window_size, sample_shape)
    number_of_rois = np.ceil(sample_shape / model_window_size)
    strides = np.floor((sample_shape - model_window_size) / (number_of_rois - 1))
    idx = np.where(np.isnan(strides))[0]
    strides[idx] = model_window_size[idx]
    strides = strides.astype(int)

    min_strides = np.ceil(model_window_size * 0.66).astype(np.int32)
    # throwaway = sample_shape - ((np.array(number_of_rois) - 1) * strides + model_window_size)

    if any(strides < min_strides): # if strides overlap too much with model window
        number_of_rois -= (strides < min_strides).astype(np.int32)    # choose one less roi and recompute
        strides = np.floor((sample_shape - model_window_size) / (number_of_rois - 1))
        idx = np.where(np.isnan(strides))[0]
        strides[idx] = model_window_size[idx]
        strides = strides.astype(int)

    if any(strides < min_strides):
        raise Exception(f'Your strides {strides} overlap too much. '
                        f'Make window size larger so strides are > 2/3 of Model window size {min_strides}')
    return strides
