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
from scipy.ndimage import gaussian_filter
import pandas as pd
import seaborn as sns
import zarr
from tqdm.contrib import itertools
from tqdm import trange
from tifffile import imread, imwrite
from scipy.spatial import KDTree
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.patches as patches
from line_profiler_pycharm import profile
from skimage.filters import window
from tifffile import TiffFile
from astropy import convolution
from skimage.feature import peak_local_max
from csbdeep.utils.tf import limit_gpu_memory
from skimage.transform import resize

limit_gpu_memory(allow_growth=True, fraction=None, total_memory=None)
from csbdeep.models import CARE

try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter
except ImportError as e:
    logging.warning(f"Cupy not supported on your system: {e}")

from vis import plot_mip, savesvg
from utils import round_to_even, gaussian_kernel

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@profile
def measure_noise(a: np.ndarray, axis: Optional[int] = None) -> np.float32:
    """ Return estimated noise (standard deviation) """
    noise = np.nanstd(a, axis=axis)
    return noise


@profile
def measure_snr(signal_img: np.ndarray,
                noise_img: Optional[np.ndarray] = None,
                axis: Optional[int] = None,
                ) -> int:
    """ Return estimated signal-to-noise ratio or inf if the given image has no noise """

    signal = np.nanmax(signal_img, axis=axis) - np.nanmedian(signal_img, axis=axis)
    noise = measure_noise(signal_img if noise_img is None else noise_img, axis=axis)
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
        # avoid z edges
        window_z = window(('tukey', pad_width[0]), padded.shape[0])**2
        # window_y = window(('tukey', pad_width[1]), padded.shape[1])
        # window_x = window(('tukey', pad_width[2]), padded.shape[2])
        # zv, yv, xv = np.meshgrid(window_z, window_y, window_x, indexing='ij', copy=True)
        if isinstance(padded, np.ndarray):
            pass
        else:
            window_z = cp.array(window_z)  # GPU array
        padded *= window_z[..., np.newaxis, np.newaxis]
        return padded


def resize_image(image, crop_shape: Union[tuple, list], interpolate: bool = False):
    if np.iscomplexobj(image):
        if interpolate:
            real = resize(np.real(image), output_shape=crop_shape, anti_aliasing=False, order=1, preserve_range=True)
            imag = resize(np.imag(image), output_shape=crop_shape, anti_aliasing=False, order=1, preserve_range=True)
            return real + 1j * imag
        else:  # only center crop
            real = resize_with_crop_or_pad(np.real(image), crop_shape=crop_shape, mode='constant')
            imag = resize_with_crop_or_pad(np.imag(image), crop_shape=crop_shape, mode='constant')
            return real + 1j * imag
    else:
        if interpolate:
            # factors =  tuple([np.round(image.shape[i]/crop_shape[i]).astype(int) for i in range(3)])
            # print(f"{factors=}, {crop_shape=}, {image.shape=}")
            # downscaled_image = downscale_local_mean(image, factors=factors).astype(np.float32)
            # downscaled_image = resize_with_crop_or_pad(downscaled_image, crop_shape=crop_shape, mode='constant')
            # return downscaled_image
            return resize(
                image.astype(np.float32),
                output_shape=crop_shape,
                anti_aliasing=False,
                order=1,
                preserve_range=True
            )
        else:
            return resize_with_crop_or_pad(
                image.astype(np.float32),
                crop_shape=crop_shape,
                mode='constant'
            ).astype(np.float32)


def na_and_background_filter(
    image: np.ndarray,
    na_mask: np.ndarray,  # light sheet NA mask
    low_sigma: float,   # unused
    high_sigma: Union[float] = None,  # high_sigma: Sets threshold for removing low frequencies (i.e. non-uniform bkgrd)
    mode: str = 'nearest',
    cval: int = 0,
    truncate: float = 4.0,
    min_psnr: int = 5,
):
    """
    Use the sample API as dog filter.
    Args:
        image:
        samplepsfgen:
        low_sigma:
        high_sigma:
        mode:
        cval:
        truncate:
        min_psnr:

    Returns:

    """
    spatial_dims = image.ndim

    dtype = np.float32
    low_sigma = np.array(low_sigma, dtype=dtype, ndmin=1)
    if high_sigma is None:
        high_sigma = low_sigma * 1.6
    else:
        high_sigma = np.array(high_sigma, dtype=dtype, ndmin=1)

    if len(high_sigma) != 1 and len(high_sigma) != spatial_dims:
        raise ValueError('high_sigma must have length equal to number of spatial dimensions of input')

    high_sigma = high_sigma * np.ones(spatial_dims, dtype=dtype)

    im2 = np.empty_like(image, dtype=dtype)  # need to supply gauss filter output with the data type we want
    if isinstance(image, np.ndarray):
        # CPU only.  Call scipy function
        im2 = gaussian_filter(image, high_sigma, mode=mode, cval=cval, truncate=truncate, output=im2)  # blurred
    else:
        im2 = cp_gaussian_filter(image, high_sigma, mode=mode, cval=cval, truncate=truncate, output=im2)  # blurred

    # fourier filter
    fourier = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(tukey_window(image, alpha=0.1))))
    fourier[na_mask == 0] = 0
    im1 = np.real(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(fourier))))  # needs to be 'real' not abs in this case

    return combine_filtered_imgs(image, im1, im2, min_psnr=min_psnr, dtype=dtype)
    # return im2


def combine_filtered_imgs(
    original_image: np.ndarray,
    im1_sharper: np.ndarray,
    im2_low_freqs_to_subtract: np.ndarray,
    min_psnr: int,
    dtype
) -> np.ndarray:
    """
    Combines the two filters (real space images) to produce filtered_img = (im1 - im2) - noise.
    Uses std_dev(im2) to determine if image is spare (we can increase background subtraction by noise) or dense

    Args:
        original_image: Raw data
        im1_sharper: Just highest frequencies
        im2_low_freqs_to_subtract: Just lowest frequencies (aka the non-uniform background to be subtracted)
        min_psnr:
        dtype:

    Returns:
        filtered_img : 3D cupy or numpy array

    """
    mask = tukey_window(np.ones_like(original_image, dtype=dtype))  # GPU or CPU array
    mask[mask < .9] = np.nan

    mask[mask >= .9] = 1
    noise_img = (original_image - im2_low_freqs_to_subtract) * mask
    # if blurred shows little std deviation: this is sparse, will want to more aggressively subtract
    if np.std(im2_low_freqs_to_subtract[mask == 1]) < 3:
        # estimate signal from processed (remove salt pepper noise), estimate noise from original image.
        filtered_img = im1_sharper - im2_low_freqs_to_subtract
        snr = measure_snr(filtered_img * mask, noise_img=noise_img)
        if snr > min_psnr:  # sparse, yet SNR of original image is good
            noise = np.std(original_image - im2_low_freqs_to_subtract)  # increase the bkgrd subtraction by the noise
            # logger.info(f"Sparse, yet SNR of original image ({snr}) is above {min_psnr}), increasing bkgrd subtraction by {noise}")
            return im1_sharper - (im2_low_freqs_to_subtract + noise)
        else:  # sparse, and SNR of original image is poor
            logger.warning(f"Dropping sparse image for poor SNR {snr} < {min_psnr}")
            return np.zeros_like(original_image)  # return zeros

    else:  # This is a dense image
        filtered_img = im1_sharper - im2_low_freqs_to_subtract
        snr = measure_snr(filtered_img * mask, noise_img=noise_img)
        if snr < min_psnr:  # SNR poor
            logger.warning(f"Dropping  dense image for poor SNR {snr} < {min_psnr}")
            return np.zeros_like(original_image)  # return zeros
        else:
            return filtered_img  # SNR good. Return filtered image.


def dog(
    image: np.ndarray,
    min_psnr: int,
    low_sigma: float,
    high_sigma: Union[float] = None,
    mode: str = 'nearest',
    cval: int = 0,
    truncate: float = 4.0,
):
    """
    Image is a cp.ndarray, processing is performed on GPU, otherwise CPU

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
        min_psnr: if SNR is poor, the image returned will be all zeros..

    Returns:
        filtered_image : ndarray
    """

    # CPU or GPU version
    spatial_dims = image.ndim

    np_dtype = np.float32

    low_sigma = np.array(low_sigma, dtype=np_dtype, ndmin=1)
    if high_sigma is None:
        high_sigma = low_sigma * 1.6
    else:
        high_sigma = np.array(high_sigma, dtype=np_dtype, ndmin=1)

    if len(low_sigma) != 1 and len(low_sigma) != spatial_dims:
        raise ValueError('low_sigma must have length equal to number of spatial dimensions of input')

    if len(high_sigma) != 1 and len(high_sigma) != spatial_dims:
        raise ValueError('high_sigma must have length equal to number of spatial dimensions of input')

    low_sigma = low_sigma * np.ones(spatial_dims, dtype=np_dtype)
    high_sigma = high_sigma * np.ones(spatial_dims, dtype=np_dtype)

    if any(high_sigma < low_sigma):
        raise ValueError('high_sigma must be equal to or larger than low_sigma for all axes')

    im1 = np.empty_like(image, dtype=np_dtype)  # need to supply gauss filter output with the data type we want
    im2 = np.empty_like(image, dtype=np_dtype)  # need to supply gauss filter output with the data type we want
    if isinstance(image, np.ndarray):
        # CPU only.  Call scipy function
        if all(low_sigma) > 0:
            im1 = gaussian_filter(image, low_sigma, mode=mode, cval=cval, truncate=truncate, output=im1)  # sharper
        else:
            im1 = np.zeros_like(image, dtype=np_dtype)
        im2 = gaussian_filter(image, high_sigma, mode=mode, cval=cval, truncate=truncate, output=im2)  # blurred
    else:
        # GPU only.  Call cupy function
        if all(low_sigma) > 0:
            im1 = cp_gaussian_filter(image, low_sigma, mode=mode, cval=cval, truncate=truncate, output=im1)  # sharper
        else:
            im1 = np.zeros_like(image, dtype=np_dtype)
        im2 = cp_gaussian_filter(image, high_sigma, mode=mode, cval=cval, truncate=truncate, output=im2)  # blurred

    return combine_filtered_imgs(image, im1, im2, min_psnr=min_psnr, dtype=np_dtype)


@profile
def remove_background_noise(
        image,
        read_noise_bias: float = 5,
        method: str = 'fourier_filter',  # 'difference_of_gaussians', fourier_filter
        high_sigma: float = 3.0,
        low_sigma: float = 0.7,
        min_psnr: int = 5,
        na_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """ Remove background noise from a given image volume.
        Difference of gaussians (DoG) works best via bandpass (reject past nyquist, reject
        non-uniform DC/background/scattering).
        Runs on GPU.
        Also checks if sparse, returns zeros if filtered image doesn't have enough signal.

    Args:
        na_mask: light sheet NA mask to filter to when using fourier filter
        image (np.ndarray or cp.ndarray): 3D image volume
        read_noise_bias (float, optional): When method="mode", empty pixels will still be non-zero due to read noise of camera.
            This value increases the amount subtracted to put empty pixels at zero. Defaults to 5.
        method (str, optional): method for background subtraction. Defaults to 'difference_of_gaussians'.
        high_sigma: Sets threshold for removing low frequencies (i.e. non-uniform bkgrd)
        low_sigma:  Sets threshold for removing high frequencies (i.e. beyond OTF support)
        min_psnr: Will blank image if filtered image does not meet this SNR minimum. min_psnr=0 disables this threshold

    Returns:
        _type_: cp.ndarray or np.ndarray
    """

    try:
        if not isinstance(image, cp.ndarray):
            image = cp.array(image)
    except Exception:
        logger.warning(f"No CUDA-capable device is detected. 'image' will be type {type(image)}")

    if method == 'mode':
        mode = st.mode(image, axis=None).mode
        mode = int(mode[0]) if isinstance(mode, (list, tuple, np.ndarray)) else int(mode)
        image -= mode + read_noise_bias

    elif method == 'difference_of_gaussians' or method == 'dog':
        image = dog(image, low_sigma=low_sigma, high_sigma=high_sigma, min_psnr=min_psnr)

    elif method == 'fourier_filter':
        if na_mask is None:
            raise ValueError("NA mask is None")

        if image.shape != na_mask.shape:
            na_mask = resize_with_crop_or_pad(na_mask, crop_shape=image.shape)

        image = na_and_background_filter(
            image,
            low_sigma=low_sigma,
            high_sigma=high_sigma,
            na_mask=na_mask,
            min_psnr=min_psnr
        )

    else:
        raise Exception(f"Unknown method '{method}' for remove_background_noise functions.")

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
    if isinstance(image, np.ndarray):
        pass
    else:
        w = cp.array(w)  # make this a GPU array.
    image *= w[np.newaxis, ...]   # apply 2D window over 3D volume as a cylinder
    return image


@profile
def prep_sample(
    sample: Union[np.array, Path],
    return_psnr: bool = False,
    normalize: bool = True,
    windowing: bool = True,
    sample_voxel_size: tuple = (.2, .097, .097),
    model_fov: Any = None,
    remove_background: bool = True,
    read_noise_bias: float = 5,
    plot: Any = None,
    min_psnr: int = 5,
    expand_dims: bool = True,
    na_mask: Optional[np.ndarray] = None,
    remove_background_noise_method: str = 'fourier_filter',  # 'fourier_filter' or 'difference_of_gaussians'
    denoiser: Optional[Union[Path, CARE]] = None,
    denoiser_window_size: tuple = (32, 64, 64),
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
        min_psnr: Will blank image if filtered image does not meet this SNR minimum. min_psnr=0 disables this threshold

    Returns:
        _type_: 3D array (or series of 3D arrays)
    """
    # remove_background_noise_method = 'difference_of_gaussians'
    sample_path = ''
    if isinstance(sample, Path):
        sample_path = sample.name
        with TiffFile(sample) as tif:
            sample = tif.asarray()

        if expand_dims:
            sample = np.expand_dims(sample, axis=-1)

    # convert to 32bit cupy if available
    try:
        sample = cp.array(sample, dtype=cp.float32)
    except:
        pass

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

        fig, axes = plt.subplots(3, ncols=3, figsize=(12, 12))

        plot_mip(
            vol=sample if isinstance(sample, np.ndarray) else cp.asnumpy(sample),
            xy=axes[0, 0],
            xz=axes[0, 1],
            yz=axes[0, 2],
            dxy=sample_voxel_size[-1],
            dz=sample_voxel_size[0],
            label=rf'Input (MIP) {sample.shape} [$\gamma$=.5]'
        )

        axes[0, 0].set_title(
            f'${int(sample_voxel_size[0]*1000)}^Z$, '
            f'${int(sample_voxel_size[1]*1000)}^Y$, '
            f'${int(sample_voxel_size[2]*1000)}^X$ (nm)'
        )
        axes[0, 1].set_title(f"PSNR: {measure_snr(sample)}")
    
    if denoiser is not None:
        sample = denoise_image(
            image=sample,
            denoiser=denoiser,
            denoiser_window_size=denoiser_window_size,
        )

    if remove_background:
        sample = remove_background_noise(
            sample,
            read_noise_bias=read_noise_bias,
            min_psnr=min_psnr,
            na_mask=na_mask,
            method=remove_background_noise_method
        )
    psnr = measure_snr(sample)

    # logger.info(f'{plot} plot min = {np.nanmin(sample)}. plot max = {np.nanmax(sample)}  plot 98th = {np.percentile(sample, 98)}  plot 5th = {np.percentile(sample, 5)}')
    if plot is not None:
        axes[1, 1].set_title(f"PSNR: {psnr}")
        background_subtraction_text = remove_background_noise_method
        background_subtraction_text = 'dog' if background_subtraction_text == 'difference_of_gaussians' else background_subtraction_text
        background_subtraction_text = 'Fourier filter' if background_subtraction_text == 'fourier_filter' else background_subtraction_text
        plot_mip(
            vol=sample if isinstance(sample, np.ndarray) else cp.asnumpy(sample),
            xy=axes[1, 0],
            xz=axes[1, 1],
            yz=axes[1, 2],
            dxy=sample_voxel_size[-1],
            dz=sample_voxel_size[0],
            label=f'{background_subtraction_text} '
                  r'[$\gamma$=.5]'
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
            vol=sample if isinstance(sample, np.ndarray) else cp.asnumpy(sample),
            xy=axes[-1, 0],
            xz=axes[-1, 1],
            yz=axes[-1, 2],
            dxy=sample_voxel_size[-1],
            dz=sample_voxel_size[0],
            label=rf'Processed, {sample.shape} [$\gamma$=.5]'
        )
        savesvg(fig, f'{plot}_preprocessing.svg')

    if return_psnr:
        logger.info(f"PSNR: {psnr:4}   {sample_path}")
        return psnr
    else:
        return sample.astype(np.float32) if isinstance(sample, np.ndarray) else cp.asnumpy(sample).astype(np.float32)


@profile
def find_roi(
    image: Union[Path, np.array],
    savepath: Path,
    window_size: tuple = (64, 64, 64),
    plot: Any = None,
    num_rois: Any = None,
    min_dist: Any = 1,
    max_dist: Any = None,
    min_intensity: Any = 100,
    max_neighbor: int = 50,
    voxel_size: tuple = (.200, .097, .097),
    kernel_size: int = 15,
    min_psnr: float = 10.0,
    zborder: int = 10,
    prep: Optional[partial] = None,
):
    savepath.mkdir(parents=True, exist_ok=True)
    savepath_unprocessed = Path(f"{savepath}_unprocessed")
    savepath_unprocessed.mkdir(parents=True, exist_ok=True)

    pd.set_option("display.precision", 2)
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })
    
    if isinstance(image, Path):
        image = imread(image).astype(np.float32)
    
    blurred_image = remove_background_noise(image, method='difference_of_gaussians', min_psnr=0)
    blurred_image = blurred_image if isinstance(blurred_image, np.ndarray) else cp.asnumpy(blurred_image)
    blurred_image = gaussian_filter(blurred_image, sigma=1.1)

    # exclude values close to the edge in Z for finding our template
    restricted_blurred = blurred_image.copy()
    restricted_blurred[0: zborder] = 0
    restricted_blurred[blurred_image.shape[0] - zborder:blurred_image.shape[0]] = 0
    max_poi = list(np.unravel_index(np.nanargmax(restricted_blurred, axis=None), restricted_blurred.shape))
    
    kernel = gaussian_kernel(kernlen=[kernel_size] * 3, std=1)
    # init_pos = [p - kernel_size // 2 for p in max_poi]
    # kernel = blurred_image[
    #     init_pos[0]:init_pos[0] + kernel_size,
    #     init_pos[1]:init_pos[1] + kernel_size,
    #     init_pos[2]:init_pos[2] + kernel_size,
    # ]
    
    # convolve template with the input image
    convolved_image = convolution.convolve_fft(
        blurred_image,
        kernel,
        allow_huge=True,
        boundary='fill',
        nan_treatment='fill',
        fill_value=0,
        normalize_kernel=False
    )
    convolved_image -= st.mode(convolved_image, axis=None)[0]
    convolved_image /= np.nanmax(convolved_image)

    pois = []
    detected_peaks = peak_local_max(
        convolved_image,
        min_distance=round(np.mean(window_size)/4),
        threshold_rel=.1,
        exclude_border=True,
        p_norm=2,
        num_peaks=num_rois
    ).astype(int)

    logger.info(f"Found {len(detected_peaks)} peaks from peak_local_max (limited to {num_rois})")
    candidates_map = np.zeros_like(image)
    if len(detected_peaks) == 0:
        p = max_poi
        intensity = image[p[0], p[1], p[2]]
        
        candidates_map[p[0], p[1], p[2]] = 1
        pois.append([p[0], p[1], p[2], intensity])
    
    elif len(detected_peaks) == 1:
        p = detected_peaks[0]
        intensity = image[p[0], p[1], p[2]]
        candidates_map[p[0], p[1], p[2]] = 1
        pois.append([p[0], p[1], p[2], intensity])
    
    else:
        for p in detected_peaks:
            peak_value = max(image[p[0], p[1], p[2]], blurred_image[p[0], p[1], p[2]])
            
            try:
                fov = convolved_image[
                      p[0] - (min_dist + 1):p[0] + (min_dist + 1),
                      p[1] - (min_dist + 1):p[1] + (min_dist + 1),
                      p[2] - (min_dist + 1):p[2] + (min_dist + 1),
                      ]

                if np.nanmax(fov) > convolved_image[p[0], p[1], p[2]]:
                    continue  # we are not at the summit if a max nearby is available.
                else:
                    candidates_map[p[0], p[1], p[2]] = peak_value
                    pois.append([p[0], p[1], p[2], peak_value])  # keep peak
            
            except Exception:
                # keep peak if we are at the border of the image
                candidates_map[p[0], p[1], p[2]] = peak_value
                pois.append([p[0], p[1], p[2], peak_value])

    pois = pd.DataFrame(pois, columns=['z', 'y', 'x', 'intensity'])
    # filter out points too close to the edge
    if len(pois) > 1:
        edge = 8
        lzedge = pois['z'] >= window_size[0]//edge
        hzedge = pois['z'] <= image.shape[0] - window_size[0] // edge
        lyedge = pois['y'] >= window_size[1]//edge
        hyedge = pois['y'] <= image.shape[1] - window_size[1] // edge
        lxedge = pois['x'] >= window_size[2]//edge
        hxedge = pois['x'] <= image.shape[2] - window_size[2] // edge
        pois = pois[lzedge & hzedge & lyedge & hyedge & lxedge & hxedge]

    if len(detected_peaks) == 0:
        p = max_poi
        intensity = image[p[0], p[1], p[2]]

        candidates_map[p[0], p[1], p[2]] = 1
        pois.append([p[0], p[1], p[2], intensity])

    pois.sort_values(by='intensity', ascending=False, inplace=True)
    pois.reset_index(inplace=True)
    points = pois[['z', 'y', 'x']].values
    scaled_peaks = np.zeros_like(pois)
    scaled_peaks[:, 0] = points[:, 0] * voxel_size[0]
    scaled_peaks[:, 1] = points[:, 1] * voxel_size[1]
    scaled_peaks[:, 2] = points[:, 2] * voxel_size[2]

    if len(scaled_peaks) > 1:
        kd = KDTree(scaled_peaks)
        num_nearest = len(scaled_peaks)
        dist, idx = kd.query(scaled_peaks, k=num_nearest, workers=-1)

        for n in range(1, num_nearest):
            pois[f'dist_{n}'] = dist[:, n]
            pois[f'nn_ids_{n}'] = idx[:, n]

        neighbor_dists = pois.columns[pois.columns.str.startswith('dist_')].tolist()  # column names
        neighbor_ids = pois.columns[pois.columns.str.startswith('nn_ids_')].tolist()  # column names
        pois['winners'] = 1
        # print(pois)

        for index, row in pois.iterrows():
            if pois.loc[index, 'winners']:
                losers_ids = row[neighbor_ids].astype(int)[np.array(row[neighbor_dists] < min_dist)].values

                # if this POI has only one loser near it.
                if len(losers_ids) == 1 and losers_ids[0] > index:

                    threshold = np.mean(window_size) / 2
                    # if this POI is very close to that single other POI loser:
                    if len(row[neighbor_ids].astype(int)[np.array(row[neighbor_dists] < threshold)].values) > 0:
                        # shift this POI (at most 1/2 window_size) so that both are within the FOV.
                        merge_id = losers_ids[0]
                        new_x = round((row['x'] + pois['x'][merge_id])/2)
                        new_y = round((row['y'] + pois['y'][merge_id])/2)
                        new_z = round((row['z'] + pois['z'][merge_id])/2)

                        logger.info(f"Merging ROI {index} with ROI {merge_id}, threshold {threshold:.1f}, shifting by "
                                    f"{new_z - row['z']:3.0f}, "
                                    f"{new_y - row['y']:3.0f}, "
                                    f"{new_x - row['x']:3.0f} (Z,Y,X) pixels.")
                        pois['x'][index] = new_x
                        pois['y'][index] = new_y
                        pois['z'][index] = new_z
                losers_ids = losers_ids[losers_ids > index]  # only kill losers with less intensity than current row
                pois['winners'][losers_ids] = 0
        logger.info(f"After winner selection, {pois['winners'].sum()} winners remain.")
    else:
        pois['winners'] = 1
    print(pois)

    # if plot:
    #     fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    #     sns.scatterplot(ax=axes[0], x=pois['dist_1'], y=pois['intensity'], s=5, color="C0")
    #     sns.kdeplot(ax=axes[0], x=pois['dist_1'], y=pois['intensity'], levels=5, color="grey", linewidths=1)
    #     axes[0].set_ylabel('Intensity')
    #     axes[0].set_xlabel('Distance (microns)')
    #     axes[0].set_yscale('log')
    #     axes[0].set_ylim(10 ** 0, None)
    #     axes[0].set_xlim(0, None)
    #     axes[0].grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)
    #
    #     x = np.sort(pois['dist_1'])
    #     y = np.arange(len(x)) / float(len(x))
    #     axes[1].plot(x, y, color='dimgrey')
    #     axes[1].set_xlabel('Distance (microns)')
    #     axes[1].set_ylabel('CDF')
    #     axes[1].set_xlim(0, None)
    #     axes[1].grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)
    #
    #     sns.histplot(ax=axes[2], data=pois, x="dist_1", kde=True)
    #     axes[2].set_xlabel('Distance')
    #     axes[2].set_xlim(0, None)
    #     axes[2].grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)
    #     savesvg(fig, f'{plot}_detected_rois.svg')

    # if min_dist is not None:
    #     logger.info(f'{min_dist =} um')
    #     pois = pois[pois['dist_1'] >= min_dist]
    pois = pois[pois['winners'] == True]


    if max_dist is not None:
        logger.info(f'{max_dist =} um')
        pois = pois[pois['dist_1'] <= max_dist]

    if min_intensity is not None and min_intensity > 0:
        pois = pois[pois['intensity'] >= min_intensity]

    neighbors = pois.columns[pois.columns.str.startswith('dist_1')].tolist()
    min_dist = np.min(window_size)*np.min(voxel_size)
    pois['neighbors'] = pois[pois[neighbors] <= min_dist].count(axis=1)



    # pois = pois[pois['neighbors'] <= max_neighbor]

    # if plot:
    #     fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    #     sns.scatterplot(ax=axes[0], x=pois['dist_1'], y=pois['intensity'], s=5, color="C0")
    #     sns.kdeplot(ax=axes[0], x=pois['dist_1'], y=pois['intensity'], levels=5, color="grey", linewidths=1)
    #     axes[0].set_ylabel('Intensity')
    #     axes[0].set_xlabel('Distance')
    #     axes[0].set_xlim(0, None)
    #     axes[0].grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)
    #
    #     x = np.sort(pois['dist_1'])
    #     y = np.arange(len(x)) / float(len(x))
    #     axes[1].plot(x, y, color='dimgrey')
    #     axes[1].set_xlabel('Distance')
    #     axes[1].set_ylabel('CDF')
    #     axes[1].set_xlim(0, None)
    #     axes[1].grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)
    #
    #     sns.histplot(ax=axes[2], data=pois, x='dist_1', kde=True)
    #     axes[2].set_xlabel('Distance')
    #     axes[2].set_xlim(0, None)
    #     axes[2].grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)
    #     savesvg(fig, f'{plot}_selected_rois.svg')


    pois = pois.head(num_rois)
    pois.to_csv(f"{plot}_stats.csv")

    pois = pois[['z', 'y', 'x']].values[:num_rois]
    widths = [w // 2 for w in window_size]

    height_of_titles = 0.1
    height_of_plot = convolved_image.shape[1] + convolved_image.shape[0]
    height_ratios = [convolved_image.shape[1]/height_of_plot + height_of_titles, convolved_image.shape[0]/height_of_plot + height_of_titles]
    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharey=False, sharex=True, height_ratios=height_ratios)
        for ax, mip_directions in enumerate([0,1]):
            axes[ax].imshow(
                np.nanmax(convolved_image, axis=mip_directions),
                aspect='equal',
                cmap='Greys_r',
            )

            for p in range(pois.shape[0]):
                if ax == 0:
                    # axes[ax].plot(pois[p, 2], pois[p, 1], marker='.', ls='', color=f'C{p}')
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
                    # axes[ax].plot(pois[p, 2], pois[p, 0], marker='.', ls='', color=f'C{p}')
                    axes[ax].add_patch(patches.Rectangle(
                        xy=(pois[p, 2] - window_size[2] // 2, pois[p, 0] - window_size[0] // 2),
                        width=window_size[1],
                        height=window_size[2],
                        fill=None,
                        color=f'C{p}',
                        alpha=1
                    ))
                    axes[ax].set_title('XZ')
        fig.tight_layout()
        savesvg(fig, f'{plot}_mips.svg')
        logger.info(f'Saved {plot}_mips.svg')

    rois = []
    poi_map = np.zeros_like(image)
    # ztiles = np.ceil(np.array(image.shape[0]) / window_size[0]).astype(int)
    ztiles = 1
    zslab_size = image.shape[0] / ztiles
    ytiles = 1
    xtiles = np.ceil(len(pois) / ztiles).astype(int)
    xtiles_counter = {z: 0 for z in range(ztiles)}

    desc = f"Saving rois and (so slowly) plotting svgs: {pois.shape[0]}" if plot else f"Saving rois: {pois.shape[0]}"
    for p in trange(pois.shape[0], desc=desc, file=sys.stdout, unit='file'):
    
        if p < len(pois):
            start = [
                pois[p, s] - widths[s] if pois[p, s] >= widths[s] else 0
                for s in range(3)
            ]
            end = [
                pois[p, s] + widths[s] if pois[p, s] + widths[s] < image.shape[s] else image.shape[s]
                for s in range(3)
            ]
            r = image[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
            poi_map[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = np.full(r.shape, int(p))
            
            if r.size != 0:
                z = np.floor(pois[p, 0] / zslab_size).astype(int)
                logger.info(f'{zslab_size=}, poi located at z={pois[p, 0]}, zslab={z}')
                y = ytiles - 1
                x = xtiles_counter[z]
                xtiles_counter[z] += 1
                
                tile = f"z{z}-y{y}-x{x}"
                imwrite(savepath_unprocessed / f"{tile}.tif", r, compression='deflate', dtype=np.float32)

                if prep is not None:
                    r = prep(r, plot=savepath / f"{tile}" if plot else None)
                    # r = prep(r, plot=None)  # Can't plot. plotting broken.

                imwrite(savepath / f"{tile}.tif", r, compression='deflate', dtype=np.float32)
                rois.append(savepath / f"{tile}.tif")
    
    scaled_heatmap = (image - np.nanpercentile(image[image > 0], 1)) / \
                     (np.nanpercentile(image[image > 0], 99) - np.nanpercentile(image[image > 0], 1))
    scaled_heatmap = np.clip(scaled_heatmap, a_min=0, a_max=1)  # this helps see the volume data in _clusters.tif
    
    poi_colors = np.split(
        np.array(sns.color_palette('tab20', n_colors=(len(pois) * ztiles))) * 255,
        ztiles,
    )  # list of colors for each z tiles
    
    colormap = []
    for cc in poi_colors:  # for each z tile's colors
        colormap.extend([[0, 0, 0], *cc])  # append the same zero color at the front
    colormap = np.array(colormap)
    rgb_map = colormap[poi_map.astype(np.ubyte)] * scaled_heatmap[..., np.newaxis]
    imwrite(
        f'{plot}_selected_rois.tif',
        rgb_map.astype(np.ubyte),
        photometric='rgb',
        resolution=window_size[1:],
        metadata={'axes': 'ZYXS'},
        compression='deflate',
    )
    return np.array(sorted(rois)), ztiles, ytiles, xtiles


@profile
def get_tiles(
    path: Union[Path, np.array],
    savepath: Path,
    window_size: tuple = (64, 64, 64),
    strides: Optional[tuple] = None,
    save_files: bool = True,
    save_file_type: str = '.tif',
    prep: Optional[partial] = None,
    plot: bool = False
):
    savepath_unprocessed = Path(f"{savepath}_unprocessed")

    savepath.mkdir(parents=True, exist_ok=True)
    savepath_unprocessed.mkdir(parents=True, exist_ok=True)

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
        dataset = imread(path).astype(np.float32)
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

    tiles = {}
    for i, (z, y, x) in enumerate(itertools.product(
        range(ztiles), range(nrows), range(ncols),
        desc=f"Locating tiles: {[windows.shape[0]]}",
        bar_format='{l_bar}{bar}{r_bar} {elapsed_s:.1f}s elapsed',
        unit=' tile',
        file=sys.stdout
    )):
        name = f"z{z}-y{y}-x{x}"

        imwrite(savepath_unprocessed / f"{name}.tif", windows[i], compression='deflate', dtype=np.float32)

        if prep is not None:
            w = prep(windows[i],  plot=savepath / f"{name}" if plot else None)
        else:
            w = windows[i]

        if np.all(w == 0):
            tiles[name] = dict(
                path=savepath / f"{name}.tif",
                ignored=True,
            )
        else:
            if save_files:
                if save_file_type == '.npy':
                    np.save(savepath / f"{name}.npy", w)
                if save_file_type == '.npz':
                    np.savez_compressed(savepath / f"{name}.npz", w)
                else:
                    imwrite(savepath / f"{name}.tif", w, compression='deflate', dtype=np.float32)


            tiles[name] = dict(
                path=savepath / f"{name}{save_file_type}",
                ignored=False,
            )

    tiles = pd.DataFrame.from_dict(tiles, orient='index')
    return tiles, ztiles, nrows, ncols


def optimal_rolling_strides(model_psf_fov, sample_voxel_size, sample_shape, overlap_factor: float = 0.8):
    model_window_size = (
        round_to_even(model_psf_fov[0] / sample_voxel_size[0]),
        round_to_even(model_psf_fov[1] / sample_voxel_size[1]),
        round_to_even(model_psf_fov[2] / sample_voxel_size[2]),
    )  # number of sample voxels that make up a model psf.

    model_window_size = np.minimum(model_window_size, sample_shape)
    number_of_rois = np.ceil(sample_shape / (model_window_size * overlap_factor))
    strides = np.floor((sample_shape - model_window_size) / (number_of_rois - 1))
    idx = np.where(np.isnan(strides))[0]
    strides[idx] = model_window_size[idx]
    strides = strides.astype(int)

    min_strides = np.ceil(model_window_size * 0.66).astype(np.int32)
    # throwaway = sample_shape - ((np.array(number_of_rois) - 1) * strides + model_window_size)

    if any(strides < min_strides):  # if strides overlap too much with model window
        number_of_rois -= (strides < min_strides).astype(np.int32)    # choose one less roi and recompute
        strides = np.floor((sample_shape - model_window_size) / (number_of_rois - 1))
        idx = np.where(np.isnan(strides))[0]
        strides[idx] = model_window_size[idx]
        strides = strides.astype(int)

    if any(strides < min_strides):
        raise Exception(f'Your strides {strides} overlap too much. '
                        f'Make window size larger so strides are > 2/3 of Model window size {min_strides}')
    return strides


def denoise_image(
    image: np.ndarray,
    denoiser: Union[Path, CARE],
    denoiser_window_size: tuple = (32, 64, 64),
    batch_size: int = 96
):
    n_tiles = np.ceil(image.shape / (np.array(denoiser_window_size) * np.cbrt(batch_size))).astype(int)
    # batch_factor = max(np.floor(np.cbrt(np.prod(n_tiles, axis=None) / batch_size)).astype(int), 1)
    # n_tiles = np.ceil(n_tiles / float(batch_factor)).astype(int)
    
    if isinstance(denoiser, Path):
        logger.info(f"Loading denoiser model: {denoiser}")
        denoiser = CARE(config=None, name=denoiser.name, basedir=denoiser.parent)
        logger.info(f"{denoiser.name} loaded")
    
    elif isinstance(denoiser, CARE):
        logger.info(f"Denoising image {image.shape} [w/ {denoiser.name}]: {n_tiles=}, {denoiser_window_size=}")
    else:
        raise Exception(f"Unknown denoiser type: {denoiser}")

    logger.info(f"Denoising image {image.shape} [w/ {denoiser.name}]: {n_tiles=}, {denoiser_window_size=}")
    denoised = denoiser.predict(
        image.get() if isinstance(image, cp.ndarray) else image,
        axes='ZYX',
        n_tiles=n_tiles
    )
    denoised[denoised < 0.0] = 0.0
    
    if isinstance(image, np.ndarray):
        return denoised
    else:
        return cp.array(denoised)  # make this a GPU array.
