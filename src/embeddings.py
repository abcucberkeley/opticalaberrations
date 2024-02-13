import matplotlib
matplotlib.use('Agg')
import time
import logging
import sys
from typing import Any, Union, Optional
from functools import lru_cache

import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.set_loglevel('error')
from tifffile import imwrite

import numpy as np
from tqdm import tqdm
from functools import partial
from pathlib import Path
from skimage.filters import window
from skimage.restoration import unwrap_phase
from skimage.feature import peak_local_max
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import RegularGridInterpolator
from line_profiler_pycharm import profile
from scipy import ndimage
from astropy import convolution
from skspatial.objects import Plane, Points

try:
    import cupy as cp
    from cupyx.scipy.ndimage import rotate, map_coordinates
except ImportError as e:
    from scipy.ndimage import rotate, map_coordinates
    logging.warning(f"Cupy not supported on your system: {e}")

from preprocessing import resize_image, measure_snr, measure_noise
from utils import multiprocess, gaussian_kernel, fft, ifft, normalize_otf
from vis import savesvg, plot_interference, plot_embeddings

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@profile
def principle_planes(emb):
    return np.stack([
        emb[emb.shape[0] // 2, :, :],
        emb[:, emb.shape[1] // 2, :],
        emb[:, :, emb.shape[2] // 2],
    ], axis=0)


@profile
def spatial_planes(emb, max_zsupport: int=10):
    midplane = emb.shape[0] // 2
    return np.stack([
        emb[midplane, :, :],
        np.nanmean(emb[midplane:midplane + 5, :, :], axis=0),
        np.nanmean(emb[midplane + 5:midplane + max_zsupport, :, :], axis=0),
    ], axis=0)


@profile
def average_planes(emb):
    return np.stack([
        np.nanmean(emb, axis=0),
        np.nanmean(emb, axis=1),
        np.nanmean(emb, axis=2),
    ], axis=0)


@profile
def rotary_slices(emb):
    z = np.linspace(-1, 1, emb.shape[0])
    y = np.linspace(-1, 1, emb.shape[1])
    x = np.linspace(-1, 1, emb.shape[2])  # grid vectors with 0,0,0 at the center
    interp = RegularGridInterpolator((z, y, x), emb)

    # match desired X output size. Might want to go to k_max instead of 1 (the border)?
    rho = np.linspace(0, 1, emb.shape[0])

    # cut "rotary_slices" number of planes from 0 to 90 degrees.
    angles = np.linspace(0, np.pi / 2, 3)

    # meshgrid with each "slice" being a rotary cut, z being the vertical axis, and rho being the horizontal axis.
    hz = np.linspace(0, 1, emb.shape[0])
    m_angles, m_z, m_rho = np.meshgrid(angles, hz, rho, indexing='ij')

    m_x = m_rho * np.cos(m_angles)  # the slice coords in cartesian coordinates
    m_y = m_rho * np.sin(m_angles)

    return interp((m_z, m_y, m_x))  # 3D interpolate


@profile
def spatial_quadrants(emb):
    # get quadrants for each axis by doing two consecutive splits
    zquadrants = [q for half in np.split(emb, 2, axis=1) for q in np.split(half, 2, axis=2)]
    yquadrants = [q for half in np.split(emb, 2, axis=0) for q in np.split(half, 2, axis=2)]
    xquadrants = [q for half in np.split(emb, 2, axis=0) for q in np.split(half, 2, axis=1)]

    cz, cy, cx = [s // 2 for s in emb.shape]  # indices for the principle planes
    planes = [0, .1, .2, .3]  # offsets of the desired planes starting from the principle planes

    '''
        0: top-left quadrant
        1: top-right quadrant
        2: bottom-left quadrant
        3: bottom-right quadrant
    '''
    placement = [
        3, 2,  # first row
        1, 0,  # second row
    ]

    # figure out indices for the desired planes
    zplanes = [np.ceil(cz + (cz * p)).astype(int) for p in planes]
    xy = np.concatenate([  # combine vertical quadrants (columns)
        np.concatenate([  # combine horizontal quadrants (rows)
            zquadrants[placement[0]][zplanes[0], :, :],
            zquadrants[placement[1]][zplanes[1], :, :]
        ], axis=1),
        np.concatenate([  # combine horizontal quadrants (rows)
            zquadrants[placement[2]][zplanes[2], :, :],
            zquadrants[placement[3]][zplanes[3], :, :]
        ], axis=1)
    ], axis=0)

    yplanes = [int(cy + (cy * p)) for p in planes]
    xz = np.concatenate([
        np.concatenate([
            yquadrants[placement[0]][:, yplanes[0], :],
            yquadrants[placement[1]][:, yplanes[1], :]
        ], axis=1),
        np.concatenate([
            yquadrants[placement[2]][:, yplanes[2], :],
            yquadrants[placement[3]][:, yplanes[3], :]
        ], axis=1)
    ], axis=0)

    xplanes = [int(cx + (cx * p)) for p in planes]
    yz = np.concatenate([
        np.concatenate([
            xquadrants[placement[0]][:, :, xplanes[0]],
            xquadrants[placement[1]][:, :, xplanes[1]]
        ], axis=1),
        np.concatenate([
            xquadrants[placement[2]][:, :, xplanes[2]],
            xquadrants[placement[3]][:, :, xplanes[3]]
        ], axis=1)
    ], axis=0)

    return np.stack([xy, xz, yz], axis=0)


@profile
def remove_phase_ramp(masked_phase, plot):
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))

    for i in range(masked_phase.shape[0]):
        phase_slice = masked_phase[i].copy()
        x = y = np.arange(0, phase_slice.shape[0])
        X, Y = np.meshgrid(x, y)

        # ignores zero points
        points = [(x, y, v) for x, y, v in zip(X.ravel(), Y.ravel(), phase_slice.ravel()) if v != 0]
        points = Points(points)
        plane = Plane.best_fit(points)
        X, Y, Z = plane.to_mesh((x - plane.point[0]), y - plane.point[1])
        Z[phase_slice == 0] = 0.
        masked_phase[i] -= Z

        if plot is not None:
            axes[i, 0].set_title(f"phase_slice")
            axes[i, 0].imshow(phase_slice, vmin=-.5, vmax=.5, cmap='Spectral_r')
            axes[i, 0].axis('off')

            axes[i, 1].set_title(f"Z")
            axes[i, 1].imshow(Z, vmin=-.5, vmax=.5, cmap='Spectral_r')
            axes[i, 1].axis('off')

            axes[i, 2].set_title(f"emb[{i}] - Z")
            axes[i, 2].imshow(masked_phase[i], vmin=-.5, vmax=.5, cmap='Spectral_r')
            axes[i, 2].axis('off')
            savesvg(fig, f"{plot}_phase_ramp.svg")

    return np.nan_to_num(masked_phase, nan=0, neginf=0, posinf=0)


@profile
def remove_interference_pattern(
        psf,
        otf: Optional[np.ndarray] = None,
        plot: Optional[str] = None,
        pois: Optional[np.ndarray] = None,
        min_distance: int = 5,
        kernel_size: int = 15,
        max_num_peaks: int = 20,
        windowing: bool = True,
        window_size: tuple = (21, 21, 21),
        plot_interference_pattern: bool = False,
        min_psnr: float = 10.0,
        zborder: int = 10
):
    """
    Normalize interference pattern from the given FFT
    Args:
        psf: input image
        otf: FFT of the given input
        plot: a toggle for visualization
        pois: pre-defined mask of the exact bead locations
        min_distance: minimum distance for detecting adjacent beads
        kernel_size: size of the window for template matching
    """
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.autolimit_mode': 'round_numbers'
    })

    if otf is None:
        otf = fft(psf)

    blured_psf = ndimage.gaussian_filter(psf, sigma=1.1)

    # exclude values close to the edge in Z for finding our template
    restricted_blurred = blured_psf.copy()
    restricted_blurred[0: zborder] = 0
    restricted_blurred[blured_psf.shape[0]-zborder:blured_psf.shape[0]] = 0

    # get max pixel in the restricted_blurred image
    max_poi = list(np.unravel_index(np.nanargmax(restricted_blurred, axis=None), restricted_blurred.shape))

    # crop a window around the object for template matching
    template_poi = max_poi.copy()
    half_length = kernel_size // 2
    template_poi[0] = np.clip(template_poi[0], a_min=half_length, a_max=(psf.shape[0] - half_length) - 1)
    template_poi[1] = np.clip(template_poi[1], a_min=half_length, a_max=(psf.shape[1] - half_length) - 1)
    template_poi[2] = np.clip(template_poi[2], a_min=half_length, a_max=(psf.shape[2] - half_length) - 1)

    measured_snr = measure_snr(psf)
    high_snr = measured_snr > 30  # SNR good enough for template
    if high_snr:
        # logger.info(f'Using template. {measured_snr=} > 30')
        init_pos = [p-half_length for p in template_poi]
        kernel = blured_psf[
            init_pos[0]:init_pos[0]+kernel_size,
            init_pos[1]:init_pos[1]+kernel_size,
            init_pos[2]:init_pos[2]+kernel_size,
        ]
        effective_kernel_width = kernel_size // 2
    else:  # SNR isn't good enough for template, use a gaussian kernel
        # logger.info(f'Using gaussian kernel. {measured_snr=} <= 30')
        effective_kernel_width = 1
        kernel = gaussian_kernel(kernlen=[kernel_size]*3, std=effective_kernel_width)

    # convolve template with the input image
    convolved_psf = convolution.convolve_fft(
        blured_psf,
        kernel,
        allow_huge=True,
        boundary='fill',
        nan_treatment='fill',
        fill_value=0,
        normalize_kernel=False
    )
    convolved_psf -= np.nanmin(convolved_psf)
    convolved_psf /= np.nanmax(convolved_psf)

    if pois is None:
        # Bead detection
        pois = []
        detected_peaks = peak_local_max(
            convolved_psf,
            min_distance=min_distance,
            threshold_rel=.3,
            exclude_border=int(np.floor(effective_kernel_width)),
            p_norm=2,
            num_peaks=max_num_peaks
        ).astype(int)

        beads = np.zeros_like(psf)

        if len(detected_peaks) == 0 and high_snr:
            p = max_poi
            beads[p[0], p[1], p[2]] = 1
            pois.append(p)

        elif len(detected_peaks) == 1:
            p = detected_peaks[0]
            beads[p[0], p[1], p[2]] = 1
            pois.append(p)

        else:
            for p in detected_peaks:
                try:
                    fov = convolved_psf[
                        p[0]-(min_distance+1):p[0]+(min_distance+1),
                        p[1]-(min_distance+1):p[1]+(min_distance+1),
                        p[2]-(min_distance+1):p[2]+(min_distance+1),
                    ]
                    peak_value = max(psf[p[0], p[1], p[2]], blured_psf[p[0], p[1], p[2]])

                    if np.nanmax(fov) > convolved_psf[p[0], p[1], p[2]]:
                        continue    # we are not at the summit if a max nearby is available.
                    else:
                        beads[p[0], p[1], p[2]] = peak_value
                        pois.append(p)  # keep peak

                except Exception:
                    # keep peak if we are at the border of the image
                    beads[p[0], p[1], p[2]] = peak_value
                    pois.append(p)

        pois = np.array(pois)
    else:
        beads = pois.copy()
        beads[beads < .05] = 0.
        pois = np.array([[z, y, x] for z, y, x in zip(*np.nonzero(beads))])

    noise = measure_noise(psf)
    baseline = np.nanmedian(psf)
    good_psnr = np.zeros(pois.shape[0], dtype=bool)
    psnrs = np.zeros(pois.shape[0], dtype=float)
    for i, p in enumerate(pois):
        psnrs[i] = (np.nanmax(
            psf[
                max(0, p[0] - (min_distance + 1)):min(psf.shape[0], p[0] + (min_distance + 1)),
                max(0, p[1] - (min_distance + 1)):min(psf.shape[1], p[1] + (min_distance + 1)),
                max(0, p[2] - (min_distance + 1)):min(psf.shape[2], p[2] + (min_distance + 1)),
        ]) - baseline) / noise
        good_psnr[i] = psnrs[i] > min_psnr

    # logger.info(f"{len(pois)} objects detected. "
    #             f"Of these, {np.count_nonzero(good_psnr)} were above {min_psnr} min_psnr.  "
    #             f"Worst SNR = {np.min(psnrs).astype(int)}")
    pois = pois[good_psnr]  # remove points that are below peak snr

    psf_peaks = np.zeros_like(psf)  # create a volume masked around each peak, don't go past vol bounds
    for p in pois:
        psf_peaks[
            max(0, p[0] - (min_distance + 1)):min(psf.shape[0], p[0] + (min_distance + 1)),
            max(0, p[1] - (min_distance + 1)):min(psf.shape[1], p[1] + (min_distance + 1)),
            max(0, p[2] - (min_distance + 1)):min(psf.shape[2], p[2] + (min_distance + 1)),
        ] = psf[
            max(0, p[0] - (min_distance + 1)):min(psf.shape[0], p[0] + (min_distance + 1)),
            max(0, p[1] - (min_distance + 1)):min(psf.shape[1], p[1] + (min_distance + 1)),
            max(0, p[2] - (min_distance + 1)):min(psf.shape[2], p[2] + (min_distance + 1)),
        ]

    if pois.shape[0] > 0:  # found anything?
        interference_pattern = fft(beads)

        if np.all(beads == 0) or np.all(interference_pattern == 0):
            logger.error("Bad interference pattern")
            return otf

        corrected_otf = otf / interference_pattern

        # code to look at the realspace psf that goes into the phase emb
        # midplane = corrected_otf.shape[0] // 2
        # corrected_otf[:midplane - 20] = 0
        # corrected_otf[midplane-5:midplane+5] = 0
        # corrected_otf[midplane + 20:] = 0

        # # code to recenter the corrected_psf to remove phase ramps. (uses heavy gaussian blur)
        # corrected_psf = ifft(corrected_otf)
        # convolved_corrected_psf = ndimage.gaussian_filter(corrected_psf, sigma=7)
        # center_of_corrected_psf = np.unravel_index(np.argmax(convolved_corrected_psf, axis=None), corrected_psf.shape)
        # pixel_shift_for_corrected_psf = np.array(center_of_corrected_psf) - (np.array(corrected_psf.shape) // 2)
        # # If we put bead at the center of 64 cube, it will have a phase ramp, so don't do the next line.
        # corrected_otf *= pix_shift_to_phase_ramp(pixel_shift_for_corrected_psf, corrected_otf.shape)

        if windowing: # and not high_snr:
            window_border = np.floor((corrected_otf.shape - np.array(window_size)) // 2).astype(int)
            window_extent = corrected_otf.shape - window_border * 2

            # pad needs amount on both sides of each axis.
            window_border = np.vstack((window_border, window_border)).transpose()
            windowing_function = np.pad(window(('tukey', 0.8), window_extent), pad_width=window_border)

            corrected_psf = ifft(corrected_otf) * windowing_function
            corrected_otf = fft(corrected_psf)
        else:
            corrected_psf = ifft(corrected_otf)

        corrected_psf /= np.nanmax(corrected_psf)

        if np.all(corrected_psf == 0) or np.any(np.isnan(corrected_psf)):
            logger.error("Couldn't remove interference pattern")
            return otf

        if plot is not None:
            plot_interference(
                    plot,
                    plot_interference_pattern,
                    pois=pois,
                    min_distance=min_distance,
                    beads=beads,
                    convolved_psf=convolved_psf,
                    psf_peaks=psf_peaks,
                    corrected_psf=corrected_psf,
                    kernel=kernel,
                    interference_pattern=interference_pattern
                )
            imwrite(f'{plot}_corrected_psf.tif', data=corrected_psf.astype(np.float32), compression='deflate', dtype=np.float32)

        return corrected_otf
    else:
        # logger.warning("No objects were detected")

        if plot is not None:
            fig, axes = plt.subplots(
                nrows=5 if plot_interference_pattern else 4,
                ncols=3,
                figsize=(8, 11),
                sharey=False,
                sharex=False
            )
            transparency = 0.6
            marker_color = 'blue'
            for ax in range(3):
                if ax == 0:
                    axes[0, ax].plot(p[2], p[1], marker='x', ls='', color=marker_color)
                    axes[2, ax].plot(p[2], p[1], marker='x', ls='', color=marker_color, alpha=transparency)
                    axes[2, ax].add_patch(patches.Rectangle(
                        xy=(p[2] - min_distance, p[1] - min_distance),
                        width=min_distance * 2,
                        height=min_distance * 2,
                        fill=None,
                        color=marker_color,
                        alpha=transparency
                    ))
                elif ax == 1:
                    axes[0, ax].plot(p[2], p[0], marker='x', ls='', color=marker_color)
                    axes[2, ax].plot(p[2], p[0], marker='x', ls='', color=marker_color, alpha=transparency)
                    axes[2, ax].add_patch(patches.Rectangle(
                        xy=(p[2] - min_distance, p[0] - min_distance),
                        width=min_distance * 2,
                        height=min_distance * 2,
                        fill=None,
                        color=marker_color,
                        alpha=transparency
                    ))

                elif ax == 2:
                    axes[0, ax].plot(p[1], p[0], marker='x', ls='', color=marker_color)
                    axes[2, ax].plot(p[1], p[0], marker='x', ls='', color=marker_color, alpha=transparency)
                    axes[2, ax].add_patch(patches.Rectangle(
                        xy=(p[1] - min_distance, p[0] - min_distance),
                        width=min_distance * 2,
                        height=min_distance * 2,
                        fill=None,
                        color=marker_color,
                        alpha=transparency
                    ))
                m1 = axes[0, ax].imshow(np.nanmax(psf, axis=ax), cmap='hot', aspect='auto')
                m2 = axes[1, ax].imshow(np.nanmax(kernel, axis=ax), cmap='hot', aspect='auto')
                m3 = axes[2, ax].imshow(np.nanmax(convolved_psf, axis=ax), cmap='Greys_r', alpha=.66, aspect='auto')

            for ax, m, label in zip(
                    range(3),
                    [m1, m2, m3],
                    [f'Inputs (MIP)', 'Kernel', 'Peak detection\n(No objects were detected)']
            ):
                divider = make_axes_locatable(axes[ax, -1])
                cax = divider.append_axes("right", size="5%", pad=0.1)
                cb = plt.colorbar(m, cax=cax)
                cax.yaxis.set_label_position("right")
                cax.set_ylabel(label)

            for ax in axes.flatten():
                ax.axis('off')

            axes[0, 0].set_title('XY')
            axes[0, 1].set_title('XZ')
            axes[0, 2].set_title('YZ')
            savesvg(fig, f'{plot}_interference_pattern.svg')

        return np.zeros_like(otf)


def pix_shift_to_phase_ramp(pix_shift, array_shape):
    """

    Args:
        pix_shift (3 elements): Amount of real space pixels to shift the image by
        array_shape (3 elements): the image shape

    Returns:
        The complex phase factor (e^i*phi), a 3D phase ramp,
        that can multiply the 3D FFT by to get the desired shift.

    """
    zz, yy, xx = np.mgrid[
        0:array_shape[0],
        0:array_shape[1],
        0:array_shape[2],
    ]
    delta = 2 * np.pi * np.array(pix_shift) / np.array(array_shape)
    ramp3d = zz * delta[0] + yy * delta[1] + xx * delta[2]
    ramp3d -= np.mean(ramp3d)  # make ramp centered about zero.
    return np.exp(1j*ramp3d)



@profile
def compute_emb(
        otf: np.ndarray,
        iotf: np.ndarray,
        val: str,
        na_mask: Optional[np.ndarray] = None,
        ratio: bool = False,
        norm: bool = True,
        log10: bool = False,
        embedding_option: Any = 'spatial_planes',
        freq_strength_threshold: float = 0.,
    model_psf_shape: tuple = (64, 64, 64),
    interpolate_embeddings: bool = False
):
    """
    Gives the "lower dimension" representation of the data that will be shown to the model.
    We usually do val='abs' to compute the alpha embeddings, with "ratio=True"
    We usually do val='angle' to compute the phase embeddings, with "ratio=False"

    Args:
        otf: fft of the input data. Must have the same frequency spacing as iotf (which means the same real-space FOV in microns)
        iotf: ideal theoretical or empirical OTF (complex number)
        na_mask: theoretical NA support mask
        val: what to compute (either 'real', 'imag' imaginary, or phase 'angle' from the complex OTF)
        ratio: optional toggle to return ratio of data to ideal OTF
        norm: optional toggle to normalize the data [0, 1]
        log10: optional toggle to take log10 of the FFT
        freq_strength_threshold: threshold to filter out frequencies below given threshold (percentage to peak)
        model_psf_shape: shape for the planes of interests that the model was trained on (POIs)
        embedding_option: type of embedding to use
            (`principle_planes`,  'pp'): return principle planes only (middle planes)
            (`spatial_planes`,    'sp'): return
            (`average_planes`,    'ap'): return average of each axis
            (`rotary_slices`,     'rs'): return three radial slices
            (`spatial_quadrants`, 'sq'): return four different spatial planes in each quadrant
             or just return the full stack if nothing is passed
    Returns:
        embedding astype(np.float32)
    """
    if na_mask is None:
        na_mask = np.abs(iotf)
        threshold = np.nanpercentile(na_mask.flatten(), 65)
        na_mask = np.where(na_mask < threshold, na_mask, 1.)
        na_mask = np.where(na_mask >= threshold, na_mask, 0.).astype(bool)
    else:
        na_mask = na_mask.astype(bool)

    if val == 'real':
        emb = np.real(otf)

    elif val == 'imag':
        emb = np.imag(otf)

    elif val == 'angle':
        emb = np.angle(otf)
        emb = np.nan_to_num(emb, nan=0, neginf=0, posinf=0)
    
    elif val == 'abs':
        emb = np.abs(otf).astype(np.float32)

    else:
        raise Exception(f'invalid choice, {val=}, for compute_emb')
    
    if emb.shape != model_psf_shape:
        emb = resize_image(emb, crop_shape=model_psf_shape, interpolate=interpolate_embeddings)
    
    if norm:
        emb = normalize_otf(emb, freq_strength_threshold=freq_strength_threshold)

    if ratio:
        iotf = np.abs(iotf).astype(np.float32)
        
        if iotf.shape != emb.shape:
            iotf = resize_image(iotf, crop_shape=emb.shape, interpolate=interpolate_embeddings)
        
        emb /= iotf
        emb = np.nan_to_num(emb, nan=0, neginf=0, posinf=0)
    
    if na_mask.shape != emb.shape:
        na_mask = resize_image(na_mask, crop_shape=emb.shape, interpolate=interpolate_embeddings)
    
    if val == 'angle':
        try:
            # unwrap phase if we have at least 100 nonzero points left in emb.
            if len(np.ma.nonzero(emb)[0]) > 100:
                emb = np.ma.masked_array(emb, mask=~na_mask, fill_value=0)
                emb = unwrap_phase(emb)
                emb = emb.filled(0)
                emb = np.nan_to_num(emb, nan=0, neginf=0, posinf=0)
        except TimeoutError as e:
            logger.warning(f"`unwrap_phase`: {e}")

    emb *= na_mask

    if log10:
        emb = np.log10(emb)
        emb = np.nan_to_num(emb, nan=0, posinf=0, neginf=0)

    if embedding_option.lower() == 'principle_planes' or embedding_option.lower() == 'pp':
        return principle_planes(emb)

    elif embedding_option.lower() == 'spatial_planes' or embedding_option.lower() == 'sp':
        return spatial_planes(emb, max_zsupport=10)

    elif embedding_option.lower() == 'spatial_planes10' or embedding_option.lower() == 'sp':
        return spatial_planes(emb, max_zsupport=10)

    elif embedding_option.lower() == 'spatial_planes20' or embedding_option.lower() == 'sp':
        return spatial_planes(emb, max_zsupport=20)

    elif embedding_option.lower() == 'spatial_planes1020' or embedding_option.lower() == 'sp':
        return spatial_planes(emb, max_zsupport=20 if val == 'angle' else 10)

    elif embedding_option.lower() == 'average_planes' or embedding_option.lower() == 'ap':
        return average_planes(emb)

    elif embedding_option.lower() == 'rotary_slices' or embedding_option.lower() == 'rs':
        return rotary_slices(emb)

    elif embedding_option.lower() == 'spatial_quadrants' or embedding_option.lower() == 'sq':
        return spatial_quadrants(emb)

    else:
        logger.warning(f"embedding_option is unrecognized : {embedding_option}")
        return emb.astype(np.float32)


@lru_cache(typed=True)
def rotate_coords(
    shape: Union[tuple, np.ndarray],
    digital_rotations: int,
    axes: tuple = (-2, -1),
):
    """
        Calculates the coordinates to interpolate at, to achieve a range of rotations between 0 and 360,
         inclusive of zero and 360.  This set of coordinates can be fed into a map_coordinates function to do the interpolation.

         Done on GPU if cupy is available.

        Args:
            shape: (number of emb, height of emb, width of emb)
            digital_rotations: number of rotations
            axes: which axes are in the plane in which to perform the rotation. (-2, -1) will be clockwise rotations in XY plane.

        Returns:
            cp.array of 3 coords (z,y,x), 361 angles, 6 emb, height of emb, width of emb

        """
    t = time.time()
    gpu_support = 'cupy' in sys.modules
    dtype = np.float16
    all_coords_cache_path = Path(__file__).parent / f'all_coords_cache_3x{digital_rotations}x{shape[0]}x{shape[1]}x{shape[2]}_{dtype.__name__}.npy'

    if digital_rotations == 361 and axes == (-2, -1) and shape == (6, 64, 64) and all_coords_cache_path.exists():
        if gpu_support:
            all_coords = cp.load(all_coords_cache_path)
        else:
            all_coords = np.load(all_coords_cache_path)
        # elapsed_time = time.time() - t
        # print(f'----load rotate_coords---- {elapsed_time:0.2f} seconds')   # between 80ms and 550ms
    else:
        rotations = np.linspace(0, 360, digital_rotations)
        print(f'need to ----generating rotate_coords---- {all_coords_cache_path}')

        if gpu_support:  # rotated coordinates don't need many decimals
            coords = cp.array(
                cp.meshgrid(
                    cp.arange(shape[0]),
                    cp.arange(shape[1]) + .5, # when size is even: FFT is offcenter by 1, and we rotate about pixel seam.
                    cp.arange(shape[2]) + .5, # when size is even: FFT is offcenter by 1, and we rotate about pixel seam.
                    indexing='ij'
                )
            )
            all_coords = cp.zeros((3, digital_rotations, *shape), dtype=dtype)

            for i, angle in enumerate(rotations):
                all_coords[:, i] = rotate(
                    coords,
                    angle=angle,
                    reshape=False,
                    axes=axes,
                    output=dtype,  # output will be floats
                    prefilter=False,
                    order=1,
                ) if angle % 360 != 0 else coords # rot(zero degrees) != original coordinates because sin(0) !=0

        else:
            coords = np.array(
                np.meshgrid(
                    np.arange(shape[0]),
                    np.arange(shape[1]) + .5, # when size is even: FFT is offcenter by 1, and we rotate about pixel seam.
                    np.arange(shape[2]) + .5, # when size is even: FFT is offcenter by 1, and we rotate about pixel seam.
                    indexing='ij'
                )
            )
            all_coords = np.zeros((3, digital_rotations, *shape), dtype=dtype)

            for i, angle in enumerate(rotations):
                all_coords[:, i] = rotate(
                    coords,
                    angle=angle,
                    reshape=False,
                    axes=axes,
                    output=dtype,  # output will be floats
                    prefilter=False,
                    order=1,
                )

        # Again, rotation by zero degrees doesn't always become a "no operation". We need to enforce that between emb
        # dimension otherwise, we will mix between the six embeddings.
        for emb in range(shape[0]):
            all_coords[0, :, emb, :, :] = emb

        elapsed_time = time.time() - t
        print(f'----generating rotate_coords---- {elapsed_time:0.2f} seconds')  # ~1-4 seconds

        if not all_coords_cache_path.exists():
            np.save(all_coords_cache_path, all_coords)

    return all_coords   # 3 coords (z,y,x), 361 angles, 6 emb, height of emb, width of emb.  ~53 MB


@profile
def rotate_embeddings(
    emb: np.ndarray,
    digital_rotations: int = 361,
    plot: Any = None,
    debug_rotations: bool = False,
):
    gpu_support = 'cupy' in sys.modules
    coordinates = rotate_coords(shape=emb.shape, digital_rotations=digital_rotations)

    if gpu_support:
        emb = cp.asnumpy(
            map_coordinates(cp.array(emb),
                            coordinates=coordinates,
                            output=cp.float32,
                            order=1,
                            prefilter=False)
        )
    else:
        emb = map_coordinates(emb,
                              coordinates=coordinates,
                              output=np.float32,
                              order=1,
                              prefilter=False)

    if debug_rotations and plot:
        rotation_labels = np.linspace(0, 360, digital_rotations)
        emb_to_plot = 5         # which of the six emb to plot 0, 1, 2, 3, 4, or 5
        for i, angle in enumerate(tqdm(
                rotation_labels,
                desc=f"Generating digital rotations [{plot.name}]",
                file=sys.stdout,
        )):
            for plane in range(emb.shape[1]):
                imwrite(f'{plot}_rot{angle:05}.tif', emb[i, emb_to_plot, :, :].astype(np.float32), compression='deflate', dtype=np.float32)

    return emb


@profile
def fourier_embeddings(
    inputs: Union[np.array, tuple],
    iotf: np.array,
    na_mask: Optional[np.ndarray] = None,
    ratio: bool = True,
    norm: bool = True,
    padsize: Any = None,
    no_phase: bool = False,
    alpha_val: str = 'abs',
    phi_val: str = 'angle',
    plot: Any = None,
    log10: bool = False,
    input_coverage: float = 1.0,
    freq_strength_threshold: float = 0.01,
    pois: Any = None,
    remove_interference: bool = True,
    plot_interference: bool = False,  # because it's broken.
    embedding_option: str = 'spatial_planes',
    digital_rotations: Optional[int] = None,
    model_psf_shape: tuple = (64, 64, 64),
    debug_rotations: bool = False,
    interpolate_embeddings: bool = False,
):
    """
    Gives the "lower dimension" representation of the data that will be shown to the model.

    Args:
        inputs: 3D array.
        iotf: ideal theoretical or empirical OTF
        na_mask: theoretical NA support mask
        ratio: Returns ratio of data to ideal PSF,
            which helps put all the FFT voxels on a similar scale. Otherwise, straight values.
        norm: optional toggle to normalize the data [0, 1]
        padsize: pad the input to the desired size for the FFT
        no_phase: ignore/drop the phase component of the FFT
        alpha_val: use absolute values of the FFT `abs`, or the real portion `real`.
        phi_val: use the FFT phase in unwrapped radians `angle`, or the imaginary portion `imag`.
        plot: optional toggle to visualize embeddings
        remove_interference: a toggle to normalize out the interference pattern from the OTF
        pois: masked array of the pois of interest to compute the interference pattern between objects
        input_coverage: optional crop to the realspace image
        log10: optional toggle to take log10 of the FFT
        freq_strength_threshold: threshold to filter out frequencies below given threshold (percentage to peak)
        digital_rotations: optional digital rotations to the embeddings
        model_psf_shape: shape for the planes of interests that the model was trained on (POIs)
        embedding_option: type of embedding to use.
            Capitalizing on the radial symmetry of the FFT,
            we have a few options to minimize the size of the embedding.
        debug_rotations: Print out the rotations if true
        interpolate_embeddings: downsample/upsample embeddings to match model's input size
    """

    if isinstance(inputs, tuple):
        psf, otf = inputs
    else:
        psf = inputs
        otf = fft(inputs, padsize=padsize)

    if psf.ndim == 4:
        psf = np.squeeze(psf)
        otf = np.squeeze(otf)

    if input_coverage != 1.:
        psf = resize_image(psf, crop_shape=[int(s * input_coverage) for s in psf.shape])
    
    if np.all(psf == 0):
        if digital_rotations is not None:
            emb = np.zeros((digital_rotations, 3, *model_psf_shape[1:])) \
                if no_phase else np.zeros((digital_rotations, 6, *model_psf_shape[1:]))
        else:
            emb = np.zeros((3, *model_psf_shape[1:])) \
                if no_phase else np.zeros((6, *model_psf_shape[1:]))

    else:
        
        if no_phase:
            emb = compute_emb(
                otf,
                iotf,
                na_mask=na_mask,
                val=alpha_val,
                ratio=ratio,
                norm=norm,
                log10=log10,
                embedding_option=embedding_option,
                freq_strength_threshold=freq_strength_threshold,
                model_psf_shape=model_psf_shape,
                interpolate_embeddings=interpolate_embeddings
            )
        else:
            use_reconstructed_otf = False   # option to use reconstructed otf for alpha embedding
            if remove_interference and use_reconstructed_otf:
                otf = remove_interference_pattern(
                    psf,
                    otf,
                    plot=plot if plot_interference else None,
                    pois=pois,
                    windowing=True
                )

            alpha = compute_emb(
                otf,
                iotf,
                na_mask=na_mask,
                val=alpha_val,
                ratio=ratio,
                norm=norm,
                log10=log10,
                embedding_option=embedding_option,
                freq_strength_threshold=freq_strength_threshold,
                model_psf_shape=model_psf_shape,
                interpolate_embeddings=interpolate_embeddings
            )

            if remove_interference and not use_reconstructed_otf:
                otf = remove_interference_pattern(
                    psf,
                    otf,
                    plot=plot if plot_interference else None,
                    pois=pois,
                    windowing=True
                )

            phi = compute_emb(
                otf,
                iotf,
                na_mask=na_mask,
                val=phi_val,
                ratio=False,
                norm=False,
                log10=False,
                embedding_option='spatial_planes',
                freq_strength_threshold=freq_strength_threshold,
                model_psf_shape=model_psf_shape,
                interpolate_embeddings=interpolate_embeddings
            )

            emb = np.concatenate([alpha, phi], axis=0)

        if plot is not None:
            plt.style.use("default")
            plot_embeddings(
                inputs=psf,
                emb=emb,
                save_path=plot
            )

        if digital_rotations is not None:
            emb = rotate_embeddings(
                emb=emb,
                digital_rotations=digital_rotations,
                plot=plot,
                debug_rotations=debug_rotations
            )

    if emb.shape[-1] != 1:
        emb = np.expand_dims(emb, axis=-1)

    return emb


@profile
def rolling_fourier_embeddings(
        rois: np.array,
        iotf: np.array,
        na_mask: Optional[np.ndarray] = None,
        ratio: bool = True,
        norm: bool = True,
        no_phase: bool = False,
        alpha_val: str = 'abs',
        phi_val: str = 'angle',
        plot: Any = None,
        log10: bool = False,
        freq_strength_threshold: float = 0.01,
        embedding_option: str = 'spatial_planes',
        digital_rotations: Optional[int] = None,
        model_psf_shape: tuple = (64, 64, 64),
        debug_rotations: bool = False,
        remove_interference: bool = True,
        plot_interference: bool = False,
        cpu_workers: int = -1,
        nrows: Optional[int] = None,
        ncols: Optional[int] = None,
        ztiles: Optional[int] = None
):
    """
    Gives the "lower dimension" representation of the data that will be shown to the model.

    Args:
        rois: an array of 3D tiles to generate an average embedding
        iotf: ideal theoretical or empirical OTF
        na_mask: theoretical NA support mask
        ratio: Returns ratio of data to ideal PSF,
            which helps put all the FFT voxels on a similar scale. Otherwise, straight values.
        norm: optional toggle to normalize the data [0, 1]
        padsize: pad the input to the desired size for the FFT
        no_phase: ignore/drop the phase component of the FFT
        alpha_val: use absolute values of the FFT `abs`, or the real portion `real`.
        phi_val: use the FFT phase in unwrapped radians `angle`, or the imaginary portion `imag`.
        plot: optional toggle to visualize embeddings
        log10: optional toggle to take log10 of the FFT
        freq_strength_threshold: threshold to filter out frequencies below given threshold (percentage to peak)
        digital_rotations: optional digital rotations to the embeddings
        model_psf_shape: shape for the planes of interests (POIs)
        embedding_option: type of embedding to use.
            Capitalizing on the radial symmetry of the FFT,
            we have a few options to minimize the size of the embedding.
    """

    if np.all(rois == 0):
        if digital_rotations is not None:
            emb = np.zeros((digital_rotations, 3, *model_psf_shape[1:])) \
                if no_phase else np.zeros((digital_rotations, 6, *model_psf_shape[1:]))
        else:
            emb = np.zeros((3, *model_psf_shape[1:])) \
                if no_phase else np.zeros((6, *model_psf_shape[1:]))

    else:  # filter out blank images
        if plot is not None:
            original_rois = rois.copy() # save a copy to plot correct order of tiles
        rois = rois[[~np.all(r == 0) for r in rois]].astype(np.float32)

        otfs = multiprocess(
            func=fft,
            jobs=rois,
            cores=cpu_workers,
            desc='Compute FFTs'
        )
        avg_otf = np.nanmean(otfs, axis=0)
        
        if avg_otf.shape != model_psf_shape:
            avg_otf = resize_image(avg_otf, crop_shape=model_psf_shape)
        
        if iotf.shape != model_psf_shape:
            iotf = resize_image(iotf, crop_shape=model_psf_shape)
        
        if na_mask.shape != iotf.shape:
            na_mask = resize_image(na_mask.astype(np.float32), crop_shape=iotf.shape)
        
        if no_phase:
            emb = compute_emb(
                np.abs(avg_otf),
                iotf,
                na_mask=na_mask,
                val=alpha_val,
                ratio=ratio,
                norm=norm,
                log10=log10,
                embedding_option=embedding_option,
                freq_strength_threshold=freq_strength_threshold,
            )
        else:
            alpha = compute_emb(
                np.abs(avg_otf),
                iotf,
                na_mask=na_mask,
                val=alpha_val,
                ratio=ratio,
                norm=norm,
                log10=log10,
                embedding_option=embedding_option,
                freq_strength_threshold=freq_strength_threshold,
            )

            if remove_interference:
                window_size = (21, 21, 21)
                interference = partial(
                    remove_interference_pattern,
                    plot=None,
                    windowing=True,
                    window_size=window_size,
                )
                phi_otfs = multiprocess(
                    func=interference,
                    jobs=rois,          # remove interference using original real space data
                    cores=cpu_workers,
                    desc='Remove interference patterns'
                )

                # could also filter "No objects were detected" cases if remove_interference_pattern returned a flag for that.
                found_spots_in_tile = np.zeros(phi_otfs.shape[0], dtype=bool)
                for m in range(phi_otfs.shape[0]):
                    found_spots_in_tile[m] = not np.array_equal(phi_otfs[m], otfs[m], equal_nan=True)

                phi_otfs = phi_otfs[found_spots_in_tile]
                avg_otf = np.nanmean(phi_otfs, axis=0)
                avg_otf = resize_image(avg_otf, crop_shape=iotf.shape)

                if plot_interference:
                    gamma = 0.5
                    avg_psf = ifft(avg_otf)  # this will have ipsf voxel size (a different voxel size than sample).

                    fig, axes = plt.subplots(
                        nrows=4,
                        ncols=3,
                        figsize=(8, 11),
                        sharey=False,
                        sharex=False
                    )
                    for row in range(min(3, phi_otfs.shape[0])):
                        # this will have ipsf voxel size (a different voxel size than sample).
                        phi_psf = ifft(resize_image(phi_otfs[row], crop_shape=window_size))
                        for ax in range(3):
                            m5 = axes[row, ax].imshow(np.nanmax(phi_psf, axis=ax)**gamma, cmap='magma')

                        label = f'Reconstructed\nTile {row} of {phi_otfs.shape[0]}. $\gamma$={gamma}'

                        divider = make_axes_locatable(axes[row, -1])
                        cax = divider.append_axes("right", size="5%", pad=0.1)
                        cb = plt.colorbar(m5, cax=cax)
                        cax.yaxis.set_label_position("right")
                        cax.set_ylabel(label)

                    for ax in range(3):
                        m5 = axes[3, ax].imshow(np.nanmax(avg_psf, axis=ax)**gamma, cmap='magma')

                    label = f'Reconstructed\navg $\gamma$={gamma}'

                    divider = make_axes_locatable(axes[-1, -1])
                    cax = divider.append_axes("right", size="5%", pad=0.1)
                    cb = plt.colorbar(m5, cax=cax)
                    cax.yaxis.set_label_position("right")
                    cax.set_ylabel(label)

                    for ax in axes.flatten():
                        ax.axis('off')

                    axes[0, 0].set_title('XY')
                    axes[0, 1].set_title('XZ')
                    axes[0, 2].set_title('YZ')
                    savesvg(fig, f'{plot}_avg_interference_pattern.svg')

            phi = compute_emb(
                avg_otf,
                iotf,
                na_mask=na_mask,
                val=phi_val,
                ratio=False,
                norm=False,
                log10=False,
                embedding_option='spatial_planes',
                freq_strength_threshold=freq_strength_threshold,
            )

            emb = np.concatenate([alpha, phi], axis=0)

        if emb.shape[1:] != model_psf_shape[1:]:
            emb = resize_image(emb, crop_shape=(3 if no_phase else 6, *model_psf_shape[1:]), interpolate=True)

        if plot is not None:
            plt.style.use("default")
            plot_embeddings(
                inputs=original_rois,
                emb=emb,
                save_path=plot,
                nrows=nrows,
                ncols=ncols,
                ztiles=ztiles
            )

        if digital_rotations is not None:
            emb = rotate_embeddings(
                emb=emb,
                digital_rotations=digital_rotations,
                plot=plot,
                debug_rotations=debug_rotations
            )

    if emb.shape[-1] != 1:
        emb = np.expand_dims(emb, axis=-1)

    return emb


@profile
def measure_fourier_snr(
        a: np.ndarray,
        plot: Optional[Union[str, Path]],
        threshold: float = 1e-6,
        wavelength: float = .510,
        axial_voxel_size: float = .097,
        lateral_voxel_size: float = .097,
        psnr: int = 200,
) -> int:
    """ Return estimated signal-to-noise ratio or inf if the given image has no noise """

    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.autolimit_mode': 'round_numbers'
    })

    # na_mask = np.abs(iotf)
    # threshold = np.nanpercentile(na_mask.flatten(), 65)
    # na_mask = np.where(na_mask < threshold, na_mask, 1.)
    # na_mask = np.where(na_mask >= threshold, na_mask, 0.).astype(bool)

    from synthetic import SyntheticPSF
    samplepsfgen = SyntheticPSF(
        psf_type='../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat',
        psf_shape=a.shape,
        lam_detection=wavelength,
        x_voxel_size=lateral_voxel_size,
        y_voxel_size=lateral_voxel_size,
        z_voxel_size=axial_voxel_size
    )

    a /= np.nanmax(a)
    otf = np.abs(fft(a))
    otf /= np.nanmax(otf)
    otf[otf < threshold] = threshold

    spsf = samplepsfgen.single_psf(
        phi=0,
        normed=True,
        meta=False,
    )
    sotf = np.abs(fft(spsf))
    sotf /= np.nanmax(sotf)
    sotf[sotf < threshold] = threshold

    ipsf = samplepsfgen.single_psf(
        phi=0,
        normed=True,
        meta=False,
    )
    iotf = np.abs(fft(ipsf))
    iotf /= np.nanmax(iotf)
    iotf[iotf < threshold] = threshold

    mid_plane = [s//2 for s in otf.shape]

    if plot is not None:
        fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharex='col', sharey='row')

        kz = otf[:, mid_plane[1], mid_plane[2]]
        ky = otf[mid_plane[0], :, mid_plane[2]]
        kx = otf[mid_plane[0], mid_plane[1], :]

        skz = sotf[:, mid_plane[1], mid_plane[2]]
        sky = sotf[mid_plane[0], :, mid_plane[2]]
        skx = sotf[mid_plane[0], mid_plane[1], :]

        ikz = iotf[:, mid_plane[1], mid_plane[2]]
        iky = iotf[mid_plane[0], :, mid_plane[2]]
        ikx = iotf[mid_plane[0], mid_plane[1], :]

        axes[0, 0].plot(np.arange(ikz.size), ikz, color='grey', label=r'Theoretical ($|\mathscr{\hat{F}}|$)', zorder=0)
        axes[0, 0].plot(np.arange(kz.size), kz, color='C0', label='Observed')
        axes[0, 0].plot(np.arange(skz.size), skz, color='C1', label=f'Simulated (PSNR={psnr})')
        axes[1, 0].plot(np.arange(kz.size), kz/ikz, color='C0', label='Observed')
        axes[1, 0].plot(np.arange(skz.size), skz/ikz, color='C1', label=f'Simulated (PSNR={psnr})')
        axes[0, 0].set_xlim(0, kz.size)
        axes[0, 0].set_ylim(threshold, 1)

        axes[0, 1].plot(np.arange(iky.size), iky, color='grey', label=r'Theoretical ($|\mathscr{\hat{F}}|$)', zorder=0)
        axes[0, 1].plot(np.arange(ky.size), ky, color='C0', label='Observed')
        axes[0, 1].plot(np.arange(sky.size), sky, color='C1', label=f'Simulated (PSNR={psnr})')
        axes[1, 1].plot(np.arange(ky.size), ky/iky, color='C0', label='Observed')
        axes[1, 1].plot(np.arange(sky.size), sky/iky, color='C1', label=f'Simulated (PSNR={psnr})')
        axes[0, 1].set_xlim(0, ky.size)
        axes[0, 1].set_ylim(threshold, 1)

        axes[0, 2].plot(np.arange(ikx.size), ikx, color='grey', label=r'Theoretical ($|\mathscr{\hat{F}}|$)', zorder=0)
        axes[0, 2].plot(np.arange(kx.size), kx, color='C0', label='Observed')
        axes[0, 2].plot(np.arange(skx.size), skx, color='C1', label=f'Simulated (PSNR={psnr})')
        axes[1, 2].plot(np.arange(kx.size), kx/ikx, color='C0', label='Observed')
        axes[1, 2].plot(np.arange(skx.size), skx/ikx, color='C1', label=f'Simulated (PSNR={psnr})')
        axes[0, 2].set_xlim(0, kx.size)
        axes[0, 2].set_ylim(threshold, 1)

        for i in range(3):
            axes[0, i].set_yscale('log')
            axes[1, i].set_yscale('log')
            axes[0, i].spines.right.set_visible(False)
            axes[1, i].spines.right.set_visible(False)
            axes[0, i].spines.left.set_visible(False)
            axes[1, i].spines.left.set_visible(False)
            axes[0, i].spines.top.set_visible(False)
            axes[1, i].spines.top.set_visible(False)
            axes[1, i].set_xlabel('Frequency')
            axes[0, i].grid(True, which="both", axis='y', lw=1, ls='--', zorder=0, alpha=.5)
            axes[1, i].grid(True, which="both", axis='y', lw=1, ls='--', zorder=0, alpha=.5)

        axes[0, 0].set_ylabel(r'$|\mathscr{F}|$')
        axes[1, 0].set_ylabel(r'$|\mathscr{F}| / |\mathscr{\hat{F}}|$')
        axes[0, 0].set_title(r'$k$(XY)')
        axes[0, 1].set_title(r'$k$(XZ)')
        axes[0, 2].set_title(r'$k$(YZ)')
        axes[0, 0].legend(frameon=False, ncol=3, loc='upper left', bbox_to_anchor=(0, 1.3))

        savesvg(fig, savepath=plot)

