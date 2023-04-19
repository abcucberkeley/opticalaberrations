import matplotlib
matplotlib.use('Agg')

import logging
import sys
from typing import Any, Union, Optional

import matplotlib.pyplot as plt
plt.set_loglevel('error')

import numpy as np
from tqdm import tqdm
from functools import partial
from pathlib import Path
from skimage.filters import scharr, window
from skimage.restoration import unwrap_phase
from skimage.feature import peak_local_max
from skimage.transform import resize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import RegularGridInterpolator
from line_profiler_pycharm import profile
from scipy import ndimage
from astropy import convolution
from skspatial.objects import Plane, Points
from multiprocessing import Pool

try:
    import cupy as cp
    from cupyx.scipy.ndimage import rotate
except ImportError as e:
    from scipy.ndimage import rotate
    logging.warning(f"Cupy not supported on your system: {e}")

import preprocessing
from utils import resize_with_crop_or_pad, multiprocess
from vis import savesvg, plot_interference, plot_embeddings

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@profile
def fft(inputs, padsize=None):
    if padsize is not None:
        shape = inputs.shape[1]
        size = shape * (padsize / shape)
        pad = int((size - shape) // 2)
        inputs = np.pad(inputs, ((pad, pad), (pad, pad), (pad, pad)), 'constant', constant_values=0)

    otf = np.fft.ifftshift(inputs)
    otf = np.fft.fftn(otf)
    otf = np.fft.fftshift(otf)
    return otf


@profile
def ifft(otf):
    psf = np.fft.fftshift(otf)
    psf = np.fft.ifftn(psf)
    psf = np.abs(np.fft.ifftshift(psf))
    return psf


@profile
def normalize(emb, otf, freq_strength_threshold: float = 0.):
    emb /= np.nanpercentile(np.abs(otf), 99.99)
    emb[emb > 1] = 1
    emb[emb < -1] = -1
    emb = np.nan_to_num(emb, nan=0)

    if freq_strength_threshold != 0.:
        emb[np.abs(emb) < freq_strength_threshold] = 0.

    return emb


@profile
def principle_planes(emb):
    return np.stack([
        emb[emb.shape[0] // 2, :, :],
        emb[:, emb.shape[1] // 2, :],
        emb[:, :, emb.shape[2] // 2],
    ], axis=0)


@profile
def spatial_planes(emb):
    midplane = emb.shape[0] // 2
    return np.stack([
        emb[midplane, :, :],
        np.mean(emb[midplane:midplane + 5, :, :], axis=0),
        np.mean(emb[midplane + 5:midplane + 10, :, :], axis=0),
    ], axis=0)


@profile
def average_planes(emb):
    return np.stack([
        np.mean(emb, axis=0),
        np.mean(emb, axis=1),
        np.mean(emb, axis=2),
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

    return np.nan_to_num(masked_phase, nan=0)


def gaussian_kernel(kernlen: tuple = (21, 21, 21), std=3):
    """Returns a 2D Gaussian kernel array."""
    x = np.arange((-kernlen[2] // 2)+1, (-kernlen[2] // 2)+1 + kernlen[2], 1)
    y = np.arange((-kernlen[1] // 2)+1, (-kernlen[1] // 2)+1 + kernlen[1], 1)
    z = np.arange((-kernlen[0] // 2)+1, (-kernlen[0] // 2)+1 + kernlen[0], 1)
    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
    kernel = np.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * std ** 2))
    return kernel / np.sum(kernel)


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
        min_psnr: float = 15.0,
        async_plot: bool = True
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
    if otf is None: otf = fft(psf)
    blured_psf = ndimage.gaussian_filter(psf, sigma=1.1)

    # get max pixel in the image
    half_length = kernel_size // 2
    poi = list(np.unravel_index(np.argmax(blured_psf, axis=None), blured_psf.shape))

    # crop a window around the object for template matching
    poi[0] = np.clip(poi[0], a_min=half_length, a_max=(psf.shape[0] - half_length) - 1)
    poi[1] = np.clip(poi[1], a_min=half_length, a_max=(psf.shape[1] - half_length) - 1)
    poi[2] = np.clip(poi[2], a_min=half_length, a_max=(psf.shape[2] - half_length) - 1)
    init_pos = [p-half_length for p in poi]
    kernel = blured_psf[
        init_pos[0]:init_pos[0]+kernel_size,
        init_pos[1]:init_pos[1]+kernel_size,
        init_pos[2]:init_pos[2]+kernel_size,
    ]

    # convolve template with the input image
    effective_kernel_width = 1
    kernel = gaussian_kernel(kernlen=[kernel_size]*3, std=effective_kernel_width)
    convolved_psf = convolution.convolve_fft(blured_psf, kernel, allow_huge=True, boundary='fill')
    convolved_psf -= np.nanmin(convolved_psf)
    convolved_psf /= np.nanmax(convolved_psf)

    if pois is None:
        # Bead detection
        pois = []
        detected_peaks = peak_local_max(
            convolved_psf,
            min_distance=min_distance,
            threshold_rel=.3,
            threshold_abs=np.std(psf),
            exclude_border=int(np.floor(effective_kernel_width)),
            p_norm=2,
            num_peaks=max_num_peaks
        ).astype(int)

        beads = np.zeros_like(psf)
        for p in detected_peaks:
            try:
                fov = convolved_psf[
                    p[0]-(min_distance+1):p[0]+(min_distance+1),
                    p[1]-(min_distance+1):p[1]+(min_distance+1),
                    p[2]-(min_distance+1):p[2]+(min_distance+1),
                ]
                if np.max(fov) > convolved_psf[p[0], p[1], p[2]]:
                    continue    # we are not at the summit if a max nearby is available.
                else:
                    beads[p[0], p[1], p[2]] = psf[p[0], p[1], p[2]]
                    pois.append(p)  # keep peak

            except Exception:
                # keep peak if we are at the border of the image
                beads[p[0], p[1], p[2]] = psf[p[0], p[1], p[2]]
                pois.append(p)

        pois = np.array(pois)
    else:
        beads = pois.copy()
        beads[beads < .05] = 0.
        pois = np.array([[z, y, x] for z, y, x in zip(*np.nonzero(beads))])

    psf_peaks = np.zeros_like(psf)  # create a volume masked around each peak, don't go past vol bounds
    noise = preprocessing.measure_noise(psf)
    baseline = np.median(psf)
    good_psnr = np.zeros(pois.shape[0], dtype=bool)
    for i, p in enumerate(pois):
        good_psnr[i] = (np.max(
            psf[
            max(0, p[0] - (min_distance + 1)):min(psf.shape[0], p[0] + (min_distance + 1)),
            max(0, p[1] - (min_distance + 1)):min(psf.shape[1], p[1] + (min_distance + 1)),
            max(0, p[2] - (min_distance + 1)):min(psf.shape[2], p[2] + (min_distance + 1)),
            ]) - baseline) / noise > min_psnr

    #logger.info(f"{pois.shape[0]} objects detected. {np.count_nonzero(good_psnr)} were above {min_psnr} min_psnr")
    pois = pois[good_psnr]  # remove points that are below peak snr

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

    if pois.shape[0] > 0:
        interference_pattern = fft(beads)
        corrected_otf = otf / interference_pattern

        if windowing:
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

        if plot is not None:
            if async_plot:
                Pool(1).apply_async(plot_interference(
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
                ))
            else:
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

            for ax in range(3):
                m1 = axes[0, ax].imshow(np.nanmax(psf, axis=ax), cmap='hot')
                m2 = axes[1, ax].imshow(np.nanmax(kernel, axis=ax), cmap='hot')
                m3 = axes[2, ax].imshow(np.nanmax(convolved_psf, axis=ax), cmap='Greys_r', alpha=.66)

            for ax, m, label in zip(
                    range(3),
                    [m1, m2, m3],
                    [f'Inputs (MIP)', 'Kernel', 'Peak detection\n(No objects were detected)']
            ):
                cax = inset_axes(axes[ax, -1], width="10%", height="100%", loc='center right', borderpad=-3)
                cb = plt.colorbar(m, cax=cax)
                cax.yaxis.set_label_position("right")
                cax.set_ylabel(label)

            for ax in axes.flatten():
                ax.axis('off')

            axes[0, 0].set_title('XY')
            axes[0, 1].set_title('XZ')
            axes[0, 2].set_title('YZ')
            savesvg(fig, f'{plot}_interference_pattern.svg')

        return otf


@profile
def compute_emb(
        otf: np.ndarray,
        iotf: np.ndarray,
        val: str,
        ratio: bool = False,
        norm: bool = True,
        log10: bool = False,
        embedding_option: Any = 'spatial_planes',
        freq_strength_threshold: float = 0.,
):
    """
    Gives the "lower dimension" representation of the data that will be shown to the model.

    Args:
        otf: fft of the input data. Must have the same frequency spacing as iotf (which means the same real-space FOV in microns)
        iotf: ideal theoretical or empirical OTF
        val: what to compute (either 'real', 'imag' imaginary, or phase 'angle' from the complex OTF)
        ratio: optional toggle to return ratio of data to ideal OTF
        norm: optional toggle to normalize the data [0, 1]
        log10: optional toggle to take log10 of the FFT
        freq_strength_threshold: threshold to filter out frequencies below given threshold (percentage to peak)
        embedding_option: type of embedding to use
            (`principle_planes`,  'pp'): return principle planes only (middle planes)
            (`spatial_planes`,    'sp'): return
            (`average_planes`,    'ap'): return average of each axis
            (`rotary_slices`,     'rs'): return three radial slices
            (`spatial_quadrants`, 'sq'): return four different spatial planes in each quadrant
             or just return the full stack if nothing is passed
    """
    na_mask = np.abs(iotf)
    threshold = np.nanpercentile(na_mask.flatten(), 65)
    na_mask = np.where(na_mask < threshold, na_mask, 1.)
    na_mask = np.where(na_mask >= threshold, na_mask, 0.).astype(bool)

    if otf.shape != iotf.shape:
        real = resize_with_crop_or_pad(np.real(otf), crop_shape=iotf.shape)  # only center crop
        imag = resize_with_crop_or_pad(np.imag(otf), crop_shape=iotf.shape)  # only center crop
        otf = real + 1j * imag

    if val == 'real':
        emb = np.real(otf)

    elif val == 'imag':
        emb = np.imag(otf)

    elif val == 'angle':
        emb = otf / np.nanpercentile(np.abs(otf), 99.99)

        if freq_strength_threshold != 0.:
            emb[np.abs(emb) < freq_strength_threshold] = 0.
            na_mask[np.abs(emb) < freq_strength_threshold] = 0.

        emb = np.angle(emb)
        emb = np.nan_to_num(emb, nan=0, neginf=0, posinf=0)

        try:
            if len(np.ma.nonzero(emb)[0]) > 100:
                emb = np.ma.masked_array(emb, mask=~na_mask, fill_value=0)
                emb = unwrap_phase(emb)
                emb = emb.filled(0)
                emb = np.nan_to_num(emb, nan=0, neginf=0, posinf=0)
        except TimeoutError as e:
            logger.warning(f"`unwrap_phase`: {e}")

    else:
        emb = np.abs(otf)

    if norm:
        emb = normalize(emb, otf, freq_strength_threshold=freq_strength_threshold)

    if ratio:
        emb /= iotf
        emb = np.nan_to_num(emb, nan=0)

    emb *= na_mask

    if log10:
        emb = np.log10(emb)
        emb = np.nan_to_num(emb, nan=0, posinf=0, neginf=0)

    if embedding_option.lower() == 'principle_planes' or embedding_option.lower() == 'pp':
        return principle_planes(emb)

    elif embedding_option.lower() == 'spatial_planes' or embedding_option.lower() == 'sp':
        return spatial_planes(emb)

    elif embedding_option.lower() == 'average_planes' or embedding_option.lower() == 'ap':
        return average_planes(emb)

    elif embedding_option.lower() == 'rotary_slices' or embedding_option.lower() == 'rs':
        return rotary_slices(emb)

    elif embedding_option.lower() == 'spatial_quadrants' or embedding_option.lower() == 'sq':
        return spatial_quadrants(emb)

    else:
        logger.warning(f"embedding_option is unrecognized : {embedding_option}")
        return emb


def rotate_embeddings(
    emb: np.ndarray,
    digital_rotations: np.ndarray = np.linspace(0, 360, 360).astype(int),
    plot: Any = None,
    debug_rotations: bool = False
):
    gpu_support = 'cupy' in sys.modules

    if gpu_support:
        memarray = cp.array(emb)
    else:
        memarray = emb.copy()

    if debug_rotations:
        emb = np.zeros((digital_rotations.shape[0], *emb.shape))

        for i, angle in enumerate(tqdm(
                digital_rotations,
                desc=f"Generating digital rotations [{plot.name}]"
                if plot is not None else "Generating digital rotations",
        )):
            for plane in range(emb.shape[1]):
                r = rotate(memarray[plane], angle=angle, reshape=False, axes=(-2, -1))
                if gpu_support:
                    r = cp.asnumpy(r)
                emb[i, plane, :, :] = r

                fig = plt.figure()
                plt.imshow(emb[i, 0, :, :])
                savesvg(fig, f'{plot}_rot{angle}.svg')

    else:
        if gpu_support:
            emb = np.array([
                cp.asnumpy(rotate(memarray, angle=angle, reshape=False, axes=(-2, -1)))
                for angle in tqdm(
                    digital_rotations,
                    desc=f"Generating digital rotations [{plot.name}]"
                    if plot is not None else "Generating digital rotations",
                )
            ])
        else:
            emb = np.array([
                rotate(memarray, angle=angle, reshape=False, axes=(-2, -1))
                for angle in tqdm(
                    digital_rotations,
                    desc=f"Generating digital rotations [{plot.name}]"
                    if plot is not None else "Generating digital rotations",
                )
            ])

    return emb


@profile
def fourier_embeddings(
        inputs: Union[np.array, tuple],
        iotf: np.array,
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
        embedding_option: str = 'spatial_planes',
        edge_filter: bool = False,
        digital_rotations: Any = None,
        poi_shape: tuple = (64, 64),
        debug_rotations: bool = False,
        async_plot: bool = True
):
    """
    Gives the "lower dimension" representation of the data that will be shown to the model.

    Args:
        inputs: 3D array.
        iotf: ideal theoretical or empirical OTF
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
        edge_filter: a toggle for running an edge filter pass on the alpha embeddings
        digital_rotations: optional digital rotations to the embeddings
        poi_shape: shape for the planes of interests (POIs)
        embedding_option: type of embedding to use.
            Capitalizing on the radial symmetry of the FFT,
            we have a few options to minimize the size of the embedding.
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
        psf = resize_with_crop_or_pad(psf, crop_shape=[int(s * input_coverage) for s in psf.shape])

    if no_phase:
        emb = compute_emb(
            otf,
            iotf,
            val=alpha_val,
            ratio=ratio,
            norm=norm,
            log10=log10,
            embedding_option=embedding_option,
            freq_strength_threshold=freq_strength_threshold,
        )
        if edge_filter:
            emb = scharr(emb)
            emb /= np.nanpercentile(emb, 90)
            emb[emb > 1] = 1
    else:

        alpha = compute_emb(
            otf,
            iotf,
            val=alpha_val,
            ratio=ratio,
            norm=norm,
            log10=log10,
            embedding_option=embedding_option,
            freq_strength_threshold=freq_strength_threshold,
        )

        if edge_filter:
            alpha = scharr(alpha)
            alpha /= np.nanpercentile(alpha, 90)
            alpha[alpha > 1] = 1

        if remove_interference:
            otf = remove_interference_pattern(
                psf,
                otf,
                plot=plot,
                pois=pois,
                windowing=True,
                async_plot=async_plot
            )

        phi = compute_emb(
            otf,
            iotf,
            val=phi_val,
            ratio=False,
            norm=False,
            log10=False,
            embedding_option='spatial_planes',
            freq_strength_threshold=freq_strength_threshold,
        )

        emb = np.concatenate([alpha, phi], axis=0)

    if emb.shape[1:] != poi_shape:
        emb = resize(emb, output_shape=(3 if no_phase else 6, *poi_shape))
        # emb = resize_with_crop_or_pad(emb, crop_shape=(3 if no_phase else 6, *poi_shape))

    if plot is not None:
        plt.style.use("default")
        if async_plot:
            Pool(1).apply_async(plot_embeddings(
                inputs=psf,
                emb=emb,
                save_path=plot
            ))
        else:
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
        ratio: bool = True,
        norm: bool = True,
        no_phase: bool = False,
        alpha_val: str = 'abs',
        phi_val: str = 'angle',
        plot: Any = None,
        log10: bool = False,
        freq_strength_threshold: float = 0.01,
        embedding_option: str = 'spatial_planes',
        digital_rotations: Any = None,
        poi_shape: tuple = (64, 64),
        debug_rotations: bool = False,
        remove_interference: bool = True,
        cpu_workers: int = -1,
        nrows: Optional[int] = None,
        ncols: Optional[int] = None,
        ztiles: Optional[int] = None,
        async_plot: bool = True
):
    """
    Gives the "lower dimension" representation of the data that will be shown to the model.

    Args:
        rois: an array of 3D tiles to generate an average embedding
        iotf: ideal theoretical or empirical OTF
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
        poi_shape: shape for the planes of interests (POIs)
        embedding_option: type of embedding to use.
            Capitalizing on the radial symmetry of the FFT,
            we have a few options to minimize the size of the embedding.
    """

    otfs = multiprocess(
        func=fft,
        jobs=rois,
        cores=cpu_workers,
        desc='Compute FFTs'
    )
    avg_otf = resize_with_crop_or_pad(np.mean(otfs, axis=0), crop_shape=iotf.shape)

    if no_phase:
        emb = compute_emb(
            avg_otf,
            iotf,
            val=alpha_val,
            ratio=ratio,
            norm=norm,
            log10=log10,
            embedding_option=embedding_option,
            freq_strength_threshold=freq_strength_threshold,
        )
    else:
        alpha = compute_emb(
            avg_otf,
            iotf,
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

            avg_otf = resize_with_crop_or_pad(np.mean(phi_otfs, axis=0), crop_shape=iotf.shape)

            if plot:
                gamma = 0.5
                avg_psf = resize_with_crop_or_pad(ifft(avg_otf), crop_shape=window_size)

                fig, axes = plt.subplots(
                    nrows=4,
                    ncols=3,
                    figsize=(8, 11),
                    sharey=False,
                    sharex=False
                )
                for row in range(min(3, phi_otfs.shape[0])):
                    phi_psf = resize_with_crop_or_pad(ifft(phi_otfs[row]), crop_shape=window_size)
                    for ax in range(3):
                        m5 = axes[row, ax].imshow(np.nanmax(phi_psf, axis=ax)**gamma, cmap='magma')

                    label = f'Reconstructed\nTile {row} of {phi_otfs.shape[0]}. $\gamma$={gamma}'

                    cax = inset_axes(axes[row, -1], width="10%", height="90%", loc='center right', borderpad=-2)
                    cb = plt.colorbar(m5, cax=cax)
                    cax.yaxis.set_label_position("right")
                    cax.set_ylabel(label)

                    for ax in axes.flatten():
                        ax.axis('off')

                for ax in range(3):
                    m5 = axes[3, ax].imshow(np.nanmax(avg_psf, axis=ax)**gamma, cmap='magma')

                label = f'Reconstructed\navg $\gamma$={gamma}'

                cax = inset_axes(axes[-1,-1], width="10%", height="90%", loc='center right', borderpad=-2)
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
            val=phi_val,
            ratio=False,
            norm=False,
            log10=False,
            embedding_option='spatial_planes',
            freq_strength_threshold=freq_strength_threshold,
        )

        emb = np.concatenate([alpha, phi], axis=0)

    if emb.shape[1:] != poi_shape:
        emb = resize(emb, output_shape=(3 if no_phase else 6, *poi_shape))
        # emb = resize_with_crop_or_pad(emb, crop_shape=(3 if no_phase else 6, *poi_shape))

    if plot is not None:
        plt.style.use("default")
        if async_plot:
            Pool(1).apply_async(plot_embeddings(
                inputs=rois,
                emb=emb,
                save_path=plot,
                nrows=nrows,
                ncols=ncols,
                ztiles=ztiles
            ))
        else:
            plot_embeddings(
                inputs=rois,
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
        psf_type='../lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat',
        snr=psnr,
        psf_shape=a.shape,
        lam_detection=wavelength,
        x_voxel_size=lateral_voxel_size,
        y_voxel_size=lateral_voxel_size,
        z_voxel_size=axial_voxel_size
    )

    a /= np.max(a)
    otf = np.abs(fft(a))
    otf /= np.max(otf)
    otf[otf < threshold] = threshold

    spsf = samplepsfgen.single_psf(
        phi=0,
        normed=True,
        noise=True,
        meta=False,
    )
    sotf = np.abs(fft(spsf))
    sotf /= np.max(sotf)
    sotf[sotf < threshold] = threshold

    ipsf = samplepsfgen.single_psf(
        phi=0,
        normed=True,
        noise=False,
        meta=False,
    )
    iotf = np.abs(fft(ipsf))
    iotf /= np.max(iotf)
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

