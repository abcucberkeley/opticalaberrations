import matplotlib
matplotlib.use('Agg')

import logging
import sys
from typing import Any, Union

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage.filters import scharr, window
from skimage.restoration import unwrap_phase
from skimage.feature import peak_local_max
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import RegularGridInterpolator
from line_profiler_pycharm import profile
from scipy import ndimage
import matplotlib.patches as patches
from astropy import convolution
from cupyx.scipy.ndimage import rotate
from skspatial.objects import Plane, Points
import matplotlib.gridspec as gridspec

from utils import resize_with_crop_or_pad

try:
    import cupy as cp
except ImportError as e:
    logging.warning(f"Cupy not supported on your system: {e}")

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
def rescale(emb, otf, freq_strength_threshold: float = 0.):
    emb /= np.nanpercentile(np.abs(otf), 99.99)
    emb[emb > 1] = 1
    emb[emb < -1] = -1
    emb = np.nan_to_num(emb, nan=0)

    if freq_strength_threshold != 0.:
        emb[np.abs(emb) < freq_strength_threshold] = 0.

    return emb


def center_crop(inputs, window_size):
    center = [(i - 1) // 2 for i in inputs.shape]
    return inputs[
      center[0]-window_size//2:center[0]+window_size//2,
      center[1]-window_size//2:center[1]+window_size//2,
      center[2]-window_size//2:center[2]+window_size//2,
    ]


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
def plot_embeddings(
        inputs: np.array,
        emb: np.array,
        save_path: Any,
        gamma: float = .5
):
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.autolimit_mode': 'round_numbers'
    })

    step = .1
    vmin = int(np.floor(np.nanpercentile(emb[0], 1))) if np.any(emb[0] < 0) else 0
    vmax = int(np.ceil(np.nanpercentile(emb[0], 99))) if vmin < 0 else 3
    vcenter = 1 if vmin == 0 else 0

    cmap = np.vstack((
        plt.get_cmap('GnBu_r' if vmin == 0 else 'GnBu_r', 256)(
            np.linspace(0, 1 - step, int(abs(vcenter - vmin) / step))
        ),
        [1, 1, 1, 1],
        plt.get_cmap('YlOrRd' if vmax == 3 else 'OrRd', 256)(
            np.linspace(0, 1 + step, int(abs(vcenter - vmax) / step))
        )
    ))
    cmap = mcolors.ListedColormap(cmap)

    if emb.shape[0] == 3:
        fig, axes = plt.subplots(2, 3, figsize=(8, 6))
    else:
        fig, axes = plt.subplots(3, 3, figsize=(8, 8))

    m = axes[0, 0].imshow(np.max(inputs, axis=0) ** gamma, cmap='hot', vmin=0, vmax=1)
    axes[0, 1].imshow(np.max(inputs, axis=1) ** gamma, cmap='hot', vmin=0, vmax=1)
    axes[0, 2].imshow(np.max(inputs, axis=2) ** gamma, cmap='hot', vmin=0, vmax=1)
    cax = inset_axes(axes[0, 2], width="10%", height="100%", loc='center right', borderpad=-3)
    cb = plt.colorbar(m, cax=cax)
    cax.yaxis.set_label_position("right")
    cax.set_ylabel(rf'Input (MIP) [$\gamma$={gamma}]')

    m = axes[1, 0].imshow(emb[0], cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 1].imshow(emb[1], cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 2].imshow(emb[2], cmap=cmap, vmin=vmin, vmax=vmax)
    cax = inset_axes(axes[1, 2], width="10%", height="100%", loc='center right', borderpad=-3)
    cb = plt.colorbar(m, cax=cax)
    cax.yaxis.set_label_position("right")
    cax.set_ylabel(r'Embedding ($\alpha$)')

    if emb.shape[0] > 3:
        p_vmin = -1
        p_vmax = 1
        p_vcenter = 0

        p_cmap = np.vstack((
            plt.get_cmap('GnBu_r' if p_vmin == 0 else 'GnBu_r', 256)(
                np.linspace(0, 1 - step, int(abs(p_vcenter - p_vmin) / step))
            ),
            [1, 1, 1, 1],
            plt.get_cmap('YlOrRd' if p_vmax == 3 else 'OrRd', 256)(
                np.linspace(0, 1 + step, int(abs(p_vcenter - p_vmax) / step))
            )
        ))
        p_cmap = mcolors.ListedColormap(p_cmap)

        m = axes[-1, 0].imshow(emb[3], cmap=p_cmap, vmin=p_vmin, vmax=p_vmax)
        axes[-1, 1].imshow(emb[4], cmap=p_cmap, vmin=p_vmin, vmax=p_vmax)
        axes[-1, 2].imshow(emb[5], cmap=p_cmap, vmin=p_vmin, vmax=p_vmax)
        cax = inset_axes(axes[-1, 2], width="10%", height="100%", loc='center right', borderpad=-3)
        cb = plt.colorbar(m, cax=cax)
        cax.yaxis.set_label_position("right")
        cax.set_ylabel(r'Embedding ($\varphi$)')

    for ax in axes.flatten():
        ax.axis('off')

    if save_path == True:
        plt.show()
    else:
        plt.savefig(f'{save_path}_embeddings.svg', dpi=300, bbox_inches='tight', pad_inches=.25)
        # plt.savefig(f'{save_path}_embeddings.png', dpi=300, bbox_inches='tight', pad_inches=.25)


@profile
def shift_otf(psf, otf, plot, window_size=8):
    """ Center around most isolated bead """
    beads = peak_local_max(
        psf,
        min_distance=window_size*2,
        threshold_rel=.33,
        exclude_border=False,
        p_norm=2,
        num_peaks=1
    ).astype(np.float64)

    center = [(i - 1) // 2 for i in psf.shape]
    shift = np.mean(beads, axis=0) - center

    z = np.arange(0, psf.shape[0])
    y = np.arange(0, psf.shape[1])
    x = np.arange(0, psf.shape[2])
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')

    slope = [(shift[i] * 2 * np.pi) / psf.shape[i] for i in range(3)]
    shifted_otf = otf * np.e ** (1j * (Z * slope[0] + Y * slope[1] + X * slope[2]))

    # get a realspace image of the shifted OTF
    shifted_image = ifft(shifted_otf)

    # compute a new OTF of the most isolated bead
    shifted_otf = fft(center_crop(shifted_image, window_size=window_size))

    if plot is not None:
        fig, axes = plt.subplots(1, 3, figsize=(8, 4), sharey=False, sharex=False)
        for ax in range(3):
            axes[ax].imshow(np.nanmax(shifted_image, axis=ax), aspect='equal', cmap='Greys_r')

            for p in range(beads.shape[0]):
                if ax == 0:
                    axes[ax].plot(beads[p, 2], beads[p, 1], marker='.', ls='', color=f'C{p}')
                elif ax == 1:
                    axes[ax].plot(beads[p, 2], beads[p, 0], marker='.', ls='', color=f'C{p}')
                elif ax == 2:
                    axes[ax].plot(beads[p, 1], beads[p, 0], marker='.', ls='', color=f'C{p}')

            axes[ax].axis('off')

        plt.tight_layout()
        plt.savefig(f'{plot}_shift.svg', bbox_inches='tight', dpi=300, pad_inches=.25)

    return shifted_otf


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
            plt.savefig(f"{plot}_phase_ramp.svg")

    return np.nan_to_num(masked_phase, nan=0)


@profile
def remove_interference_pattern(psf, otf, plot, pois=None, min_distance=5, kernel_size=15, max_num_peaks=100, windowing=True):
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
    # we're actually doing cross-corr NOT convolution
    convolved_psf = convolution.convolve_fft(blured_psf, kernel, allow_huge=True, boundary='fill')
    convolved_psf -= np.nanmin(convolved_psf)
    convolved_psf /= np.nanmax(convolved_psf)

    if pois is None:
        # Bead detection
        pois = []
        detected_peaks = peak_local_max(
            convolved_psf,
            min_distance=min_distance,
            threshold_rel=.1,
            exclude_border=0,
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
                    continue
                else:
                    beads[p[0], p[1], p[2]] = psf[p[0], p[1], p[2]]
                    pois.append(p)

            except Exception:
                # keep peak if we are at the border of the image
                beads[p[0], p[1], p[2]] = psf[p[0], p[1], p[2]]
                pois.append(p)

        pois = np.array(pois)
    else:
        beads = pois.copy()
        beads[beads < .05] = 0.
        pois = np.array([[z, y, x] for z, y, x in zip(*np.nonzero(beads))])

    if pois.shape[0] > 0:
        # logger.info(f"Detected objects: {pois.shape[0]}")

        interference_pattern = fft(beads)
        corrected_otf = otf / interference_pattern
        if windowing:
            corrected_psf = ifft(corrected_otf) * window(('tukey', 0.8), corrected_otf.shape)
            corrected_otf = fft(corrected_psf)
        else:
            corrected_psf = ifft(corrected_otf)

        corrected_psf /= np.nanmax(corrected_psf)

        if plot is not None:
            fig, axes = plt.subplots(5, 3, figsize=(8, 11), sharey=False, sharex=False)
            for ax in range(3):
                for p in range(pois.shape[0]):
                    if ax == 0:
                        axes[2, ax].plot(pois[p, 2], pois[p, 1], marker='.', ls='', color=f'C{p}')
                        axes[2, ax].add_patch(patches.Rectangle(
                            xy=(pois[p, 2] - half_length, pois[p, 1] - half_length),
                            width=kernel_size,
                            height=kernel_size,
                            fill=None,
                            color=f'C{p}',
                            alpha=1
                        ))
                    elif ax == 1:
                        axes[2, ax].plot(pois[p, 2], pois[p, 0], marker='.', ls='', color=f'C{p}')
                        axes[2, ax].add_patch(patches.Rectangle(
                            xy=(pois[p, 2] - half_length, pois[p, 0] - half_length),
                            width=kernel_size,
                            height=kernel_size,
                            fill=None,
                            color=f'C{p}',
                            alpha=1
                        ))

                    elif ax == 2:
                        axes[2, ax].plot(pois[p, 1], pois[p, 0], marker='.', ls='', color=f'C{p}')
                        axes[2, ax].add_patch(patches.Rectangle(
                            xy=(pois[p, 1] - half_length, pois[p, 0] - half_length),
                            width=kernel_size,
                            height=kernel_size,
                            fill=None,
                            color=f'C{p}',
                            alpha=1
                        ))
                m1 = axes[0, ax].imshow(np.nanmax(psf, axis=ax), cmap='hot')
                m2 = axes[1, ax].imshow(np.nanmax(kernel, axis=ax), cmap='hot')
                m3 = axes[2, ax].imshow(np.nanmax(convolved_psf, axis=ax), cmap='Greys_r', alpha=.66)

                interference = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=axes[3, ax], wspace=0.05, hspace=0)
                ax1 = fig.add_subplot(interference[0])
                ax1.imshow(np.nanmax(beads, axis=ax), cmap='hot')
                ax1.axis('off')
                ax1.set_title(r'$\mathcal{S}$')

                ax2 = fig.add_subplot(interference[1])
                m4 = ax2.imshow(np.nanmax(abs(interference_pattern), axis=ax), cmap='magma')
                ax2.axis('off')
                ax2.set_title(r'$|\mathscr{F}(\mathcal{S})|$')

                m5 = axes[-1, ax].imshow(np.nanmax(corrected_psf, axis=ax), cmap='hot')

            for ax, m, label in zip(
                    range(5),
                    [m1, m2, m3, m4, m5],
                    [r'Inputs', 'Kernel', 'Peak detection', 'Interference', r'Reconstructed']
            ):
                cax = inset_axes(axes[ax, -1], width="10%", height="90%", loc='center right', borderpad=-3)
                cb = plt.colorbar(m, cax=cax)
                cax.yaxis.set_label_position("right")
                cax.set_ylabel(label)

            for ax in axes.flatten():
                ax.axis('off')

            axes[0, 0].set_title('XY')
            axes[0, 1].set_title('XZ')
            axes[0, 2].set_title('YZ')
            axes[-1, 1].set_title(
                r"$\mathscr{F}^{-1} \left( \mathscr{F}(\mathscr{i}) / \mathscr{F}(\mathcal{S}) \right)$"
            )

            plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.2, wspace=0.1)
            plt.savefig(f'{plot}_interference_pattern.svg', bbox_inches='tight', dpi=300, pad_inches=.25)
            # plt.savefig(f'{plot}_interference_pattern.png', bbox_inches='tight', dpi=300, pad_inches=.25)

        return corrected_otf
    else:
        logger.warning("No objects were detected")

        if plot is not None:
            fig, axes = plt.subplots(3, 3, figsize=(8, 8), sharey=False, sharex=False)

            for ax in range(3):
                m1 = axes[0, ax].imshow(np.nanmax(psf, axis=ax), cmap='Greys_r', alpha=.66)
                m2 = axes[1, ax].imshow(np.nanmax(kernel, axis=ax), cmap='Greys_r')
                m3 = axes[2, ax].imshow(np.nanmax(convolved_psf, axis=ax), cmap='Greys_r')

            for ax, m, label in zip(
                    range(3),
                    [m1, m2, m3],
                    [f'Inputs (MIP)', 'Kernel', 'Detected POIs']
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

            plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.35, wspace=0.1)
            plt.savefig(f'{plot}_interference_pattern.svg', bbox_inches='tight', dpi=300, pad_inches=.25)

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
        real = resize_with_crop_or_pad(np.real(otf), crop_shape=iotf.shape) # only center crop
        imag = resize_with_crop_or_pad(np.imag(otf), crop_shape=iotf.shape) # only center crop
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
        emb = np.ma.masked_array(emb, mask=~na_mask, fill_value=0)
        if len(np.ma.nonzero(emb)[0]) > 100:
            emb = unwrap_phase(emb)
        emb = emb.filled(0)
        emb = np.nan_to_num(emb, nan=0)

    else:
        emb = np.abs(otf)

    if norm:
        emb = rescale(emb, otf, freq_strength_threshold=freq_strength_threshold)

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
        poi_shape: tuple = (64, 64)
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
            otf = remove_interference_pattern(psf, otf, plot=plot, pois=pois, windowing=True)

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

    if emb.shape != poi_shape:
        emb = resize_with_crop_or_pad(emb, crop_shape=(3 if no_phase else 6, *poi_shape))

    if plot is not None:
        plt.style.use("default")
        plot_embeddings(inputs=psf, emb=emb, save_path=plot)

    if digital_rotations is not None:
        gpu_embeddings = cp.array(emb)
        emb = np.array([
            cp.asnumpy(rotate(gpu_embeddings, angle=angle, reshape=False, axes=(-2, -1)))
            for angle in tqdm(digital_rotations, desc=f"Generating digital rotations")
        ])
        del gpu_embeddings

    if emb.shape[-1] != 1:
        emb = np.expand_dims(emb, axis=-1)

    return emb
