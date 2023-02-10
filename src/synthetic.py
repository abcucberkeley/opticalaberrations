import matplotlib

matplotlib.use('Agg')

import logging
import sys
from typing import Any, Union

import numpy as np
import cupy as cp
from pathlib import Path
from skimage import transform
from functools import partial
import multiprocessing as mp
from typing import Iterable
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage.filters import sobel, scharr
from skimage.restoration import unwrap_phase
from skimage.feature import peak_local_max, match_template
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import RegularGridInterpolator
from line_profiler_pycharm import profile
from tifffile import TiffFile
from skspatial.objects import Plane, Points
from scipy import ndimage
import matplotlib.patches as patches
from astropy import convolution

from psf import PsfGenerator3D
from wavefront import Wavefront
from preprocessing import prep_sample

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SyntheticPSF:

    def __init__(
            self,
            amplitude_ranges=(-.1, .1),
            psf_type='widefield',
            distribution='single',
            embedding_option='spatial_planes',
            mode_weights='pyramid',
            signed=True,
            rotate=False,
            gamma=.75,
            n_modes=55,
            order='ansi',
            batch_size=100,
            psf_shape=(64, 64, 64),
            x_voxel_size=.108,
            y_voxel_size=.108,
            z_voxel_size=.2,
            na_detection=1.0,
            lam_detection=.510,
            refractive_index=1.33,
            snr=(10, 50),
            mean_background_noise=100,
            sigma_background_noise=(5, 10),
            cpu_workers=-1
    ):
        """
        Args:
            amplitude_ranges: range tuple, array, or wavefront object (in microns)
            psf_type: widefield or confocal
            distribution: desired distribution for the amplitudes
            gamma: optional exponent of the powerlaw distribution
            signed: optional flag to generate a symmetric (pos/neg) semi-distributions for the given range of amplitudes,
                otherwise just positive amplitudes only
            n_modes: number of zernike modes to describe the aberration
            order: eg noll or ansi, default is ansi
            batch_size: number of samples per batch
            psf_shape: shape of input psf, eg (z, y, x)
            x_voxel_size: (x) lateral sampling rate in microns
            y_voxel_size: (y) lateral sampling rate in microns
            z_voxel_size: (z) axial sampling rate in microns
            na_detection: numerical aperture of detection objective
            lam_detection: wavelength in microns
            refractive_index: refractive index
            snr: scalar or range for a uniform signal-to-noise ratio dist
            cpu_workers: number of CPU threads to use for generating PSFs
            embedding_option: type of fourier embedding to use
        """

        self.n_modes = n_modes
        self.psf_type = psf_type
        self.order = order
        self.refractive_index = refractive_index
        self.lam_detection = lam_detection
        self.na_detection = na_detection
        self.batch_size = batch_size
        self.mean_background_noise = mean_background_noise
        self.sigma_background_noise = sigma_background_noise
        self.x_voxel_size = x_voxel_size  # desired voxel size
        self.y_voxel_size = y_voxel_size
        self.z_voxel_size = z_voxel_size
        self.voxel_size = (z_voxel_size, y_voxel_size, x_voxel_size)
        self.snr = snr
        self.cpu_workers = cpu_workers
        self.distribution = distribution
        self.mode_weights = mode_weights
        self.gamma = gamma
        self.signed = signed
        self.rotate = rotate
        self.embedding_option = embedding_option

        self.psf_shape = (psf_shape[0], psf_shape[1], psf_shape[2])
        self.amplitude_ranges = amplitude_ranges

        self.psfgen = PsfGenerator3D(
            psf_shape=self.psf_shape,
            units=self.voxel_size,
            lam_detection=self.lam_detection,
            n=self.refractive_index,
            na_detection=self.na_detection,
            psf_type=psf_type
        )

        # ideal psf (theoretical, no noise)
        self.ipsf = self.theoretical_psf(normed=True)
        self.iotf = np.abs(self.fft(self.ipsf, padsize=None))
        self.iotf = self._normalize(self.iotf, self.iotf)

    @profile
    def update_ideal_psf_with_empirical(
            self,
            ideal_empirical_psf: Union[Path, np.ndarray],
            voxel_size: tuple = (.2, .108, .108),
            remove_background: bool = True,
            normalize: bool = True
    ):
        """ 

        Args:
            ideal_empirical_psf (Union[Path, np.ndarray]): _description_
            voxel_size (tuple, optional): voxel size of empirical data. Defaults to (.2, .108, .108).
            remove_background (bool, optional): _description_. Defaults to True.
            normalize (bool, optional): _description_. Defaults to True.
        """
        logger.info(f"Updating ideal PSF with empirical PSF")

        if isinstance(ideal_empirical_psf, np.ndarray):
            # assume PSF has been pre-processed already
            self.ipsf = ideal_empirical_psf
        else:
            with TiffFile(ideal_empirical_psf) as tif:
                self.ipsf = tif.asarray()
                tif.close()

            self.ipsf = prep_sample(
                np.squeeze(self.ipsf),
                model_voxel_size=self.voxel_size,
                sample_voxel_size=voxel_size,
                remove_background=remove_background,
                normalize=normalize
            )

        self.iotf = np.abs(self.fft(self.ipsf, padsize=None))
        self.iotf = np.nan_to_num(self.iotf, nan=0)

        if self.iotf.shape != self.psf_shape:
            self.iotf = transform.rescale(
                self.iotf,
                (
                    self.psf_shape[0] / self.iotf.shape[0],
                    self.psf_shape[1] / self.iotf.shape[1],
                    self.psf_shape[2] / self.iotf.shape[2],
                ),
                order=3,
                anti_aliasing=True,
            )
        self.iotf = self._normalize(self.iotf, self.iotf)

    def _randuniform(self, var):
        """Returns a random number (uniform chance) in the range provided by var. If var is a scalar, var is simply returned.

        Args:
            var : (as scalar) Returned as is.
            var : (as list) Range to provide a random number

        Returns:
            _type_: ndarray or scalar. Random sample from the range provided.

        """
        var = (var, var) if np.isscalar(var) else var

        # star unpacks a list, so that var's values become the separate arguments here
        return np.random.uniform(*var)

    def _normal_noise(self, mean, sigma, size):
        mean = self._randuniform(mean)
        sigma = self._randuniform(sigma)
        return np.random.normal(loc=mean, scale=sigma, size=size).astype(np.float32)

    def _poisson_noise(self, image):
        return np.random.poisson(lam=image).astype(np.float32) - image

    def _random_noise(self, image, mean, sigma):
        normal_noise_img = self._normal_noise(mean=mean, sigma=sigma, size=image.shape)
        poisson_noise_img = self._poisson_noise(image=image)
        noise = normal_noise_img + poisson_noise_img
        return noise

    @profile
    def _crop(self, psf: np.array, jitter: float = 0.):
        # the coordinate of the 2x larger fov psf center
        centroid = np.array([i // 2 for i in psf.shape]) - 1

        # width of the desired psf
        wz, wy, wx = self.psf_shape[0] // 2, self.psf_shape[1] // 2, self.psf_shape[2] // 2

        z = np.arange(0, psf.shape[0], dtype=int)
        y = np.arange(0, psf.shape[1], dtype=int)
        x = np.arange(0, psf.shape[2], dtype=int)

        # Add a random offset to the center
        if jitter:
            centroid += np.array([
                np.random.randint(-jitter // s, jitter // s)  # max.jitter is in microns
                for s in self.voxel_size
            ])

        # figure out the coordinates of the cropped image
        cz = np.arange(centroid[0] - wz, centroid[0] + wz, dtype=int)
        cy = np.arange(centroid[1] - wy, centroid[1] + wy, dtype=int)
        cx = np.arange(centroid[2] - wx, centroid[2] + wx, dtype=int)
        cz, cy, cx = np.meshgrid(cz, cy, cx, indexing='ij')

        interp = RegularGridInterpolator((z, y, x), psf)
        cropped_psf = interp((cz, cy, cx))
        return cropped_psf

    @profile
    def theoretical_psf(self, normed: bool = True):
        """Generates an unabberated PSF of the "desired" PSF shape and voxel size, centered.

        Args:
            normed (bool, optional): normalized will set maximum to 1. Defaults to True.

        Returns:
            _type_: 3D PSF
        """
        phi = Wavefront(
            amplitudes=np.zeros(self.n_modes),
            order=self.order,
            distribution=self.distribution,
            mode_weights=self.mode_weights,
            modes=self.n_modes,
            gamma=self.gamma,
            lam_detection=self.lam_detection
        )

        psf = self.psfgen.incoherent_psf(phi)

        if normed:
            psf /= np.max(psf)

        return psf

    @profile
    def fft(self, inputs, padsize=None):
        if padsize is not None:
            shape = inputs.shape[1]
            size = shape * (padsize / shape)
            pad = int((size - shape) // 2)
            inputs = np.pad(inputs, ((pad, pad), (pad, pad), (pad, pad)), 'constant', constant_values=0)

        try:
            otf = cp.fft.ifftshift(cp.array(inputs))
            otf = cp.fft.fftn(otf)
            otf = cp.fft.fftshift(otf)
            return cp.asnumpy(otf)
        except Exception:
            otf = np.fft.ifftshift(inputs)
            otf = np.fft.fftn(otf)
            otf = np.fft.fftshift(otf)
            return otf

    @profile
    def ifft(self, otf):
        try:
            psf = cp.fft.fftshift(cp.array(otf))
            psf = cp.fft.ifftn(psf)
            psf = cp.abs(np.fft.ifftshift(psf))
            return cp.asnumpy(psf)
        except Exception:
            psf = np.fft.fftshift(otf)
            psf = np.fft.ifftn(psf)
            psf = np.abs(np.fft.ifftshift(psf))
            return psf

    @profile
    def na_mask(self):
        """
        OTF Mask is going to be binary thresholded ideal theoretical OTF
        """
        ipsf = self.theoretical_psf(normed=True)
        mask = np.abs(self.fft(ipsf, padsize=None))

        threshold = np.nanpercentile(mask.flatten(), 65)
        mask = np.where(mask < threshold, mask, 1.)
        mask = np.where(mask >= threshold, mask, 0.)
        return mask

    @profile
    def plot_embeddings(
            self,
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
    def _normalize(self, emb, otf, freq_strength_threshold: float = 0.):
        emb /= np.nanpercentile(np.abs(otf), 99.99)
        emb[emb > 1] = 1
        emb[emb < -1] = -1
        emb = np.nan_to_num(emb, nan=0)

        if freq_strength_threshold != 0.:
            emb[np.abs(emb) < freq_strength_threshold] = 0.

        return emb

    @profile
    def principle_planes(self, emb):
        return np.stack([
            emb[emb.shape[0] // 2, :, :],
            emb[:, emb.shape[1] // 2, :],
            emb[:, :, emb.shape[2] // 2],
        ], axis=0)

    @profile
    def spatial_planes(self, emb):
        midplane = emb.shape[0] // 2
        return np.stack([
            emb[midplane, :, :],
            np.mean(emb[midplane:midplane + 5, :, :], axis=0),
            np.mean(emb[midplane + 5:midplane + 10, :, :], axis=0),
        ], axis=0)

    @profile
    def average_planes(self, emb):
        return np.stack([
            np.mean(emb, axis=0),
            np.mean(emb, axis=1),
            np.mean(emb, axis=2),
        ], axis=0)

    @profile
    def rotary_slices(self, emb):
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
    def spatial_quadrants(self, emb):
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

    def center_crop(self, inputs, window_size):
        center = [(i - 1) // 2 for i in inputs.shape]
        return inputs[
          center[0]-window_size//2:center[0]+window_size//2,
          center[1]-window_size//2:center[1]+window_size//2,
          center[2]-window_size//2:center[2]+window_size//2,
        ]

    @profile
    def shift_otf(self, psf, otf, plot, window_size=8):
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
        shifted_image = self.ifft(shifted_otf)

        # compute a new OTF of the most isolated bead
        shifted_otf = self.fft(self.center_crop(shifted_image, window_size=window_size))

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
    def remove_interference_pattern(self, psf, otf, plot, peaks=None, min_distance=5, kernel_size=15):
        """
        Normalize interference pattern from the given FFT
        Args:
            psf: input image
            otf: FFT of the given input
            plot: a toggle for visualization
            peaks: pre-defined mask of the exact bead locations
            min_distance: minimum distance for detecting adjacent beads
            kernel_size: size of the window for template matching
        """
        blured_psf = ndimage.gaussian_filter(psf, sigma=1.1)

        # get max pixel in the image
        half_length = kernel_size // 2
        poi = np.unravel_index(np.argmax(blured_psf, axis=None), psf.shape)

        # crop a window around the object for template matching
        poi = np.clip(poi, a_min=half_length, a_max=(psf.shape[0]-half_length)-1)
        init_pos = [p-half_length for p in poi]
        kernel = blured_psf[
            init_pos[0]:init_pos[0]+kernel_size,
            init_pos[1]:init_pos[1]+kernel_size,
            init_pos[2]:init_pos[2]+kernel_size,
        ]

        # convolve template with the input image
        # we're actually doing cross-corr NOT convolution
        convolued_psf = convolution.convolve_fft(blured_psf, kernel, allow_huge=True, boundary='wrap')
        convolued_psf -= np.nanmin(convolued_psf)
        convolued_psf /= np.nanmax(convolued_psf)

        if peaks is None:
            # Bead detection
            peaks = []
            detected_peaks = peak_local_max(
                convolued_psf,
                min_distance=min_distance,
                threshold_rel=.05,
                exclude_border=0,
                p_norm=2,
                num_peaks=100
            ).astype(int)

            beads = np.zeros_like(psf)
            for p in detected_peaks:
                try:
                    fov = convolued_psf[
                        p[0]-(min_distance+1):p[0]+(min_distance+1),
                        p[1]-(min_distance+1):p[1]+(min_distance+1),
                        p[2]-(min_distance+1):p[2]+(min_distance+1),
                    ]
                    if np.max(fov) > convolued_psf[p[0], p[1], p[2]]:
                        continue
                    else:
                        beads[p[0], p[1], p[2]] = psf[p[0], p[1], p[2]]
                        peaks.append(p)

                except Exception:
                    # keep peak if we are at the border of the image
                    beads[p[0], p[1], p[2]] = psf[p[0], p[1], p[2]]
                    peaks.append(p)

            peaks = np.array(peaks)
        else:
            beads = peaks.copy()
            beads[beads < .05] = 0.
            peaks = np.array([[z, y, x] for z, y, x in zip(*np.nonzero(beads))])

        if peaks.shape[0] > 0:
            # logger.info(f"Detected objects: {peaks.shape[0]}")

            interference_pattern = self.fft(beads)
            corrected_otf = otf / interference_pattern

            corrected_psf = self.ifft(corrected_otf)
            corrected_psf /= np.nanmax(corrected_psf)

            if plot is not None:
                fig, axes = plt.subplots(4, 3, figsize=(8, 8), sharey=False, sharex=False)

                for ax in range(3):
                    for p in range(peaks.shape[0]):
                        if ax == 0:

                            axes[0, ax].plot(peaks[p, 2], peaks[p, 1], marker='.', ls='', color=f'C{p}')
                            axes[2, ax].plot(peaks[p, 2], peaks[p, 1], marker='.', ls='', color=f'C{p}')
                            axes[0, ax].add_patch(patches.Rectangle(
                                xy=(peaks[p, 2] - half_length, peaks[p, 1] - half_length),
                                width=kernel_size,
                                height=kernel_size,
                                fill=None,
                                color=f'C{p}',
                                alpha=1
                            ))
                        elif ax == 1:
                            axes[0, ax].plot(peaks[p, 2], peaks[p, 0], marker='.', ls='', color=f'C{p}')
                            axes[2, ax].plot(peaks[p, 2], peaks[p, 0], marker='.', ls='', color=f'C{p}')
                            axes[0, ax].add_patch(patches.Rectangle(
                                xy=(peaks[p, 2] - half_length, peaks[p, 0] - half_length),
                                width=kernel_size,
                                height=kernel_size,
                                fill=None,
                                color=f'C{p}',
                                alpha=1
                            ))

                        elif ax == 2:
                            axes[0, ax].plot(peaks[p, 1], peaks[p, 0], marker='.', ls='', color=f'C{p}')
                            axes[2, ax].plot(peaks[p, 1], peaks[p, 0], marker='.', ls='', color=f'C{p}')
                            axes[0, ax].add_patch(patches.Rectangle(
                                xy=(peaks[p, 1] - half_length, peaks[p, 0] - half_length),
                                width=kernel_size,
                                height=kernel_size,
                                fill=None,
                                color=f'C{p}',
                                alpha=1
                            ))

                    m1 = axes[0, ax].imshow(np.nanmax(psf, axis=ax), cmap='Greys_r', alpha=.66)
                    m2 = axes[1, ax].imshow(np.nanmax(kernel, axis=ax), cmap='magma')
                    m3 = axes[2, ax].imshow(np.nanmax(convolued_psf, axis=ax), cmap='Greys_r')
                    m4 = axes[-1, ax].imshow(np.nanmax(corrected_psf, axis=ax), cmap='hot')

                for ax, m, label in zip(
                        range(4),
                        [m1, m2, m3, m4],
                        [f'Inputs (MIP)', 'Kernel', 'Detected POIs', f'Normalized (MIP)']
                ):
                    cax = inset_axes(axes[ax, -1], width="10%", height="100%", loc='center right', borderpad=-3)
                    cb = plt.colorbar(m, cax=cax)
                    cax.yaxis.set_label_position("right")
                    cax.set_ylabel(label)

                for ax in axes.flatten():
                    ax.axis('off')

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
                    m3 = axes[2, ax].imshow(np.nanmax(convolued_psf, axis=ax), cmap='Greys_r')

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

                plt.savefig(f'{plot}_interference_pattern.svg', bbox_inches='tight', dpi=300, pad_inches=.25)
                # plt.savefig(f'{plot}_interference_pattern.png', bbox_inches='tight', dpi=300, pad_inches=.25)

            return otf


    @profile
    def remove_phase_ramp(self, masked_phase, plot):
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
    def compute_emb(
            self,
            otf: np.ndarray,
            val: str,
            ratio: bool,
            norm: bool = True,
            na_mask: bool = True,
            log10: bool = False,
            embedding_option: Any = 'spatial_planes',
            freq_strength_threshold: float = 0.,
    ):
        """
        Gives the "lower dimension" representation of the data that will be shown to the model.

        Args:
            otf: fft of the input data
            val: what to compute (either 'real', 'imag'inary, or phase 'angle' from the complex OTF)
            ratio: optional toggle to return ratio of data to ideal OTF
            norm: optional toggle to normalize the data [0, 1]
            na_mask: optional toggle to apply the NA mask
            log10: optional toggle to take log10 of the FFT
            iotf: ideal empirical OTF
            freq_strength_threshold: threshold to filter out frequencies below given threshold (percentage to peak)
            embedding_option: type of embedding to use
                (`principle_planes`,  'pp'): return principle planes only (middle planes)
                (`spatial_planes`,    'sp'): return 
                (`average_planes`,    'ap'): return average of each axis
                (`rotary_slices`,     'rs'): return three radial slices
                (`spatial_quadrants`, 'sq'): return four different spatial planes in each quadrant
                 or just return the full stack if nothing is passed
        """
        mask = self.na_mask().astype(bool)

        if otf.shape != self.psf_shape:
            real = transform.rescale(
                np.real(otf),
                (
                    self.psf_shape[0] / otf.shape[0],
                    self.psf_shape[1] / otf.shape[1],
                    self.psf_shape[2] / otf.shape[2],
                ),
                order=3,
                anti_aliasing=True,
            )
            imag = transform.rescale(
                np.imag(otf),
                (
                    self.psf_shape[0] / otf.shape[0],
                    self.psf_shape[1] / otf.shape[1],
                    self.psf_shape[2] / otf.shape[2],
                ),
                order=3,
                anti_aliasing=True,
            )
            otf = real + 1j * imag

        if val == 'real':
            emb = np.real(otf)

        elif val == 'imag':
            emb = np.imag(otf)

        elif val == 'angle':
            emb = otf / np.nanpercentile(np.abs(otf), 99.99)

            if freq_strength_threshold != 0.:
                emb[np.abs(emb) < freq_strength_threshold] = 0.
                mask[np.abs(emb) < freq_strength_threshold] = 0.

            emb = np.angle(emb)
            emb = np.ma.masked_array(emb, mask=~mask, fill_value=0)
            emb = unwrap_phase(emb)
            emb = emb.filled(0)
            emb = np.nan_to_num(emb, nan=0)

        else:
            emb = np.abs(otf)

        if norm:
            emb = self._normalize(emb, otf, freq_strength_threshold=freq_strength_threshold)

        if ratio:
            emb /= self.iotf
            emb = np.nan_to_num(emb, nan=0)

        if na_mask:
            emb *= mask

        if log10:
            emb = np.log10(emb)
            emb = np.nan_to_num(emb, nan=0, posinf=0, neginf=0)

        if embedding_option.lower() == 'principle_planes' or embedding_option.lower() == 'pp':
            return self.principle_planes(emb)

        elif embedding_option.lower() == 'spatial_planes' or embedding_option.lower() == 'sp':
            return self.spatial_planes(emb)

        elif embedding_option.lower() == 'average_planes' or embedding_option.lower() == 'ap':
            return self.average_planes(emb)

        elif embedding_option.lower() == 'rotary_slices' or embedding_option.lower() == 'rs':
            return self.rotary_slices(emb)

        elif embedding_option.lower() == 'spatial_quadrants' or embedding_option.lower() == 'sq':
            return self.spatial_quadrants(emb)

        else:
            logger.warning(f"embedding_option is unrecognized : {embedding_option}")
            return emb

    @profile
    def embedding(
            self,
            inputs: Union[np.array, tuple],
            na_mask: bool = True,
            ratio: bool = True,
            norm: bool = True,
            padsize: Any = None,
            no_phase: bool = False,
            alpha_val: str = 'abs',
            phi_val: str = 'angle',
            plot: Any = None,
            log10: bool = False,
            freq_strength_threshold: float = 0.01,
            peaks: Any = None,
            remove_interference: bool = True,
            embedding_option: Any = None,
            edge_filter: bool = False,
    ):
        """
        Gives the "lower dimension" representation of the data that will be shown to the model.

        Args:
            inputs: 3D array.
            na_mask: optional toggle to apply the NA mask
            ratio: Returns ratio of data to ideal PSF,
                which helps put all the FFT voxels on a similar scale. Otherwise, straight values.
            norm: optional toggle to normalize the data [0, 1]
            padsize: pad the input to the desired size for the FFT
            no_phase: ignore/drop the phase component of the FFT
            alpha_val: use absolute values of the FFT `abs`, or the real portion `real`.
            phi_val: use the FFT phase in unwrapped radians `angle`, or the imaginary portion `imag`.
            plot: optional toggle to visualize embeddings
            remove_interference: a toggle to normalize out the interference pattern from the OTF
            peaks: masked array of the peaks of interest to compute the interference pattern between objects
            log10: optional toggle to take log10 of the FFT
            freq_strength_threshold: threshold to filter out frequencies below given threshold (percentage to peak)
            edge_filter: a toggle for running an edge filter pass on the alpha embeddings
            embedding_option: type of embedding to use.
                Capitalizing on the radial symmetry of the FFT,
                we have a few options to minimize the size of the embedding.
        """
        if isinstance(inputs, tuple):
            psf, otf = inputs
        else:
            psf = inputs
            otf = self.fft(inputs, padsize=padsize)

        if psf.ndim == 4:
            psf = np.squeeze(psf)
            otf = np.squeeze(otf)

        if no_phase:
            emb = self.compute_emb(
                otf,
                val=alpha_val,
                ratio=ratio,
                na_mask=na_mask,
                norm=norm,
                log10=log10,
                embedding_option=self.embedding_option if embedding_option is None else embedding_option,
                freq_strength_threshold=freq_strength_threshold,
            )
            if edge_filter:
                emb = scharr(emb)
                emb /= np.nanpercentile(emb, 90)
                emb[emb > 1] = 1
        else:
            alpha = self.compute_emb(
                otf,
                val=alpha_val,
                ratio=ratio,
                na_mask=na_mask,
                norm=norm,
                log10=log10,
                embedding_option=self.embedding_option if embedding_option is None else embedding_option,
                freq_strength_threshold=freq_strength_threshold,
            )

            if edge_filter:
                alpha = scharr(alpha)
                alpha /= np.nanpercentile(alpha, 90)
                alpha[alpha > 1] = 1

            if remove_interference:
                try:
                    otf = self.remove_interference_pattern(psf, otf, plot=plot, peaks=peaks)
                except TimeoutError as e:
                    logger.warning(e)

            phi = self.compute_emb(
                otf,
                val=phi_val,
                ratio=False,
                na_mask=na_mask,
                norm=False,
                log10=False,
                embedding_option='spatial_planes',
                freq_strength_threshold=freq_strength_threshold,
            )

            emb = np.concatenate([alpha, phi], axis=0)

        if plot is not None:
            plt.style.use("default")
            self.plot_embeddings(inputs=psf, emb=emb, save_path=plot)

        if psf.ndim == 4:
            return np.expand_dims(emb, axis=-1)
        else:
            return emb


    @profile
    def single_psf(
            self,
            phi: Any = None,
            normed: bool = True,
            noise: bool = False,
            meta: bool = False,
            no_phase: bool = False,
            snr_post_aberration: bool = True
    ):
        """
        Args:
            phi: wavefront object
            normed: a toggle to normalize PSF
            noise: a toggle to add noise
            meta: return extra variables for debugging
            no_phase: used only when meta=true.
            snr_post_aberration: increase photons in abberated psf to match snr of ideal psf
        """

        if not isinstance(phi, Wavefront):
            phi = Wavefront(
                phi, order=self.order,
                distribution=self.distribution,
                mode_weights=self.mode_weights,
                modes=self.n_modes,
                gamma=self.gamma,
                signed=self.signed,
                rotate=self.rotate,
                lam_detection=self.lam_detection,
            )

        psf = self.psfgen.incoherent_psf(phi)
        snr = self._randuniform(self.snr)

        if snr_post_aberration:
            psf *= snr ** 2
        else:
            total_counts_ideal = np.sum(self.ipsf)  # peak value of ideal psf is 1, self.ipsf = 1
            total_counts = total_counts_ideal * snr ** 2  # total photons of ideal psf with desired SNR
            total_counts_abr = np.sum(psf)  # total photons in abberated psf

            # scale abberated psf to have the same number of photons as ideal psf
            # (e.g. abberation doesn't destroy/create light)
            psf *= total_counts / total_counts_abr

        if noise:
            rand_noise = self._random_noise(
                image=psf,
                mean=self.mean_background_noise,
                sigma=self.sigma_background_noise,
            )
            psf += rand_noise

        psnr = np.sqrt(np.max(psf))  # peak snr will drop as abberation smooshes psf
        maxcount = np.max(psf)

        if normed:
            psf /= np.max(psf)

        if meta:
            if no_phase:
                phi.amplitudes = np.abs(phi.amplitudes)

            return psf, phi.amplitudes, psnr, maxcount
        else:
            return psf

    @profile
    def single_otf(
        self,
        phi: Any = None,
        normed: bool = True,
        noise: bool = False,
        meta: bool = False,
        na_mask: bool = False,
        ratio: bool = False,
        padsize: Any = None,
        log10: bool = False,
        plot: Any = None,
        no_phase: bool = False
    ):

        psf = self.single_psf(
            phi=phi,
            normed=normed,
            noise=noise,
            meta=meta,
            no_phase=no_phase
        )

        if meta:
            psf, y, psnr, maxcount = psf

        emb = self.embedding(
            psf,
            na_mask=na_mask,
            ratio=ratio,
            padsize=padsize,
            log10=log10,
            plot=plot,
            no_phase=no_phase,
        )

        if meta:
            if no_phase:
                y = np.abs(y)

            return emb, y, psnr, maxcount
        else:
            return emb

    def batch(self, func: Any, samples: Iterable):

        if self.cpu_workers == 1:
            logs = []
            for i in samples:
                logs.append(func(i))

        elif self.cpu_workers == -1:
            with mp.pool.ThreadPool(mp.cpu_count()) as p:
                logs = list(p.imap(func, samples))

        elif self.cpu_workers > 1:
            with mp.pool.ThreadPool(self.cpu_workers) as p:
                logs = list(p.imap(func, samples))

        else:
            logging.error('Jobs must be a positive integer')
            return False
        return logs

    def generator(self, debug=False, otf=False):
        if otf:
            gen = partial(
                self.single_otf,
                normed=True,
                noise=True,
                meta=True,
                na_mask=True,
                ratio=True,
                padsize=None
            )
        else:
            gen = partial(
                self.single_psf,
                meta=True,
                normed=True,
                noise=True,
            )

        while True:
            inputs, amplitudes, psnrs, maxcounts = zip(
                *self.batch(gen, [self.amplitude_ranges]*self.batch_size)
            )

            x = np.expand_dims(np.stack(inputs, axis=0), -1)
            y = np.stack(amplitudes, axis=0)
            psnrs = np.stack(psnrs, axis=0)
            maxcounts = np.stack(maxcounts, axis=0)

            if debug:
                yield x, y, psnrs, maxcounts
            else:
                yield x, y
