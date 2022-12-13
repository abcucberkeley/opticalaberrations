import matplotlib
matplotlib.use('Agg')

import logging
import sys
from typing import Any

import numpy as np
from skimage import transform
from functools import partial
import multiprocessing as mp
from typing import Iterable
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tifffile import imsave
from tqdm import trange
from skimage.filters import window
from skimage.restoration import unwrap_phase
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy.lib.stride_tricks import sliding_window_view
from scipy.interpolate import RegularGridInterpolator
from line_profiler_pycharm import profile

from psf import PsfGenerator3D
from wavefront import Wavefront

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
        distribution='dirichlet',
        mode_weights='uniform',
        bimodal=False,
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
        sigma_background_noise=4,
        cpu_workers=-1
    ):
        """
        Args:
            amplitude_ranges: range tuple, array, or wavefront object (in microns)
            psf_type: widefield or confocal
            distribution: desired distribution for the amplitudes
            gamma: optional exponent of the powerlaw distribution
            bimodal: optional flag to generate a symmetric (pos/neg) semi-distributions for the given range of amplitudes, otherwise just positive amplitudes only
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
        self.x_voxel_size = x_voxel_size    # desired voxel size
        self.y_voxel_size = y_voxel_size
        self.z_voxel_size = z_voxel_size
        self.voxel_size = (z_voxel_size, y_voxel_size, x_voxel_size)
        self.snr = snr
        self.cpu_workers = cpu_workers
        self.distribution = distribution
        self.mode_weights = mode_weights
        self.gamma = gamma
        self.bimodal = bimodal
        self.rotate = rotate

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

        self.ipsf = self.theoretical_psf(normed=True)      # ipsf = ideal psf (theoretical, no noise)
        self.iotf = self.fft(self.ipsf, padsize=None)      # iotf = ideal otf

    def _normal_noise(self, mean, sigma, size):
        return np.random.normal(loc=mean, scale=sigma, size=size).astype(np.float32)

    def _poisson_noise(self, image):
        return np.random.poisson(lam=image).astype(np.float32) - image

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
                np.random.randint(-jitter//s, jitter//s)  # max.jitter is in microns
                for s in self.voxel_size
            ])

        # figure out the coordinates of the cropped image
        cz = np.arange(centroid[0]-wz, centroid[0]+wz, dtype=int)
        cy = np.arange(centroid[1]-wy, centroid[1]+wy, dtype=int)
        cx = np.arange(centroid[2]-wx, centroid[2]+wx, dtype=int)
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
        psf /= np.max(psf) if normed else psf
        return psf

    @profile
    def fft(self, inputs, padsize=None,):
        if padsize is not None:
            shape = inputs.shape[1]
            size = shape * (padsize / shape)
            pad = int((size - shape)//2)
            inputs = np.pad(inputs, ((pad, pad), (pad, pad), (pad, pad)), 'constant', constant_values=0)

        otf = np.fft.ifftshift(inputs)
        otf = np.fft.fftn(otf)
        otf = np.fft.fftshift(otf)
        return otf

    @profile
    def na_mask(self):
        mask = np.abs(self.iotf)
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
        no_phase: bool = False,
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
        # plt.style.use("dark_background")

        step = .1
        vmin = -1 if np.any(emb[0] < 0) else 0
        vmax = 1 if vmin < 0 else 3
        vcenter = 1 if vmin == 0 else 0

        cmap = np.vstack((
            plt.get_cmap('terrain' if vmin == 0 else 'GnBu_r', 256)(
                np.linspace(0, 1 - step, int(abs(vcenter - vmin) / step))
            ),
            [1, 1, 1, 1],
            plt.get_cmap('YlOrRd' if vmax == 3 else 'OrRd', 256)(
                np.linspace(0, 1 + step, int(abs(vcenter - vmax) / step))
            )
        ))
        cmap = mcolors.ListedColormap(cmap)

        if no_phase:
            fig, axes = plt.subplots(2, 3, figsize=(8, 8))
        else:
            fig, axes = plt.subplots(3, 3, figsize=(8, 8))

        m = axes[0, 0].imshow(np.max(inputs, axis=0)**.5, cmap='hot', vmin=0, vmax=1)
        axes[0, 1].imshow(np.max(inputs, axis=1)**.5, cmap='hot', vmin=0, vmax=1)
        axes[0, 2].imshow(np.max(inputs, axis=2)**.5, cmap='hot', vmin=0, vmax=1)
        cax = inset_axes(axes[0, 2], width="10%", height="100%", loc='center right', borderpad=-3)
        cb = plt.colorbar(m, cax=cax)
        cax.yaxis.set_label_position("right")
        cax.set_ylabel('Input (MIP)')

        m = axes[1, 0].imshow(emb[0], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1, 1].imshow(emb[1], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1, 2].imshow(emb[2], cmap=cmap, vmin=vmin, vmax=vmax)
        cax = inset_axes(axes[1, 2], width="10%", height="100%", loc='center right', borderpad=-3)
        cb = plt.colorbar(m, cax=cax)
        cax.yaxis.set_label_position("right")
        cax.set_ylabel(r'Embedding ($\alpha$)')

        if not no_phase:
            p_vmin = -1 if np.any(emb[3] < 0) else 0
            p_vmax = 1 if p_vmin < 0 else 3
            p_vcenter = 1 if p_vmin == 0 else 0

            p_cmap = np.vstack((
                plt.get_cmap('terrain' if p_vmin == 0 else 'GnBu_r', 256)(
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
            plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight', pad_inches=.25)

    @profile
    def _normalize(self, emb, otf, freq_strength_threshold: float = 0.):
        emb /= np.nanpercentile(np.abs(otf), 99.99)
        emb[emb > 1] = 1
        emb[emb < -1] = -1
        emb = np.nan_to_num(emb, nan=0)

        if freq_strength_threshold != 0.:
            emb[(emb > 0) * (emb < freq_strength_threshold)] = 0.
            emb[(emb < 0) * (emb > -1 * freq_strength_threshold)] = 0.
        return emb

    @profile
    def compute_emb(
        self,
        otf: np.ndarray,
        val: str,
        ratio: bool,
        norm: bool,
        na_mask: bool,
        log10: bool,
        principle_planes: bool,
        freq_strength_threshold: float
    ):
        mask = self.na_mask()
        iotf = np.abs(self.iotf)

        if val == 'real':
            emb = np.real(otf)
        elif val == 'imag':
            emb = np.imag(otf)
        elif val == 'angle':
            emb = np.angle(otf)
            emb = unwrap_phase(emb)
        else:
            emb = np.abs(otf)

        if norm:
            iotf = self._normalize(iotf, iotf)
            emb = self._normalize(emb, otf, freq_strength_threshold=freq_strength_threshold)

        if emb.shape != self.psf_shape:
            emb = transform.rescale(
                emb,
                (
                    self.psf_shape[0] / emb.shape[0],
                    self.psf_shape[1] / emb.shape[1],
                    self.psf_shape[2] / emb.shape[2],
                ),
                order=3,
                anti_aliasing=True,
            )

        if ratio:
            emb /= iotf
            emb = np.nan_to_num(emb, nan=0)

        if na_mask:
            emb *= mask

        if log10:
            emb = np.log10(emb)
            emb = np.nan_to_num(emb, nan=0, posinf=0, neginf=0)

        if principle_planes:
            emb = np.stack([
                emb[emb.shape[0] // 2, :, :],
                emb[:, emb.shape[1] // 2, :],
                emb[:, :, emb.shape[2] // 2],
            ], axis=0)

        return emb

    @profile
    def embedding(
        self,
        psf: np.array,
        na_mask: bool = True,
        ratio: bool = True,
        norm: bool = True,
        padsize: Any = None,
        no_phase: bool = False,
        alpha_val: str = 'abs',
        phi_val: str = 'angle',
        plot: Any = None,
        log10: bool = False,
        principle_planes: bool = True,
        freq_strength_threshold: float = 0.01,
    ):
        """Gives the "lower dimension" representation of the data that will be shown to the model.
        Mostly this is used to return the three principle planes from the 3D OTF.

        Args:
            psf (np.array): 3D PSF.
            na_mask (bool, optional): _description_. Defaults to True.
            ratio (bool, optional): Returns ratio of data to ideal PSF, which helps put all the FFT voxels on a similiar scale. Otherwise straight values. Defaults ratio=True.
            norm (bool, optional): _description_. Defaults to True.
            padsize (Any, optional): _description_. Defaults to None.
            no_phase (bool, optional): _description_. Defaults to False.
            alpha_val (str, optional): _description_. Defaults to 'abs'.
            phi_val (str, optional): show the FFT phase in unwrapped radians 'angle' or the imaginary portion 'imag'. Defaults to 'angle'.
            plot (Any, optional): _description_. Defaults to None.
            log10 (bool, optional): _description_. Defaults to False.
            principle_planes (bool, optional): _description_. Defaults to True.
            freq_strength_threshold (float, optional): _description_. Defaults to 0.01.

        Returns:
            _type_: _description_
        """
        if psf.ndim == 4:
            psf = np.squeeze(psf)

        otf = self.fft(psf, padsize=padsize)

        if no_phase:
            emb = self.compute_emb(
                otf,
                val=alpha_val,
                ratio=ratio,
                na_mask=na_mask,
                norm=norm,
                log10=log10,
                principle_planes=principle_planes,
                freq_strength_threshold=freq_strength_threshold,
            )
        else:
            alpha = self.compute_emb(
                otf,
                val=alpha_val,
                ratio=ratio,
                na_mask=na_mask,
                norm=norm,
                log10=log10,
                principle_planes=principle_planes,
                freq_strength_threshold=freq_strength_threshold,
            )

            phi = self.compute_emb(
                otf,
                val=phi_val,
                ratio=ratio,
                na_mask=na_mask,
                norm=norm,
                log10=log10,
                principle_planes=principle_planes,
                freq_strength_threshold=freq_strength_threshold,
            )

            emb = np.concatenate([alpha, phi], axis=0)

        if plot is not None and principle_planes:
            self.plot_embeddings(inputs=psf, emb=emb, save_path=plot, no_phase=no_phase)

        if psf.ndim == 4:
            return np.expand_dims(emb, axis=-1)
        else:
            return emb

    @profile
    def rolling_embedding(
        self,
        psf: np.array,
        na_mask: bool = True,
        apodization: bool = True,
        ratio: bool = True,
        strides: int = 32,
        padsize: Any = None,
        plot: Any = None,
        log10: bool = False,
        principle_planes: bool = False,
    ):
        if psf.ndim == 4:
            psf = np.squeeze(psf)

        windows = np.reshape(
            sliding_window_view(psf, window_shape=self.psf_shape)[::strides, ::strides, ::strides],
            (-1, *self.psf_shape)  # stack windows
        )

        embeddings = []
        for w in trange(windows.shape[0], desc='Sliding windows'):
            inputs = windows[w]
            inputs /= np.nanpercentile(inputs, 99.99)
            inputs[inputs > 1] = 1
            inputs = np.nan_to_num(inputs, nan=0)

            if apodization:
                circular_mask = window(('general_gaussian', 10 / 3, 2.5 * 10), inputs.shape)
                inputs *= circular_mask

                # corner_mask = np.zeros_like(inputs, dtype=int)
                # corner_mask[1:-1, 1:-1, 1:-1] = 1.
                # corner_mask = distance_transform_edt(corner_mask, return_distances=True)
                # corner_mask = .5 - (.5 * np.cos((np.pi*corner_mask)/apodization_dist))
                # corner_mask[
                #     apodization_dist:inputs.shape[0] - apodization_dist,
                #     apodization_dist:inputs.shape[1] - apodization_dist,
                #     apodization_dist:inputs.shape[2] - apodization_dist,
                # ] = 1.
                # # corner_mask = gaussian(corner_mask, sigma=2)
                #
                # import matplotlib.pyplot as plt
                # fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                # axes[0].imshow(circular_mask[circular_mask.shape[0]//2, :, :], cmap='magma')
                # axes[0].set_title('Circular mask')
                # axes[1].imshow(corner_mask[corner_mask.shape[0]//2, :, :], cmap='magma')
                # axes[1].set_title('Corner mask')
                # plt.show()

            embeddings.append(
                self.embedding(
                    psf=inputs,
                    na_mask=na_mask,
                    ratio=ratio,
                    padsize=padsize,
                    log10=log10,
                    principle_planes=principle_planes,
                    plot=f"{plot}_window_{w}"
                )
            )

        if principle_planes:
            embeddings = np.array(embeddings)
            emb = np.vstack([np.nanmax(embeddings[:, :3], axis=0), np.nanmean(embeddings[:, 3:], axis=0)])
        else:
            embeddings = np.array(embeddings)
            alpha = np.nanmax(embeddings[:, 0], axis=0)
            phi = embeddings[:, 1]
            phi_pos = np.nanmax(phi*(phi >= 0), axis=0)
            phi_neg = np.nanmin(phi*(phi < 0), axis=0)
            emb = np.stack([alpha, phi_pos+phi_neg])

            imsave(f"{plot}_alpha.tif", emb[0])
            imsave(f"{plot}_phi.tif", emb[1])

        if plot is not None and principle_planes:
            self.plot_embeddings(inputs=psf, emb=emb, save_path=plot)

        return emb

    @profile
    def single_psf(
        self,
        phi: Any = None,
        normed: bool = True,
        noise: bool = False,
        meta: bool = False,
        no_phase: bool = False,
    ):
        """
        Args:
            phi: wavefront object
            normed: a toggle to normalize PSF
            noise: a toggle to add noise
            meta: return extra variables for debugging
        """
        snr = self._randuniform(self.snr)

        if not isinstance(phi, Wavefront):
            phi = Wavefront(
                phi, order=self.order,
                distribution=self.distribution,
                mode_weights=self.mode_weights,
                modes=self.n_modes,
                gamma=self.gamma,
                bimodal=self.bimodal,
                rotate=self.rotate,
                lam_detection=self.lam_detection,
            )

        psf = self.psfgen.incoherent_psf(phi) * snr**2

        rand_noise = self._random_noise(
            image=psf,
            mean=self.mean_background_noise,
            sigma=self.sigma_background_noise,
        )
        noisy_psf = rand_noise + psf if noise else psf
        psnr = np.sqrt(np.max(noisy_psf))
        maxcount = np.max(noisy_psf)

        noisy_psf /= np.max(noisy_psf) if normed else noisy_psf

        if meta:
            if no_phase:
                phi.amplitudes = np.abs(phi.amplitudes)

            return noisy_psf, phi.amplitudes, psnr, maxcount
        else:
            return noisy_psf

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
