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
        gamma=1.5,
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
        max_jitter=0,
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
            max_jitter: randomly move the center point within a given limit (microns)
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
        self.max_jitter = max_jitter
        self.x_voxel_size = x_voxel_size
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
        self.theoretical_psf_shape = (2 * psf_shape[0], 2 * psf_shape[1], 2 * psf_shape[2])
        self.amplitude_ranges = amplitude_ranges

        self.psfgen = PsfGenerator3D(
            psf_shape=self.theoretical_psf_shape,
            units=self.voxel_size,
            lam_detection=self.lam_detection,
            n=self.refractive_index,
            na_detection=self.na_detection,
            psf_type=psf_type
        )

        self.ipsf = self.theoretical_psf(normed=True, noise=False)
        self.iotf = self.fft(self.ipsf, padsize=None)

    def _normal_noise(self, mean, sigma, size):
        return np.random.normal(loc=mean, scale=sigma, size=size).astype(np.float32)

    def _poisson_noise(self, image):
        return np.random.poisson(lam=image).astype(np.float32) - image

    def _randuniform(self, var):
        var = (var, var) if np.isscalar(var) else var
        return np.random.uniform(*var)

    def _random_noise(self, image, mean, sigma):
        normal_noise_img = self._normal_noise(mean=mean, sigma=sigma, size=image.shape)
        poisson_noise_img = self._poisson_noise(image=image)
        noise = normal_noise_img + poisson_noise_img
        return noise

    def _crop(self, psf: np.array, voxel_size: tuple, jitter: bool = False):
        centroid = np.array([i // 2 for i in psf.shape])
        mz, my, mx = self.psf_shape[0] // 2, self.psf_shape[1] // 2, self.psf_shape[2] // 2

        if jitter and self.max_jitter != 0:
            centroid += np.array([np.random.randint(-self.max_jitter / s, self.max_jitter / s) for s in voxel_size])

        # wrap edges
        lz = mz if (centroid[0] - mz) < 0 else centroid[0] - mz
        ly = my if (centroid[1] - my) < 0 else centroid[1] - my
        lx = mx if (centroid[2] - mx) < 0 else centroid[2] - mx

        hz = psf.shape[0]-mz if (centroid[0] + mz) > psf.shape[0] else centroid[0] + mz
        hy = psf.shape[1]-my if (centroid[1] + my) > psf.shape[1] else centroid[1] + my
        hx = psf.shape[2]-mx if (centroid[2] + mx) > psf.shape[2] else centroid[2] + mx

        cropped_psf = psf[lz:hz, ly:hy, lx:hx]
        cropped_psf = transform.resize(cropped_psf, self.psf_shape, order=3)
        return cropped_psf

    def _axial_resample(
        self, vol: np.array,
        axial_voxel_size: float,
        lateral_voxel_size: float,
        zplanes: Any,
    ):
        step = int(np.round(axial_voxel_size / lateral_voxel_size, 0))
        indices = np.arange(self.theoretical_psf_shape[0])

        if zplanes is None:
            start = np.random.randint(step)
            targets = np.arange(vol.shape[0])[start::step]
            mask = np.zeros_like(indices)
            np.put(mask, targets, np.ones_like(targets))

        elif np.isscalar(zplanes) and zplanes == 0:
            mask = np.ones_like(indices)

        else:
            if zplanes.ndim > 1:
                zplanes = zplanes.reshape(len(indices), )

            if np.any(zplanes > 1):
                mask = np.zeros_like(indices)
                np.put(mask, zplanes, np.ones_like(zplanes))
            else:
                mask = zplanes

        vol = vol[indices[np.where(mask)], :, :]
        scaled_psf = transform.resize(
            vol,
            self.theoretical_psf_shape,
            order=3,
        )
        return scaled_psf, mask

    def theoretical_psf(self, normed: bool = True, snr: int = 1000, noise: bool = False):
        x_voxel_size = self._randuniform(self.x_voxel_size)
        y_voxel_size = self._randuniform(self.y_voxel_size)
        z_voxel_size = self._randuniform(self.z_voxel_size)
        voxel_size = (z_voxel_size, y_voxel_size, x_voxel_size)

        phi = Wavefront(
            amplitudes=np.zeros(self.n_modes),
            order=self.order,
            distribution=self.distribution,
            mode_weights=self.mode_weights,
            modes=self.n_modes,
            gamma=self.gamma,
            lam_detection=self.lam_detection
        )

        if noise:
            psf = self.psfgen.incoherent_psf(phi) * snr * self.mean_background_noise
            rand_noise = self._random_noise(
                image=psf,
                mean=self.mean_background_noise,
                sigma=self.sigma_background_noise,
            )
            psf = rand_noise + psf if noise else psf
        else:
            psf = self.psfgen.incoherent_psf(phi)

        psf, zplanes = self._axial_resample(
            psf,
            axial_voxel_size=z_voxel_size,
            lateral_voxel_size=max([x_voxel_size, y_voxel_size]),
            zplanes=0
        )

        psf = self._crop(psf, voxel_size=voxel_size)
        psf /= np.max(psf) if normed else psf
        return psf

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

    def na_mask(self):
        mask = np.abs(self.iotf)
        threshold = np.nanpercentile(mask.flatten(), 65)
        mask = np.where(mask < threshold, mask, 1.)
        mask = np.where(mask >= threshold, mask, 0.)
        return mask

    def plot_embeddings(
        self,
        psf: np.array,
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

        p_vmin = -1 if np.any(emb[3] < 0) else 0
        p_vmax = 1 if p_vmin < 0 else 3

        vcenter = 1 if vmin == 0 else 0
        p_vcenter = 1 if p_vmin == 0 else 0

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

        if no_phase:
            fig, axes = plt.subplots(2, 3, figsize=(8, 8))
        else:
            fig, axes = plt.subplots(3, 3, figsize=(8, 8))

        m = axes[0, 0].imshow(np.max(psf, axis=0)**.5, cmap='hot', vmin=0, vmax=1)
        axes[0, 1].imshow(np.max(psf, axis=1)**.5, cmap='hot', vmin=0, vmax=1)
        axes[0, 2].imshow(np.max(psf, axis=2).T**.5, cmap='hot', vmin=0, vmax=1)
        cax = inset_axes(axes[0, 2], width="10%", height="100%", loc='center right', borderpad=-3)
        cb = plt.colorbar(m, cax=cax)
        cax.yaxis.set_label_position("right")
        cax.set_ylabel('Input (MIP)')

        m = axes[1, 0].imshow(emb[0], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1, 1].imshow(emb[1], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1, 2].imshow(emb[2].T, cmap=cmap, vmin=vmin, vmax=vmax)
        cax = inset_axes(axes[1, 2], width="10%", height="100%", loc='center right', borderpad=-3)
        cb = plt.colorbar(m, cax=cax)
        cax.yaxis.set_label_position("right")
        cax.set_ylabel(r'Embedding ($\alpha$)')

        if not no_phase:
            m = axes[-1, 0].imshow(emb[3], cmap=p_cmap, vmin=p_vmin, vmax=p_vmax)
            axes[-1, 1].imshow(emb[4], cmap=p_cmap, vmin=p_vmin, vmax=p_vmax)
            axes[-1, 2].imshow(emb[5].T, cmap=p_cmap, vmin=p_vmin, vmax=p_vmax)
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
        gamma: float = 1.,
        freq_strength_threshold: float = 0.01,
    ):
        if psf.ndim == 4:
            psf = np.squeeze(psf)

        otf = self.fft(psf, padsize=padsize)

        if alpha_val == 'real':
            alpha = np.real(otf)
            iotf = np.real(self.iotf)
        else:
            alpha = np.abs(otf)
            iotf = np.abs(self.iotf)

        if phi_val == 'imag':
            phi = np.imag(otf)
        elif phi_val == 'angle':
            phi = np.angle(otf)
            phi = unwrap_phase(phi)
        else:
            phi = np.abs(otf)

        if norm:
            iotf /= np.nanpercentile(np.abs(iotf), 99.99)
            iotf[iotf > 1] = 1
            iotf[iotf < -1] = -1
            iotf = np.nan_to_num(iotf, nan=0)

            alpha /= np.nanpercentile(np.abs(otf), 99.99)
            alpha[alpha > 1] = 1
            alpha[alpha < -1] = -1
            alpha = np.nan_to_num(alpha, nan=0)

            phi /= np.nanpercentile(np.abs(otf), 99.99)
            phi[phi > 1] = 1
            phi[phi < -1] = -1
            phi = np.nan_to_num(phi, nan=0)

        if freq_strength_threshold != 0.:
            alpha[(alpha > 0) * (alpha < freq_strength_threshold)] = 0.
            alpha[(alpha < 0) * (alpha > -1 * freq_strength_threshold)] = 0.

            phi[(phi > 0) * (phi < freq_strength_threshold)] = 0.
            phi[(phi < 0) * (phi > -1 * freq_strength_threshold)] = 0.

        if alpha.shape != self.psf_shape:
            alpha = transform.rescale(
                alpha,
                (
                    self.psf_shape[0] / alpha.shape[0],
                    self.psf_shape[1] / alpha.shape[1],
                    self.psf_shape[2] / alpha.shape[2],
                ),
                order=3,
                anti_aliasing=True,
            )

        if phi.shape != self.psf_shape:
            phi = transform.rescale(
                phi,
                (
                    self.psf_shape[0] / phi.shape[0],
                    self.psf_shape[1] / phi.shape[1],
                    self.psf_shape[2] / phi.shape[2],
                ),
                order=3,
                anti_aliasing=True,
            )

        if ratio:
            alpha = alpha ** gamma
            alpha /= iotf
            alpha = np.nan_to_num(alpha, nan=0)

            phi /= iotf
            phi = np.nan_to_num(phi, nan=0)

        if na_mask:
            mask = self.na_mask()
            alpha *= mask
            phi *= mask

        if log10:
            alpha = np.log10(alpha)
            alpha = np.nan_to_num(alpha, nan=0, posinf=0, neginf=0)

            phi = np.log10(phi)
            phi = np.nan_to_num(phi, nan=0, posinf=0, neginf=0)

        if principle_planes:
            if no_phase:
                emb = np.stack([
                    alpha[alpha.shape[0] // 2, :, :],
                    alpha[:, alpha.shape[1] // 2, :],
                    alpha[:, :, alpha.shape[2] // 2],
                ], axis=0)
            else:
                emb = np.stack([
                    alpha[alpha.shape[0] // 2, :, :],
                    alpha[:, alpha.shape[1] // 2, :],
                    alpha[:, :, alpha.shape[2] // 2],
                    phi[phi.shape[0] // 2, :, :],
                    phi[:, phi.shape[1] // 2, :],
                    phi[:, :, phi.shape[2] // 2],
                ], axis=0)
        else:
            emb = np.stack([alpha, phi], axis=0)
            imsave(f"{plot}_alpha.tif", alpha)
            imsave(f"{plot}_phi.tif", phi)

        if plot is not None and principle_planes:
            self.plot_embeddings(psf=psf, emb=emb, save_path=plot, no_phase=no_phase)

        if psf.ndim == 4:
            return np.expand_dims(emb, axis=-1)
        else:
            return emb

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
            self.plot_embeddings(psf=psf, emb=emb, save_path=plot)

        return emb

    def single_psf(
        self,
        phi: Any = None,
        zplanes: Any = None,
        normed: bool = True,
        noise: bool = False,
        augmentation: bool = False,
        meta: bool = False,
        no_phase: bool = False,
    ):
        """
        Args:
            phi: wavefront object
            zplanes: one-hot encoded mask for target indices
            normed: a toggle to normalize PSF
            noise: a toggle to add noise
            augmentation: a toggle for data augmentation
            meta: return extra variables for debugging
        """
        snr = self._randuniform(self.snr)
        x_voxel_size = self._randuniform(self.x_voxel_size)
        y_voxel_size = self._randuniform(self.y_voxel_size)
        z_voxel_size = self._randuniform(self.z_voxel_size)
        voxel_size = (z_voxel_size, y_voxel_size, x_voxel_size)

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

        noisy_psf, zplanes = self._axial_resample(
            noisy_psf,
            axial_voxel_size=z_voxel_size,
            lateral_voxel_size=max([x_voxel_size, y_voxel_size]),
            zplanes=zplanes
        )

        if augmentation:
            noisy_psf = self._crop(noisy_psf, voxel_size=voxel_size, jitter=True)
            # noisy_psf = noisy_psf ** np.random.uniform(low=.25, high=1.25)
        else:
            noisy_psf = self._crop(noisy_psf, voxel_size=voxel_size)

        noisy_psf /= np.max(noisy_psf) if normed else noisy_psf

        if meta:
            if no_phase:
                phi.amplitudes = np.abs(phi.amplitudes)

            return noisy_psf, phi.amplitudes, psnr, zplanes, maxcount
        else:
            return noisy_psf

    def single_otf(
        self,
        phi: Any = None,
        zplanes: Any = None,
        normed: bool = True,
        noise: bool = False,
        augmentation: bool = False,
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
            zplanes=zplanes,
            normed=normed,
            noise=noise,
            augmentation=augmentation,
            meta=meta,
            no_phase=no_phase
        )

        if meta:
            psf, y, psnr, zplanes, maxcount = psf

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

            return emb, y, psnr, zplanes, maxcount
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
                zplanes=0,
                normed=True,
                noise=True,
                augmentation=True,
                meta=True,
                na_mask=True,
                ratio=True,
                padsize=None
            )
        else:
            gen = partial(
                self.single_psf,
                zplanes=0,
                meta=True,
                normed=True,
                noise=True,
                augmentation=True
            )

        while True:
            inputs, amplitudes, psnrs, zplanes, maxcounts = zip(
                *self.batch(gen, [self.amplitude_ranges]*self.batch_size)
            )

            x = np.expand_dims(np.stack(inputs, axis=0), -1)

            y = np.stack(amplitudes, axis=0)
            psnrs = np.stack(psnrs, axis=0)
            zplanes = np.stack(zplanes, axis=0)
            maxcounts = np.stack(maxcounts, axis=0)

            if debug:
                yield x, y, psnrs, zplanes, maxcounts
            else:
                yield x, y
