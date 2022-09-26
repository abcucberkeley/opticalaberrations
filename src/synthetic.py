import matplotlib
matplotlib.use('TkAgg')

import logging
import sys
from typing import Any

import numpy as np
from skimage import transform
from skimage.filters import gaussian
from functools import partial
import multiprocessing as mp
from typing import Iterable
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tifffile import imsave
from tqdm import trange
from skimage.filters import window
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
        dtype='widefield',
        distribution='dirichlet',
        bimodal=False,
        gamma=1.5,
        n_modes=15,
        order='ansi',
        batch_size=1,
        psf_shape=(64, 64, 64),
        x_voxel_size=.15,
        y_voxel_size=.15,
        z_voxel_size=.6,
        na_detection=1.0,
        lam_detection=.605,
        refractive_index=1.33,
        snr=(10, 50),
        mean_background_noise=100,
        sigma_background_noise=4,
        max_jitter=1,
        cpu_workers=1
    ):
        """
        Args:
            amplitude_ranges: range tuple, array, or wavefront object (in microns)
            dtype: widefield or confocal
            distribution: desired distribution for the amplitudes
            gamma: optional exponent of the powerlaw distribution
            bimodal: optional flag to generate a symmetric (pos/neg) semi-distributions for the given range of amplitudes
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
        self.dtype = dtype
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
        self.gamma = gamma
        self.bimodal = bimodal

        self.psf_shape = (psf_shape[0], psf_shape[1], psf_shape[2])
        self.theoretical_psf_shape = (2 * psf_shape[0], 2 * psf_shape[1], 2 * psf_shape[2])
        self.amplitude_ranges = amplitude_ranges

        self.psfgen = PsfGenerator3D(
            psf_shape=self.theoretical_psf_shape,
            units=self.voxel_size,
            lam_detection=self.lam_detection,
            n=self.refractive_index,
            na_detection=self.na_detection,
            dtype=dtype
        )

        self.ipsf = self.theoretical_psf(normed=True, noise=False)
        self.iotf, self.iphase = self.fft(self.ipsf, padsize=None)

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

    def fft(self, inputs, padsize=None, gaussian_filter=None):

        if padsize is not None:
            shape = inputs.shape[1]
            size = shape * (padsize / shape)
            pad = int((size - shape)//2)
            inputs = np.pad(inputs, ((pad, pad), (pad, pad), (pad, pad)), 'constant', constant_values=0)

        otf = np.fft.fftn(inputs)
        otf = np.fft.fftshift(otf)

        phi = np.angle(otf)
        phi = np.unwrap(phi)
        alpha = np.abs(otf)

        if gaussian_filter is not None:
            alpha = gaussian(alpha, gaussian_filter)
            phi = gaussian(phi, gaussian_filter)

        alpha /= np.nanpercentile(alpha, 99.99)
        alpha[alpha > 1] = 1
        alpha = np.nan_to_num(alpha, nan=0)

        phi /= np.nanpercentile(phi, 99.99)
        phi[phi > 1] = 1
        phi[phi < -1] = -1
        phi = np.nan_to_num(phi, nan=0)

        return alpha, phi

    def na_mask(self):
        mask = self.iotf
        if self.dtype == 'widefield':
            threshold = np.nanpercentile(mask.flatten(), 55)
        else:
            threshold = np.nanpercentile(mask.flatten(), 80)

        # logger.info(f'NA-threshold: {threshold}')
        mask = np.where(mask < threshold, mask, 1.)
        mask = np.where(mask >= threshold, mask, 0.)
        return mask

    def plot_embeddings(
        self,
        psf: np.array,
        emb: np.array,
        save_path: Any,
        no_phase: bool = False,
        log10: bool = False,
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

        if log10:
            vmin, vmax, vcenter, step = -2, 2, 0, .1
        else:
            vmin, vmax, vcenter, step = 0, 2, 1, .1

        highcmap = plt.get_cmap('YlOrRd', 256)
        lowcmap = plt.get_cmap('terrain', 256)
        low = np.linspace(0, 1 - step, int(abs(vcenter - vmin) / step))
        high = np.linspace(0, 1 + step, int(abs(vcenter - vmax) / step))
        cmap = np.vstack((lowcmap(low), [1, 1, 1, 1], highcmap(high)))
        cmap = mcolors.ListedColormap(cmap)

        if no_phase:
            fig, axes = plt.subplots(2, 3, figsize=(8, 11))
        else:
            fig, axes = plt.subplots(3, 3, figsize=(8, 11))

        m = axes[0, 0].imshow(np.max(psf, axis=0), cmap='hot', vmin=0, vmax=1)
        axes[0, 1].imshow(np.max(psf, axis=1), cmap='hot', vmin=0, vmax=1)
        axes[0, 2].imshow(np.max(psf, axis=2).T, cmap='hot', vmin=0, vmax=1)
        cax = inset_axes(axes[0, 2], width="10%", height="100%", loc='center right', borderpad=-3)
        cb = plt.colorbar(m, cax=cax)
        cax.yaxis.set_label_position("right")
        cax.set_ylabel('Input (maxproj)')

        m = axes[1, 0].imshow(emb[0], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1, 1].imshow(emb[1], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1, 2].imshow(emb[2].T, cmap=cmap, vmin=vmin, vmax=vmax)
        cax = inset_axes(axes[1, 2], width="10%", height="100%", loc='center right', borderpad=-3)
        cb = plt.colorbar(m, cax=cax)
        cax.yaxis.set_label_position("right")
        cax.set_ylabel(r'Embedding ($\alpha$)')

        if not no_phase:
            m = axes[-1, 0].imshow(emb[3], cmap='coolwarm', vmin=-.5, vmax=.5)
            axes[-1, 1].imshow(emb[4], cmap='coolwarm', vmin=-.5, vmax=.5)
            axes[-1, 2].imshow(emb[5].T, cmap='coolwarm', vmin=-.5, vmax=.5)
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
        padsize: Any = None,
        plot: Any = None,
        log10: bool = False,
        principle_planes: bool = True,
        gamma: float = 1.,
        no_phase: bool = False,
    ):
        if psf.ndim == 4:
            psf = np.squeeze(psf)

        amp, phase = self.fft(psf, padsize=padsize)

        if psf.shape != self.psf_shape:
            amp = transform.rescale(
                amp,
                (
                    self.psf_shape[0] / amp.shape[0],
                    self.psf_shape[1] / amp.shape[1],
                    self.psf_shape[2] / amp.shape[2],
                ),
                order=3,
                anti_aliasing=True,
            )

            phase = transform.rescale(
                phase,
                (
                    self.psf_shape[0] / phase.shape[0],
                    self.psf_shape[1] / phase.shape[1],
                    self.psf_shape[2] / phase.shape[2],
                ),
                order=3,
                anti_aliasing=True,
            )

        if ratio:
            amp = amp ** gamma
            amp /= self.iotf
            amp = np.nan_to_num(amp, nan=0)
            # phase /= self.iphase
            # phase = np.nan_to_num(phase, nan=0)

        if na_mask:
            mask = self.na_mask()
            amp *= mask
            phase *= mask

        if log10:
            amp = np.log10(amp)
            amp = np.nan_to_num(amp, nan=0, posinf=0, neginf=0)

            phase = np.log10(phase)
            phase = np.nan_to_num(phase, nan=0, posinf=0, neginf=0)

        if principle_planes:
            if no_phase:
                emb = np.stack([
                    amp[amp.shape[0] // 2, :, :],
                    amp[:, amp.shape[1] // 2, :],
                    amp[:, :, amp.shape[2] // 2],
                ], axis=0)
            else:
                emb = np.stack([
                    amp[amp.shape[0] // 2, :, :],
                    amp[:, amp.shape[1] // 2, :],
                    amp[:, :, amp.shape[2] // 2],
                    phase[phase.shape[0] // 2, :, :],
                    phase[:, phase.shape[1] // 2, :],
                    phase[:, :, phase.shape[2] // 2],
                ], axis=0)
        else:
            emb = np.stack([amp, phase], axis=0)
            imsave(f"{plot}_alpha.tif", amp)
            imsave(f"{plot}_phi.tif", phase)

        if plot is not None and principle_planes:
            self.plot_embeddings(psf=psf, emb=emb, save_path=plot, log10=log10, no_phase=no_phase)

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
                modes=self.n_modes,
                gamma=self.gamma,
                bimodal=self.bimodal,
                lam_detection=self.lam_detection
            )

        psf = self.psfgen.incoherent_psf(phi) * snr * self.mean_background_noise

        rand_noise = self._random_noise(
            image=psf,
            mean=self.mean_background_noise,
            sigma=self.sigma_background_noise,
        )
        noisy_psf = rand_noise + psf if noise else psf
        psnr = (np.max(psf) / np.mean(rand_noise))
        maxcount = np.max(noisy_psf)

        noisy_psf, zplanes = self._axial_resample(
            noisy_psf,
            axial_voxel_size=z_voxel_size,
            lateral_voxel_size=max([x_voxel_size, y_voxel_size]),
            zplanes=zplanes
        )

        if augmentation:
            noisy_psf = self._crop(noisy_psf, voxel_size=voxel_size, jitter=True)
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
            no_phase=no_phase
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
