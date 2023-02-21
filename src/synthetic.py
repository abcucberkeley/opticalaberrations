import matplotlib

matplotlib.use('Agg')

import logging
import sys
from typing import Any, Union

import numpy as np
from pathlib import Path
from skimage import transform
from functools import partial
import multiprocessing as mp
from typing import Iterable
from line_profiler_pycharm import profile
from tifffile import TiffFile

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
    def theoretical_psf(self, normed: bool = True):
        """Generates an unaberrated PSF of the "desired" PSF shape and voxel size, centered.

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
    def fft(self, inputs, padsize=None):
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
    def ifft(self, otf):
        psf = np.fft.fftshift(otf)
        psf = np.fft.ifftn(psf)
        psf = np.abs(np.fft.ifftshift(psf))
        return psf


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
    def single_psf(
        self,
        phi: Any = None,
        normed: bool = True,
        noise: bool = False,
        meta: bool = False,
        no_phase: bool = False,
        snr_post_aberration: bool = True,
        lls_defocus_offset: Any = None
    ):
        """
        Args:
            phi: wavefront object
            normed: a toggle to normalize PSF
            noise: a toggle to add noise
            meta: return extra variables for debugging
            no_phase: used only when meta=true.
            snr_post_aberration: increase photons in aberrated psf to match snr of ideal psf
            lls_defocus_offset: optional shift of the excitation and detection focal plan (microns)
        """

        if isinstance(lls_defocus_offset, tuple):
            lls_defocus_offset = self._randuniform(lls_defocus_offset)

        if not isinstance(phi, Wavefront):
            phi = Wavefront(
                phi,
                order=self.order,
                distribution=self.distribution,
                mode_weights=self.mode_weights,
                modes=self.n_modes,
                gamma=self.gamma,
                signed=self.signed,
                rotate=self.rotate,
                lam_detection=self.lam_detection,
            )

        psf = self.psfgen.incoherent_psf(phi, lls_defocus_offset=lls_defocus_offset)
        snr = self._randuniform(self.snr)

        if snr_post_aberration:
            psf *= snr ** 2
        else:
            total_counts_ideal = np.sum(self.ipsf)  # peak value of ideal psf is 1, self.ipsf = 1
            total_counts = total_counts_ideal * snr ** 2  # total photons of ideal psf with desired SNR
            total_counts_abr = np.sum(psf)  # total photons in aberrated psf

            # scale aberrated psf to have the same number of photons as ideal psf
            # (e.g. aberration doesn't destroy/create light)
            psf *= total_counts / total_counts_abr

        if noise:
            rand_noise = self._random_noise(
                image=psf,
                mean=self.mean_background_noise,
                sigma=self.sigma_background_noise,
            )
            psf += rand_noise

        psnr = np.sqrt(np.max(psf))  # peak snr will drop as aberration smooshes psf
        maxcount = np.max(psf)

        if normed:
            psf /= np.max(psf)

        if meta:
            if no_phase:
                phi.amplitudes = np.abs(phi.amplitudes)

            return psf, phi.amplitudes, psnr, maxcount, lls_defocus_offset
        else:
            return psf

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

    def generator(self, debug=False):
        gen = partial(
            self.single_psf,
            meta=True,
            normed=True,
            noise=True,
        )

        while True:
            inputs, amplitudes, psnrs, maxcounts, lls_defocus_offsets = zip(
                *self.batch(gen, [self.amplitude_ranges]*self.batch_size)
            )

            x = np.expand_dims(np.stack(inputs, axis=0), -1)
            y = np.stack(amplitudes, axis=0)
            psnrs = np.stack(psnrs, axis=0)
            maxcounts = np.stack(maxcounts, axis=0)
            lls_defocus_offsets = np.stack(lls_defocus_offsets, axis=0)

            if debug:
                yield x, y, psnrs, maxcounts, lls_defocus_offsets
            else:
                yield x, y
