import matplotlib
matplotlib.use('Agg')

import logging
import sys
from typing import Any, Union

import matplotlib.pyplot as plt
plt.set_loglevel('error')

import numpy as np
from pathlib import Path
from functools import partial
import multiprocessing as mp
from typing import Iterable
from line_profiler_pycharm import profile
from skimage.draw import line
from typing import Optional

from psf import PsfGenerator3D
from wavefront import Wavefront
from preprocessing import prep_sample
from utils import randuniform, round_to_even, fft, normalize_otf

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
            lls_excitation_profile=None,
            distribution='single',
            embedding_option='spatial_planes',
            mode_weights='pyramid',
            signed=True,
            rotate=False,
            gamma=.75,
            n_modes=15,
            order='ansi',
            batch_size=100,
            psf_shape=(64, 64, 64),
            x_voxel_size=.097,
            y_voxel_size=.097,
            z_voxel_size=.2,
            na_detection=1.0,
            lam_detection=.510,
            refractive_index=1.33,
            pupil_mag_file: Optional[Path] = Path(__file__).parent.parent.resolve() / "calibration" / "aang" / "PSF" / "510nm_mag.tif",
            cpu_workers=-1,
            skip_remove_background_ideal_psf: bool = False,
            use_theoretical_widefield_simulator: bool = False,
    ):
        """
        Args:
            amplitude_ranges: range tuple, array, or wavefront object (in microns)
            psf_type: widefield, 2photon, confocal, or a path to an LLS excitation profile
            distribution: desired distribution for the amplitudes
            gamma: optional exponent of the powerlaw distribution
            signed: optional flag to generate a symmetric (pos/neg) semi-distributions for the given range of amplitudes,
                otherwise just positive amplitudes only
            n_modes: number of zernike modes to describe the aberration
            order: 'noll' or 'ansi', default is ansi
            batch_size: number of samples per batch
            psf_shape: shape of input psf, eg (z, y, x)
            x_voxel_size: (x) lateral sampling rate in microns
            y_voxel_size: (y) lateral sampling rate in microns
            z_voxel_size: (z) axial sampling rate in microns
            na_detection: numerical aperture of detection objective
            lam_detection: wavelength in microns
            refractive_index: refractive index
            cpu_workers: number of CPU threads to use for generating PSFs
            embedding_option: type of fourier embedding to use
        """

        self.n_modes = n_modes
        self.order = order
        self.refractive_index = refractive_index
        self.lam_detection = lam_detection  # wavelength in microns
        self.na_detection = na_detection
        self.batch_size = batch_size
        self.x_voxel_size = x_voxel_size  # desired voxel size
        self.y_voxel_size = y_voxel_size
        self.z_voxel_size = z_voxel_size
        self.voxel_size = (z_voxel_size, y_voxel_size, x_voxel_size)
        self.cpu_workers = cpu_workers
        self.distribution = distribution
        self.mode_weights = mode_weights
        self.gamma = gamma
        self.signed = signed
        self.rotate = rotate
        self.embedding_option = embedding_option
        self.psf_shape = (psf_shape[0], psf_shape[1], psf_shape[2])
        self.amplitude_ranges = amplitude_ranges
        self.psf_type = psf_type
        self.pupil_mag_file = pupil_mag_file
        self.skip_remove_background_ideal_psf = skip_remove_background_ideal_psf
        self.use_theoretical_widefield_simulator = use_theoretical_widefield_simulator

        yumb_axial_support_index, yumb_lateral_support_index = self.calc_max_support_index(
            psf_type='../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat',
            wavelength=.510,
        )

        axial_support_index, lateral_support_index = self.calc_max_support_index(
            psf_type=self.psf_type,
            wavelength=self.lam_detection,
        )

        if axial_support_index != 0:
            self.axial_scalar = 1 if self.psf_type == 'widefield' else yumb_axial_support_index / axial_support_index
        else:
            self.axial_scalar = 1

        if lateral_support_index != 0:
            self.lateral_scalar = yumb_lateral_support_index / lateral_support_index
        else:
            self.lateral_scalar = 1

        self.fov_scaler = (self.axial_scalar, self.lateral_scalar, self.lateral_scalar)
        logger.info(f"FOV scalar: {self.psf_type} => (axial: {self.axial_scalar:.2f}), (lateral: {self.lateral_scalar:.2f})")

        self.psf_fov = tuple(np.array(self.psf_shape) * np.array(self.voxel_size) * np.array(self.fov_scaler))
        self.adjusted_psf_shape = tuple(round_to_even(i) for i in np.array(self.psf_shape) * np.array(self.fov_scaler))  # same physical fov

        self.psfgen = PsfGenerator3D(
            psf_shape=self.adjusted_psf_shape,
            units=self.voxel_size,
            lam_detection=self.lam_detection,
            n=self.refractive_index,
            na_detection=self.na_detection,
            psf_type=self.psf_type,
            lls_excitation_profile=lls_excitation_profile,
            pupil_mag_file=self.pupil_mag_file,
        )
        self.lls_excitation_profile = self.psfgen.lls_excitation_profile

        # ideal psf (theoretical, no noise)
        self.ipsf = self.theoretical_psf(normed=True)
        self.na_mask = self.create_na_mask(ipsf=self.ipsf)

        # preprocess ideal PSF with DoG filter
        if not skip_remove_background_ideal_psf:
            self.ipsf = prep_sample(
                self.ipsf,
                sample_voxel_size=self.voxel_size,
                model_fov=self.psf_fov,
                remove_background=True,
                normalize=True,
                min_psnr=0,
                na_mask=self.na_mask
            )

        self.iotf = fft(self.ipsf, padsize=None)
        self.iotf = normalize_otf(self.iotf)
        self.iotf *= self.na_mask

    @profile
    def calc_max_support_index(
        self,
        psf_type: str = '../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat',
        wavelength: float = .510,
    ):
        """
        Find how much OTF support there is for this psf_type.  This is to be used only in a ratio to find how to
        scale the FOVs for non-lightsheets.  This will make the embeddings of non-lightsheets fill a similar space
        in their 64 x 64 images, so the model can generalize.

        For axial support: Start from 2/3rd along kx, then go up (in kz) and measure how far to hit the na_mask.
        For lateral support: March from midpoint to (0, 0), and find the distance along that line hits the na_mask.
        """
        zm, ym, xm = (i // 2 for i in self.psf_shape)

        vxz = line(r0=zm, c0=2 * xm // 3, r1=0, c1=2 * xm // 3)
        vxy = line(r0=ym, c0=xm, r1=0, c1=0)

        phi = Wavefront(
            amplitudes=np.zeros(self.n_modes),
            order=self.order,
            distribution=self.distribution,
            mode_weights=self.mode_weights,
            modes=self.n_modes,
            gamma=self.gamma,
            lam_detection=wavelength
        )

        gen = PsfGenerator3D(
            psf_shape=self.psf_shape,
            units=self.voxel_size,
            lam_detection=wavelength,
            n=self.refractive_index,
            na_detection=self.na_detection,
            psf_type=psf_type,
            pupil_mag_file=self.pupil_mag_file,
        )

        ipsf = gen.incoherent_psf(phi)
        ipsf /= np.nanmax(ipsf)

        iotf = np.abs(fft(ipsf, padsize=None))
        iotf = normalize_otf(iotf)
        iotf *= self.create_na_mask(ipsf=ipsf)

        axial_support_index = next((i for i, z in enumerate(iotf[vxz[0], ym, vxz[1]]) if z == 0), 0)    # march until 0
        lateral_support_index = next((i for i, x in enumerate(iotf[zm, vxy[0], vxy[1]]) if x == 0), 0)  # march until 0
        return axial_support_index, lateral_support_index

    @profile
    def update_ideal_psf_with_empirical(
            self,
            ideal_empirical_preprocessed_psf: np.ndarray,
    ):
        logger.info(f"Updating ideal PSF with empirical PSF")

        # PSF has been pre-processed already
        self.ipsf = ideal_empirical_preprocessed_psf

        self.iotf = fft(self.ipsf, padsize=None)
        self.iotf = normalize_otf(self.iotf)
        self.iotf *= self.na_mask

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
    def create_na_mask(self, ipsf: np.ndarray, threshold: Optional[float] = 8e-3) -> np.ndarray:
        """
        OTF Mask by binary thresholding ideal theoretical OTF

        Args:
            threshold: where to threshold after normalizing to the OTF max

        Returns:
            3D array where ==1 inside NA_Mask, ==0 outside NA mask

        """

        mask = np.abs(fft(ipsf))
        mask /= np.nanmax(mask)

        if threshold is None:
            threshold = np.nanpercentile(mask.flatten(), 65)

        mask = np.where(mask < threshold, mask, 1.)
        mask = np.where(mask >= threshold, mask, 0.)
        return mask.astype(np.float32)

    @profile
    def single_psf(
        self,
        phi: Any = None,
        normed: bool = True,
        meta: bool = False,
        no_phase: bool = False,
        lls_defocus_offset: Any = None
    ):
        """
        Args:
            phi: Wavefront object or array of amplitudes of Zernike polynomials (or path to array)
            normed: a toggle to normalize PSF by the psf max
            meta: return extra variables for debugging
            no_phase: used only when meta=true.
            lls_defocus_offset: optional shift of the excitation and detection focal plan (microns)
        """
        if not hasattr(phi, 'order'):
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

        if isinstance(lls_defocus_offset, tuple):
            lls_defocus_offset = randuniform(lls_defocus_offset)

            # if phi.peak2valley(na=1.0) <= 1.:
            #     lls_defocus_offset = randuniform(lls_defocus_offset)
            # else:
            #     lls_defocus_offset = 0.

        psf = self.psfgen.incoherent_psf(
            phi,
            lls_defocus_offset=lls_defocus_offset,
            use_theoretical_widefield_simulator=self.use_theoretical_widefield_simulator
        )

        if normed:
            psf /= np.max(psf)

        if meta:
            if no_phase:
                phi.amplitudes = np.abs(phi.amplitudes)

            return psf, phi.amplitudes, lls_defocus_offset
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
        )

        while True:
            inputs, amplitudes, lls_defocus_offsets = zip(
                *self.batch(gen, [self.amplitude_ranges]*self.batch_size)
            )

            x = np.expand_dims(np.stack(inputs, axis=0), -1)
            y = np.stack(amplitudes, axis=0)
            lls_defocus_offsets = np.stack(lls_defocus_offsets, axis=0)

            if debug:
                yield x, y, lls_defocus_offsets
            else:
                yield x, y
