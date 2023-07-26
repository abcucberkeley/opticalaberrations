"""
BSD 3-Clause License
Copyright (c) 2020, Debayan Saha, Martin Weigert, Uwe Schmidt
All rights reserved.
"""

import logging
import sys
from pathlib import Path
from copy import deepcopy

import numpy as np
from skimage.transform import rescale
from line_profiler_pycharm import profile
from astropy.convolution import convolve_fft
import h5py

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PsfGenerator3D:
    """
    3D PSF generator, courtesy of Martin Weigert (https://github.com/maweigert)
    """

    def __init__(self, psf_shape, units, lam_detection, n, na_detection, psf_type='widefield'):
        """
        Args:
            psf_shape: tuple, psf shape as (z,y,x), e.g. (64,64,64)
            units: tuple, voxel size in microns, e.g. (0.1,0.1,0.1)
            lam_detection: scalar, wavelength in microns, e.g. 0.5
            n: scalar, refractive index, eg 1.33
            na_detection: scalar, numerical aperture of detection objective, eg 1.1
            psf_type: widefield or confocal
        """

        psf_shape = tuple(psf_shape)
        units = tuple(units)

        self.n = n
        self.Nz, self.Ny, self.Nx = psf_shape
        self.dz, self.dy, self.dx = units

        # self.dz = np.random.uniform(
        #     np.min([self.dx, self.dy]),
        #     self.dz
        # )

        self.na_detection = na_detection
        self.lam_detection = lam_detection
        self.kcut = 1. * self.na_detection / self.lam_detection

        kx = np.fft.fftfreq(self.Nx, self.dx)
        ky = np.fft.fftfreq(self.Ny, self.dy)

        idx = np.arange(self.Nz) - self.Nz // 2
        kz = self.dz * idx
        self.theoretical_psf(kx=kx, ky=ky, kz=kz)
        self.psf_type = psf_type

        if (isinstance(self.psf_type, Path) or isinstance(self.psf_type, str)) and Path(self.psf_type).exists():
            with h5py.File(self.psf_type, 'r') as file:
                self.psf_type = file.get('DitheredxzPSFCrossSection')[:, 0]

    @profile
    def theoretical_psf(self, kx, ky, kz):
        KZ3, KY3, KX3 = np.meshgrid(kz, ky, kx, indexing="ij")
        KR3 = np.sqrt(KX3 ** 2 + KY3 ** 2)

        # the cutoff in fourier domain
        kmask3 = (KR3 <= self.kcut)
        H = np.sqrt(1. * self.n ** 2 - KR3 ** 2 * self.lam_detection ** 2)

        out_ind = np.isnan(H)
        kprop = np.exp(-2.j * np.pi * KZ3 / self.lam_detection * H)
        kprop[out_ind] = 0.

        KY2, KX2 = np.meshgrid(ky, kx, indexing="ij")
        KR2 = np.hypot(KX2, KY2)

        self.kbase = kmask3 * kprop
        self.krho = KR2 / self.kcut
        self.kphi = np.arctan2(KY2, KX2)
        self.kmask2 = (KR2 <= self.kcut)

    @profile
    def masked_phase_array(self, phi, normed=True):
        """
        Returns masked Zernike polynomial for back focal plane, masked according to the setup

        Args:
            phi: Zernike/ZernikeWavefront object
            normed: boolean, multiplied by normalization factor, eg True
        """
        return self.kmask2 * phi.phase(self.krho, self.kphi, normed=normed, outside=None)

    @profile
    def coherent_psf(self, phi, lam_excitation=None):
        """
        Returns the coherent psf for a given wavefront phi

        Args:
            phi: Zernike/ZernikeWavefront object
        """
        lam = self.lam_detection if lam_excitation is None else lam_excitation
        phi = self.masked_phase_array(phi)
        ku = self.kbase * np.exp(2.j * np.pi * phi / lam)
        res = np.fft.ifftn(ku, axes=(1, 2))
        return np.fft.fftshift(res, axes=(0,))

    def widefield_psf(self, phi, lam_excitation=None):
        psf = np.abs(self.coherent_psf(phi, lam_excitation=lam_excitation)) ** 2
        psf = np.array([p / np.sum(p) for p in psf])
        psf = np.fft.fftshift(psf)
        psf /= np.max(psf)
        return psf

    @profile
    def incoherent_psf(self, phi, lls_defocus_offset=None):
        """
        Returns the incoherent psf for a given wavefront phi
           (which is just the squared absolute value of the coherent one)
           The psf is normalized such that the sum intensity on each plane equals one

        Args:
            phi: Zernike/ZernikeWavefront object
            lls_defocus_offset: the offset between the excitation and detection focal plan (microns)
        """
        ideal_phi = deepcopy(phi)
        ideal_phi.zernikes = {k: 0. for k, v in ideal_phi.zernikes.items()}

        _psf = self.widefield_psf(phi)

        if self.psf_type == 'widefield':
            pass

        elif self.psf_type == '2photon':
            exc_psf = self.widefield_psf(ideal_phi, lam_excitation=.920)
            _psf = (exc_psf ** 2) * _psf

        elif self.psf_type == 'confocal':
            lam_excitation = .488
            exc_psf = self.widefield_psf(ideal_phi, lam_excitation=lam_excitation)

            eff_pixel_size = lam_excitation / self.n * 0.1
            au = 0.61 * (self.lam_detection / self.n) / self.na_detection
            Z, Y, X = np.ogrid[-200:201, -200:201, -200:201]

            f = 1  # number of AUs
            circ_func = Y ** 2 + X ** 2 <= (f * au / eff_pixel_size) ** 2
            circ_func = circ_func & (Z == 0)

            circ_func = rescale(
                circ_func.astype(np.float32),
                (eff_pixel_size/self.dz, eff_pixel_size/self.dy, eff_pixel_size/self.dx),
                order=0,
            )

            w = _psf.shape[0]//2
            focal_plane = np.array(circ_func.shape)//2
            circ_func = circ_func[
                focal_plane[0]-w:focal_plane[0]+w,
                focal_plane[1]-w:focal_plane[1]+w,
                focal_plane[2]-w:focal_plane[2]+w,
            ]

            det_psf = convolve_fft(
                _psf,
                circ_func,
                allow_huge=True,
                normalize_kernel=False,
                nan_treatment='fill',
                fill_value=0
            ).astype(np.float32)

            det_psf[det_psf < 0] = 0  # clip negative small values
            _psf = exc_psf * det_psf

        else:
            lattice_profile = self.psf_type

            if lls_defocus_offset is not None:
                if np.isscalar(lls_defocus_offset):
                    w = lattice_profile.shape[0] // 2
                    focal_plane_index = w + np.round(lls_defocus_offset / .0367).astype(int)
                    lattice_profile = lattice_profile[
                        max(0, focal_plane_index-w):min(focal_plane_index+w, lattice_profile.shape[0])
                    ]
                else:
                    logger.error(f"Unknown format for `lls_defocus_offset`: {lls_defocus_offset}")

            lattice_profile = rescale(
                lattice_profile,
                (.0367/self.dz),
                order=3,
            )

            w = _psf.shape[0]//2
            focal_plane_index = lattice_profile.shape[0] // 2
            lattice_profile = lattice_profile[
                focal_plane_index-w:focal_plane_index+w,
                np.newaxis,
                np.newaxis
            ]
            _psf *= lattice_profile

        _psf /= np.max(_psf)
        return _psf.astype(np.float32)
