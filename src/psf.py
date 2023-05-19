"""
BSD 3-Clause License
Copyright (c) 2020, Debayan Saha, Martin Weigert, Uwe Schmidt
All rights reserved.
"""

import logging
import sys
from pathlib import Path

import numpy as np
from skimage.transform import rescale
from line_profiler_pycharm import profile
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
            print(f'loading my psf {self.psf_type}')
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
    def coherent_psf(self, phi):
        """
        Returns the coherent psf for a given wavefront phi

        Args:
            phi: Zernike/ZernikeWavefront object
        """
        phi = self.masked_phase_array(phi)
        ku = self.kbase * np.exp(2.j * np.pi * phi / self.lam_detection)
        res = np.fft.ifftn(ku, axes=(1, 2))
        return np.fft.fftshift(res, axes=(0,))

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
        _psf = np.abs(self.coherent_psf(phi)) ** 2
        _psf = np.array([p / np.sum(p) for p in _psf])
        _psf = np.fft.fftshift(_psf)
        _psf /= np.max(_psf)

        if self.psf_type == 'widefield':
            pass
        elif self.psf_type == 'confocal':
            _psf = _psf**2
            _psf /= np.max(_psf)
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

        return _psf.astype(np.float32)
