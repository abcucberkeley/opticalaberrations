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

    def masked_phase_array(self, phi, normed=True):
        """
        Returns masked Zernike polynomial for back focal plane, masked according to the setup

        Args:
            phi: Zernike/ZernikeWavefront object
            normed: boolean, multiplied by normalization factor, eg True
        """
        return self.kmask2 * phi.phase(self.krho, self.kphi, normed=normed, outside=None)

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

    def incoherent_psf(self, phi):
        """
        Returns the incoherent psf for a given wavefront phi
           (which is just the squared absolute value of the coherent one)
           The psf is normalized such that the sum intensity on each plane equals one

        Args:
            phi: Zernike/ZernikeWavefront object
        """
        _psf = np.abs(self.coherent_psf(phi)) ** 2
        _psf = np.array([p / np.sum(p) for p in _psf])
        _psf = np.fft.fftshift(_psf)
        _psf /= np.max(_psf)

        if self.psf_type == 'widefield':
            return _psf
        elif self.psf_type == 'confocal':
            _psf = _psf**2
            _psf /= np.max(_psf)
            return _psf
        else:
            if isinstance(self.psf_type, str) or isinstance(self.psf_type, Path):
                with h5py.File(self.psf_type, 'r') as file:
                    lattice = file.get('DitheredxzPSFCrossSection')[:, 0]
            else:
                lattice = self.psf_type

            lattice = rescale(
                lattice,
                (.0367/self.dz),
                order=3,
                anti_aliasing=True,
            )
            w = _psf.shape[0]//2
            center = lattice.shape[0]//2
            lattice = lattice[center-w:center+w, np.newaxis, np.newaxis]
            _psf = _psf * lattice
            return _psf
