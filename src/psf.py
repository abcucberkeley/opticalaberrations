"""
BSD 3-Clause License
Copyright (c) 2020, Debayan Saha, Martin Weigert, Uwe Schmidt
All rights reserved.
"""

import logging
import sys
from pathlib import Path
from copy import deepcopy
from typing import Optional, Union

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

    def __init__(
            self,
            psf_shape: tuple,
            units: tuple,
            lam_detection: float,
            n: float,
            na_detection: float,
            psf_type: Union[str, Path],
            lls_excitation_profile: Optional[np.ndarray] = None
    ):
        """
        Args:
            psf_shape: tuple, psf shape as (z,y,x), e.g. (64,64,64)
            units: tuple, voxel size in microns, e.g. (0.1,0.1,0.1)
            lam_detection: scalar, wavelength in microns, e.g. 0.5
            n: scalar, refractive index, e.g. 1.33
            na_detection: scalar, numerical aperture of detection objective, e.g. 1.1
            psf_type: 'widefield', '2photon', 'confocal', or a Path to an LLS excitation .mat profile
            lls_excitation_profile: None when psf_type=[widefield, 2photon, or confocal],
                otherwise an array storage for an LLS excitation profile loaded from disk
        """

        psf_shape = tuple(psf_shape)
        units = tuple(units)

        self.n = n  # refractive index, e.g. 1.33
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

        if isinstance(lls_excitation_profile, np.ndarray) and lls_excitation_profile.size != 0:
            self.lls_excitation_profile = lls_excitation_profile
        else:
            # load from file and populate lls_excitation_profile, otherwise self.psf_type is a codeword.
            if isinstance(self.psf_type, Path) or isinstance(self.psf_type, str):
                path = Path(self.psf_type)

                # check if given file exists
                if path.exists():
                    with h5py.File(path, 'r') as file:
                        self.lls_excitation_profile = file.get('DitheredxzPSFCrossSection')[:, 0]

                # check if given filename exists in the lattice dir and try to load it from there instead
                elif Path(f"{Path(__file__).parent.parent.resolve()}/lattice/{path.name}").exists():

                    with h5py.File(f"{Path(__file__).parent.parent.resolve()}/lattice/{path.name}", 'r') as file:
                        self.lls_excitation_profile = file.get('DitheredxzPSFCrossSection')[:, 0]

                else:
                    self.lls_excitation_profile = None

        if self.lls_excitation_profile is not None and self.lls_excitation_profile.shape[0] != psf_shape[0]:
            lls_profile_dz = 0.1        # media wavelengths per pix, used when generating the .mat LLS file.
            lam_excitation = .488       # microns per wavelength, used when generating the .mat LLS file.
            eff_pixel_size = lam_excitation / self.n * lls_profile_dz   # microns per pix in the .mat LLS file

            self.lls_excitation_profile = rescale(
                self.lls_excitation_profile,
                (eff_pixel_size / self.dz),
                order=3,
            )

            w = psf_shape[0] // 2
            focal_plane_index = self.lls_excitation_profile.shape[0] // 2
            self.lls_excitation_profile = self.lls_excitation_profile[
                focal_plane_index - w:focal_plane_index + w,
                np.newaxis,
                np.newaxis
            ]

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

    def widefield_psf(self, phi):
        psf = np.abs(self.coherent_psf(phi)) ** 2
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
        ideal_phi.zernikes = {k: 0. for k, v in ideal_phi.zernikes.items()}  # set aberration to zero for ideal_psf

        # initalize the total PSF with (aberrated) widefield detection PSF
        _psf = self.widefield_psf(phi)

        if self.lls_excitation_profile is not None:

            if lls_defocus_offset is not None:
                if np.isscalar(lls_defocus_offset):
                    w = self.lls_excitation_profile.shape[0] // 2
                    focal_plane_index = w + np.round(lls_defocus_offset / self.dz).astype(int)
                    defocused_lls_excitation_profile = self.lls_excitation_profile[
                        max(0, focal_plane_index - w):min(focal_plane_index + w, self.lls_excitation_profile.shape[0])
                    ]
                    _psf *= defocused_lls_excitation_profile
                else:
                    raise Exception(f"Unknown format for `lls_defocus_offset`: {lls_defocus_offset}")
            else:
                _psf *= self.lls_excitation_profile

        elif self.psf_type == 'widefield':
            pass

        elif self.psf_type == '2photon':
            # we only have one lamda defined in this class: self.lam_detection. That should already be set with .920
            exc_psf = self.widefield_psf(phi)
            _psf = exc_psf ** 2

        elif self.psf_type == 'confocal':
            lls_profile_dz = 0.1    # (0.1 media wavelengths)
            lam_excitation = .488
            f = 1  # number of AUs

            exc_psf = self.widefield_psf(phi)

            eff_pixel_size = lam_excitation / self.n * lls_profile_dz       # microns per pixel
            au = 0.61 * (self.lam_detection / self.n) / self.na_detection   # airy radius in microns
            Z, Y, X = np.ogrid[-200:201, -200:201, -200:201]  # pix pitch=eff_pixel_size (0.1 media wavelengths)

            circ_func = Y ** 2 + X ** 2 <= (f * au / eff_pixel_size) ** 2
            circ_func = circ_func & (Z == 0)

            circ_func = rescale(
                circ_func.astype(np.float32),
                (eff_pixel_size/self.dz, eff_pixel_size/self.dy, eff_pixel_size/self.dx),
                order=0,
            )   # downscale voxels to go from ~36nm to 100nm voxel size. Fails if circ_func has even number of pixels.

            # clip to the number of voxels we want (e.g. _psf.shape), won't work if _psf.shape is odd.
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
            raise Exception(f"Unknown PSF type: {self.psf_type}")

        _psf /= np.max(_psf)
        return _psf.astype(np.float32)  # return total PSF
