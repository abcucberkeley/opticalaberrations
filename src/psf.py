"""
BSD 3-Clause License
Copyright (c) 2020, Debayan Saha, Martin Weigert, Uwe Schmidt
All rights reserved.
"""

import logging
import sys
import cmath
from functools import lru_cache

from pathlib import Path
from copy import deepcopy
from typing import Optional, Union
import numpy as np

try:
    import cupy as cp
except ImportError as e:
    logging.warning(f"Cupy not supported on your system: {e}")

from skimage.transform import rescale
from line_profiler_pycharm import profile
from astropy.convolution import convolve_fft
from scipy.interpolate import RegularGridInterpolator
import h5py
from tifffile import imread, imwrite
from pyotf.pyotf.otf import HanserPSF
from utils import microns2waves

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def round_to_even(n):
    answer = round(n)
    if not answer % 2:
        return int(answer)
    if abs(answer + 1 - n) < abs(answer - 1 - n):
        return int(answer + 1)
    else:
        return int(answer - 1)


def cart2pol(x, y):
    """Convert cartesian (x, y) to polar (rho, phi_in_radians)"""
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)  # Note role reversal: the "y-coordinate" is 1st parameter, the "x-coordinate" is 2nd.
    return rho, phi


nprect = np.vectorize(cmath.rect)  # stupid cmath.rect can't handle 2D arrays.

@profile
def complex_pupil_array(dx: float,
                        nx: float,
                        wavefront,  # wavefront object
                        na_detection: float,
                        lam_detection: float,   # in microns
                        pupil_mag_file: Path,
                        ) -> np.ndarray:
    """
    Calculate the pupil field at the supplied 2D arrays of kx and ky (usually given after fft shift)
    Args:
        dx: spacing in XY plane in microns
        nx: size of X (and) Y dimension
        wavefront: Wavefront object
        na_detection: numerical aperture
        lam_detection: detection wavelength in microns
        pupil_mag_file: tif file with pupil magnitude

    Returns:
        complex_pupil: 2D array ready for pyotf's apply_pupil. (usually np.fft.fftshift() needed to save as image.)

    """

    pupil_magnitude, rho, theta = pupil_magnitude_array(dx, nx, na_detection, lam_detection, pupil_mag_file)
    pupil_phase = wavefront.phase(rho=rho, theta=theta, normed=True, outside=None)  # 2D array
    pupil_phase = microns2waves(pupil_phase, wavelength=lam_detection) * 2 * np.pi  # Convert wavefront phase to radians

    complex_pupil = nprect(pupil_magnitude, pupil_phase)  # numpy's cartesian to complex
    return complex_pupil


@lru_cache()
def pupil_magnitude_array(dx: float, nx: float, na_detection: float, lam_detection: float, pupil_mag_file: Path):
    """
    Wrapper function to allow for lru cache
    Args:
        dx: spacing in XY plane in microns
        nx: size of X (and) Y dimension
        na_detection: numerical aperture
        lam_detection: detection wavelength in microns
        pupil_mag_file: tif file with pupil magnitude

    Returns:
        pupil_magnitude: 2d float array
        rho: 2D array for rho in polar coordinates in units of pupil radii
        theta: 2D array for polar angle in units of radians
    """

    k = np.fft.fftfreq(nx, dx)
    pyotf_model_kxx, pyotf_model_kyy = np.meshgrid(k, k)
    if pupil_mag_file is not None:
        try:
            pupil_kx, pupil_ky, pupil_mag = load_pupil_file(pupil_mag_file)
            interp = RegularGridInterpolator(
                points=(pupil_ky, pupil_kx),  # ky then kx
                values=pupil_mag,
                bounds_error=False,     # allow for padding with zeros
                fill_value=0,           # allow for padding with zeros
                method='linear'
            )
            pupil_magnitude = interp((pyotf_model_kyy, pyotf_model_kxx))  # ky then kx
        except:
            logger.error(f'Cannot load one of: \n'
                         f'{pupil_mag_file.resolve()}\n'
                         f'{pupil_mag_file.with_suffix("").resolve()}_kxx.tif\n'
                         f'{pupil_mag_file.with_suffix("").resolve()}_kxy.tif')
            pupil_magnitude = np.ones_like(pyotf_model_kxx)     # Use flat pupil
    else:
        pupil_magnitude = np.ones_like(pyotf_model_kxx)
    kr, theta = cart2pol(pyotf_model_kxx, pyotf_model_kyy)  # kx then ky
    diff_limit = na_detection / lam_detection  # see _gen_pupil
    rho = kr / diff_limit
    pupil_magnitude[rho > 1] = 0  # avoid where we've extrapolated past diffraction limit
    return pupil_magnitude, rho, theta


@lru_cache()
def load_pupil_file(pupil_mag_file: Path):
    pupil_mag = imread(pupil_mag_file)
    pupil_kx = imread(f"{pupil_mag_file.with_suffix('')}_kxx.tif")[0, :]  # 1D row
    pupil_ky = imread(f"{pupil_mag_file.with_suffix('')}_kyy.tif")[:, 0]  # 1D col
    return pupil_kx, pupil_ky, pupil_mag


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
            pupil_mag_file: Path = Path(__file__).parent.parent.resolve() / "calibration" / "aang" / "PSF" / "510nm_mag.tif",
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
        self.psf_type = psf_type
        self.pupil_mag_file = pupil_mag_file

        # gpu_support = 'cupy' in sys.modules
        gpu_support = False  # Seems faster

        zrange = -(np.arange(self.Nz) - (self.Nz + 1) // 2) * self.dz  # set z direction to match old psf gen method
        if gpu_support:
            zrange = cp.asarray(zrange)

        self.pyotf_gen = HanserPSF(
            wl=self.lam_detection,
            na=self.na_detection,
            ni=self.n,
            res=self.dx,
            size=self.Nx,
            zres=self.dz,
            zsize=self.Nz,
            vec_corr="none",    # we will overwrite the pupil magnitude: Set to 'none' to match retrieve_phase()
            condition="none",   # we will overwrite the pupil magnitude: Set to 'none' to match retrieve_phase()
            gpu=gpu_support,
            zrange=zrange  # set z direction to match old psf gen method
        )


        if isinstance(lls_excitation_profile, np.ndarray) and lls_excitation_profile.size != 0:
            self.excitation_profile = lls_excitation_profile
            self.lls_excitation_profile = lls_excitation_profile
        else:
            lls_excitation_profile = self.load_excitation_profile()
            if lls_excitation_profile is not None and lls_excitation_profile.shape[0] != psf_shape[0]:

                # High resolution excitation profile
                self.excitation_profile = self.rescale_excitation_profile_voxel_size(
                    lls_excitation_profile=lls_excitation_profile,
                )
                w = psf_shape[0] // 2
                focal_plane_index = self.excitation_profile.shape[0] // 2
                self.lls_excitation_profile = self.excitation_profile[
                    focal_plane_index - w:focal_plane_index - w + psf_shape[0],
                    np.newaxis,
                    np.newaxis
                ]
            else:
                self.excitation_profile = lls_excitation_profile
                self.lls_excitation_profile = lls_excitation_profile

    @profile
    def load_excitation_profile(self):
        # load from file and populate lls_excitation_profile, otherwise self.psf_type is a codeword.
        if isinstance(self.psf_type, Path) or isinstance(self.psf_type, str):
            path = Path(self.psf_type)

            # check if given file exists
            if path.exists():
                with h5py.File(path, 'r') as file:
                    lls_excitation_profile = file.get('DitheredxzPSFCrossSection')[:, 0]

            # check if given filename exists in the lattice dir and try to load it from there instead
            elif Path(f"{Path(__file__).parent.parent.resolve()}/lattice/{path.name}").exists():

                with h5py.File(f"{Path(__file__).parent.parent.resolve()}/lattice/{path.name}", 'r') as file:
                    lls_excitation_profile = file.get('DitheredxzPSFCrossSection')[:, 0]

            else:
                lls_excitation_profile = None

        return lls_excitation_profile

    @profile
    def rescale_excitation_profile_voxel_size(self, lls_excitation_profile):
        lls_profile_dz = 0.1        # media wavelengths per pix, used when generating the .mat LLS file.
        lam_excitation = .488       # microns per wavelength, used when generating the .mat LLS file.
        eff_pixel_size = lam_excitation / self.n * lls_profile_dz   # microns per pix in the .mat LLS file

        return rescale(
            lls_excitation_profile,
            (eff_pixel_size / self.dz),
            order=3,
        )

    @profile
    def defocus_excitation_profile(self, desired_shape: int, offset=None):

        if np.isscalar(offset):
            w = desired_shape // 2
            focal_plane_index = (self.excitation_profile.shape[0] // 2) + np.round(offset / self.dz).astype(int)

            defocused_lls_excitation_profile = self.excitation_profile[
                max(0, focal_plane_index - w):min(focal_plane_index + w, self.excitation_profile.shape[0]),
                np.newaxis,
                np.newaxis
            ]
            return defocused_lls_excitation_profile
        else:
            raise Exception(f"Unknown format for `lls_defocus_offset`: {offset}")

    @profile
    def widefield_psf(self, wavefront) -> np.ndarray:
        """
        Calculate 3D PSF for the given parameters in 'self', and the abberation in the Wavefront object 'wavefront'

        Args:
            wavefront: Wavefront class

        Returns:
            psf: 3D np.ndarray
        """
        # psf = np.abs(self.coherent_psf(wavefront)) ** 2
        # psf = np.array([p / np.sum(p) for p in psf])
        # psf = np.fft.fftshift(psf)
        # psf /= np.max(psf)

        pupil = complex_pupil_array(
            dx=self.dx,
            nx=self.Nx,
            wavefront=wavefront,
            na_detection=self.na_detection,
            lam_detection=self.lam_detection,
            pupil_mag_file=self.pupil_mag_file
        )   # Interpolate pupil

        # imwrite(f"{pupil_mag_file.with_suffix('')}_interp.tif", fftshift(np.abs(pupil)).astype(np.float32))
        # imwrite(f"{pupil_mag_file.with_suffix('')}_phase_interp.tif", fftshift(np.angle(pupil)).astype(np.float32))

        if self.pyotf_gen.gpu:
            pupil = cp.array(pupil)

        psf = np.squeeze(self.pyotf_gen.gen_psf(pupil_base=pupil))  # PSF amplitude
        psf = np.abs(psf) ** 2                                      # PSF intensity
        psf /= np.max(psf)
        if (self.Nz, self.Ny, self.Nx) != psf.shape:
            raise Exception(f'PSF sizes dont match. {self.Nz, self.Ny, self.Nx} vs {psf.shape=}')
        if self.pyotf_gen.gpu:
            psf = cp.asnumpy(psf)
        return psf

    @profile
    def incoherent_psf(self, wavefront, lls_defocus_offset=None):
        """
        Returns the incoherent psf for a given wavefront
           (which is just the squared absolute value of the coherent one)
           The psf is normalized such that the sum intensity on each plane equals one

        Args:
            wavefront: Wavefront object
            lls_defocus_offset: the offset between the excitation and detection focal plan (microns)
        """
        ideal_wavefront = deepcopy(wavefront)
        ideal_wavefront.zernikes = {k: 0. for k, v in ideal_wavefront.zernikes.items()}  # set aberration to 0 for ideal

        # Initialize the total PSF with (aberrated) widefield detection PSF
        _psf = self.widefield_psf(wavefront)

        if self.lls_excitation_profile is not None:

            if lls_defocus_offset is not None and lls_defocus_offset != 0:
                _psf *= self.defocus_excitation_profile(offset=lls_defocus_offset, desired_shape=_psf.shape[0])
            else:
                _psf *= self.lls_excitation_profile

        elif self.psf_type == 'widefield':
            pass

        elif self.psf_type == '2photon':
            # we only have one lamda defined in this class: self.lam_detection. That should already be set with .920
            exc_psf = self.widefield_psf(wavefront)
            _psf = exc_psf ** 2

        elif self.psf_type == 'confocal':
            media_wavelength = 0.1
            lam_excitation = .488
            f = 1  # number of AUs

            exc_psf = self.widefield_psf(wavefront)

            eff_pixel_size = lam_excitation / self.n * media_wavelength       # microns per pixel
            au = 0.61 * (self.lam_detection / self.n) / self.na_detection   # airy radius in microns
            w = _psf.shape[0] // 2
            lateral_extent = round_to_even(2 * w * self.dy / eff_pixel_size)
            axial_extent = 2 * w

            # pix pitch=eff_pixel_size (0.1 media wavelengths)
            Z, Y, X = np.ogrid[
                -axial_extent:axial_extent+1,
                -lateral_extent:lateral_extent+1,
                -lateral_extent:lateral_extent+1
            ]

            circ_func = Y ** 2 + X ** 2 <= (f * au / eff_pixel_size) ** 2
            circ_func = circ_func & (Z == 0)

            circ_func = rescale(
                circ_func.astype(np.float32),
                (1, eff_pixel_size/self.dy, eff_pixel_size/self.dx),
                order=0,
            )   # downscale voxels to go from ~36nm to 100nm voxel size. Fails if circ_func has even number of pixels.

            # clip to the number of voxels we want (e.g. _psf.shape), won't work if _psf.shape is odd.

            focal_plane = np.array(circ_func.shape)//2
            circ_func = circ_func[
                focal_plane[0]-w:focal_plane[0]+w,
                focal_plane[1]-w:focal_plane[1]+w,
                focal_plane[2]-w:focal_plane[2]+w,
            ]
            circ_func /= np.max(circ_func)

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
