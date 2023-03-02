import logging
import sys

import numpy as np
from zernike import Zernike, rho_theta, nm_polynomial
from pathlib import Path
from tifffile import imread
from typing import Union

from distributions import uniform_weights, decayed_weights, pyramid_weights, pick_modes
from distributions import single, bimodal, multinomial, powerlaw, dirichlet, uniform
from functools import lru_cache

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Wavefront:
    """
        Encapsulates the wavefront defined by Zernike polynomials

        amplitudes: Amplitudes of Zernike polynomials
        order: Zernike nomenclature, 'noll' or 'ansi', default is 'ansi'
        lam_detection: wavelength in microns
        modes: number of modes to be selected from.
        mode_weights: how likely to pick modes to be selected. 'pyramid', 'decay', or all else uniform (uniform=default)
        distribution: amplitude distribution 'single', 'powerlaw', 'dirichlet', or 'mixed'=random choice of distribution
        rotate: optional toggle to assign a random rotation between any given mode and its twin
        signed: optional toggle to get positive amplitudes only or a mixture
    """

    def __init__(
        self,
        amplitudes: Union[None, np.ndarray, list, tuple, dict],
        order: str = 'ansi',
        modes: int = 55,
        lam_detection: float = .510,
        distribution: str = 'single',
        mode_weights: str = 'uniform',
        gamma: float = .75,
        signed: bool = True,
        rotate: bool = False,

    ):
        self.ranges = amplitudes
        self.order = order
        self.lam_detection = lam_detection
        self.prefixed = [0, 1, 2, 4] if order == 'ansi' else [0, 1, 2, 3]
        self.modes = modes
        self.gamma = gamma
        self.signed = signed
        self.rotate = rotate

        # Provide the probabilities (aka weights) over the desired range of modes.
        # Don't include "prefixed": piston,tip,tilt,defocus.
        if mode_weights == 'pyramid':
            self.mode_weights = pyramid_weights(
                num_modes=self.modes - len(self.prefixed),
                order=self.order,
                prefixed=self.prefixed,
                starting_ansi_index=15,
            )
        elif mode_weights == 'decay':
            self.mode_weights = decayed_weights(num_modes=self.modes - len(self.prefixed))
        else:
            self.mode_weights = uniform_weights(num_modes=self.modes - len(self.prefixed))

        self.distribution = np.random.choice(['single', 'bimodal', 'multinomial', 'powerlaw', 'dirichlet'], size=1)[0] \
            if distribution == 'mixed' else distribution

        if np.isscalar(self.ranges) or isinstance(self.ranges, tuple):
            lims = (self.ranges-.0001, self.ranges+.0001) if np.isscalar(self.ranges) else self.ranges

            ## amps is an array with size matched to self.modes (e.g. 55).
            if self.distribution == 'single':
                #  The first element has a random value picked from the range given by "lims", all others zeros
                amps = single(num_modes=self.modes, range_lims=lims, signed=self.signed)

            elif self.distribution == 'bimodal':
                amps = bimodal(num_modes=self.modes, range_lims=lims, signed=self.signed)

            elif self.distribution == 'multinomial':
                amps = multinomial(num_modes=self.modes, range_lims=lims, signed=self.signed)

            elif self.distribution == 'powerlaw':
                amps = powerlaw(num_modes=self.modes, range_lims=lims, signed=self.signed, gamma=self.gamma)

            elif self.distribution == 'dirichlet':
                amps = dirichlet(num_modes=self.modes, range_lims=lims, signed=self.signed)

            else:  # draw amplitude for each zernike mode from a uniform dist
                amps = uniform(num_modes=self.modes, range_lims=lims)

            amplitudes = np.zeros(self.modes)      # initialize modes 55
            # pick order of modes from most "interesting" to least (51 modes)
            moi = pick_modes(num_modes=self.modes, prefixed=self.prefixed, mode_weights=self.mode_weights)

            # assign amplitudes  amps[:len(moi)] is 51, so amplitudes[piston, tip,tilt, defocus] will always be zero
            amplitudes[moi] = amps[:len(moi)]
            amplitudes = self._formatter(amplitudes, order)

            self.zernikes = {
                Zernike(j, order=order): a
                for j, a in amplitudes.items()
            }

            if self.rotate:
                for j, a in amplitudes.items():
                    if a != 0:
                        z = Zernike(j, order=order)
                        twin = Zernike((z.n, z.m * -1), order=order)

                        if z.m != 0 and self.zernikes.get(twin) is not None:
                            a = np.sqrt(self.zernikes[z] ** 2 + self.zernikes[twin] ** 2)
                            randomangle = np.random.uniform(
                                low=0,
                                high=2 * np.pi if self.signed else np.pi / 2
                            )
                            self.zernikes[z] = a * np.cos(randomangle)
                            self.zernikes[twin] = a * np.sin(randomangle)

        elif isinstance(amplitudes, Path) or isinstance(amplitudes, str):
            amplitudes = self._fit_zernikes(amplitudes)
            amplitudes = self._formatter(amplitudes, order)
            self.zernikes = {
                Zernike(j, order=order): a
                for j, a in amplitudes.items()
            }
        else:
            amplitudes = self._formatter(amplitudes, order)
            self.zernikes = {
                Zernike(j, order=order): a
                for j, a in amplitudes.items()
            }

        self.amplitudes_noll = np.array(
            self._dict_to_list({z.index_noll: a for z, a in self.zernikes.items()})[1:]
        )
        self.amplitudes_noll_waves = np.array(
            self._dict_to_list({z.index_noll: self._microns2waves(a) for z, a in self.zernikes.items()})[1:]
        )

        self.amplitudes_ansi = np.array(
            self._dict_to_list({z.index_ansi: a for z, a in self.zernikes.items()})
        )
        self.amplitudes_ansi_waves = np.array(
            self._dict_to_list({z.index_ansi: self._microns2waves(a) for z, a in self.zernikes.items()})
        )

        self.amplitudes = np.array([self.zernikes[k] for k in sorted(self.zernikes.keys())])

        self.twins = {}
        for mode in self.zernikes:
            twin = mode.twin()
            if mode.index_ansi not in self.prefixed:
                if mode.index_ansi == twin.index_ansi:
                    self.twins[mode] = None

                elif mode.index_ansi < twin.index_ansi:
                    self.twins[mode] = twin

    def __len__(self):
        return len(self.zernikes)

    def __add__(self, other):
        if np.isscalar(other):
            return self.amplitudes + other
        else:
            return self.amplitudes + other.amplitudes

    def __sub__(self, other):
        if np.isscalar(other):
            return self.amplitudes - other
        else:
            return self.amplitudes - other.amplitudes

    def __mul__(self, other):
        if np.isscalar(other):
            return self.amplitudes * other
        else:
            return self.amplitudes * other.amplitudes

    def __truediv__(self, other):
        if np.isscalar(other):
            return self.amplitudes / other
        else:
            return self.amplitudes / other.amplitudes

    def _waves2microns(self, w):
        return w * self.lam_detection

    def _microns2waves(self, w):
        return w / self.lam_detection

    def _formatter(self, values, order):
        if isinstance(values, dict):
            return values

        elif isinstance(values, np.ndarray):
            values = tuple(values.ravel())
            offset = 1 if str(order).lower() == 'noll' else 0
            indices = range(offset, offset + len(values))
            return dict(zip(indices, values))

        elif isinstance(values, (tuple, list)):
            offset = 1 if str(order).lower() == 'noll' else 0
            indices = range(offset, offset + len(values))
            return dict(zip(indices, values))

        else:
            raise ValueError("Could not identify the data type for dictionary formation")

    def _dict_to_list(self, kv):
        max_key = max(kv.keys())
        out = [0] * (max_key + 1)
        for k, v in kv.items():
            out[k] = v
        return out

    def phase(self, rho, theta, normed=True, outside=None):
        """
            For creation of phase defined as a weighted sum of Zernike polynomial with a given polar co-ordinate system

            :param rho: 2D square array,  radial axis
            :param theta: 2D square array, azimuthal axis
            :param normed: boolean, whether the Zernike polynomials are normalized, default is True
            :param outside: scalar, outside padding of the spherical disc defined within a square grid, default is none
            :return: 2D array, wavefront computed for rho and theta
        """
        return np.sum(
            [a * z.phase(rho=rho, theta=theta, normed=normed, outside=outside) for z, a in self.zernikes.items()],
            axis=0
        )

    def polynomial(self, size, normed=True, outside=np.nan):
        """
            For visualization of weighted sum of Zernike polynomials on a disc of unit radius

            :param size: integer, Defines the shape of square grid, e.g. 256 or 512
            :param normed: Whether the Zernike polynomials are normalized, default is True
            :param outside: scalar, Outside padding of the spherical disc defined within a square grid, default is np.nan
            :return: 2D array, weighted sums of Zernike polynomials computed on a disc of unit radius defined within a square grid
        """
        return np.sum(
            [self._microns2waves(a) * z.polynomial(size=size, normed=normed, outside=outside)
             for z, a in self.zernikes.items()],
            axis=0
        )

    def wave(self, size=55, normed=True):
        return np.flip(np.rot90(self.polynomial(size=size, normed=normed)), axis=0)

    @lru_cache(maxsize=None)
    def na_mask(self, na: float = 1.0, wavefrontshape: tuple = (256,256)):
        center = (int(wavefrontshape[0] / 2), int(wavefrontshape[1] / 2))
        Y, X = np.ogrid[:wavefrontshape[0], :wavefrontshape[1]]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        return dist_from_center <= (na * wavefrontshape[0]) / 2

    def peak2valley(self, na: float = 1.0) -> float:
        """ measure peak-to-valley of the aberration in waves"""
        wavefront = self.wave(256)
        wavefront *= self.na_mask(na=na, wavefrontshape=wavefront.shape)
        return abs(np.nanmax(wavefront) - np.nanmin(wavefront))

    def _fit_zernikes(self, wavefront, rotate=True, microns=True):
        wavefront = np.ascontiguousarray(imread(wavefront).astype(float))

        if microns:
            wavefront *= self.lam_detection  # convert waves to microns before fitting.

        if rotate:
            wavefront = np.flip(np.rot90(wavefront), axis=0)

        zernikes = [Zernike(i) for i in range(self.modes)]
        # crop to where pupil is
        wavefront = wavefront[:, ~np.isnan(wavefront).all(axis=0)]
        wavefront = wavefront[~np.isnan(wavefront).all(axis=1), :]

        rho, theta = rho_theta(wavefront.shape[0])
        valid = rho <= 1 & ~np.isnan(wavefront)

        rho = rho[valid].flatten()
        theta = theta[valid].flatten()
        pupil_displacement = wavefront[valid].flatten()

        Z = np.array([nm_polynomial(z.n, z.m, rho=rho, theta=theta) for z in zernikes])
        coeffs, residuals, rank, s = np.linalg.lstsq(Z.T, pupil_displacement, rcond=None)
        coeffs[self.prefixed] = 0.
        return coeffs
