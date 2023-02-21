"""
BSD 3-Clause License
Copyright (c) 2020, Debayan Saha, Martin Weigert, Uwe Schmidt
All rights reserved.
"""

import logging
import sys

import numpy as np
from zernike import Zernike, rho_theta, nm_polynomial
from pathlib import Path
from tifffile import imread

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Wavefront:
    """
        Encapsulates the wavefront defined by Zernike polynomials

        :param amplitudes: dictionary, nd array, tuple or list, Amplitudes of Zernike polynomials
        :param order: string, Zernike nomenclature, 'noll' or 'ansi', default is 'ansi'
        :param lam_detection: wavelength in microns
        :param modes: number of modes to be selected from.
        :param mode_weights: How likely to pick modes to be selected. 'pyramid', 'decay', or all else uniform (uniform=default)
        :param distribution: Once modes are selected. Amplitude distribution 'single', 'powerlaw', 'dirichlet', or 'mixed'=random choice of distribution      

    """

    def __init__(
        self,
        amplitudes,
        order='ansi',
        modes=55,
        lam_detection=.510,
        distribution=None,
        gamma=.75,
        signed=True,
        rotate=False,
        mode_weights='uniform'
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
            self.mode_weights = self._pyramid_weights(num_modes=self.modes - len(self.prefixed))
        elif mode_weights == 'decay':
            self.mode_weights = self._decayed_weights(num_modes=self.modes - len(self.prefixed))
        else:
            self.mode_weights = self._uniform_weights(num_modes=self.modes - len(self.prefixed))

        self.distribution = np.random.choice(['single', 'bimodal', 'multinomial', 'powerlaw', 'dirichlet'], size=1)[0] \
            if distribution == 'mixed' else distribution

        if np.isscalar(self.ranges) or isinstance(self.ranges, tuple):
            lims = (self.ranges-.0001, self.ranges+.0001) if np.isscalar(self.ranges) else self.ranges

            ## amps is an array with size matched to self.modes (e.g. 55).
            if self.distribution == 'single':
                #  The first element has a random value picked from the range given by "lims", all others zeros
                amps = self._single(lims)

            elif self.distribution == 'bimodal':
                amps = self._bimodal(lims)

            elif self.distribution == 'multinomial':
                amps = self._multinomial(lims)

            elif self.distribution == 'powerlaw':
                amps = self._powerlaw(lims)

            elif self.distribution == 'dirichlet':
                amps = self._dirichlet(lims)

            else:  # draw amplitude for each zernike mode from a uniform dist
                amps = np.random.uniform(*lims, size=self.modes)

            amplitudes = np.zeros(self.modes)      # initialize modes 55
            moi = self._pick_modes()                # pick order of modes from most "interesting" to least (51 modes)
            amplitudes[moi] = amps[:len(moi)]       # assign amplitudes  amps[:len(moi)] is 51, so amplitudes[piston, tip,tilt, defocus] will always be zero
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

    def _uniform_weights(self, num_modes):
        weights = np.ones(num_modes).astype(float)
        weights /= np.sum(weights)  # normalize probabilities for choosing any given mode
        return weights

    def _decayed_weights(self, num_modes):
        weights = np.arange(1, num_modes + 1)[::-1].astype(float)
        weights /= np.sum(weights)  # normalize probabilities for choosing any given mode
        return weights

    def _pyramid_weights(self, num_modes, starting_ansi_index=15):
        hashtable = {
            Zernike(j, order=self.order): a
            for j, a in self._formatter(np.zeros(self.modes), self.order).items()
        }

        i = starting_ansi_index - len(self.prefixed) if starting_ansi_index >= len(self.prefixed) else 0

        weights = np.ones(num_modes)
        for z, a in hashtable.items():
            if z.index_ansi not in self.prefixed and z.index_ansi >= starting_ansi_index:
                weights[i] /= abs(z.m) + 2
                i += 1

        weights /= np.sum(weights)
        return weights

    def _pick_modes(self):
        """Return the number modes (with piston, tip, tilt, defocus removed) in an order given by the probabilites given by mode_weights.  Like an NBA draft order selection.  The highest probability team will get the #1 draft pick the highest amount of times.
        """
        modes = np.arange(self.modes).astype(int)
        modes = np.delete(modes, self.prefixed)  # remove bias, tip, tilt, and defocus from being selected
        options = np.random.choice(a=modes, p=self.mode_weights, size=1000) # we need to draw self.modes number of unique modes.  If we draw a mode that is already picked, we just throw that result away and redraw.  To do this we just draw a lot (e.g. 1000 times), and remove duplicates.
        u, picked = np.sort(np.unique(options,  return_index=True))         # remove duplicates, and by just retaining the first occurances (np.unique's return_index array) but it's ordered by u, so we just sort.
        return options[picked]

    def _single(self, range_lims):
        amplitudes = np.zeros(self.modes)

        if self.signed:
            amp = np.random.choice([
                np.random.uniform(*range_lims),
                np.random.uniform(*-np.array(range_lims))
            ])
        else:
            amp = np.random.uniform(*range_lims)

        amplitudes[0] = amp
        return amplitudes

    def _bimodal(self, range_lims):
        amplitudes = np.zeros(self.modes)

        if self.signed:
            a = np.random.choice([
                np.random.uniform(*range_lims),
                np.random.uniform(*-np.array(range_lims))
            ])
        else:
            a = np.random.uniform(*range_lims)

        frac = np.random.uniform(low=0, high=1)
        amplitudes[0] = a * frac
        amplitudes[1] = a * (1 - frac)
        return amplitudes

    def _multinomial(self, range_lims, maxpeaks=6):
        amplitudes = np.zeros(self.modes)

        if self.signed:
            a = np.random.choice([
                np.random.uniform(*range_lims),
                np.random.uniform(*-np.array(range_lims))
            ])
        else:
            a = np.random.uniform(*range_lims)

        dmodes = int(np.random.uniform(low=3, high=maxpeaks))

        for i in range(dmodes):
            amplitudes[i] = a/dmodes

        return amplitudes

    def _powerlaw(self, range_lims):
        weights = np.random.pareto(self.gamma, size=self.modes)
        weights /= np.sum(weights)
        weights = np.sort(weights)[::-1]
        amplitudes = np.random.uniform(*range_lims) * weights

        if self.signed:
            amplitudes *= np.random.choice([-1, 1], size=self.modes)

        return amplitudes

    def _dirichlet(self, range_lims):
        """ sum of the coefficients will add up to the desired peak2peak aberration """
        sign = 0 if self.signed else np.sum(np.sign(range_lims))

        # draw negative and positive random numbers that add up to 1
        if sign == 0:
            pos_weights = np.random.dirichlet(np.ones(self.modes), size=1)[0] * 2
            neg_weights = np.random.dirichlet(np.ones(self.modes), size=1)[0] * -1
            weights = pos_weights + neg_weights
            amplitudes = weights * np.random.choice([
                np.random.uniform(*range_lims),
                np.random.uniform(*-np.array(range_lims))
            ])
        else:
            weights = np.random.dirichlet(np.ones(self.modes), size=1)[0]
            amplitudes = weights * np.random.uniform(*range_lims)

        amplitudes = np.sort(amplitudes)[::-1]
        return amplitudes

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
