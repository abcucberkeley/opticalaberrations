"""
BSD 3-Clause License
Copyright (c) 2020, Debayan Saha, Martin Weigert, Uwe Schmidt
All rights reserved.
"""

import matplotlib
matplotlib.use('Agg')

import logging
import sys

import numpy as np
from zernike import Zernike

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
        :param order: string, Zernike nomenclature, eg noll or ansi, default is ansi
        :param lam_detection: wavelength in microns
    """
    _prefixed = {
        (0, 0):  0,
        (1, -1): 0,
        (1, 1):  0,
        (2, 0):  0,
    }

    def __init__(
        self,
        amplitudes,
        order='ansi',
        modes=60,
        lam_detection=.605,
        distribution=None,
        gamma=1.5,
        bimodal=True,
    ):
        self.ranges = amplitudes
        self.order = order
        self.lam_detection = lam_detection
        self.prefixed = [0, 1, 2, 4] if order == 'ansi' else [0, 1, 2, 4]
        self.length = modes
        self.gamma = gamma
        self.bimodal = bimodal

        self.distribution = np.random.choice(['powerlaw', 'dirichlet'], size=1)[0] \
            if distribution == 'mixed' else distribution

        if np.isscalar(self.ranges) or isinstance(self.ranges, tuple):
            lims = (self.ranges-.001, self.ranges+.001) if np.isscalar(self.ranges) else self.ranges

            if self.distribution == 'single':
                amplitudes = self._single(lims)

            elif self.distribution == 'powerlaw':
                amplitudes = self._powerlaw(lims)

            elif self.distribution == 'dirichlet':
                amplitudes = self._dirichlet(lims)

            else:  # draw amplitude for each zernike mode from a uniform dist
                amplitudes = np.random.uniform(*lims, size=self.length)

        amplitudes[self.prefixed] = 0.

        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # fig, axes = plt.subplots(2, 1, figsize=(4, 8))
        # sns.histplot(amplitudes, ax=axes[0], kde=True, color='dimgrey')
        # axes[0].set_xlim(np.min(amplitudes), np.max(amplitudes))
        # sns.barplot(x=np.arange(len(amplitudes)), y=amplitudes, ax=axes[1], palette='Accent')
        # axes[1].set_ylim(np.min(amplitudes), np.max(amplitudes))
        # axes[1].set_xlim(0, 60)
        # axes[1].set_xticks(np.arange(0, 65, 5))
        # axes[1].set_title(f"{round(self._microns2waves(sum(amplitudes)), 3)}$\lambda$")
        # plt.show()

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

    def _single(self, range_lims):
        idx = np.random.randint(4, self.length)
        amplitudes = np.zeros(self.length)

        if self.bimodal:
            amplitudes[idx] = np.random.choice([
                np.random.uniform(*range_lims),
                np.random.uniform(*-np.array(range_lims))
            ])
        else:
            amplitudes[idx] = np.random.uniform(*range_lims)

        return amplitudes

    def _dirichlet(self, range_lims, length=None):
        """ sum of the coefficients will add up to the desired peak2peak aberration """
        length = self.length if length is None else length
        sign = 0 if self.bimodal else np.sum(np.sign(range_lims))

        # draw negative and positive random numbers that add up to 1
        if sign == 0:
            pos_weights = np.random.dirichlet(np.ones(length), size=1)[0] * 2
            neg_weights = np.random.dirichlet(np.ones(length), size=1)[0] * -1
            weights = pos_weights + neg_weights
            amplitudes = weights * np.random.choice([
                np.random.uniform(*range_lims),
                np.random.uniform(*-np.array(range_lims))
            ])
        else:
            weights = np.random.dirichlet(np.ones(length), size=1)[0]
            amplitudes = weights * np.random.uniform(*range_lims)

        return amplitudes

    def _powerlaw(self, range_lims):
        weights = np.random.pareto(self.gamma, size=self.length)
        weights = weights / np.sum(weights)
        amplitudes = np.random.uniform(*range_lims) * weights

        if self.bimodal:
            amplitudes *= np.random.choice([-1, 1], size=self.length)

        return amplitudes

    def _waves2microns(self, w):
        return (self.lam_detection / 2 * np.pi) * w

    def _microns2waves(self, w):
        return (2 * np.pi / self.lam_detection) * w

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

    def wave(self, size=55):
        return np.flip(np.rot90(self.polynomial(size=size)), axis=0)
