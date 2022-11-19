"""
BSD 3-Clause License
Copyright (c) 2020, Debayan Saha, Martin Weigert, Uwe Schmidt
All rights reserved.
"""

import logging
import sys
from functools import lru_cache

import numpy as np
from scipy.special import binom

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def nm_to_noll(n, m):
    j = (n * (n + 1)) // 2 + abs(m)
    if m > 0 and n % 4 in (0, 1):
        return j
    if m < 0 and n % 4 in (2, 3):
        return j
    if m >= 0 and n % 4 in (2, 3):
        return j + 1
    if m <= 0 and n % 4 in (0, 1):
        return j + 1
    assert False


def nm_to_ansi(n, m):
    return (n * (n + 2) + m) // 2


def nm_normalization(n, m):
    """the norm of the zernike mode n,m in born/wolf convention
    i.e. sqrt( \int | z_nm |^2 )
    """
    return np.sqrt((1. + (m == 0)) / (2. * n + 2))


def nm_polynomial(n, m, rho, theta, normed=True):
    """returns the zernike polyonimal by classical n,m enumeration

    if normed=True, then they form an orthonormal system

        \int z_nm z_n'm' = delta_nn' delta_mm'

        and the first modes are

        z_nm(0,0)  = 1/sqrt(pi)*
        z_nm(1,-1) = 1/sqrt(pi)* 2r cos(phi)
        z_nm(1,1)  = 1/sqrt(pi)* 2r sin(phi)
        z_nm(2,0)  = 1/sqrt(pi)* sqrt(3)(2 r^2 - 1)
        ...
        z_nm(4,0)  = 1/sqrt(pi)* sqrt(5)(6 r^4 - 6 r^2 +1)
        ...

    if normed =False, then they follow the Born/Wolf convention
        (i.e. min/max is always -1/1)

        \int z_nm z_n'm' = (1.+(m==0))/(2*n+2) delta_nn' delta_mm'

        z_nm(0,0)  = 1
        z_nm(1,-1) = r cos(phi)
        z_nm(1,1)  =  r sin(phi)
        z_nm(2,0)  = (2 r^2 - 1)
        ...
        z_nm(4,0)  = (6 r^4 - 6 r^2 +1)
    """
    if abs(m) > n:
        logging.error(ValueError(" |m| <= n ! ( %s <= %s)" % (m, n)))

    if (n - m) % 2 == 1:
        return 0 * rho + 0 * theta

    radial = 0
    m0 = abs(m)

    for k in range((n - m0) // 2 + 1):
        radial += (-1.) ** k * binom(n - k, k) * binom(n - 2 * k, (n - m0) // 2 - k) * rho ** (n - 2 * k)

    radial *= (rho <= 1.)

    if normed:
        prefac = 1. / nm_normalization(n, m)
    else:
        prefac = 1.
    if m >= 0:
        return prefac * radial * np.cos(m0 * theta)
    else:
        return prefac * radial * np.sin(m0 * theta)

@lru_cache(maxsize=32)
def rho_theta(size):
    r = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(r, r, indexing='ij')
    rho = np.hypot(X, Y)
    theta = np.arctan2(Y, X)
    return rho, theta


@lru_cache(maxsize=32)
def outside_mask(size):
    rho, theta = rho_theta(size)
    return nm_polynomial(0, 0, rho, theta, normed=False) < 1


class Zernike:
    """
        Encapsulates Zernike polynomials

        :param index: string, integer or tuple, index of Zernike polynomial e.g. 'defocus', 4, (2,2)
        :param order: string, defines the Zernike nomenclature if index is an integer, eg noll or ansi, default is noll
    """

    _nm_pairs = set((n, m) for n in range(200) for m in range(-n, n + 1, 2))
    _noll_to_nm = dict(zip((nm_to_noll(*nm) for nm in _nm_pairs), _nm_pairs))
    _ansi_to_nm = dict(zip((nm_to_ansi(*nm) for nm in _nm_pairs), _nm_pairs))

    def __init__(self, index, order='ansi'):
        super().__setattr__('_mutable', True)

        if isinstance(index, (list, tuple)) and len(index) == 2:
            self.n, self.m = int(index[0]), int(index[1])
            (self.n, self.m) in self._nm_pairs \
            or logging.error(ValueError(
                "Your input for index is list/tuple : Could not identify the n,m order of Zernike polynomial"))

        elif isinstance(index, int):
            order = str(order).lower()
            order in ('noll', 'ansi') \
            or logging.error(ValueError("Your input for index is int : Could not identify the Zernike nomenclature/order"))

            if order == 'noll':
                index in self._noll_to_nm \
                or logging.error(ValueError(
                    "Your input for index is int and input for Zernike nomenclature is "
                    "Noll: Could not identify the Zernike polynomial with this index"))
                self.n, self.m = self._noll_to_nm[index]

            elif order == 'ansi':
                index in self._ansi_to_nm \
                or logging.error(ValueError(
                    "Your input for index is int and input for Zernike nomenclature is "
                    "ANSI: Could not identify the Zernike polynomial with this index"))
                self.n, self.m = self._ansi_to_nm[index]
        else:
            logging.error(ValueError("Could not identify your index input, we accept strings, lists and tuples only"))

        self.index_noll = nm_to_noll(self.n, self.m)
        self.index_ansi = nm_to_ansi(self.n, self.m)
        self._mutable = False

    def polynomial(self, size, normed=True, outside=np.nan):
        """
            For visualization of Zernike polynomial on a disc of unit radius

            :param size: integer, Defines the shape of square grid, e.g. 256 or 512
            :param normed: boolean, Whether the Zernike polynomials are normalized, default is True
            :param outside: scalar, Outside padding of the spherical disc defined within a square grid, default is np.nan
            :return: 2D array, Zernike polynomial computed on a disc of unit radius defined within a square grid  
        """

        np.isscalar(size) and int(size) > 0 or logging.error(ValueError())
        return self.phase(*rho_theta(int(size)), normed=normed, outside=outside)

    def phase(self, rho, theta, normed=True, outside=None):
        """
            For creation of a Zernike polynomial  with a given polar co-ordinate system

            :param rho: 2D square array,  radial axis
            :param theta: 2D square array, azimuthal axis
            :param normed: boolean, whether the Zernike polynomials are normalized, default is True
            :param outside: scalar, outside padding of the spherical disc defined within a square grid, default is None
            :return: 2D array, Zernike polynomial computed for rho and theta
        """
        (isinstance(rho, np.ndarray) and rho.ndim == 2 and rho.shape[0] == rho.shape[1]) or logging.error(
            ValueError('Only 2D square array for radial co-ordinate is accepted'))

        (isinstance(theta, np.ndarray) and theta.shape == rho.shape) or logging.error(
            ValueError('Only 2D square array for azimutha co-ordinate is accepted'))

        np.isscalar(normed) or logging.error(ValueError())
        outside is None or np.isscalar(outside) or logging.error(
            ValueError("Only scalar constant value for outside is accepted"))
        w = nm_polynomial(self.n, self.m, rho, theta, normed=bool(normed))

        if outside is not None:
            w[nm_polynomial(0, 0, rho, theta, normed=False) < 1] = outside
        return w

    def __hash__(self):
        return hash((self.n, self.m))

    def __eq__(self, other):
        return isinstance(other, Zernike) and (self.n, self.m) == (other.n, other.m)

    def __lt__(self, other):
        return self.index_ansi < other.index_ansi

    def __setattr__(self, *args):
        if self._mutable:
            super().__setattr__(*args)
        else:
            logging.error(AttributeError('Zernike is immutable'))

    def __repr__(self):
        return f'Zernike(n={self.n:2}, m={self.m:2}, noll={self.index_noll:2}, ansi={self.index_ansi:2})'

    @property
    def nm_pairs(self):
        return self._nm_pairs

    def ansi_to_nm(self, k):
        return self._ansi_to_nm[k]
