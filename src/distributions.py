
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


def uniform(num_modes, range_lims):
    return np.random.uniform(*range_lims, size=num_modes)


def single(num_modes, range_lims, signed=True):
    amplitudes = np.zeros(num_modes)

    if signed:
        amp = np.random.choice([
            np.random.uniform(*range_lims),
            np.random.uniform(*-np.array(range_lims))
        ])
    else:
        amp = np.random.uniform(*range_lims)

    amplitudes[0] = amp
    return amplitudes


def bimodal(num_modes, range_lims, signed=True):
    amplitudes = np.zeros(num_modes)

    if signed:
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


def multinomial(num_modes, range_lims, signed=True, maxpeaks=6):
    amplitudes = np.zeros(num_modes)

    if signed:
        a = np.random.choice([
            np.random.uniform(*range_lims),
            np.random.uniform(*-np.array(range_lims))
        ])
    else:
        a = np.random.uniform(*range_lims)

    dmodes = int(np.random.uniform(low=3, high=maxpeaks))

    for i in range(dmodes):
        amplitudes[i] = a/ dmodes

    return amplitudes


def powerlaw(num_modes, range_lims, signed=True, gamma=.75):
    weights = np.random.pareto(gamma, size=num_modes)
    weights /= np.sum(weights)
    weights = np.sort(weights)[::-1]
    amplitudes = np.random.uniform(*range_lims) * weights

    if signed:
        amplitudes *= np.random.choice([-1, 1], size=num_modes)

    return amplitudes


def dirichlet(num_modes: int, range_lims: tuple, signed=True):
    '''

    Args:
        num_modes: total number of modes (e.g. 15)
        range_lims: tuple (min, max)
        signed: if True, amplitudes will have positive and negative values

    Returns:
        array of length "num_modes", which is descendingly sorted.The np.sum() of the array (e.g. not the Euclidean sum)
         will add up to the desired peak2peak aberration

    '''

    sign = 0 if signed else np.sum(np.sign(range_lims))

    # draw negative and positive random numbers that add up to 1
    if sign == 0:
        # positive and negative aberrations
        # num_modes = 11 in the case of Zernike's up to 15 since we skip tip, tilt, piston.
        pos_weights = np.random.dirichlet(np.ones(num_modes), size=1)[0] * 2
        neg_weights = np.random.dirichlet(np.ones(num_modes), size=1)[0] * -1
        weights = pos_weights + neg_weights
        amplitudes = weights * np.random.choice([
            np.random.uniform(*range_lims),
            np.random.uniform(*-np.array(range_lims))
        ])
    else:
        weights = np.random.dirichlet(np.ones(num_modes), size=1)[0]
        amplitudes = weights * np.random.uniform(*range_lims)

    amplitudes = np.sort(amplitudes)[::-1]
    return amplitudes


def uniform_weights(num_modes):
    weights = np.ones(num_modes).astype(float)
    weights /= np.sum(weights)  # normalize probabilities for choosing any given mode
    return weights


def decayed_weights(num_modes):
    weights = np.arange(1, num_modes + 1)[::-1].astype(float)
    weights /= np.sum(weights)  # normalize probabilities for choosing any given mode
    return weights


def pyramid_weights(num_modes, order='ansi', prefixed=(0, 1, 2, 4), starting_ansi_index=15):
    offset = 1 if str(order).lower() == 'noll' else 0
    indices = range(offset, offset + num_modes)
    zernikes = [Zernike(j, order=order) for j in indices]

    i = starting_ansi_index - len(prefixed) if starting_ansi_index >= len(prefixed) else 0

    weights = np.ones(num_modes)
    for z in zernikes:
        if z.index_ansi not in prefixed and z.index_ansi >= starting_ansi_index:
            weights[i] /= abs(z.m) + 2
            i += 1

    weights /= np.sum(weights)
    return weights


def pick_modes(num_modes, prefixed=(0, 1, 2, 4), mode_weights='uniform'):
    """
        Return the number modes in an order given by the probabilities set by mode_weights
        (without piston, tip, tilt, defocus).
        Like an NBA draft order selection:
        Highest probability team will get the #1 draft pick (and appear first in the array) the highest amount of times.
    """
    modes = np.arange(num_modes).astype(int)

    # remove bias, tip, tilt, and defocus from being selected
    modes = np.delete(modes, prefixed)

    # we need to draw self.modes number of unique modes.  If we draw a mode that is already picked,
    # we just throw that result away and redraw.
    # To do this we just draw a lot (e.g. 1000 times), and remove duplicates.
    options = np.random.choice(a=modes, p=mode_weights, size=1000)

    # remove duplicates, and by just retaining the first occurrences
    # (np.unique's return_index array) but it's ordered by u, so we just sort.
    u, picked = np.sort(np.unique(options,  return_index=True))

    return options[picked]
