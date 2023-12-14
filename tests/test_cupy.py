import logging
import sys
sys.path.append('.')
sys.path.append('./src')
sys.path.append('./tests')
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")

import pytest


@pytest.mark.run(order=1)
def test_cupy(kargs):
    import cupy as cp
    import numpy as np

    # Create a NumPy array
    a = np.array([1, 2, 3])

    # Create a CuPy array from the NumPy array
    b = cp.array(a)

    # Test if the two arrays are equal
    assert np.array_equal(a, cp.asnumpy(b))

    # Test if the CuPy array is on the GPU
    assert b.device.id >= 0