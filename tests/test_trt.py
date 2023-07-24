
import sys
from typing import Any
import numpy as np

sys.path.append('.')
sys.path.append('./src')
sys.path.append('./tests')

import warnings
warnings.filterwarnings("ignore")

import logging
logger = logging.getLogger('')

import pytest
from time import time
from src import convert
from src import backend


@pytest.mark.run(order=1)
def test_bulid_runtime(kargs):

    samples = np.array([convert.create_test_sample(kargs['model']) for _ in range(100)])
    embeddings, zernikes = np.stack(np.array(samples)[:, 0]), np.stack(np.array(samples)[:, 1])

    results_trt, timer_trt = convert.convert2trt(
        model_path=kargs['model'].with_suffix('')/'model',
        embeddings=embeddings,
        zernikes=zernikes,
        overwrite=True
    )

    model = backend.load(kargs['model'])
    timeit = time()
    results_tf = model.predict(embeddings, batch_size=100)
    timer_tf = time() - timeit

    np.testing.assert_allclose(results_trt, results_tf, atol=1e-2)

    logging.info(f"Runtime for TF backend: {embeddings.shape} - {timer_tf:.2f} sec.")
    logging.info(f"Runtime for native TRT backend: {embeddings.shape} - {timer_trt:.2f} sec.")

