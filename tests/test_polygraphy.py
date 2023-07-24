
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
def test_polygraphy_trt(kargs):

    samples = np.array([convert.create_test_sample(kargs['model']) for _ in range(100)])
    embeddings, zernikes = np.stack(np.array(samples)[:, 0]), np.stack(np.array(samples)[:, 1])

    results_polygraphy, timer_polygraphy = convert.convert2polygraphy(
        model_path=kargs['model'].with_suffix('')/'model',
        embeddings=embeddings,
        zernikes=zernikes,
        backend='trt',
        overwrite=False
    )

    model = backend.load(kargs['model'])
    timeit = time()
    results_tf = model.predict(embeddings, batch_size=100)
    timer_tf = time() - timeit

    np.testing.assert_allclose(results_polygraphy, results_tf, atol=1e-2)

    logging.info(f"Runtime for TF backend: {embeddings.shape} - {timer_tf:.2f} sec.")
    logging.info(f"Runtime for polygraphy.trt backend: {embeddings.shape} - {timer_polygraphy:.2f} sec.")


@pytest.mark.run(order=2)
def test_polygraphy_engine(kargs):

    samples = np.array([convert.create_test_sample(kargs['model']) for _ in range(100)])
    embeddings, zernikes = np.stack(np.array(samples)[:, 0]), np.stack(np.array(samples)[:, 1])

    results_polygraphy_engine, timer_polygraphy_engine = convert.convert2polygraphy(
        model_path=kargs['model'].with_suffix('')/'model',
        embeddings=embeddings,
        zernikes=zernikes,
        backend='engine',
        overwrite=True
    )

    model = backend.load(kargs['model'])
    timeit = time()
    results_tf = model.predict(embeddings, batch_size=100)
    timer_tf = time() - timeit

    np.testing.assert_allclose(results_polygraphy_engine, results_tf, atol=1e-2)

    logging.info(f"Runtime for TF backend: {embeddings.shape} - {timer_tf:.2f} sec.")
    logging.info(f"Runtime for polygraphy.engine backend: {embeddings.shape} - {timer_polygraphy_engine:.2f} sec.")

