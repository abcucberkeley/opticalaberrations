
import sys
from typing import Any
import numpy as np

sys.path.append('../tests')
sys.path.append('./src')
sys.path.append('./tests')

import warnings
warnings.filterwarnings("ignore")

import pytest
from time import time
from src import convert
from src import backend


@pytest.mark.run(order=1)
def test_build(kargs):

    samples = np.array([convert.create_test_sample(kargs['model']) for _ in range(100)])
    embeddings, zernikes = np.stack(np.array(samples)[:, 0]), np.stack(np.array(samples)[:, 1])

    convert.convert2onnx(
        model_path=kargs['model'].with_suffix('')/'model',
        embeddings=embeddings,
        zernikes=zernikes,
        overwrite=True
    )


@pytest.mark.run(order=2)
def test_runtime(kargs):

    samples = np.array([convert.create_test_sample(kargs['model']) for _ in range(100)])
    embeddings, zernikes = np.stack(np.array(samples)[:, 0]), np.stack(np.array(samples)[:, 1])

    results_onnx, timer_onnx = convert.convert2onnx(
        model_path=kargs['model'].with_suffix('')/'model',
        embeddings=embeddings,
        zernikes=zernikes,
        overwrite=False
    )

    model = backend.load(kargs['model'])
    timeit = time()
    results_tf = model.predict(embeddings, batch_size=kargs['batch_size'])
    timer_tf = time() - timeit

    np.testing.assert_allclose(results_onnx, results_tf, atol=1e-2)

    logger.info(f"Runtime for TF backend: {embeddings.shape} - {timer_tf:.2f} sec.")
    logger.info(f"Runtime for ONNX backend: {embeddings.shape} - {timer_onnx:.2f} sec.")
