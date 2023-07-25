from functools import partial

import matplotlib
matplotlib.use('Agg')

import numexpr
numexpr.set_num_threads(numexpr.detect_number_of_cores())

import matplotlib.pyplot as plt
plt.set_loglevel('error')

import time
import subprocess
import logging
import sys
from pathlib import Path
from typing import Any

import tensorflow as tf
import numpy as np

import utils
import backend
from wavefront import Wavefront

import onnx
import tf2onnx
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from polygraphy.backend.trt import TrtRunner, EngineFromBytes


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
tf.get_logger().setLevel(logging.ERROR)


def batchify(arr, batch_size):
    for i in range(0, arr.shape[0], batch_size):
        yield arr[i:i + batch_size]


def create_test_sample(
    model_path: Path,
    dtype: Any = np.float32,
):
    gen = backend.load_metadata(model_path)
    phi = Wavefront(
        (0, .15),
        modes=gen.n_modes,
        distribution='mixed',
        signed=True,
        rotate=True,
        mode_weights='pyramid',
        lam_detection=gen.lam_detection,
    )

    psf, zernikes, y_lls_defocus = gen.single_psf(
        phi=phi,
        normed=True,
        meta=True,
        lls_defocus_offset=(0, 0)
    )

    inputs = utils.add_noise(psf)
    emb = backend.preprocess(
        inputs,
        modelpsfgen=gen,
        digital_rotations=None,
        remove_background=True,
        normalize=True,
    )
    return emb.astype(dtype), zernikes.astype(dtype)


def convert2onnx(
    model_path: Path,
    embeddings: np.ndarray,
    zernikes: np.ndarray,
    dtype: Any = np.float32,
    atol: float = 1e-2,
    overwrite: bool = True,
    batch_size: int = 512,
):

    # input_signature = [tf.TensorSpec((batch_size, *input_shape[1:]), dtype=dtype, name='embeddings')]
    # onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature)
    # onnx.save(onnx_model, f"{model_path}.onnx")
    # del onnx_model

    if overwrite or not Path(f"{model_path}.onnx").exists():
        subprocess.call(
            f"python -m tf2onnx.convert "
            f"--saved-model {model_path} "
            f"--output={model_path}.onnx "
            f"--rename-inputs embeddings "
            f"--rename-outputs zernikes "
            f"--target tensorrt "
            f"--verbose ",
            shell=True,
        )

    # load onnx model
    sess = ort.InferenceSession(f"{model_path}.onnx", providers=['CUDAExecutionProvider'])

    timeit = time.time()
    results_ort = sess.run(["zernikes"], {"embeddings": embeddings})[0]
    timer = time.time() - timeit

    try:
        np.testing.assert_allclose(results_ort, zernikes, atol=atol)
    except AssertionError as e:
        logger.info(e)

    return results_ort, timer


def convert2trt(
    model_path: Path,
    embeddings: np.ndarray,
    zernikes: np.ndarray,
    dtype: Any = np.float32,
    atol: float = 1e-2,
    overwrite: bool = True,
    batch_size: int = 512,
):
    if not Path(f"{model_path}.onnx").exists():
        subprocess.call(
            f"python -m tf2onnx.convert "
            f"--saved-model {model_path} "
            f"--output={model_path}.onnx "
            f"--rename-inputs embeddings "
            f"--rename-outputs zernikes "
            f"--target tensorrt "
            f"--verbose ",
            shell=True,
        )

    if overwrite or not Path(f"{model_path}.trt").exists():
        subprocess.call(
            f"/usr/src/tensorrt/bin/trtexec "
            f"--verbose "
            f"--onnx={model_path}.onnx "
            f"--saveEngine={model_path}.trt "
            f"--best",
            shell=True,
        )

    timeit = time.time()

    f = open(f"{model_path}.trt", "rb")
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    output = np.empty_like(zernikes, dtype=dtype)

    # Allocate device memory
    d_input = cuda.mem_alloc(1 * embeddings.nbytes)
    d_output = cuda.mem_alloc(1 * output.nbytes)

    bindings = [int(d_input), int(d_output)]
    stream = cuda.Stream()

    def predict(batch):  # result gets copied into output
        # Transfer input data to device
        cuda.memcpy_htod_async(d_input, batch, stream)
        # Execute model
        context.execute_async_v2(bindings, stream.handle, None)
        # Transfer predictions back
        cuda.memcpy_dtoh_async(output, d_output, stream)
        # Syncronize threads
        stream.synchronize()
        return output

    results_trt = predict(embeddings).astype(dtype)

    timer = time.time() - timeit

    try:
        np.testing.assert_allclose(results_trt, zernikes, atol=atol)
    except AssertionError as e:
        logger.info(e)

    return results_trt, timer


def convert2polygraphy(
    model_path: Path,
    embeddings: np.ndarray,
    zernikes: np.ndarray,
    dtype: Any = np.float32,
    atol: float = 1e-2,
    backend: str = 'engine',
    overwrite: bool = True,
    batch_size: int = 512,
):

    if not Path(f"{model_path}.onnx").exists():
        subprocess.call(
            f"python -m tf2onnx.convert "
            f"--saved-model {model_path} "
            f"--output={model_path}.onnx "
            f"--rename-inputs embeddings "
            f"--rename-outputs zernikes "
            f"--target tensorrt "
            f"--verbose ",
            shell=True,
        )

    if backend == 'trt' and not Path(f"{model_path}.trt").exists():
        subprocess.call(
            f"/usr/src/tensorrt/bin/trtexec "
            f"--verbose "
            f"--onnx={model_path}.onnx "
            f"--saveEngine={model_path}.trt "
            f"--best",
            shell=True,
        )
    else:
        if overwrite or not Path(f"{model_path}.engine").exists():
            n, z, y, x, c = embeddings.shape
            subprocess.call(
                f"/usr/src/tensorrt/bin/trtexec "
                f"--verbose "
                f"--onnx={model_path}.onnx "
                f"--saveEngine={model_path}.engine "
                f"--minShapes=embeddings:1x{z}x{y}x{x}x{c} "
                f"--optShapes=embeddings:512x{z}x{y}x{x}x{c} "
                f"--maxShapes=embeddings:1024x{z}x{y}x{x}x{c} "
                f"--best",
                shell=True,
            )

    timeit = time.time()

    if backend == 'trt':
        f = open(f"{model_path}.trt", "rb")
        engine = EngineFromBytes(f.read())
        runner = TrtRunner(engine)

        with runner:
            results_trt = np.concatenate([
                runner.infer({"embeddings": emb[np.newaxis, ...]})["zernikes"]
                for emb in embeddings
            ], axis=0)

    else:
        f = open(f"{model_path}.engine", "rb")
        engine = EngineFromBytes(f.read())
        runner = TrtRunner(engine)

        with runner:
            results_trt = np.concatenate([
                runner.infer({"embeddings": batch})["zernikes"]
                for batch in batchify(embeddings, batch_size=batch_size)
            ], axis=0)

    timer = time.time() - timeit

    try:
        np.testing.assert_allclose(results_trt, zernikes, atol=atol)
    except AssertionError as e:
        logger.info(e)

    return results_trt, timer


def optimize_model(
    model_path: Path,
    dtype: Any = np.float32,
    atol: float = 1e-2,
    modelformat: str = 'trt',
    number_of_samples: int = 10000,
    batch_size: int = 768,
):

    if not Path(f'{model_path.parent}/embeddings_{number_of_samples}.npy').exists():
        logger.info(f"Creating [{number_of_samples}] test samples")

        samples = utils.multiprocess(
            jobs=np.repeat(f"{model_path}.h5", number_of_samples),
            func=create_test_sample,
            cores=8,
            desc=f"Creating test samples"
        )
        embeddings, zernikes = np.stack(samples[:, 0]), np.stack(samples[:, 1])

        np.save(f'{model_path.parent}/embeddings_{number_of_samples}', embeddings)
        np.save(f'{model_path.parent}/zernikes_{number_of_samples}', zernikes)

    else:
        logger.info(f"Loading [{number_of_samples}] test samples")
        embeddings = np.load(f'{model_path.parent}/embeddings_{number_of_samples}.npy')
        zernikes = np.load(f'{model_path.parent}/zernikes_{number_of_samples}.npy')

    logger.info(f"Evaluating samples with [{modelformat}] model")

    if modelformat == 'onnx':
        results, timer = convert2onnx(
            model_path=model_path,
            embeddings=embeddings,
            zernikes=zernikes,
            dtype=dtype,
            atol=atol,
            overwrite=False,
            batch_size=batch_size
        )

    elif modelformat == 'polygraphy':
        results, timer = convert2polygraphy(
            model_path=model_path,
            embeddings=embeddings,
            zernikes=zernikes,
            dtype=dtype,
            atol=atol,
            overwrite=False,
            backend='engine',
            batch_size=batch_size
        )

    elif modelformat == 'trt':
        results, timer = convert2trt(
            model_path=model_path,
            embeddings=embeddings,
            zernikes=zernikes,
            dtype=dtype,
            atol=atol,
            overwrite=False,
            batch_size=batch_size
        )

    else:
        timer = 0.
        results = np.zeros_like(zernikes)
        logger.error(f"Unknown model format: {modelformat}")

    # load tf model
    model = backend.load(model_path)

    timeit = time.time()
    results_tf = model.predict(embeddings, batch_size=512)
    timer_tf = time.time() - timeit

    try:
        np.testing.assert_allclose(results_tf, zernikes, atol=atol)
    except AssertionError as e:
        logger.info(e)

    logger.info(f"Runtime for {embeddings.shape} samples")
    logger.info(f"TF backend [batchsize=512]: {number_of_samples/timer_tf:.0f} pred/sec [{timer_tf:.2f}]")
    logger.info(f"{modelformat.upper()} backend [batchsize={batch_size}]: {number_of_samples/timer:.0f} pred/sec [{timer:.2f}]")
    np.testing.assert_allclose(results, results_tf, atol=1e-2)
    return results, timer
