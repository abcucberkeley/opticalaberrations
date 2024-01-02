import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.set_loglevel('error')

import os
import time
import logging
import sys
from pathlib import Path
from typing import Any

import tensorflow as tf
import numpy as np

import onnx
import tf2onnx
import onnxruntime as ort
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.compiler.tensorrt import trt_convert as tftrt

import backend


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


def convert2tftrt(
    model_path: Path,
    dtype: Any = np.float32,
    optimal_batch_size: int = 512,
):
    timeit = time.time()

    model = backend.load(f"{model_path}.h5")

    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    # inspect the layers operations inside your frozen graph definition and see the name of its input and output tensors
    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    # serialize the frozen graph and its text representation to disk.
    tf.io.write_graph(
        graph_or_graph_def=frozen_func.graph,
        logdir=str(Path(model_path).parent),
        name=f"{model_path}.pb",
        as_text=False
    )

    # Optional
    tf.io.write_graph(
        graph_or_graph_def=frozen_func.graph,
        logdir=str(Path(model_path).parent),
        name=f"{model_path}.pbtxt",
        as_text=True
    )

    return time.time() - timeit


def convertpb2tftrt(
    model_path: Path,
    dtype: Any = np.float32,
    optimal_batch_size: int = 512,
):
    timeit = time.time()

    converter = tftrt.TrtGraphConverterV2(
        input_saved_model_dir=f"{model_path}.h5",
        precision_mode=tftrt.TrtPrecisionMode.FP16,
        use_dynamic_shape=True,
        allow_build_at_runtime=False,
        dynamic_shape_profile_strategy='Optimal',
        max_workspace_size_bytes=1 << 32,
        maximum_cached_engines=100,
        minimum_segment_size=3,
    )

    # Converter method used to partition and optimize TensorRT compatible segments
    converter.convert()

    # Optionally, build TensorRT engines before deployment to save time at runtime
    # Note that this is GPU specific, and as a rule of thumb, we recommend building at runtime
    def input_fn():
        yield tf.convert_to_tensor(np.random.random((optimal_batch_size, 6, 64, 64, 1)).astype(dtype))

    converter.build(input_fn=input_fn)

    converter.save(f"{model_path}/tftrt")
    converter.summary()

    return time.time() - timeit


def convert2onnx(
    model_path: Path,
    dtype: Any = np.float32,
    optimal_batch_size: int = 512,
):
    timeit = time.time()

    model = backend.load(f"{model_path}.h5")
    # tfmodel = backend.load(f'{modeldir}/tf')
    #
    # for l1, l2 in zip(tfmodel.layers, model.layers):
    #     for w1, w2 in zip(l1.get_weights(), l2.get_weights()):
    #         logger.info(f"{l1.name}: {w1.shape} \t {l2.name}: {w2.shape}")
    #         try:
    #             np.testing.assert_equal(w1, w2)
    #         except AssertionError as e:
    #             logger.error(e)
    #     print('-'*100)

    input_signature = (tf.TensorSpec((None, 6, 64, 64, 1), tf.float32, name="embeddings"),)

    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        opset=18,
        input_signature=input_signature,
        target='tensorrt',
        output_path=f"{model_path}.onnx",
    )
    return time.time() - timeit


def benchmark(path, dataset, modelformat='tftrt'):
    timeit = time.time()

    if modelformat == 'onnx':
        options = ort.SessionOptions()
        sess = ort.InferenceSession(
            path,
            sess_options=options,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            # providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        )

        predictions = []
        for batch in dataset.as_numpy_iterator():
            preds = sess.run(['regressor'], {'embeddings': batch})[0]
            predictions.extend(preds)

    elif modelformat == 'tftrt':
        graph = tf.saved_model.load(path, tags=[tag_constants.SERVING])
        model = graph.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

        predictions = []
        for batch in dataset.as_numpy_iterator():
            preds = model(batch)
            predictions.extend(preds[next(iter(preds))].numpy())

    else:
        model = backend.load(path)
        predictions = model.predict(dataset)

    return predictions, time.time() - timeit


def optimize_model(
    model_path: Path,
    dtype: Any = np.float32,
    modelformat: str = 'onnx',
    number_of_samples: int = 10240,
    batch_size: int = 128,
):
    logger.info(f"Compiling {model_path.name} => {modelformat}")

    if modelformat == 'onnx':
        convert2onnx(
            model_path=f"{model_path}",
            dtype=dtype,
            optimal_batch_size=batch_size
        )
    elif modelformat == 'tftrt':
        convert2tftrt(
            model_path=f"{model_path}",
            dtype=dtype,
            optimal_batch_size=batch_size
        )
    else:
        logger.error(f"Unknown model format: {modelformat}")

    logger.info(f"Creating [{number_of_samples}] test samples")

    dataset = tf.data.Dataset.from_tensor_slices(
        np.random.random((number_of_samples, 6, 64, 64, 1)).astype(np.float32)
    )
    dataset = dataset.cache()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.with_options(options)

    logger.info(f"Evaluating samples with [{modelformat}] model")
    preds_tf, timer_tf = benchmark(f"{model_path}.h5", dataset=dataset, modelformat='keras')
    preds, timer = benchmark(f"{model_path}.onnx", dataset=dataset, modelformat=modelformat)

    logger.info(f"Runtime for {number_of_samples} samples [{batch_size=}]")
    logger.info(f"TF backend: {number_of_samples/timer_tf:.0f} prediction/sec.")
    logger.info(f"{modelformat.upper()} backend: {number_of_samples/timer:.0f} prediction/sec.")
    np.testing.assert_allclose(preds_tf, preds, rtol=1e-3, atol=1e-3)

