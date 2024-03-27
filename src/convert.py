import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.set_loglevel('error')

from pathlib import Path
from typing import Any
import subprocess
import time
import numpy as np

subprocess.call(
    "pip install -U --extra-index-url https://pypi.ngc.nvidia.com triton-model-navigator[tensorflow]",
    shell=True
)
import model_navigator as nav

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.compiler.tensorrt import trt_convert as tftrt

import onnx
import tf2onnx
import onnxruntime as ort
import pycuda.driver as cuda

import backend
import trt_utils


def batchify(arr, batch_size):
    for i in range(0, arr.shape[0], batch_size):
        yield arr[i:i + batch_size]


def convert_h5_to_graph(
    model_path: Path,
):
    model = backend.load(model_path)

    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    frozen = convert_variables_to_constants_v2(full_model)
    frozen.graph.as_graph_def()

    input_names = [t.name for t in model.inputs]
    output_names = [t.name for t in model.outputs]
    
    print(f"Frozen model inputs: {input_names}: {frozen.inputs}")
    print(f"Frozen model outputs: {output_names}: {frozen.outputs}")

    # model_path.with_suffix('').mkdir(parents=True, exist_ok=True)
    
    tf.io.write_graph(
        graph_or_graph_def=frozen.graph,
        logdir=str(Path(model_path).parent),
        name=f"{model_path.with_suffix('')}_frozen_graph.pb",
        as_text=False
    )

    tf.io.write_graph(
        graph_or_graph_def=frozen.graph,
        logdir=str(Path(model_path).parent),
        name=f"{model_path.with_suffix('')}_frozen_graph.pbtxt",
        as_text=True
    )   

    return frozen, input_names, output_names


def convert_h5_to_pb(
    model_path: Path,
):
    timeit = time.time()
    
    model = backend.load(model_path)
    tf.saved_model.save(model, str(model_path.with_suffix('')))
    subprocess.call(
        f"saved_model_cli show --all --dir {model_path.with_suffix('')}",
        shell=True,
    )
    
    return time.time() - timeit


def convert_pb_to_tftrt(
    model_path: Path,
    dtype: Any = 'float16',
    optimal_batch_size: int = 128,
):
    timeit = time.time()
    
    converter = tftrt.TrtGraphConverterV2(
        input_saved_model_dir=str(model_path.with_suffix('')),
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
        yield tf.convert_to_tensor(np.random.random((optimal_batch_size, 6, 64, 64, 1)).astype('float32'))

    converter.build(input_fn=input_fn)

    converter.save(str(model_path.with_suffix('.tftrt')))
    converter.summary()

    return time.time() - timeit


def convert_h5_to_onnx(
    model_path: Path,
    optimal_batch_size: int = 128,
):
    timeit = time.time()
    
    model = backend.load(model_path)
    input_signature = (tf.TensorSpec((None, 6, 64, 64, 1), 'float16', name="embeddings"),)

    tf2onnx.convert.from_keras(
        model,
        opset=17,
        input_signature=input_signature,
        target='tensorrt',
        output_path=str(model_path.with_suffix('.onnx')),
    )
    
    return time.time() - timeit


def convert_onnx_to_trt(
    model_path: Path,
    dtype: Any = 'float16',
    optimal_batch_size: int = 128,
):
    timeit = time.time()
    
    subprocess.call(
        f"/usr/src/tensorrt/bin/trtexec "
        f"--onnx={model_path.with_suffix('.onnx')} "
        f"--saveEngine={model_path.with_suffix('.trt')} "
        "--workspace=16000 "
        # f"--explicitBatch "
        # f"--fp16 "
        # f"--verbose "
        f"--best ",
        shell=True,
    )
    
    return time.time() - timeit


def convert_pb_to_onnx(
    model_path: Path,
    dtype: Any = 'float16',
    optimal_batch_size: int = 128,
):
    timeit = time.time()
    
    subprocess.call(
        f"python -m tf2onnx.convert "
        f"--saved-model={model_path.with_suffix('')} "
        f"--output={model_path.with_suffix('.onnx')} "
        f"--opset=17 "
        f"--target=tensorrt "
        f"--verbose ",
        shell=True,
    )   
    
    return time.time() - timeit


def benchmark(path, dataset, modelformat='tftrt'):

    timeit = time.time()
    
    if modelformat == 'onnx':
        model = onnx.load(path)
        out_names =[node.name for node in model.graph.output]
        input_names = [node.name for node in model.graph.input]
        del model 
        
        options = ort.SessionOptions()
        sess = ort.InferenceSession(
            path,
            sess_options=options,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            # providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        )

        predictions = []
        for batch in dataset.as_numpy_iterator():
            preds = sess.run(out_names, {input_names[0]: batch})[0]
            predictions.extend(preds)

    elif modelformat == 'pb' or modelformat == 'tftrt':
        graph = tf.saved_model.load(path, tags=[tag_constants.SERVING])
        model = graph.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

        predictions = []
        for batch in dataset:
            preds = model(batch)
            predictions.extend(preds[next(iter(preds))].numpy())

    elif modelformat == 'trt':
        stream = cuda.Stream()
        with trt_utils.compile_engine(path) as engine:
            with engine.create_execution_context() as context:

                predictions = []
                for batch in dataset.as_numpy_iterator():
                    preds = trt_utils.predict(
                        context=context,
                        stream=stream,
                        batch=batch,
                    )
                    predictions.extend(preds)

    else:  # tf/keras model
        model = backend.load(path)
        predictions = model.predict(dataset)

    return predictions, time.time() - timeit


def convert_model(
    model_path: Path,
    dtype: Any = 'float16',
    modelformat: str = 'trt',
    number_of_samples: int = 10*1024,
    batch_size: int = 128,
):
    print(f"Compiling {model_path.name} => {modelformat}")

    if modelformat == 'pb':
        convert_h5_to_pb(
            model_path=model_path,
        )  
    elif modelformat == 'tftrt':
        convert_h5_to_pb(
            model_path=model_path,
        )  
        convert_pb_to_tftrt(
            model_path=model_path,
            dtype=dtype,
            optimal_batch_size=batch_size
        )
    elif modelformat == 'onnx':
        convert_h5_to_onnx(
            model_path=model_path,
            optimal_batch_size=batch_size
        )
    elif modelformat == 'trt':
        convert_h5_to_pb(
            model_path=model_path,
        ) 
        convert_pb_to_onnx(
            model_path=model_path,
            dtype=dtype,
            optimal_batch_size=batch_size
        )
        convert_onnx_to_trt(
            model_path=model_path,
            dtype=dtype,
            optimal_batch_size=batch_size
        )
    else:
        print(f"Unknown model format: {modelformat}")

    print(f"Creating [{number_of_samples}] test samples")

    dataset = tf.data.Dataset.from_tensor_slices(
        np.random.random((number_of_samples, 6, 64, 64, 1)).astype('float32')
    )
    dataset = dataset.cache()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.with_options(options)

    print(f"Evaluating samples with [{modelformat}] model")
    preds, timer = benchmark(
        model_path.with_suffix(f".{modelformat}") if modelformat != 'pb' else model_path.with_suffix(''), 
        dataset=dataset, 
        modelformat=modelformat
    )
    preds_tf, timer_tf = benchmark(
        model_path, 
        dataset=dataset, 
        modelformat='keras'
    )

    print(f"Runtime for {number_of_samples} samples [{batch_size=}]")
    print(f"TF backend: {number_of_samples/timer_tf:.0f} prediction/sec.")
    print(f"{modelformat.upper()} backend: {number_of_samples/timer:.0f} prediction/sec.")
    np.testing.assert_allclose(preds_tf, preds, rtol=1e-3, atol=1e-3)


def verify_func(ys_runner, ys_expected):
    """Define verify function that compares outputs of the original model and the optimized model."""
    for y_runner, y_expected in zip(ys_runner, ys_expected):
        if not all(
            np.allclose(a, b, atol=1.0e-3) for a, b in zip(y_runner.values(), y_expected.values())
        ):
            return False
    return True


def optimize_model(
    model_path: Path,
    dtype: Any = 'float16',
    batch_size: int = 1024,
):

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    model = backend.load(model_path)
    embeddings = [{"input__0": np.random.rand(1, *model.input_shape[1:]).astype(dtype)} for _ in range(10)]
    # embeddings = [np.random.uniform(size=(batch_size, *model.input_shape[1:])).astype(dtype)]
    
    # package = nav.tensorflow.optimize(
    #     model=model,
    #     verify_func=verify_func,
    #     dataloader=embeddings,
    #     custom_configs=(
    #         nav.TensorRTConfig(
    #             precision=(nav.TensorRTPrecision.FP16,), #nav.TensorRTPrecision.FP32),
    #             # run_max_batch_size_search=True
    #         ),
    #     ),
    #     verbose=True,
    #     debug=True,
    #     batching=True
    # )

    # nav.package.save(package, model_path.with_suffix('.nav'), override=True)

    convert_h5_to_onnx(model_path=model_path)
     
    package = nav.onnx.optimize(
        model=model_path.with_suffix('.onnx'),   
        dataloader=embeddings,
        target_formats=(nav.Format.TENSORRT,),
        optimization_profile=nav.OptimizationProfile(max_batch_size=batch_size),
        custom_configs=[nav.TensorRTConfig(precision=(nav.TensorRTPrecision.FP16,))],
        verbose=True,
    )
    nav.package.save(package, model_path.with_suffix('.nav'), override=True)
