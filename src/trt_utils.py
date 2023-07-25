import logging
import numpy as np
import sys

import onnx
import tf2onnx
import onnxruntime as ort
import pycuda.driver as cuda
import tensorrt as trt
import pycuda.autoinit  # automatically manage CUDA context creation and cleanup.

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = trt.Logger(trt.Logger.WARNING)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def compile_engine(model_path):
    f = open(model_path, "rb")
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(f.read())
    return engine


def allocate_buffers(engine, inputs):
    allocated_input_buffers = cuda.mem_alloc(inputs.nbytes)

    dshape = engine.get_binding_shape(1)
    dshape[0] *= inputs.shape[0]

    dtype = trt.nptype(engine.get_binding_dtype(engine[1]))
    allocated_output_buffers = cuda.mem_alloc(np.zeros(dshape, dtype=dtype).nbytes)

    bindings = [int(allocated_input_buffers), int(allocated_output_buffers)]
    return allocated_input_buffers, allocated_output_buffers, bindings


def predict(context, inputs, stream):
    allocated_input_buffers, allocated_output_buffers, bindings = allocate_buffers(context.engine, inputs=inputs)
    outputs = np.zeros_like(allocated_output_buffers, dtype=inputs.dtype)

    cuda.memcpy_htod_async(allocated_input_buffers, inputs, stream)
    context.execute_async_v2(bindings, stream.handle, None)
    cuda.memcpy_dtoh_async(outputs, allocated_output_buffers, stream)
    stream.synchronize()

    return outputs
