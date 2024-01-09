import logging
import numpy as np
import sys

import pycuda.driver as cuda
import tensorrt as trt
import pycuda.autoinit  # automatically manage CUDA context creation and cleanup.

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = trt.Logger(trt.Logger.WARNING)


def compile_engine(model_path):
    f = open(model_path, "rb")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(f.read())
    return engine


def predict(context, batch, stream):
    
    in_shape = context.engine.get_binding_shape(0)
    out_shape = context.engine.get_binding_shape(1)
    in_shape[0] = batch.shape[0]  # update batch size
    out_shape[0] = batch.shape[0]  # update batch size

    outputs = np.zeros(out_shape, dtype=batch.dtype)
    allocated_input_buffers = cuda.mem_alloc(batch.nbytes)
    allocated_output_buffers = cuda.mem_alloc(outputs.nbytes)
    bindings = [int(allocated_input_buffers), int(allocated_output_buffers)]

    cuda.memcpy_htod_async(allocated_input_buffers, batch, stream)
    context.execute_async_v2(bindings, stream.handle, None)
    cuda.memcpy_dtoh_async(outputs, allocated_output_buffers, stream)
    stream.synchronize()

    return outputs
