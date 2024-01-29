import logging
import sys
import time
import numpy as np
from pathlib import Path

import tensorflow as tf
from pytriton.client import ModelClient
from pytriton.model_config import ModelConfig, Tensor

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
tf.get_logger().setLevel(logging.ERROR)


batch_size = 32
image_size = (6, 64, 64, 1)
batch = np.random.uniform(size=(batch_size,) + image_size).astype('float16')

logger.info(f"Input: {batch.shape}")

with ModelClient(url='localhost', model_name="trt-fp16") as client:
    logger.info("Sending request")
    result_dict = client.infer_batch(batch)

logger.info(f"results: {result_dict}")