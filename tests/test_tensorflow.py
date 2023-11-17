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
def test_tensorflow(kargs):
    physical_devices = tf.config.list_physical_devices('GPU')

    strategy = tf.distribute.MirroredStrategy(
        devices=[f"{physical_devices[i].device_type}:{i}" for i in range(len(physical_devices))]
    )

    gpu_workers = strategy.num_replicas_in_sync

    if gpu_workers > 0:
        gpu_model = tf.config.experimental.get_device_details(physical_devices[0])['device_name']
    else:
        gpu_model = None

    logging.info(f'Number of active GPUs: {gpu_workers}, {gpu_model}')
    assert gpu_workers > 0
