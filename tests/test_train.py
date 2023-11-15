
import sys
sys.path.append('.')
sys.path.append('./src')
sys.path.append('./tests')

import logging
import warnings
warnings.filterwarnings("ignore")

import pytest
import tensorflow as tf
from pathlib import Path

from src import train


@pytest.mark.run(order=1)
def test_zernike_model(kargs):

    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    strategy = tf.distribute.MirroredStrategy()

    gpu_workers = strategy.num_replicas_in_sync
    logging.info(f'Number of active GPUs: {gpu_workers}')

    with strategy.scope():
        train.train_model(
            dataset=Path(f"{kargs['repo']}/dataset/example/YuMB_lambda510/z200-y108-x108/z64-y64-x64/z15/"),
            outdir=Path(f"{kargs['repo']}/models/tests/yumb"),
            psf_type=kargs['psf_type'],
            x_voxel_size=kargs['lateral_voxel_size'],
            y_voxel_size=kargs['lateral_voxel_size'],
            z_voxel_size=kargs['axial_voxel_size'],
            modes=kargs['num_modes'],
            wavelength=kargs['wavelength'],
            batch_size=kargs['batch_size'],
            warmup=1,
            epochs=5,
            strategy=strategy
        )


@pytest.mark.run(order=2)
def test_defocus_model(kargs):

    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    strategy = tf.distribute.MirroredStrategy()

    gpu_workers = strategy.num_replicas_in_sync
    logging.info(f'Number of active GPUs: {gpu_workers}')

    with strategy.scope():
        train.train_model(
            dataset=Path(f"{kargs['repo']}/dataset/example/YuMB_lambda510/z200-y108-x108/z64-y64-x64/z15/"),
            outdir=Path(f"{kargs['repo']}/models/tests/yumb"),
            psf_type=kargs['psf_type'],
            x_voxel_size=kargs['lateral_voxel_size'],
            y_voxel_size=kargs['lateral_voxel_size'],
            z_voxel_size=kargs['axial_voxel_size'],
            modes=kargs['num_modes'],
            wavelength=kargs['wavelength'],
            batch_size=kargs['batch_size'],
            warmup=1,
            epochs=5,
            strategy=strategy,
            defocus_only=True,
        )


@pytest.mark.run(order=3)
def test_zernike_defocus_model(kargs):

    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    strategy = tf.distribute.MirroredStrategy()

    gpu_workers = strategy.num_replicas_in_sync
    logging.info(f'Number of active GPUs: {gpu_workers}')

    with strategy.scope():
        train.train_model(
            dataset=Path(f"{kargs['repo']}/dataset/example/YuMB_lambda510/z200-y108-x108/z64-y64-x64/z15/"),
            outdir=Path(f"{kargs['repo']}/models/tests/yumb"),
            psf_type=kargs['psf_type'],
            x_voxel_size=kargs['lateral_voxel_size'],
            y_voxel_size=kargs['lateral_voxel_size'],
            z_voxel_size=kargs['axial_voxel_size'],
            modes=kargs['num_modes'],
            wavelength=kargs['wavelength'],
            batch_size=kargs['batch_size'],
            warmup=1,
            epochs=5,
            strategy=strategy,
            lls_defocus=True,
        )
