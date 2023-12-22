
import sys
sys.path.append('.')
sys.path.append('./src')
sys.path.append('./tests')

import logging
import warnings
warnings.filterwarnings("ignore")

import pytest
import tensorflow as tf
import shutil
from pathlib import Path

from src import train
from src import multipoint_dataset
from src.synthetic import SyntheticPSF


@pytest.mark.run(order=1)
def test_training_dataset(kargs):

    gen = SyntheticPSF(
        order='ansi',
        n_modes=15,
        distribution='mixed',
        mode_weights='pyramid',
        signed=True,
        rotate=True,
        psf_type='../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat',
        lam_detection=kargs['wavelength'],
        psf_shape=[64, 64, 64],
        x_voxel_size=kargs['lateral_voxel_size'],
        y_voxel_size=kargs['lateral_voxel_size'],
        z_voxel_size=kargs['axial_voxel_size'],
    )

    for i in range(5):
        multipoint_dataset.create_synthetic_sample(
            filename=f'{i}',
            npoints=5,
            fill_radius=.66,
            generators={str(kargs['psf_type']): gen},
            upsampled_generators={str(kargs['psf_type']): gen},
            modes=kargs['num_modes'],
            savedir=Path(f"{kargs['repo']}/dataset/training_dataset"),
            distribution=gen.distribution,
            mode_dist=gen.mode_weights,
            gamma=.75,
            randomize_voxel_size=False,
            emb=True,
            signed=True,
            random_crop=None,
            rotate=True,
            noise=True,
            normalize=True,
            min_amplitude=.1,
            max_amplitude=.2,
            min_lls_defocus_offset=-2,
            max_lls_defocus_offset=2,
            min_photons=100000,
            max_photons=200000,
        )


@pytest.mark.run(order=2)
def test_zernike_model(kargs):

    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    strategy = tf.distribute.MirroredStrategy()

    gpu_workers = strategy.num_replicas_in_sync
    logging.info(f'Number of active GPUs: {gpu_workers}')
    
    subfolder = (f"z{int(kargs['axial_voxel_size']*1000)}-"
                 f"y{int(kargs['lateral_voxel_size']*1000)}-"
                 f"x{int(kargs['lateral_voxel_size']*1000)}")
    print(f"\n{subfolder=}\n")

    # clean out existing model
    outdir = Path(f"{kargs['repo']}/models/pytests/yumb_zernike_model")
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    with strategy.scope():
        train.train_model(
            network='prototype',
            dataset=Path(f"{kargs['repo']}/dataset/training_dataset/YuMB_lambda510/{subfolder}/z64-y64-x64/z15/"),
            outdir=outdir,
            psf_type=kargs['psf_type'],
            x_voxel_size=kargs['lateral_voxel_size'],
            y_voxel_size=kargs['lateral_voxel_size'],
            z_voxel_size=kargs['axial_voxel_size'],
            modes=kargs['num_modes'],
            wavelength=kargs['wavelength'],
            batch_size=kargs['batch_size']//3,
            warmup=1,
            epochs=5,
        )


@pytest.mark.run(order=3)
def test_defocus_model(kargs):

    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    strategy = tf.distribute.MirroredStrategy()

    gpu_workers = strategy.num_replicas_in_sync
    logging.info(f'Number of active GPUs: {gpu_workers}')

    subfolder = (f"z{int(kargs['axial_voxel_size']*1000)}-"
                 f"y{int(kargs['lateral_voxel_size']*1000)}-"
                 f"x{int(kargs['lateral_voxel_size']*1000)}")

    # clean out existing model
    outdir = Path(f"{kargs['repo']}/models/pytests/yumb_defocus_model")
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    with strategy.scope():
        train.train_model(
            network='prototype',
            dataset=Path(f"{kargs['repo']}/dataset/training_dataset/YuMB_lambda510/{subfolder}/z64-y64-x64/z15/"),
            outdir=outdir,
            psf_type=kargs['psf_type'],
            x_voxel_size=kargs['lateral_voxel_size'],
            y_voxel_size=kargs['lateral_voxel_size'],
            z_voxel_size=kargs['axial_voxel_size'],
            modes=kargs['num_modes'],
            wavelength=kargs['wavelength'],
            batch_size=kargs['batch_size']//3,
            warmup=1,
            epochs=5,
            defocus_only=True,
        )


@pytest.mark.run(order=4)
def test_zernike_defocus_model(kargs):

    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    strategy = tf.distribute.MirroredStrategy()

    gpu_workers = strategy.num_replicas_in_sync
    logging.info(f'Number of active GPUs: {gpu_workers}')

    subfolder = (f"z{int(kargs['axial_voxel_size']*1000)}-"
                 f"y{int(kargs['lateral_voxel_size']*1000)}-"
                 f"x{int(kargs['lateral_voxel_size']*1000)}")

    # clean out existing model
    outdir = Path(f"{kargs['repo']}/models/pytests/yumb_zern_defocus_model")
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)


    with strategy.scope():
        train.train_model(
            network='prototype',
            dataset=Path(f"{kargs['repo']}/dataset/training_dataset/YuMB_lambda510/{subfolder}/z64-y64-x64/z15/"),
            outdir=outdir,
            psf_type=kargs['psf_type'],
            x_voxel_size=kargs['lateral_voxel_size'],
            y_voxel_size=kargs['lateral_voxel_size'],
            z_voxel_size=kargs['axial_voxel_size'],
            modes=kargs['num_modes'],
            wavelength=kargs['wavelength'],
            batch_size=kargs['batch_size']//3,
            warmup=1,
            epochs=5,
            lls_defocus=True,
        )


@pytest.mark.run(order=5)
def test_finetune_zernike_model(kargs):

    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)

    strategy = tf.distribute.MirroredStrategy()

    gpu_workers = strategy.num_replicas_in_sync
    logging.info(f'Number of active GPUs: {gpu_workers}')

    subfolder = (f"z{int(kargs['axial_voxel_size']*1000)}-"
                 f"y{int(kargs['lateral_voxel_size']*1000)}-"
                 f"x{int(kargs['lateral_voxel_size']*1000)}")

    # clean out existing model
    outdir = Path(f"{kargs['repo']}/models/pytests/yumb_zernike_model_finetuned")
    if outdir.exists() and outdir.is_dir():
        shutil.rmtree(outdir)

    with strategy.scope():
        train.train_model(
            network='prototype',
            dataset=Path(f"{kargs['repo']}/dataset/training_dataset/YuMB_lambda510/{subfolder}/z64-y64-x64/z15/"),
            outdir=outdir,
            psf_type=kargs['psf_type'],
            x_voxel_size=kargs['lateral_voxel_size'],
            y_voxel_size=kargs['lateral_voxel_size'],
            z_voxel_size=kargs['axial_voxel_size'],
            modes=kargs['num_modes'],
            wavelength=kargs['wavelength'],
            batch_size=kargs['batch_size']//3,
            warmup=1,
            epochs=5,
            finetune=Path(f"{kargs['repo']}/models/tests/yumb_zernike_model"),
        )
