import logging
import sys
from functools import partial
from line_profiler_pycharm import profile

import pandas as pd
from tifffile import TiffFile
from pathlib import Path
import numpy as np

import tensorflow as tf
import ujson

import embeddings
from wavefront import Wavefront
from zernike import Zernike
from synthetic import SyntheticPSF
from utils import multiprocess, resize_with_crop_or_pad

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
tf.get_logger().setLevel(logging.ERROR)


@profile
def get_image(path):
    if isinstance(path, tf.Tensor):
        path = Path(str(path.numpy(), "utf-8"))
    else:
        path = Path(str(path))

    with TiffFile(path) as tif:
        img = tif.asarray()

        if np.isnan(np.sum(img)):
            logger.error("NaN!")

    img = np.expand_dims(img, axis=-1)
    return img


def get_metadata(path, codename: str):
    try:
        if isinstance(path, tf.Tensor):
            path = Path(str(path.numpy(), "utf-8"))
        else:
            path = Path(str(path))

        with open(path.with_suffix('.json')) as f:
            hashtbl = ujson.load(f)

        return hashtbl[codename]

    except KeyError:
        return None


@profile
def get_sample(
        path,
        no_phase=False,
        metadata=False,
        input_coverage=1.0,
        embedding_option='spatial_planes',
        iotf=None,
        lls_defocus: bool = False,
        defocus_only: bool = False
):
    try:
        if isinstance(path, tf.Tensor):
            path = Path(str(path.numpy(), "utf-8"))
        else:
            path = Path(str(path))

        with open(path.with_suffix('.json')) as f:
            hashtbl = ujson.load(f)

        snr = hashtbl['snr']
        npoints = hashtbl['npoints']

        try:
            lls_defocus_offset = hashtbl['lls_defocus_offset']
        except KeyError:
            lls_defocus_offset = 0.

        if defocus_only:
            zernikes = [lls_defocus_offset]
        else:
            zernikes = hashtbl['zernikes']

            if lls_defocus:
                zernikes.append(lls_defocus_offset)

        try:
            p2v = hashtbl['peak2peak']
        except KeyError:
            p2v = Wavefront(zernikes, lam_detection=float(hashtbl['wavelength'])).peak2valley()

        try:
            avg_min_distance = hashtbl['avg_min_distance']
        except KeyError:
            avg_min_distance = 0.

        if metadata:
            return zernikes, snr, p2v, npoints, avg_min_distance, str(path)
        else:
            img = get_image(path)

            if input_coverage != 1.:
                img = resize_with_crop_or_pad(img, crop_shape=[int(s * input_coverage) for s in img.psf_shape])

            if img.shape[0] == img.shape[1] and iotf is not None:
                img = embeddings.fourier_embeddings(
                    img,
                    iotf=iotf,
                    padsize=None,
                    alpha_val='abs',
                    phi_val='angle',
                    remove_interference=True,
                    embedding_option=embedding_option,
                )

            if no_phase and img.shape[0] == 6:
                img = img[:3]
                wave = Wavefront(zernikes)

                for i, a in enumerate(zernikes):
                    mode = Zernike(i)
                    twin = Zernike((mode.n, mode.m * -1))

                    if mode.index_ansi > twin.index_ansi:
                        continue
                    else:
                        if mode.m != 0 and wave.zernikes.get(twin) is not None:
                            if np.sign(a) == -1:
                                zernikes[mode.index_ansi] *= -1
                                zernikes[twin.index_ansi] *= -1
                        else:
                            zernikes[i] = np.abs(a)

            return img, zernikes

    except Exception as e:
        logger.warning(f"Corrupted file {path}: {e}")


@profile
def check_sample(path):
    try:
        with open(path.with_suffix('.json')) as f:
            pass

        with TiffFile(path) as tif:
            pass
        return 1

    except Exception as e:
        logger.warning(f"Corrupted file {path}: {e}")
        return path


@profile
def check_criteria(
    file,
    distribution='/',
    embedding='',
    modes='',
    max_amplitude=1.,
    photons_range=None,
):
    path = str(file)
    amp = float(str([s for s in file.parts if s.startswith('amp_')][0]).split('-')[-1].replace('p', '.'))
    snr = tuple(map(int, str([s.strip('psnr_') for s in file.parts if s.startswith('psnr_')][0]).split('-')))

    if distribution in path \
            and embedding in path \
            and f"z{modes}" in path \
            and amp <= max_amplitude \
            and photons_range is None or (photons_range is not None and photons_range[0] <= snr[0] and snr[1] <= photons_range[1]) \
            and check_sample(file) == 1:
        return path


@profile
def load_dataset(
    datadir,
    split=None,
    multiplier=1,
    samplelimit=None,
    distribution='/',
    embedding='',
    modes='',
    max_amplitude=1.,
    photons_range=None,
):
    check = partial(
        check_criteria,
        distribution=distribution,
        embedding=embedding,
        modes=modes,
        max_amplitude=max_amplitude,
        photons_range=photons_range
    )
    files = multiprocess(
        func=check,
        jobs=Path(datadir).rglob('*[!_gt|!_realspace].tif'),
        cores=-1,
        desc='Loading dataset hashtable'
    )
    files = [f for f in files if f is not None]
    logger.info(f'Registered files: {len(files)}')

    if samplelimit is not None:
        files = np.random.choice(files, samplelimit, replace=False).tolist()
        logger.info(f'Selected files: {len(files)}')

    dataset_size = len(files) * multiplier
    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.repeat(multiplier)

    if split is not None:
        val_size = int(np.ceil(dataset_size * split))
        train = ds.skip(val_size)
        val = ds.take(val_size)

        logger.info(
            f"`{datadir}`: "
            f"training [{tf.data.experimental.cardinality(train).numpy()}], "
            f"validation [{tf.data.experimental.cardinality(val).numpy()}]"
        )
        return train, val
    else:
        logger.info(
            f"`{datadir}`: "
            f"dataset [{tf.data.experimental.cardinality(ds).numpy()}]"
        )
        return ds


@profile
def check_dataset(datadir):
    jobs = multiprocess(func=check_sample, jobs=Path(datadir).rglob('*.tif'), cores=-1)
    corrupted = [j for j in jobs if j != 1]
    corrupted = pd.DataFrame(corrupted, columns=['path'])
    logger.info(f"Corrupted files [{corrupted.index.shape[0]}]")
    print(corrupted)
    corrupted.to_csv(datadir/'corrupted.csv', header=False, index=False)
    return corrupted


@profile
def collect_dataset(
    datadir,
    split=None,
    multiplier=1,
    distribution='/',
    embedding='',
    modes='',
    samplelimit=None,
    max_amplitude=1.,
    no_phase=False,
    input_coverage=1.,
    embedding_option='spatial_planes',
    photons_range=None,
    iotf=None,
    metadata=False,
    lls_defocus: bool = False
):
    if metadata:
        # amps, snr, peak2peak, npoints, avg_min_distance, filename
        dtypes = [tf.float32, tf.int32, tf.float32, tf.float32, tf.float32, tf.string]
    else:
        # img, amps
        dtypes = [tf.float32, tf.float32]

    load = partial(
        get_sample,
        no_phase=no_phase,
        input_coverage=input_coverage,
        iotf=iotf,
        embedding_option=embedding_option,
        metadata=metadata,
        lls_defocus=lls_defocus
    )

    if split is not None:
        train_data, val_data = load_dataset(
            datadir,
            split=split,
            modes=modes,
            multiplier=multiplier,
            samplelimit=samplelimit,
            embedding=embedding,
            distribution=distribution,
            max_amplitude=max_amplitude,
            photons_range=photons_range
        )

        train = train_data.map(lambda x: tf.py_function(load, [x], dtypes))
        val = val_data.map(lambda x: tf.py_function(load, [x], dtypes))

        for i in train.take(1):
            logger.info(f"Input: {i[0].numpy().shape}")
            logger.info(f"Output: {i[1].numpy().shape}")

        logger.info(f"Training samples: {tf.data.experimental.cardinality(train).numpy()}")
        logger.info(f"Validation samples: {tf.data.experimental.cardinality(val).numpy()}")

        return train, val

    else:
        data = load_dataset(
            datadir,
            modes=modes,
            multiplier=multiplier,
            samplelimit=samplelimit,
            embedding=embedding,
            distribution=distribution,
            max_amplitude=max_amplitude,
            photons_range=photons_range
        )

        data = data.map(lambda x: tf.py_function(load, [x], dtypes))

        if not metadata:
            for i in data.take(1):
                logger.info(f"Input: {i[0].numpy().shape}")
                logger.info(f"Output: {i[1].numpy().shape}")

            logger.info(f"Samples: {tf.data.experimental.cardinality(data).numpy()}")

        return data


@profile
def create_dataset(config, split=None):
    master = tf.data.Dataset.from_tensor_slices([f'generator{i}' for i in range(4)])

    train = master.interleave(
        lambda x: tf.data.Dataset.from_generator(
            SyntheticPSF(**config).generator,
            output_signature=(
                tf.TensorSpec(shape=(None, *config['psf_shape']), dtype=tf.float32),
                tf.TensorSpec(shape=(None, config['n_modes']), dtype=tf.float32)
            )
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if split is not None:
        val = master.interleave(
            lambda x: tf.data.Dataset.from_generator(
                SyntheticPSF(**config).generator,
                output_signature=(
                    tf.TensorSpec(shape=(None, *config['psf_shape']), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, config['n_modes']), dtype=tf.float32)
                )
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return train, val
    else:
        return train


