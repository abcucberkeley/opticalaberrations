import logging
import sys
from functools import partial
from line_profiler_pycharm import profile

import pandas as pd
from tifffile import TiffFile
from pathlib import Path
import numpy as np

import tensorflow as tf
from skimage.filters import sobel, scharr
import ujson

from wavefront import Wavefront
from zernike import Zernike
from synthetic import SyntheticPSF
from utils import multiprocess

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
tf.get_logger().setLevel(logging.ERROR)


@profile
def get_image(path):
    with TiffFile(path) as tif:
        img = tif.asarray()
        tif.close()

        if np.isnan(np.sum(img)):
            logger.error("NaN!")

    img = np.expand_dims(img, axis=-1)
    return img


@profile
def get_sample(path, no_phase=False, metadata=False, edge_filter=False):
    try:
        if isinstance(path, tf.Tensor):
            path = Path(str(path.numpy(), "utf-8"))
        else:
            path = Path(str(path))

        with open(path.with_suffix('.json')) as f:
            hashtbl = ujson.load(f)
            f.close()

        img = get_image(path)
        amps = hashtbl['zernikes']
        snr = hashtbl['snr']
        p2v = hashtbl['peak2peak']
        npoints = hashtbl['npoints']

        if no_phase and img.shape[0] == 6:
            img = img[:3]
            wave = Wavefront(amps)

            for i, a in enumerate(amps):
                mode = Zernike(i)
                twin = Zernike((mode.n, mode.m * -1))

                if mode.index_ansi > twin.index_ansi:
                    continue
                else:
                    if mode.m != 0 and wave.zernikes.get(twin) is not None:
                        if np.sign(a) == -1:
                            amps[mode.index_ansi] *= -1
                            amps[twin.index_ansi] *= -1
                    else:
                        amps[i] = np.abs(a)

        if edge_filter:
            alpha = img[:3]
            alpha = scharr(alpha)
            alpha /= np.nanpercentile(alpha, 90)
            alpha[alpha > 1] = 1
            img[:3] = alpha

        if metadata:
            return img, amps, snr, p2v, npoints, str(path)

        else:
            return img, amps

    except Exception as e:
        logger.warning(f"Corrupted file {path}: {e}")


@profile
def check_sample(path):
    try:
        with open(path.with_suffix('.json')) as f:
            f.close()

        with TiffFile(path) as tif:
            tif.close()
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
    snr_range=None,
):
    path = str(file)
    amp = float(str([s for s in file.parts if s.startswith('amp_')][0]).split('-')[-1].replace('p', '.'))
    snr = tuple(map(int, str([s.strip('psnr_') for s in file.parts if s.startswith('psnr_')][0]).split('-')))

    if distribution in path \
            and embedding in path \
            and f"z{modes}" in path \
            and amp <= max_amplitude \
            and snr_range is None or (snr_range is not None and snr_range[0] <= snr[0] and snr[1] <= snr_range[1]) \
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
    snr_range=None,
):
    check = partial(
        check_criteria,
        distribution=distribution,
        embedding=embedding,
        modes=modes,
        max_amplitude=max_amplitude,
        snr_range=snr_range
    )
    files = multiprocess(check, Path(datadir).rglob('*[!_gt].tif'), cores=1, desc='Loading dataset hashtable')
    files = [f for f in files if f is not None]

    if samplelimit is not None:
        files = np.random.choice(files, samplelimit, replace=False).tolist()

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
    jobs = multiprocess(check_sample, Path(datadir).rglob('*.tif'), cores=-1)
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
    metadata=False,
    snr_range=None
):
    if metadata:
        # img, amps, snr, peak2peak, npoints, filename
        dtypes = [tf.float32, tf.float32, tf.int32, tf.float32, tf.float32, tf.string]
    else:
        # img, amps
        dtypes = [tf.float32, tf.float32]

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
            snr_range=snr_range
        )

        func = partial(get_sample, no_phase=no_phase, metadata=metadata)
        train = train_data.map(lambda x: tf.py_function(func, [x], dtypes))
        val = val_data.map(lambda x: tf.py_function(func, [x], dtypes))

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
            snr_range=snr_range
        )
        func = partial(get_sample, no_phase=no_phase, metadata=metadata)
        data = data.map(lambda x: tf.py_function(func, [x], dtypes))

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


