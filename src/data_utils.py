import logging
import sys

from tifffile import imread, imsave, TiffFile
from pathlib import Path
import numpy as np

import tensorflow as tf
import ujson
from synthetic import SyntheticPSF

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
tf.get_logger().setLevel(logging.ERROR)


def get_image(path):
    with TiffFile(path) as tif:
        img = tif.asarray()

        if np.isnan(np.sum(img)):
            logger.error("NaN!")

    img = np.expand_dims(img, axis=-1)
    return img


def get_sample(path):
    path = Path(str(path.numpy(), "utf-8"))
    with open(path.with_suffix('.json')) as f:
        hashtbl = ujson.load(f)

    amps = hashtbl['zernikes']
    img = get_image(path)
    return img, amps


def load_dataset(datadir, split=None, multiplier=1, samplelimit=None):
    files = [str(p) for p in Path(datadir).rglob('*.tif')]
    if samplelimit:
        files = files[:samplelimit]

    dataset_size = len(files) * multiplier
    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.repeat(multiplier)

    if split is not None:
        val_size = int(dataset_size * split)
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


def collect_dataset(
        datadir,
        split=None,
        multiplier=1,
        distribution='/',
        samplelimit=None,
        max_amplitude=.25,
):
    train_data, val_data = None, None
    classes = [
        c for c in Path(datadir).rglob('*/')
        if c.is_dir()
           and len(list(c.glob('*.tif'))) > 0
           and distribution in str(c)
           and float(str(c.name.split('-')[-1].replace('p', '.'))) <= max_amplitude
    ]

    for c in classes:
        if split is not None:
            t, v = load_dataset(c, multiplier=multiplier, split=split, samplelimit=samplelimit)
            train_data = t if train_data is None else train_data.concatenate(t)
            val_data = v if val_data is None else val_data.concatenate(v)
        else:
            data = load_dataset(c, multiplier=multiplier, samplelimit=samplelimit)
            if 'train' in str(c):
                train_data = data if train_data is None else train_data.concatenate(data)
            else:
                val_data = data if val_data is None else val_data.concatenate(data)

    train = train_data.map(lambda x: tf.py_function(get_sample, [x], [tf.float32, tf.float32]))
    val = val_data.map(lambda x: tf.py_function(get_sample, [x], [tf.float32, tf.float32]))

    for img, y in val.take(1):
        logger.info(f"Input: {img.numpy().shape}")
        logger.info(f"Output: {y.numpy().shape}")

    logger.info(f"Training samples: {tf.data.experimental.cardinality(train).numpy()}")
    logger.info(f"Validation samples: {tf.data.experimental.cardinality(val).numpy()}")

    return train, val


def create_dataset(config):
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
