import sys
import logging
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf

from wavefront import Wavefront
from synthetic import SyntheticPSF

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FourierAttention(layers.Layer):

    def __init__(
        self,
        na_det=1.0,
        x_voxel_size=.15,
        y_voxel_size=.15,
        z_voxel_size=.6,
        lambda_det=.605,
        refractive_index=1.33,
        **kwargs
    ):
        super(FourierAttention, self).__init__(**kwargs)
        self.na_det = na_det
        self.x_voxel_size = x_voxel_size * 1000
        self.y_voxel_size = y_voxel_size * 1000
        self.z_voxel_size = z_voxel_size * 1000
        self.lambda_det = lambda_det * 1000
        self.refractive_index = refractive_index

    def get_config(self):
        config = super(FourierAttention, self).get_config()
        config.update({
            "na_det": self.na_det,
            "x_voxel_size": self.x_voxel_size,
            "y_voxel_size": self.y_voxel_size,
            "z_voxel_size": self.z_voxel_size,
            "lambda_det": self.lambda_det,
            "refractive_index": self.refractive_index,
        })
        return config

    def build(self, input_shape):
        super(FourierAttention, self).build(input_shape)

    def _theoretical_otf(self, psf_shape):
        psfgen = SyntheticPSF(
            n_modes=60,
            amplitude_ranges=0,
            psf_shape=psf_shape,
            lam_detection=self.lambda_det,
            x_voxel_size=self.x_voxel_size,
            y_voxel_size=self.y_voxel_size,
            z_voxel_size=self.z_voxel_size,
            snr=1000,
            max_jitter=0,
        )

        otf = psfgen.single_otf(
            phi=Wavefront(np.zeros(60), lam_detection=self.lambda_det),
            zplanes=0,
            normed=True,
            noise=False,
            augmentation=False,
            na_mask=False,
            padsize=None
        )

        otf = tf.convert_to_tensor(otf, dtype=tf.float32)
        otf = tf.expand_dims(tf.expand_dims(otf, axis=0), axis=-1)
        return otf

    def _pad(self, inputs, padsize):
        shape = inputs.shape[1]
        size = shape * (padsize / shape)
        pads = int((size - shape) // 2)
        pads = tf.constant([[pads, pads], [pads, pads], [pads, pads]])
        padded = tf.pad(inputs, pads, "CONSTANT", constant_values=0)
        return padded

    def _fft(self, inputs, padsize=None, gamma=1.0):
        if padsize is not None:
            inputs = self._pad(inputs, padsize)

        # compute the 3-dimensional FFT over the three innermost dimensions
        fft = tf.transpose(inputs, perm=[0, 4, 1, 2, 3])
        fft = tf.signal.fft3d(tf.cast(fft, tf.complex64))
        fft = tf.transpose(fft, perm=[0, 2, 3, 4, 1])

        fft = tf.pow(tf.math.abs(fft)+1e-10, gamma)
        fft = tf.signal.fftshift(fft)

        maxx = tf.reduce_max(fft, axis=(1, 2, 3), keepdims=True)
        fft = tf.math.divide_no_nan(fft, maxx)
        return fft

    def _rel_ratio(self, inputs):
        iotf = self._theoretical_otf(psf_shape=inputs.shape[1:-1])
        rel = tf.math.divide_no_nan(inputs, iotf)

        one = tf.constant(1., dtype=inputs.dtype)
        gt1 = tf.math.greater(rel, one)
        rel = tf.where(gt1, one, rel)
        rel = one - rel
        return rel

    def call(self, inputs, **kwargs):

        try:
            att = layers.Lambda(self._fft)(inputs)
            tf.debugging.check_numerics(att, message='Checking FFT')
        except Exception as e:
            logger.error(f'Tensor [{att}] had NaN values: {e}')

        try:
            att = layers.Lambda(self._rel_ratio)(att)
            tf.debugging.check_numerics(att, message='Checking REL-Ratio')
        except Exception as e:
            logger.error(f'Tensor [{att}] had NaN values: {e}')

        zplane = att[:, att.shape[1]//2, :, :, :]
        yplane = att[:, :, att.shape[2]//2, :, :]
        xplane = att[:, :, :, att.shape[3]//2, :]
        return tf.stack([zplane, yplane, xplane], axis=1)
