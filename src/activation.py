import logging
import sys

import numpy as np
from tensorflow.keras import layers
import tensorflow as tf

from synthetic import SyntheticPSF

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MaskedActivation(layers.Layer):

    def __init__(
        self,
        activation='relu',
        mask_shape=64,
        na_det=1.0,
        psf_type='widefield',
        lambda_det=.510,
        x_voxel_size=.108,
        y_voxel_size=.108,
        z_voxel_size=.2,
        refractive_index=1.33,
        **kwargs
    ):
        super(MaskedActivation, self).__init__(**kwargs)
        self.activation = str(activation).lower()
        self.mask_shape = mask_shape
        self.na_det = na_det
        self.lambda_det = lambda_det
        self.psf_type = psf_type
        self.x_voxel_size = x_voxel_size
        self.y_voxel_size = y_voxel_size
        self.z_voxel_size = z_voxel_size
        self.refractive_index = refractive_index

    def get_config(self):
        config = super(MaskedActivation, self).get_config()
        config.update({
            "activation": self.activation,
            "psf_type": self.psf_type,
            "na_det": self.na_det,
            "lambda_det": self.lambda_det,
            "x_voxel_size": self.x_voxel_size,
            "y_voxel_size": self.y_voxel_size,
            "z_voxel_size": self.z_voxel_size,
            "refractive_index": self.refractive_index,
            "mask_shape": self.mask_shape,
        })
        return config

    def build(self, input_shape):
        super(MaskedActivation, self).build(input_shape)
        self.mask = self._theoretical_mask(mask_shape=3*[self.mask_shape], planes=input_shape[1])

    def _theoretical_mask(self, mask_shape, planes):
        psfgen = SyntheticPSF(
            n_modes=55,
            amplitude_ranges=0,
            psf_shape=mask_shape,
            dtype=self.psf_type,
            lam_detection=self.lambda_det,
            x_voxel_size=self.x_voxel_size,
            y_voxel_size=self.y_voxel_size,
            z_voxel_size=self.z_voxel_size,
            snr=1000,
            max_jitter=0,
        )

        mask = psfgen.na_mask()

        if planes == 3:
            mask = np.stack([
                mask[mask.shape[0] // 2, :, :],
                mask[:, mask.shape[1] // 2, :],
                mask[:, :, mask.shape[2] // 2],
            ], axis=0)
        else:
            mask = np.stack([
                mask[mask.shape[0] // 2, :, :],
                mask[:, mask.shape[1] // 2, :],
                mask[:, :, mask.shape[2] // 2],
                mask[mask.shape[0] // 2, :, :],
                mask[:, mask.shape[1] // 2, :],
                mask[:, :, mask.shape[2] // 2],
            ], axis=0)

        mask = tf.convert_to_tensor(mask, dtype=tf.float32)
        mask = tf.expand_dims(tf.expand_dims(mask, axis=0), axis=-1)
        return mask

    def call(self, inputs, **kwargs):
        mask = self.mask

        if self.mask_shape != inputs.shape[-2]:
            mask = layers.AveragePooling3D(
                pool_size=(1, self.mask_shape//inputs.shape[-3], self.mask_shape//inputs.shape[-2]),
            )(mask)

        x = layers.multiply([inputs, mask])

        if self.activation == 'none':
            return x
        else:
            return layers.Activation(self.activation)(x)
