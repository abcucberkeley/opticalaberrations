import logging
import sys
from abc import ABC

import tensorflow as tf
from tensorflow.keras import layers
from base import Base
from activation import MaskedActivation
from depthwiseconv import DepthwiseConv3D
from spatial import SpatialAttention
from stem import Stem

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Baseline(Base, ABC):
    def __init__(
            self,
            mask_shape=64,
            na_det=1.0,
            lambda_det=.605,
            x_voxel_size=.15,
            y_voxel_size=.15,
            z_voxel_size=.6,
            refractive_index=1.33,
            depth_scalar=1.0,
            width_scalar=1.0,
            activation='gelu',
            mul=False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.mask_shape = mask_shape
        self.na_det = na_det
        self.lambda_det = lambda_det
        self.x_voxel_size = x_voxel_size
        self.y_voxel_size = y_voxel_size
        self.z_voxel_size = z_voxel_size
        self.refractive_index = refractive_index
        self.depth_scalar = depth_scalar
        self.width_scalar = width_scalar
        self.activation = activation
        self.mul = mul

        self.avg = layers.GlobalAvgPool3D()

    def _calc_channels(self, channels, width_scalar):
        return int(tf.math.ceil(width_scalar * channels))

    def _calc_repeats(self, repeats, depth_scalar):
        return int(tf.math.ceil(depth_scalar * repeats))

    def _cab(self, inputs, filters, expansion=4):
        x = layers.LayerNormalization(axis=-1, epsilon=1e-6)(inputs)

        x = SpatialAttention(
            channels=filters,
            ratio=.25,
            activation=self.activation,
            mask_shape=self.mask_shape,
            na_det=self.na_det,
            refractive_index=self.refractive_index,
            lambda_det=self.lambda_det,
            x_voxel_size=self.x_voxel_size,
            y_voxel_size=self.y_voxel_size,
            z_voxel_size=self.z_voxel_size,
        )(x)

        x = DepthwiseConv3D(kernel_size=(1, 3, 3), depth_multiplier=1, padding='same')(x)

        x = layers.Conv3D(filters=filters*expansion, kernel_size=1, padding='same')(x)
        x = MaskedActivation(
            activation=self.activation,
            mask_shape=self.mask_shape,
            na_det=self.na_det,
            refractive_index=self.refractive_index,
            lambda_det=self.lambda_det,
            x_voxel_size=self.x_voxel_size,
            y_voxel_size=self.y_voxel_size,
            z_voxel_size=self.z_voxel_size,
        )(x)

        x = layers.Conv3D(filters=filters, kernel_size=1, padding='same')(x)

        return layers.add([inputs, x])

    def call(self, inputs, training=True, **kwargs):
        m = Stem(
            filters=self._calc_channels(24, width_scalar=self.width_scalar),
            activation=self.activation,
            mask_shape=self.mask_shape,
            na_det=self.na_det,
            refractive_index=self.refractive_index,
            lambda_det=self.lambda_det,
            x_voxel_size=self.x_voxel_size,
            y_voxel_size=self.y_voxel_size,
            z_voxel_size=self.z_voxel_size,
            mul=self.mul
        )(inputs)

        for i, r in enumerate([2, 4, 6, 2]):
            if i > 0:
                m = layers.Conv3D(
                    filters=self._calc_channels(2 * m.shape[-1], width_scalar=self.width_scalar),
                    strides=(1, 2, 2),
                    kernel_size=1,
                )(m)

            res = m
            for _ in range(self._calc_repeats(r, depth_scalar=self.depth_scalar)):
                m = self._cab(m, filters=m.shape[-1])
            m = layers.add([res, m])

        m = self.avg(m)
        return self.regressor(m)
