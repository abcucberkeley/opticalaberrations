import logging
import sys
from abc import ABC

import tensorflow as tf
from tensorflow.keras import layers

from base import Base
from depthwiseconv import DepthwiseConv3D
from stochasticdepth import StochasticDepth

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Baseline(Base, ABC):
    def __init__(
            self,
            depth_scalar=1.0,
            width_scalar=1.0,
            activation='gelu',
            dropout_rate=.1,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.depth_scalar = depth_scalar
        self.width_scalar = width_scalar
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.avg = layers.GlobalAvgPool3D()

    def _calc_channels(self, channels, width_scalar):
        return int(tf.math.ceil(width_scalar * channels))

    def _calc_repeats(self, repeats, depth_scalar):
        return int(tf.math.ceil(depth_scalar * repeats))

    def _block(self, inputs, filters, kernel_size, expansion=4):
        x = DepthwiseConv3D(kernel_size=kernel_size, depth_multiplier=1, padding='same')(inputs)
        x = layers.LayerNormalization(axis=-1, epsilon=1e-6)(x)
        x = layers.Conv3D(filters=filters*expansion, kernel_size=1, padding='same', activation=self.activation)(x)
        x = layers.Conv3D(filters=filters, kernel_size=1, padding='same')(x)
        return StochasticDepth(survival_probability=1-self.dropout_rate)([inputs, x])

    def call(self, inputs, training=True, **kwargs):
        m = layers.Conv3D(
            filters=self._calc_channels(96, width_scalar=self.width_scalar),
            kernel_size=(1, 4, 4),
            strides=(1, 4, 4),
            padding="VALID",
        )(inputs)
        m = layers.LayerNormalization(axis=-1, epsilon=1e-6)(m)

        for i, r in enumerate([2, 2, 2, 2]):
            if i > 0:
                m = layers.Conv3D(
                    filters=self._calc_channels(2 * m.shape[-1], width_scalar=self.width_scalar),
                    strides=(1, 2, 2),
                    kernel_size=(1, 2, 2),
                )(m)

            res = m
            for _ in range(self._calc_repeats(r, depth_scalar=self.depth_scalar)):
                m = self._block(m, kernel_size=(1, 7, 7), filters=m.shape[-1])
            m = layers.add([res, m])
        
        m = layers.LayerNormalization(axis=-1, epsilon=1e-6)(m)
        m = self.avg(m)
        return self.regressor(m)
