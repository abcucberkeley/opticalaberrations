import logging
import sys
from abc import ABC

import tensorflow as tf
from tensorflow.keras import layers
from depthwiseconv import DepthwiseConv3D
from base import Base

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
            **kwargs
    ):
        super().__init__(**kwargs)
        self.depth_scalar = depth_scalar
        self.width_scalar = width_scalar
        self.activation = activation
        self.avg = layers.GlobalAvgPool3D()

    def _calc_channels(self, channels, width_scalar):
        return int(tf.math.ceil(width_scalar * channels))

    def _calc_repeats(self, repeats, depth_scalar):
        return int(tf.math.ceil(depth_scalar * repeats))

    def _stem(self, inputs, filters):
        dwc3 = DepthwiseConv3D(kernel_size=(3, 3, 3), depth_multiplier=filters//2, padding='same')(inputs)
        dwc7 = DepthwiseConv3D(kernel_size=(7, 7, 7), depth_multiplier=filters//2, padding='same')(inputs)
        return layers.concatenate([dwc3, dwc7])

    def _attention(self, inputs, filters, ratio=.25):
        att = layers.GlobalAvgPool3D()(inputs)
        att = layers.Dense(max(1, int(filters * ratio)), activation=self.activation)(att)
        att = layers.Dense(filters, activation='sigmoid')(att)
        return layers.multiply([inputs, att])

    def _cab(self, inputs, filters, kernel_size, expansion=4):
        x = layers.LayerNormalization(axis=-1, epsilon=1e-6)(inputs)
        x = self._attention(x, filters=filters, ratio=.25)
        x = DepthwiseConv3D(kernel_size=kernel_size, depth_multiplier=1, padding='same')(x)
        x = layers.Conv3D(filters=filters*expansion, kernel_size=1, padding='same', activation=self.activation)(x)
        x = layers.Conv3D(filters=filters, kernel_size=1, padding='same')(x)
        return layers.add([inputs, x])

    def call(self, inputs, training=True, **kwargs):
        m = self._stem(inputs, filters=self._calc_channels(16, width_scalar=self.width_scalar))

        for i, r in enumerate([2, 4, 6, 2]):
            if i > 0:
                m = layers.Conv3D(
                    filters=self._calc_channels(2 * m.shape[-1], width_scalar=self.width_scalar),
                    strides=2,
                    kernel_size=1,
                )(m)

            res = m
            for _ in range(self._calc_repeats(r, depth_scalar=self.depth_scalar)):
                m = self._cab(m, kernel_size=(7, 7, 7), filters=m.shape[-1])
            m = layers.add([res, m])

        m = self.avg(m)
        return self.regressor(m)
