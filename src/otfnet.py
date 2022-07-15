
import logging
import sys
from abc import ABC

import tensorflow as tf
from tensorflow.keras import layers
from base import Base
from activation import MaskedActivation


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OTFNet(Base, ABC):

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
        init_channels=16,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mask_shape = mask_shape
        self.na_det = na_det
        self.lambda_det = lambda_det
        self.x_voxel_size = x_voxel_size
        self.y_voxel_size = y_voxel_size
        self.z_voxel_size = z_voxel_size
        self.refractive_index = refractive_index
        self.init_channels = init_channels
        self.depth_scalar = depth_scalar
        self.width_scalar = width_scalar
        self.activation = tf.nn.relu

    def _calc_channels(self, channels, width_scalar):
        return int(tf.math.ceil(width_scalar * channels))

    def _calc_repeats(self, repeats, depth_scalar):
        return int(tf.math.ceil(depth_scalar * repeats))

    def _transition(self, inputs, opt='strides', filters=None):
        iz, ih, iw = inputs.shape[1:-1]

        if opt == 'strides':
            return layers.Conv3D(filters=filters, strides=(1, 2, 2), kernel_size=(1, 1, 1))(inputs)

        elif opt == 'maxpool':
            return layers.MaxPooling3D(pool_size=(1, 2, 2))(inputs)

        elif opt == 'vsplit':
            return layers.Cropping3D(cropping=((0, 0), (0, 0), (0, iw//2)), name='vsplit')(inputs)

        elif opt == 'hsplit':
            return layers.Cropping3D(cropping=((0, 0), (0, ih//2), (0, 0)), name='hsplit')(inputs)

        else:
            return layers.AveragePooling3D(pool_size=(1, 2, 2))(inputs)

    def _convbn(self, inputs, filters, kernel, norm=False, padding='same'):

        x = layers.Conv3D(
            filters=filters,
            strides=1,
            dilation_rate=1,
            kernel_size=kernel,
            padding=padding,
            #kernel_initializer=self.conv_init,
        )(inputs)

        if norm:
            x = layers.LayerNormalization(axis=-1)(x)

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

        return x

    def _conv_block(self, inputs, filters):
        x = self._convbn(inputs, filters=filters, kernel=(1, 3, 3))
        x = self._convbn(x, filters=filters, kernel=(1, 3, 3))
        return x

    def _stem(self, inputs):
        m = MaskedActivation(
            activation=self.activation,
            mask_shape=self.mask_shape,
            na_det=self.na_det,
            refractive_index=self.refractive_index,
            lambda_det=self.lambda_det,
            x_voxel_size=self.x_voxel_size,
            y_voxel_size=self.y_voxel_size,
            z_voxel_size=self.z_voxel_size,
        )(inputs)
        m = self._conv_block(m, filters=self._calc_channels(self.init_channels, width_scalar=self.width_scalar))
        m = self._transition(m, opt='avgpool')
        return m

    def call(self, inputs, training=True, **kwargs):
        m = self._stem(inputs)

        for i in range(self._calc_repeats(repeats=3, depth_scalar=self.depth_scalar)):
            m = self._conv_block(
                m,
                filters=self._calc_channels(2*(i+1)*self.init_channels, width_scalar=self.width_scalar),
            )
            m = self._transition(m, opt='avgpool')

        m = self.flat(m)
        m = layers.Dense(64, activation=self.activation)(m)
        m = layers.Dense(64, activation=self.activation)(m)
        m = layers.Dense(64, activation=self.activation)(m)

        return self.regressor(m)
