import logging
import sys
from abc import ABC

import tensorflow as tf
from tensorflow.keras import layers

from activation import MaskedActivation
from base import Base
from depthwiseconv import DepthwiseConv3D
from spatial import SpatialAttention
from stem import Stem
from stochasticdepth import StochasticDepth

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CAB(layers.Layer):
    def __init__(
            self,
            filters,
            mask_shape=64,
            na_det=1.0,
            psf_type='widefield',
            lambda_det=.605,
            x_voxel_size=.15,
            y_voxel_size=.15,
            z_voxel_size=.6,
            refractive_index=1.33,
            depth_multiplier=1,
            expansion=4,
            dropout_rate=.1,
            activation='gelu',
            **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.mask_shape = mask_shape
        self.na_det = na_det
        self.lambda_det = lambda_det
        self.psf_type = psf_type
        self.x_voxel_size = x_voxel_size
        self.y_voxel_size = y_voxel_size
        self.z_voxel_size = z_voxel_size
        self.refractive_index = refractive_index
        self.activation = activation
        self.depth_multiplier = depth_multiplier
        self.expansion = int(expansion)
        self.dropout_rate = dropout_rate

        self.dwc3 = DepthwiseConv3D(kernel_size=(1, 3, 3), depth_multiplier=self.depth_multiplier, padding='same')
        self.dwc7 = DepthwiseConv3D(kernel_size=(1, 7, 7), depth_multiplier=self.depth_multiplier, padding='same')

        self.expand = layers.Conv3D(filters=self.expansion*self.filters, kernel_size=1, padding='same')
        self.conv = layers.Conv3D(filters=self.filters, kernel_size=1, padding='same')

        self.ln = layers.LayerNormalization(axis=-1, epsilon=1e-6)

        self.drop_path = StochasticDepth(survival_probability=1-self.dropout_rate)

        self.sca = SpatialAttention(
            channels=self.filters,
            ratio=.25,
            activation=self.activation,
            mask_shape=self.mask_shape,
            na_det=self.na_det,
            psf_type=self.psf_type,
            refractive_index=self.refractive_index,
            lambda_det=self.lambda_det,
            x_voxel_size=self.x_voxel_size,
            y_voxel_size=self.y_voxel_size,
            z_voxel_size=self.z_voxel_size,
        )

        self.act = MaskedActivation(
            activation=self.activation,
            mask_shape=self.mask_shape,
            na_det=self.na_det,
            psf_type=self.psf_type,
            refractive_index=self.refractive_index,
            lambda_det=self.lambda_det,
            x_voxel_size=self.x_voxel_size,
            y_voxel_size=self.y_voxel_size,
            z_voxel_size=self.z_voxel_size,
        )

    def build(self, input_shape):
        super(CAB, self).build(input_shape)

    def get_config(self):
        config = super(CAB, self).get_config()
        config.update({
            "filters": self.filters,
            "mask_shape": self.mask_shape,
            "na_det": self.na_det,
            "psf_type": self.psf_type,
            "refractive_index": self.refractive_index,
            "x_voxel_size": self.x_voxel_size,
            "y_voxel_size": self.y_voxel_size,
            "z_voxel_size": self.z_voxel_size,
            "activation": self.activation,
            "depth_multiplier": self.depth_multiplier,
            "expansion": self.expansion,
            "dropout_rate": self.dropout_rate,
        })
        return config

    def call(self, inputs, training=False, **kwargs):
        x = self.ln(inputs)
        x = self.sca(x)
        x = layers.concatenate([self.dwc3(x), self.dwc7(x)])

        x = self.expand(x)
        x = self.act(x)
        x = self.conv(x)
        return self.drop_path([inputs, x], training=training)


class TB(layers.Layer):
    def __init__(
            self,
            filters,
            opt='strides',
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.opt = opt

        self.maxpool = layers.MaxPooling3D(pool_size=(1, 2, 2))
        self.avgpool = layers.AveragePooling3D(pool_size=(1, 2, 2))
        self.strides = layers.Conv3D(filters=self.filters, strides=(1, 2, 2), kernel_size=1)
        self.conv = layers.Conv3D(filters=self.filters, strides=1, kernel_size=1, padding='same')

    def build(self, input_shape):
        super(TB, self).build(input_shape)

        iz, ih, iw = input_shape[1:-1]
        self.vsplit = layers.Cropping3D(cropping=((0, 0), (0, 0), (0, iw // 2)), name='vsplit')
        self.hsplit = layers.Cropping3D(cropping=((0, 0), (0, ih // 2), (0, 0)), name='hsplit')

    def get_config(self):
        config = super(TB, self).get_config()
        config.update({
            "filters": self.filters,
            "opt": self.opt,
        })
        return config

    def call(self, inputs, **kwargs):

        if self.opt == 'strides':
            x = self.strides(inputs)

        elif self.opt == 'vsplit':
            x = self.vsplit(inputs)
            x = self.conv(x)

        elif self.opt == 'hsplit':
            x = self.hsplit(inputs)
            x = self.conv(x)

        elif self.opt == 'maxpool':
            x = self.maxpool(inputs)
            x = self.conv(x)

        else:
            x = self.avgpool(inputs)
            x = self.conv(x)

        return x


class OpticalResNet(Base, ABC):
    def __init__(
            self,
            depth_scalar=1.0,
            width_scalar=1.0,
            activation='gelu',
            dropout_rate=0.05,
            mul=False,
            no_phase=False,
            mask_shape=64,
            na_det=1.0,
            psf_type='widefield',
            lambda_det=.605,
            x_voxel_size=.15,
            y_voxel_size=.15,
            z_voxel_size=.6,
            refractive_index=1.33,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.depth_scalar = depth_scalar
        self.width_scalar = width_scalar
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.avg = layers.GlobalAvgPool3D()
        self.mul = mul
        self.no_phase = no_phase
        self.mask_shape = mask_shape
        self.na_det = na_det
        self.psf_type = psf_type
        self.lambda_det = lambda_det
        self.x_voxel_size = x_voxel_size
        self.y_voxel_size = y_voxel_size
        self.z_voxel_size = z_voxel_size
        self.refractive_index = refractive_index

    def _calc_channels(self, channels, width_scalar):
        return int(tf.math.ceil(width_scalar * channels))

    def _calc_repeats(self, repeats, depth_scalar):
        return int(tf.math.ceil(depth_scalar * repeats))

    def call(self, inputs, training=False, **kwargs):

        m = Stem(
            filters=self._calc_channels(24, width_scalar=self.width_scalar),
            activation=self.activation,
            mask_shape=self.mask_shape,
            na_det=self.na_det,
            psf_type=self.psf_type,
            refractive_index=self.refractive_index,
            lambda_det=self.lambda_det,
            x_voxel_size=self.x_voxel_size,
            y_voxel_size=self.y_voxel_size,
            z_voxel_size=self.z_voxel_size,
            mul=self.mul,
            no_phase=self.no_phase
        )(inputs)

        for i, r in enumerate([2, 4, 6, 2]):
            if i > 0:
                m = TB(
                    opt="strides",
                    filters=self._calc_channels(2 * m.shape[-1], width_scalar=self.width_scalar),
                )(m)

            res = m
            for _ in range(self._calc_repeats(r, depth_scalar=self.depth_scalar)):
                m = CAB(
                    filters=m.shape[-1],
                    activation=self.activation,
                    mask_shape=self.mask_shape,
                    na_det=self.na_det,
                    psf_type=self.psf_type,
                    refractive_index=self.refractive_index,
                    lambda_det=self.lambda_det,
                    x_voxel_size=self.x_voxel_size,
                    y_voxel_size=self.y_voxel_size,
                    z_voxel_size=self.z_voxel_size,
                    dropout_rate=self.dropout_rate/(i+1)
                )(m, training=training)
            m = layers.add([res, m])

        m = self.avg(m)
        return self.regressor(m)
