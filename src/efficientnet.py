import logging
import sys
from abc import ABC

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from base import Base

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EfficientNet(Base, ABC):
    """
    Modified version of the original 2D EfficientNet

    EfficientNet V1: Rethinking Model Scaling for Convolutional Neural Networks:
        https://arxiv.org/abs/2104.00298

    EfficientNetV2: Smaller Models and Faster Training
        https://arxiv.org/abs/2104.00298

    TODO: NAS search for a baseline model
    """
    def __init__(
        self,
        depth_coefficient=1,
        width_coefficient=1,
        depth_divisor=8,
        dropout_rate=.3,
        se_ratio=.25,
        activation=tf.nn.relu6,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.depth_coefficient = depth_coefficient
        self.width_coefficient = width_coefficient
        self.depth_divisor = depth_divisor
        self.dropout_rate = dropout_rate
        self.activation = activation

        self.conv_kernel = {
            'class_name': 'VarianceScaling',
            'config': {
                'mode': 'fan_out',
                'scale': 2.0,
                'distribution': 'normal'
            }
        }

        self.fc_kernel = {
            'class_name': 'VarianceScaling',
            'config': {
                'mode': 'fan_out',
                'scale': 1./3,
                'distribution': 'uniform'
            }
        }

        self.fmb_blocks = [
            {
                'repeats': self._calc_repeats(1, self.depth_coefficient),
                'configs': dict(
                    input_filters=self._calc_filters(8, self.width_coefficient, self.depth_divisor),
                    output_filters=self._calc_filters(16, self.width_coefficient, self.depth_divisor),
                    kernel_size=3, strides=2, expand_ratio=1,
                    se_ratio=se_ratio, activation=activation,
                )
            },
            {
                'repeats': self._calc_repeats(1, self.depth_coefficient),
                'configs': dict(
                    input_filters=self._calc_filters(16, self.width_coefficient, self.depth_divisor),
                    output_filters=self._calc_filters(32, self.width_coefficient, self.depth_divisor),
                    kernel_size=3, strides=1, expand_ratio=4,
                    se_ratio=se_ratio, activation=activation,
                )
            },
            {
                'repeats': self._calc_repeats(1, self.depth_coefficient),
                'configs': dict(
                    input_filters=self._calc_filters(32, self.width_coefficient, self.depth_divisor),
                    output_filters=self._calc_filters(64, self.width_coefficient, self.depth_divisor),
                    kernel_size=3, strides=2, expand_ratio=6,
                    se_ratio=se_ratio, activation=activation,
                )
            },
            {
                'repeats': self._calc_repeats(1, self.depth_coefficient),
                'configs': dict(
                    input_filters=self._calc_filters(64, self.width_coefficient, self.depth_divisor),
                    output_filters=self._calc_filters(128, self.width_coefficient, self.depth_divisor),
                    kernel_size=3, strides=2, expand_ratio=6,
                    se_ratio=se_ratio, activation=activation,
                )
            },
        ]

        self.n_blocks = sum([block['repeats'] for block in self.fmb_blocks])

    def _calc_filters(self, filters, width_coefficient, depth_divisor):
        filters *= width_coefficient
        new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
        new_filters = max(depth_divisor, new_filters)

        if new_filters < 0.9 * filters:
            new_filters += depth_divisor
        return int(new_filters)

    def _calc_repeats(self, repeats, depth_coefficient):
        return int(np.ceil(depth_coefficient * repeats))

    def se_block(self, inputs, input_filters, activation, ratio=.25):
        """
        Squeeze-and-Excitation Networks:
            https://arxiv.org/abs/1709.01507
        """

        se = layers.GlobalAveragePooling3D()(inputs)
        se = layers.Reshape((1, 1, 1, input_filters))(se)

        se = layers.Conv3D(
            filters=max(1, int(input_filters * ratio)),
            kernel_size=1,
            activation=activation,
            padding='same',
            kernel_initializer=self.conv_kernel
        )(se)

        se = layers.Conv3D(
            filters=input_filters,
            kernel_size=1,
            activation='sigmoid',
            padding='same',
            kernel_initializer=self.conv_kernel
        )(se)

        return layers.multiply([inputs, se])

    def fused_mb_block(
        self,
        inputs,
        input_filters,
        output_filters,
        expand_ratio,
        se_ratio,
        strides,
        activation,
        kernel_size,
        dropout_rate,
    ):
        """
        MobileNetV2: Inverted Residuals and Linear Bottlenecks:
            https://arxiv.org/abs/1801.04381


        EfficientNet-EdgeTPU: Creating Accelerator-Optimized Neural Networks with AutoML:
            https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html
        """

        filters = input_filters * expand_ratio

        x = layers.Conv3D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            strides=strides,
            kernel_initializer=self.conv_kernel
        )(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)

        x = self.se_block(
            x,
            input_filters=filters,
            activation=activation,
            ratio=se_ratio
        )

        x = layers.Conv3D(
            filters=output_filters,
            kernel_size=1,
            padding="same",
            kernel_initializer=self.conv_kernel
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)

        x = layers.Dropout(dropout_rate)(x)
        return x

    def call(self, inputs, training=True, **kwargs):
        # Head
        m = self.conv3d_bn(
            inputs,
            filters=self._calc_filters(8, self.width_coefficient, self.depth_divisor),
            kernel_size=3,
            strides=1,
            activation=self.activation,
            kernel_initializer=self.conv_kernel
        )

        # MB blocks
        i = 0
        for block in self.fmb_blocks:
            for _ in range(block['repeats']):
                m = self.fused_mb_block(
                    m,
                    dropout_rate=.1 + (self.dropout_rate * i/self.n_blocks),
                    **block['configs']
                )
                i += 1

        # Output
        m = self.conv3d_bn(
            m,
            filters=self._calc_filters(128, self.width_coefficient, self.depth_divisor),
            kernel_size=1,
            strides=1,
            activation=self.activation,
            kernel_initializer=self.conv_kernel
        )
        m = self.global_avepool(m)
        m = self.fc(m, nodes=256, training=training, kernel_initializer=self.fc_kernel)
        return self.regressor(m)
