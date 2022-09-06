
import sys
import logging
import tensorflow as tf
from tensorflow.keras import layers
from activation import MaskedActivation

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SpatialAttention(layers.Layer):
    """
        Squeeze-and-Attention Networks for Semantic Segmentation
            https://arxiv.org/abs/1909.03402

        Squeeze-and-Excitation Networks:
            https://arxiv.org/abs/1709.01507
    """

    def __init__(
            self,
            channels,
            ratio=.25,
            mask_shape=64,
            na_det=1.0,
            lambda_det=.605,
            psf_type='widefield',
            x_voxel_size=.15,
            y_voxel_size=.15,
            z_voxel_size=.6,
            refractive_index=1.33,
            depth_scalar=1.0,
            width_scalar=1.0,
            activation='gelu',
            **kwargs
    ):
        super(SpatialAttention, self).__init__(**kwargs)
        self.channels = channels
        self.ratio = ratio
        self.mask_shape = mask_shape
        self.na_det = na_det
        self.lambda_det = lambda_det
        self.psf_type = psf_type
        self.x_voxel_size = x_voxel_size
        self.y_voxel_size = y_voxel_size
        self.z_voxel_size = z_voxel_size
        self.refractive_index = refractive_index
        self.depth_scalar = depth_scalar
        self.width_scalar = width_scalar
        self.activation = activation

        self.mask = MaskedActivation(
            activation=None,
            mask_shape=self.mask_shape,
            na_det=self.na_det,
            refractive_index=self.refractive_index,
            lambda_det=self.lambda_det,
            psf_type=self.psf_type,
            x_voxel_size=self.x_voxel_size,
            y_voxel_size=self.y_voxel_size,
            z_voxel_size=self.z_voxel_size,
        )

        self.squeeze = layers.Dense(
            max(1, int(self.channels * self.ratio)),
            activation=self.activation
        )

        self.excitation = layers.Dense(
            self.channels,
            activation='sigmoid'
        )

    def global_average(self, inputs):
        weights = tf.math.reduce_mean(inputs, axis=[2, 3])
        return layers.Reshape((inputs.shape[1], 1, 1, self.channels))(weights)

    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        config.update({
            "channels": self.channels,
            "ratio": self.ratio,
            "activation": self.activation,
            "mask_shape": self.mask_shape,
            "na_det": self.na_det,
            "lambda_det": self.lambda_det,
            "psf_type": self.psf_type,
            "x_voxel_size": self.x_voxel_size,
            "y_voxel_size": self.y_voxel_size,
            "z_voxel_size": self.z_voxel_size,
            "refractive_index": self.refractive_index,
            "depth_scalar": self.depth_scalar,
            "width_scalar": self.width_scalar,
        })
        return config

    def build(self, input_shape):
        super(SpatialAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        att = self.mask(inputs)
        att = self.global_average(att)
        att = self.squeeze(att)
        att = self.excitation(att)
        return layers.multiply([inputs, att])
