
import sys
import logging
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from activation import MaskedActivation

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Stem(layers.Layer):
    def __init__(
            self,
            filters=16,
            mask_shape=64,
            na_det=1.0,
            psf_type='widefield',
            lambda_det=.605,
            x_voxel_size=.15,
            y_voxel_size=.15,
            z_voxel_size=.6,
            refractive_index=1.33,
            activation='gelu',
            mul=False,
            **kwargs
    ):
        super(Stem, self).__init__(**kwargs)
        self.mask_shape = mask_shape
        self.na_det = na_det
        self.psf_type = psf_type
        self.lambda_det = lambda_det
        self.x_voxel_size = x_voxel_size
        self.y_voxel_size = y_voxel_size
        self.z_voxel_size = z_voxel_size
        self.refractive_index = refractive_index
        self.activation = activation
        self.mul = mul

        self.kernels = [3, 7]
        self.filters = filters // len(self.kernels)

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

        self.convs = [
            layers.Conv3D(filters=self.filters, kernel_size=(1, k, k), padding='same')
            for k in self.kernels
        ]

    def get_config(self):
        config = super(Stem, self).get_config()
        config.update({
            "filters": self.filters,
            "activation": self.activation,
            "mul": self.mul,
            "mask_shape": self.mask_shape,
            "psf_type": self.psf_type,
            "na_det": self.na_det,
            "lambda_det": self.lambda_det,
            "x_voxel_size": self.x_voxel_size,
            "y_voxel_size": self.y_voxel_size,
            "z_voxel_size": self.z_voxel_size,
            "refractive_index": self.refractive_index,
        })
        return config

    def build(self, input_shape):
        super(Stem, self).build(input_shape)

    def gaussian_filter3D(self, inputs, gkernel):
        outputs = []
        for i in range(inputs.shape[1]):
            filters = []
            for s in range(self.filters):
                s = tfa.image.gaussian_filter2d(
                    inputs[:, i],
                    filter_shape=(gkernel, gkernel),
                    sigma=gkernel/3,
                    padding='CONSTANT'
                )
                filters.append(s)
            outputs.append(layers.concatenate(filters, axis=-1))

        return tf.stack(outputs, axis=1)

    def call(self, inputs, training=True, **kwargs):
        embedding = []
        for gkernel, cc in zip(self.kernels, self.convs):
            alpha = cc(inputs[:, :3])
            alpha = self.act(alpha)

            phi = self.gaussian_filter3D(inputs[:, 3:], gkernel=gkernel)
            phi = self.act(phi)
            mu_alpha = tf.math.reduce_mean(alpha, axis=[2, 3], keepdims=True)
            std_alpha = tf.math.reduce_std(alpha, axis=[2, 3], keepdims=True)
            phi = (phi * mu_alpha) / (std_alpha + 1e-6)

            emb = layers.concatenate([alpha, phi], axis=1)

            if self.mul:
                emb = layers.multiply([alpha, phi])

            embedding.append(emb)

        m = layers.concatenate(embedding)
        return m
