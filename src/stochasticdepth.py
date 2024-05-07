import logging
import sys

import tensorflow as tf
from tensorflow.keras import layers

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StochasticDepth(layers.Layer):
    """
    Deep Networks with Stochastic Depth: https://arxiv.org/abs/1603.09382
    https://github.com/tensorflow/addons/blob/v0.20.0/tensorflow_addons/layers/stochastic_depth.py#L5-L90
    """
    def __init__(self, survival_probability: float = .5, **kwargs):
        super().__init__(**kwargs)
        self.survival_probability = survival_probability

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(StochasticDepth, self).get_config()
        config.update({
            "survival_probability": self.survival_probability,
        })
        return config

    def call(self, inputs, training=False, **kwargs):
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError("input must be a list of length 2.")

        shortcut, residual = inputs

        # Random bernoulli variable indicating whether the branch should be kept or not
        b_l = tf.keras.backend.random_bernoulli([], p=self.survival_probability, dtype=shortcut.dtype)

        def _call_train():
            return shortcut + b_l * residual

        def _call_test():
            return shortcut + self.survival_probability * residual

        return tf.keras.backend.in_train_phase(_call_train, _call_test, training=training)
