"""
Adopted 2D ConvNextV2 from pytorch to 3D ConvNextV2 in tensorflow

MIT License
=======================================================================

Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
=======================================================================

Attribution-NonCommercial 4.0 International

=======================================================================

Creative Commons is not a party to its public
licenses. Notwithstanding, Creative Commons may elect to apply one of
its public licenses to material it publishes and in those instances
will be considered the “Licensor.” The text of the Creative Commons
public licenses is dedicated to the public domain under the CC0 Public
Domain Dedication. Except for the limited purpose of indicating that
material is shared under a Creative Commons public license or as
otherwise permitted by the Creative Commons policies published at
creativecommons.org/policies, Creative Commons does not authorize the
use of the trademark "Creative Commons" or any other trademark or logo
of Creative Commons without its prior written consent including,
without limitation, in connection with any unauthorized modifications
to any of its public licenses or any other arrangements,
understandings, or agreements concerning use of licensed material. For
the avoidance of doubt, this paragraph does not form part of the
public licenses.

Creative Commons may be contacted at creativecommons.org.

"""


import logging
import sys

import tensorflow as tf
from tensorflow.keras import layers, Sequential

from base import Base

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GRN(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.epsilon = 1e-6

    def build(self, input_shape):

        self.gamma = self.add_weight(
            name=f"gamma_{self.name}",
            shape=(1, 1, 1, 1, input_shape[-1]),
            initializer=tf.zeros_initializer(),
        )
        self.beta = self.add_weight(
            name=f"beta_{self.name}",
            shape=(1, 1, 1, 1, input_shape[-1]),
            initializer=tf.zeros_initializer(),
        )
    
    def get_config(self):
        config = super(GRN, self).get_config()
        return config
    def call(self, inputs):
        gamma = tf.cast(self.gamma, tf.float32)
        beta = tf.cast(self.beta, tf.float32)
        x = tf.cast(inputs, tf.float32)
        
        Gx = tf.pow(
            (tf.reduce_sum(tf.pow(x, 2), axis=(1, 2, 3), keepdims=True) + self.epsilon),
            0.5,
        )
        Nx = Gx / tf.reduce_mean(Gx, axis=-1, keepdims=True) + self.epsilon
        
        result = gamma * (x * Nx) + beta + x
        return tf.cast(result, inputs.dtype)


class StochasticDepth(layers.Layer):
    """Stochastic Depth module.
    It is also referred to as Drop Path in `timm`.
    References:
        (1) github.com:rwightman/pytorch-image-models
    """

    def __init__(self, drop_path, **kwargs):
        super(StochasticDepth, self).__init__(**kwargs)
        self.drop_path = drop_path
    
    def get_config(self):
        config = super(StochasticDepth, self).get_config()
        config.update({
            "drop_path": self.drop_path,
        })
        return config
    
    def call(self, inputs, training=None):
        if training:
            keep_prob = 1 - self.drop_path
            shape = (tf.shape(inputs)[0],) + (1,) * (len(tf.shape(inputs)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (inputs / keep_prob) * random_tensor
        return inputs


class Block(layers.Layer):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dim = dim
        self.drop_path = drop_path
        
        self.dwconv = layers.Conv3D(dim, kernel_size=(1, 7, 7), padding="same", groups=dim)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.pwconv1 = layers.Dense(dim * 4)
        self.act = layers.Activation("gelu")
        self.grn = GRN()
        self.pwconv2 = layers.Dense(dim)
        
        self.drop_path = (
            StochasticDepth(drop_path)
            if drop_path > 0.0
            else layers.Activation("linear")
        )
    
    def build(self, input_shape):
        super(Block, self).build(input_shape)
    
    def get_config(self):
        config = super(Block, self).get_config()
        config.update({
            "dim": self.dim,
            "drop_path": self.drop_path,
        })
        return config
    
    def call(self, inputs, training=None):
        shortcut = inputs
        x = self.dwconv(inputs)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return shortcut + self.drop_path(x)


class Baseline(Base):
    def __init__(
        self,
        repeats=(3, 3, 27, 3),
        heads=(96, 192, 384, 768), # channel projections
        drop_path_rate=0.,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.heads = heads
        self.repeats = repeats
        self.drop_path_rate = drop_path_rate
        self.dp_rates = [d for d in tf.linspace(0.0, self.drop_path_rate, sum(self.repeats))]
        
    def call(self, inputs, **kwargs):

        x = inputs

        cur = 0
        for i in range(4):
            if i == 0:
                x = Sequential(
                    [
                        layers.Conv3D(self.heads[i], kernel_size=(1, 4, 4), strides=(1, 4, 4)),
                        layers.LayerNormalization(epsilon=1e-6),
                    ]
                )(x)
            else:
                x = Sequential(
                    [
                        layers.LayerNormalization(epsilon=1e-6),
                        layers.Conv3D(self.heads[i], kernel_size=(1, 2, 2), strides=(1, 2, 2)),
                    ]
                )(x)

            for j in range(self.repeats[i]):
                x = Block(
                    dim=self.heads[i],
                    drop_path=self.dp_rates[cur + j],
                )(x)
            cur += self.repeats[i]
        
        x = layers.GlobalAvgPool3D()(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return self.regressor(x)
