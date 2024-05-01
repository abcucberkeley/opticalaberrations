import logging
import sys
"""ConvNeXt models for Keras.

References:

- [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
  (CVPR 2022)
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras import initializers
from tensorflow.keras import layers, Sequential

from base import Base

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================



BASE_DOCSTRING = """Instantiates the {name} architecture.

  References:
    - [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
    (CVPR 2022)

  For image classification use cases, see
  [this page for detailed examples](
  https://keras.io/api/applications/#usage-examples-for-image-classification-models).
  For transfer learning use cases, make sure to read the
  [guide to transfer learning & fine-tuning](
    https://keras.io/guides/transfer_learning/).

  The `base`, `large`, and `xlarge` models were first pre-trained on the
  ImageNet-21k dataset and then fine-tuned on the ImageNet-1k dataset. The
  pre-trained parameters of the models were assembled from the
  [official repository](https://github.com/facebookresearch/ConvNeXt). To get a
  sense of how these parameters were converted to Keras compatible parameters,
  please refer to
  [this repository](https://github.com/sayakpaul/keras-convnext-conversion).

  Note: Each Keras Application expects a specific kind of input preprocessing.
  For ConvNeXt, preprocessing is included in the model using a `Normalization`
  layer.  ConvNeXt models expect their inputs to be float or uint8 tensors of
  pixels with values in the [0-255] range.

  When calling the `summary()` method after instantiating a ConvNeXt model,
  prefer setting the `expand_nested` argument `summary()` to `True` to better
  investigate the instantiated model.

  Returns:
    A `keras.Model` instance.
"""


class StochasticDepth(layers.Layer):
    """Stochastic Depth module.

    It performs batch-wise dropping rather than sample-wise. In libraries like
    `timm`, it's similar to `DropPath` layers that drops residual paths
    sample-wise.

    References:
      - https://github.com/rwightman/pytorch-image-models

    Args:
      drop_path_rate (float): Probability of dropping paths. Should be within
        [0, 1].

    Returns:
      Tensor either with the residual path dropped or kept.
    """

    def __init__(self, drop_path_rate, **kwargs):
        super().__init__(**kwargs)
        self.drop_path_rate = drop_path_rate

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path_rate
            # Random bernoulli variable indicating whether the branch should be kept or not or not
            b_l = tf.keras.backend.random_bernoulli([], p=keep_prob, dtype=x.dtype)
            return x * b_l
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({"drop_path_rate": self.drop_path_rate})
        return config


class LayerScale(layers.Layer):
    """Layer scale module.

    References:
      - https://arxiv.org/abs/2103.17239

    Args:
      init_values (float): Initial value for layer scale. Should be within
        [0, 1].
      projection_dim (int): Projection dimensionality.

    Returns:
      Tensor multiplied to the scale.
    """

    def __init__(self, init_values, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="gamma",
            shape=(self.projection_dim,),
            initializer=initializers.Constant(self.init_values),
            trainable=True,
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_values": self.init_values,
                "projection_dim": self.projection_dim,
            }
        )
        return config


def ConvNeXtBlock(
    projection_dim,
    drop_path_rate=0.0,
    layer_scale_init_value=1e-6,
):
    """ConvNeXt block.

    References:
    - https://arxiv.org/abs/2201.03545
    - https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

    Notes:
      In the original ConvNeXt implementation (linked above), the authors use
      `Dense` layers for pointwise convolutions for increased efficiency.
      Following that, this implementation also uses the same.

    Args:
      projection_dim (int): Number of filters for convolution layers. In the
        ConvNeXt paper, this is referred to as projection dimension.
      drop_path_rate (float): Probability of dropping paths. Should be within
        [0, 1].
      layer_scale_init_value (float): Layer scale value. Should be a small float
        number.
      name: name to path to the keras layer.

    Returns:
      A function representing a ConvNeXtBlock block.
    """
    def apply(inputs):
        x = inputs

        x = layers.Conv3D(
            filters=projection_dim,
            kernel_size=7,
            padding="same",
            groups=projection_dim,
        )(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Dense(4 * projection_dim)(x)
        x = layers.Activation("gelu")(x)
        x = layers.Dense(projection_dim)(x)

        if layer_scale_init_value is not None:
            x = LayerScale(
                layer_scale_init_value,
                projection_dim,
            )(x)
        if drop_path_rate:
            layer = StochasticDepth(drop_path_rate)
        else:
            layer = layers.Activation("linear")

        return inputs + layer(x)

    return apply


class Baseline(Base):
    def __init__(
        self,
        repeats=(3, 3, 27, 3),
        heads=(128, 256, 512, 1024), # channel projections
        layer_scale_init_value=1e-6,
        dropath_rate=0.,
        num_stages=4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.heads = heads
        self.repeats = repeats
        self.layer_scale_init_value = layer_scale_init_value
        self.num_stages = num_stages
        
        # Stochastic depth schedule.
        # This is referred from the original ConvNeXt codebase:
        # https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py#L86
        self.depth_drop_rates = [
            float(x) for x in np.linspace(0.0, dropath_rate, sum(repeats))
        ]
        
        self.stem = Sequential(
            [
                layers.Conv3D(
                    heads[0],
                    kernel_size=(1, 8, 8),
                    strides=(1, 8, 8),
                ),
                layers.LayerNormalization(
                    epsilon=1e-6,
                ),
            ],
        )
        
        self.norm = layers.LayerNormalization(axis=-1, epsilon=1e-6)
        self.avg = layers.GlobalAveragePooling3D()
        
    def call(self, inputs, **kwargs):

        m = inputs
        
        cur = 0
        for i in range(self.num_stages):
            if i == 0:
                m = self.stem(m)
            else:
                m = layers.LayerNormalization(epsilon=1e-6)(m)
                m = layers.Conv3D(self.heads[i], kernel_size=(1, 2, 2), strides=(1, 2, 2))(m)
        
            for j in range(self.repeats[i]):
                m = ConvNeXtBlock(
                    projection_dim=self.heads[i],
                    drop_path_rate=self.depth_drop_rates[cur + j],
                    layer_scale_init_value=self.layer_scale_init_value,
                )(m)
            cur += self.repeats[i]
        
        m = self.norm(m)
        m = self.avg(m)
        return self.regressor(m)
