import logging
import sys

from tensorflow.keras import layers

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ROI(layers.Layer):
    def __init__(self, crop_shape, **kwargs):
        super().__init__(**kwargs)
        self.crop_shape = crop_shape

    def build(self, input_shape):
        super(ROI, self).build(input_shape)
        new_shape = tuple((i - s) // 2 for s, i in zip(self.crop_shape, input_shape[1:-1]))
        self.crop = layers.Cropping3D(
            cropping=new_shape,
            name='center_crop'
        )

    def get_config(self):
        config = super(ROI, self).get_config()
        config.update({
            "crop_shape": self.crop_shape,
        })
        return config

    def call(self, inputs, training=True, **kwargs):
        return self.crop(inputs)
