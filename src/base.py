
import logging
import sys
from abc import ABC

from tensorflow.keras import Model
from tensorflow.keras import layers

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Base(Model, ABC):

    def __init__(self, modes=15, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flat = layers.Flatten()
        self.regressor = layers.Dense(modes, activation='linear', name='regressor')
        self.classifier = layers.Dense(modes, activation='softmax', name='classifier')

    def build(self, input_shape):
        inputs = layers.Input(shape=input_shape)
        outputs = self.call(inputs)
        return Model(inputs=inputs, outputs=outputs, name=self.name)
