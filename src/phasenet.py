
import logging
import sys
from abc import ABC

from tensorflow.keras import layers
from base import Base

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PhaseNet(Base, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = (3, 3, 3)
        self.pool_size = (1, 2, 2)
        self.activation = 'tanh'
        self.padding = 'same'

    def call(self, inputs, training=True, **kwargs):
        m = layers.Conv3D(8, kernel_size=self.kernel_size, activation=self.activation, padding=self.padding)(inputs)
        m = layers.Conv3D(8, kernel_size=self.kernel_size, activation=self.activation, padding=self.padding)(m)
        m = layers.MaxPooling3D(pool_size=self.pool_size)(m)

        m = layers.Conv3D(16, kernel_size=self.kernel_size, activation=self.activation, padding=self.padding)(m)
        m = layers.Conv3D(16, kernel_size=self.kernel_size, activation=self.activation, padding=self.padding)(m)
        m = layers.MaxPooling3D(pool_size=self.pool_size)(m)

        m = layers.Conv3D(32, kernel_size=self.kernel_size, activation=self.activation, padding=self.padding)(m)
        m = layers.Conv3D(32, kernel_size=self.kernel_size, activation=self.activation, padding=self.padding)(m)
        m = layers.MaxPooling3D(pool_size=self.pool_size)(m)

        m = layers.Conv3D(64, kernel_size=self.kernel_size, activation=self.activation, padding=self.padding)(m)
        m = layers.Conv3D(64, kernel_size=self.kernel_size, activation=self.activation, padding=self.padding)(m)
        m = layers.MaxPooling3D(pool_size=self.pool_size)(m)

        m = layers.Conv3D(128, kernel_size=self.kernel_size, activation=self.activation, padding=self.padding)(m)
        m = layers.Conv3D(128, kernel_size=self.kernel_size, activation=self.activation, padding=self.padding)(m)

        if inputs.shape[0] == 1:
            m = layers.MaxPooling3D(pool_size=(1, 2, 2))(m)
        else:
            m = layers.MaxPooling3D(pool_size=(2, 2, 2))(m)

        m = self.flat(m)
        m = layers.Dense(64, activation=self.activation)(m)
        m = layers.Dense(64, activation=self.activation)(m)
        return self.regressor(m)
