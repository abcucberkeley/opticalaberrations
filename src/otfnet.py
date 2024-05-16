import logging
import sys

import torch
import torch.nn as nn

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OTFNet(nn.Module):

    def __init__(self, name='OTFNet', input_shape=(1, 6, 64, 64, 1), modes=15, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.input_shape = input_shape
        self.modes = modes
        self.kernel_size = (1, 3, 3)
        self.pool_size = (1, 2, 2)
        self.activation = nn.Tanh
        self.padding = 'same'

        
        self.features = nn.Sequential(
            nn.Conv3d(self.input_shape[-1], 8, kernel_size=self.kernel_size, padding=self.padding),
            self.activation(),
            nn.Conv3d(8, 8, kernel_size=self.kernel_size, padding=self.padding),
            self.activation(),
            nn.MaxPool3d(kernel_size=self.pool_size),

            nn.Conv3d(8, 16, kernel_size=self.kernel_size, padding=self.padding),
            self.activation(),
            nn.Conv3d(16, 16, kernel_size=self.kernel_size, padding=self.padding),
            self.activation(),
            nn.MaxPool3d(kernel_size=self.pool_size),

            nn.Conv3d(16, 32, kernel_size=self.kernel_size, padding=self.padding),
            self.activation(),
            nn.Conv3d(32, 32, kernel_size=self.kernel_size, padding=self.padding),
            self.activation(),
            nn.MaxPool3d(kernel_size=self.pool_size),

            nn.Conv3d(32, 64, kernel_size=self.kernel_size, padding=self.padding),
            self.activation(),
            nn.Conv3d(64, 64, kernel_size=self.kernel_size, padding=self.padding),
            self.activation(),
            nn.MaxPool3d(kernel_size=self.pool_size),

            nn.Conv3d(64, 128, kernel_size=self.kernel_size, padding=self.padding),
            self.activation(),
            nn.Conv3d(128, 128, kernel_size=self.kernel_size, padding=self.padding),
            self.activation(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            
            nn.Flatten(),
            nn.Linear(1536, 64),
            self.activation(),
            nn.Linear(64, 64),
            self.activation(),
        )
        
        self.regressor = nn.Linear(64, self.modes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.permute(x, (0, -1, 1, 2, 3))  # (B, Z, Y, X, C) -> (B, C, Z, Y, X)
        x = self.features(x)
        return self.regressor(x)
