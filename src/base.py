
import logging
import sys
import torch
import torch.nn as nn
from functools import reduce
 
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Base(nn.Module):
    
    def __init__(self, input_shape=(1, 6, 64, 64, 1), modes=15, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        in_features = reduce(lambda x, y: x*y, input_shape)
        self.regressor = nn.Linear(
            in_features=in_features, 
            out_features=modes
        )

    def forward(self, x):
        x = torch.flatten(x)
        x = self.regressor(x)
        return x
