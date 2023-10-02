
from pathlib import Path
import os
import numpy as np
from tifffile import TiffFile, imwrite, imread, TiffWriter
from experimental import decon

folder = Path(r'U:\Data\TestsForThayer\20230928_fish_clta_mNG')
cam_A = 'CamA'
cam_B = 'CamB'

completed_list = []
before_files = folder.rglob(f'before*{cam_A}*stack0000_*00??t_tiles_predictions.csv')
for file in before_files:
    print(f'\n------------------- Decon tiles from : {file}')
    completed_list += decon(
            model_pred=file,  # predictions  _tiles_predictions.csv
            iters=30,
            plot=False,
    )
print('Done. Output files:')
print(completed_list)