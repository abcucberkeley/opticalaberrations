import pytest
from pathlib import Path
from operator import floordiv
import numpy as np


@pytest.fixture(scope="session")
def kargs():
    repo = Path.cwd()
    num_modes = 15
    input_shape = (256, 256, 256)  # z-y-x
    window_size = (64, 64, 64)  # z-y-x
    num_tiles = np.prod(tuple(map(floordiv, input_shape, window_size)))
    digital_rotations = range(0, 361)

    kargs = dict(
        inputs=repo / f'examples/single/single.tif',
        input_shape=input_shape,
        embeddings_shape=(6, 64, 64, 1),
        digital_rotations=digital_rotations,
        rotations_shape=(len(digital_rotations), 6, 64, 64, 1),
        window_size=window_size,
        num_tiles=num_tiles,
        tiles_shape=(num_modes, num_tiles+5),
        num_modes=num_modes,
        zernikes_shape=(num_modes, 3),
        model=repo / f'pretrained_models/lattice_yumb_x108um_y108um_z200um/opticalnet-{num_modes}.h5',
        dm_calibration=repo/'calibration/aang/28_mode_calibration.csv',
        psf_type=repo/'lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat',
        pois=repo/f'examples/single/results/Detection3D.mat',
        ideal_psf=repo/'examples/psf.tif',
        prev=None,
        dm_state=None,
        wavelength=.510,
        dm_damping_scalar=1.0,
        lateral_voxel_size=.108,
        axial_voxel_size=.2,
        freq_strength_threshold=.01,
        prediction_threshold=0.,
        num_predictions=1,
        batch_size=128,
        plot=True,
        plot_rotations=True,
        ignore_modes=[],
        # extra `aggregate_predictions` flags
        majority_threshold=.5,
        min_percentile=1,
        max_percentile=99,
        final_prediction='mean',
        ignore_tile=[],
        # extra `predict_rois` flags
        num_rois=10,
        min_intensity=200,
        minimum_distance=.5,
    )

    return kargs