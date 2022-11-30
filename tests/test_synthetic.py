import sys
import numpy as np
sys.path.append("src")              # needed to get to the src folder of the project so that everything below will import from src

from synthetic import SyntheticPSF

wavelength = 0.515
lateral_voxel_size = 0.108
axial_voxel_size = 0.2


# any function that starts or ends with "test" will be discovered and run.
def test_PSF():
    psfgen = SyntheticPSF(
        snr=100,
        lam_detection=wavelength,
        x_voxel_size=lateral_voxel_size,
        y_voxel_size=lateral_voxel_size,
        z_voxel_size=axial_voxel_size,
    )

    psfgenGPU = SyntheticPSF(
        snr=100,
        lam_detection=wavelength,
        x_voxel_size=lateral_voxel_size,
        y_voxel_size=lateral_voxel_size,
        z_voxel_size=axial_voxel_size,
    )

    # just a placeholder for a test.  Maybe we check if a gpu implementation is identical or.
    assert np.array_equal(psfgen.theoretical_psf()      , psfgenGPU.theoretical_psf()     )


