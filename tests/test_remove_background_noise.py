
import sys
sys.path.append('.')
sys.path.append('./src')

import warnings
warnings.filterwarnings("ignore")

import logging
logger = logging.getLogger('')

from pathlib import Path
import numpy as np
import cupy as cp
from tifffile import imwrite
from src.synthetic import SyntheticPSF
from src.preprocessing import na_and_background_filter, dog


import imageio
import pytest

@pytest.mark.run(order=1)
def test_filter(kargs):
    """
    Will generate the filter response for removing background filters (dog and na_and_background_filter)

    """
    high_sigma = 3.0  # sets the low frequency cutoff.
    low_sigma  = 0.7

    # Create lattice SyntheticPSF so we can get the NA Mask
    samplepsfgen = SyntheticPSF(
        order='ansi',
        n_modes=kargs['num_modes'],
        distribution='mixed',
        mode_weights='pyramid',
        signed=True,
        rotate=True,
        psf_type='../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat',
        lam_detection=kargs['wavelength'],
        psf_shape=[64, 64, 64],
        x_voxel_size=kargs['lateral_voxel_size'],
        y_voxel_size=kargs['lateral_voxel_size'],
        z_voxel_size=kargs['axial_voxel_size'],
    )

    # Setup test images: "realsp" and "fourier"
    # real space image = a single voxel at the center of a volume.  This will create a uniform otf.
    realsp = cp.zeros(samplepsfgen.psf_shape)   # [64, 64, 64]
    center = np.array(realsp.shape) // 2
    realsp[center[2], center[1], center[0]] = 1
    realsp -= np.mean(realsp)                   # set DC frequency to zero so it doesn't overload otf.
    fourier = samplepsfgen.fft(realsp)

    base_folder = Path("./filter")
    base_folder.mkdir(exist_ok=True)
    imwrite(f'{base_folder}/fourier.tif', np.abs(cp.asnumpy(fourier)).astype(np.float32))
    imwrite(f'{base_folder}/realsp.tif', np.abs(cp.asnumpy(realsp)).astype(np.float32))

    print(f"\n\nUsing {high_sigma=}, {low_sigma=}")

    # filter the real space image. Remove DC.
    print(f'Testing "na_and_background_filter"...')
    FFTfiltered_realsp = na_and_background_filter(realsp, low_sigma=low_sigma, high_sigma=high_sigma, samplepsfgen=samplepsfgen)
    FFTfiltered_realsp -= np.mean(FFTfiltered_realsp)

    print(f'Testing "dog"...\n')
    dogfiltered_realsp = dog(realsp, low_sigma=low_sigma, high_sigma=high_sigma)
    dogfiltered_realsp -= np.mean(dogfiltered_realsp)

    imwrite(f'{base_folder}/FFTfiltered_realsp.tif', np.abs(cp.asnumpy(FFTfiltered_realsp)).astype(np.float32))
    imwrite(f'{base_folder}/dogfiltered_realsp.tif', np.abs(cp.asnumpy(dogfiltered_realsp)).astype(np.float32))

    FFTfourier = samplepsfgen.fft(FFTfiltered_realsp)
    dogfourier = samplepsfgen.fft(dogfiltered_realsp)

    FFTfiltered_otf = np.abs(cp.asnumpy(FFTfourier)).astype(np.float32)
    dogfiltered_otf = np.abs(cp.asnumpy(dogfourier)).astype(np.float32)

    imwrite(f'{base_folder}/FFTfiltered_otf.tif', FFTfiltered_otf)
    imwrite(f'{base_folder}/dogfiltered_otf.tif', dogfiltered_otf)
    print(f'3D Frequency supports saved to:\n'
            f'{Path(base_folder / "FFTfiltered_otf.tif").resolve()}\n'
            f'{Path(base_folder / "dogfiltered_otf.tif").resolve()}\n')

    # Save principle planes
    imageio.imsave(f'{base_folder}FFTfiltered_otf_XY.png', FFTfiltered_otf[center[2],:,:])
    imageio.imsave(f'{base_folder}/dogfiltered_otf_XY.png', dogfiltered_otf[center[2],:,:])
    imageio.imsave(f'{base_folder}/FFTfiltered_otf_XZ.png', FFTfiltered_otf[:,center[1],:])
    imageio.imsave(f'{base_folder}/dogfiltered_otf_XZ.png', dogfiltered_otf[:,center[1],:])
    imageio.imsave(f'{base_folder}/FFTfiltered_otf_YZ.png', np.transpose(FFTfiltered_otf[:,:,center[0]]))
    imageio.imsave(f'{base_folder}/dogfiltered_otf_YZ.png', np.transpose(dogfiltered_otf[:,:,center[0]]))

