import logging
import sys
sys.path.append('.')
sys.path.append('./src')

import warnings
warnings.filterwarnings("ignore")

import math
import pytest
import numpy as np
import cupy as cp
from pathlib import Path
from tifffile import imwrite
import matplotlib.image as plt

from src import backend
from src import preprocessing
from src.synthetic import SyntheticPSF
from src.preprocessing import na_and_background_filter, dog
from src.utils import fft


@pytest.mark.run(order=1)
def test_psnr(kargs):
    sample = backend.load_sample(kargs['inputs'])

    psnr = preprocessing.prep_sample(
        sample,
        remove_background=True,
        return_psnr=True,
        plot=None,
        normalize=False,
    )
    assert math.isclose(psnr, 30, rel_tol=1)


@pytest.mark.run(order=2)
def test_remove_background_noise(kargs):
    """
    Will generate the filter response for removing background filters (dog and na_and_background_filter)

    """
    high_sigma = 3.0  # sets the low frequency cutoff.
    low_sigma = 0.7

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
    na_mask = samplepsfgen.na_mask()

    # Setup test images: "realsp" and "fourier"
    # real space image = a single voxel at the center of a volume.  This will create a uniform otf.
    realsp = cp.zeros(samplepsfgen.psf_shape)   # [64, 64, 64]
    center = np.array(realsp.shape) // 2
    realsp[center[2], center[1], center[0]] = 1
    realsp -= np.mean(realsp)                   # set DC frequency to zero so it doesn't overload otf.
    fourier = fft(realsp)

    base_folder = Path(f"{kargs['repo']}/preprocessing")
    base_folder.mkdir(exist_ok=True)
    imwrite(f'{base_folder}/fourier.tif', np.abs(cp.asnumpy(fourier)).astype(np.float32), compression='deflate')
    imwrite(f'{base_folder}/realsp.tif', np.abs(cp.asnumpy(realsp)).astype(np.float32), compression='deflate')

    logging.info(f"Using {high_sigma=}, {low_sigma=}")
    logging.info(f'Testing "dog" on CPU...')
    dogfiltered_realsp = dog(cp.asnumpy(realsp), low_sigma=low_sigma, high_sigma=high_sigma, min_psnr=1)
    dogfiltered_realsp -= np.mean(cp.asnumpy(dogfiltered_realsp))

    logging.info(f'Testing "dog" on GPU...')
    dogfiltered_realsp_GPU = dog(realsp, low_sigma=low_sigma, high_sigma=high_sigma, min_psnr=1)
    dogfiltered_realsp_GPU -= cp.mean(dogfiltered_realsp_GPU)

    logging.info(f'Checking if GPU "dog" and CPU "dog" agree...')
    np.testing.assert_allclose(cp.asnumpy(dogfiltered_realsp_GPU), dogfiltered_realsp, rtol=0, atol=1e-7)

    # filter the real space image. Remove DC.
    logging.info(f'Testing "na_and_background_filter"...')
    FFTfiltered_realsp = na_and_background_filter(
        realsp, low_sigma=low_sigma, high_sigma=high_sigma, na_mask=na_mask, min_psnr=1
    )
    FFTfiltered_realsp -= np.mean(FFTfiltered_realsp)

    imwrite(f'{base_folder}/FFTfiltered_realsp.tif', np.abs(cp.asnumpy(FFTfiltered_realsp)).astype(np.float32), compression='deflate')
    imwrite(f'{base_folder}/dogfiltered_realsp.tif', np.abs(cp.asnumpy(dogfiltered_realsp)).astype(np.float32), compression='deflate')

    FFTfourier = fft(FFTfiltered_realsp)
    dogfourier = fft(dogfiltered_realsp)

    FFTfiltered_otf = np.abs(cp.asnumpy(FFTfourier)).astype(np.float32)
    dogfiltered_otf = np.abs(cp.asnumpy(dogfourier)).astype(np.float32)

    imwrite(f'{base_folder}/FFTfiltered_otf.tif', FFTfiltered_otf, compression='deflate')
    imwrite(f'{base_folder}/dogfiltered_otf.tif', dogfiltered_otf, compression='deflate')
    logging.info(f'3D Frequency supports saved to: {Path(base_folder / "FFTfiltered_otf.tif").resolve()}')
    logging.info(f'3D Frequency supports saved to: {Path(base_folder / "dogfiltered_otf.tif").resolve()}')

    # Save principle planes
    plt.imsave(f'{base_folder}/FFTfiltered_otf_XY.png', FFTfiltered_otf[center[2], :, :])
    plt.imsave(f'{base_folder}/dogfiltered_otf_XY.png', dogfiltered_otf[center[2], :, :])
    plt.imsave(f'{base_folder}/FFTfiltered_otf_XZ.png', FFTfiltered_otf[:, center[1], :])
    plt.imsave(f'{base_folder}/dogfiltered_otf_XZ.png', dogfiltered_otf[:, center[1], :])
    plt.imsave(f'{base_folder}/FFTfiltered_otf_YZ.png', np.transpose(FFTfiltered_otf[:, :, center[0]]))
    plt.imsave(f'{base_folder}/dogfiltered_otf_YZ.png', np.transpose(dogfiltered_otf[:, :, center[0]]))


@pytest.mark.run(order=3)
def test_preprocessing(kargs):
    sample_voxel_size = (
        kargs['axial_voxel_size'],
        kargs['lateral_voxel_size'],
        kargs['lateral_voxel_size']
    )
    sample = backend.load_sample(kargs['inputs'])

    sample = preprocessing.prep_sample(
        sample,
        sample_voxel_size=sample_voxel_size,
        remove_background=True,
        normalize=True,
        plot=kargs['inputs'].with_suffix('') if kargs['plot'] else None,
    )
    assert sample.shape == kargs['input_shape']

