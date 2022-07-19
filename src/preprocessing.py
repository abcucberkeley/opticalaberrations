import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tifffile import imread
from skimage import transform
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def resize_with_crop_or_pad(psf: np.array, crop_shape: tuple, **kwargs):
    rank = len(crop_shape)
    index = [[0, psf.shape[d]] for d in range(rank)]
    pad = [[0, 0] for _ in range(rank)]
    slicer = [slice(None)] * rank

    for i in range(rank):
        if psf.shape[i] < crop_shape[i]:
            pad[i][0] = (crop_shape[i] - psf.shape[i]) // 2
            pad[i][1] = crop_shape[i] - psf.shape[i] - pad[i][0]
        else:
            index[i][0] = int(np.floor((psf.shape[i] - crop_shape[i]) / 2.))
            index[i][1] = index[i][0] + crop_shape[i]

        slicer[i] = slice(index[i][0], index[i][1])
    return np.pad(psf[slicer], pad, **kwargs)


def resize(vol, voxel_size: tuple, crop_shape: tuple, reference_voxel_size: tuple = (.1, .1, .1), debug=False):
    resampled_vol = transform.rescale(
        vol,
        (
            reference_voxel_size[0]/voxel_size[0],
            reference_voxel_size[1]/voxel_size[1],
            reference_voxel_size[2]/voxel_size[2],
        ),
        order=3
    )
    resized_psf = resize_with_crop_or_pad(resampled_vol, crop_shape)

    if debug:
        vol = vol ** .5
        vol = np.nan_to_num(vol)

        fig, axes = plt.subplots(3, 3, figsize=(8, 11))

        axes[0, 1].set_title(str(vol.shape))
        m = axes[0, 0].imshow(vol[vol.shape[0]//2, :, :], cmap='hot', vmin=0, vmax=1)
        axes[0, 1].imshow(vol[:, vol.shape[1]//2, :], cmap='hot', vmin=0, vmax=1)
        axes[0, 2].imshow(vol[:, :, vol.shape[2]//2], cmap='hot', vmin=0, vmax=1)
        cax = inset_axes(axes[0, 2], width="10%", height="100%", loc='center right', borderpad=-2)
        cb = plt.colorbar(m, cax=cax)
        cax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axes[0, 0].set_ylabel('Input')

        axes[1, 1].set_title(str(resampled_vol.shape))
        m = axes[1, 0].imshow(resampled_vol[resampled_vol.shape[0]//2, :, :], cmap='hot', vmin=0, vmax=1)
        axes[1, 1].imshow(resampled_vol[:, resampled_vol.shape[1]//2, :], cmap='hot', vmin=0, vmax=1)
        axes[1, 2].imshow(resampled_vol[:, :, resampled_vol.shape[2]//2], cmap='hot', vmin=0, vmax=1)
        cax = inset_axes(axes[1, 2], width="10%", height="100%", loc='center right', borderpad=-2)
        cb = plt.colorbar(m, cax=cax)
        cax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axes[1, 0].set_ylabel('Resampled')

        axes[2, 1].set_title(str(resized_psf.shape))
        m = axes[2, 0].imshow(resized_psf[resized_psf.shape[0]//2, :, :], cmap='hot', vmin=0, vmax=1)
        axes[2, 1].imshow(resized_psf[:, resized_psf.shape[1]//2, :], cmap='hot', vmin=0, vmax=1)
        axes[2, 2].imshow(resized_psf[:, :, resized_psf.shape[2]//2], cmap='hot', vmin=0, vmax=1)
        cax = inset_axes(axes[2, 2], width="10%", height="100%", loc='center right', borderpad=-2)
        cb = plt.colorbar(m, cax=cax)
        cax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axes[2, 0].set_ylabel('Resized')
        plt.show()

    return resized_psf


def prep_psf(path: Path, input_shape: tuple, voxel_size: tuple):
    psf = imread(path)
    psf = psf.transpose(0, 2, 1)
    psf = psf / np.max(psf)
    psf = resize(psf, voxel_size=voxel_size, crop_shape=tuple(3*[input_shape[-1]]))
    return psf
