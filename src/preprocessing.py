import logging
import sys
from pathlib import Path
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import gridspec
from tifffile import imread, imsave
from skimage import transform, filters
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


def resize(vol, voxel_size: tuple, crop_shape: tuple, sample_voxel_size: tuple = (.1, .1, .1), debug=False):
    def plot(cls, img):
        if img.shape[0] == 6:
            vmin, vmax, vcenter, step = 0, 2, 1, .1
            highcmap = plt.get_cmap('YlOrRd', 256)
            lowcmap = plt.get_cmap('YlGnBu_r', 256)
            low = np.linspace(0, 1 - step, int(abs(vcenter - vmin) / step))
            high = np.linspace(0, 1 + step, int(abs(vcenter - vmax) / step))
            cmap = np.vstack((lowcmap(low), [1, 1, 1, 1], highcmap(high)))
            cmap = mcolors.ListedColormap(cmap)

            for i in range(3):
                inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=cls[i], wspace=0.1, hspace=0.1)
                ax = fig.add_subplot(inner[0])
                m = ax.imshow(img[i], cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel(r'$\alpha = |\tau| / |\hat{\tau}|$')
                ax = fig.add_subplot(inner[1])
                ax.imshow(img[i+3], cmap='coolwarm', vmin=vmin, vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel(r'$\varphi = \angle \tau$')
                cls[i].axis('off')
        else:
            m = cls[0].imshow(np.max(img, axis=0), cmap='hot', vmin=0, vmax=1)
            cls[1].imshow(np.max(img, axis=1), cmap='hot', vmin=0, vmax=1)
            cls[2].imshow(np.max(img, axis=2).T, cmap='hot', vmin=0, vmax=1)

        cax = inset_axes(cls[2], width="10%", height="50%", loc='center right', borderpad=-2)
        cb = plt.colorbar(m, cax=cax)
        cax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    resampled_vol = transform.rescale(
        vol,
        (
            sample_voxel_size[0]/voxel_size[0],
            sample_voxel_size[1]/voxel_size[1],
            sample_voxel_size[2]/voxel_size[2],
        ),
        order=3,
        anti_aliasing=True,
    )
    resized_psf = resize_with_crop_or_pad(resampled_vol, crop_shape)

    if debug is not None:
        fig, axes = plt.subplots(3, 3, figsize=(8, 11))

        axes[0, 1].set_title(f"{str(vol.shape)} @ {sample_voxel_size}")
        axes[0, 0].set_ylabel('Input (maxproj)')
        plot(axes[0, :], vol)

        axes[1, 1].set_title(f"{str(resampled_vol.shape)} @ {voxel_size}")
        axes[1, 0].set_ylabel('Resampled (maxproj)')
        plot(axes[1, :], resampled_vol)
        imsave(f'{debug}_resampled_psf.tif', resampled_vol)

        axes[2, 1].set_title(str(resized_psf.shape))
        axes[2, 0].set_ylabel('Resized (maxproj)')
        plot(axes[2, :], resized_psf)
        imsave(f'{debug}_resized_psf.tif', resized_psf)

        if debug == True:
            plt.show()
        else:
            plt.savefig(f'{debug}_rescaling.png', dpi=300, bbox_inches='tight', pad_inches=.25)

    return resized_psf


def prep_psf(path: Path, input_shape: tuple, voxel_size: tuple):
    psf = imread(path)
    psf = psf.transpose(0, 2, 1)
    psf = psf / np.max(psf)
    psf = resize(psf, voxel_size=voxel_size, crop_shape=tuple(3*[input_shape[-1]]))
    return psf
