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

from synthetic import SyntheticPSF

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fft(inputs, padsize=None, gaussian_filter=None):
    if padsize is not None:
        shape = inputs.shape[1]
        size = shape * (padsize / shape)
        pad = int((size - shape) // 2)
        inputs = np.pad(inputs, ((pad, pad), (pad, pad), (pad, pad)), 'constant', constant_values=0)

    otf = np.fft.fftn(inputs)
    otf = np.fft.fftshift(otf)

    phi = np.angle(otf)
    phi = np.unwrap(phi)
    alpha = np.abs(otf)

    if gaussian_filter is not None:
        alpha = filters.gaussian(alpha, gaussian_filter)
        phi = filters.gaussian(phi, gaussian_filter)

    alpha /= np.nanpercentile(alpha, 99.9)
    alpha[alpha > 1] = 1
    alpha = np.nan_to_num(alpha, nan=0)

    phi /= np.nanpercentile(phi, 99.9)
    phi[phi > 1] = 1
    phi[phi < -1] = -1
    phi = np.nan_to_num(phi, nan=0)

    return alpha, phi


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


def embedding(
    psf: np.array,
    psfgen: SyntheticPSF,
    crop_shape: np.array = (6, 64, 64),
    sample_voxel_size: np.array = (.1, .1, .1),
    na_mask: bool = True,
    ratio: bool = True,
    debug: Any = None,
    padsize: Any = None,
):
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
                m = ax.imshow(np.transpose(img[i]) if i == 2 else img[i], cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel(r'$\alpha = |\tau| / |\hat{\tau}|$')
                ax = fig.add_subplot(inner[1])
                ax.imshow(np.transpose(img[i+3]) if i == 2 else img[i+3], cmap='coolwarm', vmin=-1, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel(r'$\varphi = \angle \tau$')
                cls[i].axis('off')

        else:
            m = cls[0].imshow(np.max(img, axis=0), cmap='hot', vmin=0, vmax=1)
            cls[0].axis('off')
            cls[1].imshow(np.max(img, axis=1), cmap='hot', vmin=0, vmax=1)
            cls[1].axis('off')
            cls[2].imshow(np.max(img, axis=2).T, cmap='hot', vmin=0, vmax=1)
            cls[2].axis('off')

        cax = inset_axes(cls[2], width="10%", height="50%", loc='center right', borderpad=-2)
        cb = plt.colorbar(m, cax=cax)
        cax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    if psf.ndim == 4:
        psf = np.squeeze(psf)

    rescaled_psf = transform.resize(
        psf,
        (
            int(psf.shape[0] * (sample_voxel_size[0]/psfgen.voxel_size[0])),
            int(psf.shape[1] * (sample_voxel_size[1]/psfgen.voxel_size[1])),
            int(psf.shape[2] * (sample_voxel_size[2]/psfgen.voxel_size[2])),
        ),
        order=3
    )

    amp, phase = fft(rescaled_psf, padsize=padsize)

    middle_plane = amp.shape[1]//2
    emb = np.stack([
        amp[middle_plane, :, :],
        amp[:, middle_plane, :],
        amp[:, :, middle_plane],
        phase[middle_plane, :, :],
        phase[:, middle_plane, :],
        phase[:, :, middle_plane],
    ], axis=0)

    resized_emb = resize_with_crop_or_pad(emb, crop_shape)
    middle_plane = resized_emb.shape[1] // 2

    if ratio:
        resized_emb[:3] /= psfgen.iotf[middle_plane, middle_plane, middle_plane]
        resized_emb[:3] = np.nan_to_num(resized_emb[:3], nan=0)

    if na_mask:
        mask = psfgen.na_mask()
        resized_emb[:3] *= mask[middle_plane, middle_plane, middle_plane]
        resized_emb[3:] *= mask[middle_plane, middle_plane, middle_plane]

    if debug is not None:
        fig, axes = plt.subplots(4, 3, figsize=(8, 11))

        axes[0, 1].set_title(f"Input (maxproj) {str(psf.shape)} @ {sample_voxel_size}")
        plot(axes[0, :], psf)

        axes[1, 1].set_title(f"Resampled (maxproj) {str(rescaled_psf.shape)} @ {psfgen.voxel_size}")
        plot(axes[1, :], rescaled_psf)
        imsave(f'{debug}_rescaled_psf.tif', rescaled_psf)

        axes[2, 1].set_title(f"Embeddings {str(emb.shape)} @ {psfgen.voxel_size}")
        plot(axes[2, :], emb)
        imsave(f'{debug}_embeddings.tif', emb)

        axes[-1, 1].set_title(f"Resized embeddings {str(resized_emb.shape)}")
        plot(axes[-1, :], resized_emb)
        imsave(f'{debug}_resized_embeddings.tif', resized_emb)

        if debug == True:
            plt.show()
        else:
            plt.savefig(f'{debug}_rescaling.png', dpi=300, bbox_inches='tight', pad_inches=.25)

    if psf.ndim == 4:
        return np.expand_dims(resized_emb, axis=-1)
    else:
        return resized_emb
