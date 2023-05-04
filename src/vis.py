import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.set_loglevel('error')

import warnings
from pathlib import Path
from functools import partial
import logging
import sys
import itertools
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import ImageGrid
from numpy.lib.stride_tricks import sliding_window_view
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from typing import Any, Union, Optional
import numpy as np
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from line_profiler_pycharm import profile
from matplotlib import colors

from wavefront import Wavefront
import re


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


def savesvg(
    fig: plt.Figure,
    savepath: Union[Path, str],
    top: float = 0.9,
    bottom: float = 0.1,
    left: float = 0.1,
    right: float = 0.9,
    hspace: float = 0.35,
    wspace: float = 0.1
):

    plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)
    plt.savefig(savepath, bbox_inches='tight', dpi=300, pad_inches=.25)

    if Path(savepath).suffix == '.svg':
        # Read in the file
        with open(savepath, 'r') as f:
            filedata = f.read()

        # Replace the target string
        filedata = re.sub('height="[0-9]+(\.[0-9]+)pt"', '', filedata)
        filedata = re.sub('width="[0-9]+(\.[0-9]+)pt"', '', filedata)

        # Write the file out again
        with open(savepath, 'w') as f:
            f.write(filedata)


def plot_mip(xy, xz, yz, vol, label='', gamma=.5, cmap='hot', dxy=.108, dz=.2, colorbar=True, aspect=None):
    def formatter(x, pos, dd):
        return f'{np.ceil(x * dd).astype(int):1d}'

    vol = vol ** gamma
    vol = np.nan_to_num(vol)

    if xy is not None:
        m = xy.imshow(np.max(vol, axis=0), cmap=cmap, aspect=aspect)
        xy.yaxis.set_ticks_position('right')
        xy.xaxis.set_major_formatter(partial(formatter, dd=dxy))
        xy.yaxis.set_major_formatter(partial(formatter, dd=dxy))
        xy.xaxis.set_major_locator(plt.MaxNLocator(6))
        xy.yaxis.set_major_locator(plt.MaxNLocator(6))
        xy.set_xlabel('XY ($\mu$m)')

    if xz is not None:
        m = xz.imshow(np.max(vol, axis=1), cmap=cmap, aspect=aspect)
        xz.yaxis.set_ticks_position('right')
        xz.xaxis.set_major_formatter(partial(formatter, dd=dxy))
        xz.yaxis.set_major_formatter(partial(formatter, dd=dz))
        xz.xaxis.set_major_locator(plt.MaxNLocator(6))
        xz.yaxis.set_major_locator(plt.MaxNLocator(6))
        xz.set_xlabel('XZ ($\mu$m)')

    if yz is not None:
        m = yz.imshow(np.max(vol, axis=2), cmap=cmap, aspect=aspect)
        yz.yaxis.set_ticks_position('right')
        yz.xaxis.set_major_formatter(partial(formatter, dd=dxy))
        yz.yaxis.set_major_formatter(partial(formatter, dd=dz))
        yz.xaxis.set_major_locator(plt.MaxNLocator(6))
        yz.yaxis.set_major_locator(plt.MaxNLocator(6))
        yz.set_xlabel('YZ ($\mu$m)')

    if colorbar:
        if xy is not None:
            cax = inset_axes(xy, width="10%", height="100%", loc='center left', borderpad=-5)
        elif xz is not None:
            cax = inset_axes(xz, width="10%", height="100%", loc='center left', borderpad=-5)
        else:
            cax = inset_axes(yz, width="10%", height="100%", loc='center left', borderpad=-5)

        cb = plt.colorbar(m, cax=cax)
        cax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        cax.set_ylabel(f"{label}")
        cax.yaxis.set_label_position("left")

    return m


def plot_wavefront(
    iax,
    phi,
    label=None,
    nas=(.65, .75, .85, .95),
    vcolorbar=False,
    hcolorbar=False,
    vmin=None,
    vmax=None,
):
    def formatter(x, pos):
        val_str = '{:.1g}'.format(x)
        if np.abs(x) > 0 and np.abs(x) < 1:
            return val_str.replace("0", "", 1)
        else:
            return val_str

    def na_mask(radius):
        center = (int(phi.shape[0]/2), int(phi.shape[1]/2))
        Y, X = np.ogrid[:phi.shape[0], :phi.shape[1]]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
        mask = dist_from_center <= radius
        return mask

    dlimit = .05
    step = .025

    if vmin is None:
        vmin = np.floor(np.nanmin(phi)*2)/4     # round down to nearest 0.25 wave
        vmin = -1*dlimit if vmin > -0.01 else vmin

    if vmax is None:
        vmax = np.ceil(np.nanmax(phi)*2)/4  # round up to nearest 0.25 wave
        vmax = dlimit if vmax < 0.01 else vmax

    # highcmap = plt.get_cmap('magma_r', 256)
    # middlemap = plt.get_cmap('gist_gray', 256)
    # lowcmap = plt.get_cmap('gist_earth_r', 256)
    #
    # ll = np.arange(vmin, -1*dlimit+step, step)
    # hh = np.arange(dlimit, vmax+step, step)

    # wave_cmap = np.vstack((
    #     lowcmap(.66 * ll / ll.min()),
    #     middlemap([.8, .9, 1, .9, .8]),
    #     highcmap(.66 * hh / hh.max())
    # ))
    # wave_cmap = mcolors.ListedColormap(wave_cmap)

    mat = iax.imshow(
        phi,
        cmap='Spectral_r',
        vmin=vmin,
        vmax=vmax,
    )

    pcts = []
    for d in nas:
        r = (d * phi.shape[0]) / 2
        circle = patches.Circle((50, 50), r, ls='--', ec="dimgrey", fc="none", zorder=3)
        iax.add_patch(circle)

        mask = phi * na_mask(radius=r)
        pcts.append((np.nanquantile(mask, .05), np.nanquantile(mask, .95)))

    phi = phi.flatten()

    if label is not None:
        p2v = abs(np.nanmin(phi) - np.nanmax(phi))
        err = '\n'.join([
            f'$NA_{{{na:.2f}}}$={abs(p[1]-p[0]):.2f}$\lambda$'
            for na, p in zip(nas, pcts)
        ])
        iax.set_title(f'{label}\n{err}')
        iax.set_title(f'{label} [{p2v:.2f}$\lambda$]\n{err}')

    iax.axis('off')
    iax.set_aspect("equal")

    if vcolorbar:
        cax = inset_axes(iax, width="10%", height="100%", loc='center right', borderpad=-3)
        cbar = plt.colorbar(mat, cax=cax, extend='both', format=formatter)
        cbar.ax.set_title(r'$\lambda$', pad=10)
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.yaxis.set_label_position('left')

    if hcolorbar:
        cax = inset_axes(iax, width="100%", height="10%", loc='lower center', borderpad=-1)
        cbar = plt.colorbar(mat, cax=cax, extend='both', format=formatter, orientation='horizontal')
        cbar.ax.xaxis.set_ticks_position('bottom')
        cbar.ax.xaxis.set_label_position('top')

    return mat


def diagnostic_assessment(
        psf: np.array,
        gt_psf: np.array,
        predicted_psf: np.array,
        corrected_psf: np.array,
        psnr: Any,
        maxcounts: Any,
        y: Wavefront,
        pred: Wavefront,
        save_path: Path,
        y_lls_defocus: Any = None,
        p_lls_defocus: Any = None,
        display: bool = False,
        psf_cmap: str = 'hot',
        gamma: float = .5,
        bar_width: float = .35,
        dxy: float = .108,
        dz: float = .2,
        pltstyle: Any = None,
        transform_to_align_to_DM: bool = False,
):
    if pltstyle is not None: plt.style.use(pltstyle)

    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.autolimit_mode': 'round_numbers'
    })

    if len(psf.shape) > 3:
        psf = np.squeeze(psf, axis=-1)
        psf = np.squeeze(psf, axis=0)

    if not np.isscalar(psnr):
        psnr = psnr[0]

    if not np.isscalar(maxcounts):
        maxcounts = maxcounts[0]

    y_wave = y.wave(size=100)
    pred_wave = pred.wave(size=100)
    diff = y_wave - pred_wave

    fig = plt.figure(figsize=(17, 17))
    gs = fig.add_gridspec(5 if gt_psf is None else 6, 4)

    ax_gt = fig.add_subplot(gs[:2, 0])
    ax_pred = fig.add_subplot(gs[:2, 1])
    ax_diff = fig.add_subplot(gs[:2, 2])
    cax = fig.add_axes([0.05, 0.7, 0.02, .175])

    # input
    ax_xy = fig.add_subplot(gs[2, 0])
    ax_xz = fig.add_subplot(gs[2, 1])
    ax_yz = fig.add_subplot(gs[2, 2])

    # predictions
    ax_pxy = fig.add_subplot(gs[-2, 0])
    ax_pxz = fig.add_subplot(gs[-2, 1])
    ax_pyz = fig.add_subplot(gs[-2, 2])

    # corrected
    ax_cxy = fig.add_subplot(gs[-1, 0])
    ax_cxz = fig.add_subplot(gs[-1, 1])
    ax_cyz = fig.add_subplot(gs[-1, 2])

    if p_lls_defocus is not None:
        ax_zcoff = fig.add_subplot(gs[:-1, -1])
        ax_defocus = fig.add_subplot(gs[-1, -1])
    else:
        ax_zcoff = fig.add_subplot(gs[:, -1])

    dlimit = .25    #hardcap the extreme limits to 0.25

    vmin = np.floor(np.nanmin(y_wave) * 2) / 2  # round down to nearest 0.5 wave
    vmin = -1 * dlimit if vmin > -0.01 else vmin

    vmax = np.ceil(np.nanmax(y_wave) * 2) / 2  # round up to nearest 0.5 wave
    vmax = dlimit if vmax < 0.01 else vmax

    mat = plot_wavefront(ax_gt, y_wave, label='Ground truth', vmin=vmin, vmax=vmax)
    plot_wavefront(ax_pred, pred_wave, label='Predicted', vmin=vmin, vmax=vmax)
    plot_wavefront(ax_diff, diff, label='Residuals', vmin=vmin, vmax=vmax)

    cbar = fig.colorbar(
        mat,
        cax=cax,
        fraction=0.046,
        pad=0.04,
        extend='both',
        format=FormatStrFormatter("%.2g"),
        # spacing='proportional',
    )
    cbar.ax.set_title(r'$\lambda$', pad=20)
    cbar.ax.yaxis.set_ticks_position('left')

    ax_cxy.set_xlabel(r'XY ($\mu$m)')
    ax_cxz.set_xlabel(r'XZ ($\mu$m)')
    ax_cyz.set_xlabel(r'YZ ($\mu$m)')
    ax_xz.set_title(f"PSNR: {psnr:.2f}")
    ax_yz.set_title(f"Max photon count: {maxcounts:.0f}")

    if transform_to_align_to_DM:
        psf = np.transpose(np.rot90(psf, k=2, axes=(1, 2)), axes=(0, 2, 1))    # 180 rotate, then transpose

    plot_mip(
        xy=ax_xy,
        xz=ax_xz,
        yz=ax_yz,
        vol=psf,
        label=f'Input (MIP) [$\gamma$={gamma}]',
        cmap=psf_cmap,
        dxy=dxy,
        dz=dz
    )
    plot_mip(
        xy=ax_pxy,
        xz=ax_pxz,
        yz=ax_pyz,
        vol=predicted_psf,
        label=f'Predicted [$\gamma$={gamma}]',
        cmap=psf_cmap,
        dxy=dxy,
        dz=dz
    )
    plot_mip(
        xy=ax_cxy,
        xz=ax_cxz,
        yz=ax_cyz,
        vol=corrected_psf,
        label=f'Corrected [$\gamma$={gamma}]',
        cmap=psf_cmap,
        dxy=dxy,
        dz=dz
    )

    if gt_psf is not None:
        ax_xygt = fig.add_subplot(gs[3, 0])
        ax_xzgt = fig.add_subplot(gs[3, 1])
        ax_yzgt = fig.add_subplot(gs[3, 2])
        plot_mip(
            xy=ax_xygt,
            xz=ax_xzgt,
            yz=ax_yzgt,
            vol=gt_psf,
            label=f'Simulated [$\gamma$={gamma}]')

    ax_zcoff.barh(
        np.arange(len(pred.amplitudes)) - bar_width / 2,
        pred.amplitudes,
        capsize=10,
        alpha=.75,
        color='C0',
        align='center',
        ecolor='C0',
        label='Predictions',
        height=bar_width
    )
    ax_zcoff.barh(
        np.arange(len(y.amplitudes)) + bar_width / 2,
        y.amplitudes,
        capsize=10,
        alpha=.75,
        color='C1',
        align='center',
        ecolor='C1',
        label='Ground truth',
        height=bar_width
    )
    ax_zcoff.set_ylim((-1, len(pred.amplitudes)))
    ax_zcoff.set_yticks(range(0, len(pred.amplitudes)))
    ax_zcoff.spines.right.set_visible(False)
    ax_zcoff.spines.left.set_visible(False)
    ax_zcoff.spines.top.set_visible(False)
    ax_zcoff.grid(True, which="both", axis='x', lw=1, ls='--', zorder=0)
    ax_zcoff.set_xlabel(r'Zernike coefficients ($\mu$m RMS)')
    handles, labels = ax_zcoff.get_legend_handles_labels()
    ax_zcoff.legend(reversed(handles), reversed(labels), frameon=False, loc='upper center', bbox_to_anchor=(.5, 1.05))
    ax_zcoff.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_zcoff.yaxis.tick_right()

    if p_lls_defocus is not None:
        ax_defocus.barh(
            [1 - bar_width / 2],
            [p_lls_defocus],
            capsize=10,
            alpha=.75,
            color='C0',
            align='center',
            ecolor='C0',
            label='Predictions',
            height=bar_width
        )
        ax_defocus.barh(
            [1 + bar_width / 2],
            [y_lls_defocus],
            capsize=10,
            alpha=.75,
            color='C1',
            align='center',
            ecolor='C1',
            label='Ground truth',
            height=bar_width
        )

        ax_defocus.set_xlabel(f'LLS defocus ($\mu$m)')
        ax_defocus.spines['top'].set_visible(False)
        ax_defocus.spines['left'].set_visible(False)
        ax_defocus.spines['right'].set_visible(False)
        ax_defocus.set_yticks([])
        ax_defocus.grid(True, which="both", axis='x', lw=1, ls='--', zorder=0)
        ax_defocus.axvline(0, ls='-', color='k')
        ax_defocus.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    for ax in [ax_gt, ax_pred, ax_diff]:
        ax.axis('off')

    savesvg(fig, f'{save_path}.svg')

    if display:
        plt.tight_layout()
        plt.show()


@profile
def diagnosis(pred: Wavefront, save_path: Path, pred_std: Any = None, lls_defocus: float = 0.):

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.autolimit_mode': 'round_numbers'
    })

    pred_wave = pred.wave(size=100)

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(4, 3)

    if lls_defocus != 0.:
        ax_wavefront = fig.add_subplot(gs[:-1, -1])
        ax_zcoff = fig.add_subplot(gs[:, :-1])
    else:
        ax_wavefront = fig.add_subplot(gs[:, -1])
        ax_zcoff = fig.add_subplot(gs[:, :-1])

    plot_wavefront(
        ax_wavefront,
        pred_wave,
        label='Predicted wavefront',
        vcolorbar=True,
    )

    if pred_std is not None:
        ax_zcoff.bar(
            range(len(pred.amplitudes)),
            pred.amplitudes,
            yerr=pred_std.amplitudes,
            capsize=2,
            color='dimgrey',
            alpha=.75,
            align='center',
            ecolor='lightgrey',
        )
    else:
        ax_zcoff.bar(
            range(len(pred.amplitudes)),
            pred.amplitudes,
            capsize=2,
            color='dimgrey',
            alpha=.75,
            align='center',
            ecolor='k',
        )

    ax_zcoff.set_ylabel(f'Zernike coefficients ($\mu$m RMS)')
    ax_zcoff.spines['top'].set_visible(False)
    ax_zcoff.spines['left'].set_visible(False)
    ax_zcoff.spines['right'].set_visible(False)
    ax_zcoff.grid(True, which="both", axis='y', lw=1, ls='--', zorder=0)
    ax_zcoff.set_xticks(range(len(pred.amplitudes)), minor=True)
    ax_zcoff.set_xticks(range(0, len(pred.amplitudes)+5, min(5, int(np.ceil(len(pred.amplitudes)+5)/8))), minor=False) # at least 8 ticks
    ax_zcoff.set_xlim((-.5, len(pred.amplitudes)))
    ax_zcoff.axhline(0, ls='--', color='r', alpha=.5)

    if lls_defocus != 0.:
        ax_defocus = fig.add_subplot(gs[-1, -1])

        data = [lls_defocus]
        bars = ax_defocus.barh(range(len(data)), data)

        for bar in bars:
            bar.set_zorder(1)
            bar.set_facecolor("none")
            x, y = bar.get_xy()
            w, h = bar.get_width(), bar.get_height()
            grad = np.atleast_2d(np.linspace(0, 1 * w / max(data), 256))
            ax_defocus.imshow(
                grad,
                extent=[x, x + w, y, y + h],
                aspect="auto",
                zorder=0,
                cmap='magma'
            )

        ax_defocus.set_title(f'LLS defocus ($\mu$m)')
        ax_defocus.spines['top'].set_visible(False)
        ax_defocus.spines['left'].set_visible(False)
        ax_defocus.spines['right'].set_visible(False)
        ax_defocus.set_yticks([])
        ax_defocus.grid(True, which="both", axis='x', lw=1, ls='--', zorder=0)
        ax_defocus.axvline(0, ls='-', color='k')
        ax_defocus.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax_defocus.set_xticklabels(ax_defocus.get_xticks(), rotation=45)

    savesvg(fig, f'{save_path}.svg')


@profile
def prediction(
    original_image,
    corrected_image,
    save_path,
    cmap='hot',
    gamma=.5
):
    def slice(xy, zx, vol, label='', maxproj=True):

        if vol.shape[-1] == 3:
            m = xy.imshow(vol[:, :, 0], cmap=cmap, vmin=0, vmax=1)
            zx.imshow(vol[:, :, 1], cmap=cmap, vmin=0, vmax=1)
        else:
            vol = vol ** gamma
            vol = np.nan_to_num(vol)

            if maxproj:
                m = xy.imshow(np.max(vol, axis=0), cmap=cmap, vmin=0, vmax=1)
                zx.imshow(np.max(vol, axis=1), cmap=cmap, vmin=0, vmax=1)
            else:
                mid_plane = vol.shape[0] // 2
                m = xy.imshow(vol[mid_plane, :, :], cmap=cmap, vmin=0, vmax=1)
                zx.imshow(vol[:, mid_plane, :], cmap=cmap, vmin=0, vmax=1)

        cax = inset_axes(zx, width="10%", height="100%", loc='center right', borderpad=-3)
        cb = plt.colorbar(m, cax=cax)
        cax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        cax.yaxis.set_label_position("right")
        cb.ax.set_ylabel(rf"$\gamma$={gamma}")
        xy.set_ylabel(label)
        return m

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.autolimit_mode': 'round_numbers'
    })

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 2)

    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xz = fig.add_subplot(gs[0, 1])

    ax_pxy = fig.add_subplot(gs[1, 0])
    ax_pxz = fig.add_subplot(gs[1, 1])

    ax_pxy.set_title('XY')
    ax_pxz.set_title('XZ')

    slice(ax_xy, ax_xz, original_image, label='Input (MIP)', maxproj=True)
    slice(ax_pxy, ax_pxz, corrected_image, label='Corrected (MIP)', maxproj=True)
    savesvg(fig, f'{save_path}.svg')

@profile
def tiles(
    data: np.ndarray,
    save_path: Path,
    strides: tuple = (64, 64, 64),
    window_size: tuple = (64, 64, 64),
    gamma: float = .5,
):
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.autolimit_mode': 'round_numbers'
    })
    ztiles = np.array_split(range(data.shape[0]), data.shape[0]//window_size[0])

    for z, idx in enumerate(ztiles):
        sample = np.max(data[idx], axis=0)
        tiles = sliding_window_view(sample, window_shape=[window_size[1], window_size[2]])[::strides[1], ::strides[2]]
        nrows, ncols = tiles.shape[0], tiles.shape[1]
        tiles = np.reshape(tiles, (-1, window_size[1], window_size[2]))

        fig = plt.figure(figsize=(3*nrows, 3*ncols))
        grid = ImageGrid(
            fig, 111,
            nrows_ncols=(nrows, ncols),
            share_all=True,
            label_mode="L",
            cbar_location="top",
            cbar_mode="single",
            axes_pad=.2,
            cbar_pad=.2
        )

        i = 0
        for y in range(nrows):
            for x in range(ncols):
                im = grid[i].imshow(tiles[i], cmap='hot', vmin=np.nanmin(sample), vmax=np.nanmax(sample), aspect='equal')
                grid[i].set_title(f"z{z}-y{y}-x{x}", pad=1)
                grid[i].axis('off')
                i += 1

        cbar = grid.cbar_axes[0].colorbar(im)
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.set_yticks([])
        cbar.ax.set_xlabel(rf"$\gamma$={gamma}")
        savesvg(fig, f'{save_path}_z{z}.svg')


@profile
def wavefronts(
    predictions: pd.DataFrame,
    ztiles: int,
    nrows: int,
    ncols: int,
    save_path: Path,
    wavelength: float = .510,
    threshold: float = .01,
    scale: str = 'mean',
):

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.autolimit_mode': 'round_numbers'
    })
    final_pred = Wavefront(predictions[scale].values, lam_detection=wavelength)
    pred_wave = final_pred.wave(size=100)
    fig, ax = plt.subplots()
    mat = plot_wavefront(ax, pred_wave)

    for z in range(ztiles):
        fig = plt.figure(figsize=(3*nrows, 3*ncols))
        grid = ImageGrid(
            fig, 111,
            nrows_ncols=(nrows, ncols),
            share_all=True,
            label_mode="L",
            cbar_location="top",
            cbar_mode="single",
            axes_pad=.2,
            cbar_pad=.2
        )

        i = 0
        for y in range(nrows):
            for x in range(ncols):
                try:
                    roi = f"z{z}-y{y}-x{x}"
                    pred = Wavefront(predictions[roi].values, lam_detection=wavelength)
                    pred_wave = pred.wave(size=100)
                    plot_wavefront(grid[i], pred_wave)
                    grid[i].set_title(roi, pad=1)
                except Exception:
                    grid[i].axis('off')

                i += 1

        cbar = grid.cbar_axes[0].colorbar(mat)
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.set_yticks([])
        cbar.ax.set_title(f'$\lambda = {wavelength}~\mu$m')
        savesvg(fig, f'{save_path}_z{z}.svg')


def sign_correction(
    init_preds_wave,
    init_preds_wave_error,
    followup_preds_wave,
    followup_preds_wave_error,
    preds_wave,
    preds_error,
    percent_changes,
    percent_changes_error,
    savepath,
    bar_width: float = .35
):
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.autolimit_mode': 'round_numbers'
    })

    fig, axes = plt.subplots(3, 1, figsize=(16, 8))

    axes[0].bar(
        np.arange(len(preds_wave)) - bar_width / 2,
        init_preds_wave,
        yerr=init_preds_wave_error,
        capsize=5,
        alpha=.75,
        color='grey',
        align='center',
        ecolor='grey',
        label='Initial',
        width=bar_width
    )
    axes[0].bar(
        np.arange(len(preds_wave)) + bar_width / 2,
        followup_preds_wave,
        yerr=followup_preds_wave_error,
        capsize=5,
        alpha=.75,
        color='red',
        align='center',
        ecolor='grey',
        label='Followup',
        width=bar_width
    )

    axes[0].legend(frameon=False, loc='upper left')
    axes[0].set_xlim((-1, len(preds_wave)))
    axes[0].set_xticks(range(0, len(preds_wave)))
    axes[0].spines.right.set_visible(False)
    axes[0].spines.left.set_visible(False)
    axes[0].spines.top.set_visible(False)
    axes[0].grid(True, which="both", axis='y', lw=1, ls='--', zorder=0)
    axes[0].set_ylabel(r'Zernike coefficients ($\mu$m RMS)')

    axes[1].plot(np.zeros_like(percent_changes), '--', color='lightgrey')
    axes[1].bar(
        range(len(preds_wave)),
        percent_changes,
        yerr=percent_changes_error,
        capsize=10,
        color='C2',
        alpha=.75,
        align='center',
        ecolor='grey',
    )
    axes[1].set_xlim((-1, len(preds_wave)))
    axes[1].set_xticks(range(0, len(preds_wave)))
    axes[1].set_ylim((-100, 100))
    axes[1].set_yticks(range(-100, 125, 25))
    axes[1].set_yticklabels(['-100+', '-75', '-50', '-25', '0', '25', '50', '75', '100+'])
    axes[1].spines.right.set_visible(False)
    axes[1].spines.left.set_visible(False)
    axes[1].spines.top.set_visible(False)
    axes[1].grid(True, which="both", axis='y', lw=1, ls='--', zorder=0)
    axes[1].set_ylabel(f'Percent change')

    axes[2].plot(np.zeros_like(preds_wave), '--', color='lightgrey')
    axes[2].bar(
        range(len(preds_wave)),
        preds_wave,
        yerr=preds_error,
        capsize=10,
        alpha=.75,
        color='C0',
        align='center',
        ecolor='grey',
        label='Predictions',
    )
    axes[2].set_xlim((-1, len(preds_wave)))
    axes[2].set_xticks(range(0, len(preds_wave)))
    axes[2].spines.right.set_visible(False)
    axes[2].spines.left.set_visible(False)
    axes[2].spines.top.set_visible(False)
    axes[2].grid(True, which="both", axis='y', lw=1, ls='--', zorder=0)
    axes[2].set_ylabel(r'Zernike coefficients ($\mu$m RMS)')
    savesvg(fig, f'{savepath}.svg')


def sign_eval(
    inputs,
    followup_inputs,
    savepath,
    gamma=.5,
    cmap='hot'
):
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.autolimit_mode': 'round_numbers'
    })

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    inputs = np.squeeze(inputs) ** gamma
    inputs = np.nan_to_num(inputs)

    followup_inputs = np.squeeze(followup_inputs) ** gamma
    followup_inputs = np.nan_to_num(followup_inputs)

    for i in range(3):
        m = axes[0, i].imshow(np.max(inputs, axis=i), cmap=cmap, vmin=0, vmax=1)
        m = axes[1, i].imshow(np.max(followup_inputs, axis=i), cmap=cmap, vmin=0, vmax=1)

        if i == 2:
            for k in range(2):
                cax = inset_axes(axes[k, i], width="10%", height="100%", loc='center right', borderpad=-3)
                cb = plt.colorbar(m, cax=cax)
                cax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                cax.yaxis.set_label_position("right")
                cb.ax.set_ylabel(rf"$\gamma$={gamma}")

    axes[0, 0].set_ylabel('Input (MIP)')
    axes[1, 0].set_ylabel('Followup (MIP)')
    savesvg(fig, f'{savepath}.svg')


def compare_mips(
    results: dict,
    save_path: Path,
    psf_cmap: str = 'magma',
    gamma: float = .5,
    dxy: float = .108,
    dz: float = .2,
    pltstyle: Any = None,
    transform_to_align_to_DM: bool = False,
):
    if pltstyle is not None: plt.style.use(pltstyle)

    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.autolimit_mode': 'round_numbers'
    })

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 3)

    if transform_to_align_to_DM:
        # 180 rotate, then transpose
        noao_img = np.transpose(np.rot90(results['noao_img'], k=2, axes=(1, 2)), axes=(0, 2, 1))
        ml_img = np.transpose(np.rot90(results['ml_img'], k=2, axes=(1, 2)), axes=(0, 2, 1))
        sh_img = np.transpose(np.rot90(results['gt_img'], k=2, axes=(1, 2)), axes=(0, 2, 1))

    plot_mip(
        xy=fig.add_subplot(gs[0, 0]),
        xz=fig.add_subplot(gs[0, 1]),
        yz=fig.add_subplot(gs[0, 2]),
        label=f'Input [$\gamma$={gamma}]',
        vol=noao_img,
        cmap=psf_cmap,
        dxy=dxy,
        dz=dz,
        aspect='auto'
    )

    plot_mip(
        xy=fig.add_subplot(gs[1, 0]),
        xz=fig.add_subplot(gs[1, 1]),
        yz=fig.add_subplot(gs[1, 2]),
        label=f'SH [$\gamma$={gamma}]',
        vol=sh_img,
        cmap=psf_cmap,
        dxy=dxy,
        dz=dz,
        aspect='auto'
    )

    plot_mip(
        xy=fig.add_subplot(gs[2, 0]),
        xz=fig.add_subplot(gs[2, 1]),
        yz=fig.add_subplot(gs[2, 2]),
        vol=ml_img,
        label=f'Model [$\gamma$={gamma}]',
        cmap=psf_cmap,
        dxy=dxy,
        dz=dz,
        aspect='auto'
    )

    savesvg(fig, f'{save_path}.svg')


def compare_iterations(
    results: dict,
    num_iters: int,
    save_path: Path,
    psf_cmap: str = 'hot',
    gamma: float = .5,
    dxy: float = .108,
    dz: float = .2,
    pltstyle: Any = None,
):
    if pltstyle is not None: plt.style.use(pltstyle)

    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.autolimit_mode': 'round_numbers'
    })

    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, num_iters)

    vmin = -.5
    vmax = .5

    for i in range(num_iters):
        p = results[i]['ml_wavefront']
        if p is not None:
            p_wave = p.wave(size=100)
            ax_ml = fig.add_subplot(gs[1, i])
            plot_wavefront(ax_ml, p_wave, label='P2V', vmin=vmin, vmax=vmax, nas=[.95, .85])

        y = results[i]['gt_wavefront']
        if y is not None:
            y_wave = y.wave(size=100)
            ax_sh = fig.add_subplot(gs[2, i])
            mat = plot_wavefront(ax_sh, y_wave, label='P2V', vmin=vmin, vmax=vmax, nas=[.95, .85])

        if i == 0:
            for ax, label in zip((ax_ml, ax_sh), ('OpticalNet', 'Shack–Hartmann')):
                cax = inset_axes(ax, width="10%", height="100%", loc='center left', borderpad=-5)
                cbar = fig.colorbar(
                    mat,
                    cax=cax,
                    fraction=0.046,
                    pad=0.04,
                    extend='both',
                    format=FormatStrFormatter("%.2g"),
                )
                cbar.ax.set_title(r'$\lambda$', pad=20)
                cbar.ax.yaxis.set_ticks_position('right')
                cbar.ax.yaxis.set_label_position('left')
                cbar.ax.set_ylabel(label)

        ax_img = fig.add_subplot(gs[0, i])

        if i == 0:
            plot_mip(
                xy=ax_img,
                xz=None,
                yz=None,
                gamma=gamma,
                label='OpticalNet',
                vol=results['noao_img'],
                cmap=psf_cmap,
                dxy=dxy,
                dz=dz,
                colorbar=True,
            )
            ax_img.set_title(f'No AO')
        else:
            plot_mip(
                xy=ax_img,
                xz=None,
                yz=None,
                gamma=gamma,
                vol=results[i]['ml_img'],
                cmap=psf_cmap,
                dxy=dxy,
                dz=dz,
                colorbar=False,
            )

            ax_img.set_title(f'Round {i}\nXY ($\mu$m)')
        ax_img.set_xlabel('')

        for ax in [ax_ml, ax_sh]:
            ax.axis('off')

    # plot_mip(
    #     xy=ml_ax,
    #     xz=None,
    #     yz=None,
    #     gamma=gamma,
    #     label='OpticalNet',
    #     vol=results['ml_img'],
    #     cmap=psf_cmap,
    #     dxy=dxy,
    #     dz=dz,
    #     colorbar=True,
    # )
    # ml_ax.set_xlabel('')
    # ml_ax.set_title('XY ($\mu$m)')

    # plot_mip(
    #     xy=gt_ax,
    #     xz=None,
    #     yz=None,
    #     gamma=gamma,
    #     label='Shack–Hartmann',
    #     vol=results['gt_img'],
    #     cmap=psf_cmap,
    #     dxy=dxy,
    #     dz=dz,
    #     colorbar=True,
    # )
    # gt_ax.set_xlabel('')
    # gt_ax.set_title('XY ($\mu$m)')

    plt.subplots_adjust(top=.9, bottom=.1, left=.1, right=.9, hspace=.1, wspace=.2)
    plt.savefig(f'{save_path}.png', bbox_inches='tight', dpi=300, pad_inches=.25)
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight', dpi=300, pad_inches=.25)

    fig = plt.figure(figsize=(8, 12))
    gs = fig.add_gridspec(6, 3)
    zz = 20

    for i in range(3):
        r = i * 2
        noao_ax = fig.add_subplot(gs[r, 0])
        gt_ax = fig.add_subplot(gs[r, 1])
        ml_ax = fig.add_subplot(gs[r, 2])
        ml_axz = fig.add_subplot(gs[r+1, :])

        plot_mip(
            xy=noao_ax,
            xz=None,
            yz=None,
            gamma=gamma,
            vol=results['noao_img'][i*zz:(i+1)*zz],
            cmap=psf_cmap,
            dxy=dxy,
            dz=dz,
            colorbar=True,
            label=f'{int(i*zz*dz):1d}$-${int((i+1)*zz*dz):1d}$~\mu$m'
        )
        noao_ax.set_xlabel('')

        plot_mip(
            xy=gt_ax,
            xz=None,
            yz=None,
            gamma=gamma,
            vol=results['gt_img'][i*zz:(i+1)*zz],
            cmap=psf_cmap,
            dxy=dxy,
            dz=dz,
            colorbar=False,
        )
        gt_ax.set_xlabel('')

        plot_mip(
            xy=ml_ax,
            xz=ml_axz,
            yz=None,
            gamma=gamma,
            vol=results['ml_img'][i*zz:(i+1)*zz],
            cmap=psf_cmap,
            dxy=dxy,
            dz=dz,
            colorbar=False,
        )
        ml_ax.set_xlabel('')
        ml_axz.set_xlabel('')

        if i == 0:
            noao_ax.set_title('No AO\nXY ($\mu$m)')
            gt_ax.set_title('Shack–Hartmann\nXY ($\mu$m)')
            ml_ax.set_title('OpticalNet\nXY ($\mu$m)')
        ml_axz.set_ylabel('OpticalNet\nXZ ($\mu$m)')

    plt.subplots_adjust(top=.9, bottom=.1, left=.1, right=.9, hspace=.01, wspace=.2)
    plt.savefig(f'{save_path}_depth.png', bbox_inches='tight', dpi=300, pad_inches=.25)
    plt.savefig(f'{save_path}_depth.pdf', bbox_inches='tight', dpi=300, pad_inches=.25)


def plot_interference(
        plot,
        plot_interference_pattern,
        pois,
        min_distance,
        beads,
        convolved_psf,
        psf_peaks,
        corrected_psf,
        kernel,
        interference_pattern
):
    fig, axes = plt.subplots(
        nrows=5 if plot_interference_pattern else 4,
        ncols=3,
        figsize=(10, 11),
        sharey=False,
        sharex=False
    )
    transparency = 0.6
    for ax in range(3):
        for p in range(pois.shape[0]):
            if ax == 0:
                axes[0, ax].plot(pois[p, 2], pois[p, 1], marker='x', ls='', color=f'C{p}')
                axes[2, ax].plot(pois[p, 2], pois[p, 1], marker='x', ls='', color=f'C{p}', alpha=transparency)
                axes[2, ax].add_patch(patches.Rectangle(
                    xy=(pois[p, 2] - min_distance, pois[p, 1] - min_distance),
                    width=min_distance * 2,
                    height=min_distance * 2,
                    fill=None,
                    color=f'C{p}',
                    alpha=transparency
                ))
            elif ax == 1:
                axes[0, ax].plot(pois[p, 2], pois[p, 0], marker='x', ls='', color=f'C{p}')
                axes[2, ax].plot(pois[p, 2], pois[p, 0], marker='x', ls='', color=f'C{p}', alpha=transparency)
                axes[2, ax].add_patch(patches.Rectangle(
                    xy=(pois[p, 2] - min_distance, pois[p, 0] - min_distance),
                    width=min_distance * 2,
                    height=min_distance * 2,
                    fill=None,
                    color=f'C{p}',
                    alpha=transparency
                ))

            elif ax == 2:
                axes[0, ax].plot(pois[p, 1], pois[p, 0], marker='x', ls='', color=f'C{p}')
                axes[2, ax].plot(pois[p, 1], pois[p, 0], marker='x', ls='', color=f'C{p}', alpha=transparency)
                axes[2, ax].add_patch(patches.Rectangle(
                    xy=(pois[p, 1] - min_distance, pois[p, 0] - min_distance),
                    width=min_distance * 2,
                    height=min_distance * 2,
                    fill=None,
                    color=f'C{p}',
                    alpha=transparency
                ))
        m1 = axes[0, ax].imshow(np.nanmax(psf_peaks, axis=ax), cmap='hot')
        m2 = axes[1, ax].imshow(np.nanmax(kernel, axis=ax), cmap='hot')
        m3 = axes[2, ax].imshow(np.nanmax(convolved_psf, axis=ax), cmap='Greys_r', alpha=.66)

        if plot_interference_pattern:
            interference = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=axes[3, ax], wspace=0.05, hspace=0)
            ax1 = fig.add_subplot(interference[0])
            ax1.imshow(np.nanmax(beads, axis=ax), cmap='hot')
            ax1.axis('off')
            ax1.set_title(r'$\mathcal{S}$')

            ax2 = fig.add_subplot(interference[1])
            m4 = ax2.imshow(np.nanmax(abs(interference_pattern), axis=ax), cmap='magma')
            ax2.axis('off')
            ax2.set_title(r'$|\mathscr{F}(\mathcal{S})|$')

        m5 = axes[-1, ax].imshow(np.nanmax(corrected_psf, axis=ax), cmap='hot')

    for ax, m, label in zip(
        range(5) if plot_interference_pattern else range(4),
        [m1, m2, m3, m4, m5] if plot_interference_pattern else [m1, m2, m3, m5],
        [
            f'Inputs ({pois.shape[0]} peaks)',
            'Kernel',
            'Peak detection',
            'Interference',
            'Reconstructed'
        ]
        if plot_interference_pattern else [
            f'Inputs ({pois.shape[0]} peaks)',
            'kernel',
            'Peak detection',
            'Reconstructed'
        ]
    ):
        cax = inset_axes(axes[ax, -1], width="10%", height="90%", loc='center right', borderpad=-3)
        cb = plt.colorbar(m, cax=cax)
        cax.yaxis.set_label_position("right")
        cax.set_ylabel(label)

    for ax in axes.flatten():
        ax.axis('off')

    axes[0, 0].set_title('XY')
    axes[0, 1].set_title('XZ')
    axes[0, 2].set_title('YZ')

    savesvg(fig, f'{plot}_interference_pattern.svg')


@profile
def plot_embeddings(
        inputs: np.array,
        emb: np.array,
        save_path: Any,
        gamma: float = .5,
        nrows: Optional[int] = None,
        ncols: Optional[int] = None,
        ztiles: Optional[int] = None,
):
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.autolimit_mode': 'round_numbers'
    })

    step = .1
    vmin = int(np.floor(np.nanpercentile(emb[0], 1))) if np.any(emb[0] < 0) else 0
    vmax = int(np.ceil(np.nanpercentile(emb[0], 99))) if vmin < 0 else 3
    vcenter = 1 if vmin == 0 else 0

    cmap = np.vstack((
        plt.get_cmap('GnBu_r' if vmin == 0 else 'GnBu_r', 256)(
            np.linspace(0, 1 - step, int(abs(vcenter - vmin) / step))
        ),
        [1, 1, 1, 1],
        plt.get_cmap('YlOrRd' if vmax == 3 else 'OrRd', 256)(
            np.linspace(0, 1 + step, int(abs(vcenter - vmax) / step))
        )
    ))
    cmap = mcolors.ListedColormap(cmap)

    if emb.shape[0] == 3:
        fig, axes = plt.subplots(2, 3, figsize=(8, 6))
    else:
        fig, axes = plt.subplots(3, 3, figsize=(8, 8))

    if inputs.ndim == 4:
        if ncols is None or nrows is None:
            inputs = np.max(inputs, axis=0)  # show max projections of all z-tiles
            for c in range(10, 0, -1):
                if inputs.shape[0] > c and not inputs.shape[0] % c:
                    ncols = c
                    break

            nrows = inputs.shape[0] // ncols

        for proj in range(3):
            grid = gridspec.GridSpecFromSubplotSpec(
                nrows, ncols, subplot_spec=axes[0, proj], wspace=.01, hspace=.01
            )

            for idx, (i, j) in enumerate(itertools.product(range(nrows), range(ncols))):
                ax = fig.add_subplot(grid[i, j])
                m = ax.imshow(np.max(inputs[idx], axis=proj) ** gamma, cmap='hot', vmin=0, vmax=1)
                ax.axis('off')

        cax = inset_axes(axes[0, 2], width="10%", height="100%", loc='center right', borderpad=-3)
        cb = plt.colorbar(m, cax=cax)
        cax.yaxis.set_label_position("right")
        cax.set_ylabel(rf'Input (MIP) [$\gamma$={gamma}]')
    else:
        m = axes[0, 0].imshow(np.max(inputs, axis=0) ** gamma, cmap='hot', vmin=0, vmax=1)
        axes[0, 1].imshow(np.max(inputs, axis=1) ** gamma, cmap='hot', vmin=0, vmax=1)
        axes[0, 2].imshow(np.max(inputs, axis=2) ** gamma, cmap='hot', vmin=0, vmax=1)
        cax = inset_axes(axes[0, 2], width="10%", height="100%", loc='center right', borderpad=-3)
        cb = plt.colorbar(m, cax=cax)
        cax.yaxis.set_label_position("right")
        cax.set_ylabel(rf'Input (MIP) [$\gamma$={gamma}]')

    m = axes[1, 0].imshow(emb[0], cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 1].imshow(emb[1], cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 2].imshow(emb[2], cmap=cmap, vmin=vmin, vmax=vmax)
    cax = inset_axes(axes[1, 2], width="10%", height="100%", loc='center right', borderpad=-3)
    cb = plt.colorbar(m, cax=cax)
    cax.yaxis.set_label_position("right")
    cax.set_ylabel(r'Embedding ($\alpha$)')

    if emb.shape[0] > 3:
        p_vmin = -1
        p_vmax = 1
        p_vcenter = 0

        p_cmap = np.vstack((
            plt.get_cmap('GnBu_r' if p_vmin == 0 else 'GnBu_r', 256)(
                np.linspace(0, 1 - step, int(abs(p_vcenter - p_vmin) / step))
            ),
            [1, 1, 1, 1],
            plt.get_cmap('YlOrRd' if p_vmax == 3 else 'OrRd', 256)(
                np.linspace(0, 1 + step, int(abs(p_vcenter - p_vmax) / step))
            )
        ))
        p_cmap = mcolors.ListedColormap(p_cmap)

        m = axes[-1, 0].imshow(emb[3], cmap=p_cmap, vmin=p_vmin, vmax=p_vmax)
        axes[-1, 1].imshow(emb[4], cmap=p_cmap, vmin=p_vmin, vmax=p_vmax)
        axes[-1, 2].imshow(emb[5], cmap=p_cmap, vmin=p_vmin, vmax=p_vmax)
        cax = inset_axes(axes[-1, 2], width="10%", height="100%", loc='center right', borderpad=-3)
        cb = plt.colorbar(m, cax=cax)
        cax.yaxis.set_label_position("right")
        cax.set_ylabel(r'Embedding ($\varphi$)')

    for ax in axes.flatten():
        ax.axis('off')

    if save_path == True:
        plt.show()
    else:
        savesvg(fig, f'{save_path}_embeddings.svg')


@profile
def plot_rotations(results: Path):
    plt.style.use("default")
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.autolimit_mode': 'round_numbers'
    })

    dataframe = pd.read_csv(results, header=0, index_col=0)
    n_modes = int(np.nanmax(dataframe['twin']) + 1)
    wavefront = Wavefront(np.zeros(n_modes))
    rotations = dataframe['angle'].unique()

    fig = plt.figure(figsize=(15, 20 * round(n_modes / 15)))
    gs = fig.add_gridspec(len(wavefront.twins.keys()), 2)

    for row, (mode, twin) in enumerate(wavefront.twins.items()):
        df = dataframe[dataframe['mode'] == mode.index_ansi]

        xdata = df.twin_angle.values
        ydata = df.pred_twin_angle.values
        rhos = df.rhos.values

        data_mask = df.valid_points.values.astype(bool)
        fraction_of_kept_points = data_mask.sum() / len(data_mask)
        fitted_twin_angle = df.fitted_twin_angle.values

        rho = df.aggr_rho.values[0]
        stdev = df.aggr_std_dev.values[0]
        mse = df.mse.values[0]
        confident = df.confident.values[0].astype(bool)

        if rho > 0 and confident:
            title_color = 'g'
        else:
            title_color = 'C0' if confident else 'r'

        if twin is not None:
            ax = fig.add_subplot(gs[row, 0])
            fit_ax = fig.add_subplot(gs[row, 1])

            ax.plot(rotations, df.init_pred_mode, label=f"m{mode.index_ansi}")
            ax.plot(rotations, df.init_pred_twin, label=f"m{twin.index_ansi}")

            ax.set_xlim(0, 360)
            ax.set_xticks(range(0, 405, 45))
            ax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)
            ax.set_ylim(-np.max(rhos), np.max(rhos))
            ax.legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(.5, 1.15))
            ax.set_ylabel('Amplitude ($\mu$m RMS)')
            ax.set_xlabel('Digital rotation (deg)')

            # plot fit line from zero to end of data
            fit_ax.plot(xdata, fitted_twin_angle, color=title_color, lw='.75')
            fit_ax.scatter(xdata[data_mask], ydata[data_mask], s=2, color='grey')

            fit_ax.set_title(
                # f'm{mode.index_ansi}={preds[mode.index_ansi]:.3f}, '
                # f'm{twin.index_ansi}={preds[twin.index_ansi]:.3f} '
                f'$\\rho$={rho:.3f} $\mu$RMS, '
                f'$\\rho/\\sigma={rho / stdev:.3f}, \\sigma$={stdev:.3f}, '
                f'MSE={mse:.0f}',
                color=title_color
            )

            fit_ax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)
            fit_ax.set_ylabel('Predicted Twin angle (deg)', rotation=-90, labelpad=15)
            fit_ax.yaxis.set_label_position("right")
            fit_ax.set_xlabel('Digitially rotated Twin angle (deg)')
            fit_ax.set_xticks(range(0, int(np.max(xdata)), 90))
            fit_ax.set_yticks(np.insert(np.arange(-90, np.nanmax(ydata[data_mask]), 180), 0, 0))
            fit_ax.set_xlim(0, 360 * np.abs(mode.m))

            ax.scatter(rotations[data_mask == 0], rhos[data_mask == 0], s=1.5, color='pink', zorder=3)
            ax.scatter(rotations[data_mask == 1], rhos[data_mask == 1], s=1.5, color='black', zorder=3)
        else:
            ax = fig.add_subplot(gs[row, 0])
            ax.plot(
                rotations,
                df.init_pred_mode,
                label=f'm{mode.index_ansi}: '
                      f'$\\rho$={rho:.3f} $\mu$RMS, '
                      f'$\\rho/\\sigma={rho / stdev:.3f}, \\sigma$={stdev:.3f}',
                color=title_color
            )

            ax.set_xlim(0, 360)
            ax.set_xticks(range(0, 405, 45))
            ax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)
            ax.set_ylim(min(np.min(-np.abs(df.init_pred_mode)), -0.01), max(np.max(np.abs(df.init_pred_mode)), 0.01))
            ax.legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(.5, 1.15))
            ax.set_ylabel('Amplitude ($\mu$ RMS)')
            ax.set_xlabel('Digital rotation (deg)')

    savesvg(fig, results.with_suffix('.svg'))


@profile
def plot_volume(
        vol: np.ndarray,
        results: pd.DataFrame,
        save_path: Union[Path, str],
        window_size: tuple,
        wavelength: float = .510,
        dxy: float = .108,
        dz: float = .2,
        gamma: float = .5,
        proj_ax: int = 2
):
    def formatter(x, pos, dd):
        return f'{np.ceil(x * dd).astype(int):1d}'

    plt.style.use("default")
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.autolimit_mode': 'round_numbers'
    })

    vol = np.max(vol, axis=proj_ax)
    ztiles = sliding_window_view(
        vol, window_shape=[window_size[0], vol.shape[1], 3]
    )[::window_size[0], ::vol.shape[1]]

    nrows, ncols = ztiles.shape[0], ztiles.shape[1]
    ztiles = np.reshape(ztiles, (-1, window_size[0], vol.shape[1], 3))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, nrows*3))

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for i, (k, j) in enumerate(itertools.product(range(nrows), range(ncols))):
        proj = ztiles[i]

        im = axes[i].contourf(
            np.max(proj, axis=-1),
            cmap='tab20',
            vmin=np.nanmin(proj),
            vmax=np.nanmax(proj),
            aspect='auto'
        )

        depth = np.arange(proj.shape[0]*i, proj.shape[0]*(i+1))
        labels = [int(round(x * dz, 0)) for x in depth]
        axes[i].set_yticks(np.arange(len(depth)))
        axes[i].set_yticklabels(labels)
        axes[i].yaxis.set_major_locator(plt.MaxNLocator(8))
        axes[i].grid(True, which="both", axis='both', lw=.5, ls='--', zorder=0, alpha=.66)

        if i == 0:
            axes[i].xaxis.set_major_formatter(partial(formatter, dd=dxy))
            axes[i].xaxis.set_major_locator(plt.MaxNLocator(vol.shape[1]*dxy//5))
        else:
            axes[i].set_xticks([])

        wavefront = Wavefront(
            results[f'z{i}'].values,
            order='ansi',
            lam_detection=wavelength
        )

        wax = inset_axes(axes[i], width="25%", height="100%", loc='center right', borderpad=-15)
        w = plot_wavefront(wax, wavefront.wave(size=100), vcolorbar=True, label='Average', nas=[.95, .85])

    axes[0].set_xlabel('X ($\mu$m)' if proj_ax == 1 else 'Y ($\mu$m)')
    axes[0].xaxis.set_label_position("top")
    axes[0].xaxis.set_ticks_position("top")
    axes[0].set_ylabel('Z ($\mu$m)')

    # cax = inset_axes(axes[-1], width="100%", height="10%", loc='lower center', borderpad=-2)
    # cb = plt.colorbar(im, cax=cax, orientation='horizontal')
    # cax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    # cax.set_xlabel(rf'Input (MIP)')
    # cax.xaxis.set_label_position("bottom")
    # cax.xaxis.set_ticks_position("bottom")

    savesvg(fig, save_path, hspace=.01, wspace=.01)


@profile
def plot_isoplanatic_patchs(
    results: pd.DataFrame,
    clusters: pd.DataFrame,
    save_path: Union[Path, str],
):

    plt.style.use("default")
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.autolimit_mode': 'round_numbers'
    })

    xtiles = len(results.index.get_level_values('x').unique())
    ytiles = len(results.index.get_level_values('y').unique())
    ztiles = len(results.index.get_level_values('z').unique())
    nrows = ztiles * ytiles

    mode_hashtable = {}
    w = Wavefront(np.zeros(results.index.get_level_values('mode').unique().shape[0]))
    for i, (mode, twin) in enumerate(w.twins.items()):
        if twin is not None:
            mode_hashtable[mode.index_ansi] = f'$Z^{mode.n}_{{{mode.index_ansi},{twin.index_ansi}}}$'
            mode_hashtable[twin.index_ansi] = f'$Z^{mode.n}_{{{mode.index_ansi},{twin.index_ansi}}}$'
        else:
            mode_hashtable[mode.index_ansi] = f'$Z^{mode.n}_{{{mode.index_ansi}}}$'

    results.reset_index(inplace=True)
    results['cat'] = results['mode'].map(mode_hashtable)
    results.set_index(['z', 'y', 'x'], inplace=True)

    fig, axes = plt.subplots(nrows=nrows, ncols=xtiles, figsize=(15, 50), subplot_kw={'projection': 'polar'})

    for zi, yi, xi in itertools.product(range(ztiles), range(ytiles), range(xtiles)):
        row = yi + (zi * ytiles)
        roi = f"z{zi}-y{yi}-x{xi}"
        m = results.loc[(zi, yi, xi)]
        m = m.groupby('cat', as_index=False).mean()
        cc = clusters.loc[(zi, yi, xi), 'cluster']

        # pred = Wavefront(results.loc[(zi, yi, xi), 'prediction'].values, lam_detection=.510)
        # pred_wave = pred.wave(size=100)
        # plot_wavefront(axes[row, xi], pred_wave)

        theta = np.arange(m.shape[0] + 1) / float(m.shape[0]) * 2 * np.pi
        values = m['weight'].values
        values = np.append(values, values[0])

        l1, = axes[row, xi].plot(theta, values, color='k', marker="o")
        axes[row, xi].set_xticks(theta[:-1], m['cat'], color='dimgrey')
        axes[row, xi].tick_params(axis='both', which='major', pad=10)
        axes[row, xi].set_yticklabels([])
        axes[row, xi].set_title(roi, pad=1)
        # axes[row, xi].fill(theta, values, 'grey', alpha=0.25)
        axes[row, xi].patch.set_facecolor(colors.to_rgba(f'C{cc}'))
        axes[row, xi].patch.set_alpha(0.25)

    savesvg(fig, save_path, hspace=.4, wspace=0)
