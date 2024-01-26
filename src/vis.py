import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.set_loglevel('error')

import re
import warnings
import pandas as pd
from pathlib import Path
from functools import partial
import logging
import sys
import itertools
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import ImageGrid
from numpy.lib.stride_tricks import sliding_window_view
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FormatStrFormatter, LogFormatterMathtext
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from itertools import cycle

from typing import Any, Union, Optional
import numpy as np
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from line_profiler_pycharm import profile
from matplotlib import colors

from wavefront import Wavefront
from zernike import Zernike


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
        with open(savepath, 'r', encoding="utf-8") as f:
            filedata = f.read()

        # Replace the target string
        filedata = re.sub('height="[0-9]+(\.[0-9]+)pt"', '', filedata)
        filedata = re.sub('width="[0-9]+(\.[0-9]+)pt"', '', filedata)

        # Write the file out again
        with open(savepath, 'w', encoding="utf-8") as f:
            f.write(filedata)


def plot_mip(
    xy,
    xz,
    yz,
    vol,
    label='',
    gamma=.5,
    cmap='hot',
    dxy=.097,
    dz=.2,
    colorbar=True,
    aspect='auto',
    log=False,
    mip=True,
    ticks=True,
    normalize=False
):
    def formatter(x, pos, dd):
        return f'{np.ceil(x * dd).astype(int):1d}'

    if log:
        vmin, vmax, step = 1e-4, 1, .025
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        vol[vol < vmin] = vmin
    else:
        vol = vol ** gamma
        vol /= vol.max()
        vol = np.nan_to_num(vol)
        vmin, vmax, step = 0, 1, .025
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    if xy is not None:
        if mip:
            v = np.max(vol, axis=0)
        else:
            v = vol[vol.shape[0]//2, :, :]

        mat = xy.imshow(v, cmap=cmap, aspect=aspect, norm=norm if normalize else None)

        xy.set_xlabel(r'XY ($\mu$m)')
        if ticks:
            xy.yaxis.set_ticks_position('right')
            xy.xaxis.set_major_formatter(partial(formatter, dd=dxy))
            xy.yaxis.set_major_formatter(partial(formatter, dd=dxy))
            xy.xaxis.set_major_locator(plt.MaxNLocator(6))
            xy.yaxis.set_major_locator(plt.MaxNLocator(6))
        else:
            xy.axis('off')

    if xz is not None:
        if mip:
            v = np.max(vol, axis=1)
        else:
            v = vol[:, vol.shape[0] // 2, :]

        mat = xz.imshow(v, cmap=cmap, aspect=aspect, norm=norm if normalize else None)

        xz.set_xlabel(r'XZ ($\mu$m)')
        if ticks:
            xz.yaxis.set_ticks_position('right')
            xz.xaxis.set_major_formatter(partial(formatter, dd=dxy))
            xz.yaxis.set_major_formatter(partial(formatter, dd=dz))
            xz.xaxis.set_major_locator(plt.MaxNLocator(6))
            xz.yaxis.set_major_locator(plt.MaxNLocator(6))
        else:
            xz.axis('off')

    if yz is not None:
        if mip:
            v = np.max(vol, axis=2)
        else:
            v = vol[:, :, vol.shape[0] // 2]

        mat = yz.imshow(v, cmap=cmap, aspect=aspect, norm=norm if normalize else None)

        yz.set_xlabel(r'YZ ($\mu$m)')
        if ticks:
            yz.yaxis.set_ticks_position('right')
            yz.xaxis.set_major_formatter(partial(formatter, dd=dxy))
            yz.yaxis.set_major_formatter(partial(formatter, dd=dz))
            yz.xaxis.set_major_locator(plt.MaxNLocator(6))
            yz.yaxis.set_major_locator(plt.MaxNLocator(6))
        else:
            yz.axis('off')

    if colorbar:
        divider = make_axes_locatable(xy)
        cax = divider.append_axes("left", size="5%", pad=0.1)
        cb = plt.colorbar(
            mat,
            cax=cax,
            format=LogFormatterMathtext() if log else FormatStrFormatter("%.1f"),
        )

        cb.ax.set_ylabel(f"{label}")
        cb.ax.yaxis.set_label_position("left")
        cb.ax.yaxis.set_ticks_position('left')
        cb.ax.get_xaxis().get_major_formatter().labelOnlyBase = False

    return


def plot_wavefront(
    iax,
    phi,
    rms=None,
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

    if vmin is None:
        vmin = np.floor(np.nanmin(phi)*2)/4     # round down to nearest 0.25 wave
        vmin = -1*dlimit if vmin > -0.01 else vmin

    if vmax is None:
        vmax = np.ceil(np.nanmax(phi)*2)/4  # round up to nearest 0.25 wave
        vmax = dlimit if vmax < 0.01 else vmax

    cmap = 'Spectral_r'
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mat = iax.imshow(phi, cmap=cmap, norm=norm)

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
            f'$NA_{{{na:.2f}}}$={p2v if na == 1 else abs(p[1]-p[0]):.2f}$\lambda$ (P2V)'
            for na, p in zip(nas, pcts)
        ])
        if label == '':
            iax.set_title(err)
        else:
            if rms is not None:
                iax.set_title(f'{label} RMS[{rms:.2f}$\lambda$]\n{err}\n$NA_{{1.0}}=${p2v:.2f}$\lambda$ (P2V)')
            else:
                iax.set_title(f'{label} [{p2v:.2f}$\lambda$] (P2V)\n{err}')

    iax.axis('off')
    iax.set_aspect("equal")

    if vcolorbar:
        divider = make_axes_locatable(iax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(
            mat,
            cax=cax,
            extend='both',
            format=formatter,
        )
        cbar.ax.set_title(r'$\lambda$', pad=10)
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.yaxis.set_label_position('left')

    if hcolorbar:
        divider = make_axes_locatable(iax)
        cax = divider.append_axes("bottom", size="5%", pad=0.1)
        cbar = plt.colorbar(
            mat,
            cax=cax,
            extend='both',
            orientation='horizontal',
            format=formatter,
        )
        cbar.ax.xaxis.set_ticks_position('bottom')
        cbar.ax.xaxis.set_label_position('top')

    return mat


def diagnostic_assessment(
        psf: np.array,
        gt_psf: np.array,
        predicted_psf: np.array,
        corrected_psf: np.array,
        photons: Any,
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
        dxy: float = .097,
        dz: float = .2,
        pltstyle: Any = None,
        transform_to_align_to_DM: bool = False,
        display_otf: bool = False,
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

    if not np.isscalar(photons):
        photons = photons[0]

    if not np.isscalar(maxcounts):
        maxcounts = maxcounts[0]

    y_wave = y.wave(size=100)
    pred_wave = pred.wave(size=100)
    diff = y_wave - pred_wave

    fig = plt.figure(figsize=(13, 15))
    gs = fig.add_gridspec(4 if gt_psf is None else 5, 4)

    ax_gt = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[0, 1])
    ax_diff = fig.add_subplot(gs[0, 2])
    cax = fig.add_axes([0.05, 0.75, 0.02, .15])

    # input
    ax_xy = fig.add_subplot(gs[1, 0])
    ax_xz = fig.add_subplot(gs[1, 1])
    ax_yz = fig.add_subplot(gs[1, 2])

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
    cbar.ax.set_ylabel(rf'$\lambda$ = {y.lam_detection*1000:.0f}nm')

    ax_cxy.set_xlabel(r'XY ($\mu$m)')
    ax_cxz.set_xlabel(r'XZ ($\mu$m)')
    ax_cyz.set_xlabel(r'YZ ($\mu$m)')
    ax_xz.set_title(f"Total photons: {photons:.1G}")
    ax_yz.set_title(f"Max counts: {maxcounts:.1G}")

    if transform_to_align_to_DM:
        psf = np.transpose(np.rot90(psf, k=2, axes=(1, 2)), axes=(0, 2, 1))    # 180 rotate, then transpose

    if display_otf:
        from embeddings import fft
        psf = np.abs(fft(psf))
        psf /= np.max(psf)

        gt_psf = np.abs(fft(gt_psf))
        gt_psf /= np.max(gt_psf)

        predicted_psf = np.abs(fft(predicted_psf))
        predicted_psf /= np.max(predicted_psf)

        corrected_psf = np.abs(fft(corrected_psf))
        corrected_psf /= np.max(corrected_psf)

    plot_mip(
        xy=ax_xy,
        xz=ax_xz,
        yz=ax_yz,
        vol=psf,
        label=rf'Input (MIP) [$\gamma$={gamma}]',
        cmap=psf_cmap,
        dxy=dxy,
        dz=dz,
        ticks=False
    )
    plot_mip(
        xy=ax_pxy,
        xz=ax_pxz,
        yz=ax_pyz,
        vol=predicted_psf,
        label=rf'Predicted [$\gamma$={gamma}]',
        cmap=psf_cmap,
        dxy=dxy,
        dz=dz,
        ticks=False
    )
    plot_mip(
        xy=ax_cxy,
        xz=ax_cxz,
        yz=ax_cyz,
        vol=corrected_psf,
        label=rf'Corrected [$\gamma$={gamma}]',
        cmap=psf_cmap,
        dxy=dxy,
        dz=dz,
    )

    if gt_psf is not None:
        ax_xygt = fig.add_subplot(gs[2, 0])
        ax_xzgt = fig.add_subplot(gs[2, 1])
        ax_yzgt = fig.add_subplot(gs[2, 2])
        plot_mip(
            xy=ax_xygt,
            xz=ax_xzgt,
            yz=ax_yzgt,
            vol=gt_psf,
            label=rf'Simulated [$\gamma$={gamma}]',
            ticks=False
        )

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

    savesvg(fig, f'{save_path}.svg', wspace=.15, hspace=.15)

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
    pred_rms = np.linalg.norm(pred.amplitudes_noll_waves)

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
        rms=pred_rms,
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

        divider = make_axes_locatable(zx)
        cax = divider.append_axes("right", size="5%", pad=0.1)
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
                divider = make_axes_locatable(axes[k, i])
                cax = divider.append_axes("right", size="5%", pad=0.1)
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
    dxy: float = .097,
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
        label=rf'Input [$\gamma$={gamma}]',
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
        label=rf'SH [$\gamma$={gamma}]',
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
        label=rf'Model [$\gamma$={gamma}]',
        cmap=psf_cmap,
        dxy=dxy,
        dz=dz,
        aspect='auto'
    )

    savesvg(fig, f'{save_path}.svg')


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
        m1 = axes[0, ax].imshow(np.nanmax(psf_peaks, axis=ax), cmap='hot', aspect='auto')
        m2 = axes[1, ax].imshow(np.nanmax(kernel, axis=ax), cmap='hot', aspect='auto')
        m3 = axes[2, ax].imshow(np.nanmax(convolved_psf, axis=ax), cmap='Greys_r', alpha=.66, aspect='auto')

        if plot_interference_pattern:
            interference = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=axes[3, ax], wspace=0.05, hspace=0)
            ax1 = fig.add_subplot(interference[0])
            ax1.imshow(np.nanmax(beads, axis=ax), cmap='hot', aspect='auto')
            ax1.axis('off')
            ax1.set_title(r'$\mathcal{S}$')

            ax2 = fig.add_subplot(interference[1])
            m4 = ax2.imshow(np.nanmax(abs(interference_pattern), axis=ax), cmap='magma', aspect='auto')
            ax2.axis('off')
            ax2.set_title(r'$|\mathscr{F}(\mathcal{S})|$')

        m5 = axes[-1, ax].imshow(np.nanmax(corrected_psf, axis=ax), cmap='hot', aspect='auto')

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
        divider = make_axes_locatable(axes[ax, -1])
        cax = divider.append_axes("right", size="5%", pad=0.1)
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
        icmap: str = 'hot',
        aspect: str = 'auto'
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
        plt.get_cmap('YlOrRd' if vmax != 1 else 'OrRd', 256)(
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

                try:
                    if np.max(inputs[idx], axis=None) > 0 :
                        m = ax.imshow(np.max(inputs[idx], axis=proj) ** gamma, cmap=icmap, aspect=aspect)

                except IndexError: # if we dropped a tile due to poor SNR
                    m = ax.imshow(np.zeros_like(np.max(inputs[0], axis=proj)), cmap=icmap, aspect=aspect)

                ax.axis('off')
            axes[0, proj].axis('off')

        divider = make_axes_locatable(axes[0, 0])
        cax = divider.append_axes("left", size="5%", pad=0.1)
        cb = plt.colorbar(m, cax=cax)
        cax.yaxis.set_label_position("left")
        cax.yaxis.set_ticks_position('left')
        cax.set_ylabel(rf'Input (MIP) [$\gamma$={gamma}]')
    else:
        m = axes[0, 0].imshow(np.max(inputs**gamma, axis=0), cmap=icmap, aspect=aspect)
        axes[0, 1].imshow(np.max(inputs**gamma, axis=1), cmap=icmap, aspect=aspect)
        axes[0, 2].imshow(np.max(inputs**gamma, axis=2), cmap=icmap, aspect=aspect)

        divider = make_axes_locatable(axes[0, 0])
        cax = divider.append_axes("left", size="5%", pad=0.1)
        cb = plt.colorbar(m, cax=cax)
        cax.yaxis.set_label_position("left")
        cax.yaxis.set_ticks_position('left')
        cax.set_ylabel(rf'Input (MIP) [$\gamma$={gamma}]')

    m = axes[1, 0].imshow(emb[0], cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 1].imshow(emb[1], cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 2].imshow(emb[2], cmap=cmap, vmin=vmin, vmax=vmax)

    divider = make_axes_locatable(axes[1, 0])
    cax = divider.append_axes("left", size="5%", pad=0.1)
    cb = plt.colorbar(m, cax=cax)
    cax.yaxis.set_label_position("left")
    cax.yaxis.set_ticks_position('left')
    cax.set_ylabel(r'Embedding ($\alpha$)')

    if emb.shape[0] > 3:
        # phase embedding limit = 95th percentile or 0.25, round to nearest 1/2 rad
        p_vmax = max(np.ceil(np.nanpercentile(np.abs(emb[3:]), 95)*2)/2, .25)
        p_vmin = -p_vmax
        p_vcenter = 0
        step = p_vmax/10

        p_cmap = np.vstack((
            plt.get_cmap('GnBu_r' if p_vmin == 0 else 'GnBu_r', 256)(
                np.linspace(0, 1, int(abs(p_vcenter - p_vmin) / step))
            ),
            [1, 1, 1, 1],
            plt.get_cmap('YlOrRd' if p_vmax == 3 else 'OrRd', 256)(
                np.linspace(0, 1, int(abs(p_vcenter - p_vmax) / step))
            )
        ))
        p_cmap = mcolors.ListedColormap(p_cmap)

        m = axes[-1, 0].imshow(emb[3], cmap=p_cmap, vmin=p_vmin, vmax=p_vmax)
        axes[-1, 1].imshow(emb[4], cmap=p_cmap, vmin=p_vmin, vmax=p_vmax)
        axes[-1, 2].imshow(emb[5], cmap=p_cmap, vmin=p_vmin, vmax=p_vmax)

        divider = make_axes_locatable(axes[-1, 0])
        cax = divider.append_axes("left", size="5%", pad=0.1)
        cb = plt.colorbar(m, cax=cax, format=lambda x, _: f"{x:.1f}")
        cax.yaxis.set_label_position("left")
        cax.yaxis.set_ticks_position('left')
        cax.set_ylabel(r'Embedding ($\varphi$, radians)')

    for ax in axes.flatten():
        ax.axis('off')

    if save_path == True:
        plt.show()
    else:
        savesvg(fig, f'{save_path}_embeddings.svg')
        plt.savefig(f'{save_path}_embeddings.png')


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
        aggr_mode_amp = df.aggr_mode_amp.values[0]
        aggr_twin_amp = df.aggr_twin_amp.values[0]
        fitted_twin_angle_b = df.fitted_twin_angle_b.values[0]
        mse = df.mse.values[0]
        confident = df.confident.values[0].astype(bool)

        if np.abs(rho) > 0 and confident:
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
                f'[{aggr_mode_amp:.3f}, {aggr_twin_amp:.3f}] '
                #f'$\\rho$={rho:.3f} $\mu$RMS, '
                f'$\\rho/\\sigma={rho / stdev:.3f}, \\sigma$={stdev:.3f}, '
                f'MSE={mse:.0f}, '
                f'$\\angle({fitted_twin_angle_b:.0f}^{{\\degree}}_{{twin}},{fitted_twin_angle_b/np.abs(mode.m):.0f}\\degree)$',
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
        dxy: float = .097,
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

        divider = make_axes_locatable(axes[i])
        wax = divider.append_axes("right", size="25%", pad=0.3)
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


def plot_beads_dataset(
    results: dict,
    residuals: pd.DataFrame,
    savepath: Path,
    psf_cmap: str = 'hot',
    gamma: float = .5,
    dxy: float = .097,
    dz: float = .2,
    wavelength: float = .510,
    pltstyle: Any = None,
    custom_colormap: bool = True,
    transform_to_align_to_DM: bool = True
):
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.autolimit_mode': 'round_numbers'
    })

    heatmaps = residuals.drop_duplicates(subset=['eval_file', 'na'])
    heatmaps = heatmaps[["na", "iteration_index", "p2v_gt", "p2v_residual", "p2v_pred", "modes", "mode_1", "mode_2"]]

    for val, label in zip(
            ["p2v_gt", "p2v_residual", "p2v_pred"],
            [
                rf"Aberration (peak-to-valley, $\lambda = {int(wavelength * 1000)}~nm$)",
                rf"Disagreement (peak-to-valley, $\lambda = {int(wavelength * 1000)}~nm$)",
                rf"Prediction (peak-to-valley, $\lambda = {int(wavelength * 1000)}~nm$)",
            ]
    ):

        fig = plt.figure(figsize=(16, 11))
        gs = fig.add_gridspec(4, 6)
        zernikes_dict = list(set(Zernike(int(j)) for j in range(15)))
        heatmap1 = fig.add_subplot(gs[:, -2])
        heatmap85 = fig.add_subplot(gs[:, -1])

        for i, modes in enumerate(['05-05', '03-05', '05-08', '10-12']):
            zernikes = list(set(Zernike(int(j)) for j in modes.split('-')))

            if len(zernikes) == 1:
                zlabel = f"$Z_{{n={zernikes[0].n}}}^{{m={zernikes[0].m}}}$"
            else:
                zlabel = f"$Z_{{n={zernikes[0].n}}}^{{m={zernikes[0].m}}}$" \
                        f" + $Z_{{n={zernikes[1].n}}}^{{m={zernikes[1].m}}}$"

            k = results[('before', modes)]
            r1 = results[('after0', modes)]
            r2 = results[('after1', modes)]

            wf_mip = fig.add_subplot(gs[i, 0])
            wf_wavefront = inset_axes(wf_mip, width="40%", height="40%", loc='lower right', borderpad=0)

            ls_mip = fig.add_subplot(gs[i, 1])
            ml_wavefront = inset_axes(ls_mip, width="40%", height="40%", loc='lower right', borderpad=0)

            diff_mip = fig.add_subplot(gs[i, 2])
            diff_wavefront = inset_axes(diff_mip, width="40%", height="40%", loc='lower right', borderpad=0)

            diff_mip2 = fig.add_subplot(gs[i, 3])
            diff_wavefront2 = inset_axes(diff_mip2, width="40%", height="40%", loc='lower right', borderpad=0)

            plot_mip(
                xy=wf_mip,
                xz=None,
                yz=None,
                gamma=gamma,
                vol=np.transpose(np.rot90(k['gt_img'], k=2, axes=(1, 2)), axes=(0, 2, 1))
                    if transform_to_align_to_DM else k['gt_img'],
                cmap=psf_cmap,
                dxy=dxy,
                dz=dz,
                colorbar=False,
            )
            wf_mip.axis('on')
            wf_mip.set_title('Iteration 0\nPhaseRetrieval\nWF' if i == 0 else '')
            wf_mip.set_ylabel(modes)
            wf_mip.set_xlabel('')
            wf_mip.set_yticks([])
            wf_mip.set_xticks([])

            if i == 0:
                scalebar = AnchoredSizeBar(
                    wf_mip.transData,
                    2/dxy,
                    r'2 $\mu$m',
                    'upper right',
                    pad=0.25,
                    color='white',
                    frameon=False,
                    size_vertical=1
                )
                wf_mip.add_artist(scalebar)

            plot_wavefront(
                wf_wavefront,
                k['gt_wavefront'].wave(size=100),
                label=None,
                vmin=-.75,
                vmax=.75,
                nas=[1.0, .85],
                # hcolorbar=True if i == 3 else False, ## mplib breaks with "_raw_ticks istep=np.nonzero(large_steps)[0][0] IndexError: index 0 is out of bounds for axis 0 with size 0
            )

            plot_mip(
                xy=ls_mip,
                xz=None,
                yz=None,
                gamma=gamma,
                vol=np.transpose(np.rot90(k['ml_img'], k=2, axes=(1, 2)), axes=(0, 2, 1))
                    if transform_to_align_to_DM else k['ml_img'],
                cmap=psf_cmap,
                dxy=dxy,
                dz=dz,
                colorbar=False,
            )
            ls_mip.axis('on')
            ls_mip.set_title('Iteration 0\nOpticalNet\nLLSM' if i == 0 else '')
            ls_mip.set_xlabel('')
            ls_mip.set_yticks([])
            ls_mip.set_xticks([])

            plot_wavefront(
                ml_wavefront,
                k['ml_wavefront'].wave(size=100),
                label=None,
                vmin=-.75,
                vmax=.75,
                nas=[1.0, .85],
                # hcolorbar=True if i == 3 else False, ## mplib breaks with "_raw_ticks istep=np.nonzero(large_steps)[0][0] IndexError: index 0 is out of bounds for axis 0 with size 0
            )

            plot_mip(
                xy=diff_mip,
                xz=None,
                yz=None,
                gamma=gamma,
                vol=np.transpose(np.rot90(r1['ml_img'], k=2, axes=(1, 2)), axes=(0, 2, 1))
                    if transform_to_align_to_DM else r1['ml_img'],
                cmap=psf_cmap,
                dxy=dxy,
                dz=dz,
                colorbar=False,
            )
            diff_mip.axis('on')
            diff_mip.set_title('Iteration 1\nOpticalNet\nLLSM' if i == 0 else '')
            diff_mip.set_xlabel('')
            diff_mip.set_yticks([])
            diff_mip.set_xticks([])

            plot_wavefront(
                diff_wavefront,
                r1['diff_wavefront'].wave(size=100),
                label=None,
                vmin=-.75,
                vmax=.75,
                nas=[1.0, .85],
                # hcolorbar=True if i == 3 else False, ## mplib breaks with "_raw_ticks istep=np.nonzero(large_steps)[0][0] IndexError: index 0 is out of bounds for axis 0 with size 0
            )

            plot_mip(
                xy=diff_mip2,
                xz=None,
                yz=None,
                gamma=gamma,
                vol=np.transpose(np.rot90(r2['ml_img'], k=2, axes=(1, 2)), axes=(0, 2, 1))
                    if transform_to_align_to_DM else r2['ml_img'],
                cmap=psf_cmap,
                dxy=dxy,
                dz=dz,
                colorbar=False,
            )
            diff_mip2.axis('on')
            diff_mip2.set_title('Iteration 2\nOpticalNet\nLLSM' if i == 0 else '')
            diff_mip2.set_xlabel('')
            diff_mip2.set_yticks([])
            diff_mip2.set_xticks([])

            plot_wavefront(
                diff_wavefront2, r2['diff_wavefront'].wave(size=100),
                label=None,
                vmin=-.75,
                vmax=.75,
                nas=[1.0, .85],
                # hcolorbar=True if i == 3 else False,  ## mplib breaks with "_raw_ticks istep=np.nonzero(large_steps)[0][0] IndexError: index 0 is out of bounds for axis 0 with size 0
            )

        for k, (heatmapax, na) in enumerate(zip([heatmap1, heatmap85], [1.0, .85])):

            g = heatmaps[heatmaps['na'] == na].pivot(index="iteration_index", columns="modes",  values=val).T
            levels = np.arange(0, 1.75 if val == 'p2v_gt' else 1.25, .05)

            if custom_colormap:
                vmin, vmax, vcenter, step = levels[0], levels[-1], .5, .05
                highcmap = plt.get_cmap('magma_r', 256)
                lowcmap = plt.get_cmap('GnBu_r', 256)
                low = np.linspace(0, 1 - step, int(abs(vcenter - vmin) / step))
                high = np.linspace(0, 1 + step, int(abs(vcenter - vmax) / step))
                cmap = np.vstack((lowcmap(low), [1, 1, 1, 1], highcmap(high)))
                cmap = mcolors.ListedColormap(cmap)
                im = heatmapax.imshow(g.values, cmap=cmap, aspect='auto', vmin=levels[0], vmax=levels[-1])
            else:
                # colors = sns.color_palette('magma_r', n_colors=len(levels))
                # cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors, extend="max")
                # im = ax.imshow(g.values.T, cmap=cmap, norm=norm, aspect='auto')
                im = heatmapax.imshow(g.values, cmap='magma_r', aspect='auto', vmin=levels[0], vmax=levels[-1])

            heatmapax.yaxis.set_ticks_position('right')
            heatmapax.yaxis.set_label_position('right')

            heatmapax.set(
                yticks=range(g.shape[0]),
                xticks=range(g.shape[1]),
                xticklabels=g.columns
            )

            if k == 1:
                heatmapax.set_yticklabels(g.index)
                heatmapax.set_ylabel('Initial modes (ANSI index)')
            else:
                heatmapax.set_yticklabels([])

            heatmapax.set_xlabel(f'Iteration ($NA_{{{na:.2f}}}$)')

        cbar_ax = inset_axes(
            heatmap1,
            width="200%",
            height="2%",
            loc='upper left',
            bbox_to_anchor=(0, .28, 1, .75),
            bbox_transform=heatmap1.transAxes,
        )
        cbar = plt.colorbar(
            im,
            cax=cbar_ax,
            extend='max',
            spacing='proportional',
            orientation="horizontal",
            ticks=np.arange(0, 1.75 if val == 'p2v_gt' else 1.25, .25),
        )
        cbar_ax.set_title(label)
        cbar_ax.xaxis.set_ticks_position('top')
        cbar_ax.xaxis.set_label_position('top')

        plt.subplots_adjust(top=.9, bottom=.1, left=.1, right=.9, hspace=.06, wspace=.06)
        savesvg(plt,f'{savepath}_{val}.svg')
        plt.savefig(f'{savepath}_{val}.png', dpi=300,  pad_inches=.25)
        plt.savefig(f'{savepath}_{val}.pdf', dpi=300,  pad_inches=.25)
        logger.info(f'{savepath}_{val}')


def compare_ao_iterations(
    results: dict,
    num_iters: int,
    save_path: Path,
    psf_cmap: str = 'hot',
    fft_cmap: str = 'hot',
    gamma: float = .5,
    dxy: float = .097,
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

    fig = plt.figure(figsize=(12, 16))
    gs = fig.add_gridspec(4, num_iters)

    vmin = -.5
    vmax = .5

    noao_otf = results[0]['ml_img_fft'] / np.nanmax(results[0]['ml_img_fft'])
    noao_otf_hist = noao_otf.flatten()

    for i in range(num_iters):
        ax_img = fig.add_subplot(gs[0, i])
        ax_fft = fig.add_subplot(gs[1, i])

        if i == 0:
            otf = noao_otf
            plot_mip(
                xy=ax_img,
                xz=None,
                yz=None,
                gamma=gamma,
                label=rf'OpticalNet [$\gamma$={gamma}]',
                vol=results['noao_img'],
                cmap=psf_cmap,
                dxy=dxy,
                dz=dz,
                colorbar=True,
            )
            ax_img.set_title(f'No AO')
            ax_img.axis('off')

            scalebar = AnchoredSizeBar(
                ax_img.transData,
                5 / dxy,
                r'5 $\mu$m',
                'lower left',
                pad=0.1,
                color='white',
                frameon=False,
                size_vertical=1
            )
            ax_img.add_artist(scalebar)

            plot_mip(
                xy=ax_fft,
                xz=None,
                yz=None,
                label='OTF Strength',
                vol=otf,
                cmap=fft_cmap,
                dxy=dxy,
                dz=dz,
                colorbar=True,
                log=True,
                mip=False
            )
            contours = ax_fft.contour(
                np.nanmax(results['iotf'], axis=0),
                levels=[0, 1],
                origin='lower',
                linestyles='dashed',
                colors='green'
            )
            ax_fft.axis('off')
        else:
            ax_img.set_title(f'Round {i}')
            otf = results[i]['ml_img_fft'] / np.nanmax(results[0]['ml_img_fft'])
            otf_hist = otf.flatten()

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
            ax_img.axis('off')

            plot_mip(
                xy=ax_fft,
                xz=None,
                yz=None,
                vol=otf,
                cmap=fft_cmap,
                dxy=dxy,
                dz=dz,
                colorbar=False,
                log=True,
                mip=False
            )
            contours = ax_fft.contour(
                np.nanmax(results['iotf'], axis=0),
                levels=[0, 1],
                origin='lower',
                linestyles='dashed',
                colors='green'
            )
            ax_fft.axis('off')

        ax_hist = inset_axes(ax_fft, height="25%", width="100%", loc='upper center', borderpad=-5)
        ax_hist.hist(noao_otf_hist, density=True, bins=500, log=True, color='lightgrey', zorder=3)

        if i > 0:
            ax_hist.hist(otf_hist, density=True, bins=500, log=True, color=f'C0', alpha=.75, zorder=0)
            ax_hist.set_yticklabels([])

        ax_hist.grid(True, which="both", axis='y', lw=.5, ls='--', zorder=0, alpha=.5)
        ax_hist.spines['right'].set_visible(False)
        ax_hist.spines['top'].set_visible(False)
        ax_hist.spines['left'].set_visible(False)
        ax_hist.set_xlim(10**-3, 10**-.5)
        ax_hist.set_xticks([10**-3, 10**-2.5, 10**-2, 10**-1.5, 10**-1, 10**-.5])
        ax_hist.set_yticks([10**-5, 10**-3, 10**-1, 10], fontsize=10)
        ax_hist.set_xscale('log')
        ax_hist.tick_params(axis='both', labelsize=10)

        ax_img.set_xlabel('')
        ax_fft.set_xlabel('')

        p = results[i]['ml_wavefront']
        if p is not None:
            p_wave = p.wave(size=100)
            ax_ml = fig.add_subplot(gs[-2, i])
            plot_wavefront(ax_ml, p_wave, label='P2V', vmin=vmin, vmax=vmax, nas=[.95, .85])

        y = results[i]['gt_wavefront']
        if y is not None:
            y_wave = y.wave(size=100)
            ax_sh = fig.add_subplot(gs[-1, i])
            mat = plot_wavefront(ax_sh, y_wave, label='P2V', vmin=vmin, vmax=vmax, nas=[.95, .85])

        if i == 0:
            for ax, label in zip((ax_ml, ax_sh), ('OpticalNet', 'Shack–Hartmann')):
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("left", size="5%", pad=0.1)
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

        for ax in [ax_ml, ax_sh]:
            ax.axis('off')

    plt.subplots_adjust(top=.9, bottom=.1, left=.1, right=.9, hspace=.01, wspace=.01)
    plt.savefig(f'{save_path}.png', bbox_inches='tight', dpi=300, pad_inches=.25)
    plt.savefig(f'{save_path}.svg', bbox_inches='tight', dpi=300, pad_inches=.25)
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight', dpi=300, pad_inches=.25)

    fig = plt.figure(figsize=(8, 14))
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
        noao_ax.axis('off')
        scalebar = AnchoredSizeBar(
            noao_ax.transData,
            5 / dxy,
            r'5 $\mu$m',
            'lower left',
            pad=0.1,
            color='white',
            frameon=False,
            size_vertical=1
        )
        noao_ax.add_artist(scalebar)

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
        gt_ax.axis('off')
        scalebar = AnchoredSizeBar(
            gt_ax.transData,
            5 / dxy,
            r'5 $\mu$m',
            'lower left',
            pad=0.1,
            color='white',
            frameon=False,
            size_vertical=1
        )
        gt_ax.add_artist(scalebar)

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
        ml_ax.axis('off')
        scalebar = AnchoredSizeBar(
            ml_ax.transData,
            5 / dxy,
            r'5 $\mu$m',
            'lower left',
            pad=0.1,
            color='white',
            frameon=False,
            size_vertical=1
        )
        ml_ax.add_artist(scalebar)

        ml_axz.set_xlabel('')
        ml_axz.axis('off')
        scalebar = AnchoredSizeBar(
            ml_axz.transData,
            5 / dz,
            r'5 $\mu$m',
            'lower left',
            pad=0.1,
            color='white',
            frameon=False,
            size_vertical=1
        )
        ml_axz.add_artist(scalebar)

        if i == 0:
            noao_ax.set_title('No AO')
            gt_ax.set_title('Shack–Hartmann')
            ml_ax.set_title('OpticalNet')
        ml_axz.set_title('OpticalNet (XZ)')

    plt.subplots_adjust(top=.9, bottom=.1, left=.1, right=.9, hspace=0, wspace=0)
    plt.savefig(f'{save_path}_mips.png', bbox_inches='tight', dpi=300, pad_inches=.25)
    plt.savefig(f'{save_path}_mips.svg', bbox_inches='tight', dpi=300, pad_inches=.25)
    plt.savefig(f'{save_path}_mips.pdf', bbox_inches='tight', dpi=300, pad_inches=.25)


def otf_diagnosis(
        psfs: Union[np.ndarray, list],
        save_path: Union[Path, str],
        labels: list,
        lateral_voxel_size: float=0.097,
        axial_voxel_size: float=0.1,
        na_detection: float=1.0,
        lam_detection: float=.510,
        refractive_index: float=1.33,
        otf_floor: float = 0.5e-5,
):
    from embeddings import fft
    from preprocessing import resize_with_crop_or_pad

    plt.style.use("default")
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 8,
        'axes.autolimit_mode': 'round_numbers'
    })
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), layout="constrained")

    psfs = np.array(psfs)
    if psfs.ndim == 3:
        psfs = np.expand_dims(psfs, axis=0)  # make 4D array

    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)

    for i in range(psfs.shape[0]):
        linestyle = next(linecycler)
        voxel_size = np.array([axial_voxel_size, lateral_voxel_size, lateral_voxel_size])
        desired_cubic_fov = np.min(np.array(psfs[i].shape) * voxel_size)    # field of view in um to crop to

        psf = resize_with_crop_or_pad(psfs[i], np.round(desired_cubic_fov / voxel_size).astype(int))

        otf = np.abs(fft(psf))
        kx = np.fft.fftshift(np.fft.fftfreq(psf.shape[2], lateral_voxel_size/lam_detection))
        kz = np.fft.fftshift(np.fft.fftfreq(psf.shape[0], axial_voxel_size/lam_detection))
        otf /= np.max(otf)
        G = np.array(otf.shape)

        midpt = G // 2

        fattest_column = np.round(midpt[2] + G[2]*na_detection/refractive_index/4).astype(np.int32)

        LateralXWFCrossSection = np.squeeze(otf[midpt[0], midpt[1], :])               # line cut along x-axis
        LateralYWFCrossSection = np.squeeze(otf[midpt[0],        :, midpt[2]])        # line cut along y-axis
        AxialWFCrossSection =    np.squeeze(otf[:,        midpt[1], midpt[2]])        # line cut along z-axis
        BowtieWFCrossSection =   np.squeeze(otf[:,        midpt[1], fattest_column])  # line cut along z-axis @ bowtie

        axes[0].semilogy(kx, LateralXWFCrossSection,  lw='.75', linestyle=linestyle, label=labels[i])
        axes[1].semilogy(kz, BowtieWFCrossSection,    lw='.75', linestyle=linestyle, label=labels[i])
        axes[2].semilogy(kx, LateralYWFCrossSection,  lw='.75', linestyle=linestyle, label=labels[i])

    axes[0].legend(title='Lateral X WF\nCrossSection')
    axes[1].legend(title='Bowtie WF\nCrossSection')
    axes[2].legend(title='Lateral Y WF\nCrossSection')

    axes[0].set_xlabel('kx (1/$\lambda$)')
    axes[0].set_ylabel('OTF mag')
    axes[1].set_xlabel('kz (1/$\lambda$)')
    axes[2].set_xlabel('ky (1/$\lambda$)')
    axes[0].set_ylim(top=1, bottom=otf_floor)
    axes[1].set_ylim(top=1, bottom=otf_floor)
    axes[2].set_ylim(top=1, bottom=otf_floor)
    otf_diags_path = f'{save_path}_otf_diagnosis.svg'
    savesvg(fig, otf_diags_path)
    logger.info(f'OTF diagnosis saved to : {Path(otf_diags_path).resolve()}')
