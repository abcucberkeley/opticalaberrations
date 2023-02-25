from functools import partial

import matplotlib
matplotlib.use('Agg')

import warnings
from pathlib import Path
import logging
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import ImageGrid
from numpy.lib.stride_tricks import sliding_window_view
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from typing import Any
import numpy as np
import matplotlib.patches as patches
from line_profiler_pycharm import profile

from wavefront import Wavefront


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


def plot_mip(xy, xz, yz, vol, label='', gamma=.5, cmap='hot', dxy=.108, dz=.2):
    def formatter(x, pos, dd):
        return f'{np.ceil(x * dd).astype(int):1d}'

    vol = vol ** gamma
    vol = np.nan_to_num(vol)

    m = xy.imshow(np.max(vol, axis=0), cmap=cmap)
    xz.imshow(np.max(vol, axis=1), cmap=cmap)
    yz.imshow(np.max(vol, axis=2), cmap=cmap)

    cax = inset_axes(xy, width="10%", height="100%", loc='center left', borderpad=-5)
    cb = plt.colorbar(m, cax=cax)
    cax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    cax.set_ylabel(f"{label}")
    cax.yaxis.set_label_position("left")

    xy.yaxis.set_ticks_position('right')
    xy.xaxis.set_major_formatter(partial(formatter, dd=dxy))
    xy.yaxis.set_major_formatter(partial(formatter, dd=dxy))
    xy.xaxis.set_major_locator(plt.MaxNLocator(6))
    xy.yaxis.set_major_locator(plt.MaxNLocator(6))

    xz.yaxis.set_ticks_position('right')
    xz.xaxis.set_major_formatter(partial(formatter, dd=dxy))
    xz.yaxis.set_major_formatter(partial(formatter, dd=dz))
    xz.xaxis.set_major_locator(plt.MaxNLocator(6))
    xz.yaxis.set_major_locator(plt.MaxNLocator(6))

    yz.yaxis.set_ticks_position('right')
    yz.xaxis.set_major_formatter(partial(formatter, dd=dxy))
    yz.yaxis.set_major_formatter(partial(formatter, dd=dz))
    yz.xaxis.set_major_locator(plt.MaxNLocator(6))
    yz.yaxis.set_major_locator(plt.MaxNLocator(6))

    xy.set_xlabel('XY ($\mu$m)')
    yz.set_xlabel('XZ ($\mu$m)')
    xz.set_xlabel('YZ ($\mu$m)')
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

    dlimit = .1
    step = .1

    if vmin is None:
        vmin = np.floor(np.nanmin(phi)*2)/2     # round down to nearest 0.5 wave
        vmin = -1*dlimit if vmin > -0.01 else vmin

    if vmax is None:
        vmax = np.ceil(np.nanmax(phi)*2)/2  # round up to nearest 0.5 wave
        vmax = dlimit if vmax < 0.01 else vmax

    highcmap = plt.get_cmap('magma_r', 256)
    middlemap = plt.get_cmap('gist_gray', 256)
    lowcmap = plt.get_cmap('gist_earth_r', 256)

    ll = np.arange(vmin, -1*dlimit+step, step)
    hh = np.arange(dlimit, vmax+step, step)

    wave_cmap = np.vstack((
        lowcmap(.66 * ll / ll.min()),
        middlemap([.9, 1, .9]),
        highcmap(.66 * hh / hh.max())
    ))
    wave_cmap = mcolors.ListedColormap(wave_cmap)

    mat = iax.imshow(
        phi,
        cmap=wave_cmap,
        vmin=ll.min(),
        vmax=hh.max(),
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
        display: bool = False,
        psf_cmap: str = 'hot',
        gamma: float = .5,
        bar_width: float = .35,
        dxy: float = .108,
        dz: float = .2,
        pltstyle = None,
        transform_to_align_to_DM = False,
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

    fig = plt.figure(figsize=(17, 15))
    gs = fig.add_gridspec(5 if gt_psf is None else 6, 4)

    ax_gt = fig.add_subplot(gs[:2, 0])
    ax_pred = fig.add_subplot(gs[:2, 1])
    ax_diff = fig.add_subplot(gs[:2, 2])
    cax = fig.add_axes([0.1, 0.725, 0.02, .175])

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

    ax_zcoff = fig.add_subplot(gs[:, -1])

    dlimit = .25
    vmin = np.round(np.nanmin(y_wave))
    vmin = -1*dlimit if np.abs(vmin) == 0 else vmin

    vmax = np.round(np.nanmax(y_wave))
    vmax = dlimit if np.abs(vmax) == 0 else vmax

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

    ax_cxy.set_xlabel('XY ($\mu$m)')
    ax_cxz.set_xlabel('XZ ($\mu$m)')
    ax_cyz.set_xlabel('YZ ($\mu$m)')
    ax_xy.set_title(f"$\gamma$: {gamma:.2f}")
    ax_xz.set_title(f"PSNR: {psnr:.2f}")
    ax_yz.set_title(f"Max photon count: {maxcounts:.0f}")

    if transform_to_align_to_DM:
        psf = np.transpose(np.rot90(psf, k=2, axes=(1, 2)), axes=(0, 2, 1))    # 180 rotate, then transpose
    plot_mip(
        xy=ax_xy,
        xz=ax_xz,
        yz=ax_yz,
        vol=psf,
        label='Input (MIP)',
        cmap=psf_cmap,
        dxy=dxy,
        dz=dz
    )
    plot_mip(
        xy=ax_pxy,
        xz=ax_pxz,
        yz=ax_pyz,
        vol=predicted_psf,
        label='Predicted',
        cmap=psf_cmap,
        dxy=dxy,
        dz=dz
    )
    plot_mip(
        xy=ax_cxy,
        xz=ax_cxz,
        yz=ax_cyz,
        vol=corrected_psf,
        label='Corrected',
        cmap=psf_cmap,
        dxy=dxy,
        dz=dz
    )

    if gt_psf is not None:
        ax_xygt = fig.add_subplot(gs[3, 0])
        ax_xzgt = fig.add_subplot(gs[3, 1])
        ax_yzgt = fig.add_subplot(gs[3, 2])
        plot_mip(xy=ax_xygt, xz=ax_xzgt, yz=ax_yzgt, vol=gt_psf, label='Simulated')

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

    for ax in [ax_gt, ax_pred, ax_diff]:
        ax.axis('off')

    plt.subplots_adjust(top=0.95, right=0.95, wspace=.2, hspace=.2)
    plt.savefig(f'{save_path}.svg', dpi=300, bbox_inches='tight', pad_inches=.25)
    # plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight', pad_inches=.25)

    if display:
        plt.tight_layout()
        plt.show()


@profile
def diagnosis(pred: Wavefront, save_path: Path, pred_std: Any = None):

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
    gs = fig.add_gridspec(1, 3)
    ax_wavefront = fig.add_subplot(gs[0, -1])
    ax_zcoff = fig.add_subplot(gs[0, :-1])

    plot_wavefront(
        ax_wavefront,
        pred_wave,
        label='Predicted wavefront',
        vcolorbar=True,
        nas=(.55, .65, .75, .85, .95, 1.)
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

    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.35, wspace=0.1)
    plt.savefig(f'{save_path}.svg', dpi=300, bbox_inches='tight', pad_inches=.1)


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

    plt.subplots_adjust(top=0.95, right=0.95, wspace=.2)
    plt.savefig(f'{save_path}.svg', dpi=300, bbox_inches='tight', pad_inches=.25)


@profile
def tiles(
    data: np.ndarray,
    save_path: Path,
    strides: int = 64,
    window_size: int = 64,
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
    ztiles = np.array_split(range(data.shape[0]), data.shape[0]//window_size)

    for z, idx in enumerate(ztiles):
        sample = np.max(data[idx], axis=0)
        tiles = sliding_window_view(sample, window_shape=[window_size, window_size])[::strides, ::strides]
        nrows, ncols = tiles.shape[0], tiles.shape[1]
        tiles = np.reshape(tiles, (-1, window_size, window_size))

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
                im = grid[i].imshow(tiles[i], cmap='hot', vmin=0, vmax=1, aspect='equal')
                grid[i].set_title(f"z{z}-y{y}-x{x}", pad=1)
                grid[i].axis('off')
                i += 1

        cbar = grid.cbar_axes[0].colorbar(im)
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.set_yticks([])
        cbar.ax.set_xlabel(rf"$\gamma$={gamma}")

        plt.savefig(f'{save_path}_z{z}.svg', dpi=300, bbox_inches='tight', pad_inches=.25)


@profile
def wavefronts(
    predictions: pd.DataFrame,
    ztiles: int,
    nrows: int,
    ncols: int,
    save_path: Path,
    wavelength: float = .605,
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
                roi = f"z{z}-y{y}-x{x}"
                pred = Wavefront(predictions[roi].values, lam_detection=wavelength)
                pred_wave = pred.wave(size=100)
                plot_wavefront(grid[i], pred_wave)
                grid[i].set_title(roi, pad=1)
                i += 1

        cbar = grid.cbar_axes[0].colorbar(mat)
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.set_yticks([])
        cbar.ax.set_title(f'$\lambda = {wavelength}~\mu$m')

        plt.savefig(f'{save_path}_z{z}.svg', dpi=300, bbox_inches='tight', pad_inches=.25)


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

    plt.tight_layout()
    plt.savefig(f'{savepath}.svg', dpi=300, bbox_inches='tight', pad_inches=.25)
    # plt.savefig(f'{savepath}.png', dpi=300, bbox_inches='tight', pad_inches=.25)


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

    plt.subplots_adjust(top=0.95, right=0.95, wspace=.2)
    plt.savefig(f'{savepath}.svg', dpi=300, bbox_inches='tight', pad_inches=.25)
    # plt.savefig(f'{savepath}.png', dpi=300, bbox_inches='tight', pad_inches=.25)
