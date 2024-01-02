import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.set_loglevel('error')

import warnings
from pathlib import Path
import logging
import sys
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import raster_geometry as rg
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from tifffile import imwrite
import numpy as np
import seaborn as sns
from tqdm import tqdm, trange
import matplotlib.patches as patches
from astropy import convolution
import tensorflow as tf
from line_profiler_pycharm import profile

import utils
from wavefront import Wavefront
from zernike import Zernike
from synthetic import SyntheticPSF
from embeddings import fourier_embeddings
from preprocessing import prep_sample


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


@profile
def plot_zernike_pyramid(amp=.1, wavelength=.510, weighted=False):
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
    })

    cmaps = ['Reds', 'OrRd', 'YlOrRd_r', 'YlGn', 'Greens', 'YlGnBu', 'Blues', 'BuPu', 'Purples', 'PuRd', 'pink']
    cmaps = [sns.color_palette(c, n_colors=256, as_cmap=True) for c in cmaps]

    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height

    for nth_order in range(1, 11):
        for k, savepath in enumerate([
            Path(f'../data/zernikes/{nth_order}th_zernike_pyramid'),
            Path(f'../data/zernikes/{nth_order}th_zernike_pyramid_db')
        ]):
            if weighted:
                savepath = Path(f"{savepath.parent}/{savepath.name}_weighted")

            if k == 0:
                plt.style.use('default')
            else:
                plt.style.use('dark_background')

            fig = plt.figure(figsize=(3*nth_order, 2*nth_order))
            gs = fig.add_gridspec(nth_order+1, 2*nth_order+1)

            for n in trange(nth_order+1, file=sys.stdout):
                for i, m in enumerate(range(-nth_order, nth_order+1)):
                    ax = fig.add_subplot(gs[n, i])
                    ax.axis('off')

                    if (n == 0 and m == 0) or (n > 0):
                        try:
                            z = Zernike((n, m))
                            w = Wavefront({z.index_ansi: amp}, lam_detection=wavelength).wave(size=100)

                            # if n == 0 and m == 0:
                            #     mode = f"$\lambda$ = {wavelength} $\mu$m\n" \
                            #            f"Amplitude={amp} $\mu$m RMS\n\n"\
                            #            f"{round(np.nanmax(w) - np.nanmin(w), 2):.2f} $\lambda$\n" \
                            #            f"{z.index_ansi}: $Z_{{n={z.n}}}^{{m={z.m}}}$"
                            # else:
                            #     mode = f"{round(np.nanmax(w) - np.nanmin(w), 2):.2f} $\lambda$\n" \
                            #            f"{z.index_ansi}: $Z_{{n={z.n}}}^{{m={z.m}}}$"

                            mode = f"{round(np.nanmax(w) - np.nanmin(w), 2):.2f} $\lambda$\n" \
                                   f"{z.index_ansi}: $Z_{{n={z.n}}}^{{m={z.m}}}$"
                            ax.set_title(mode)

                            if weighted:
                                if z.index_ansi in [0, 1, 2, 4]:
                                    mat = ax.imshow(
                                        w,
                                        cmap='Greys',
                                        vmin=-.5,
                                        vmax=.5,
                                    )
                                elif n >= 2 and n <= 4:
                                    mat = ax.imshow(
                                        w,
                                        cmap='Spectral_r',
                                        vmin=-.5,
                                        vmax=.5,
                                    )
                                else:
                                    mat = ax.imshow(
                                        w,
                                        cmap=cmaps[abs(m)],
                                        vmin=-.5,
                                        vmax=.5,
                                    )
                            else:
                                mat = ax.imshow(
                                    w,
                                    cmap='Spectral_r',
                                    vmin=-.5,
                                    vmax=.5,
                                )

                            ax.set_yticks([])
                            ax.set_xticks([])

                        except ValueError:

                            if weighted and n > 4:
                                ax.text(
                                    0.5 * (left + right), 0.5 * (bottom + top),
                                    f"1/{abs(m) + 2}",
                                    horizontalalignment='center',
                                    verticalalignment='center',
                                    transform=ax.transAxes
                                )
                            continue

            plt.subplots_adjust(top=0.95, right=0.95, wspace=.2)
            plt.savefig(savepath.with_suffix('.png'), bbox_inches='tight', pad_inches=.25)
            plt.savefig(savepath.with_suffix('.pdf'), bbox_inches='tight', pad_inches=.25)
            plt.savefig(savepath.with_suffix('.svg'), bbox_inches='tight', pad_inches=.25)


@profile
def plot_embedding_pyramid(
        res=64,
        n_modes=60,
        wavelength=.510,
        x_voxel_size=.125,
        y_voxel_size=.125,
        z_voxel_size=.2,
        psf_type='../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat',
        datadir='../data/embeddings',
        embedding_option='spatial_planes',
):
    def formatter(x, pos):
        val_str = '{:.1g}'.format(x)
        if np.abs(x) > 0 and np.abs(x) < 1:
            return val_str.replace("0", "", 1)
        else:
            return val_str

    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
    })

    vmin, vmax, vcenter, step = 0, 2, 1, .1

    alpha_cmap = np.vstack((
        plt.get_cmap('GnBu_r', 256)(
            np.linspace(0, 1 - step, int(abs(vcenter - vmin) / step))
        ),
        [1, 1, 1, 1],
        plt.get_cmap('YlOrRd', 256)(
            np.linspace(0, 1 + step, int(abs(vcenter - vmax) / step))
        )
    ))
    alpha_cmap = mcolors.ListedColormap(alpha_cmap)

    phi_cmap = np.vstack((
        plt.get_cmap('GnBu_r', 256)(
            np.linspace(0, 1 - step, int(abs(vcenter - vmin) / step))
        ),
        [1, 1, 1, 1],
        plt.get_cmap('OrRd', 256)(
            np.linspace(0, 1 + step, int(abs(vcenter - vmax) / step))
        )
    ))
    phi_cmap = mcolors.ListedColormap(phi_cmap)

    gen = SyntheticPSF(
        psf_type=psf_type,
        amplitude_ranges=(-1, 1),
        n_modes=n_modes,
        lam_detection=wavelength,
        psf_shape=3 * [res],
        x_voxel_size=x_voxel_size,
        y_voxel_size=y_voxel_size,
        z_voxel_size=z_voxel_size,
        cpu_workers=-1,
    )
    waves = np.arange(-.3, .35, step=.05).round(2)

    for nth_order in range(2, 11):
        for amp in tqdm(waves, file=sys.stdout):
            title = f"{int(np.sign(amp))}x{str(np.abs(amp).round(3)).replace('0.', 'p')}"

            embeddings = {}
            for n in range(nth_order + 1):
                for i, m in enumerate(range(-nth_order, nth_order + 1)):
                    if (n == 0 and m == 0) or (n > 0):
                        try:
                            z = Zernike((n, m))
                            wavefront = Wavefront({z.index_ansi: amp}, lam_detection=wavelength)

                            psf = gen.single_psf(
                                wavefront,
                                normed=True,
                                meta=False,
                            )
                            psf /= np.sum(psf)
                            psf *= 100000
                            psf = utils.add_noise(psf)

                            psf = prep_sample(
                                psf,
                                sample_voxel_size=gen.voxel_size,
                                model_fov=gen.psf_fov,
                                remove_background=True,
                                normalize=True,
                                min_psnr=0,
                                na_mask=gen.na_mask
                            )

                            emb = fourier_embeddings(
                                psf,
                                iotf=gen.iotf,
                                na_mask=gen.na_mask,
                                no_phase=False,
                                remove_interference=False,
                                embedding_option=embedding_option,
                            )
                            embeddings[z] = emb
                        except ValueError:
                            continue

            for plane in trange(6):
                outdir = Path(f'{datadir}/{embedding_option}/{nth_order}th/POI_{plane}')
                outdir.mkdir(exist_ok=True, parents=True)

                fig = plt.figure(figsize=(3 * nth_order, 2 * nth_order))
                gs = fig.add_gridspec(nth_order + 1, 2 * nth_order + 1)

                for n in range(nth_order + 1):
                    for i, m in enumerate(range(-nth_order, nth_order + 1)):
                        ax = fig.add_subplot(gs[n, i])
                        ax.axis('off')

                        if (n == 0 and m == 0) or (n > 0):
                            try:
                                z = Zernike((n, m))
                                w = Wavefront({z.index_ansi: amp}, lam_detection=wavelength).wave(size=100)
                                abr = round((np.nanmax(w) - np.nanmin(w)) * np.sign(amp), 1)

                                if n == 0 and m == 0:
                                    mode = f"$\lambda$ = {wavelength} $\mu$m\n" \
                                           f"Amplitude={amp} $\mu$m RMS\n\n" \
                                           f"{abr} $\lambda$\n" \
                                           f"{z.index_ansi}: $Z_{{n={z.n}}}^{{m={z.m}}}$"
                                else:
                                    mode = f"{abr} $\lambda$\n" \
                                           f"{z.index_ansi}: $Z_{{n={z.n}}}^{{m={z.m}}}$"

                                plt.figure(fig.number)
                                mat = ax.imshow(
                                    embeddings[z][plane],
                                    cmap=alpha_cmap if plane < 3 else phi_cmap,
                                    vmin=vmin if plane < 3 else -.5,
                                    vmax=vmax if plane < 3 else .5,
                                )

                                cax = inset_axes(ax, width="100%", height="10%", loc='lower center', borderpad=-1)
                                cbar = plt.colorbar(mat, cax=cax, extend='both', format=formatter, orientation='horizontal')
                                cbar.ax.xaxis.set_ticks_position('bottom')
                                cbar.ax.xaxis.set_label_position('top')
                                cbar.ax.set_xticks([
                                    vmin if plane < 3 else -.5,
                                    vcenter if plane < 3 else 0,
                                    vmax if plane < 3 else .5,
                                ])
                                ax.set_title(mode)

                            except ValueError:
                                continue

                plt.subplots_adjust(top=0.95, right=0.95, wspace=.2)
                plt.savefig(f"{outdir}/{title}.png", bbox_inches='tight', pad_inches=.25)


@profile
def plot_training_dist(n_samples=10, batch_size=10, wavelength=.510):
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    for dist in ['single', 'bimodal', 'multinomial', 'powerlaw', 'dirichlet', 'mixed']:
        psfargs = dict(
            n_modes=55,
            psf_type='../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat',
            distribution=dist,
            mode_weights='pyramid',
            signed=True,
            rotate=True,
            gamma=.75,
            lam_detection=wavelength,
            amplitude_ranges=(0, 1),
            psf_shape=(32, 32, 32),
            x_voxel_size=.125,
            y_voxel_size=.125,
            z_voxel_size=.2,
            batch_size=batch_size,
            cpu_workers=-1,
        )

        n_batches = n_samples // batch_size
        peaks = []
        zernikes = pd.DataFrame([], columns=range(1, psfargs['n_modes'] + 1))

        ## Testing dataset
        min_amps = np.arange(0, .30, .01).round(3)
        max_amps = np.arange(.01, .31, .01).round(3)

        for mina, maxa in zip(min_amps, max_amps):
            psfargs['amplitude_ranges'] = (mina, maxa)
            for _, (psfs, ys) in zip(range(n_batches), SyntheticPSF(**psfargs).generator()):
                zernikes = zernikes.append(
                    pd.DataFrame(ys, columns=range(1, psfargs['n_modes'] + 1)),
                    ignore_index=True
                )
                ps = [Wavefront(p, lam_detection=wavelength).peak2valley() for p in ys]
                logger.info(f'Range[{mina}, {maxa}]')
                peaks.extend(ps)

        logger.info(zernikes.round(2))

        fig, (pax, cax, zax) = plt.subplots(1, 3, figsize=(16, 4))

        sns.histplot(peaks, kde=True, ax=pax, color='dimgrey')

        pax.set_xlabel(
            'peak-to-valley aberration\n'
            rf'($\lambda = {int(wavelength*1000)}~nm$)'
        )
        pax.set_ylabel(rf'Samples')

        zernikes = np.abs(zernikes)
        zernikes = zernikes.loc[(zernikes != 0).any(axis=1)]
        zernikes = zernikes.div(zernikes.sum(axis=1), axis=0)
        logger.info(zernikes.round(2))

        dmodes = (zernikes[zernikes > .05]).count(axis=1)
        hist, bins = np.histogram(dmodes, bins=zernikes.columns.values)
        idx = (hist > 0).nonzero()
        hist = hist / hist.sum()

        if len(idx[0]) != 0:
            bars = sns.barplot(bins[idx], hist[idx], ax=cax, palette='Accent')
            for index, label in enumerate(bars.get_xticklabels()):
                if index % 2 == 0:
                    label.set_visible(True)
                else:
                    label.set_visible(False)

        cax.set_xlabel(
            f'Number of highly influential modes\n'
            rf'$\alpha_i / \sum_{{k=1}}^{{{psfargs["n_modes"]}}}{{\alpha_{{k}}}} > 5\%$'
        )

        modes = zernikes.sum(axis=0)
        modes /= modes.sum(axis=0)

        cmap = sns.color_palette("viridis", len(modes))
        rank = modes.argsort().argsort()
        bars = sns.barplot(modes.index-1, modes.values, ax=zax, palette=np.array(cmap[::-1])[rank])

        for index, label in enumerate(bars.get_xticklabels()):
            if index % 4 == 0:
                label.set_visible(True)
            else:
                label.set_visible(False)

        zax.set_xlabel(f'Influential modes (ANSI)')

        pax.grid(True, which="both", axis='both', lw=.5, ls='--', zorder=0)
        cax.grid(True, which="both", axis='both', lw=.5, ls='--', zorder=0)
        zax.grid(True, which="both", axis='both', lw=.5, ls='--', zorder=0)

        name = f'{psfargs["distribution"]}_{psfargs["n_modes"]}modes_gamma_{str(psfargs["gamma"]).replace(".", "p")}'
        plt.savefig(
            f'../data/{name}.png',
            dpi=300, bbox_inches='tight', pad_inches=.25
        )


def plot_fov(n_modes=55, wavelength=.605, psf_cmap='hot', x_voxel_size=.15, y_voxel_size=.15, z_voxel_size=.6):
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    waves = np.round(np.arange(0, .5, step=.1), 2)
    res = [128, 64, 32]
    offsets = [0, 32, 48]
    savedir = '../data/fov/ratio_150x-150y-600z'
    logger.info(waves)

    for i in range(3, n_modes):
        fig = plt.figure(figsize=(35, 55))
        gs = fig.add_gridspec(len(waves)*len(res), 8)

        grid = {}
        for a, j in zip(waves, np.arange(0, len(waves)*len(res), step=3)):
            for k, r in enumerate(res):
                for c in range(8):
                    grid[(a, r, c)] = fig.add_subplot(gs[j+k, c])

        # from pprint import pprint
        # pprint(grid)

        for j, amp in enumerate(tqdm(waves, desc=f'Mode [#{i}]', file=sys.stdout)):
            phi = np.zeros(n_modes)
            phi[i] = amp
            w = Wavefront(phi, order='ansi', lam_detection=wavelength)

            for r in res:
                gen = SyntheticPSF(
                    amplitude_ranges=(-1, 1),
                    n_modes=n_modes,
                    lam_detection=wavelength,
                    psf_shape=3*[r],
                    x_voxel_size=x_voxel_size,
                    y_voxel_size=y_voxel_size,
                    z_voxel_size=z_voxel_size,
                    cpu_workers=-1,
                )
                window = gen.single_psf(w, normed=True)
                #window = center_crop(psf, crop_shape=tuple(3 * [r]))

                fft = np.fft.fftn(window)
                fft = np.fft.fftshift(fft)
                fft = np.abs(fft)
                # fft[fft == np.inf] = np.nan
                # fft[fft == -np.inf] = np.nan
                # fft[fft == np.nan] = np.min(fft)
                # fft = np.log10(fft)
                fft /= np.max(fft)

                perfect_psf = gen.single_psf(phi=Wavefront(np.zeros(n_modes)))
                perfect_fft = np.fft.fftn(perfect_psf)
                perfect_fft = np.fft.fftshift(perfect_fft)
                perfect_fft = np.abs(perfect_fft)
                perfect_fft /= np.max(perfect_fft)

                fft = fft / perfect_fft
                fft[fft > 1] = 0

                NA_det = 1.0
                n = 1.33
                lambda_det = wavelength * 1000
                kx = ky = 4 * np.pi * NA_det / lambda_det
                kz = 2 * np.pi * ((n - np.sqrt(n**2 - NA_det**2)) / lambda_det)

                N = np.array(window.shape)
                px = x_voxel_size * 1000
                py = y_voxel_size * 1000
                pz = z_voxel_size * 1000

                # get the axis lengths of the support
                hN = np.ceil((N - 1) / 2)
                a = 2 * hN[2] * (kx * px) / (2 * np.pi)
                b = 2 * hN[1] * (ky * py) / (2 * np.pi)
                c = 2 * hN[0] * (kz * pz) / (2 * np.pi)

                # formulate the ellipse
                Z, Y, X = np.mgrid[-hN[0]:hN[0], -hN[1]:hN[1], -hN[2]:hN[2]]
                mask = np.sqrt(X**2/a**2 + Y**2/b**2 + Z**2/c**2)
                mask = mask <= 1

                for ax in range(3):
                    vol = np.max(window, axis=ax) ** .5
                    grid[(amp, r, ax)].imshow(vol, cmap=psf_cmap, vmin=0, vmax=1)
                    grid[(amp, r, ax)].set_aspect('equal')

                    if ax == 0:
                        vol = fft[fft.shape[0]//2, :, :]
                        vol *= mask[mask.shape[0] // 2, :, :]
                    elif ax == 1:
                        vol = fft[:, fft.shape[1]//2, :]
                        vol *= mask[:, mask.shape[1] // 2, :]
                    else:
                        vol = fft[:, :, fft.shape[2]//2]
                        vol *= mask[:, :, mask.shape[2] // 2]

                    # vol = np.max(fft, axis=ax) ** .5
                    # vol = np.nan_to_num(vol)
                    grid[(amp, r, ax+3)].imshow(vol, vmin=0, vmax=1)
                    grid[(amp, r, ax+3)].set_aspect('equal')

                    # draw boxes
                    for z, rr in enumerate(res):
                        rect = patches.Rectangle(
                            (offsets[z], offsets[z]),
                            rr, rr,
                            linewidth=1,
                            edgecolor='w',
                            facecolor='none'
                        )
                        grid[(amp, 128, ax)].add_patch(rect)

                    grid[(amp, r, ax)].axis('off')
                    grid[(amp, r, ax+3)].axis('off')

                grid[(amp, r, 6)].semilogy(fft[:, fft.shape[0]//2, fft.shape[0]//2], '-', label='XY')
                grid[(amp, r, 6)].semilogy(fft[fft.shape[0]//2, :, fft.shape[0]//2], '--', label='XZ')
                grid[(amp, r, 6)].semilogy(fft[fft.shape[0]//2, fft.shape[0]//2, :], ':', label='YZ')
                grid[(amp, r, 6)].grid(True, which="both", axis='both', lw=.5, ls='--', zorder=0)
                grid[(amp, r, 6)].legend(frameon=False, ncol=1, bbox_to_anchor=(1.0, 1.0), loc='upper left')
                grid[(amp, r, 6)].set_aspect('equal')

                mat = grid[(amp, r, 7)].contourf(
                    w.wave(100),
                    levels=np.arange(-10, 10, step=1),
                    cmap='Spectral_r',
                    extend='both'
                )
                grid[(amp, r, 7)].axis('off')
                grid[(amp, r, 7)].set_aspect('equal')

                grid[(amp, r, 7)].set_title(f'{round(w.peak2valley())} waves')
                grid[(amp, r, 0)].set_title('XY')
                grid[(amp, r, 3)].set_title('XY')

                grid[(amp, r, 1)].set_title('XZ')
                grid[(amp, r, 4)].set_title('XZ')

                grid[(amp, r, 2)].set_title('YZ')
                grid[(amp, r, 5)].set_title('YZ')

        plt.subplots_adjust(top=0.95, right=0.95, wspace=.2)
        plt.savefig(f'{savedir}/fov_mode_{i}.pdf', bbox_inches='tight', pad_inches=.25)


def plot_embeddings(
        res=64,
        padsize=None,
        n_modes=55,
        wavelength=.510,
        x_voxel_size=.125,
        y_voxel_size=.125,
        z_voxel_size=.2,
        psf_type='../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat',
        savepath='../data/embeddings',
        embedding_option='spatial_planes',
):

    savepath = f"{savepath}/{embedding_option}/{int(wavelength*1000)}/x{int(x_voxel_size*1000)}-y{int(y_voxel_size*1000)}-z{int(z_voxel_size*1000)}"
    plt.rcParams.update({
        'font.size': 8,
        'axes.titlesize': 10,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
    })

    vmin, vmax, vcenter, step = 0, 2, 1, .1
    highcmap = plt.get_cmap('YlOrRd', 256)
    lowcmap = plt.get_cmap('YlGnBu_r', 256)
    low = np.linspace(0, 1 - step, int(abs(vcenter - vmin) / step))
    high = np.linspace(0, 1 + step, int(abs(vcenter - vmax) / step))
    cmap = np.vstack((lowcmap(low), [1, 1, 1, 1], highcmap(high)))
    cmap = mcolors.ListedColormap(cmap)

    waves = np.arange(-.3, .35, step=.05).round(3)
    # waves = np.arange(-.075, .08, step=.015).round(3) ## small
    logger.info(waves)

    gen = SyntheticPSF(
        psf_type=psf_type,
        amplitude_ranges=(-1, 1),
        n_modes=n_modes,
        lam_detection=wavelength,
        psf_shape=3*[res],
        x_voxel_size=x_voxel_size,
        y_voxel_size=y_voxel_size,
        z_voxel_size=z_voxel_size,
        cpu_workers=-1,
    )

    for mode in trange(3, n_modes):
        fig, axes = plt.subplots(6, len(waves)+1, figsize=(12, 6))

        for i, amp in enumerate(waves):
            phi = np.zeros(n_modes)
            phi[mode] = amp

            psf, amps, lls_defocus_offset = gen.single_psf(
                phi=phi,
                normed=True,
                meta=True,
            )

            wavefront = Wavefront(phi, lam_detection=gen.lam_detection)
            abr = round(wavefront.peak2valley() * np.sign(amp), 1)
            axes[0, i+1].set_title(f'{abr}$\\lambda$')

            outdir = Path(f'{savepath}/i{res}_pad_{padsize}/mode_{mode}/embeddings/')
            outdir.mkdir(exist_ok=True, parents=True)

            emb = fourier_embeddings(
                psf,
                iotf=gen.iotf,
                na_mask=gen.na_mask,
                no_phase=False,
                remove_interference=False,
                embedding_option=embedding_option,
                plot=f"{outdir}/{str(abr).replace('.', 'p')}",
            )
            imwrite(f"{outdir}/{str(abr).replace('.', 'p')}.tif", emb, compression='deflate')

            plt.figure(fig.number)
            for ax in range(6):
                if amp == waves[-1]:
                    mat = axes[ax, 0].contourf(
                        Wavefront(phi, lam_detection=wavelength).wave(100),
                        levels=np.arange(-2, 2, step=.1),
                        cmap='Spectral_r',
                        extend='both'
                    )
                    axes[ax, 0].axis('off')
                    axes[ax, 0].set_aspect('equal')

                m = axes[ax, i+1].imshow(
                    emb[ax, :, :],
                    cmap=cmap if ax < 3 else 'Spectral_r',
                    vmin=vmin if ax < 3 else -.5,
                    vmax=vmax if ax < 3 else .5,
                )
                axes[ax, i+1].set_aspect('equal')
                axes[ax, i+1].axis('off')

                cax = inset_axes(
                    axes[ax, -1],
                    width="10%",
                    height="100%",
                    loc='center right',
                    borderpad=-1
                )
                cb = plt.colorbar(m, cax=cax)
                cax.yaxis.set_label_position("right")

        plt.subplots_adjust(top=0.95, right=0.95, wspace=.2)
        plt.savefig(f'{savepath}/i{res}_pad{padsize}_mode_{mode}.pdf', bbox_inches='tight', pad_inches=.25)


def plot_rotations(
        res=64,
        padsize=None,
        n_modes=55,
        wavelength=.510,
        x_voxel_size=.125,
        y_voxel_size=.125,
        z_voxel_size=.2,
        psf_type='../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat',
        savepath='../data/rotations',
        embedding_option='spatial_planes',
):

    savepath = f"{savepath}/{embedding_option}/{int(wavelength * 1000)}/x{int(x_voxel_size * 1000)}-y{int(y_voxel_size * 1000)}-z{int(z_voxel_size * 1000)}"
    plt.rcParams.update({
        'font.size': 8,
        'axes.titlesize': 10,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
    })

    vmin, vmax, vcenter, step = 0, 2, 1, .1
    highcmap = plt.get_cmap('YlOrRd', 256)
    lowcmap = plt.get_cmap('YlGnBu_r', 256)
    low = np.linspace(0, 1 - step, int(abs(vcenter - vmin) / step))
    high = np.linspace(0, 1 + step, int(abs(vcenter - vmax) / step))
    cmap = np.vstack((lowcmap(low), [1, 1, 1, 1], highcmap(high)))
    cmap = mcolors.ListedColormap(cmap)

    degrees = np.arange(0, 195, 15).astype(int)
    logger.info(degrees)
    amp = .1

    gen = SyntheticPSF(
        psf_type=psf_type,
        amplitude_ranges=(-1, 1),
        n_modes=n_modes,
        lam_detection=wavelength,
        psf_shape=3 * [res],
        x_voxel_size=x_voxel_size,
        y_voxel_size=y_voxel_size,
        z_voxel_size=z_voxel_size,
        cpu_workers=-1,
    )

    for mode in trange(3, n_modes):
        fig, axes = plt.subplots(6, len(degrees) + 1, figsize=(12, 6))

        for i, deg in enumerate(degrees):
            axes[0, i + 1].set_title(f'{deg}$^\circ$')

            outdir = Path(f'{savepath}/i{res}_pad_{padsize}/mode_{mode}/embeddings/')
            outdir.mkdir(exist_ok=True, parents=True)

            v = np.zeros(n_modes)
            v[mode] = amp

            z = Zernike(mode)
            twin = Zernike((z.n, z.m * -1))
            wave = Wavefront(v, lam_detection=gen.lam_detection)

            if z.m != 0 and wave.zernikes.get(twin) is not None:
                v[z.index_ansi] = amp * np.cos(deg / 360 * 2 * np.pi)
                v[twin.index_ansi] = amp * np.sin(deg / 360 * 2 * np.pi)

            psf, amps, lls_defocus_offset = gen.single_psf(
                phi=v,
                normed=True,
                meta=True,
            )

            emb = fourier_embeddings(
                psf,
                iotf=gen.iotf,
                na_mask=gen.na_mask,
                no_phase=False,
                remove_interference=False,
                embedding_option=embedding_option,
                plot=f"{outdir}/amp{str(amp).replace('.', 'p')}_deg{str(deg)}",
            )
            imwrite(f"{outdir}/amp{str(amp).replace('.', 'p')}_deg{str(deg)}.tif", emb, compression='deflate')

            plt.figure(fig.number)
            for ax in range(6):
                if i == 0:
                    mat = axes[ax, 0].contourf(
                        Wavefront(v, lam_detection=wavelength).wave(100),
                        levels=np.arange(-1, 1, step=.1),
                        cmap='Spectral_r',
                        extend='both'
                    )
                    axes[ax, 0].axis('off')
                    axes[ax, 0].set_aspect('equal')

                m = axes[ax, i + 1].imshow(
                    emb[ax, :, :],
                    cmap=cmap if ax < 3 else 'Spectral_r',
                    vmin=vmin if ax < 3 else -.5,
                    vmax=vmax if ax < 3 else .5,
                )
                axes[ax, i + 1].set_aspect('equal')
                axes[ax, i + 1].axis('off')

                cax = inset_axes(
                    axes[ax, -1],
                    width="10%",
                    height="100%",
                    loc='center right',
                    borderpad=-1
                )
                cb = plt.colorbar(m, cax=cax)
                cax.yaxis.set_label_position("right")

        plt.subplots_adjust(top=0.95, right=0.95, wspace=.2)
        plt.savefig(f'{savepath}/i{res}_pad{padsize}_mode_{mode}.pdf', bbox_inches='tight', pad_inches=.25)


def plot_shapes_embeddings(
        res=64,
        padsize=None,
        shapes=5,
        wavelength=.510,
        x_voxel_size=.125,
        y_voxel_size=.125,
        z_voxel_size=.2,
        psf_type='../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat',
        savepath='../data/shapes_embeddings',
        n_modes=55,
        embedding_option='spatial_planes',
):
    """ Plot the embeddings for different puncta sizes (aka different "shapes")

    Args:
        res: resolution. Defaults to 64.
        padsize: Uh, doesn't get used here.  It will appear in the name of the folder path. Defaults to None.
        shapes: Number of puncta sizes to test. Defaults to 5 different sizes
        wavelength:   Defaults to .510 microns
        x_voxel_size: Defaults to .125 microns
        y_voxel_size: Defaults to .125 microns
        z_voxel_size: Defaults to .2   microns
        psf_type: Defaults to '../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat'.
        savepath: Defaults to '../data/shapes_embeddings'.
    """
    def sphere(image_size, radius=.5, position=.5):
        img = rg.sphere(shape=image_size, radius=radius, position=position)
        return img.astype(np.float32)

    savepath = f"{savepath}/{embedding_option}/{int(wavelength*1000)}/x{int(x_voxel_size*1000)}-y{int(y_voxel_size*1000)}-z{int(z_voxel_size*1000)}"
    plt.rcParams.update({
        'font.size': 8,
        'axes.titlesize': 10,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
    })

    vmin, vmax, vcenter, step = 0, 2, 1, .1
    highcmap = plt.get_cmap('YlOrRd', 256)
    lowcmap = plt.get_cmap('YlGnBu_r', 256)
    low = np.linspace(0, 1 - step, int(abs(vcenter - vmin) / step))
    high = np.linspace(0, 1 + step, int(abs(vcenter - vmax) / step))
    cmap = np.vstack((lowcmap(low), [1, 1, 1, 1], highcmap(high)))
    cmap = mcolors.ListedColormap(cmap)

    waves = np.arange(-.3, .35, step=.05).round(3)
    # waves = np.arange(-.075, .08, step=.015).round(3) ## small
    logger.info(waves)

    gen = SyntheticPSF(
        psf_type=psf_type,
        amplitude_ranges=(-1, 1),
        n_modes=n_modes,
        lam_detection=wavelength,
        psf_shape=3*[res],
        x_voxel_size=x_voxel_size,
        y_voxel_size=y_voxel_size,
        z_voxel_size=z_voxel_size,
        cpu_workers=-1,
    )

    for mode in trange(3, n_modes):
        for radius in trange(shapes):
            if radius == 0:
                reference = np.zeros(gen.psf_shape)
                reference[gen.psf_shape[0]//2, gen.psf_shape[1]//2, gen.psf_shape[2]//2] = 1    # single voxel
            else:
                reference = sphere(image_size=gen.psf_shape, radius=radius, position=.5)     # sphere of voxels

            outdir = Path(f'{savepath}/i{res}_pad_{padsize}_lattice/mode_{mode}/r{radius}')
            outdir.mkdir(exist_ok=True, parents=True)
            imwrite(f"{outdir}/reference_{radius}.tif", reference, compression='deflate')


            fig, axes = plt.subplots(6, len(waves)+1, figsize=(12, 6))

            for i, amp in enumerate(waves):
                phi = np.zeros(n_modes)
                phi[mode] = amp

                psf, amps, lls_defocus_offset = gen.single_psf(
                    phi=phi,
                    normed=True,
                    meta=True,
                )

                wavefront = Wavefront(phi, lam_detection=gen.lam_detection)
                abr = round(wavefront.peak2valley() * np.sign(amp), 1)
                axes[0, i+1].set_title(f'{abr}$\\lambda$')

                # inputs = detected signal, given by convolving reference (puncta) with kernel (aberrated psf)
                inputs = convolution.convolve_fft(reference, psf, allow_huge=True)
                inputs /= np.nanmax(inputs)

                outdir = Path(f'{savepath}/i{res}_pad_{padsize}_lattice/mode_{mode}/r{radius}/convolved/')
                outdir.mkdir(exist_ok=True, parents=True)
                imwrite(f"{outdir}/{str(abr).replace('.', 'p')}.tif", inputs, compression='deflate')

                emb = fourier_embeddings(
                    inputs,
                    iotf=gen.iotf,
                    na_mask=gen.na_mask,
                    no_phase=False,
                    remove_interference=False,
                    embedding_option=embedding_option,
                    plot=f"{outdir}/{str(abr).replace('.', 'p')}",
                )
                outdir = Path(f'{savepath}/i{res}_pad_{padsize}_lattice/mode_{mode}/r{radius}/embeddings/')
                outdir.mkdir(exist_ok=True, parents=True)
                imwrite(f"{outdir}/{str(abr).replace('.', 'p')}.tif", emb, compression='deflate')


                plt.figure(fig.number)
                for ax in range(6):
                    if amp == waves[-1]:
                        mat = axes[ax, 0].contourf(
                            Wavefront(phi, lam_detection=wavelength).wave(100),
                            levels=np.arange(-2, 2, step=.1),
                            cmap='Spectral_r',
                            extend='both'
                        )
                        axes[ax, 0].axis('off')
                        axes[ax, 0].set_aspect('equal')

                    m = axes[ax, i+1].imshow(
                        emb[ax, :, :],
                        cmap=cmap if ax < 3 else 'Spectral_r',
                        vmin=vmin if ax < 3 else -.5,
                        vmax=vmax if ax < 3 else .5,
                    )
                    axes[ax, i+1].set_aspect('equal')
                    axes[ax, i+1].axis('off')

                    cax = inset_axes(
                        axes[ax, -1],
                        width="10%",
                        height="100%",
                        loc='center right',
                        borderpad=-1
                    )
                    cb = plt.colorbar(m, cax=cax)
                    cax.yaxis.set_label_position("right")

            plt.subplots_adjust(top=0.95, right=0.95, wspace=.2)
            plt.savefig(f'{savepath}/i{res}_pad{padsize}_mode_{mode}_radius_{radius}.pdf', bbox_inches='tight', pad_inches=.25)


def plot_simulation(
        res=64,
        padsize=None,
        n_modes=55,
        wavelength=.605,
        x_voxel_size=.15,
        y_voxel_size=.15,
        z_voxel_size=.6,
        #savepath='../data/embeddings/seminar/x100-y100-z100',
        savepath='../data/embeddings/seminar/x150-y150-z600',
):

    waves = np.round([-.2, -.1, -.05, .05, .1, .2], 3)
    logger.info(waves)

    gen = SyntheticPSF(
        amplitude_ranges=(-1, 1),
        n_modes=n_modes,
        lam_detection=wavelength,
        psf_shape=3*[res],
        x_voxel_size=x_voxel_size,
        y_voxel_size=y_voxel_size,
        z_voxel_size=z_voxel_size,
        cpu_workers=-1,
    )

    outdir = Path(f'{savepath}/i{res}_pad_{padsize}/')
    outdir.mkdir(exist_ok=True, parents=True)

    imwrite(f"{outdir}/theoretical_psf.tif", gen.ipsf, compression='deflate')
    imwrite(f"{outdir}/theoretical_otf.tif", gen.iotf, compression='deflate')

    for mode in trange(5, n_modes):
        for amp in waves:
            phi = np.zeros(n_modes)
            phi[mode] = amp

            wavefront = Wavefront(phi, lam_detection=gen.lam_detection)
            abr = round(wavefront.peak2valley() * np.sign(amp), 1)

            embedding = gen.single_otf(
                phi=phi,
                normed=True,
                noise=True,
                na_mask=True,
                ratio=True,
                padsize=padsize,
            )

            emb = Path(f'{outdir}/mode_{mode}/embeddings')
            emb.mkdir(exist_ok=True, parents=True)
            imwrite(f"{emb}/{str(abr).replace('.', 'p')}.tif", embedding, compression='deflate')

            psf = gen.single_psf(
                phi=phi,
                normed=True,
                meta=False,
            )

            reals = Path(f'{outdir}/mode_{mode}/psfs')
            reals.mkdir(exist_ok=True, parents=True)
            imwrite(f"{reals}/{str(abr).replace('.', 'p')}.tif", psf, compression='deflate')


def plot_signal(n_modes=55, wavelength=.605):
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })
    from preprocessing import resize_with_crop_or_pad

    waves = np.arange(0, .5, step=.05)
    res = [32, 64, 96, 128, 192, 256]
    logger.info(waves)

    gen = SyntheticPSF(
        amplitude_ranges=(-1, 1),
        n_modes=n_modes,
        lam_detection=wavelength,
        psf_shape=(256, 256, 256),
        x_voxel_size=.1,
        y_voxel_size=.1,
        z_voxel_size=.1,
        cpu_workers=-1,

    )

    signal = {}
    for i in range(3, n_modes):
        signal[i] = {}

        for j, a in enumerate(tqdm(waves, desc=f'Mode [#{i}]', file=sys.stdout)):
            phi = np.zeros(n_modes)
            phi[i] = a
            w = Wavefront(phi, order='ansi', lam_detection=wavelength)

            abr = 0 if j == 0 else round(w.peak2valley())
            signal[i][abr] = {}

            psf = gen.single_psf(w, normed=True)

            # psf_cmap = 'hot'
            # fig, axes = plt.subplots(len(res), 4)

            for k, r in enumerate(res):
                window = resize_with_crop_or_pad(psf, crop_shape=tuple(3*[r]))
                signal[i][abr][r] = np.sum(window)

                # vol = window ** .5
                # vol = np.nan_to_num(vol)
                #
                # axes[k, 0].bar(range(n_modes), height=w.amplitudes)
                # m = axes[k, 1].imshow(np.max(vol, axis=0), cmap=psf_cmap, vmin=0, vmax=1)
                # axes[k, 2].imshow(np.max(vol, axis=1), cmap=psf_cmap, vmin=0, vmax=1)
                # axes[k, 3].imshow(np.max(vol, axis=2), cmap=psf_cmap, vmin=0, vmax=1)

        # plt.tight_layout()
        # plt.show()

        df = pd.DataFrame.from_dict(signal[i], orient="index")
        logger.info(df)

        total_energy = df[res[-1]].values
        df = df.apply(lambda e: e/total_energy, axis=0)
        logger.info(df)

        theoretical = df.iloc[[0]].values[0]
        rdf = df.apply(lambda row: abs(theoretical-row) / theoretical, axis=1)
        logger.info(rdf)

        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(2, 3)
        ax = fig.add_subplot(gs[0, :2])
        axw = fig.add_subplot(gs[0, 2])
        axr = fig.add_subplot(gs[1, :])

        for r in res:
            ax.plot(df[r], label=r)
            ax.set_xlim((0, None))
            ax.set_ylim((0, 1))
            ax.set_ylabel('Signal')
            ax.grid(True, which="both", axis='both', lw=.5, ls='--', zorder=0)

            axr.plot(rdf[r], label=r)
            axr.set_xlim((0, None))
            axr.set_ylim((0, 1))
            axr.set_xlabel(
                'peak-to-valley aberration'
                rf'($\lambda = {int(wavelength*1000)}~nm$)'
            )
            axr.set_ylabel('Percentage signal lost')
            axr.grid(True, which="both", axis='both', lw=.5, ls='--', zorder=0)
            axr.legend(frameon=False, loc='upper center', ncol=6)

            phi = np.zeros(n_modes)
            phi[i] = .5
            phi = Wavefront(phi, order='ansi', lam_detection=wavelength).wave(size=100)

            mat = axw.contourf(
                phi,
                cmap='Spectral_r',
                extend='both'
            )
            divider = make_axes_locatable(axw)
            top = divider.append_axes("top", size='30%', pad=0.2)
            top.hist(phi.flatten(), bins=phi.shape[0], color='grey')
            top.set_yticks([])
            top.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            top.spines['right'].set_visible(False)
            top.spines['top'].set_visible(False)
            top.spines['left'].set_visible(False)
            axw.axis('off')

            plt.tight_layout()
            plt.savefig(f'../data/signal_res_mode_{i}.png', dpi=300, bbox_inches='tight', pad_inches=.25)

    signal = pd.DataFrame.from_dict(signal, orient="index").stack().to_frame()
    signal.index.names = ['index', 'waves']
    signal = pd.concat([signal.drop([0], axis=1), signal[0].apply(pd.Series)], axis=1).reset_index()
    logger.info(signal)
    signal.to_csv('../data/signal.csv')


def plot_dmodes(
    psf: np.array,
    gen: SyntheticPSF,
    y: Wavefront,
    pred: Wavefront,
    save_path: Path,
    wavelength: float = .605,
    psf_cmap: str = 'hot',
    gamma: float = .5,
    threshold: float = .01,
):
    def wavefront(iax, phi, levels, label=''):
        mat = iax.contourf(
            phi,
            levels=levels,
            cmap=wave_cmap,
            vmin=np.min(levels),
            vmax=np.max(levels),
            extend='both'
        )
        iax.axis('off')
        iax.set_title(label)

        cax = inset_axes(iax, width="10%", height="100%", loc='center right', borderpad=-3)
        cbar = fig.colorbar(
            mat,
            cax=cax,
            fraction=0.046,
            pad=0.04,
            extend='both',
            format=FormatStrFormatter("%.2g"),
        )
        cbar.ax.set_title(r'$\lambda$')
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.yaxis.set_label_position('left')
        return mat

    def psf_slice(xy, zx, zy, vol, label=''):
        vol = vol ** gamma
        vol = np.nan_to_num(vol)

        if vol.shape[0] == 3:
            m = xy.imshow(vol[0], cmap='Spectral_r', vmin=0, vmax=1)
            zx.imshow(vol[1], cmap='Spectral_r', vmin=0, vmax=1)
            zy.imshow(vol[2], cmap='Spectral_r', vmin=0, vmax=1)
        else:
            m = xy.imshow(np.max(vol, axis=0), cmap=psf_cmap, vmin=0, vmax=1)
            zx.imshow(np.max(vol, axis=1), cmap=psf_cmap, vmin=0, vmax=1)
            zy.imshow(np.max(vol, axis=2), cmap=psf_cmap, vmin=0, vmax=1)

        cax = inset_axes(zy, width="10%", height="100%", loc='center right', borderpad=-3)
        cb = plt.colorbar(m, cax=cax)
        cax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        cax.yaxis.set_label_position("right")

        xy.set_ylabel(label)
        return m

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.autolimit_mode': 'round_numbers'
    })
    # plt.style.use("dark_background")

    if len(psf.shape) > 3:
        psf = np.squeeze(psf, axis=-1)
        psf = np.squeeze(psf, axis=0)

    y_wave = y.wave(size=100)
    step = .25
    vmax = round(np.max([
        np.abs(round(np.nanquantile(y_wave, .1), 2)),
        np.abs(round(np.nanquantile(y_wave, .9), 2))
    ]) * 4) / 4
    vmax = .25 if vmax < threshold else vmax

    highcmap = plt.get_cmap('magma_r', 256)
    middlemap = plt.get_cmap('gist_gray', 256)
    lowcmap = plt.get_cmap('gist_earth_r', 256)

    ll = np.arange(-vmax, -.25 + step, step)
    mm = [-.15, 0, .15]
    hh = np.arange(.25, vmax + step, step)
    mticks = np.concatenate((ll, mm, hh))

    levels = np.vstack((
        lowcmap(.66 * ll / ll.min()),
        middlemap([.85, .95, 1, .95, .85]),
        highcmap(.66 * hh / hh.max())
    ))
    wave_cmap = mcolors.ListedColormap(levels)

    fig = plt.figure(figsize=(15, 200))
    gs = fig.add_gridspec(64, 4)

    p_psf = gen.single_psf(pred)
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xz = fig.add_subplot(gs[0, 1])
    ax_yz = fig.add_subplot(gs[0, 2])
    ax_w = fig.add_subplot(gs[0, 3])
    psf_slice(ax_xy, ax_xz, ax_yz, p_psf, label='Prediction')
    wavefront(ax_w, pred.wave(size=100), label='Prediction', levels=mticks)

    ax_xy = fig.add_subplot(gs[1, 0])
    ax_xz = fig.add_subplot(gs[1, 1])
    ax_yz = fig.add_subplot(gs[1, 2])
    ax_w = fig.add_subplot(gs[1, 3])
    psf_slice(ax_xy, ax_xz, ax_yz, psf, label='PSF (MIP)')
    wavefront(ax_w, y_wave, label='Ground truth', levels=mticks)

    otf = fourier_embeddings(psf, iotf=gen.iotf, na_mask=gen.na_mask)
    ax_xy = fig.add_subplot(gs[2, 0])
    ax_xz = fig.add_subplot(gs[2, 1])
    ax_yz = fig.add_subplot(gs[2, 2])
    ax_w = fig.add_subplot(gs[2, 3])
    psf_slice(ax_xy, ax_xz, ax_yz, otf, label='R_rel')
    wavefront(ax_w, y_wave, label='Ground truth', levels=mticks)

    k = 0
    for i, w in enumerate(y.amplitudes):
        k += 1
        phi = np.zeros(55)
        phi[i] = w / (2 * np.pi / wavelength)
        phi = Wavefront(phi, order='ansi')

        psf = gen.single_psf(phi)
        otf = fourier_embeddings(psf, iotf=gen.iotf, na_mask=gen.na_mask,)
        ax_xy = fig.add_subplot(gs[2+k, 0])
        ax_xz = fig.add_subplot(gs[2+k, 1])
        ax_yz = fig.add_subplot(gs[2+k, 2])
        ax_w = fig.add_subplot(gs[2+k, 3])
        psf_slice(ax_xy, ax_xz, ax_yz, otf, label=f'Mode #{i}')
        wavefront(ax_w, phi.wave(100), label=f'Mode #{i}', levels=mticks)

    ax_zcoff = fig.add_subplot(gs[-1, :])
    ax_zcoff.plot(pred.amplitudes, '-o', color='C0', label='Predictions')
    ax_zcoff.plot(y.amplitudes, '-o', color='C1', label='Ground truth')
    ax_zcoff.legend(frameon=False, loc='upper center', bbox_to_anchor=(.1, 1))
    ax_zcoff.set_xticks(range(len(pred.amplitudes)))
    ax_zcoff.set_ylabel(r'Zernike coefficients ($\mu$m)')
    ax_zcoff.spines['top'].set_visible(False)
    ax_zcoff.set_xlim((0, len(pred.amplitudes)))
    ax_zcoff.grid(True, which="both", axis='both', lw=1, ls='--', zorder=0)

    plt.subplots_adjust(top=0.95, right=0.95, wspace=.2)
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight', pad_inches=.25)
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight', pad_inches=.25)


def plot_inputs(
    n_modes=15,
    x_voxel_size=.15,
    y_voxel_size=.15,
    z_voxel_size=.6,
    wavelength: float = .605,
    psf_cmap: str = 'Spectral_r',
    threshold: float = .01,
):
    def wavefront(iax, phi, levels, label=''):
        mat = iax.contourf(
            phi,
            levels=levels,
            cmap=wave_cmap,
            vmin=np.min(levels),
            vmax=np.max(levels),
            extend='both'
        )
        iax.set_aspect('equal')

        divider = make_axes_locatable(iax)
        top = divider.append_axes("top", size='30%', pad=0.2)
        top.hist(phi.flatten(), bins=phi.shape[0], color='grey')

        top.set_title(label)
        top.set_yticks([])
        top.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        top.spines['right'].set_visible(False)
        top.spines['top'].set_visible(False)
        top.spines['left'].set_visible(False)
        return mat

    def slice(xy, zx, zy, vol, label='', maxproj=True):

        if vol.shape[-1] == 3:
            m = xy.imshow(vol[:, :, 0], cmap=psf_cmap, vmin=0, vmax=1)
            zx.imshow(vol[:, :, 1], cmap=psf_cmap, vmin=0, vmax=1)
            zy.imshow(vol[:, :, 2], cmap=psf_cmap, vmin=0, vmax=1)

            if maxproj:
                m = xy.imshow(np.max(vol, axis=0), cmap=psf_cmap, vmin=0, vmax=1)
                zx.imshow(np.max(vol, axis=1), cmap=psf_cmap, vmin=0, vmax=1)
                zy.imshow(np.max(vol, axis=2), cmap=psf_cmap, vmin=0, vmax=1)
            else:
                mid_plane = vol.shape[0] // 2
                m = xy.imshow(vol[mid_plane, :, :], cmap=psf_cmap, vmin=0, vmax=1)
                zx.imshow(vol[:, mid_plane, :], cmap=psf_cmap, vmin=0, vmax=1)
                zy.imshow(vol[:, :, mid_plane], cmap=psf_cmap, vmin=0, vmax=1)

        cax = inset_axes(zy, width="10%", height="100%", loc='center right', borderpad=-3)
        cb = plt.colorbar(m, cax=cax)
        cax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        cax.yaxis.set_label_position("right")
        xy.set_ylabel(label)
        return m

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.autolimit_mode': 'round_numbers'
    })
    #plt.style.use("dark_background")

    for i in trange(5, n_modes):
        phi = np.zeros(n_modes)
        phi[i] = .05
        w = Wavefront(phi, order='ansi', lam_detection=wavelength)
        y_wave = w.wave(size=100)

        gen = SyntheticPSF(
            amplitude_ranges=(-1, 1),
            n_modes=n_modes,
            lam_detection=wavelength,
            psf_shape=3*[32],
            x_voxel_size=x_voxel_size,
            y_voxel_size=y_voxel_size,
            z_voxel_size=z_voxel_size,
            cpu_workers=-1,
        )

        inputs = gen.single_otf(w, normed=True)
        psf = gen.single_psf(w, normed=True)

        otf = np.fft.fftn(psf)
        otf = np.fft.fftshift(otf)
        otf = np.abs(otf)
        otf = np.log10(otf)
        otf /= np.max(otf)

        fig = plt.figure(figsize=(8, 8))
        gs = fig.add_gridspec(4, 5)

        ax_xy = fig.add_subplot(gs[0, 0])
        ax_xz = fig.add_subplot(gs[0, 1])
        ax_yz = fig.add_subplot(gs[0, 2])
        ax_wave = fig.add_subplot(gs[0, -1])
        cax = fig.add_axes([.995, 0.75, 0.02, .175])

        ax_pxy = fig.add_subplot(gs[1, 0])
        ax_pxz = fig.add_subplot(gs[1, 1])
        ax_pyz = fig.add_subplot(gs[1, 2])
        ax_pcuts = fig.add_subplot(gs[1, -1])

        ax_cxy = fig.add_subplot(gs[2, 0])
        ax_cxz = fig.add_subplot(gs[2, 1])
        ax_cyz = fig.add_subplot(gs[2, 2])
        ax_ccuts = fig.add_subplot(gs[2, -1])

        ax_zcoff = fig.add_subplot(gs[-1, :])

        step = .25
        vmax = round(np.max([
            np.abs(round(np.nanquantile(y_wave, .1), 2)),
            np.abs(round(np.nanquantile(y_wave, .9), 2))
        ]) * 4) / 4
        vmax = .25 if vmax < threshold else vmax

        highcmap = plt.get_cmap('magma_r', 256)
        middlemap = plt.get_cmap('gist_gray', 256)
        lowcmap = plt.get_cmap('gist_earth_r', 256)

        ll = np.arange(-vmax, -.25 + step, step)
        mm = [-.15, 0, .15]
        hh = np.arange(.25, vmax + step, step)
        mticks = np.concatenate((ll, mm, hh))

        levels = np.vstack((
            lowcmap(.66 * ll / ll.min()),
            middlemap([.85, .95, 1, .95, .85]),
            highcmap(.66 * hh / hh.max())
        ))
        wave_cmap = mcolors.ListedColormap(levels)

        mat = wavefront(ax_wave, y_wave, label='Ground truth', levels=mticks)
        ax_wave.axis('off')

        cbar = fig.colorbar(
            mat,
            cax=cax,
            fraction=0.046,
            pad=0.04,
            extend='both',
            format=FormatStrFormatter("%.2g"),
            # spacing='proportional',
        )
        cbar.ax.set_title(r'$\lambda$')
        cbar.ax.set_ylabel(f'$\lambda = {wavelength}~\mu m$')
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.yaxis.set_label_position('left')

        ax_xy.set_title('XY')
        ax_xz.set_title('ZX')
        ax_yz.set_title('ZY')

        slice(ax_xy, ax_xz, ax_yz, inputs, label='Input', maxproj=False)
        slice(ax_pxy, ax_pxz, ax_pyz, psf, label='PSF (MIP)')
        slice(ax_cxy, ax_cxz, ax_cyz, otf, label=r'OTF ($log_{10}$)', maxproj=False)

        ax_pcuts.semilogy(psf[:, psf.shape[0]//2, psf.shape[0]//2], '-', label='XY')
        ax_pcuts.semilogy(psf[psf.shape[0]//2, :, psf.shape[0]//2], '--', label='XZ')
        ax_pcuts.semilogy(psf[psf.shape[0]//2, psf.shape[0]//2, :], ':', label='YZ')
        ax_pcuts.grid(True, which="both", axis='both', lw=.5, ls='--', zorder=0)
        ax_pcuts.legend(frameon=False, ncol=1, bbox_to_anchor=(1.0, 1.0), loc='upper left')
        ax_pcuts.set_aspect('equal')

        ax_ccuts.semilogy(otf[:, otf.shape[0]//2, otf.shape[0]//2], '-', label='XY')
        ax_ccuts.semilogy(otf[otf.shape[0]//2, :, otf.shape[0]//2], '--', label='XZ')
        ax_ccuts.semilogy(otf[otf.shape[0]//2, otf.shape[0]//2, :], ':', label='YZ')
        ax_ccuts.grid(True, which="both", axis='both', lw=.5, ls='--', zorder=0)
        ax_ccuts.legend(frameon=False, ncol=1, bbox_to_anchor=(1.0, 1.0), loc='upper left')
        ax_ccuts.set_aspect('equal')

        # ax_zcoff.set_title('Zernike modes')
        ax_zcoff.plot(w.amplitudes, '-o', color='C0', label='Predictions')
        ax_zcoff.set_xticks(range(len(w.amplitudes)))
        ax_zcoff.set_ylabel(r'Zernike coefficients ($\mu$m)')
        ax_zcoff.spines['top'].set_visible(False)
        ax_zcoff.grid()

        plt.subplots_adjust(top=0.95, right=0.95, wspace=.3)
        plt.savefig(f'../data/inputs/{i}.png', dpi=300, bbox_inches='tight', pad_inches=.25)
