import matplotlib
matplotlib.use('Agg')

import warnings
from pathlib import Path
import logging
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
from numpy.lib.stride_tricks import sliding_window_view
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec
import raster_geometry as rg
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from typing import Any
from tifffile import imsave
import numpy as np
import seaborn as sns
from tqdm import tqdm, trange
import matplotlib.patches as patches
from astropy import convolution
import tensorflow as tf
from tensorflow_addons.image import gaussian_filter2d

from wavefront import Wavefront
from zernike import Zernike
from synthetic import SyntheticPSF


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


def plot_training_dist(n_samples=10, batch_size=10, wavelength=.510):
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })
    from utils import peak2peak
    from utils import microns2waves

    psfargs = dict(
        n_modes=60,
        dtype='../lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat',
        distribution='powerlaw',
        bimodal=True,
        gamma=.75,
        lam_detection=wavelength,
        amplitude_ranges=(0, 1),
        psf_shape=(32, 32, 32),
        x_voxel_size=.108,
        y_voxel_size=.108,
        z_voxel_size=.2,
        batch_size=batch_size,
        snr=30,
        max_jitter=1,
        cpu_workers=-1,
    )

    n_batches = n_samples // batch_size
    peaks = []
    zernikes = pd.DataFrame([], columns=range(1, psfargs['n_modes'] + 1))

    ## Challenging dataset
    difractionlimit = np.arange(0, 0.055, .005).round(3)  # 10 bins
    small = np.arange(.05, .1, .0025).round(3)            # 20 bins
    large = np.arange(.1, .2, .005).round(3)              # 20 bins
    extreme = np.arange(.2, .52, .02).round(3)            # 20 bins
    min_amps = np.concatenate([difractionlimit, small, large, extreme[:-1]])
    max_amps = np.concatenate([difractionlimit[1:], small, large, extreme])

    ## Easy dataset
    # min_amps = np.arange(0, .15, .01).round(3)
    # max_amps = np.arange(.01, .16, .01).round(3)

    for mina, maxa in zip(min_amps, max_amps):
        psfargs['amplitude_ranges'] = (mina, maxa)
        for _, (psfs, ys) in zip(range(n_batches), SyntheticPSF(**psfargs).generator()):
            zernikes = zernikes.append(
                pd.DataFrame(microns2waves(ys, wavelength=wavelength), columns=range(1, psfargs['n_modes'] + 1)),
                ignore_index=True
            )
            ps = list(peak2peak(ys))
            logger.info(f'Range[{mina}, {maxa}]$\mu$m \t $\\bar{{p2p}}$={np.mean(ps).round(3)}$\\lambda$')
            peaks.extend(ps)

    logger.info(zernikes.round(2))
    logger.info(f'Range[{min_amps[0]}, {max_amps[-1]}]$\mu$m \t $\\bar{{p2p}}$={np.mean(peaks).round(3)}$\\lambda$')

    fig, (pax, cax, zax) = plt.subplots(1, 3, figsize=(16, 4))

    sns.histplot(peaks, kde=True, ax=pax, color='dimgrey')

    pax.set_xlabel(
        'Peak-to-peak aberration $|P_{95} - P_{5}|$\n'
        rf'($\lambda = {int(wavelength*1000)}~nm$)'
    )
    pax.set_ylabel(rf'Samples')
    # pax.set_xlim(0, 10)

    # bars = sns.barplot(zernikes.columns.values, zernikes.mean(axis=0), ax=zax)
    # for index, label in enumerate(bars.get_xticklabels()):
    #     if index % int(.1 * zernikes.shape[1]) == 0:
    #         label.set_visible(True)
    #     else:
    #         label.set_visible(False)
    # zax.set_xlabel('Average amplitude per zernike mode')

    zernikes = np.abs(zernikes)
    dlimit = wavelength/2
    dmodes = (zernikes[zernikes > dlimit]).count(axis=1)
    hist, bins = np.histogram(dmodes, bins=zernikes.columns.values)
    hist = hist / hist.sum()
    idx = (hist > 0).nonzero()

    if len(idx[0]) != 0:
        bars = sns.barplot(bins[idx], hist[idx], ax=cax)
        for index, label in enumerate(bars.get_xticklabels()):
            if index % 2 == 0:
                label.set_visible(True)
            else:
                label.set_visible(False)

    cax.set_xlabel(
        f'Number of modes above diffraction limit\n'
        r'$\alpha_i > \dfrac{\lambda}{2NA}$'
    )


    # pct = zernikes.astype(float).quantile(.9999, axis=1)
    # dmodes = (zernikes[zernikes > pct]).count(axis=1)
    zernikes = zernikes.div(zernikes.sum(axis=1), axis=0)
    logger.info(zernikes.round(2))

    dmodes = (zernikes[zernikes > .05]).count(axis=1)
    hist, bins = np.histogram(dmodes, bins=zernikes.columns.values)
    idx = (hist > 0).nonzero()
    hist = hist / hist.sum()

    if len(idx[0]) != 0:
        bars = sns.barplot(bins[idx], hist[idx], ax=zax, palette='Accent')
        for index, label in enumerate(bars.get_xticklabels()):
            if index % 2 == 0:
                label.set_visible(True)
            else:
                label.set_visible(False)

    zax.set_xlabel(
        f'Number of highly influential modes\n'
        r'$\alpha_i / \sum_{{k=1}}^{{60}}{{\alpha_{{k}}}} > 5\%$'
    )

    pax.grid(True, which="both", axis='both', lw=.5, ls='--', zorder=0)
    cax.grid(True, which="both", axis='both', lw=.5, ls='--', zorder=0)
    zax.grid(True, which="both", axis='both', lw=.5, ls='--', zorder=0)

    name = f'{psfargs["distribution"]}_{psfargs["n_modes"]}modes_gamma_{str(psfargs["gamma"]).replace(".", "p")}'
    plt.savefig(
        f'../data/{name}.png',
        dpi=300, bbox_inches='tight', pad_inches=.25
    )


def plot_fov(n_modes=60, wavelength=.605, psf_cmap='hot', x_voxel_size=.15, y_voxel_size=.15, z_voxel_size=.6):
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })
    from utils import peak_aberration

    waves = np.round(np.arange(0, .5, step=.1), 2)
    res = [128, 64, 32]
    offsets = [0, 32, 48]
    savedir = '../data/fov/ratio_150x-150y-600z'
    logger.info(waves)

    for i in range(3, n_modes):
        fig = plt.figure(figsize=(35, 60))
        gs = fig.add_gridspec(len(waves)*len(res), 8)

        grid = {}
        for a, j in zip(waves, np.arange(0, len(waves)*len(res), step=3)):
            for k, r in enumerate(res):
                for c in range(8):
                    grid[(a, r, c)] = fig.add_subplot(gs[j+k, c])

        # from pprint import pprint
        # pprint(grid)

        for j, amp in enumerate(tqdm(waves, desc=f'Mode [#{i}]')):
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
                    snr=100,
                    max_jitter=0,
                    cpu_workers=-1,
                )
                window = gen.single_psf(w, zplanes=0, normed=True, noise=False)
                #window = center_crop(psf, crop_shape=tuple(3 * [r]))

                fft = np.fft.fftn(window)
                fft = np.fft.fftshift(fft)
                fft = np.abs(fft)
                # fft[fft == np.inf] = np.nan
                # fft[fft == -np.inf] = np.nan
                # fft[fft == np.nan] = np.min(fft)
                # fft = np.log10(fft)
                fft /= np.max(fft)

                perfect_psf = gen.single_psf(phi=Wavefront(np.zeros(n_modes)), zplanes=0)
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

                grid[(amp, r, 7)].set_title(f'{round(peak_aberration(phi))} waves')
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
        n_modes=15,
        wavelength=.510,
        x_voxel_size=.108,
        y_voxel_size=.108,
        z_voxel_size=.268,
        log10=False,
        psf_type='../lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat',
        savepath='../data/embeddings',
):
    savepath = f"{savepath}/{int(wavelength*1000)}/x{int(x_voxel_size*1000)}-y{int(y_voxel_size*1000)}-z{int(z_voxel_size*1000)}"
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })
    from utils import peak_aberration

    if log10:
        vmin, vmax, vcenter, step = -2, 2, 0, .1
    else:
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

    fig = plt.figure(figsize=(25, 60))
    nrows = (n_modes-5) * 6
    gs = fig.add_gridspec(nrows, len(waves)+1)

    grid = {}
    for mode, ax in zip(range(5, n_modes), np.round(np.arange(0, nrows, step=6))):
        for k in range(6):
            grid[(mode, k, 'wavefront')] = fig.add_subplot(gs[ax + k, 0])

            for j, w in enumerate(waves):
                grid[(mode, k, w)] = fig.add_subplot(gs[ax+k, j+1])

    gen = SyntheticPSF(
        dtype=psf_type,
        amplitude_ranges=(-1, 1),
        n_modes=n_modes,
        lam_detection=wavelength,
        psf_shape=3*[res],
        x_voxel_size=x_voxel_size,
        y_voxel_size=y_voxel_size,
        z_voxel_size=z_voxel_size,
        snr=20,
        max_jitter=0,
        cpu_workers=-1,
    )

    for mode in trange(5, n_modes):
        for amp in waves:
            phi = np.zeros(n_modes)
            phi[mode] = amp

            window, amps, snr, zplanes, maxcounts = gen.single_otf(
                phi=phi,
                zplanes=0,
                normed=True,
                noise=True,
                augmentation=True,
                meta=True,
                na_mask=True,
                ratio=True,
                padsize=padsize,
                log10=log10
            )

            abr = round(peak_aberration(phi) * np.sign(amp), 1)
            grid[(mode, 0, amp)].set_title(f'{abr}$\\lambda$')

            outdir = Path(f'{savepath}/i{res}_pad_{padsize}_lattice/mode_{mode}/ratios/')
            outdir.mkdir(exist_ok=True, parents=True)
            imsave(f"{outdir}/{str(abr).replace('.', 'p')}.tif", window)

            for ax in range(6):
                if amp == waves[-1]:
                    mat = grid[(mode, ax, 'wavefront')].contourf(
                        Wavefront(phi, lam_detection=wavelength).wave(100),
                        levels=np.arange(-10, 10, step=1),
                        cmap='Spectral_r',
                        extend='both'
                    )
                    grid[(mode, ax, 'wavefront')].axis('off')
                    grid[(mode, ax, 'wavefront')].set_aspect('equal')

                if window.shape[0] == 6:
                    vol = window[ax, :, :]
                else:
                    vol = np.max(window, axis=ax)

                m = grid[(mode, ax, amp)].imshow(
                    vol,
                    cmap=cmap if ax < 3 else 'Spectral_r',
                    vmin=vmin if ax < 3 else -1,
                    vmax=vmax if ax < 3 else 1,
                )
                grid[(mode, ax, amp)].set_aspect('equal')
                grid[(mode, ax, amp)].axis('off')

                cax = inset_axes(
                    grid[(mode, ax, waves[-1])],
                    width="10%",
                    height="100%",
                    loc='center right',
                    borderpad=-3
                )
                cb = plt.colorbar(m, cax=cax)
                cax.yaxis.set_label_position("right")

    plt.subplots_adjust(top=0.95, right=0.95, wspace=.2)
    plt.savefig(f'{savepath}/i{res}_pad{padsize}_lattice.pdf', bbox_inches='tight', pad_inches=.25)


def plot_shapes_embeddings(
        res=64,
        padsize=None,
        shapes=5,
        wavelength=.510,
        x_voxel_size=.108,
        y_voxel_size=.108,
        z_voxel_size=.2,
        log10=False,
        psf_type='../lattice/YuMB_NAlattice0.35_NAAnnulusMax0.40_NAsigma0.1.mat',
        savepath='../data/shapes_embeddings',
):
    def sphere(image_size, radius=.5, position=.5):
        img = rg.sphere(shape=image_size, radius=radius, position=position)
        return img.astype(np.float)

    savepath = f"{savepath}/{int(wavelength*1000)}/x{int(x_voxel_size*1000)}-y{int(y_voxel_size*1000)}-z{int(z_voxel_size*1000)}"
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })
    from utils import peak_aberration

    if log10:
        vmin, vmax, vcenter, step = -2, 2, 0, .1
    else:
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

    fig = plt.figure(figsize=(25, 60))
    nrows = shapes * 6
    gs = fig.add_gridspec(nrows, len(waves)+1)

    grid = {}
    for th, ax in zip(range(shapes), np.round(np.arange(0, nrows, step=6))):
        for k in range(6):
            grid[(th, k, 'wavefront')] = fig.add_subplot(gs[ax + k, 0])

            for j, w in enumerate(waves):
                grid[(th, k, w)] = fig.add_subplot(gs[ax+k, j+1])

    gen = SyntheticPSF(
        dtype=psf_type,
        amplitude_ranges=(-1, 1),
        n_modes=60,
        lam_detection=wavelength,
        psf_shape=3*[res],
        x_voxel_size=x_voxel_size,
        y_voxel_size=y_voxel_size,
        z_voxel_size=z_voxel_size,
        snr=100,
        max_jitter=0,
        cpu_workers=-1,
    )
    mode = 6

    for thickness in trange(shapes):
        if thickness == 0:
            reference = np.zeros(gen.psf_shape)
            reference[gen.psf_shape[0]//2, gen.psf_shape[1]//2, gen.psf_shape[2]//2] = 1
        else:
            reference = sphere(image_size=gen.psf_shape, radius=thickness, position=.5)

        outdir = Path(f'{savepath}/i{res}_pad_{padsize}_lattice/mode_{mode}/{thickness}')
        outdir.mkdir(exist_ok=True, parents=True)
        imsave(f"{outdir}/reference_{thickness}.tif", reference)

        for amp in waves:
            phi = np.zeros(60)
            phi[mode] = amp

            abr = round(peak_aberration(phi) * np.sign(amp), 1)
            grid[(thickness, 0, amp)].set_title(f'{abr}$\\lambda$')

            kernel = gen.single_psf(
                phi=phi,
                zplanes=0,
                normed=True,
                noise=False,
                augmentation=False,
            )
            inputs = convolution.convolve_fft(reference, kernel, allow_huge=True)
            inputs /= np.nanmax(inputs)

            outdir = Path(f'{savepath}/i{res}_pad_{padsize}_lattice/mode_{mode}/{thickness}/convolved/')
            outdir.mkdir(exist_ok=True, parents=True)
            imsave(f"{outdir}/{str(abr).replace('.', 'p')}.tif", inputs)

            emb = gen.embedding(psf=inputs, principle_planes=True)

            outdir = Path(f'{savepath}/i{res}_pad_{padsize}_lattice/mode_{mode}/{thickness}/ratios/')
            outdir.mkdir(exist_ok=True, parents=True)
            imsave(f"{outdir}/{str(abr).replace('.', 'p')}.tif", emb)

            for ax in range(6):
                if amp == waves[-1]:
                    mat = grid[(thickness, ax, 'wavefront')].contourf(
                        Wavefront(phi, lam_detection=wavelength).wave(100),
                        levels=np.arange(-10, 10, step=1),
                        cmap='Spectral_r',
                        extend='both'
                    )
                    grid[(thickness, ax, 'wavefront')].axis('off')
                    grid[(thickness, ax, 'wavefront')].set_aspect('equal')

                if emb.shape[0] == 6:
                    vol = emb[ax, :, :]
                else:
                    vol = np.max(emb, axis=ax)

                m = grid[(thickness, ax, amp)].imshow(
                    vol,
                    cmap=cmap if ax < 3 else 'Spectral_r',
                    vmin=vmin if ax < 3 else -1,
                    vmax=vmax if ax < 3 else 1,
                )
                grid[(thickness, ax, amp)].set_aspect('equal')
                grid[(thickness, ax, amp)].axis('off')

                cax = inset_axes(
                    grid[(thickness, ax, waves[-1])],
                    width="10%",
                    height="100%",
                    loc='center right',
                    borderpad=-3
                )
                cb = plt.colorbar(m, cax=cax)
                cax.yaxis.set_label_position("right")

    plt.subplots_adjust(top=0.95, right=0.95, wspace=.2)
    plt.savefig(f'{savepath}/i{res}_pad{padsize}_lattice.pdf', bbox_inches='tight', pad_inches=.25)


def plot_gaussian_filters(
        res=64,
        padsize=None,
        n_modes=15,
        wavelength=.605,
        x_voxel_size=.15,
        y_voxel_size=.15,
        z_voxel_size=.6,
        sigma=1.66,
        kernel=5,
        savepath='../data/gaussian_filters',
):
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })
    from utils import peak_aberration

    vmin, vmax, vcenter, step = 0, 2, 1, .1
    highcmap = plt.get_cmap('YlOrRd', 256)
    lowcmap = plt.get_cmap('YlGnBu_r', 256)
    low = np.linspace(0, 1 - step, int(abs(vcenter - vmin) / step))
    high = np.linspace(0, 1 + step, int(abs(vcenter - vmax) / step))
    cmap = np.vstack((lowcmap(low), [1, 1, 1, 1], highcmap(high)))
    cmap = mcolors.ListedColormap(cmap)

    phase_vmin, phase_vmax, phase_vcenter, step = -1, 1, 0, .1
    low = np.linspace(0, 1 - step, int(abs(phase_vcenter - phase_vmin) / step))
    high = np.linspace(0, 1 + step, int(abs(phase_vcenter - phase_vmax) / step))
    phase_cmap = np.vstack((lowcmap(low), [1, 1, 1, 1], highcmap(high)))
    phase_cmap = mcolors.ListedColormap(phase_cmap)

    waves = np.arange(-.08, .1, step=.02).round(3)
    logger.info(waves)

    fig = plt.figure(figsize=(25, 60))
    nrows = (n_modes-5) * 6
    gs = fig.add_gridspec(nrows, len(waves)+1)

    grid = {}
    for mode, ax in zip(range(5, n_modes), np.round(np.arange(0, nrows, step=6))):
        for k in range(6):
            grid[(mode, k, 'wavefront')] = fig.add_subplot(gs[ax + k, 0])

            for j, w in enumerate(waves):
                grid[(mode, k, w)] = fig.add_subplot(gs[ax+k, j+1])

    gen = SyntheticPSF(
        amplitude_ranges=(-1, 1),
        n_modes=n_modes,
        lam_detection=wavelength,
        psf_shape=3*[res],
        x_voxel_size=x_voxel_size,
        y_voxel_size=y_voxel_size,
        z_voxel_size=z_voxel_size,
        snr=20,
        max_jitter=0,
        cpu_workers=-1,
    )

    for mode in trange(5, n_modes):
        for amp in waves:
            phi = np.zeros(n_modes)
            phi[mode] = amp

            window, amps, snr, zplanes, maxcounts = gen.single_otf(
                phi=phi,
                zplanes=0,
                normed=True,
                noise=True,
                augmentation=True,
                meta=True,
                na_mask=True,
                ratio=True,
                padsize=padsize
            )

            abr = round(peak_aberration(phi) * np.sign(amp), 1)
            grid[(mode, 0, amp)].set_title(f'{abr}$\\lambda$')

            for ax in range(6):
                if amp == waves[-1]:
                    mat = grid[(mode, ax, 'wavefront')].contourf(
                        Wavefront(phi, lam_detection=wavelength).wave(100),
                        levels=np.arange(-10, 10, step=1),
                        cmap='Spectral_r',
                        extend='both'
                    )
                    grid[(mode, ax, 'wavefront')].axis('off')
                    grid[(mode, ax, 'wavefront')].set_aspect('equal')

                if window.shape[0] == 6:
                    vol = window[ax, :, :]
                else:
                    vol = np.max(window, axis=ax)

                if ax >= 3:
                    physical_devices = tf.config.list_physical_devices('GPU')
                    for gpu_instance in physical_devices:
                        tf.config.experimental.set_memory_growth(gpu_instance, True)

                    vol = gaussian_filter2d(
                        vol,
                        filter_shape=(kernel, kernel),
                        sigma=sigma,
                        padding='CONSTANT'
                    )

                m = grid[(mode, ax, amp)].imshow(
                    vol,
                    cmap=cmap if ax < 3 else phase_cmap,
                    vmin=vmin if ax < 3 else phase_vmin,
                    vmax=vmax if ax < 3 else phase_vmax,
                )
                grid[(mode, ax, amp)].set_aspect('equal')
                grid[(mode, ax, amp)].axis('off')

                cax = inset_axes(
                    grid[(mode, ax, waves[-1])],
                    width="10%",
                    height="100%",
                    loc='center right',
                    borderpad=-3
                )
                cb = plt.colorbar(m, cax=cax)
                cax.yaxis.set_label_position("right")

    plt.subplots_adjust(top=0.95, right=0.95, wspace=.2)
    plt.savefig(f'{savepath}/i{res}_pad{padsize}_s{round(sigma, 3)}_k{kernel}.pdf', bbox_inches='tight', pad_inches=.25)


def plot_simulation(
        res=64,
        padsize=None,
        n_modes=60,
        wavelength=.605,
        x_voxel_size=.15,
        y_voxel_size=.15,
        z_voxel_size=.6,
        #savepath='../data/embeddings/seminar/x100-y100-z100',
        savepath='../data/embeddings/seminar/x150-y150-z600',
):
    from utils import peak_aberration

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
        snr=100,
        max_jitter=0,
        cpu_workers=-1,
    )

    outdir = Path(f'{savepath}/i{res}_pad_{padsize}/')
    outdir.mkdir(exist_ok=True, parents=True)

    imsave(f"{outdir}/theoretical_psf.tif", gen.ipsf)
    imsave(f"{outdir}/theoretical_otf.tif", gen.iotf)

    for mode in trange(5, n_modes):
        for amp in waves:
            phi = np.zeros(n_modes)
            phi[mode] = amp

            abr = round(peak_aberration(phi) * np.sign(amp), 1)

            # otf = gen.single_otf(
            #     phi=phi,
            #     zplanes=0,
            #     normed=True,
            #     noise=True,
            #     augmentation=True,
            #     na_mask=True,
            #     ratio=False,
            #     padsize=padsize,
            # )
            #
            # amps = Path(f'{outdir}/mode_{mode}/amps')
            # amps.mkdir(exist_ok=True, parents=True)
            # imsave(f"{amps}/{str(abr).replace('.', 'p')}.tif", otf)

            embedding = gen.single_otf(
                phi=phi,
                zplanes=0,
                normed=True,
                noise=True,
                augmentation=True,
                na_mask=True,
                ratio=True,
                padsize=padsize,
            )

            emb = Path(f'{outdir}/mode_{mode}/embeddings')
            emb.mkdir(exist_ok=True, parents=True)
            imsave(f"{emb}/{str(abr).replace('.', 'p')}.tif", embedding)

            psf = gen.single_psf(
                phi=phi,
                zplanes=0,
                normed=True,
                noise=True,
                augmentation=True,
                meta=False,
            )

            reals = Path(f'{outdir}/mode_{mode}/psfs')
            reals.mkdir(exist_ok=True, parents=True)
            imsave(f"{reals}/{str(abr).replace('.', 'p')}.tif", psf)


def plot_signal(n_modes=60, wavelength=.605):
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })
    from preprocessing import resize_with_crop_or_pad
    from utils import peak_aberration

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
        snr=100,
        max_jitter=0,
        cpu_workers=-1,

    )

    signal = {}
    for i in range(3, n_modes):
        signal[i] = {}

        for j, a in enumerate(tqdm(waves, desc=f'Mode [#{i}]')):
            phi = np.zeros(n_modes)
            phi[i] = a
            w = Wavefront(phi, order='ansi', lam_detection=wavelength)

            abr = 0 if j == 0 else round(peak_aberration(phi))
            signal[i][abr] = {}

            psf = gen.single_psf(w, zplanes=0, normed=True, noise=False)

            # psf_cmap = 'hot'
            # fig, axes = plt.subplots(len(res), 4)

            for k, r in enumerate(res):
                window = resize_with_crop_or_pad(psf, crop_shape=tuple(3*[r]))
                signal[i][abr][r] = np.sum(window)

                # vol = window ** .5
                # vol = np.nan_to_num(vol)
                #
                # axes[k, 0].bar(range(n_modes), height=w.amplitudes_ansi_waves)
                # m = axes[k, 1].imshow(np.max(vol, axis=0), cmap=psf_cmap, vmin=0, vmax=1)
                # axes[k, 2].imshow(np.max(vol, axis=1), cmap=psf_cmap, vmin=0, vmax=1)
                # axes[k, 3].imshow(np.max(vol, axis=2).T, cmap=psf_cmap, vmin=0, vmax=1)

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
                'Peak-to-peak aberration $|P_{95} - P_{5}|$'
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


def plot_mode(savepath, df, mode_index, n_modes=60, wavelength=.605):
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    fig = plt.figure(figsize=(8, 4))
    gs = fig.add_gridspec(1, 3)
    ax = fig.add_subplot(gs[0, :2])
    axw = fig.add_subplot(gs[0, 2])

    ax.plot(df)
    ax.set_xlim((0, 12))
    ax.set_yscale('log')
    ax.set_ylim((10**-2, 10))
    ax.set_xlabel(
        'Peak-to-peak aberration $|P_{95} - P_{5}|$'
        rf'($\lambda = {int(wavelength * 1000)}~nm$)'
    )
    ax.set_ylabel('Peak-to-peak residuals')
    ax.grid(True, which="both", axis='both', lw=.5, ls='--', zorder=0)

    phi = np.zeros(n_modes)
    phi[mode_index] = .5
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
    plt.savefig(savepath, dpi=300, bbox_inches='tight', pad_inches=.25)


def plot_aberrations():
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    fig, axes = plt.subplots(4, 4, figsize=(10, 8))
    axes = axes.flatten()

    for i in range(15):
        ax = axes[i]
        idx = i
        w = Wavefront({idx: 1})
        ax.set_title(f"{Zernike(idx).ansi_to_nm(idx)}")
        mat = ax.imshow(w.wave(size=100), cmap='Spectral_r')
        ax.axis('off')

    plt.tight_layout()
    plt.colorbar(mat)
    plt.show()


def plot_psnr(psf_cmap='hot', gamma=.75):
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    def psf_slice(xy, zx, zy, vol):
        vol = vol ** gamma
        vol = np.nan_to_num(vol)
        mid_plane = vol.shape[0] // 2

        # m = xy.imshow(vol[mid_plane, :, :], cmap=psf_cmap, vmin=0, vmax=1)
        # zx.imshow(vol[:, mid_plane, :], cmap=psf_cmap, vmin=0, vmax=1)
        # zy.imshow(vol[:, :, mid_plane].T, cmap=psf_cmap, vmin=0, vmax=1)

        levels = np.arange(0, 1.01, .01)
        m = xy.contourf(vol[mid_plane, :, :], cmap=psf_cmap, levels=levels, vmin=0, vmax=1)
        zx.contourf(vol[:, mid_plane, :], cmap=psf_cmap, levels=levels, vmin=0, vmax=1)
        zy.contourf(vol[:, :, mid_plane].T, cmap=psf_cmap, levels=levels, vmin=0, vmax=1)

        cax = inset_axes(zy, width="10%", height="100%", loc='center right', borderpad=-2)
        cb = plt.colorbar(m, cax=cax)
        cax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        return m

    scales = sorted(set([int(t) for t in np.logspace(0, 2, num=8)]))
    logger.info(f"PSNRs: {scales}")

    fig, axes = plt.subplots(len(scales), 3, figsize=(8, 16))
    for i, snr in tqdm(enumerate(scales), total=len(scales)):
        psfargs = dict(
            lam_detection=.605,
            amplitude_ranges=0,
            psf_shape=(64, 64, 64),
            x_voxel_size=.1,
            y_voxel_size=.1,
            z_voxel_size=.1,
            batch_size=10,
            snr=snr,
            max_jitter=0,
            cpu_workers=-1,
        )
        psfs, ys, psnrs, zplanes, maxcounts = next(SyntheticPSF(**psfargs).generator(debug=True))
        target_psnr = np.ceil(np.nanquantile(psnrs, .95))

        psf_slice(
            xy=axes[i, 0],
            zx=axes[i, 1],
            zy=axes[i, 2],
            vol=psfs[np.random.randint(psfs.shape[0]), :, :, :, 0],
        )

        axes[i, 0].set_title(f'r-SNR: {snr}')
        axes[i, 1].set_title(f"PSNR: {target_psnr:.2f}")
        axes[i, 2].set_title(f"$\gamma$: {gamma:.2f}")

    axes[-1, 0].set_xlabel('XY')
    axes[-1, 1].set_xlabel('ZX')
    axes[-1, 2].set_xlabel('ZY')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, right=0.95, wspace=.3, hspace=.3)
    plt.savefig(f'../data/noise.png', dpi=300, bbox_inches='tight', pad_inches=.25)


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
            zy.imshow(vol[2].T, cmap='Spectral_r', vmin=0, vmax=1)
        else:
            m = xy.imshow(np.max(vol, axis=0), cmap=psf_cmap, vmin=0, vmax=1)
            zx.imshow(np.max(vol, axis=1), cmap=psf_cmap, vmin=0, vmax=1)
            zy.imshow(np.max(vol, axis=2).T, cmap=psf_cmap, vmin=0, vmax=1)

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

    p_psf = gen.single_psf(pred, zplanes=0)
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

    otf = gen.embedding(psf)
    ax_xy = fig.add_subplot(gs[2, 0])
    ax_xz = fig.add_subplot(gs[2, 1])
    ax_yz = fig.add_subplot(gs[2, 2])
    ax_w = fig.add_subplot(gs[2, 3])
    psf_slice(ax_xy, ax_xz, ax_yz, otf, label='R_rel')
    wavefront(ax_w, y_wave, label='Ground truth', levels=mticks)

    k = 0
    for i, w in enumerate(y.amplitudes_ansi_waves):
        k += 1
        phi = np.zeros(60)
        phi[i] = w / (2 * np.pi / wavelength)
        phi = Wavefront(phi, order='ansi')

        psf = gen.single_psf(phi, zplanes=0)
        otf = gen.embedding(psf)
        ax_xy = fig.add_subplot(gs[2+k, 0])
        ax_xz = fig.add_subplot(gs[2+k, 1])
        ax_yz = fig.add_subplot(gs[2+k, 2])
        ax_w = fig.add_subplot(gs[2+k, 3])
        psf_slice(ax_xy, ax_xz, ax_yz, otf, label=f'Mode #{i}')
        wavefront(ax_w, phi.wave(100), label=f'Mode #{i}', levels=mticks)

    ax_zcoff = fig.add_subplot(gs[-1, :])
    ax_zcoff.plot(pred.amplitudes_ansi_waves, '-o', color='C0', label='Predictions')
    ax_zcoff.plot(y.amplitudes_ansi_waves, '-o', color='C1', label='Ground truth')
    ax_zcoff.legend(frameon=False, loc='upper center', bbox_to_anchor=(.1, 1))
    ax_zcoff.set_xticks(range(len(pred.amplitudes_ansi_waves)))
    ax_zcoff.set_ylabel(f'Amplitudes\n($\lambda = {wavelength}~\mu m$)')
    ax_zcoff.spines['top'].set_visible(False)
    ax_zcoff.set_xlim((0, len(pred.amplitudes_ansi_waves)))
    ax_zcoff.grid(True, which="both", axis='both', lw=1, ls='--', zorder=0)

    plt.subplots_adjust(top=0.95, right=0.95, wspace=.2)
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight', pad_inches=.25)
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight', pad_inches=.25)


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
        wavelength: float = .605,
        display: bool = False,
        psf_cmap: str = 'hot',
        gamma: float = .5,
        threshold: float = .01,
):

    def wavefront(iax, phi, levels, label='', nas=(.5, .75, .9, .95, .99)):
        def na_mask(radius):
            center = (int(phi.shape[0]/2), int(phi.shape[1]/2))
            Y, X = np.ogrid[:phi.shape[0], :phi.shape[1]]
            dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
            mask = dist_from_center <= radius
            return mask

        mat = iax.contourf(
            phi,
            levels=levels,
            cmap=wave_cmap,
            vmin=np.min(levels),
            vmax=np.max(levels),
            extend='both'
        )

        pcts = []
        for d in nas:
            r = (d * phi.shape[0]) / 2
            circle = patches.Circle((50, 50), r, ls='--', ec="dimgrey", fc="none", zorder=3)
            iax.add_patch(circle)

            mask = phi * na_mask(radius=r)
            pcts.append((np.nanquantile(mask, .05), np.nanquantile(mask, .95)))

        circle = patches.Circle((50, 50), 50, ec="dimgrey", fc="none", zorder=3)
        iax.add_patch(circle)

        divider = make_axes_locatable(iax)
        top = divider.append_axes("top", size='30%', pad=0.2)
        phi = phi.flatten()
        top.hist(phi, bins=60, color='grey')

        err = '\n'.join([
            f'$P2P_{{NA={na}}}$={abs(p[1]-p[0]):.2f}\t[$P_{{05}}$={p[0]:.2f}, $P_{{95}}$={p[1]:.2f}]'
            for na, p in zip(nas, pcts)
        ])
        top.set_title(f'{label}\n{err}')
        top.set_yticks([])
        top.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        top.set_xlim((np.nanquantile(phi, 0.01), np.nanquantile(phi, 0.99)))
        top.spines['right'].set_visible(False)
        top.spines['top'].set_visible(False)
        top.spines['left'].set_visible(False)
        return mat

    def psf_slice(xy, zx, zy, vol, label=''):
        if vol.shape[0] == 6:
            vmin, vmax, vcenter, step = 0, 2, 1, .1
            highcmap = plt.get_cmap('YlOrRd', 256)
            lowcmap = plt.get_cmap('YlGnBu_r', 256)
            low = np.linspace(0, 1 - step, int(abs(vcenter - vmin) / step))
            high = np.linspace(0, 1 + step, int(abs(vcenter - vmax) / step))
            cmap = np.vstack((lowcmap(low), [1, 1, 1, 1], highcmap(high)))
            cmap = mcolors.ListedColormap(cmap)

            inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=xy, wspace=0.1, hspace=0.1)
            ax = fig.add_subplot(inner[0])
            ax.imshow(vol[0], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel('Input')
            ax.set_xlabel(r'$\alpha = |\tau| / |\hat{\tau}|$')
            ax = fig.add_subplot(inner[1])
            ax.imshow(vol[3], cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(r'$\varphi = \angle \tau$')
            xy.axis('off')

            inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=zx, wspace=0.1, hspace=0.1)
            ax = fig.add_subplot(inner[0])
            ax.imshow(vol[1], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(r'$\alpha = |\tau| / |\hat{\tau}|$')

            ax = fig.add_subplot(inner[1])
            ax.imshow(vol[4], cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(r'$\varphi = \angle \tau$')
            zx.axis('off')

            inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=zy, wspace=0.1, hspace=0.1)
            ax = fig.add_subplot(inner[0])
            m = ax.imshow(vol[2].T, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(r'$\alpha = |\tau| / |\hat{\tau}|$')

            ax = fig.add_subplot(inner[1])
            ax.imshow(vol[5].T, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(r'$\varphi = \angle \tau$')
            zy.axis('off')

            cax = inset_axes(zy, width="10%", height="100%", loc='center right', borderpad=-3)
            cb = plt.colorbar(m, cax=cax)
            cax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            cax.yaxis.set_label_position("right")

        elif vol.shape[0] == 3:
            m = xy.imshow(vol[0], cmap='Spectral_r', vmin=0, vmax=1)
            zx.imshow(vol[1], cmap='Spectral_r', vmin=0, vmax=1)
            zy.imshow(vol[2].T, cmap='Spectral_r', vmin=0, vmax=1)
        else:
            vol = vol ** gamma
            vol = np.nan_to_num(vol)

            m = xy.imshow(np.max(vol, axis=0), cmap=psf_cmap, vmin=0, vmax=1)
            zx.imshow(np.max(vol, axis=1), cmap=psf_cmap, vmin=0, vmax=1)
            zy.imshow(np.max(vol, axis=2).T, cmap=psf_cmap, vmin=0, vmax=1)

            cax = inset_axes(zy, width="10%", height="100%", loc='center right', borderpad=-3)
            cb = plt.colorbar(m, cax=cax)
            cax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            cax.set_ylabel(f"$\gamma$: {gamma:.2f}")
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

    if not np.isscalar(psnr):
        psnr = psnr[0]

    if not np.isscalar(maxcounts):
        maxcounts = maxcounts[0]

    y_wave = y.wave(size=100)
    pred_wave = pred.wave(size=100)
    diff = y_wave - pred_wave

    fig = plt.figure(figsize=(13, 20))
    gs = fig.add_gridspec(6 if gt_psf is None else 7, 3)

    ax_gt = fig.add_subplot(gs[:2, 0])
    ax_pred = fig.add_subplot(gs[:2, 1])
    ax_diff = fig.add_subplot(gs[:2, 2])
    cax = fig.add_axes([.99, 0.725, 0.02, .175])

    # input
    ax_xy = fig.add_subplot(gs[2, 0])
    ax_xz = fig.add_subplot(gs[2, 1])
    ax_yz = fig.add_subplot(gs[2, 2])

    # predictions
    ax_pxy = fig.add_subplot(gs[-3, 0])
    ax_pxz = fig.add_subplot(gs[-3, 1])
    ax_pyz = fig.add_subplot(gs[-3, 2])

    # corrected
    ax_cxy = fig.add_subplot(gs[-2, 0])
    ax_cxz = fig.add_subplot(gs[-2, 1])
    ax_cyz = fig.add_subplot(gs[-2, 2])

    ax_zcoff = fig.add_subplot(gs[-1, :])

    step = .25
    vmax = round(np.max([
        np.abs(round(np.nanquantile(y_wave, .05), 2)),
        np.abs(round(np.nanquantile(y_wave, .95), 2))
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

    mat = wavefront(ax_gt, y_wave, label='Ground truth', levels=mticks)
    wavefront(ax_pred, pred_wave, label='Predicted', levels=mticks)
    wavefront(ax_diff, diff, label='Residuals', levels=mticks)

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
    ax_xz.set_title('XZ')
    ax_yz.set_title('YZ')
    ax_xz.set_ylabel(f"PSNR: {psnr:.2f}")
    ax_yz.set_ylabel(f"Max photon count: {maxcounts:.0f}")

    psf_slice(ax_xy, ax_xz, ax_yz, psf, label='Input (MIP)')
    psf_slice(ax_pxy, ax_pxz, ax_pyz, predicted_psf, label='Predicted')
    psf_slice(ax_cxy, ax_cxz, ax_cyz, corrected_psf, label='Corrected')

    if gt_psf is not None:
        ax_xygt = fig.add_subplot(gs[3, 0])
        ax_xzgt = fig.add_subplot(gs[3, 1])
        ax_yzgt = fig.add_subplot(gs[3, 2])
        psf_slice(ax_xygt, ax_xzgt, ax_yzgt, gt_psf, label='Validation')

    # ax_zcoff.set_title('Zernike modes')
    ax_zcoff.plot(pred.amplitudes_ansi_waves, '-o', color='C0', label='Predictions')
    ax_zcoff.plot(y.amplitudes_ansi_waves, '-o', color='C1', label='Ground truth')
    ax_zcoff.legend(frameon=False, loc='upper center', bbox_to_anchor=(.075, 1))
    ax_zcoff.set_xticks(range(len(pred.amplitudes_ansi_waves)))
    ax_zcoff.set_xlim((0, len(pred.amplitudes_ansi_waves)-1))
    ax_zcoff.set_ylabel(f'Amplitudes\n($\lambda = {wavelength}~\mu m$)')
    ax_zcoff.spines['top'].set_visible(False)
    ax_zcoff.grid(True, which="both", axis='both', lw=1, ls='--', zorder=0)

    for ax in [ax_gt, ax_pred, ax_diff]:
        ax.axis('off')

    plt.subplots_adjust(top=0.95, right=0.95, wspace=.2)
    # plt.savefig(f'{save_path}.pdf', bbox_inches='tight', pad_inches=.25)
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight', pad_inches=.25)

    if display:
        plt.tight_layout()
        plt.show()


def matlab_diagnostic_assessment(
        psf: np.array,
        gt_psf: np.array,
        predicted_psf: np.array,
        corrected_psf: np.array,
        matlab_corrected_psf: np.array,
        psnr: Any,
        maxcounts: Any,
        y: Wavefront,
        pred: Wavefront,
        pred_matlab: Wavefront,
        save_path: Path,
        wavelength: float = .605,
        display: bool = False,
        psf_cmap: str = 'hot',
        gamma: float = .5,
        threshold: float = .01,
):

    def wavefront(iax, phi, levels, label='', nas=(.5, .75, .9, .95, .99), p2p=False):
        def na_mask(radius):
            center = (int(phi.shape[0]/2), int(phi.shape[1]/2))
            Y, X = np.ogrid[:phi.shape[0], :phi.shape[1]]
            dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
            mask = dist_from_center <= radius
            return mask

        mat = iax.contourf(
            phi,
            levels=levels,
            cmap=wave_cmap,
            vmin=np.min(levels),
            vmax=np.max(levels),
            extend='both'
        )

        pcts = []
        for d in nas:
            r = (d * phi.shape[0]) / 2
            circle = patches.Circle((50, 50), r, ls='--', ec="dimgrey", fc="none", zorder=3)
            iax.add_patch(circle)

            mask = phi * na_mask(radius=r)
            pcts.append((np.nanquantile(mask, .05), np.nanquantile(mask, .95)))

        circle = patches.Circle((50, 50), 50, ec="dimgrey", fc="none", zorder=3)
        iax.add_patch(circle)

        divider = make_axes_locatable(iax)
        top = divider.append_axes("top", size='30%', pad=0.2)
        phi = phi.flatten()
        top.hist(phi, bins=60, color='grey')

        if p2p:
            err = '\n'.join([
                f'$P2P_{{NA={na}}}$={abs(p[1]-p[0]):.2f}\t[$P_{{05}}$={p[0]:.2f}, $P_{{95}}$={p[1]:.2f}]'
                for na, p in zip(nas, pcts)
            ])

            top.set_title(f'{label}\n{err}')
        else:
            top.set_title(label)

        top.set_yticks([])
        top.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        top.set_xlim((np.nanquantile(phi, 0.01), np.nanquantile(phi, 0.99)))
        top.spines['right'].set_visible(False)
        top.spines['top'].set_visible(False)
        top.spines['left'].set_visible(False)
        return mat

    def psf_slice(xy, zx, zy, vol, label=''):
        if vol.shape[0] == 6:
            vmin, vmax, vcenter, step = 0, 2, 1, .1
            highcmap = plt.get_cmap('YlOrRd', 256)
            lowcmap = plt.get_cmap('YlGnBu_r', 256)
            low = np.linspace(0, 1 - step, int(abs(vcenter - vmin) / step))
            high = np.linspace(0, 1 + step, int(abs(vcenter - vmax) / step))
            cmap = np.vstack((lowcmap(low), [1, 1, 1, 1], highcmap(high)))
            cmap = mcolors.ListedColormap(cmap)

            inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=xy, wspace=0.1, hspace=0.1)
            ax = fig.add_subplot(inner[0])
            ax.imshow(vol[0], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel('Input')
            ax.set_xlabel(r'$\alpha = |\tau| / |\hat{\tau}|$')
            ax = fig.add_subplot(inner[1])
            ax.imshow(vol[3], cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(r'$\varphi = \angle \tau$')
            xy.axis('off')

            inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=zx, wspace=0.1, hspace=0.1)
            ax = fig.add_subplot(inner[0])
            ax.imshow(vol[1], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(r'$\alpha = |\tau| / |\hat{\tau}|$')

            ax = fig.add_subplot(inner[1])
            ax.imshow(vol[4], cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(r'$\varphi = \angle \tau$')
            zx.axis('off')

            inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=zy, wspace=0.1, hspace=0.1)
            ax = fig.add_subplot(inner[0])
            m = ax.imshow(vol[2].T, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(r'$\alpha = |\tau| / |\hat{\tau}|$')

            ax = fig.add_subplot(inner[1])
            ax.imshow(vol[5].T, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(r'$\varphi = \angle \tau$')
            zy.axis('off')

            cax = inset_axes(zy, width="10%", height="100%", loc='center right', borderpad=-3)
            cb = plt.colorbar(m, cax=cax)
            cax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            cax.yaxis.set_label_position("right")

        elif vol.shape[0] == 3:
            m = xy.imshow(vol[0], cmap='Spectral_r', vmin=0, vmax=1)
            zx.imshow(vol[1], cmap='Spectral_r', vmin=0, vmax=1)
            zy.imshow(vol[2].T, cmap='Spectral_r', vmin=0, vmax=1)
        else:
            vol = vol ** gamma
            vol = np.nan_to_num(vol)

            m = xy.imshow(np.max(vol, axis=0), cmap=psf_cmap, vmin=0, vmax=1)
            zx.imshow(np.max(vol, axis=1), cmap=psf_cmap, vmin=0, vmax=1)
            zy.imshow(np.max(vol, axis=2).T, cmap=psf_cmap, vmin=0, vmax=1)

            cax = inset_axes(zy, width="10%", height="100%", loc='center right', borderpad=-3)
            cb = plt.colorbar(m, cax=cax)
            cax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            cax.set_ylabel(f"$\gamma$: {gamma:.2f}")
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

    if not np.isscalar(psnr):
        psnr = psnr[0]

    if not np.isscalar(maxcounts):
        maxcounts = maxcounts[0]

    y_wave = y.wave(size=100)
    pred_wave = pred.wave(size=100)
    pred_matlab_wave = pred_matlab.wave(size=100)
    diff = y_wave - pred_wave
    matlab_diff = y_wave - pred_matlab_wave

    fig = plt.figure(figsize=(20, 35))
    gs = fig.add_gridspec(10, 3)

    ax_gt = fig.add_subplot(gs[:2, 0])
    ax_pred = fig.add_subplot(gs[:2, 1])
    ax_diff = fig.add_subplot(gs[:2, 2])
    cax = fig.add_axes([.99, 0.6, 0.02, .35])

    max_gt = fig.add_subplot(gs[2:4, 0])
    max_pred = fig.add_subplot(gs[2:4, 1])
    max_diff = fig.add_subplot(gs[2:4, 2])

    # input
    ax_xy = fig.add_subplot(gs[4, 0])
    ax_xz = fig.add_subplot(gs[4, 1])
    ax_yz = fig.add_subplot(gs[4, 2])

    # GT
    ax_xygt = fig.add_subplot(gs[5, 0])
    ax_xzgt = fig.add_subplot(gs[5, 1])
    ax_yzgt = fig.add_subplot(gs[5, 2])

    # predictions
    ax_pxy = fig.add_subplot(gs[-4, 0])
    ax_pxz = fig.add_subplot(gs[-4, 1])
    ax_pyz = fig.add_subplot(gs[-4, 2])

    # corrected
    ax_cxy = fig.add_subplot(gs[-3, 0])
    ax_cxz = fig.add_subplot(gs[-3, 1])
    ax_cyz = fig.add_subplot(gs[-3, 2])

    # matlab
    ax_mxy = fig.add_subplot(gs[-2, 0])
    ax_mxz = fig.add_subplot(gs[-2, 1])
    ax_myz = fig.add_subplot(gs[-2, 2])

    ax_zcoff = fig.add_subplot(gs[-1, :])

    step = .25
    vmax = round(np.max([
        np.abs(round(np.nanquantile(y_wave, .05), 2)),
        np.abs(round(np.nanquantile(y_wave, .95), 2))
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

    mat = wavefront(ax_gt, y_wave, label='Ground truth', levels=mticks, p2p=True)
    wavefront(ax_pred, pred_wave, label='Predicted', levels=mticks, p2p=True)
    wavefront(ax_diff, diff, label='Residuals', levels=mticks, p2p=True)

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

    mat = wavefront(max_gt, y_wave, label='Ground truth', levels=mticks)
    wavefront(max_pred, pred_matlab_wave, label='Matlab', levels=mticks)
    wavefront(max_diff, matlab_diff, label='Residuals', levels=mticks)

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
    ax_xz.set_title('XZ')
    ax_yz.set_title('YZ')
    ax_xz.set_ylabel(f"PSNR: {psnr:.2f}")
    ax_yz.set_ylabel(f"Max photon count: {maxcounts:.0f}")

    psf_slice(ax_xy, ax_xz, ax_yz, psf, label='Input')
    psf_slice(ax_xygt, ax_xzgt, ax_yzgt, gt_psf, label='Input PSF (MIP)')
    psf_slice(ax_pxy, ax_pxz, ax_pyz, predicted_psf, label='Predicted')
    psf_slice(ax_cxy, ax_cxz, ax_cyz, corrected_psf, label='Corrected')
    psf_slice(ax_mxy, ax_mxz, ax_myz, matlab_corrected_psf, label='Matlab')

    # ax_zcoff.set_title('Zernike modes')
    ax_zcoff.plot(pred.amplitudes_ansi_waves, '-o', color='C0', label='Predictions')
    ax_zcoff.plot(pred_matlab.amplitudes_ansi_waves, '-o', color='C1', label='Matlab')
    ax_zcoff.plot(y.amplitudes_ansi_waves, '-o', color='dimgrey', label='Ground truth')

    ax_zcoff.legend(frameon=False, loc='upper center', bbox_to_anchor=(.075, 1))
    ax_zcoff.set_xticks(range(len(pred.amplitudes_ansi_waves)))
    ax_zcoff.set_xlim((0, len(pred.amplitudes_ansi_waves)-1))
    ax_zcoff.set_ylabel(f'Amplitudes\n($\lambda = {wavelength}~\mu m$)')
    ax_zcoff.spines['top'].set_visible(False)
    ax_zcoff.grid(True, which="both", axis='both', lw=1, ls='--', zorder=0)

    for ax in [ax_gt, ax_pred, ax_diff, max_gt, max_pred, max_diff,]:
        ax.axis('off')

    plt.subplots_adjust(top=0.95, right=0.95, wspace=.2)
    # plt.savefig(f'{save_path}.pdf', bbox_inches='tight', pad_inches=.25)
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight', pad_inches=.25)

    if display:
        plt.tight_layout()
        plt.show()


def plot_residuals(df: pd.DataFrame, save_path, wavelength=.605, nsamples=100, label=''):
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    df = df.drop(df.index[[0, 1, 2, 4]])

    mean = np.mean(df[np.isfinite(df)], axis=0)
    stdv = np.std(df[np.isfinite(df)], axis=0)

    ax.errorbar(
        x=df.columns.values, y=mean, yerr=stdv,
        ecolor='lightgrey', lw=2,
        label=r'Mean $\pm$ stdev'
    )
    ax.grid(True, which="both", axis='both', lw=.5, ls='--', zorder=0)
    ax.set_yscale('log')
    ax.set_ylim((0.01, .5))
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(label)

    ax.set_xscale('log', subsx=[2, 4, 6, 8])
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.xaxis.set_minor_formatter(FormatStrFormatter("%d"))
    ax.set_xlim(10 ** 1, 10 ** 3)
    ax.grid(True, which="both", axis='both', lw=.5, ls='--', zorder=0)

    ax.set_yticks([.01, 0.02, .03, .05, .07, .15, .2, .3, .4], minor=True)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(axis='x', which='major', pad=10)

    divider = make_axes_locatable(ax)
    axl = divider.append_axes("top", size=2.0, pad=0, sharex=ax)

    axl.errorbar(
        x=df.columns.values, y=mean, yerr=stdv,
        ecolor='lightgrey', lw=2,
    )
    axl.set_xscale('linear')
    axl.set_ylim((0.5, 3))
    axl.spines['bottom'].set_visible(False)
    axl.xaxis.set_ticks_position('top')
    axl.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axl.grid(True, which="major", axis='both', lw=.5, ls='--', zorder=0)

    axl.set_xscale('log', subsx=[2, 3, 4, 6, 8])
    axl.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    axl.xaxis.set_minor_formatter(FormatStrFormatter("%d"))
    axl.tick_params(axis='x', which='major', pad=10)
    axl.set_xlim(10 ** 1, 10 ** 3)
    axl.grid(True, which="both", axis='both', lw=.5, ls='--', zorder=0)

    ax.set_ylabel(rf'Peak-to-peak residuals')
    axl.set_ylabel(rf'($n$ = {nsamples}; $\lambda = {wavelength}~\mu m$)')
    ax.legend(frameon=False, loc='lower center')

    plt.tight_layout()
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight', pad_inches=.25)
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight', pad_inches=.25)


def plot_mae_amps(df: pd.DataFrame, save_path, wavelength=.605):
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df.index, df['mae'], color='k')
    ax.grid(True, which="major", axis='both', lw=1, ls='--', zorder=0)
    ax.legend(frameon=False, loc='upper right')
    ax.set_ylabel(f'MAE ($\lambda = {wavelength}~\mu m$)')
    ax.set_xlabel(f'Amplitudes\n($\lambda = {wavelength}~\mu m$)')

    plt.tight_layout()
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight', pad_inches=.25)
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight', pad_inches=.25)


def plot_eval(means: pd.DataFrame, save_path, wavelength=.605, nsamples=100, label=''):
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'xtick.major.pad': 10
    })

    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.gca(projection="3d")
    fig, ax = plt.subplots(figsize=(8, 6))

    levels = [
        .15, .175, .2, .225,
        .25, .3, .35, .4, .45,
        .5, .6, .7, .8, .9,
        1, 1.25, 1.5, 1.75, 2.
    ]

    vmin, vmax, vcenter, step = 0, 2, .1, .01
    highcmap = plt.get_cmap('magma_r', 256)
    lowcmap = plt.get_cmap('GnBu_r', 256)
    low = np.linspace(0, 1 - step, int(abs(vcenter - vmin) / step))
    high = np.linspace(0, 1 + step, int(abs(vcenter - vmax) / step))
    cmap = np.vstack((lowcmap(low), [1, 1, 1, 1], highcmap(high)))
    cmap = mcolors.ListedColormap(cmap)

    contours = ax.contourf(
        means.columns.values,
        means.index.values,
        means.values,
        cmap=cmap,
        levels=levels,
        extend='both',
        linewidths=2,
        linestyles='dashed',
    )

    # ax.clabel(contours, contours.levels, inline=True, fontsize=10, colors='k')

    cax = fig.add_axes([1, 0.08, 0.03, 0.87])
    cbar = plt.colorbar(
        contours,
        cax=cax,
        fraction=0.046,
        pad=0.04,
        extend='both',
        spacing='proportional',
        format=FormatStrFormatter("%.2f")
    )

    cbar.ax.set_ylabel(
        'Peak-to-peak aberration $|P_{95} - P_{5}|$'
        rf'($\lambda = {int(wavelength*1000)}~nm$)'
    )
    cbar.ax.set_title(r'$\lambda$')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('left')

    ax.set_xlabel(f'Peak signal-to-noise ratio')
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.xaxis.set_minor_formatter(FormatStrFormatter("%d"))
    ax.set_xlim(10 ** 0, 10 ** 2)
    ax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

    if 'amplitude' in label:
        ax.set_ylabel(
            'Peak-to-peak aberration $|P_{95} - P_{5}|$'
            rf'($\lambda = {int(wavelength*1000)}~nm$)'
        )
        ax.set_yticks(np.arange(0, 11, .5), minor=True)
        ax.set_yticks(np.arange(0, 11, 1))
        ax.set_ylim(.25, 10)
    else:
        ax.set_ylabel(f"{label.replace('_', ' ').capitalize()} ($\mu m$)")

    plt.tight_layout()
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight', pad_inches=.25)
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight', pad_inches=.25)


def plot_models(df: pd.DataFrame, save_path, wavelength=.605, nsamples=100, label=''):
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    fig, ax = plt.subplots(figsize=(8, 6))

    for model in df.columns.values:
        ax.plot(df[model], label=model)

    ax.grid(True, which="both", axis='both', lw=.5, ls='--', zorder=0)
    ax.set_yscale('log')
    ax.set_ylim((0.01, .5))
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(label)

    ax.set_xscale('log', subsx=[2, 3, 4, 5, 6, 8])
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.xaxis.set_minor_formatter(FormatStrFormatter("%d"))
    ax.set_xlim(10 ** 1, 10 ** 2)
    ax.grid(True, which="both", axis='both', lw=.5, ls='--', zorder=0)

    ax.set_yticks([.01, 0.02, .03, .05, .07, .15, .2, .3, .4], minor=True)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(axis='x', which='major', pad=10)

    divider = make_axes_locatable(ax)
    axl = divider.append_axes("top", size=2.0, pad=0, sharex=ax)

    for model in df.columns:
        axl.plot(df[model], label=model)

    axl.set_xscale('linear')
    axl.set_ylim((0.5, 3))
    axl.spines['bottom'].set_visible(False)
    axl.xaxis.set_ticks_position('top')
    axl.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axl.grid(True, which="major", axis='both', lw=.5, ls='--', zorder=0)

    axl.set_xscale('log', subsx=[2, 3, 4, 5, 6, 8])
    axl.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    axl.xaxis.set_minor_formatter(FormatStrFormatter("%d"))
    axl.tick_params(axis='x', which='major', pad=10)
    axl.set_xlim(10 ** 1, 10 ** 2)
    axl.grid(True, which="both", axis='both', lw=.5, ls='--', zorder=0)

    ax.set_ylabel(rf'Peak-to-peak residuals')
    axl.set_ylabel(rf'($n$ = {nsamples}; $\lambda = {wavelength}~\mu m$)')
    ax.legend(frameon=False, loc='lower center', ncol=df.shape[1] // 2)

    plt.tight_layout()
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight', pad_inches=.25)
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight', pad_inches=.25)


def plot_residuals_per_mode(df: pd.DataFrame, save_path, wavelength=.605, nsamples=100):
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    order = pd.concat(
        [
            df.mean(axis=1).to_frame('mean'),
            df.sum(axis=1).to_frame('sum'),
            df.std(axis=1).to_frame('std'),
            df.median(axis=1).to_frame('median'),
            df
        ],
        axis=1
    )
    order = order.groupby('model')['mean', 'std', 'median', 'sum'].mean().sort_values('mean')
    logger.info(order)

    fig, axes = plt.subplots(nrows=order.shape[0], figsize=(df.shape[1] / 2, 20), sharex='all')

    for i, (model, row) in enumerate(order.iterrows()):
        axes[i].set_title(model)

        g = sns.boxplot(
            ax=axes[i],
            data=df[df.model == model],
            orient='v',
            palette="Set3",
        )
        g.set(ylim=(0, 2))

        axes[i].grid(True, which="both", axis='both', lw=.5, ls='--', zorder=0)
        axes[i].axhline(.25, color='r')

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        text = '\n'.join((
            rf"$\mu={round(row['mean'], 4)}$",
            rf"$\sigma={round(row['std'], 4)}$",
            rf"$m={round(row['median'], 4)}$",
            rf"$\Sigma={round(row['sum'], 4)}$",
        ))

        axes[i].text(0.025, 0.95, text, transform=axes[i].transAxes, va='top', bbox=props)

        axes[i].set_ylabel(
            'Residuals\n'
            rf'($n$ = {nsamples}; $\lambda = {wavelength}~\mu m$)'
        )

    plt.tight_layout()
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight', pad_inches=.25)
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight', pad_inches=.25)


def plot_convergence(df: pd.DataFrame, save_path, wavelength=.605, nsamples=100, psnr=30):
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    fig, ax = plt.subplots(figsize=(8, 6))

    for model in df['model'].unique():
        x = df[df['model'] == model]['niter'].values
        y = df[df['model'] == model]['residuals'].values
        ax.plot(x, y, label=model)

    ax.grid(True, which="both", axis='both', lw=.5, ls='--', zorder=0)
    ax.set_yscale('log')
    ax.set_ylim((0.01, 10))
    ax.set_xlim((0, df['niter'].nunique()))
    ax.set_xticks(range(df['niter'].nunique()))
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Number of iterations')
    ax.grid(True, which="both", axis='both', lw=.5, ls='--', zorder=0)

    ax.set_yticks([
        .01, 0.02, .03, .05, .07, .15,
        .25, .5, .75, 1, 1.5,
        2, 3, 4, 6, 8, 10
    ], minor=True)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
    ax.tick_params(axis='x', which='major', pad=10)

    #ax.set_title(f"PSNR: {psnr}, $n$ = {nsamples}")
    ax.set_ylabel(rf'Average peak-to-peak residuals ($\lambda = {round(wavelength*1000)}~nm$)')
    ax.legend(frameon=False, loc='lower center', ncol=4)

    plt.tight_layout()
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight', pad_inches=.25)
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight', pad_inches=.25)


def plot_inputs(
    n_modes=15,
    x_voxel_size=.15,
    y_voxel_size=.15,
    z_voxel_size=.6,
    psnr=100,
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
        else:
            #vol = vol ** gamma
            #vol = np.nan_to_num(vol)

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
            snr=psnr,
            max_jitter=0,
            cpu_workers=-1,
        )

        inputs = gen.single_otf(w, zplanes=0, normed=True, noise=False)
        psf = gen.single_psf(w, zplanes=0, normed=True, noise=False)

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
        ax_zcoff.plot(w.amplitudes_ansi_waves, '-o', color='C0', label='Predictions')
        ax_zcoff.set_xticks(range(len(w.amplitudes_ansi_waves)))
        ax_zcoff.set_ylabel(f'Amplitudes\n($\lambda = {wavelength}~\mu m$)')
        ax_zcoff.spines['top'].set_visible(False)
        ax_zcoff.grid()

        plt.subplots_adjust(top=0.95, right=0.95, wspace=.3)
        plt.savefig(f'../data/inputs/{i}.png', dpi=300, bbox_inches='tight', pad_inches=.25)


def diagnosis(
    pred: Wavefront,
    dm_before: np.array,
    dm_after: np.array,
    save_path: Path,
    wavelength: float = .605,
    threshold: float = .01,
    pred_std: Any = None
):
    def formatter(x, pos):
        val_str = '{:.1g}'.format(x)
        if np.abs(x) > 0 and np.abs(x) < 1:
            return val_str.replace("0", "", 1)
        else:
            return val_str
    def wavefront(iax, phi, levels, label='', nas=(.75, .95)):
        def na_mask(radius):
            center = (int(phi.shape[0]/2), int(phi.shape[1]/2))
            Y, X = np.ogrid[:phi.shape[0], :phi.shape[1]]
            dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
            mask = dist_from_center <= radius
            return mask

        mat = iax.contourf(
            phi,
            levels=levels,
            cmap=wave_cmap,
            vmin=np.min(levels),
            vmax=np.max(levels),
            extend='both',
        )

        pcts = []
        for d in nas:
            r = (d * phi.shape[0]) / 2
            circle = patches.Circle((50, 50), r, ls='--', ec="dimgrey", fc="none", zorder=3)
            iax.add_patch(circle)

            mask = phi * na_mask(radius=r)
            pcts.append((np.nanquantile(mask, .05), np.nanquantile(mask, .95)))

        circle = patches.Circle((50, 50), 50, ec="dimgrey", fc="none", zorder=3)
        iax.add_patch(circle)
        phi = phi.flatten()

        err = '\n'.join([
            f'$NA_{{{na}}}$={abs(p[1]-p[0]):.2f}$\lambda$'
            for na, p in zip(nas, pcts)
        ])
        iax.set_title(f'{label}\n{err}')
        iax.axis('off')
        iax.set_aspect("equal")
        return mat

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.autolimit_mode': 'round_numbers'
    })

    pred_wave = pred.wave(size=100)

    fig = plt.figure()
    gs = fig.add_gridspec(4, 3)

    ax_acts = fig.add_subplot(gs[:2, :2])
    ax_wavefornt = fig.add_subplot(gs[:2, 2])
    cax = inset_axes(ax_wavefornt, width="10%", height="100%", loc='center right', borderpad=-1)

    ax_zcoff = fig.add_subplot(gs[-2:, :])

    step = .25
    vmax = round(np.max([
        np.abs(round(np.nanquantile(pred_wave, .05), 2)),
        np.abs(round(np.nanquantile(pred_wave, .95), 2))
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

    mat = wavefront(ax_wavefornt, pred_wave, levels=mticks)

    cbar = fig.colorbar(mat, cax=cax, extend='both', format=formatter)
    cbar.ax.set_title(r'$\lambda$')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('left')

    ax_acts.plot(dm_before, ls='--', color='C1', label='Current')
    ax_acts.plot(dm_after, color='C0', label='Predictions')
    ax_acts.set_xlim(0, 68)
    ax_acts.set_title(f'DM actuators (volts)')
    ax_acts.legend(frameon=False, loc='lower left', ncol=2)
    ax_acts.grid(True, which="both", axis='both', lw=1, ls='--', zorder=0)
    ax_acts.axhline(0, ls='--', color='k', alpha=.5)

    ax_zcoff.plot(pred.amplitudes_ansi_waves, marker=".", markersize=5, color='dimgrey', label='Predictions')
    if pred_std is not None:
        ax_zcoff.fill_between(
            range(len(pred.amplitudes_ansi_waves)),
            pred.amplitudes_ansi_waves - pred_std.amplitudes_ansi_waves,
            pred.amplitudes_ansi_waves + pred_std.amplitudes_ansi_waves,
            label=r'$\pm \sigma$',
            color='gray',
            alpha=0.33
        )

    ax_zcoff.legend(frameon=False, loc='lower left', ncol=2)
    ax_zcoff.set_ylabel(f'Amplitudes ($\lambda = {int(wavelength*1000)}~nm$)')
    ax_zcoff.spines['top'].set_visible(False)
    ax_zcoff.grid(True, which="both", axis='y', lw=1, ls='--', zorder=0)
    ax_zcoff.spines['top'].set_visible(False)
    ax_zcoff.set_xticks(range(len(pred.amplitudes_ansi_waves)), minor=True)
    ax_zcoff.set_xticks(range(0, len(pred.amplitudes_ansi_waves)+5, 5), minor=False)
    ax_zcoff.set_xlim((0, len(pred.amplitudes_ansi_waves)))
    ax_zcoff.axhline(0, ls='--', color='r', alpha=.5)

    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.35, wspace=0.1)
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight', pad_inches=.1)
    plt.savefig(f'{save_path}.svg', dpi=300, bbox_inches='tight', pad_inches=.1)


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
        'legend.fontsize': 10,
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
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight', pad_inches=.25)
    plt.savefig(f'{save_path}.svg', dpi=300, bbox_inches='tight', pad_inches=.25)


def tiles(
    data: np.ndarray,
    save_path: Path,
    strides: int = 64,
    window_size: int = 64,
    gamma: float = 1.,
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

    data = data ** gamma
    ztiles = np.array_split(range(data.shape[0]), data.shape[0]//window_size)

    for z, idx in enumerate(ztiles):
        sample = np.max(data[idx], axis=0)
        tiles = sliding_window_view(sample, window_shape=[window_size, window_size])[::strides, ::strides]
        nrows, ncols = tiles.shape[0], tiles.shape[1]
        tiles = np.reshape(tiles, (-1, window_size, window_size))

        fig = plt.figure(figsize=(11, 8))
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

        plt.savefig(f'{save_path}_z{z}.png', dpi=300, bbox_inches='tight', pad_inches=.25)
        plt.savefig(f'{save_path}_z{z}.svg', dpi=300, bbox_inches='tight', pad_inches=.25)


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
    def pupil(iax, phi, levels, label='', nas=(.50, .75, .85, .95)):
        def na_mask(radius):
            center = (int(phi.shape[0]/2), int(phi.shape[1]/2))
            Y, X = np.ogrid[:phi.shape[0], :phi.shape[1]]
            dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
            mask = dist_from_center <= radius
            return mask

        mat = iax.contourf(
            phi,
            levels=levels,
            cmap=wave_cmap,
            vmin=np.min(levels),
            vmax=np.max(levels),
            extend='both'
        )

        pcts = []
        for d in nas:
            r = (d * phi.shape[0]) / 2
            circle = patches.Circle((50, 50), r, ls='--', ec="dimgrey", fc="none", zorder=3)
            iax.add_patch(circle)

            mask = phi * na_mask(radius=r)
            pcts.append((np.nanquantile(mask, .05), np.nanquantile(mask, .95)))

        circle = patches.Circle((50, 50), 50, ec="dimgrey", fc="none", zorder=3)
        iax.add_patch(circle)
        iax.axis('off')
        return mat

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.autolimit_mode': 'round_numbers'
    })
    final_pred = Wavefront(predictions[scale].values, lam_detection=wavelength)
    pred_wave = final_pred.wave(size=100)

    step = .25
    vmax = round(np.max([
        np.abs(round(np.nanquantile(pred_wave, .05), 2)),
        np.abs(round(np.nanquantile(pred_wave, .95), 2))
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

    for z in range(ztiles):
        fig = plt.figure(figsize=(11, 8))
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
                pred = Wavefront(predictions[f"p-z{z}-y{y}-x{x}"].values, lam_detection=wavelength)
                pred_wave = pred.wave(size=100)
                mat = pupil(grid[i], pred_wave, levels=mticks)
                grid[i].set_title(f"z{z}-y{y}-x{x}", pad=1)
                i += 1

        cbar = grid.cbar_axes[0].colorbar(mat)
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.set_yticks([])
        cbar.ax.set_title(f'$\lambda = {wavelength}~\mu m$')

        plt.savefig(f'{save_path}_z{z}.png', dpi=300, bbox_inches='tight', pad_inches=.25)
        plt.savefig(f'{save_path}_z{z}.svg', dpi=300, bbox_inches='tight', pad_inches=.25)
