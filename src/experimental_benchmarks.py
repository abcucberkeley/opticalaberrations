
import matplotlib
matplotlib.use('Agg')

import logging
import sys
import itertools
import subprocess
from pathlib import Path
from typing import Any, Optional
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as mcolors

import matplotlib.pyplot as plt
plt.set_loglevel('error')

import swifter
import numpy as np
import pandas as pd
import tensorflow as tf
from line_profiler_pycharm import profile
from tqdm import tqdm
from tifffile import imwrite

import utils
import backend
import eval

from wavefront import Wavefront
from synthetic import SyntheticPSF

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
tf.get_logger().setLevel(logging.ERROR)


@profile
def evaluate_phasenet(
    model: Path,
    eval_sign: str = 'signed',
    dist: str = 'mixed',
    batch_size: int = 512,
    num_objs: Optional[int] = 1,
    digital_rotations: bool = True,
    agg: str = 'median',
    na: float = 1.0,
    samples_per_bin: int = 1,
    phasenet_path: Path = Path('phasenetrepo')
):
    levels = [
        0, .05, .1, .15, .2, .25, .3, .35, .4, .45,
        .5, .6, .7, .8, .9,
        1, 1.25, 1.5, 1.75, 2., 2.5,
        3., 4., 5.,
    ]

    vmin, vmax, vcenter, step = levels[0], levels[-1], .5, .05
    highcmap = plt.get_cmap('magma_r', 256)
    lowcmap = plt.get_cmap('GnBu_r', 256)
    low = np.linspace(0, 1 - step, int(abs(vcenter - vmin) / step))
    high = np.linspace(0, 1 + step, int(abs(vcenter - vmax) / step))
    cmap = np.vstack((lowcmap(low), [1, 1, 1, 1], highcmap(high)))
    cmap = mcolors.ListedColormap(cmap)

    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.autolimit_mode': 'round_numbers'
    })

    if not phasenet_path.exists():
        subprocess.run(f"git clone https://github.com/mpicbg-csbd/phasenet.git phasenetrepo", shell=True)

    from phasenetrepo.phasenet.model import PhaseNet
    from csbdeep.utils import normalize, download_and_extract_zip_file

    download_and_extract_zip_file(
        url='https://github.com/mpicbg-csbd/phasenet/releases/download/0.1.0/model.zip',
        targetdir=f'{phasenet_path}/models/',
        verbose=1,
    )

    num_objs = 1 if num_objs is None else num_objs

    outdir = model.with_suffix('') / eval_sign / 'benchmark' / 'phasenet'
    outdir.mkdir(parents=True, exist_ok=True)
    savepath = Path(f'{outdir}/predictions_num_objs_{num_objs}.csv')
    modelspecs = backend.load_metadata(model)

    phasenet = PhaseNet(None, name='16_05_2020_11_48_14_berkeley_50planes', basedir=f'{phasenet_path}/models/')

    phasenetgen = SyntheticPSF(
        psf_type='widefield',
        lls_excitation_profile=None,
        psf_shape=(50, 50, 50),
        n_modes=modelspecs.n_modes,
        lam_detection=.510,
        x_voxel_size=.086,
        y_voxel_size=.086,
        z_voxel_size=.1,
        na_detection=1.1,
        refractive_index=1.33,
        order='ansi',
        distribution=dist,
        mode_weights='pyramid',
    )

    aberrations = [
        r for r in [
           (0, .05), (.05, .1), (.1, .15), (.15, .2), (.2, .25),
        ]
        for _ in range(samples_per_bin)
    ]

    if Path(f"{savepath}_predictions.npy").exists():
        wavefronts = np.load(f"{savepath}_wavefronts.npy", allow_pickle=True)
    else:
        wavefronts = [Wavefront(
            amplitudes=w,
            lam_detection=modelspecs.lam_detection,
            modes=modelspecs.n_modes,
            order='ansi',
            distribution=dist,
            mode_weights='pyramid',
            signed=True,
            rotate=True,
        ) for w in aberrations]
        np.save(f"{savepath}_wavefronts", wavefronts, allow_pickle=True)

    photons = np.arange(0, 1e6+1e5, 1e5)
    photons[0] = 1e5

    if savepath.exists():
        df = pd.read_csv(savepath, index_col=0, header=0)
    else:
        df = eval.eval_object(
            wavefronts=wavefronts,
            num_objs=num_objs,
            photons=photons,
            modelpath=model,
            batch_size=batch_size,
            eval_sign=eval_sign,
            savepath=savepath,
            digital_rotations=361 if digital_rotations else None
        )

    phasenet_inputs = eval.create_samples(
        wavefronts=wavefronts,
        photons=photons,
        gen=phasenetgen,
        savepath=Path(f"{savepath}_phasenet"),
        num_objs=num_objs,
    ).squeeze()
    phasenet_inputs = np.expand_dims([normalize(i) for i in phasenet_inputs], axis=-1)

    phasenet_wavefronts = [Wavefront(
        amplitudes=[0, 0, 0, 0] + list(phasenet.predict(i)),
        lam_detection=modelspecs.lam_detection,
        modes=modelspecs.n_modes,
        order='ansi',
        rotate=False,
    ) for i in phasenet_inputs]
    phasenet_predictions = np.array([w.amplitudes_ansi for w in phasenet_wavefronts])

    if Path(f"{savepath}_predictions_phasenet.npy").exists():
        phasenet_predictions = np.load(f"{savepath}_predictions_phasenet.npy")
    else:
        np.save(f"{savepath}_predictions_phasenet", phasenet_predictions)

    ys = np.stack([w.amplitudes for w, ph in itertools.product(wavefronts, photons)])
    residuals = ys - phasenet_predictions
    df['phasenet_residuals'] = [Wavefront(i, lam_detection=phasenetgen.lam_detection).peak2valley(na=na) for i in residuals]

    df['photoelectrons'] = utils.photons2electrons(df['photons'], quantum_efficiency=.82)

    x, y = 'photons', 'aberration'
    xstep, ystep = 5e4, .25

    ybins = np.arange(0, df[y].max()+ystep, ystep)
    df['ybins'] = pd.cut(df[y], ybins, labels=ybins[1:], include_lowest=True)

    if x == 'photons':
        xbins = df['photons'].values
        df['xbins'] = xbins
    else:
        xbins = np.arange(0, df[x].max()+xstep, xstep)
        df['xbins'] = pd.cut(df[x], xbins, labels=xbins[1:], include_lowest=True)

    fig, (axt, axp) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), sharey=True)

    dataframe = pd.pivot_table(df, values='residuals', index='ybins', columns='xbins', aggfunc=agg)
    dataframe = dataframe.sort_index()#.interpolate()

    phasenet_dataframe = pd.pivot_table(df, values='phasenet_residuals', index='ybins', columns='xbins', aggfunc=agg)
    phasenet_dataframe = phasenet_dataframe.sort_index()#.interpolate()

    contours = axt.contourf(
        dataframe.columns,
        dataframe.index.values,
        dataframe.values,
        cmap=cmap,
        levels=levels,
        extend='max',
        linewidths=2,
        linestyles='dashed',
    )
    axt.patch.set(hatch='/', edgecolor='lightgrey', lw=.01)

    axp.contourf(
        phasenet_dataframe.columns,
        phasenet_dataframe.index.values,
        phasenet_dataframe.values,
        cmap=cmap,
        levels=levels,
        extend='max',
        linewidths=2,
        linestyles='dashed',
    )
    axp.patch.set(hatch='/', edgecolor='lightgrey', lw=.01)

    cax = fig.add_axes([1.01, 0.08, 0.03, .9])
    cbar = plt.colorbar(
        contours,
        cax=cax,
        fraction=0.046,
        pad=0.04,
        extend='both',
        spacing='proportional',
        format=FormatStrFormatter("%.2f"),
        ticks=[0, .15, .3, .5, .75, 1., 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5],
    )

    cbar.ax.set_ylabel(rf'Residuals ({agg} peak-to-valley, $\lambda = {int(modelspecs.lam_detection*1000)}~nm$)')
    cbar.ax.set_title(r'$\lambda$')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('left')

    axt.set_ylabel(rf'Initial aberration ({agg} peak-to-valley, $\lambda = {int(modelspecs.lam_detection*1000)}~nm$)')
    axt.set_yticks(np.arange(0, 6, .5), minor=True)
    axt.set_yticks(np.arange(0, 6, 1))
    axt.set_ylim(ybins[0], ybins[-1])

    axt.set_xlim(xbins[0], xbins[-1])
    axt.set_xlabel(f'{x}')

    axt.spines['right'].set_visible(False)
    axp.spines['right'].set_visible(False)
    axt.spines['left'].set_visible(False)
    axp.spines['left'].set_visible(False)
    axt.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)
    axp.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

    plt.tight_layout()
    plt.savefig(f'{savepath}.pdf', bbox_inches='tight', pad_inches=.25)
    plt.savefig(f'{savepath}.png', dpi=300, bbox_inches='tight', pad_inches=.25)
    plt.savefig(f'{savepath}.svg', dpi=300, bbox_inches='tight', pad_inches=.25)

    return savepath
