
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
from functools import partial
from line_profiler_pycharm import profile

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
def random_samples_phasenet(
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
            digital_rotations=361 if digital_rotations else None,
            psf_type='widefield'
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


@profile
def phasenet_heatmap(
    datadir: Path,
    iter_num: int = 1,
    distribution: str = '/',
    batch_size: int = 128,
    samplelimit: Any = None,
    na: float = 1.0,
    eval_sign: str = 'signed',
    agg: str = 'median',
    modes: int = 15,
    no_beads: bool = True,
    phasenet_path: Path = Path('phasenetrepo')
):

    if not phasenet_path.exists():
        subprocess.run(f"git clone https://github.com/mpicbg-csbd/phasenet.git phasenetrepo", shell=True)

    from phasenetrepo.phasenet.model import PhaseNet
    from csbdeep.utils import normalize, download_and_extract_zip_file

    download_and_extract_zip_file(
        url='https://github.com/mpicbg-csbd/phasenet/releases/download/0.1.0/model.zip',
        targetdir=f'{phasenet_path}/models/',
        verbose=1,
    )

    if no_beads:
        savepath = phasenet_path.with_suffix('') / eval_sign / 'psf'
    else:
        savepath = phasenet_path.with_suffix('') / eval_sign / 'bead'

    savepath.mkdir(parents=True, exist_ok=True)

    if distribution != '/':
        savepath = Path(f'{savepath}/{distribution}_na_{str(na).replace("0.", "p")}')
    else:
        savepath = Path(f'{savepath}/na_{str(na).replace("0.", "p")}')

    phasenet = PhaseNet(None, name='16_05_2020_11_48_14_berkeley_50planes', basedir=f'{phasenet_path}/models/')

    phasenetgen = SyntheticPSF(
        psf_type='widefield',
        lls_excitation_profile=None,
        psf_shape=(64, 64, 64),
        n_modes=modes,
        lam_detection=.510,
        x_voxel_size=.086,
        y_voxel_size=.086,
        z_voxel_size=.1,
        na_detection=1.1,
        refractive_index=1.33,
        order='ansi',
        distribution='mixed',
        mode_weights='pyramid',
    )

    def predict(path):
        psf = backend.load_sample(path)
        psf = utils.resize_with_crop_or_pad(psf, crop_shape=(50, 50, 50))
        psf = np.expand_dims(normalize(psf), axis=-1)
        p = list(phasenet.predict(psf))
        wavefront = Wavefront(
            amplitudes=[0, 0, 0, 0] + p,
            lam_detection=phasenetgen.lam_detection,
            modes=modes,
            order='ansi',
            rotate=False,
        )
        return wavefront.amplitudes_ansi

    if iter_num == 1:
        # on first call, setup the dataframe with the 0th iteration stuff
        results = eval.collect_data(
            datapath=datadir,
            model=15,
            samplelimit=samplelimit,
            distribution=distribution,
            photons_range=None,
            npoints_range=(1, 1),
            psf_type=phasenetgen.psf_type,
            lam_detection=phasenetgen.lam_detection
        )
    else:
        # read previous results, ignoring criteria
        results = pd.read_csv(f'{savepath}_predictions.csv', header=0, index_col=0)

    prediction_cols = [col for col in results.columns if col.endswith('_prediction')]
    ground_truth_cols = [col for col in results.columns if col.endswith('_ground_truth')]
    residual_cols = [col for col in results.columns if col.endswith('_residual')]
    previous = results[results['iter_num'] == iter_num - 1]   # previous iteration = iter_num - 1

    # create realspace images for the current iteration
    paths = utils.multiprocess(
        func=partial(
            eval.generate_sample,
            iter_number=iter_num,
            savedir=savepath.resolve(),
            data=previous,
            psfgen=phasenetgen,
            no_phase=False,
            digital_rotations=None,
            no_beads=True
        ),
        jobs=previous['id'].values,
        desc=f'Generate samples ({savepath.resolve()})',
        unit=' sample',
        cores=-1
    )

    current = previous.copy()
    current['iter_num'] = iter_num
    current['file'] = paths
    current['file_windows'] = [utils.convert_to_windows_file_string(f) for f in paths]

    current[ground_truth_cols] = previous[residual_cols]
    current[prediction_cols] = np.array([predict(p) for p in paths])

    if eval_sign == 'positive_only':
        current[ground_truth_cols] = current[ground_truth_cols].abs()
        current[prediction_cols] = current[prediction_cols].abs()

    current[residual_cols] = current[ground_truth_cols].values - current[prediction_cols].values

    # compute residuals for each sample
    current['residuals'] = current.apply(
        lambda row: Wavefront(row[residual_cols].values, lam_detection=phasenetgen.lam_detection).peak2valley(na=na),
        axis=1
    )

    current['residuals_umRMS'] = current.apply(
        lambda row: np.linalg.norm(row[residual_cols].values),
        axis=1
    )

    results = results.append(current, ignore_index=True)

    if savepath is not None:
        try:
            results.to_csv(f'{savepath}_predictions.csv')
        except PermissionError:
            savepath = f'{savepath}_x'
            results.to_csv(f'{savepath}_predictions.csv')
        logger.info(f'Saved: {savepath.resolve()}_predictions.csv')


    df = results[results['iter_num'] == iter_num]
    df['photoelectrons'] = utils.photons2electrons(df['photons'], quantum_efficiency=.82)

    for x in ['photons', 'photoelectrons', 'counts', 'counts_p100', 'counts_p99']:

        if x == 'photons' or x == 'photoelectrons':
            label = f'Integrated photoelectrons'
            lims = (0, 10**6)
            pbins = np.arange(lims[0], lims[-1]+10e4, 5e4)
        elif x == 'counts':
            label = f'Integrated counts'
            lims = (4e6, 7.5e6)
            pbins = np.arange(lims[0], lims[-1]+2e5, 1e5)
        elif x == 'counts_p100':
            label = f'Max counts'
            lims = (0, 5000)
            pbins = np.arange(lims[0], lims[-1]+400, 200)
        else:
            label = f'99th percentile of counts'
            lims = (0, 300)
            pbins = np.arange(lims[0], lims[-1]+50, 25)

        df['pbins'] = pd.cut(df[x], pbins, labels=pbins[1:], include_lowest=True)
        bins = np.arange(0, 10.25, .25).round(2)
        df['ibins'] = pd.cut(
            df['aberration'],
            bins,
            labels=bins[1:],
            include_lowest=True
        )

        dataframe = pd.pivot_table(df, values='residuals', index='ibins', columns='pbins', aggfunc=agg)
        dataframe.insert(0, 0, dataframe.index.values)

        try:
            dataframe = dataframe.sort_index().interpolate()
        except ValueError:
            pass

        dataframe.to_csv(f'{savepath}_{x}.csv')
        logger.info(f'Saved: {savepath.resolve()}_{x}.csv')

        eval.plot_heatmap_p2v(
            dataframe,
            histograms=df if x == 'photons' else None,
            wavelength=phasenetgen.lam_detection,
            savepath=Path(f"{savepath}_iter_{iter_num}_{x}"),
            label=label,
            lims=lims,
            agg=agg
        )

    return savepath
