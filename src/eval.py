import itertools

import matplotlib
matplotlib.use('Agg')

from multiprocessing import Pool

import logging
import sys
from functools import partial
from pathlib import Path
from typing import Any, Optional
from matplotlib.ticker import FormatStrFormatter, LogFormatterExponent
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

import embeddings
import utils
import data_utils
import backend
import vis
import multipoint_dataset

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
def simulate_beads(psf, beads=None, photons=100000, object_size=0, num_objs=1, noise=True, fill_radius=.4):

    if beads is None:
        beads = multipoint_dataset.beads(
            image_shape=psf.shape,
            photons=photons,
            object_size=object_size,
            num_objs=num_objs,
            fill_radius=fill_radius
        )

    psf /= np.sum(psf)
    inputs = utils.fftconvolution(sample=beads, kernel=psf)

    if noise:
        inputs = utils.add_noise(inputs)
    else:  # convert image to counts
        inputs = utils.electrons2counts(inputs)

    return inputs


def generate_fourier_embeddings(
    image_id: int,
    data: pd.DataFrame,
    psfgen: SyntheticPSF,
    no_phase: bool = False,
    input_coverage: float = 1.0,
    digital_rotations: Optional[int] = None,
):
    hashtable = data[data['id'] == image_id].iloc[0].to_dict()

    f = Path(str(hashtable['file']))
    ys = [hashtable[cc] for cc in data.columns[data.columns.str.endswith('_residual')]]
    ref = np.squeeze(data_utils.get_image(f.with_stem(f'{f.stem}_gt')))

    wavefront = Wavefront(
        ys,
        modes=psfgen.n_modes,
        lam_detection=psfgen.lam_detection,
    )

    psf = psfgen.single_psf(
        phi=wavefront,
        normed=True,
        meta=False,
    )

    noisy_img = simulate_beads(
        psf=psf,
        beads=ref,
        gen=psfgen,
        photons=hashtable['photons']
    )

    emb = embeddings.fourier_embeddings(
        noisy_img,
        iotf=psfgen.iotf,
        no_phase=no_phase,
        alpha_val='abs',
        phi_val='angle',
        remove_interference=True,
        input_coverage=input_coverage,
        embedding_option=psfgen.embedding_option,
        digital_rotations=digital_rotations
    )
    return emb


@profile
def iter_evaluate(
    datapath,
    modelpath,
    niter: int = 5,
    samplelimit: int = 1,
    na: float = 1.0,
    distribution: str = '/',
    input_coverage: float = 1.0,
    threshold: float = 0.,
    no_phase: bool = False,
    batch_size: int = 100,
    photons_range: tuple = (100000, 100000),
    eval_sign: str = 'positive_only',
    digital_rotations: bool = False,
    rotations: Optional[int] = 361,
    savepath: Any = None,
):
    model = backend.load(modelpath)
    gen = backend.load_metadata(
        modelpath,
        signed=True,
        rotate=True,
        batch_size=batch_size,
        psf_shape=3*[model.input_shape[2]]
    )
    predicted_modes = model.output_shape[-1]

    metadata = data_utils.collect_dataset(
        datapath,
        modes=gen.n_modes,
        samplelimit=samplelimit,
        distribution=distribution,
        no_phase=no_phase,
        photons_range=photons_range,
        metadata=True
    )
    # this runs multiple samples (aka images) at a time.
    # ys is a 2D array, rows are each sample, columns give aberration in zernike coeffs
    metadata = np.array(list(metadata.take(-1)))
    ys = np.array([i.numpy() for i in metadata[:, 0]])[:, :predicted_modes]
    photons = np.array([i.numpy() for i in metadata[:, 1]])
    p2v = np.array([i.numpy() for i in metadata[:, 2]])
    npoints = np.array([i.numpy() for i in metadata[:, 3]])
    dists = np.array([i.numpy() for i in metadata[:, 4]])
    files = np.array([Path(str(i.numpy(), "utf-8")) for i in metadata[:, -1]])
    ids = np.arange(ys.shape[0], dtype=int)

    # 'results' is going to be written out as the .csv.
    # It holds the information from every iteration.
    # Initialize it first with the zeroth iteration.
    results = pd.DataFrame.from_dict({
        # image number where the voxel locations of the beads are given in 'file'. Constant over iterations.
        'id': ids,
        'niter': np.zeros_like(ids, dtype=int),  # iteration index.
        'aberration': p2v,  # initial p2v aberration. Constant over iterations.
        'residuals': p2v,  # remaining p2v aberration after ML correction.
        'photons': photons,  # integrated photons
        'distance': dists,  # average distance to nearst bead
        'file': files,  # file = binary image file filled with zeros except at location of beads
    })

    # make 3 more columns for every z mode,
    # all in terms of zernike coeffs. _residual will become the next iteration's ground truth.
    # iteration zero will have _prediction = zero, GT = _residual = starting aberration
    for z in range(ys.shape[-1]):
        results[f'z{z}_ground_truth'] = ys[:, z]
        results[f'z{z}_prediction'] = np.zeros_like(ys[:, z])
        results[f'z{z}_residual'] = ys[:, z]

    # ys contains the current GT aberration of every sample.
    for k in range(1, niter+1):
        before = results[results['niter'] == k - 1]
        updated_embeddings = partial(
            generate_fourier_embeddings,
            data=before,
            psfgen=gen,
            no_phase=no_phase,
            digital_rotations=rotations if digital_rotations else None,
            input_coverage=input_coverage,
        )

        check = data_utils.get_image(files[0])

        # need to get this working with digital rotations and a dynamic batchsize
        if k == 1 and (check.shape[0] == 3 or check.shape[0] == 6):
            # check if embeddings has been pre-computed
            inputs = tf.data.Dataset.from_tensor_slices(np.vectorize(str)(files))
            inputs = inputs.map(lambda x: tf.py_function(data_utils.get_image, [x], tf.float32))
        else:
            inputs = tf.data.Dataset.from_tensor_slices(ids)
            inputs = inputs.map(lambda image_id: tf.py_function(updated_embeddings, [image_id], tf.float32))

        ps, stdev = backend.predict_dataset(
            model,
            inputs,
            psfgen=gen,
            batch_size=batch_size,
            threshold=threshold,
            desc=f'Predicting (iter #{k})',
            digital_rotations=rotations if digital_rotations else None,
            plot_rotations=False,
        )

        if eval_sign == 'positive_only':
            ys = np.abs(ys)
            if len(ps.shape) > 1:
                ps = np.abs(ps)[:, :ys.shape[-1]]
            else:
                ps = np.abs(ps)[np.newaxis, :ys.shape[-1]]
        else:
            if len(ps.shape) > 1:
                ps = ps[:, :ys.shape[-1]]
            else:
                ps = ps[np.newaxis, :ys.shape[-1]]

        res = ys - ps

        current = pd.DataFrame(ids, columns=['id'])
        current['niter'] = k
        current['aberration'] = p2v
        current['residuals'] = [Wavefront(i, lam_detection=gen.lam_detection).peak2valley(na=na) for i in res]
        current['photons'] = photons
        current['neighbors'] = npoints
        current['distance'] = dists
        current['file'] = files

        for z in range(ps.shape[-1]):
            current[f'z{z}_ground_truth'] = ys[:, z]
            current[f'z{z}_prediction'] = ps[:, z]
            current[f'z{z}_residual'] = res[:, z]

        results = results.append(current, ignore_index=True)

        if savepath is not None:
            results.to_csv(f'{savepath}_predictions.csv')

        # update the aberration for the next iteration with the residue
        ys = res

    return results


@profile
def plot_heatmap(means, wavelength, savepath, label='Integrated photons', lims=(0, 100)):
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'xtick.major.pad': 10
    })

    fig, ax = plt.subplots(figsize=(8, 6))
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

    contours = ax.contourf(
        means.columns.values,
        means.index.values,
        means.values,
        cmap=cmap,
        levels=levels,
        extend='max',
        linewidths=2,
        linestyles='dashed',
    )
    ax.patch.set(hatch='/', edgecolor='lightgrey', lw=.01)

    cax = fig.add_axes([1.01, 0.08, 0.03, 0.87])
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

    cbar.ax.set_ylabel(rf'Residuals; average peak-to-valley ($\lambda = {int(wavelength * 1000)}~nm$)')
    cbar.ax.set_title(r'$\lambda$')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('left')

    ax.set_xlabel(label)
    ax.set_xlim(lims)

    ax.set_ylabel('Initial aberration (average peak-to-valley)')
    ax.set_yticks(np.arange(0, 6, .5), minor=True)
    ax.set_yticks(np.arange(0, 6, 1))
    ax.set_ylim(.25, 5)

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)
    plt.tight_layout()

    plt.savefig(f'{savepath}.pdf', bbox_inches='tight', pad_inches=.25)
    plt.savefig(f'{savepath}.png', dpi=300, bbox_inches='tight', pad_inches=.25)
    return fig


@profile
def snrheatmap(
    modelpath: Path,
    datadir: Path,
    niter: int = 1,
    distribution: str = '/',
    samplelimit: Any = None,
    na: float = 1.0,
    input_coverage: float = 1.0,
    batch_size: int = 100,
    eval_sign: str = 'positive_only',
    digital_rotations: bool = False,
):
    modelspecs = backend.load_metadata(modelpath)
    savepath = modelpath.with_suffix('') / eval_sign / f'snrheatmaps_{input_coverage}'
    savepath.mkdir(parents=True, exist_ok=True)

    if distribution != '/':
        savepath = Path(f'{savepath}/{distribution}_na_{str(na).replace("0.", "p")}')
    else:
        savepath = Path(f'{savepath}/na_{str(na).replace("0.", "p")}')

    if datadir.suffix == '.csv':
        df = pd.read_csv(datadir, header=0, index_col=0)
    else:
        df = iter_evaluate(
            niter=niter,
            modelpath=modelpath,
            datapath=datadir,
            savepath=savepath,
            samplelimit=samplelimit,
            na=na,
            input_coverage=input_coverage,
            batch_size=batch_size,
            photons_range=(0, 10**6),
            eval_sign=eval_sign,
            digital_rotations=digital_rotations
        )

    bins = np.arange(0, 10.25, .25)
    df['bins'] = pd.cut(df['aberration'], bins, labels=bins[1:], include_lowest=True)

    means = pd.pivot_table(
        df[df['niter'] == niter], values='residuals', index='bins', columns='photons', aggfunc=np.mean
    )
    means = means.sort_index().interpolate()

    logger.info(means)
    means.to_csv(f'{savepath}.csv')

    plot_heatmap(
        means,
        wavelength=modelspecs.lam_detection,
        savepath=savepath,
        label=f'Integrated photons',
        lims=(0, 10**6)
    )

    for z in range(3, modelspecs.n_modes):
        if z == 4: continue  # ignore defocus

        try:
            df[f"z{z}_ground_truth"] = df[f"z{z}_ground_truth"].swifter.apply(
                lambda x: Wavefront({z: x}, lam_detection=modelspecs.lam_detection).peak2valley()
            )
            df[f"z{z}_residual"] = df[f"z{z}_residual"].swifter.apply(
                lambda x: Wavefront({z: x}, lam_detection=modelspecs.lam_detection).peak2valley()
            )

            bins = np.arange(0, 10.25, .25)
            df[f'z{z}_bins'] = pd.cut(df[f'z{z}_ground_truth'], bins, labels=bins[1:], include_lowest=True)
            means = pd.pivot_table(df, values=f'z{z}_residual', index=f'z{z}_bins', columns='photons', aggfunc=np.mean)
            means = means.sort_index().interpolate()
            logger.info(f"z{z}")
            logger.info(means)
            means.to_csv(savepath.with_name(f"{savepath.name}_z{z}.csv"))

            plot_heatmap(
                means,
                wavelength=modelspecs.lam_detection,
                savepath=savepath.with_name(f"{savepath.name}_z{z}"),
                label=f'Integrated photons',
                lims=(0, 10**6)
            )
        except KeyError:
            logger.warning(f"No evaluation found for z{z}")


@profile
def densityheatmap(
    modelpath: Path,
    datadir: Path,
    niter: int = 1,
    distribution: str = '/',
    na: float = 1.0,
    samplelimit: Any = None,
    input_coverage: float = 1.0,
    batch_size: int = 100,
    photons_range: tuple = (100000, 100000),
    eval_sign: str = 'positive_only',
    digital_rotations: bool = False,
):
    modelspecs = backend.load_metadata(modelpath)

    savepath = modelpath.with_suffix('') / eval_sign / f'densityheatmaps_{input_coverage}'
    savepath.mkdir(parents=True, exist_ok=True)

    if distribution != '/':
        savepath = Path(f'{savepath}/{distribution}_na_{str(na).replace("0.", "p")}')
    else:
        savepath = Path(f'{savepath}/na_{str(na).replace("0.", "p")}')

    if datadir.suffix == '.csv':
        df = pd.read_csv(datadir, header=0, index_col=0)
    else:
        df = iter_evaluate(
            niter=niter,
            modelpath=modelpath,
            datapath=datadir,
            savepath=savepath,
            samplelimit=samplelimit,
            distribution=distribution,
            na=na,
            photons_range=photons_range,
            input_coverage=input_coverage,
            batch_size=batch_size,
            eval_sign=eval_sign,
            digital_rotations=digital_rotations
        )

    for col, label, lims in zip(
        ['neighbors', 'distance'],
        ['Number of objects', 'Average distance to nearest neighbor (microns)'],
        [(1, 150), (0, 4)]
    ):
        bins = np.arange(0, 10.25, .25)
        df['bins'] = pd.cut(df['aberration'], bins, labels=bins[1:], include_lowest=True)

        means = pd.pivot_table(
            df[df['niter'] == niter], values='residuals', index='bins', columns=col, aggfunc=np.mean
        )
        means = means.sort_index().interpolate()
        means.to_csv(f'{savepath}_{col}.csv')

        plot_heatmap(
            means,
            wavelength=modelspecs.lam_detection,
            savepath=f'{savepath}_{col}',
            label=label,
            lims=lims
        )

        for z in range(3, modelspecs.n_modes):
            if z == 4: continue  # ignore defocus

            try:
                df[f"z{z}_ground_truth"] = df[f"z{z}_ground_truth"].swifter.apply(
                    lambda x: Wavefront({z: x}, lam_detection=modelspecs.lam_detection).peak2valley()
                )
                df[f"z{z}_residual"] = df[f"z{z}_residual"].swifter.apply(
                    lambda x: Wavefront({z: x}, lam_detection=modelspecs.lam_detection).peak2valley()
                )

                bins = np.arange(0, 10.25, .25)
                df[f'z{z}_bins'] = pd.cut(df[f'z{z}_ground_truth'], bins, labels=bins[1:], include_lowest=True)
                means = pd.pivot_table(df, values=f'z{z}_residual', index=f'z{z}_bins', columns=col, aggfunc=np.mean)
                means = means.sort_index().interpolate()
                logger.info(f"z{z}")
                logger.info(means)
                means.to_csv(savepath.with_name(f"{savepath.name}_{col}_z{z}.csv"))

                plot_heatmap(
                    means,
                    wavelength=modelspecs.lam_detection,
                    savepath=savepath.with_name(f"{savepath.name}_{col}_z{z}"),
                    label=label,
                    lims=lims
                )
            except KeyError:
                logger.warning(f"No evaluation found for z{z}")


@profile
def iterheatmap(
    modelpath: Path,
    datadir: Path,
    niter: int = 5,
    distribution: str = '/',
    samplelimit: Any = None,
    na: float = 1.0,
    input_coverage: float = 1.0,
    no_phase: bool = False,
    batch_size: int = 1024,
    photons_range: tuple = (100000, 100000),
    eval_sign: str = 'positive_only',
    digital_rotations: bool = False,
):
    modelspecs = backend.load_metadata(modelpath)
    savepath = modelpath.with_suffix('') / eval_sign / f'iterheatmaps_{input_coverage}'
    savepath.mkdir(parents=True, exist_ok=True)

    if distribution != '/':
        savepath = Path(f'{savepath}/{distribution}_na_{str(na).replace("0.", "p")}')
    else:
        savepath = Path(f'{savepath}/na_{str(na).replace("0.", "p")}')

    if datadir.suffix == '.csv':
        df = pd.read_csv(datadir, header=0, index_col=0)
    else:
        df = iter_evaluate(
            datadir,
            savepath=savepath,
            niter=niter,
            modelpath=modelpath,
            samplelimit=samplelimit,
            na=na,
            photons_range=photons_range,
            input_coverage=input_coverage,
            no_phase=no_phase,
            batch_size=batch_size,
            eval_sign=eval_sign,
            digital_rotations=digital_rotations
        )

    means = pd.pivot_table(
        df[df['niter'] == 0], values='residuals', index='id', columns='niter', aggfunc=np.mean
    )
    for i in range(1, niter+1):
        means[i] = pd.pivot_table(
            df[df['niter'] == i], values='residuals', index='id', columns='niter', aggfunc=np.mean
        )

    bins = np.arange(0, 10.25, .25)
    means.index = pd.cut(means[0], bins, labels=bins[1:], include_lowest=True)
    means.index.name = 'bins'
    means = means.groupby("bins").agg("mean")
    means.loc[0] = pd.Series({cc: 0 for cc in means.columns})
    means = means.sort_index().interpolate()

    logger.info(means)
    means.to_csv(f'{savepath}.csv')

    plot_heatmap(
        means,
        wavelength=modelspecs.lam_detection,
        savepath=savepath,
        label=f'Number of iterations',
        lims=(0, niter)
    )

    for z in range(3, modelspecs.n_modes):
        if z == 4: continue  # ignore defocus

        try:
            df[f"z{z}_ground_truth"] = df[f"z{z}_ground_truth"].swifter.apply(
                lambda x: Wavefront({z: x}, lam_detection=modelspecs.lam_detection).peak2valley()
            )
            df[f"z{z}_residual"] = df[f"z{z}_residual"].swifter.apply(
                lambda x: Wavefront({z: x}, lam_detection=modelspecs.lam_detection).peak2valley()
            )

            means = pd.pivot_table(
                df[df['niter'] == 0], values=f"z{z}_residual", index='id', columns='niter', aggfunc=np.mean
            )
            for i in range(1, niter + 1):
                means[i] = pd.pivot_table(
                    df[df['niter'] == i], values=f"z{z}_residual", index='id', columns='niter', aggfunc=np.mean
                )

            bins = np.arange(0, 10.25, .25)
            means.index = pd.cut(means[0], bins, labels=bins[1:], include_lowest=True)
            means.index.name = 'bins'
            means = means.groupby("bins").agg("mean")
            means.loc[0] = pd.Series({cc: 0 for cc in means.columns})
            means = means.sort_index().interpolate()

            logger.info(f"z{z}")
            logger.info(means)
            means.to_csv(savepath.with_name(f"{savepath.name}_z{z}.csv"))

            plot_heatmap(
                means,
                wavelength=modelspecs.lam_detection,
                savepath=savepath.with_name(f"{savepath.name}_z{z}"),
                label=f'Number of iterations',
                lims=(0, niter)
            )
        except KeyError:
            logger.warning(f"No evaluation found for z{z}")


@profile
def random_samples(
    model: Path,
    photons: int = 1e5,
    batch_size: int = 512,
    eval_sign: str = 'signed',
    digital_rotations: bool = False,
):
    m = backend.load(model)
    m.summary()
    no_phase = True if m.input_shape[1] == 3 else False

    pool = Pool(processes=4)  # plotting = 2*calculation time, so shouldn't need more than 2-4 processes to keep up.

    for dist in ['single', 'bimodal', 'multinomial', 'powerlaw', 'dirichlet']:
        for amplitude_range in [(.05, .1), (.1, .2), (.2, .3)]:
            gen = backend.load_metadata(
                model,
                amplitude_ranges=amplitude_range,
                distribution=dist,
                signed=False if eval_sign == 'positive_only' else True,
                rotate=True,
                mode_weights='pyramid',
                psf_shape=(64, 64, 64),
            )
            for s in range(10):
                for num_objs in tqdm([1, 2, 5, 25, 50, 100, 150]):
                    reference = multipoint_dataset.beads(
                        photons=photons,
                        image_shape=gen.psf_shape,
                        object_size=0,
                        num_objs=num_objs
                    )

                    phi = Wavefront(
                        amplitude_range,
                        modes=gen.n_modes,
                        distribution=dist,
                        signed=False if eval_sign == 'positive_only' else True,
                        rotate=True,
                        mode_weights='pyramid',
                        lam_detection=gen.lam_detection,
                    )

                    # aberrated PSF without noise
                    psf, y, y_lls_defocus = gen.single_psf(
                        phi=phi,
                        normed=True,
                        meta=True,
                        lls_defocus_offset=(0, 0)
                    )

                    noisy_img = simulate_beads(psf, beads=reference, noise=True)
                    maxcounts = np.max(noisy_img)
                    noisy_img /= maxcounts

                    save_path = Path(
                        f"{model.with_suffix('')}/{eval_sign}/samples/{dist}/um-{amplitude_range[-1]}/num_objs-{num_objs:02d}"
                    )
                    save_path.mkdir(exist_ok=True, parents=True)

                    embeddings = backend.preprocess(
                        noisy_img,
                        modelpsfgen=gen,
                        digital_rotations=361 if digital_rotations else None,
                        remove_background=True,
                        normalize=True,
                        plot=save_path / f'{s}',
                    )

                    if digital_rotations:
                        res = backend.predict_rotation(
                            m,
                            embeddings,
                            save_path=save_path / f'{s}',
                            psfgen=gen,
                            no_phase=no_phase,
                            batch_size=batch_size,
                            plot=save_path / f'{s}',
                            plot_rotations=save_path / f'{s}',
                        )
                    else:
                        res = backend.bootstrap_predict(
                            m,
                            embeddings,
                            psfgen=gen,
                            no_phase=no_phase,
                            batch_size=batch_size,
                            plot=save_path / f'{s}',
                        )

                    try:
                        p, std = res
                        p_lls_defocus = None
                    except ValueError:
                        p, std, p_lls_defocus = res

                    if eval_sign == 'positive_only':
                        y = np.abs(y)
                        if len(p.shape) > 1:
                            p = np.abs(p)[:, :y.shape[-1]]
                        else:
                            p = np.abs(p)[np.newaxis, :y.shape[-1]]
                    else:
                        if len(p.shape) > 1:
                            p = p[:, :y.shape[-1]]
                        else:
                            p = p[np.newaxis, :y.shape[-1]]

                    residuals = y - p

                    p_wave = Wavefront(p, lam_detection=gen.lam_detection)
                    y_wave = Wavefront(y, lam_detection=gen.lam_detection)
                    residuals = Wavefront(residuals, lam_detection=gen.lam_detection)

                    p_psf = gen.single_psf(p_wave, normed=True)
                    gt_psf = gen.single_psf(y_wave, normed=True)

                    corrected_psf = gen.single_psf(residuals)
                    corrected_noisy_img = simulate_beads(corrected_psf, beads=reference, noise=True)
                    corrected_noisy_img /= np.max(corrected_noisy_img)

                    imwrite(save_path / f'psf_{s}.tif', noisy_img)
                    imwrite(save_path / f'corrected_psf_{s}.tif', corrected_psf)

                    task = partial(
                        vis.diagnostic_assessment,
                        psf=noisy_img,
                        gt_psf=gt_psf,
                        predicted_psf=p_psf,
                        corrected_psf=corrected_noisy_img,
                        photons=photons,
                        maxcounts=maxcounts,
                        y=y_wave,
                        pred=p_wave,
                        y_lls_defocus=y_lls_defocus,
                        p_lls_defocus=p_lls_defocus,
                        save_path=save_path / f'{s}',
                        display=False,
                        pltstyle='default'
                    )
                    _ = pool.apply_async(task)  # issue task

    pool.close()    # close the pool
    pool.join()     # wait for all tasks to complete


@profile
def eval_object(
    phi,
    modelpath,
    photons: list,
    na: float = 1.0,
    batch_size: int = 512,
    n_samples: int = 1,
    eval_sign: str = 'rotations',
    savepath: Any = None,
    digital_rotations: Optional[int] = 361
):
    model = backend.load(modelpath)
    gen = backend.load_metadata(modelpath, psf_shape=3*[model.input_shape[2]])

    df = pd.DataFrame([], columns=['aberration', 'prediction', 'residuals', 'photons'])
    wavefronts = [Wavefront(w, lam_detection=gen.lam_detection) for w in phi]
    p2v = [w.peak2valley(na=na) for w in wavefronts]
    kernels = [gen.single_psf(phi=w, normed=True) for w in wavefronts]

    inputs = np.stack([
        backend.preprocess(
            simulate_beads(psf=kernels[k], object_size=0, num_objs=1, photons=ph, noise=True, fill_radius=0),
            modelpsfgen=gen,
            digital_rotations=digital_rotations,
            remove_background=True,
            normalize=True,
            #plot=f"{savepath}_{p2v[k]}_{ph}_{i}"
        )
        for k, ph, i in itertools.product(range(len(kernels)), photons, range(n_samples))
    ], axis=0)
    ys = np.stack([phi[k] for k, ph, i in itertools.product(range(len(kernels)), photons, range(n_samples))])

    inputs = tf.data.Dataset.from_tensor_slices(inputs)

    res = backend.predict_dataset(
        model,
        inputs=inputs,
        psfgen=gen,
        batch_size=batch_size,
        save_path=[f"{savepath}_{a}_{ph}_{i}" for a, ph, i in itertools.product(p2v, photons, range(n_samples))],
        digital_rotations=digital_rotations,
        #plot_rotations=True
    )

    try:
        preds, stdev = res
    except ValueError:
        preds, stdev, lls_defocus = res

    if eval_sign == 'positive_only':
        ys = np.abs(ys)
        preds = np.abs(preds)[:, :ys.shape[-1]]

    residuals = ys - preds
    p = pd.DataFrame(
        np.stack([p2v[k] for k, ph, i in itertools.product(range(len(kernels)), photons, range(n_samples))]),
        columns=['aberration']
    )
    p['prediction'] = [Wavefront(i, lam_detection=gen.lam_detection).peak2valley(na=na) for i in preds]
    p['residuals'] = [Wavefront(i, lam_detection=gen.lam_detection).peak2valley(na=na) for i in residuals]
    p['photons'] = np.concatenate([photons for i in itertools.product(phi, range(n_samples))])

    df = df.append(p, ignore_index=True)

    if savepath is not None:
        df.to_csv(f'{savepath}_predictions.csv')

    return df


@profile
def evaluate_modes(
    model: Path,
    eval_sign: str = 'signed',
    batch_size: int = 512,
    digital_rotations: bool = True
):
    outdir = model.with_suffix('') / eval_sign / 'evalmodes'
    outdir.mkdir(parents=True, exist_ok=True)
    modelspecs = backend.load_metadata(model)

    photons = [1, 1000, 10000, 50000, 100000, 200000, 400000, 600000, 800000, 1000000]
    labels = ['1', '$10^3$', '$10^4$', '$5 \\times 10^4$', '$10^5$', '$2 \\times 10^5$', '$4 \\times 10^5$', '$6 \\times 10^5$', '$8 \\times 10^5$', '$10^6$']
    waves = np.arange(1e-5, .6, step=.05).round(2)
    aberrations = np.zeros((len(waves), modelspecs.n_modes))

    for i in range(3, modelspecs.n_modes):
        if i == 4:
            continue

        savepath = outdir / f"m{i}"

        classes = aberrations.copy()
        classes[:, i] = waves
        df = eval_object(
            phi=classes,
            photons=photons,
            modelpath=model,
            batch_size=batch_size,
            eval_sign=eval_sign,
            savepath=savepath,
            digital_rotations=361 if digital_rotations else None
        )

        bins = np.arange(0, 10.25, .25)
        df['bins'] = pd.cut(df['aberration'], bins, labels=bins[1:], include_lowest=True)
        means = pd.pivot_table(df, values='residuals', index='bins', columns='photons', aggfunc=np.mean)
        means = means.sort_index().interpolate()
        logger.info(means)

        fig = plt.figure(figsize=(8, 8))
        gs = fig.add_gridspec(4, 4)
        ax_xy = fig.add_subplot(gs[0, 0])
        ax_xz = fig.add_subplot(gs[0, 1])
        ax_yz = fig.add_subplot(gs[0, 2])
        ax_wavevfront = fig.add_subplot(gs[0, -1])
        ax = fig.add_subplot(gs[1:, :])

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

        contours = ax.contourf(
            np.arange(len(photons)),
            means.index.values,
            means.values,
            cmap=cmap,
            levels=levels,
            extend='max',
            linewidths=2,
            linestyles='dashed',
        )
        ax.patch.set(hatch='/', edgecolor='lightgrey', lw=.01)

        cax = fig.add_axes([1.01, 0.08, 0.03, 0.7])
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

        cbar.ax.set_ylabel(rf'Residuals; average peak-to-valley ($\lambda = {int(modelspecs.lam_detection*1000)}~nm$)')
        cbar.ax.set_title(r'$\lambda$')
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.yaxis.set_label_position('left')

        ax.set_xlabel(f'Integrated photons')
        ax.set_xticks(np.arange(len(photons)))
        ax.set_xticklabels(labels)

        ax.set_ylabel('Initial aberration (average peak-to-valley)')
        ax.set_yticks(np.arange(0, 6, .5), minor=True)
        ax.set_yticks(np.arange(0, 6, 1))
        ax.set_ylim(.25, 5)

        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

        phi = np.zeros_like(classes[-1, :])
        phi[i] = .2
        gen = backend.load_metadata(model, psf_shape=(64, 64, 64))
        w = Wavefront(phi, lam_detection=gen.lam_detection)
        kernel = gen.single_psf(
            phi=w,
            normed=True,
            meta=False,
        )
        ax_xy.imshow(np.max(kernel, axis=0)**.5, vmin=0, vmax=1, cmap='hot')
        ax_xz.imshow(np.max(kernel, axis=1)**.5, vmin=0, vmax=1, cmap='hot')
        ax_yz.imshow(np.max(kernel, axis=2)**.5, vmin=0, vmax=1, cmap='hot')
        ax_wavevfront.imshow(w.wave(size=100), vmin=-1, vmax=1, cmap='Spectral_r')

        for a, t in zip([ax_xy, ax_xz, ax_yz, ax_wavevfront], ['XY', 'XZ', 'YZ', 'Wavefront']):
            a.axis('off')
            a.set_title(t)

        plt.tight_layout()

        plt.savefig(f'{savepath}.pdf', bbox_inches='tight', pad_inches=.25)
        plt.savefig(f'{savepath}.png', dpi=300, bbox_inches='tight', pad_inches=.25)
        plt.savefig(f'{savepath}.svg', dpi=300, bbox_inches='tight', pad_inches=.25)
