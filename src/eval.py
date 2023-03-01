import matplotlib
matplotlib.use('Agg')

from multiprocessing import Pool

import logging
import sys
from functools import partial
from pathlib import Path
from typing import Any
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as mcolors

import matplotlib.pyplot as plt
plt.set_loglevel('error')

import swifter
import numpy as np
import pandas as pd
import tensorflow as tf
from preprocessing import remove_background_noise
from line_profiler_pycharm import profile
from tqdm import tqdm
from tifffile import imsave

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
def simulate_beads(psf, gen, snr, object_size=0, num_objs=1, beads=None, noise=None):
    if beads is None:
        beads = multipoint_dataset.beads(
            gen=gen,
            object_size=object_size,
            num_objs=num_objs,
            radius=.1
        )

    img = utils.fftconvolution(sample=beads, kernel=psf)
    img *= snr ** 2

    if noise is None:
        noise = gen._random_noise(
            image=img,
            mean=gen.mean_background_noise,
            sigma=gen.sigma_background_noise
        )

    noisy_img = noise + img
    noisy_img = remove_background_noise(noisy_img)
    noisy_img /= np.max(noisy_img)
    return noisy_img


def generate_fourier_embeddings(
    image_id: int,
    data: pd.DataFrame,
    psfgen: SyntheticPSF,
    no_phase: bool = False,
    input_coverage: float = 1.0,
    digital_rotations: Any = None,
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
        noise=False,
        meta=False,
    )

    noisy_img = simulate_beads(
        psf=psf,
        beads=ref,
        gen=psfgen,
        snr=hashtable['snr']
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
    snr_range: tuple = (21, 30),
    eval_sign: str = 'positive_only',
    digital_rotations: bool = False,
    rotations: np.ndarray = np.arange(0, 360 + 1, 1).astype(int),
    savepath: Any = None,
):
    model = backend.load(modelpath)
    gen = backend.load_metadata(
        modelpath,
        snr=1000,
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
        snr_range=snr_range,
        metadata=True
    )
    # this runs multiple samples (aka images) at a time.
    # ys is a 2D array, rows are each sample, columns give aberration in zernike coeffs
    metadata = np.array(list(metadata.take(-1)))
    ys = np.array([i.numpy() for i in metadata[:, 0]])[:, :predicted_modes]
    snrs = np.array([i.numpy() for i in metadata[:, 1]])
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
        'snr': snrs,  # signal-to-noise
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
        current['snr'] = snrs
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
def plot_heatmap(means, wavelength, savepath, label=f'Peak signal-to-noise ratio', lims=(0, 100)):
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

    cbar.ax.set_ylabel(rf'Average peak-to-valley residuals ($\lambda = {int(wavelength * 1000)}~nm$)')
    cbar.ax.set_title(r'$\lambda$')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('left')

    ax.set_xlabel(label)
    ax.set_xlim(lims)
    ax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

    ax.set_ylabel(
        'Average peak-to-valley aberration'
        rf'($\lambda = {int(wavelength * 1000)}~nm$)'
    )
    ax.set_yticks(np.arange(0, 6, .5), minor=True)
    ax.set_yticks(np.arange(0, 6, 1))
    ax.set_ylim(.25, 5)

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
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
            snr_range=(0, 100),
            eval_sign=eval_sign,
            digital_rotations=digital_rotations
        )

    bins = np.arange(0, 10.25, .25)
    df['bins'] = pd.cut(df['aberration'], bins, labels=bins[1:], include_lowest=True)

    means = pd.pivot_table(
        df[df['niter'] == niter], values='residuals', index='bins', columns='snr', aggfunc=np.mean
    )
    means = means.sort_index().interpolate()

    logger.info(means)
    means.to_csv(f'{savepath}.csv')

    plot_heatmap(
        means,
        wavelength=modelspecs.lam_detection,
        savepath=savepath,
        label=f'Peak signal-to-noise ratio',
        lims=(0, 100)
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
            means = pd.pivot_table(df, values=f'z{z}_residual', index=f'z{z}_bins', columns='snr', aggfunc=np.mean)
            means = means.sort_index().interpolate()
            logger.info(f"z{z}")
            logger.info(means)
            means.to_csv(savepath.with_name(f"{savepath.name}_z{z}.csv"))

            plot_heatmap(
                means,
                wavelength=modelspecs.lam_detection,
                savepath=savepath.with_name(f"{savepath.name}_z{z}"),
                label=f'Peak signal-to-noise ratio',
                lims=(0, 100)
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
    snr_range: tuple = (21, 30),
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
            snr_range=snr_range,
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
    snr_range: tuple = (21, 30),
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
            snr_range=snr_range,
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
    psnr: int = 30,
    batch_size: int = 128,
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
                snr=1000,
                amplitude_ranges=amplitude_range,
                distribution=dist,
                signed=False if eval_sign == 'positive_only' else True,
                rotate=True,
                mode_weights='pyramid',
                psf_shape=(64, 64, 64),
                mean_background_noise=0,
            )
            for s in range(10):
                for num_objs in tqdm([1, 2, 5, 25, 50, 100, 150]):
                    reference = multipoint_dataset.beads(
                        gen=gen,
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
                    psf, y, snr, maxcounts, lls_defocus_offset = gen.single_psf(
                        phi=phi,
                        normed=True,
                        noise=False,
                        meta=True,
                    )

                    img = utils.fftconvolution(sample=reference, kernel=psf)
                    img *= psnr ** 2

                    rand_noise = gen._random_noise(
                        image=img,
                        mean=0,
                        sigma=gen.sigma_background_noise
                    )
                    noisy_img = rand_noise + img
                    maxcounts = np.max(noisy_img)
                    noisy_img /= maxcounts

                    save_path = Path(
                        f"{model.with_suffix('')}/{eval_sign}/samples/{dist}/um-{amplitude_range[-1]}/num_objs-{num_objs:02d}"
                    )
                    save_path.mkdir(exist_ok=True, parents=True)

                    if digital_rotations:
                        res = backend.predict_rotation(
                            m,
                            noisy_img,
                            psfgen=gen,
                            no_phase=no_phase,
                            batch_size=batch_size,
                            plot=save_path / f'{s}',
                            plot_rotations=save_path / f'{s}',
                        )
                    else:
                        res = backend.bootstrap_predict(
                            m,
                            noisy_img,
                            psfgen=gen,
                            no_phase=no_phase,
                            batch_size=batch_size,
                            plot=save_path / f'{s}',
                        )

                    try:
                        p, std = res
                    except ValueError:
                        p, std, lls_defocus = res

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

                    p_psf = gen.single_psf(p_wave, normed=True, noise=True)
                    gt_psf = gen.single_psf(y_wave, normed=True, noise=True)

                    corrected_psf = gen.single_psf(residuals)
                    corrected_noisy_img = utils.fftconvolution(sample=reference, kernel=corrected_psf)
                    corrected_noisy_img *= psnr ** 2
                    corrected_noisy_img = rand_noise + corrected_noisy_img
                    corrected_noisy_img /= np.max(corrected_noisy_img)

                    imsave(save_path / f'psf_{s}.tif', noisy_img)
                    imsave(save_path / f'corrected_psf_{s}.tif', corrected_psf)

                    task = partial(
                        vis.diagnostic_assessment,
                        psf=noisy_img,
                        gt_psf=gt_psf,
                        predicted_psf=p_psf,
                        corrected_psf=corrected_noisy_img,
                        psnr=psnr,
                        maxcounts=maxcounts,
                        y=y_wave,
                        pred=p_wave,
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
    na: float = 1.0,
    batch_size: int = 100,
    snr_range: tuple = (21, 30),
    n_samples: int = 10,
    eval_sign: str = 'positive_only',
    savepath: Any = None,
):
    model = backend.load(modelpath)
    gen = backend.load_metadata(modelpath, psf_shape=3*[model.input_shape[2]])
    no_phase = True if model.input_shape[1] == 3 else False

    w = Wavefront(phi, lam_detection=gen.lam_detection)
    p2v = w.peak2valley(na=na)
    df = pd.DataFrame([], columns=['aberration', 'prediction', 'residuals', 'object_size'])

    kernel = gen.single_psf(
        phi=w,
        normed=True,
        noise=False,
        meta=False,
    )

    k = np.where(phi > 0)[0]
    for isize in tqdm([0, 1, 2, 3, 4, 5], desc=f"Evaluate different sizes [mode #{k}]"):
        psnr = gen._randuniform(snr_range)
        inputs = np.array([
            simulate_beads(
                psf=kernel,
                gen=gen,
                object_size=isize,
                num_objs=1,
                snr=psnr,
            )
            for i in range(n_samples)
        ])[..., np.newaxis]
        ys = np.array([phi for i in inputs])

        if eval_sign == 'rotations':
            res = backend.predict_rotation(
                model,
                inputs,
                psfgen=gen,
                batch_size=batch_size,
                no_phase=no_phase,
                cpu_workers=1
            )
        else:
            res = backend.bootstrap_predict(
                model,
                inputs,
                psfgen=gen,
                batch_size=batch_size,
                n_samples=1,
                no_phase=no_phase,
                cpu_workers=1
            )

        try:
            preds, stdev = res
        except ValueError:
            preds, stdev, lls_defocus = res

        if eval_sign == 'positive_only':
            ys = np.abs(ys)
            if len(preds.shape) > 1:
                preds = np.abs(preds)[:, :ys.shape[-1]]
            else:
                preds = np.abs(preds)[np.newaxis, :ys.shape[-1]]
        else:
            if len(preds.shape) > 1:
                preds = preds[:, :ys.shape[-1]]
            else:
                preds = preds[np.newaxis, :ys.shape[-1]]

        residuals = ys - preds

        p = pd.DataFrame([p2v for i in inputs], columns=['aberration'])
        p['prediction'] = [Wavefront(i, lam_detection=gen.lam_detection).peak2valley(na=na) for i in preds]
        p['residuals'] = [Wavefront(i, lam_detection=gen.lam_detection).peak2valley(na=na) for i in residuals]
        p['object_size'] = 1 if isize == 0 else isize * 2

        for z in range(preds.shape[-1]):
            p[f'z{z}_ground_truth'] = ys[:, z]
            p[f'z{z}_prediction'] = preds[:, z]
            p[f'z{z}_residual'] = residuals[:, z]

        df = df.append(p, ignore_index=True)

        if savepath is not None:
            df.to_csv(f'{savepath}_predictions.csv')

    return df


@profile
def evaluate_modes(model: Path, eval_sign: str = 'positive_only'):
    outdir = model.with_suffix('') / eval_sign / 'evalmodes'
    outdir.mkdir(parents=True, exist_ok=True)
    modelspecs = backend.load_metadata(model)

    waves = np.arange(1e-5, .75, step=.05)
    aberrations = np.zeros((len(waves), modelspecs.n_modes))

    for i in range(3, modelspecs.n_modes):
        if i == 4:
            continue

        savepath = outdir / f"m{i}"

        classes = aberrations.copy()
        classes[:, i] = waves
        job = partial(eval_object, modelpath=model, eval_sign=eval_sign, savepath=savepath)
        preds = utils.multiprocess(job, list(classes), cores=-1)
        df = pd.DataFrame([]).append(preds, ignore_index=True)

        bins = np.arange(0, 10.25, .25)
        df['bins'] = pd.cut(df['aberration'], bins, labels=bins[1:], include_lowest=True)

        means = pd.pivot_table(df, values='residuals', index='bins', columns='object_size', aggfunc=np.mean)
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

        cbar.ax.set_ylabel(rf'Average peak-to-valley residuals ($\lambda = 510~nm$)')
        cbar.ax.set_title(r'$\lambda$')
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.yaxis.set_label_position('left')

        ax.set_xlabel(f'Diameter of the simulated object (pixels)')
        ax.set_xlim(1, 10)
        ax.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)

        ax.set_ylabel(
            'Average peak-to-valley aberration'
            rf' ($\lambda = 510~nm$)'
        )
        ax.set_yticks(np.arange(0, 6, .5), minor=True)
        ax.set_yticks(np.arange(0, 6, 1))
        ax.set_ylim(.25, 5)

        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        phi = np.zeros_like(classes[-1, :])
        phi[i] = .2
        gen = backend.load_metadata(model, psf_shape=(64, 64, 64))
        w = Wavefront(phi, lam_detection=gen.lam_detection)
        kernel = gen.single_psf(
            phi=w,
            normed=True,
            noise=False,
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
