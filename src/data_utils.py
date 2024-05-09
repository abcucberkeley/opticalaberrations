import logging
import os
import sys
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import ujson
from line_profiler_pycharm import profile
from tifffile import TiffFile
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

import embeddings
import vis
from preprocessing import resize_with_crop_or_pad
from utils import multiprocess
from wavefront import Wavefront
from zernike import Zernike

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@profile
def get_image(path):
    if isinstance(path, torch.Tensor):
        path = Path(str(path.numpy(), "utf-8"))
    else:
        path = Path(str(path))
    
    if path.suffix == '.npy':
        with np.load(path) as arr:
            img = arr
    
    elif path.suffix == '.npz':
        with np.load(path) as data:
            img = data['arr_0']
    
    elif path.suffix == '.tif':
        with TiffFile(path) as tif:
            img = tif.asarray()
    else:
        raise Exception(f"Unknown file format {path.suffix}")
    
    if np.isnan(np.sum(img)):
        logger.error("NaN!")
    
    if img.shape[-1] != 1:  # add a channel dim
        img = np.expand_dims(img, axis=-1)
    
    return img.astype(np.float32)


def get_metadata(path, codename: str):
    try:
        if isinstance(path, torch.Tensor):
            path = Path(str(path.numpy(), "utf-8"))
        else:
            path = Path(str(path))
        
        with open(path.with_suffix('.json')) as f:
            hashtbl = ujson.load(f)
        
        return hashtbl[codename]
    
    except KeyError:
        return None


@profile
def get_sample(
    path,
    no_phase=False,
    metadata=False,
    input_coverage=1.0,
    embedding_option='spatial_planes',
    iotf=None,
    lls_defocus: bool = False,
    defocus_only: bool = False
):
    if isinstance(path, torch.Tensor):
        path = Path(str(path.numpy(), "utf-8"))
    else:
        path = Path(str(path))
    
    with open(path.with_suffix('.json')) as f:
        hashtbl = ujson.load(f)
    
    npoints = int(hashtbl.get('npoints', 1))
    photons = hashtbl.get('photons', 0)
    counts = hashtbl.get('counts', 0)
    counts_mode = hashtbl.get('counts_mode', 0)
    counts_percentiles = hashtbl.get('counts_percentiles', np.zeros(100))
    
    lls_defocus_offset = np.nan_to_num(hashtbl.get('lls_defocus_offset', 0), nan=0)
    avg_min_distance = np.nan_to_num(hashtbl.get('avg_min_distance', 0), nan=0)
    
    if defocus_only:
        zernikes = [lls_defocus_offset]
    else:
        zernikes = hashtbl['zernikes']
        
        if lls_defocus:
            zernikes.append(lls_defocus_offset)
    
    zernikes = np.array(zernikes).astype('float32')
    
    try:
        umRMS = hashtbl['umRMS']
    except KeyError:
        umRMS = np.linalg.norm(hashtbl['zernikes'])
    
    try:
        p2v = hashtbl['peak2peak']
    except KeyError:
        p2v = Wavefront(zernikes, lam_detection=float(hashtbl['wavelength'])).peak2valley()
    
    if metadata:
        return zernikes, photons, counts, counts_mode, counts_percentiles, p2v, umRMS, npoints, avg_min_distance, str(
            path)
    
    else:
        img = get_image(path).astype('float32')
        
        if input_coverage != 1.:
            img = resize_with_crop_or_pad(img, crop_shape=[int(s * input_coverage) for s in img.shape])
        
        if img.shape[0] == img.shape[1] and iotf is not None:
            img = embeddings.fourier_embeddings(
                img,
                iotf=iotf,
                padsize=None,
                alpha_val='abs',
                phi_val='angle',
                remove_interference=True,
                embedding_option=embedding_option,
            )
        
        if no_phase and img.shape[0] == 6:
            img = img[:3]
            wave = Wavefront(zernikes)
            
            for i, a in enumerate(zernikes):
                mode = Zernike(i)
                twin = Zernike((mode.n, mode.m * -1))
                
                if mode.index_ansi > twin.index_ansi:
                    continue
                else:
                    if mode.m != 0 and wave.zernikes.get(twin) is not None:
                        if np.sign(a) == -1:
                            zernikes[mode.index_ansi] *= -1
                            zernikes[twin.index_ansi] *= -1
                    else:
                        zernikes[i] = np.abs(a)
        
        return img, zernikes


@profile
def check_sample(path):
    try:
        with open(path.with_suffix('.json')) as f:
            ujson.load(f)
        
        with TiffFile(path) as tif:
            tif.asarray()
        return 1
    
    except Exception as e:
        logger.warning(f"Corrupted file {path}: {e}")
        return path


@profile
def check_criteria(
    file,
    distribution='/',
    embedding='',
    modes=-1,
    max_amplitude=1.,
    photons_range=None,
    npoints_range=None,
):
    path = str(file)
    amp = float(str([s for s in file.parts if s.startswith('amp_')][0]).split('-')[-1].replace('p', '.'))
    photons = tuple(map(int, str([s.strip('photons_') for s in file.parts if s.startswith('photons_')][0]).split('-')))
    npoints = int([s.strip('npoints_') for s in file.parts if s.startswith('npoints')][0])
    modes = '' if modes - 1 else str(modes)
    
    if 'iter' not in path \
            and (distribution == '/' or distribution in path) \
            and embedding in path \
            and f"z{modes}" in path \
            and amp <= max_amplitude \
            and ((npoints_range[0] <= npoints <= npoints_range[1]) if npoints_range is not None else True) \
            and (
    (photons_range[0] <= photons[0] and photons[1] <= photons_range[1]) if photons_range is not None else True) \
            and check_sample(file) == 1:  # access file system only after everything else has passed.
        return path
    else:
        return None


@profile
def collect_files(
    datadir,
    samplelimit=None,
    distribution='/',
    embedding='',
    modes=-1,
    max_amplitude=1.,
    photons_range=None,
    npoints_range=None,
    filename_pattern: str = r"*[!_gt|!_realspace|!_noisefree|!_predictions_psf|!_corrected_psf|!_reconstructed_psf].tif",
    shuffle=True,
    cpu_workers: int = -1,
):
    if not Path(datadir).exists():
        raise Exception(f"The 'datadir' does not exist: {datadir}")
    s1 = f'Searching for files that meet:'
    s2 = f'npoints_range=({int(npoints_range[0]):,} to {int(npoints_range[1]):,} objects),' if npoints_range is not None else ""
    s3 = f'photons_range=({int(photons_range[0]):,} to {int(photons_range[1]):,} photons),' if photons_range is not None else ""
    s4 = f'{max_amplitude=}, number of {modes=}.'
    s5 = f'In data directory: {Path(datadir).resolve()} which exists={Path(datadir).exists()}'
    logger.info(" ".join([s1, s2, s3, s4]))
    logger.info(s5)
    
    check = partial(
        check_criteria,
        distribution=distribution,
        embedding=embedding,
        modes=modes,
        max_amplitude=max_amplitude,
        photons_range=photons_range,
        npoints_range=npoints_range,
    )
    candidate_files = sorted(Path(datadir).rglob(filename_pattern))
    files = multiprocess(
        func=check,
        jobs=candidate_files,
        cores=cpu_workers,
        desc='Loading dataset hashtable',
        unit=' .tif candidates checked'
    )
    try:
        files = [f for f in files if f is not None]
        logger.info(f'.tif files that meet criteria: {len(files)} files')
    except TypeError:
        raise Exception(f'No files that meet criteria out of {len(candidate_files)} candidate files, '
                        f'{sum(len(files) for _, _, files in os.walk(datadir))} total files, '
                        f'in data directory: {Path(datadir).resolve()} which exists={Path(datadir).exists()}')
    
    if samplelimit is not None:
        files = np.random.choice(files, min(samplelimit, len(files)), replace=False).tolist()
        logger.info(f'.tif files selected ({samplelimit=}): {len(files)} files')
    
    if shuffle:
        np.random.shuffle(files)
    
    return files


@profile
def check_dataset(
    datadir,
    filename_pattern: str = r"*[!_gt|!_realspace|!_noisefree|!_predictions_psf|!_corrected_psf|!_reconstructed_psf].tif"
):
    jobs = multiprocess(func=check_sample, jobs=sorted(Path(datadir).rglob(filename_pattern)), cores=-1)
    corrupted = [j for j in jobs if j != 1]
    corrupted = pd.DataFrame(corrupted, columns=['path'])
    logger.info(f"Corrupted files [{corrupted.index.shape[0]}]")
    print(corrupted)
    corrupted.to_csv(datadir / 'corrupted.csv', header=False, index=False)
    return corrupted


def dataset_statistics(
    datadir,
    filename_pattern: str = r"*.json",
    n_samples: int = 5000,
):
    zern2name = {
        0: "piston",
        # 1st order
        1: "tilt",
        2: "tip",
        # 2nd order
        3: "oblique astigmatism",
        4: "defocus",
        5: "vertical astigmatism",
        # 3rd order
        6: "vertical trefoil",
        7: "vertical coma",
        8: "horizontal coma",
        9: "oblique trefoil",
        # 4th order
        10: "oblique quadrafoil",  # sometimes called tetrafoil
        11: "oblique secondary astigmatism",
        12: "primary spherical",
        13: "vertical secondary astigmatism",
        14: "vertical quadrafoil",
    }
    np.set_printoptions(edgeitems=10, linewidth=100000,
                        formatter=dict(float=lambda x: "%.3g" % x))
    
    datadir = Path(datadir)
    stats = pd.DataFrame()
    logger.info(f"Finding '{filename_pattern}' files in '{datadir.stem}'...")
    files = list(Path(datadir).rglob(filename_pattern))
    logger.info(f"Found {len(files)} files. Shuffling...")
    np.random.shuffle(files)
    n_files = min(n_samples, len(files))
    logger.info(f"Loading {n_files} files...")
    data = open(files[0]).read()
    df = pd.json_normalize(ujson.loads(data))
    zern_matrix = np.vstack(df.zernikes.values)
    number_of_modes = int(zern_matrix.shape[-1])
    zern_matrix = np.full((n_files, number_of_modes), np.nan)  # init
    
    fast = True
    if fast:
        for i in tqdm(range(n_files), unit='files'):
            data = open(files[i]).read()
            zern_matrix[i] = np.vstack(pd.json_normalize(ujson.loads(data)).zernikes)
    else:
        for f in tqdm(files, unit='files'):
            # print(f)
            data = open(f).read()
            df = pd.json_normalize(ujson.loads(data))
            max_counts = df['counts_percentiles'].values[-1][-1]
            df['psnr'] = (max_counts - df['mean_background_offset']) / df['sigma_background_noise']
            
            stats = stats.append(df, ignore_index=True)
            zern_matrix = np.vstack(stats.zernikes.values)
    
    # Bar plot for number of (non-zero) samples of this mode being present in the dataset.
    fig, ax = plt.subplots(1, 1)  # type:plt.Figure, plt.Axes
    
    mean_of_all = np.mean(np.abs(zern_matrix))
    logger.info(
        f"Mean of the absolute value of all zernike amplitudes (mae if we just guessed all zeros) = {mean_of_all:.3E}")
    
    sns.barplot(np.count_nonzero(zern_matrix, axis=0), ax=ax, color='dimgrey')
    # ax.set_title(f'Path = {files[0].parent}', wrap=True)
    ax.set_ylabel(f'# of non-zero values out of {n_files}')
    ax.set_xlabel(f'Zernike mode index')
    
    output_figure = Path(datadir / 'number_of_samples_in_dataset_per_mode.svg')
    vis.savesvg(fig, f"{output_figure}")
    logger.info(f"Saved: {output_figure.resolve()}")
    
    #
    # Plot zernike distribution
    fig, ax = plt.subplots(2, 1, sharex=True)  # type:plt.Figure, plt.Axes
    output_figure = Path(datadir / 'heatmap.svg')
    sorted_zern = zern_matrix.copy()
    sorted_zern.sort(axis=0)
    ax[0].imshow(np.flip(sorted_zern, axis=0).transpose(), aspect='auto', interpolation='none')
    ax[0].set_ylim(2.5, number_of_modes - 0.5)
    ax[0].set_yticks([3, 6, 10, 11, 12])
    ax[0].grid(axis='y', linestyle='--', linewidth=0.5)
    ax[0].set_ylabel(f'Zernike')
    
    sorted_zern = np.abs(zern_matrix.copy())
    sorted_zern.sort(axis=0)
    ax[1].imshow(np.flip(sorted_zern, axis=0).transpose(), aspect='auto', interpolation='none')
    ax[1].set_ylim(2.5, number_of_modes - 0.5)
    ax[1].set_yticks([3, 6, 10, 11, 12, ])
    ax[1].grid(axis='y', linestyle='--', linewidth=0.5, which='both')
    ax[1].set_ylabel(f'Zernike (abs)')
    
    ax[1].set_xlabel(f'Sample')
    ax[1].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=zern_matrix.shape[0]))
    ax[0].set_title('Distribution of amplitudes\n(sorted along each Z mode)')
    
    vis.savesvg(fig, f"{output_figure}")
    logger.info(f"Saved: {output_figure.resolve()}")
    
    #
    # Histogram for angles per aberration.
    
    zern_degrees = pd.DataFrame()
    
    wavefront = Wavefront(zern_matrix[0])
    for row, (mode, twin) in enumerate(wavefront.twins.items()):
        if twin is not None:
            prime = zern_matrix[:, mode.index_ansi]
            twin = zern_matrix[:, twin.index_ansi]
            x = prime
            y = twin
            angle = np.rad2deg(np.arctan2(y, x)) % 360
            zern_str = zern2name[mode.index_ansi]
            zern_degrees[zern_str] = angle
        
        else:  # mode has m=0 (spherical,...), or twin isn't within the 55 modes.
            pass
    
    # plot for each:
    fig, axes = plt.subplots(zern_degrees.shape[-1], 1, sharey=True, sharex=True)  # type:plt.Figure, plt.Axes
    i = 0
    for col in range(zern_degrees.shape[-1]):
        column_name = zern_degrees.columns[col]
        data = zern_degrees[zern_degrees[column_name] != 0][column_name]
        axes[i].hist(data, bins=18)
        column_name = column_name.replace(' ', '\n')
        axes[i].set_ylabel(f"{column_name}\nTotal\n{len(data)}")
        axes[i].set_xlim(0, 360)
        i += 1
    
    axes[0].set_title(f'Angle distribution out of {n_files} Files', wrap=True)
    axes[-1].set_xlabel(f'Degrees')
    axes[-1].set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    output_figure = Path(datadir / f'1Dhisto.svg')
    
    vis.savesvg(fig, f"{output_figure}")
    logger.info(f"Saved: {output_figure.resolve()}")
    
    # fig, axs = plt.subplots(number_of_modes, 1, tight_layout=True, sharey=True, sharex=True) # type:plt.Figure, plt.Axes
    # for z in range(number_of_modes):
    #     sns.histplot(zern_matrix[:, z]!=0, kde=True, ax=axs, color='dimgrey', bins=20)
    
    # sns.ecdfplot(stats, x='psnr', ax=axs[1], stat="percent")
    # axs[1].set_ylim(0,100)
    # sns.histplot(stats['psnr'].values, kde=True, ax=axs[0], color='dimgrey', bins=20)
    # sns.histplot(stats['psnr'].values, kde=True, ax=axs[0], color='dimgrey', bins=20)
    
    # fig.tight_layout()


class DataFinder(Dataset):
    def __init__(
        self,
        datadir,
        distribution='/',
        embedding='',
        modes=-1,
        samplelimit=None,
        max_amplitude=1.,
        no_phase=False,
        input_coverage=1.,
        embedding_option='spatial_planes',
        photons_range=None,
        npoints_range=None,
        iotf=None,
        metadata=False,
        lls_defocus: bool = False,
        defocus_only: bool = False,
        filename_pattern: str = r"*[!_gt|!_realspace|!_noisefree|!_predictions_psf|!_corrected_psf|!_reconstructed_psf].tif",
        cpu_workers: int = -1,
        model_input_shape: tuple = (6, 64, 64, 1)
    ):
        super(Dataset, self).__init__()
        
        self.datadir = datadir
        self.distribution = distribution
        self.embedding = embedding
        self.modes = modes
        self.samplelimit = samplelimit
        self.max_amplitude = max_amplitude
        self.no_phase = no_phase
        self.input_coverage = input_coverage
        self.embedding_option = embedding_option
        self.photons_range = photons_range
        self.npoints_range = npoints_range
        self.iotf = iotf
        self.metadata = metadata
        self.lls_defocus = lls_defocus
        self.defocus_only = defocus_only
        self.filename_pattern = filename_pattern
        self.cpu_workers = cpu_workers
        self.model_input_shape = model_input_shape
        
        self.files = collect_files(
            datadir,
            modes=self.modes,
            samplelimit=self.samplelimit,
            embedding=self.embedding,
            distribution=self.distribution,
            max_amplitude=self.max_amplitude,
            photons_range=self.photons_range,
            npoints_range=self.npoints_range,
            filename_pattern=self.filename_pattern,
            cpu_workers=self.cpu_workers
        )
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        
        return get_sample(
            path=path,
            no_phase=self.no_phase,
            input_coverage=self.input_coverage,
            iotf=self.iotf,
            embedding_option=self.embedding_option,
            metadata=self.metadata,
            lls_defocus=self.lls_defocus,
            defocus_only=self.defocus_only
        )


@profile
def collect_dataset(
    datadir,
    split=None,
    multiplier=1,
    batch_size=1,
    distribution='/',
    embedding='',
    modes=-1,
    samplelimit=None,
    max_amplitude=1.,
    no_phase=False,
    input_coverage=1.,
    embedding_option='spatial_planes',
    photons_range=None,
    npoints_range=None,
    iotf=None,
    metadata=False,
    lls_defocus: bool = False,
    defocus_only: bool = False,
    filename_pattern: str = r"*[!_gt|!_realspace|!_noisefree|!_predictions_psf|!_corrected_psf|!_reconstructed_psf].tif",
    cpu_workers: int = -1,
    model_input_shape: tuple = (6, 64, 64, 1)
):
    """
    Returns:
        metadata=True -> (amps, photons, counts, peak2peak, umRMS, npoints, avg_min_distance, filename)
        metadata=False-> img & zern
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    dataset = DataFinder(
        datadir=datadir,
        distribution=distribution,
        embedding=embedding,
        modes=modes,
        samplelimit=samplelimit,
        max_amplitude=max_amplitude,
        no_phase=no_phase,
        input_coverage=input_coverage,
        embedding_option=embedding_option,
        photons_range=photons_range,
        npoints_range=npoints_range,
        iotf=iotf,
        metadata=metadata,
        lls_defocus=lls_defocus,
        defocus_only=defocus_only,
        filename_pattern=filename_pattern,
        cpu_workers=cpu_workers,
        model_input_shape=model_input_shape
    )
    
    if split is not None:
        val_size = round(len(dataset) * split)
        train, val = random_split(dataset, lengths=[len(dataset) - val_size, val_size])
        
        train = DataLoader(
            train,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            # num_workers=cpu_workers
        )
        val = DataLoader(
            val,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            # num_workers=cpu_workers
        )
        
        i = next(iter(train))
        logger.info(f"Input: {i[0].shape}")
        logger.info(f"Output: {i[1].shape}")
        logger.info(f"Training batches: {len(train)}")
        logger.info(f"Validation batches: {len(val)}")
        
        return train, val
    
    else:
        
        data = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            # num_workers=cpu_workers,
            # sampler=DistributedSampler(dataset)
        )
        
        if not metadata:
            i = next(iter(data))
            logger.info(f"Input: {i[0].shape}")
            logger.info(f"Output: {i[1].shape}")
            logger.info(f"Batches: {len(data)}")
        
        return data


def dataset_statistics(
        datadir,
        filename_pattern: str = r"*.json",
        n_samples: int = 5000,
        ):

    zern2name = {
        0: "piston",
        # 1st order
        1: "tilt",
        2: "tip",
        # 2nd order
        3: "oblique astigmatism",
        4: "defocus",
        5: "vertical astigmatism",
        # 3rd order
        6: "vertical trefoil",
        7: "vertical coma",
        8: "horizontal coma",
        9: "oblique trefoil",
        # 4th order
        10: "oblique quadrafoil",  # sometimes called tetrafoil
        11: "oblique secondary astigmatism",
        12: "primary spherical",
        13: "vertical secondary astigmatism",
        14: "vertical quadrafoil",
    }
    np.set_printoptions(edgeitems=10, linewidth=100000,
                        formatter=dict(float=lambda x: "%.3g" % x))

    datadir = Path(datadir)
    stats = pd.DataFrame()
    logger.info(f"Finding '{filename_pattern}' files in '{datadir.stem}'...")
    files = list(Path(datadir).rglob(filename_pattern))
    logger.info(f"Found {len(files)} files. Shuffling...")
    np.random.shuffle(files)
    n_files = min(n_samples, len(files))
    logger.info(f"Loading {n_files} files...")
    data = open(files[0]).read()
    df = pd.json_normalize(ujson.loads(data))
    zern_matrix = np.vstack(df.zernikes.values)
    number_of_modes = int(zern_matrix.shape[-1])
    zern_matrix = np.full((n_files, number_of_modes), np.nan) # init

    fast = True
    if fast:
        for i in tqdm(range(n_files), unit='files'):
            data = open(files[i]).read()
            zern_matrix[i] = np.vstack(pd.json_normalize(ujson.loads(data)).zernikes)
    else:
        for f in tqdm(files, unit='files'):
            # print(f)
            data = open(f).read()
            df = pd.json_normalize(ujson.loads(data))
            max_counts = df['counts_percentiles'].values[-1][-1]
            df['psnr'] = (max_counts - df['mean_background_offset']) / df['sigma_background_noise']

            stats = stats.append(df, ignore_index=True)
            zern_matrix = np.vstack(stats.zernikes.values)

    # Bar plot for number of (non-zero) samples of this mode being present in the dataset.
    fig, ax = plt.subplots(1, 1) # type:plt.Figure, plt.Axes

    mean_of_all = np.mean(np.abs(zern_matrix))
    logger.info(f"Mean of the absolute value of all zernike amplitudes (mae if we just guessed all zeros) = {mean_of_all:.3E}")

    sns.barplot(np.count_nonzero(zern_matrix, axis=0), ax=ax, color='dimgrey')
    # ax.set_title(f'Path = {files[0].parent}', wrap=True)
    ax.set_ylabel(f'# of non-zero values out of {n_files}')
    ax.set_xlabel(f'Zernike mode index')

    output_figure = Path(datadir / 'number_of_samples_in_dataset_per_mode.svg')
    vis.savesvg(fig, f"{output_figure}")
    logger.info(f"Saved: {output_figure.resolve()}")

    #
    # Plot zernike distribution
    fig, ax = plt.subplots(2, 1, sharex=True)  # type:plt.Figure, plt.Axes
    output_figure = Path(datadir / 'heatmap.svg')
    sorted_zern = zern_matrix.copy()
    sorted_zern.sort(axis=0)
    ax[0].imshow(np.flip(sorted_zern, axis=0).transpose(), aspect='auto', interpolation='none')
    ax[0].set_ylim(2.5, number_of_modes-0.5)
    ax[0].set_yticks([3,6,10,11,12])
    ax[0].grid(axis='y', linestyle='--', linewidth=0.5)
    ax[0].set_ylabel(f'Zernike')

    sorted_zern = np.abs(zern_matrix.copy())
    sorted_zern.sort(axis=0)
    ax[1].imshow(np.flip(sorted_zern, axis=0).transpose(), aspect='auto', interpolation='none')
    ax[1].set_ylim(2.5, number_of_modes-0.5)
    ax[1].set_yticks([3, 6, 10, 11, 12,])
    ax[1].grid(axis='y',  linestyle='--', linewidth=0.5, which='both')
    ax[1].set_ylabel(f'Zernike (abs)')

    ax[1].set_xlabel(f'Sample')
    ax[1].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=zern_matrix.shape[0]))
    ax[0].set_title('Distribution of amplitudes\n(sorted along each Z mode)')

    vis.savesvg(fig, f"{output_figure}")
    logger.info(f"Saved: {output_figure.resolve()}")

    #
    # Histogram for angles per aberration.

    zern_degrees = pd.DataFrame()

    wavefront = Wavefront(zern_matrix[0])
    for row, (mode, twin) in enumerate(wavefront.twins.items()):
        if twin is not None:
            prime = zern_matrix[:, mode.index_ansi]
            twin = zern_matrix[:, twin.index_ansi]
            x = prime
            y = twin
            angle = np.rad2deg(np.arctan2(y, x)) % 360
            zern_str = zern2name[mode.index_ansi]
            zern_degrees[zern_str] = angle

        else:  # mode has m=0 (spherical,...), or twin isn't within the 55 modes.
            pass

    # plot for each:
    fig, axes = plt.subplots(zern_degrees.shape[-1], 1, sharey=True, sharex=True)  # type:plt.Figure, plt.Axes
    i=0
    for col in range(zern_degrees.shape[-1]):

        column_name = zern_degrees.columns[col]
        data = zern_degrees[zern_degrees[column_name] != 0][column_name]
        axes[i].hist(data, bins=18)
        column_name = column_name.replace(' ', '\n')
        axes[i].set_ylabel(f"{column_name}\nTotal\n{len(data)}")
        axes[i].set_xlim(0, 360)
        i += 1

    axes[0].set_title(f'Angle distribution out of {n_files} Files', wrap=True)
    axes[-1].set_xlabel(f'Degrees')
    axes[-1].set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    output_figure = Path(datadir / f'1Dhisto.svg')

    vis.savesvg(fig,  f"{output_figure}")
    logger.info(f"Saved: {output_figure.resolve()}")

    # fig, axs = plt.subplots(number_of_modes, 1, tight_layout=True, sharey=True, sharex=True) # type:plt.Figure, plt.Axes
    # for z in range(number_of_modes):
    #     sns.histplot(zern_matrix[:, z]!=0, kde=True, ax=axs, color='dimgrey', bins=20)

    # sns.ecdfplot(stats, x='psnr', ax=axs[1], stat="percent")
    # axs[1].set_ylim(0,100)
    # sns.histplot(stats['psnr'].values, kde=True, ax=axs[0], color='dimgrey', bins=20)
    # sns.histplot(stats['psnr'].values, kde=True, ax=axs[0], color='dimgrey', bins=20)

    # fig.tight_layout()