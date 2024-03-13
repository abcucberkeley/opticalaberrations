from functools import partial

import matplotlib

matplotlib.use('Agg')

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

plt.set_loglevel('error')

from pathlib import Path
from typing import Optional, Any
import numpy as np

import pandas as pd
import seaborn as sns
from tifffile import imwrite
from line_profiler_pycharm import profile

from scipy.optimize import curve_fit
from skimage.feature import blob_dog, blob_log, peak_local_max
from skimage.morphology import extrema
from skimage.filters import window
from matplotlib.ticker import PercentFormatter

import utils
import vis
from preprocessing import remove_background_noise, resize_with_crop_or_pad

import logging

logger = logging.getLogger('')

try:
	import cupy as cp
except ImportError as e:
	logging.warning(f"Cupy not supported on your system: {e}")


def plot_detections(
	image: np.ndarray,
	detections: pd.DataFrame,
	save_path: Path,
	axial_voxel_size: float,
	lateral_voxel_size: float,
	kde_color='grey',
	cdf_color='k',
	hist_color='lightgrey',
):
	num_peaks_detected = detections.shape[0]
	median = np.median(detections['sigma'])
	mean = np.mean(detections['sigma'])
	values, counts = np.unique(detections['sigma'].round(1).values, return_counts=True)
	mode = values[counts.argmax()]
	logger.info(rf"$\sigma$: {mean=:.2f}, {median=:.2f}, {mode=:.2f}")
	
	fig, axes = plt.subplots(3, 1, figsize=(11, 8))
	
	vis.plot_mip(
		vol=image,
		xy=axes[0],
		xz=axes[1],
		yz=None,
		dxy=lateral_voxel_size,
		dz=axial_voxel_size,
		cmap='gray',
		colorbar=False,
		aspect=None
	)
	axes[0].set_ylabel(r'Input (MIP) [$\gamma$=.5]')
	axes[1].set_ylabel(r'Input (MIP) [$\gamma$=.5]')
	
	sns.scatterplot(
		ax=axes[0],
		data=detections,
		x=detections.x,
		y=detections.y,
		hue=detections.sigma,
		size=detections.fwhm,
		sizes=(5, 15),
		legend=False,
		palette='magma'
	)
	
	sns.scatterplot(
		ax=axes[1],
		data=detections,
		x=detections.x,
		y=detections.z,
		hue=detections.sigma,
		size=detections.fwhm,
		sizes=(5, 15),
		legend=False,
		palette='magma'
	)
	
	axes[-1].scatter([0], [0], label=f'POIs={num_peaks_detected}', color='grey')
	axes[-1].axvline(mean, c='C0', ls=':', lw=2, label=f'Mean={mean:.2f}', zorder=3)
	axes[-1].axvline(median, c='C1', ls='--', lw=2, label=f'Median={median:.2f}', zorder=3)
	axes[-1].axvline(mode, c='C2', ls=':', lw=2, label=f'Mode={mode:.2f}', zorder=3)
	
	ax1t = axes[-1].twinx()
	ax1t = sns.histplot(
		ax=ax1t,
		data=detections,
		x='sigma',
		stat='percent',
		kde=True,
		bins=50,
		color=hist_color,
		element="step",
	)
	ax1t.lines[0].set_color(kde_color)
	ax1t.tick_params(axis='y', labelcolor=kde_color, color=kde_color)
	ax1t.set_ylabel('KDE', color=kde_color)
	ax1t.set_ylim(0, 15)
	ax1t.set_xlim(0, detections.sigma.max())
	ax1t.yaxis.set_major_formatter(PercentFormatter())
	
	ax1 = sns.histplot(
		ax=axes[-1],
		data=detections,
		x='sigma',
		stat='proportion',
		color=cdf_color,
		bins=50,
		element="poly",
		fill=False,
		cumulative=True,
		label='CDF'
	)
	
	ax1.tick_params(axis='y', labelcolor=cdf_color, color=cdf_color)
	ax1.set_ylabel('CDF', color=cdf_color)
	ax1.set_ylim(0, 1)
	ax1.set_yticks(np.arange(0, 1.2, .2))
	ax1.set_xlim(detections.sigma.min(), detections.sigma.max())
	ax1.set_xlabel(r"$\sigma$")
	ax1.grid(True, which="both", axis='both', lw=.25, ls='--', zorder=0)
	
	cdf = plt.Line2D([0], [0], label='KDE', color=kde_color)
	handles, labels = axes[-1].get_legend_handles_labels()
	handles.extend([cdf])
	axes[-1].legend(handles=handles, frameon=False, ncol=1)
	
	vis.savesvg(fig, save_path)


@profile
def measure_sigma(
	peak: np.ndarray,
	image: np.ndarray,
	axial_voxel_size: float,
	lateral_voxel_size: float,
	meshgrid: Optional[np.ndarray] = None,
	window_size: tuple = (11, 11, 11),
	plot: Optional[Path] = None,
):
	def gauss_3d(meshgrid, amplitude, zc, yc, xc, background, sigma):
		zz, yy, xx = meshgrid
		exp = ((xx - xc) ** 2 + (yy - yc) ** 2 + (zz - zc) ** 2) / (2 * sigma ** 2)
		g = amplitude * np.exp(-exp) + background
		return g.ravel()
	
	fov_start = [
		peak[s] - window_size[s] if peak[s] >= window_size[s] else 0
		for s in range(3)
	]
	fov_end = [
		peak[s] + window_size[s] + 1 if peak[s] + window_size[s] < image.shape[s] else image.shape[s]
		for s in range(3)
	]
	
	half_width = [w // 2 for w in window_size]
	fov = image[fov_start[0]:fov_end[0], fov_start[1]:fov_end[1], fov_start[2]:fov_end[2]]
	fov *= window(('tukey', 1.25), fov.shape)
	
	psf = resize_with_crop_or_pad(fov, crop_shape=window_size, mode='constant')
	psf /= np.sum(psf)
	
	if meshgrid is None:
		zz = np.linspace(0, psf.shape[0], psf.shape[0])
		yy = np.linspace(0, psf.shape[1], psf.shape[1])
		xx = np.linspace(0, psf.shape[2], psf.shape[2])
		zgrid, ygrid, xgrid = np.meshgrid(zz, yy, xx, indexing="ij")
	else:
		zgrid, ygrid, xgrid = meshgrid
	
	bg = np.median(psf)
	amp = np.max(psf) - bg
	
	try:
		popt, pcov = curve_fit(
			gauss_3d,
			(zgrid, ygrid, xgrid),
			psf.ravel(),
			p0=[amp, *half_width, bg, 1],
		)
		
		amplitude, zc, yc, xc, background, sigma = popt
		amplitude_err, zc_err, yc_err, xc_err, background_err, sigma_err = np.sqrt(np.diag(pcov))
		
		if plot is not None:
			outdir = Path(plot)
			outdir.mkdir(exist_ok=True, parents=True)
			
			filename = f"{outdir}/z{peak[0]}-y{peak[1]}-x{peak[2]}"
			
			fitted = gauss_3d((zgrid, ygrid, xgrid), *popt)
			fitted = fitted.reshape(*psf.shape)
			fitted /= np.sum(fitted)
			
			imwrite(f"{filename}_fov.tif", fov.astype(np.float32), compression='deflate', dtype=np.float32)
			imwrite(f"{filename}_psf.tif", psf.astype(np.float32), compression='deflate', dtype=np.float32)
			imwrite(f"{filename}_estimated.tif", fitted.astype(np.float32), compression='deflate', dtype=np.float32)
			
			fig, axes = plt.subplots(3, 3, figsize=(11, 8))
			vis.plot_mip(
				vol=fov,
				xy=axes[0, 0],
				xz=axes[0, 1],
				yz=axes[0, 2],
				dxy=lateral_voxel_size,
				dz=axial_voxel_size,
				label=r'FOV (MIP) [$\gamma$=.5]',
				cmap='hot',
				colorbar=True
			)
			vis.plot_mip(
				vol=psf,
				xy=axes[1, 0],
				xz=axes[1, 1],
				yz=axes[1, 2],
				dxy=lateral_voxel_size,
				dz=axial_voxel_size,
				label=r'PSF (MIP) [$\gamma$=.5]',
				cmap='hot',
				colorbar=True
			)
			
			vis.plot_mip(
				vol=fitted,
				xy=axes[-1, 0],
				xz=axes[-1, 1],
				yz=axes[-1, 2],
				dxy=lateral_voxel_size,
				dz=axial_voxel_size,
				label=r'Fit (MIP) [$\gamma$=.5]',
				cmap='hot',
				colorbar=True
			)
			
			axes[0, 1].set_title(f"$\sigma$={sigma}")
			
			vis.savesvg(fig, Path(f"{filename}.svg"))
		
		return [sigma, sigma_err]
	except RuntimeError:
		return [-1, -1]


def find_laplacian_of_gaussian_blobs(
	image: np.ndarray,
	min_sigma: float = 0.,
	max_sigma: float = 5,
	num_sigma: int = 20,
	overlap: float = .05,
	exclude_border: int = 11,
):
	blobs = blob_log(
		image,
		min_sigma=min_sigma,
		max_sigma=max_sigma,
		num_sigma=num_sigma,
		threshold_rel=0.25,
		overlap=overlap,
		exclude_border=exclude_border,
	)
	
	df = pd.DataFrame(blobs, columns=['z', 'y', 'x', 'sigma'])
	return df


def find_difference_of_gaussians_blobs(
	image: np.ndarray,
	min_sigma: float = 0.,
	max_sigma: float = 5,
	threshold_rel: float = .25,
	overlap: float = .05,
	exclude_border: int = 11,
):
	blobs = blob_dog(
		image,
		min_sigma=min_sigma,
		max_sigma=max_sigma,
		threshold_rel=threshold_rel,
		overlap=overlap,
		exclude_border=exclude_border,
	)
	
	df = pd.DataFrame(blobs, columns=['z', 'y', 'x', 'sigma'])
	return df


def find_peak_local_max(
	image: np.ndarray,
	save_path: Path,
	axial_voxel_size: float,
	lateral_voxel_size: float,
	min_distance: int = 5,
	threshold_rel: float = .25,
	window_size: tuple = (11, 11, 11),
	plot_gaussian_fits: bool = True,
	exclude_border: int = 11,
	cpu_workers: int = -1,
	max_num_peaks: int = 100,
):
	zz = np.linspace(0, window_size[0], window_size[0])
	yy = np.linspace(0, window_size[1], window_size[1])
	xx = np.linspace(0, window_size[2], window_size[2])
	meshgrid = np.meshgrid(zz, yy, xx, indexing="ij")
	
	detected_peaks = peak_local_max(
		image,
		min_distance=min_distance,
		threshold_rel=threshold_rel,
		exclude_border=exclude_border,
		num_peaks=max_num_peaks,
		p_norm=2,
	).astype(int)
	
	df = pd.DataFrame(detected_peaks, columns=['z', 'y', 'x'])
	num_peaks_detected = df.shape[0]
	
	estimate_sigma = partial(
		measure_sigma,
		image=image,
		axial_voxel_size=axial_voxel_size,
		lateral_voxel_size=lateral_voxel_size,
		meshgrid=meshgrid,
		window_size=window_size,
		plot=Path(f"{save_path}_gaussian_fits") if plot_gaussian_fits else None
	)
	
	results = utils.multiprocess(
		func=estimate_sigma,
		jobs=df.values,
		desc=f"Estimating sigma for [{df.shape[0]}] peaks",
		cores=cpu_workers
	)
	df['sigma'] = results[..., 0]
	df['perr'] = results[..., -1]
	
	# drop detections with high error
	df = df[(df.perr < 1) & (df.perr > -1)]
	df = df[df.sigma > 0]
	df.sort_values('sigma', inplace=True)
	
	logger.info(
		f"Dropped [{num_peaks_detected - df.shape[0]}] detections with high error"
	)
	return df


def find_peaks(
	image: np.ndarray,
	save_path: Path,
	axial_voxel_size: float,
	lateral_voxel_size: float,
	window_size: tuple = (11, 11, 11),
	h_maxima_threshold: Any = None,
	exclude_border: int = 11,
	plot_gaussian_fits: bool = True,
	cpu_workers: int = -1,
):
	zz = np.linspace(0, window_size[0], window_size[0])
	yy = np.linspace(0, window_size[1], window_size[1])
	xx = np.linspace(0, window_size[2], window_size[2])
	meshgrid = np.meshgrid(zz, yy, xx, indexing="ij")
	
	h = np.percentile(image, 95) if h_maxima_threshold is None else h_maxima_threshold
	h_maxima = extrema.h_maxima(image, h=h)
	df = pd.DataFrame(np.transpose(np.nonzero(h_maxima)), columns=['z', 'y', 'x'])
	num_peaks_detected = df.shape[0]
	logger.info(f"Found [{num_peaks_detected}] candidates using threshold {h=:.2f}")
	
	# drop peaks too close to the edge
	lzedge = df['z'] >= window_size[0] // exclude_border
	hzedge = df['z'] <= image.shape[0] - window_size[0] // exclude_border
	lyedge = df['y'] >= window_size[1] // exclude_border
	hyedge = df['y'] <= image.shape[1] - window_size[1] // exclude_border
	lxedge = df['x'] >= window_size[2] // exclude_border
	hxedge = df['x'] <= image.shape[2] - window_size[2] // exclude_border
	df = df[lzedge & hzedge & lyedge & hyedge & lxedge & hxedge]
	
	logger.info(
		f"Dropped {num_peaks_detected - df.shape[0]}/{num_peaks_detected} detections "
		f"because they're too close to the edge [{exclude_border=}]"
	)
	num_peaks_detected = df.shape[0]
	
	estimate_sigma = partial(
		measure_sigma,
		image=image,
		axial_voxel_size=axial_voxel_size,
		lateral_voxel_size=lateral_voxel_size,
		meshgrid=meshgrid,
		window_size=window_size,
		plot=Path(f"{save_path}_gaussian_fits") if plot_gaussian_fits else None
	)
	
	results = utils.multiprocess(
		func=estimate_sigma,
		jobs=df.values,
		desc=f"Estimating sigma for [{df.shape[0]}] peaks",
		cores=cpu_workers
	)
	df['sigma'] = results[..., 0]
	df['perr'] = results[..., -1]
	
	# drop detections with high error
	df = df[(df.perr < 1) & (df.perr > -1)]
	df = df[df.sigma > 0]
	df.sort_values('sigma', inplace=True)
	
	logger.info(
		f"Dropped {num_peaks_detected - df.shape[0]}/{num_peaks_detected} detections with high error"
	)
	return df


def detect_peaks(
	image: np.ndarray,
	save_path: Path,
	axial_voxel_size: float = .200,
	lateral_voxel_size: float = .097,
	plot: bool = False,
	plot_gaussian_fits: bool = False,
	remove_background: bool = True,
	cpu_workers: int = -1,
	window_size: tuple = (11, 11, 11),
	h_maxima_threshold: Any = None,
	exclude_border: int = 11,
	method: str = 'find_peaks',
):
	if remove_background:
		image = remove_background_noise(image, method='difference_of_gaussians')
		if isinstance(image, cp.ndarray):
			image = image.get()
	
	if method == 'find_laplacian_of_gaussian_blobs':
		df = find_laplacian_of_gaussian_blobs(
			image,
			min_sigma=0,
			max_sigma=5,
			num_sigma=20,
			overlap=.05,
			exclude_border=exclude_border,
		)
	
	elif method == 'find_difference_of_gaussians_blobs':
		df = find_difference_of_gaussians_blobs(
			image,
			min_sigma=0,
			max_sigma=5,
			threshold_rel=0.25,
			overlap=.05,
			exclude_border=exclude_border,
		)
	
	elif method == 'find_peak_local_max':
		df = find_peak_local_max(
			image=image,
			save_path=save_path,
			axial_voxel_size=axial_voxel_size,
			lateral_voxel_size=lateral_voxel_size,
			plot_gaussian_fits=plot_gaussian_fits,
			window_size=window_size,
			exclude_border=exclude_border,
			cpu_workers=cpu_workers,
		)
	
	else:
		df = find_peaks(
			image=image,
			save_path=save_path,
			axial_voxel_size=axial_voxel_size,
			lateral_voxel_size=lateral_voxel_size,
			plot_gaussian_fits=plot_gaussian_fits,
			h_maxima_threshold=h_maxima_threshold,
			window_size=window_size,
			exclude_border=exclude_border,
			cpu_workers=cpu_workers,
		)
	
	df['fwhm'] = df.sigma.apply(utils.sigma2fwhm)
	df['sigma_lateral_nm'] = df.sigma * lateral_voxel_size * 1000
	df['sigma_axial_nm'] = df.sigma * axial_voxel_size * 1000
	print(df[['sigma', 'fwhm', 'sigma_lateral_nm', 'sigma_axial_nm']].describe(
		percentiles=[.5, .75, .85, .9, .95, .99]
	))
	
	if plot:
		plot_detections(
			image=image,
			detections=df,
			save_path=Path(f"{save_path}_gaussian_fit.svg"),
			axial_voxel_size=axial_voxel_size,
			lateral_voxel_size=lateral_voxel_size
		)
	
	df.index.name = 'id'
	df.to_csv(f"{save_path}_gaussian_fit.csv")
	
	return df
