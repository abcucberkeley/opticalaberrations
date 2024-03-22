import logging
import sys
from abc import ABC

import numpy as np
import tensorflow as tf
from scipy.special import binom
from tensorflow.keras import layers

from base import Base
from roi import ROI

logging.basicConfig(
	stream=sys.stdout,
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Patchify(layers.Layer):
	def __init__(self, patch_size, num_patches, **kwargs):
		super().__init__(**kwargs)
		self.patch_size = patch_size
		self.num_patches = num_patches
		self.project = layers.Conv3D(
			filters=self.patch_size ** 2,
			kernel_size=(1, self.patch_size, self.patch_size),
			strides=(1, self.patch_size, self.patch_size),
			padding="VALID",
			name="conv_projection",
		)
		self.flat = layers.Reshape(
			target_shape=(-1, self.num_patches, self.patch_size ** 2),
			name="flatten_projection",
		)

		self.prenorm = layers.LayerNormalization(axis=-1, epsilon=1e-6)
	
	def build(self, input_shape):
		super(Patchify, self).build(input_shape)
	
	def get_config(self):
		config = super(Patchify, self).get_config()
		config.update({
			"patch_size": self.patch_size,
			"num_patches": self.num_patches,
		})
		return config
	
	def call(self, inputs, **kwargs):
		patches = self.project(inputs)
		patches = self.flat(patches)
		patches = layers.Reshape((inputs.shape[1], -1, patches.shape[-1]))(patches)
		patches = self.prenorm(patches)
		return patches


class PatchEncoder(layers.Layer):
	def __init__(
		self,
		num_patches,
		embedding_size,
		positional_encoding_scheme='default',
		radial_encoding_periods=1,
		radial_encoding_nth_order=4,
		**kwargs
	):
		super().__init__(**kwargs)
		self.num_patches = num_patches
		self.embedding_size = embedding_size
		self.positional_encoding_scheme = positional_encoding_scheme
		self.radial_encoding_periods = radial_encoding_periods
		self.radial_encoding_nth_order = radial_encoding_nth_order
		
		self.project_layer = layers.Dense(self.embedding_size)
		self.radial_embedding_layer = layers.Dense(self.embedding_size)
		self.positional_embedding_layer = layers.Embedding(
			input_dim=self.num_patches,
			output_dim=self.embedding_size
		)
	
	def build(self, input_shape):
		super(PatchEncoder, self).build(input_shape)
	
	def get_config(self):
		config = super(PatchEncoder, self).get_config()
		config.update({
			"num_patches": self.num_patches,
			"embedding_size": self.embedding_size,
			"positional_encoding_scheme": self.positional_encoding_scheme,
			"radial_encoding_periods": self.radial_encoding_periods,
			"radial_encoding_nth_order": self.radial_encoding_nth_order,
		})
		return config
	
	def _calc_radius(self):
		grid_size = int(np.sqrt(self.num_patches))
		d = np.linspace(-1 + 1 / grid_size, 1 - 1 / grid_size, grid_size, dtype=np.float32)
		ygrid, xgrid = np.meshgrid(d, d, indexing='ij')
		r = np.sqrt(ygrid.flatten() ** 2 + xgrid.flatten() ** 2)
		theta = np.arctan2(ygrid.flatten(), xgrid.flatten())
		return r.astype(np.float32), theta.astype(np.float32)
	
	def _nm_polynomial(self, n, m, rho, theta, normed=True):
		def _nm_normalization(n, m):
			""" return orthonormal zernike """
			return np.sqrt((1. + (m == 0)) / (2. * n + 2))
		
		if (n - m) % 2 == 1:
			poly = 0 * rho + 0 * theta
			return poly.astype(np.float32)
		
		radial = 0
		m0 = abs(m)
		
		for k in range((n - m0) // 2 + 1):
			a = binom(n - k, k)
			b = binom(n - 2 * k, (n - m0) // 2 - k)
			radial += (-1.) ** k * a * b * rho ** (n - 2 * k)
		
		# no clipping needed here
		# radial *= (rho <= 1.)
		
		if normed:  # return orthonormal zernike
			prefac = 1. / _nm_normalization(n, m)
		else:
			prefac = 1.
		
		if m >= 0:
			poly = prefac * radial * np.cos(m0 * theta)
		else:
			poly = prefac * radial * np.sin(m0 * theta)
		
		return poly.astype(np.float32)
	
	def _zernike_polynomials(self, radial_encoding_nth_order=4):
		r, theta = self._calc_radius()
		nm_pairs = set((n, m) for n in range(radial_encoding_nth_order + 1) for m in range(-n, n + 1, 2))
		polynomials = np.zeros((r.shape[0], len(nm_pairs)), dtype=np.float32)
		
		for i, (pr, pt) in enumerate(zip(r, theta)):
			for j, (n, m) in enumerate(nm_pairs):
				polynomials[i, j] = self._nm_polynomial(n=n, m=m, rho=pr, theta=pt, normed=True)
		
		return tf.constant(polynomials)
	
	def _fourier_decomposition(self, periods=1):
		r, theta = self._calc_radius()
		r = tf.constant(r, dtype=tf.float32)
		theta = tf.constant(theta, dtype=tf.float32)
		
		encodings = [r]
		for p in range(1, periods + 1):
			encodings.append(tf.sin(p * r))
			encodings.append(tf.cos(p * r))
			encodings.append(tf.sin(p * theta))
			encodings.append(tf.cos(p * theta))
		
		return tf.stack(encodings, axis=-1)
	
	def _power_decomposition(self, periods=1, radial_encoding_nth_order=4):
		r, theta = self._calc_radius()
		r = tf.constant(r, dtype=tf.float32)
		theta = tf.constant(theta, dtype=tf.float32)
		
		encodings = []
		for n in range(1, radial_encoding_nth_order + 1):
			encodings.append(tf.pow(r, n))
		
		for p in range(1, periods + 1):
			encodings.append(tf.sin(p * theta))
			encodings.append(tf.cos(p * theta))
		
		return tf.stack(encodings, axis=-1)
	
	def _rotational_symmetry(self, periods=1):
		r, theta = self._calc_radius()
		r = tf.constant(r, dtype=tf.float32)
		theta = tf.constant(theta, dtype=tf.float32)
		
		encodings = [r]
		for p in range(1, periods + 1):
			encodings.append(tf.sin(p * theta))
			encodings.append(tf.cos(p * theta))
		
		return tf.stack(encodings, axis=-1)
	
	def _patch_number(self):
		return tf.range(start=0, limit=self.num_patches, delta=1)
	
	def positional_encoding(self, inputs, scheme, periods, radial_encoding_nth_order):
		
		if scheme == 'rotational_symmetry' or scheme == 'rot_sym':
			pos = self._rotational_symmetry(periods=periods)
		
		elif scheme == 'fourier_decomposition':
			pos = self._fourier_decomposition(periods=periods)
		
		elif scheme == 'power_decomposition':
			pos = self._power_decomposition(periods=periods, radial_encoding_nth_order=radial_encoding_nth_order)
		
		elif scheme == 'zernike_polynomials':
			pos = self._zernike_polynomials(radial_encoding_nth_order=radial_encoding_nth_order)
		
		else:
			pos = self._patch_number()
		
		emb = []
		for _ in range(inputs.shape[1]):
			if scheme is None or scheme == 'default':
				emb.append(self.positional_embedding_layer(pos))
			else:
				emb.append(self.radial_embedding_layer(pos))
		
		return tf.stack(emb, axis=0)
	
	def call(
		self,
		inputs,
		periods=None,
		zernike_nth_order=None,
		scheme=None,
		**kwargs
	):
		linear_projections = self.project_layer(inputs)
		
		if scheme is not None:
			self.positional_encoding_scheme = scheme
			self.radial_encoding_periods = periods
			self.radial_encoding_nth_order = zernike_nth_order
		
		positional_embeddings = self.positional_encoding(
			inputs,
			scheme=self.positional_encoding_scheme,
			periods=self.radial_encoding_periods,
			radial_encoding_nth_order=self.radial_encoding_nth_order,
		)
		
		return linear_projections + positional_embeddings


class MLP(layers.Layer):
	def __init__(
		self,
		expand_rate,
		dropout_rate,
		activation,
		**kwargs
	):
		super().__init__(**kwargs)
		self.expand_rate = expand_rate
		self.dropout_rate = dropout_rate
		self.activation = activation
	
	def build(self, input_shape):
		super(MLP, self).build(input_shape)
		self.expand = layers.Dense(int(self.expand_rate * input_shape[-1]), activation=self.activation)
		self.proj = layers.Dense(input_shape[-1])
		self.dropout1 = layers.Dropout(self.dropout_rate)
		self.dropout2 = layers.Dropout(self.dropout_rate)
	
	def get_config(self):
		config = super(MLP, self).get_config()
		config.update({
			"expand_rate": self.expand_rate,
			"dropout_rate": self.dropout_rate,
			"activation": self.activation,
		})
		return config
	
	def call(self, inputs, **kwargs):
		x = self.expand(inputs)
		x = self.dropout1(x)
		x = self.proj(x)
		x = self.dropout2(x)
		return x


class Transformer(layers.Layer):
	def __init__(
		self,
		heads,
		dims,
		activation,
		dropout_rate,
		expand_rate=4,
		**kwargs
	):
		super().__init__(**kwargs)
		self.heads = heads
		self.dims = dims
		self.activation = activation
		self.dropout_rate = dropout_rate
		self.expand_rate = expand_rate
		
		self.dropout = layers.Dropout(self.dropout_rate)
		self.prenorm1 = layers.LayerNormalization(axis=-1, epsilon=1e-6)
		self.prenorm2 = layers.LayerNormalization(axis=-1, epsilon=1e-6)
		self.msa = layers.MultiHeadAttention(
			num_heads=self.heads,
			key_dim=self.dims,
			dropout=self.dropout_rate
		)
		self.mlp = MLP(
			expand_rate=self.expand_rate,
			dropout_rate=self.dropout_rate,
			activation=self.activation
		)
	
	def build(self, input_shape):
		super(Transformer, self).build(input_shape)
	
	def get_config(self):
		config = super(Transformer, self).get_config()
		config.update({
			"heads": self.heads,
			"dims": self.dims,
			"dropout_rate": self.dropout_rate,
			"expand_rate": self.expand_rate,
			"activation": self.activation,
		})
		return config
	
	def call(self, inputs, **kwargs):
		ln1 = self.prenorm1(inputs)
		att = self.msa(ln1, ln1)
		att = self.dropout(att)
		s1 = layers.Add()([inputs, att])
		
		ln2 = self.prenorm2(s1)
		s2 = self.mlp(ln2)
		return layers.Add()([s1, s2])


class OpticalTransformer(Base, ABC):
	def __init__(
		self,
		roi=None,
		patches=16,
		heads=(16),
		repeats=(24),
		depth_scalar=1.0,
		width_scalar=1.0,
		activation='gelu',
		dropout_rate=0.1,
		expand_rate=4,
		rho=.05,
		mul=False,
		no_phase=False,
		radial_encoding_period=1,
		positional_encoding_scheme='default',
		radial_encoding_nth_order=4,
		fixed_dropout_depth=False,
		stem=False,
		**kwargs
	):
		super().__init__(**kwargs)
		self.roi = roi
		self.stem = stem
		self.patch_size = patches[0]
		self.heads = heads
		self.repeats = repeats
		self.depth_scalar = depth_scalar
		self.width_scalar = width_scalar
		self.activation = activation
		self.dropout_rate = dropout_rate
		self.expand_rate = expand_rate
		self.avg = layers.GlobalAvgPool2D()
		self.rho = rho
		self.mul = mul
		self.no_phase = no_phase
		self.radial_encoding_period = radial_encoding_period
		self.positional_encoding_scheme = positional_encoding_scheme
		self.radial_encoding_nth_order = radial_encoding_nth_order
		self.fixed_dropout_depth = fixed_dropout_depth
	
	def _calc_channels(self, channels, width_scalar):
		return int(tf.math.ceil(width_scalar * channels))
	
	def _calc_repeats(self, repeats, depth_scalar):
		return int(tf.math.ceil(depth_scalar * repeats))
	
	def call(self, inputs, **kwargs):
		
		m = inputs
		
		if self.roi is not None:
			m = ROI(crop_shape=self.roi)(m)
		
		img_shape = self.roi[-1] if self.roi is not None else inputs.shape[-2]
		num_patches = (img_shape // self.patch_size) ** 2
		
		m = Patchify(patch_size=self.patch_size, num_patches=num_patches)(m)
		
		m = PatchEncoder(
			num_patches=num_patches,
			embedding_size=self._calc_channels(self.patch_size ** 2, width_scalar=self.width_scalar),
			positional_encoding_scheme=self.positional_encoding_scheme,
			radial_encoding_periods=self.radial_encoding_period,
			radial_encoding_nth_order=self.radial_encoding_nth_order
		)(m)
		
		for i, (h, r) in enumerate(zip(self.heads, self.repeats)):
			res = m
			
			for j in range(self._calc_repeats(r, depth_scalar=self.depth_scalar)):
				if self.fixed_dropout_depth:
					dropout_rate = self.dropout_rate
				else:
					dropout_rate = self.dropout_rate * (i + 1) / sum(self.repeats)
				
				m = Transformer(
					heads=self._calc_channels(h, width_scalar=self.width_scalar),
					dims=64,
					activation=self.activation,
					dropout_rate=dropout_rate,
					expand_rate=self.expand_rate,
				)(m)
			
			if len(self.repeats) > 1:
				m = layers.add([res, m])
		
		m = self.avg(m)
		return self.regressor(m)
