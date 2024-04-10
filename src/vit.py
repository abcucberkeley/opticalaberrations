import logging
import sys
from abc import ABC

import numpy as np
import tensorflow as tf
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
	def __init__(self, patch_size, hidden_size, **kwargs):
		super().__init__(**kwargs)
		self.patch_size = patch_size
		self.hidden_size = hidden_size
		self.project = layers.Conv3D(
			filters=self.hidden_size,
			kernel_size=(1, self.patch_size, self.patch_size),
			strides=(1, self.patch_size, self.patch_size),
			padding="VALID",
			name="conv_projection",
		)
		
		self.prenorm = layers.LayerNormalization(axis=-1, epsilon=1e-6)
	
	def build(self, input_shape):
		super(Patchify, self).build(input_shape)
	
	def get_config(self):
		config = super(Patchify, self).get_config()
		config.update({
			"patch_size": self.patch_size,
			"hidden_size": self.hidden_size,
		})
		return config
	
	def call(self, inputs, **kwargs):
		patches = self.project(inputs)
		patches = self.prenorm(patches)
		patches = layers.Reshape((inputs.shape[1], -1, patches.shape[-1]))(patches)
		return patches


class PatchEncoder(layers.Layer):
	def __init__(
		self,
		num_patches,
		embedding_size,
		positional_encoding_scheme='default',
		**kwargs
	):
		super().__init__(**kwargs)
		self.num_patches = num_patches
		self.embedding_size = embedding_size
		self.positional_encoding_scheme = positional_encoding_scheme
		
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
		})
		return config
	
	def _calc_radius(self):
		grid_size = int(np.sqrt(self.num_patches))
		d = np.linspace(-1 + 1 / grid_size, 1 - 1 / grid_size, grid_size, dtype=np.float32)
		ygrid, xgrid = np.meshgrid(d, d, indexing='ij')
		r = np.sqrt(ygrid.flatten() ** 2 + xgrid.flatten() ** 2)
		theta = np.arctan2(ygrid.flatten(), xgrid.flatten())
		return r.astype(np.float32), theta.astype(np.float32)
	
	def _rotational_symmetry(self):
		r, theta = self._calc_radius()
		r = tf.constant(r, dtype=tf.float32)
		theta = tf.constant(theta, dtype=tf.float32)
		return tf.stack([r, tf.sin(theta), tf.cos(theta)], axis=-1)
	
	def _patch_number(self):
		return tf.range(start=0, limit=self.num_patches, delta=1)
	
	def positional_encoding(self, inputs, scheme):
		
		if scheme == 'rotational_symmetry' or scheme == 'rot_sym':
			pos = self._rotational_symmetry()
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
		scheme=None,
		**kwargs
	):
		linear_projections = self.project_layer(inputs)
		
		if scheme is not None:
			self.positional_encoding_scheme = scheme
		
		positional_embeddings = self.positional_encoding(inputs, scheme=self.positional_encoding_scheme)
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


class VIT(Base, ABC):
	def __init__(
		self,
		roi=None,
		hidden_size=768,
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
		positional_encoding_scheme='default',
		fixed_dropout_depth=False,
		stem=False,
		**kwargs
	):
		super().__init__(**kwargs)
		self.hidden_size = hidden_size
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
		self.positional_encoding_scheme = positional_encoding_scheme
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
		
		m = Patchify(patch_size=self.patch_size, hidden_size=self.hidden_size)(m)
		
		m = PatchEncoder(
			num_patches=num_patches,
			embedding_size=self._calc_channels(self.hidden_size, width_scalar=self.width_scalar),
			positional_encoding_scheme=self.positional_encoding_scheme,
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
					dims=self.hidden_size,
					activation=self.activation,
					dropout_rate=dropout_rate,
					expand_rate=self.expand_rate,
				)(m)
			
			if len(self.repeats) > 1:
				m = layers.add([res, m])
		
		m = self.avg(m)
		return self.regressor(m)
