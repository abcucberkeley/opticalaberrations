import logging
import sys
from abc import ABC
from functools import partial

import tensorflow as tf
from tensorflow.keras import layers

from VideoSwin.videoswin.blocks import TFBasicLayer
from VideoSwin.videoswin.layers import TFPatchEmbed3D, TFPatchMerging
from base import Base
from roi import ROI

logging.basicConfig(
	stream=sys.stdout,
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



class Swin(Base, ABC):
	def __init__(
		self,
		roi=None,
		hidden_size=128,
		patches=(1, 8, 8),
		mini_patches=(1, 4, 4),
		heads=(2, 2, 18, 2),
		repeats=(4, 8, 16, 32),
		depth_scalar=1.0,
		width_scalar=1.0,
		activation='gelu',
		expand_rate=4,
		rho=.05,
		mul=False,
		no_phase=False,
		in_chans=1,
		qkv_bias=True,
		qk_scale=None,
		dropout_rate=0.,
		attn_drop_rate=0.,
		drop_path_rate=0.2,
		**kwargs
	):
		super().__init__(**kwargs)
		self.hidden_size = hidden_size
		self.roi = roi
		self.patch_size = patches
		self.mini_patches=mini_patches
		self.heads = heads
		self.repeats = repeats
		self.depth_scalar = depth_scalar
		self.width_scalar = width_scalar
		self.activation = activation
		self.mlp_ratio = expand_rate
		self.rho = rho
		self.mul = mul
		self.no_phase = no_phase
		self.num_layers = len(repeats)
		self.in_chans = in_chans
		self.qkv_bias=qkv_bias
		self.qk_scale=qk_scale
		self.dropout_rate=dropout_rate
		self.attn_drop_rate=attn_drop_rate
		self.drop_path_rate=drop_path_rate
		
		self.patch_embed = TFPatchEmbed3D(
			patch_size=self.mini_patches,
			embed_dim=self.hidden_size,
			norm_layer=partial(layers.LayerNormalization, epsilon=1e-05),
		)
		self.pos_drop = layers.Dropout(dropout_rate)
		self.dpr = tf.linspace(0., self.drop_path_rate, sum(self.repeats)).numpy().tolist()
		self.avg_pool = layers.GlobalAveragePooling3D()
		
		self.blocks = []
		for i in range(self.num_layers):
			layer = TFBasicLayer(
				dim=int(self.hidden_size * 2 ** i),
				depth=self.repeats[i],
				num_heads=self.heads[i],
				window_size=self.patch_size,
				mlp_ratio=self.mlp_ratio,
				qkv_bias=self.qkv_bias,
				qk_scale=self.qk_scale,
				drop=self.dropout_rate,
				attn_drop=self.attn_drop_rate,
				drop_path=self.dpr[sum(self.repeats[:i]): sum(self.repeats[: i + 1])],
				downsample=TFPatchMerging if ( i <= self.num_layers - 1) else None,
				norm_layer=partial(layers.LayerNormalization, epsilon=1e-05)
			)
			self.blocks.append(layer)
	
	def call(self, inputs, **kwargs):
		
		m = inputs
		
		if self.roi is not None:
			m = ROI(crop_shape=self.roi)(m)
		
		m = self.patch_embed(m)
		m = self.pos_drop(m)
		
		for i in range(self.num_layers):
			m = self.blocks[i](m)
		
		m = self.avg_pool(m)
		return self.regressor(m)
