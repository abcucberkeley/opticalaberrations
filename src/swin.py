import logging
import sys
from abc import ABC
from functools import partial

import numpy as np
from tensorflow.keras import layers

from VideoSwin.videoswin.blocks import SwinStage
from VideoSwin.videoswin.layers import TFPatchEmbed3D, TFPatchMerging
from base import Base

logging.basicConfig(
	stream=sys.stdout,
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Swin(Base, ABC):
	def __init__(
		self,
		hidden_size=128,
		patches=(1, 4, 4),
		window_size=(1, 8, 8),
		heads=(2, 2, 18, 2),
		repeats=(4, 8, 16, 32),
		expand_rate=4,
		in_chans=1,
		qkv_bias=True,
		qk_scale=None,
		dropout_rate=0.,
		attn_drop_rate=0.,
		drop_path_rate=.1,
		**kwargs
	):
		super().__init__(**kwargs)
		self.hidden_size = hidden_size
		self.patches = patches
		self.window_size = window_size
		self.in_chans= in_chans
		self.heads = heads
		self.repeats = repeats
		self.mlp_ratio = expand_rate
		self.num_layers = len(repeats)
		self.in_chans = in_chans
		self.qkv_bias = qkv_bias
		self.qk_scale = qk_scale
		self.dropout_rate = dropout_rate
		self.attn_drop_rate = attn_drop_rate
		self.drop_path_rate = drop_path_rate
		self.dpr = np.linspace(0., self.drop_path_rate, sum(self.repeats)).tolist()
		self.avg_pool = layers.GlobalAveragePooling3D()

	def call(self, inputs, **kwargs):
		
		m = TFPatchEmbed3D(
			patch_size=self.patches,
			embed_dim=self.hidden_size,
			norm_layer=partial(layers.LayerNormalization, epsilon=1e-05),
		)(inputs)
		
		for i in range(self.num_layers):
			m = SwinStage(
				dim=int(self.hidden_size * 2 ** i),
				depth=self.repeats[i],
				num_heads=self.heads[i],
				window_size=self.window_size,
				mlp_ratio=self.mlp_ratio,
				qkv_bias=self.qkv_bias,
				qk_scale=self.qk_scale,
				drop=self.dropout_rate,
				attn_drop=self.attn_drop_rate,
				drop_path=self.dpr[sum(self.repeats[:i]): sum(self.repeats[:i + 1])],
				downsample=TFPatchMerging if i < self.num_layers - 1 else None,
				norm_layer=partial(layers.LayerNormalization, epsilon=1e-05)
			)(m)
		
		m = self.avg_pool(m)
		return self.regressor(m)
