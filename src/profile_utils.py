import matplotlib
matplotlib.use('Agg')

import time
import sys
import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.profiler.model_analyzer import profile as tf_profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
from tbparse import SummaryReader

try:
	from keras import backend as K
except:
	from tensorflow.keras import backend as K

try:
	import cupy as cp
except ImportError as e:
	logging.warning(f"Cupy not supported on your system: {e}")


logging.basicConfig(
	stream=sys.stdout,
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



def measure_throughput(model: tf.keras.Model, number_of_samples=10*1024, batch_size=512):
	
	dataset = tf.data.Dataset.from_tensor_slices(
		np.random.random((number_of_samples, *model.input_shape[1:])).astype('float32')
	)
	dataset = dataset.cache()
	dataset = dataset.batch(batch_size)
	dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
	options = tf.data.Options()
	options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
	dataset = dataset.with_options(options)
	
	
	start_time = time.time()
	model.predict(dataset)
	timeit = time.time() - start_time
	predictions_per_second = int(round(number_of_samples / timeit, 0))
	
	logger.info(f"Runtime for {number_of_samples} samples [{batch_size=}]")
	logger.info(f"TF backend: {predictions_per_second:.0f} prediction/sec.")
	return predictions_per_second

def count_transformer_blocks(model: tf.keras.Model):
	counter = {}

	for l in model.layers:
		layer_type = l.__class__.__name__
		
		if layer_type.lower() == 'transformer':
			patch_size = int(np.sqrt(l.output_shape[-1]))
			
			if counter.get(f"p{patch_size}") is None:
				counter[f"p{patch_size}"] = 1
			else:
				counter[f"p{patch_size}"] += 1
	logger.info(counter)
	return counter


def measure_memory_usage(model: tf.keras.Model, batch_size: int = 1):
	""" https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model """
	
	shapes_mem_count = 0
	sub_model_mem_count = 0
	
	for l in model.layers:
		layer_type = l.__class__.__name__
		
		if layer_type == 'Model':
			sub_model_mem_count += measure_memory_usage(batch_size, l)
		
		single_layer_mem = 1
		out_shape = l.output_shape
		
		if type(out_shape) is list:
			out_shape = out_shape[0]
		
		for s in out_shape:
			if s is None:
				continue
			single_layer_mem *= s
		
		shapes_mem_count += single_layer_mem
	
	trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
	non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])
	
	if K.floatx() == 'float16':
		number_size = 2.0
	elif K.floatx() == 'float32':
		number_size = 4.0
	elif K.floatx() == 'float64':
		number_size = 8.0
	else:
		number_size = 4.0
	
	total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
	gbytes = np.round(total_memory / (1024.0 ** 3), 3) + sub_model_mem_count
	logger.info(f"{model.name}: {gbytes} GB")
	return gbytes


def measure_gflops(model: tf.keras.Model):
	graph = tf.function(
		model.call,
		input_signature=[tf.TensorSpec(shape=(1, *model.input_shape[1:]))]
	).get_concrete_function().graph
	
	flops = tf_profile(graph, options=ProfileOptionBuilder.float_operation()).total_float_ops
	
	gflops = np.round(flops / 1e9, 3)
	logger.info(f"{flops:,} FLOPs = {gflops} GFLOPs")
	return gflops


def load_tf_logs(path):
	reader = SummaryReader(str(path), pivot=True, extra_columns=set(['wall_time']))
	df = reader.tensors
	return df


def matmul_flops(m, n, k):
    return m * (2 * n - 1) * k

def softmax_flops(n, d):
    return n * (3 * d - 1)

def layernorm_flops(n, d):
    flops = n * d - 1  # mean
    flops += 3 * n * d - 1  # variance
    flops += 2 * n * d  # normalize
    return flops


def self_attention_flops(num_tokens, input_dim, output_dim):
	flops = 3 * matmul_flops(num_tokens, input_dim, output_dim)  # Q, K and V
	flops += matmul_flops(num_tokens, output_dim, num_tokens)  # Q*K^T
	flops += num_tokens ** 2
	flops += softmax_flops(num_tokens, num_tokens)
	flops += matmul_flops(num_tokens, num_tokens, output_dim)  # softmax
	return flops

def msa_flops(num_tokens, embed_dim, heads):
    sa_flops = self_attention_flops(
        num_tokens, embed_dim, embed_dim // heads
    )
    flops = heads * sa_flops
    flops += matmul_flops(num_tokens, embed_dim, embed_dim)
    return flops

def mlp_flops(num_tokens, embed_dim, mlp_dim):
    flops = matmul_flops(num_tokens, embed_dim, mlp_dim) # expand layer
    flops += num_tokens * mlp_dim  # activation
    flops += matmul_flops(num_tokens, mlp_dim, embed_dim)  # project layer
    return flops


def encoder_flops(num_tokens, embed_dim, heads, mlp_dim):
	flops = layernorm_flops(n=num_tokens, d=embed_dim)
	flops += msa_flops(num_tokens, embed_dim, heads)
	flops += num_tokens * embed_dim   # res connection
	
	flops += layernorm_flops(n=num_tokens, d=embed_dim)
	flops += mlp_flops(num_tokens, embed_dim, mlp_dim)
	flops += num_tokens * embed_dim   # res connection
	return flops


def decoder_flops(num_tokens, embed_dim, heads, mlp_dim):
	flops = layernorm_flops(n=num_tokens, d=embed_dim)
	flops += msa_flops(num_tokens, embed_dim, heads)
	flops += num_tokens * embed_dim  # res connection
	
	flops += layernorm_flops(n=num_tokens, d=embed_dim)
	flops += msa_flops(num_tokens, embed_dim, heads) # second msa
	flops += num_tokens * embed_dim  # res connection
	
	flops += layernorm_flops(n=num_tokens, d=embed_dim)
	flops += mlp_flops(num_tokens, embed_dim, mlp_dim)
	flops += num_tokens * embed_dim  # res connection
	return flops


def patchify_flops(num_tokens, patch_size, embed_dim):
	latent_dim = np.product(patch_size)
	flops = matmul_flops(num_tokens, latent_dim, embed_dim)
	flops += num_tokens * embed_dim  # Add positional embedding
	return flops


def encoder_transformer_flops(image_size, patch_size, layers, embed_dim, heads, mlp_dim):
	num_tokens = np.product([s // p for s, p in zip(image_size, patch_size)])
	# num_tokens += 1  # class embedding
	
	flops = layers * encoder_flops(num_tokens, embed_dim, heads, mlp_dim)
	# flops += patchify_flops(num_tokens, patch_size, embed_dim)
	
	gflops = np.round(flops / 1e9, 3)
	logger.info(f"{flops:,} FLOPs = {gflops} GFLOPs")
	return flops


def decoder_transformer_flops(image_size, patch_size, layers, embed_dim, heads, mlp_dim):
	num_tokens = np.product([s // p for s, p in zip(image_size, patch_size)])
	# num_tokens += 1  # class embedding
	
	flops = layers * decoder_flops(num_tokens, embed_dim, heads, mlp_dim)
	# flops += patchify_flops(num_tokens, patch_size, embed_dim)
	
	gflops = np.round(flops / 1e9, 3)
	logger.info(f"{flops:,} FLOPs = {gflops} GFLOPs")
	return flops

def encoder_transformer_params(layers, embed_dim, mlp_dim):
	attention = 4 * (embed_dim ** 2 + embed_dim)
	feed_forward = 2 * embed_dim * mlp_dim + embed_dim + mlp_dim
	layer_norm = 2 * embed_dim
	
	encoder = layers * (attention + feed_forward + 2 * layer_norm)
	encoder_mparams = np.round(encoder / 1e6, 0).astype(int)
	
	logger.info(f"{encoder:,} params = {encoder_mparams} M params")
	# feed_forward = 8 * embed_dim**2 + 5 * embed_dim
	# attention = (4 * embed_dim ** 2 + 4 * embed_dim) * 1
	# layer_norm = (2 * embed_dim) * 2
	# params = 12 * embed_dim**2 + 13 * embed_dim
	return encoder

def decoder_transformer_params(layers, embed_dim, mlp_dim):
	attention = 4 * (embed_dim ** 2 + embed_dim)
	feed_forward = 2 * embed_dim * mlp_dim + embed_dim + mlp_dim
	layer_norm = 2 * embed_dim
	
	decoder = layers * (2 * attention + feed_forward + 3 * layer_norm)
	decoder_mparams = np.round(decoder / 1e6, 0).astype(int)
	
	logger.info(f"{decoder:,} params = {decoder_mparams} M params")
	# feed_forward = 8 * embed_dim**2 + 5 * embed_dim
	# attention = 8 * embed_dim ** 2 + 8 * embed_dim
	# layer_norm = 6 * embed_dim
	# params = 16 * embed_dim**2 + 19 * embed_dim
	return decoder


def transformer_inference_memory_footprint(params, dtype = 'float32'):
	if dtype == 'float16':
		s = 2.0
	elif dtype == 'float32':
		s = 4.0
	elif dtype == 'float64':
		s = 8.0
	else:
		s = 4.0
	
	mem = int(s * params)
	gbytes = mem / (1024.0 ** 3)
	
	logger.info(f"{mem:,d} B = {gbytes} GB using ({dtype})")
	return gbytes


def transformer_training_memory_footprint(params, dtype = 'float32'):
	if dtype == 'float16':
		s = 2.0
	elif dtype == 'float32':
		s = 4.0
	elif dtype == 'float64':
		s = 8.0
	else:
		s = 4.0
	
	model = s * params
	grads = s * params
	opt = 2 * s * params
	
	mem = int(model + grads + opt)
	gbytes = mem / (1024.0 ** 3)
	
	logger.info(f"{mem:,d} B = {gbytes} GB using ({dtype})")
	return gbytes


def data_memory_footprint(image_size, batch_size=1, dtype='float32'):
	if dtype == 'float16':
		s = 2.0
	elif dtype == 'float32':
		s = 4.0
	elif dtype == 'float64':
		s = 8.0
	else:
		s = 4.0
	
	mem = int(s * batch_size * np.product(image_size))
	gbytes = mem / (1024.0 ** 3)
	
	logger.info(f"{mem:,d} B = {gbytes} GB using ({dtype})")
	return gbytes

def compute_time(flops, gpu="H100", unit="seconds"):
	
	if gpu == "TPUv3":
		# https://cloud.google.com/tpu/docs/v3
		peak_tensor_cores_BF16 = 123 * 10**12 # FLOPS (FLOP per S)
	
	elif gpu == "TPUv4":
		# https://cloud.google.com/tpu/docs/v4
		peak_tensor_cores_BF16 = 275 * 10**12 # FLOPS (FLOP per S)
	
	elif gpu == "TPUv5e":
		# https://cloud.google.com/tpu/docs/v5e
		peak_tensor_cores_BF16 = 197 * 10**12 # FLOPS (FLOP per S)
	
	elif gpu == "TPUv5p":
		# https://cloud.google.com/tpu/docs/v5p
		peak_tensor_cores_BF16 = 459 * 10**12 # FLOPS (FLOP per S)
	
	elif gpu == "V100":
		# https://khairy2011.medium.com/tpu-vs-gpu-vs-cerebras-vs-graphcore-a-fair-comparison-between-ml-hardware-3f5a19d89e38
		peak_tensor_cores_BF16 = 125 * 10**12
	
	elif gpu == "A100":
		# https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
		peak_tensor_cores_BF16 = 312 * 10**12
	
	elif gpu == "H100":
		# https://resources.nvidia.com/en-us-tensor-core
		peak_tensor_cores_BF16 = 756 * 10**12
		
	else:
		raise Exception("Unknown GPU device")
	
	time = flops / peak_tensor_cores_BF16
	if unit == "hours":
		time = time / (60 * 60)
	elif unit == "minutes":
		time = time / 60
	return time
	
	