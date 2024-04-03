import matplotlib
matplotlib.use('Agg')

import time
import subprocess
import sys
import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.profiler.model_analyzer import profile as tf_profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder


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
	logger.info(f"{model.name}: {gflops} Gflops")
	return gflops


def load_tf_logs(path):
	subprocess.call("pip install -U tbparse", shell=True)
	from tbparse import SummaryReader
	
	reader = SummaryReader(str(path), pivot=True, extra_columns=set(['wall_time']))
	df = reader.tensors
	return df