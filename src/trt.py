''' TensorRT 
TensorRT is an NVIDIA product that "optimizes" a model for inference (or training).  Typically by reducing the precision
of expensive operations from float32 to float16 or int8. TensorRT (TRT) converts/recompiles the model, and this part must
be run on Linux. For converting a TensorFlow model, the easiest path is through TF-TensorRT, aka tf_trt, which is included
in tensorflow installations by default. Therefore, we pull the latest NVIDIA tensorflow container and do this in docker.

Here we run a Linux WSL2 container from Windows, expose some ports (if tensorboard would work), and map our src directory 
into it ala:
docker run --name trt_docker_container --gpus all -it -p 6006:6006 -p 8008:8008  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v "C:\SPIM\Common\Calculations\Python\Phase Retrieval ML\opticalaberrations":/workspace/opticalaberrations       nvcr.io/nvidia/tensorflow:23.05-tf2-py3

The python code below will 
1. Convert the model.
2. Build the engines. (aka, compile the code for the current GPU, taking many minutes)
3. Save the converted model.
4. Run tests to see how much faster the optimized code is.

I run this typically in ipython, cutting and pasting the lines I want to write.  Keeping an eye on GPU RAM.
The helper.py is from the NVIDIA examples.

Conclusion:
The predictions can be faster, but takes a lot longer to startup.  Perhaps Transformer model may not be well suited for this?
'''

from tensorflow.python.compiler.tensorrt import trt_convert as tf_trt
from tensorflow.python.compiler.tensorrt import utils as trt_utils
from tensorflow.python.saved_model import tag_constants
import tensorflow as tf
import tensorrt as trt
from helper import ModelOptimizer # using the helper from <URL>
from helper import OptimizedModel
from tensorflow.keras.models import load_model, save_model
import numpy as np
import os

precision_dict = {
    np.float32: tf_trt.TrtPrecisionMode.FP32,
    np.float16: tf_trt.TrtPrecisionMode.FP16,
    np.int8: tf_trt.TrtPrecisionMode.INT8,
}

physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)


# User things
precision = np.float16 # Options are np.float32, np.float16, or np.int8
model_dir = '/workspace/opticalaberrations/pretrained_models/serialized/'
MAX_BATCH_SIZE=25

dummy_input_batch = np.zeros((MAX_BATCH_SIZE, 6, 64, 64, 1), dtype=np.float32) # e.g. 6 embeddings, each 64 x 64
x = tf.constant(dummy_input_batch)
trt_precision = precision_dict[precision]

# TRT converter options
max_workspace_size_bytes=int(3*1e9) # maximum GPU memory size available for TRT layers 
minimum_segment_size=5  # if TF-TRT generates too many engines (> 5-10), increase this number
# TRT logging
os.environ["TF_TRT_SHOW_DETAILED_REPORT"] = "1" # this prints the detailed nonconversion report on stdout.
output_graph_file = f"/workspace/opticalaberrations/graphviz/{trt_precision}"
# os.environ["TF_TRT_EXPORT_GRAPH_VIZ_PATH"]= f"{output_graph_file}.dot"   # output graph. transformer makes this impossible?


#################### Convert input model
if (precision == "INT8" or precision == np.int8) and calibration_data is None:
    raise(Exception("No calibration data set!"))
conversion_params = tf_trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt_precision, 
                                                                        max_workspace_size_bytes=max_workspace_size_bytes,
                                                                        use_calibration= precision == "INT8",
                                                                        minimum_segment_size=minimum_segment_size,
                                                                        allow_build_at_runtime=False,
                                                                        maximum_cached_engines=100,                                                                        
                                                                        )
converter = tf_trt.TrtGraphConverterV2(
    input_saved_model_dir=model_dir,
    conversion_params=conversion_params,
    use_dynamic_shape=True,
    dynamic_shape_profile_strategy="Optimal")

if (precision == "INT8" or precision == np.int8):
    new_predict = converter.convert(calibration_input_fn=self.calibration_data)
else:
    new_predict = converter.convert()
# converter.summary(line_length=300)

# !dot -Tpng  "{output_graph_file}.dot" -o  "{output_graph_file}.png"   # takes infinite amoutn of time.  Dont' do this!

#### Build the TRT engines (takes many minutes)
def input_fn():
   batch_size = MAX_BATCH_SIZE
   x = dummy_input_batch[0:batch_size, :]
   yield [x]

converter.build(input_fn=input_fn)
converter.save(f'/workspace/opticalaberrations/pretrained_models/trt_{trt_precision}')

# Test predictions
labeling = new_predict(x) # warmup
%timeit -n 1 labeling = new_predict(x)


######################## Load saved model and predict

#new model
saved_model_loaded = tf.saved_model.load(f'/workspace/opticalaberrations/pretrained_models/trt_{trt_precision}', tags=[tag_constants.SERVING])
new_predict = saved_model_loaded.signatures['serving_default']

#old model
from roi import ROI
import opticalnet
from tensorflow.keras.models import load_model, save_model
custom_objects = {
    "ROI": ROI,
    "Stem": opticalnet.Stem,
    "Patchify": opticalnet.Patchify,
    "Merge": opticalnet.Merge,
    "PatchEncoder": opticalnet.PatchEncoder,
    "MLP": opticalnet.MLP,
    "Transformer": opticalnet.Transformer,
}
oldmodel = load_model(model_dir, custom_objects=custom_objects)
oldmodel.predict(dummy_input_batch) # warmup
labeling = new_predict(x) # warmup
print("warmup complete")
%timeit -n 1 oldmodel.predict(dummy_input_batch)
%timeit -n 1 labeling = new_predict(x)
