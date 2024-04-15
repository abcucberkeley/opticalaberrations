import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import warnings
warnings.filterwarnings("ignore")

import logging
import sys
import time
import tensorflow as tf

import profile_utils

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
tf.get_logger().setLevel(logging.ERROR)
    

def main():
    timeit = time.time()
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)
    
    # model = vit.VIT(
    #     name='ViT',
    #     hidden_size=348,
    #     patches=[16],
    #     heads=[12],
    #     repeats=[12],
    # )
    
    # model = prototype.OpticalTransformer(
    #     name='Prototype',
    #     patches=[16],
    #     heads=[16],
    #     repeats=[16],
    # )
    
    # model = opticalnet.OpticalTransformer(
    #     name='OpticalNet',
    #     patches=[32, 16],
    #     heads=[8, 8],
    #     repeats=[8, 8],
    # )
    # model = model.build(input_shape=embeddings_shape)
    

    # num_tokens = np.product([s // p for s, p in zip(context_window, patch_size)])
    # embedding = np.product(patch_size)
    # embedding_shape =  (num_tokens, embedding)
    
    # logger.info(f"{context_window=}, {patch_size=}, {embedding_shape=}")
    
    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Input(shape=embedding_shape, dtype=tf.float16),
    #     vit.Transformer(heads=1, dims=embedding, expand_rate=4, activation='gelu', dropout_rate=0.),
    # ])
    # model.summary()
    #
    # profile_utils.measure_gflops(model)
    # profile_utils.measure_memory_usage(model, batch_size=batch_size)
    
    # profile_utils.transformer_encoder_flops(
    #     num_tokens=num_tokens,
    #     embedding=embedding,
    #     heads=1,
    #     mlp_dim=4*embedding,
    # )
    
    dtype = 'float16'
    batch_size = 1
    dimensions = {
        "2D(g)": {"t": 1, "z": 1, "y": 224, "x": 224, "c": 1},
        "2D(rgb)": {"t": 1, "z": 1, "y": 224, "x": 224, "c": 3},
        "3D(g)": {"t": 1, "z": 224, "y": 224, "x": 224, "c": 1},
        "3D(rgb)": {"t": 1, "z": 224, "y": 224, "x": 224, "c": 3},
        "4D(g)": {"t": 64, "z": 224, "y": 224, "x": 224, "c": 1},
        "4D(rgb)": {"t": 64, "z": 224, "y": 224, "x": 224, "c": 3}
    }
    
    vits = {
        "S": {"layers": 6, "heads": 6, "embedding": 384},
        "B": {"layers": 12, "heads": 12, "embedding": 768},
        "L": {"layers": 24, "heads": 16, "embedding": 1024},
        "H": {"layers": 32, "heads": 16, "embedding": 1280},
    }
    
    configs = {}
    for patch in [8, 16, 32]:
        patches = {
            "2D(g)":    {"t": 1, "z": 1, "y": patch, "x": patch, "c": 1},
            "2D(rgb)":  {"t": 1, "z": 1, "y": patch, "x": patch, "c": 3},
            "3D(g)":    {"t": 1, "z": patch, "y": patch, "x": patch, "c": 1},
            "3D(rgb)":  {"t": 1, "z": patch, "y": patch, "x": patch, "c": 3},
            "4D(g)":    {"t": patch, "z": patch, "y": patch, "x": patch, "c": 1},
            "4D(rgb)":  {"t": patch, "z": patch, "y": patch, "x": patch, "c": 3}
        }
        
        for dims in dimensions.keys():
            
            image_size = list(dimensions[dims].values())
            patch_size = list(patches[dims].values())
            memory_per_image = profile_utils.data_memory_footprint(
                image_size=image_size,
                batch_size=batch_size,
                dtype=dtype,
            )
            
            for transformer in ["encoder", "decoder", "autoencoder"]:
                layers = 6
                heads = 6
                embedding = 384

    
                if transformer == "encoder":
                    params = profile_utils.encoder_transformer_params(
                        layers=layers,
                        embed_dim=embedding,
                    )
                    flops = profile_utils.encoder_transformer_flops(
                        image_size=image_size,
                        patch_size=patch_size,
                        layers=layers,
                        embed_dim=embedding,
                        heads=heads,
                    )
                elif transformer == "decoder":
                    params = profile_utils.decoder_transformer_params(
                        layers=layers,
                        embed_dim=embedding,
                    )
                    flops = profile_utils.decoder_transformer_flops(
                        image_size=image_size,
                        patch_size=patch_size,
                        layers=layers,
                        embed_dim=embedding,
                        heads=heads,
                    )
                else:
                    eparams = profile_utils.encoder_transformer_params(
                        layers=layers,
                        embed_dim=embedding,
                    )
                    dparams = profile_utils.decoder_transformer_params(
                        layers=layers,
                        embed_dim=embedding,
                    )
                    params = eparams + dparams
                    
                    eflops = profile_utils.encoder_transformer_flops(
                        image_size=image_size,
                        patch_size=patch_size,
                        layers=layers,
                        embed_dim=embedding,
                        heads=heads,
                    )
                    
                    dflops = profile_utils.decoder_transformer_flops(
                        image_size=image_size,
                        patch_size=patch_size,
                        layers=layers,
                        embed_dim=embedding,
                        heads=heads,
                    )
                    
                    flops = eflops + dflops
                    
                
                gflops = np.round(flops / 1e9, 3)
                
                model_inference_memory_per_image = profile_utils.transformer_inference_memory_footprint(
                    params=params,
                    dtype=dtype
                )
                
                model_training_memory_per_image = profile_utils.transformer_training_memory_footprint(
                    params=params,
                    dtype=dtype
                )
                
                configs[f"{dims}/{patch} {transformer}"] = {
                    "transformer": transformer,
                    "layers": layers,
                    "heads": heads,
                    "embedding": embedding,
                    "t": dimensions[dims]["t"],
                    "x": dimensions[dims]["x"],
                    "y": dimensions[dims]["y"],
                    "z": dimensions[dims]["z"],
                    "c": dimensions[dims]["c"],
                    "pt": patches[dims]["t"],
                    "px": patches[dims]["x"],
                    "py": patches[dims]["y"],
                    "pz": patches[dims]["z"],
                    "pc": patches[dims]["c"],
                    "parameters": params,
                    "inference_flops_per_image": flops,
                    "inference_gflops_per_image": gflops,
                    "training_flops_per_image": 3 * flops,
                    "training_glops_per_image": 3 * gflops,
                    "model_inference_memory": model_inference_memory_per_image,
                    "model_training_memory": model_training_memory_per_image,
                    "memory_per_image": memory_per_image,
                }
    
            for v in vits:
                layers = vits[v]["layers"]
                heads = vits[v]["heads"]
                embedding = vits[v]["embedding"]
                
                params = profile_utils.encoder_transformer_params(
                    layers=layers,
                    embed_dim=embedding,
                )
                flops = profile_utils.encoder_transformer_flops(
                    image_size=image_size,
                    patch_size=patch_size,
                    layers=layers,
                    embed_dim=embedding,
                    heads=heads,
                )
                model_inference_memory_per_image = profile_utils.transformer_inference_memory_footprint(
                    params=params,
                    dtype=dtype
                )
                
                model_training_memory_per_image = profile_utils.transformer_training_memory_footprint(
                    params=params,
                    dtype=dtype
                )
                
                configs[f"{dims} ViT {v}/{patch}"] = {
                    "transformer": "encoder",
                    "layers": layers,
                    "heads": heads,
                    "embedding": embedding,
                    "t": dimensions[dims]["t"],
                    "x": dimensions[dims]["x"],
                    "y": dimensions[dims]["y"],
                    "z": dimensions[dims]["z"],
                    "c": dimensions[dims]["c"],
                    "pt": patches[dims]["t"],
                    "px": patches[dims]["x"],
                    "py": patches[dims]["y"],
                    "pz": patches[dims]["z"],
                    "pc": patches[dims]["c"],
                    "parameters": params,
                    "inference_flops_per_image": flops,
                    "inference_gflops_per_image": gflops,
                    "training_flops_per_image": 3 * flops,
                    "training_glops_per_image": 3 * gflops,
                    "model_inference_memory": model_inference_memory_per_image,
                    "model_training_memory": model_training_memory_per_image,
                    "memory_per_image": memory_per_image,
                }
    
    df = pd.DataFrame.from_dict(configs, orient='index')
    df = df.sort_values(['px', 'layers'], ascending=[True, True])
    df.to_csv("../data/scaling.csv")
    print(df)
    
    logger.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
