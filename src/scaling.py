from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

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
    

def scaling_transformers(dtype = 'float16', batch_size = 1, outdir=Path("../scaling")):
    dimensions = {
        "2D(g)": {"t": 1, "z": 1, "y": 256, "x": 256, "c": 1},
        "2D(rgb)": {"t": 1, "z": 1, "y": 256, "x": 256, "c": 3},
        "3D(g)": {"t": 1, "z": 256, "y": 256, "x": 256, "c": 1},
        "3D(rgb)": {"t": 1, "z": 256, "y": 256, "x": 256, "c": 3},
        "4D(g)": {"t": 64, "z": 256, "y": 256, "x": 256, "c": 1},
        "4D(rgb)": {"t": 64, "z": 256, "y": 256, "x": 256, "c": 3}
    }
    configs = {
        "S": {"layers": 12, "heads": 6, "embedding": 384, "mlp": 1536},
        "B": {"layers": 12, "heads": 12, "embedding": 768, "mlp": 3072},
        "L": {"layers": 24, "heads": 16, "embedding": 1024, "mlp": 4096},
        "H": {"layers": 32, "heads": 16, "embedding": 1280, "mlp": 5120},
        "g": {"layers": 40, "heads": 16, "embedding": 1408, "mlp": 6144},
        "G": {"layers": 48, "heads": 16, "embedding": 1664, "mlp": 8192},
        "e": {"layers": 56, "heads": 16, "embedding": 1792, "mlp": 15360},
        "22B": {"layers": 48, "heads": 48, "embedding": 6144, "mlp": 24576},
    }
    
    transformer_configs, vit_configs = {}, {}
    for patch in [8, 16, 32]:
        patches = {
            "2D(g)": {"t": 1, "z": 1, "y": patch, "x": patch, "c": 1},
            "2D(rgb)": {"t": 1, "z": 1, "y": patch, "x": patch, "c": 3},
            "3D(g)": {"t": 1, "z": patch, "y": patch, "x": patch, "c": 1},
            "3D(rgb)": {"t": 1, "z": patch, "y": patch, "x": patch, "c": 3},
            "4D(g)": {"t": patch, "z": patch, "y": patch, "x": patch, "c": 1},
            "4D(rgb)": {"t": patch, "z": patch, "y": patch, "x": patch, "c": 3}
        }
        
        for dims in dimensions.keys():
            
            image_size = list(dimensions[dims].values())
            patch_size = list(patches[dims].values())
            
            memory_per_image = profile_utils.data_memory_footprint(
                image_size=image_size,
                batch_size=batch_size,
                dtype=dtype,
            )
            
            for transformer in ["En", "De", "AE"]:
                for c in configs:
                    print(f"{dims} {c}/{patch} {transformer}")
                    layers = configs[c]["layers"]
                    heads = configs[c]["heads"]
                    embedding = configs[c]["embedding"]
                    mlp = configs[c]["mlp"]


                    if transformer == "En":
                        params = profile_utils.encoder_transformer_params(
                            layers=layers,
                            embed_dim=embedding,
                            mlp_dim=mlp
                        )
                        flops = profile_utils.encoder_transformer_flops(
                            image_size=image_size,
                            patch_size=patch_size,
                            layers=layers,
                            embed_dim=embedding,
                            heads=heads,
                            mlp_dim=mlp
                        )
                        flops_per_token = layers * profile_utils.encoder_flops(1, embedding, heads, mlp)
                    elif transformer == "De":
                        params = profile_utils.decoder_transformer_params(
                            layers=layers,
                            embed_dim=embedding,
                            mlp_dim=mlp
                        )
                        flops = profile_utils.decoder_transformer_flops(
                            image_size=image_size,
                            patch_size=patch_size,
                            layers=layers,
                            embed_dim=embedding,
                            heads=heads,
                            mlp_dim=mlp
                        )
                        flops_per_token = layers * profile_utils.decoder_flops(1, embedding, heads, mlp)
                    else:
                        eparams = profile_utils.encoder_transformer_params(
                            layers=layers,
                            embed_dim=embedding,
                            mlp_dim=mlp
                        )
                        dparams = profile_utils.decoder_transformer_params(
                            layers=layers,
                            embed_dim=embedding,
                            mlp_dim=mlp
                        )
                        params = eparams + dparams
    
                        eflops = profile_utils.encoder_transformer_flops(
                            image_size=image_size,
                            patch_size=patch_size,
                            layers=layers,
                            embed_dim=embedding,
                            heads=heads,
                            mlp_dim=mlp
                        )
                        eflops_per_token = layers * profile_utils.encoder_flops(1, embedding, heads, mlp)
    
                        dflops = profile_utils.decoder_transformer_flops(
                            image_size=image_size,
                            patch_size=patch_size,
                            layers=layers,
                            embed_dim=embedding,
                            heads=heads,
                            mlp_dim=mlp
                        )
                        dflops_per_token = layers * profile_utils.decoder_flops(1, embedding, heads, mlp)
                        
                        flops = eflops + dflops
                        flops_per_token = eflops_per_token + dflops_per_token
    
                    gflops = np.round(flops / 1e9, 3)
                    gflops_per_token = np.round(flops_per_token / 1e9, 3)
    
                    model_inference_memory_per_image = profile_utils.transformer_inference_memory_footprint(
                        params=params,
                        dtype=dtype
                    )
    
                    model_training_memory_per_image = profile_utils.transformer_training_memory_footprint(
                        params=params,
                        dtype=dtype
                    )
    
                    inference_time_per_batch = profile_utils.compute_time(flops=flops, gpu="H100", unit="seconds")
                    training_time_per_batch = profile_utils.compute_time(flops=3*flops, gpu="H100", unit="seconds")
    
                    transformer_configs[f"{dims} {c}/{patch} {transformer}"] = {
                        "data": dims,
                        "class": f"{c}/{patch}",
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
                        "flops_per_token": flops_per_token,
                        "gflops_per_token": gflops_per_token,
                        "model_inference_memory": model_inference_memory_per_image,
                        "model_training_memory": model_training_memory_per_image,
                        "memory_per_image": memory_per_image,
                        "inference_time_per_batch": inference_time_per_batch,
                        "training_time_per_batch": training_time_per_batch,
                    }
    
    transformer_scaling = pd.DataFrame.from_dict(transformer_configs, orient='index')
    transformer_scaling = transformer_scaling.sort_values(['px', 'layers'], ascending=[True, True])
    transformer_scaling.to_csv(outdir/"transformers.csv")
    print(transformer_scaling)
    return transformer_scaling
    
def scaling_vit(dtype = 'float16', batch_size = 1, outdir=Path("../scaling"), dataset_size = 10000000):
    
    vit_dimensions = {
        "2D(g)": {"t": 1, "z": 1, "y": 224, "x": 224, "c": 1},
        "2D(rgb)": {"t": 1, "z": 1, "y": 224, "x": 224, "c": 3},
        "3D(g)": {"t": 1, "z": 224, "y": 224, "x": 224, "c": 1},
        "3D(rgb)": {"t": 1, "z": 224, "y": 224, "x": 224, "c": 3},
        "4D(g)": {"t": 8, "z": 224, "y": 224, "x": 224, "c": 1},
        "4D(rgb)": {"t": 8, "z": 224, "y": 224, "x": 224, "c": 3}
    }
    vits = {
        "S": {"layers": 12, "heads": 6, "embedding": 384, "mlp": 1536},
        "B": {"layers": 12, "heads": 12, "embedding": 768, "mlp": 3072},
        "L": {"layers": 24, "heads": 16, "embedding": 1024, "mlp": 4096},
        "H": {"layers": 32, "heads": 16, "embedding": 1280, "mlp": 5120},
        "g": {"layers": 40, "heads": 16, "embedding": 1408, "mlp": 6144},
        "G": {"layers": 48, "heads": 16, "embedding": 1664, "mlp": 8192},
        "e": {"layers": 56, "heads": 16, "embedding": 1792, "mlp": 15360},
        "22B": {"layers": 48, "heads": 48, "embedding": 6144, "mlp": 24576},
    }
    
    transformer_configs, vit_configs = {}, {}
    for patch in [14, 16]:
        patches = {
            "2D(g)": {"t": 1, "z": 1, "y": patch, "x": patch, "c": 1},
            "2D(rgb)": {"t": 1, "z": 1, "y": patch, "x": patch, "c": 3},
            "3D(g)": {"t": 1, "z": patch, "y": patch, "x": patch, "c": 1},
            "3D(rgb)": {"t": 1, "z": patch, "y": patch, "x": patch, "c": 3},
            "4D(g)": {"t": 2, "z": patch, "y": patch, "x": patch, "c": 1},
            "4D(rgb)": {"t": 2, "z": patch, "y": patch, "x": patch, "c": 3}
        }
        
        for dims in vit_dimensions.keys():
            
            image_size = list(vit_dimensions[dims].values())
            patch_size = list(patches[dims].values())
            
            memory_per_image = profile_utils.data_memory_footprint(
                image_size=image_size,
                batch_size=batch_size,
                dtype=dtype,
            )
            
            for v in vits:
                print(f"{dims} ViT {v}/{patch}")
                
                layers = vits[v]["layers"]
                heads = vits[v]["heads"]
                embedding = vits[v]["embedding"]
                mlp = vits[v]["mlp"]
                
                params = profile_utils.encoder_transformer_params(
                    layers=layers,
                    embed_dim=embedding,
                    mlp_dim=mlp
                )
                flops = profile_utils.encoder_transformer_flops(
                    image_size=image_size,
                    patch_size=patch_size,
                    layers=layers,
                    embed_dim=embedding,
                    heads=heads,
                    mlp_dim=mlp
                )
                flops_per_patch = layers * profile_utils.encoder_flops(1, embedding, heads, mlp)
                
                model_inference_memory_per_image = profile_utils.transformer_inference_memory_footprint(
                    params=params,
                    dtype=dtype
                )
                
                model_training_memory_per_image = profile_utils.transformer_training_memory_footprint(
                    params=params,
                    dtype=dtype
                )
                
                inference_time_per_image = profile_utils.compute_time(flops=flops, gpu="H100", unit="seconds")
                training_time_per_image = profile_utils.compute_time(flops=3 * flops, gpu="H100", unit="seconds")
                gflops = np.round(flops / 1e9, 3)
                gflops_per_patch = np.round(flops_per_patch / 1e9, 3)
                
                patches_per_image = np.product([s // p for s, p in zip(image_size, patch_size)])
                pixels_per_patch = np.product(patch_size)
                images_per_h100 = 80 // memory_per_image
                number_h100_for_4096 = np.ceil(model_training_memory_per_image + (memory_per_image * 4096) / 80)
                number_h100_for_2048 = np.ceil(model_training_memory_per_image + (memory_per_image * 2048) / 80)
                
                """
                    ViT L/16: https://arxiv.org/pdf/2010.11929.pdf (table 6)
                    exaFLOPs = 783
                    epochs = 7
                    dataset = 303,000,000
                    TPUv3 peak FLOPS = 123 * 10**12
 
                    training_time_per_image = (783 * 10^18) / 7 / 303,000,000 / (123 * 10**12)
                    training_time_per_image = 0.00300134543
                """
                training_h100_days_per_epoch = dataset_size * training_time_per_image / 3600 / 24
                multigpu_training_per_epoch = training_h100_days_per_epoch / number_h100_for_4096
                multigpu_1024_training_per_epoch = training_h100_days_per_epoch / 1024
                
                vit_configs[f"{dims} ViT {v}/{patch}"] = {
                    "data": dims,
                    "class": f"{v}/{patch}",
                    "transformer": "encoder",
                    "layers": layers,
                    "heads": heads,
                    "embedding": embedding,
                    "t": vit_dimensions[dims]["t"],
                    "x": vit_dimensions[dims]["x"],
                    "y": vit_dimensions[dims]["y"],
                    "z": vit_dimensions[dims]["z"],
                    "c": vit_dimensions[dims]["c"],
                    "pt": patches[dims]["t"],
                    "px": patches[dims]["x"],
                    "py": patches[dims]["y"],
                    "pz": patches[dims]["z"],
                    "pc": patches[dims]["c"],
                    "parameters": params,
                    "inference_gflops_per_image": gflops,
                    "training_glops_per_image": 3 * gflops,
                    "gflops_per_patch": gflops_per_patch,
                    "model_inference_memory": model_inference_memory_per_image,
                    "model_training_memory": model_training_memory_per_image,
                    "inference_time_per_image": inference_time_per_image,
                    "training_time_per_image": training_time_per_image,
                    "training_h100_days_per_epoch": training_h100_days_per_epoch,
                    "memory_per_image": memory_per_image,
                    "patches_per_image": patches_per_image,
                    "pixels_per_patch": pixels_per_patch,
                    "images_per_h100": images_per_h100,
                    "number_h100_for_2048": number_h100_for_2048,
                    "number_h100_for_4096": number_h100_for_4096,
                    "multigpu_training_time_per_epoch": multigpu_training_per_epoch,
                    "multigpu_1024_training_time_per_epoch": multigpu_1024_training_per_epoch,
                }
    
    vit_scaling = pd.DataFrame.from_dict(vit_configs, orient='index')
    vit_scaling = vit_scaling.sort_values(['px', 'parameters', 'layers', 'heads'], ascending=[True, True, True, True])
    vit_scaling.to_csv(outdir/"vits.csv")
    print(vit_scaling)
    return vit_scaling

def plot_gflops(df, outdir):
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'xtick.major.pad': 10
    })
    
    fig, ax = plt.subplots(figsize=(8, 8))
    for cc, colormap in zip(['S/8', 'B/8', 'L/8', 'H/8', 'G/8'], ['C0', 'C1', 'C2', 'C3', 'C4']):
        data = df[df.cat == cc]
        g = sns.scatterplot(
            data=data,
            x="parameters",
            y="inference_gflops_per_image",
            c=colormap,
            ax=ax
        )

        # ax.text(
        #     data["training_gflops"] + 0.1, data["epoch_loss"], cc,
        #     horizontalalignment='left', size='medium', color=colormap, weight='semibold'
        # )

    ax.grid(True, which="major", axis='both', lw=.5, ls='--', zorder=0)
    ax.grid(True, which="minor", axis='both', lw=.25, ls='--', zorder=0)
    # ax.set_xlabel('Training compute (Billions of GFLOPs)')
    # ax.set_ylabel('Loss')
    # ax.set_yscale('log')
    # ax.set_xlim(0, 11)
    # ax.set_ylim(10 ** -3, 1)
    ax.legend(loc='lower left', ncol=1, title="", frameon=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    savepath = Path(f'{outdir}/best')
    plt.savefig(f'{savepath}.pdf', bbox_inches='tight', pad_inches=.25)
    plt.savefig(f'{savepath}.png', dpi=300, bbox_inches='tight', pad_inches=.25)
    plt.savefig(f'{savepath}.svg', dpi=300, bbox_inches='tight', pad_inches=.25)
    
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
    dataset_size = 303000000
    
    outdir = Path("../scaling")
    outdir.mkdir(parents=True, exist_ok=True)
    
    vit_scaling = scaling_vit(dtype=dtype, batch_size=batch_size, outdir=outdir, dataset_size=dataset_size)
    transformer_scaling = scaling_transformers(dtype=dtype, batch_size=batch_size, outdir=outdir)
    # plot_gflops(vit_scaling, outdir=outdir)
    
    logger.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
