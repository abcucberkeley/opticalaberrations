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


def scaling_transformers(dtype = 'float16', outdir=Path("../scaling")):
    dimensions = {
        "2D(g)": {"t": 1, "z": 1, "y": 256, "x": 256, "c": 1},
        "2D(rgb)": {"t": 1, "z": 1, "y": 256, "x": 256, "c": 3},
        "3D(g)": {"t": 1, "z": 256, "y": 256, "x": 256, "c": 1},
        "3D(rgb)": {"t": 1, "z": 256, "y": 256, "x": 256, "c": 3},
        "4D(g)": {"t": 8, "z": 256, "y": 256, "x": 256, "c": 1},
        "4D(rgb)": {"t": 8, "z": 256, "y": 256, "x": 256, "c": 3}
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
    for patch in [14, 16]:
        patches = {
            "2D(g)": {"t": 1, "z": 1, "y": patch, "x": patch, "c": 1},
            "2D(rgb)": {"t": 1, "z": 1, "y": patch, "x": patch, "c": 3},
            "3D(g)": {"t": 1, "z": patch, "y": patch, "x": patch, "c": 1},
            "3D(rgb)": {"t": 1, "z": patch, "y": patch, "x": patch, "c": 3},
            "4D(g)": {"t": 2, "z": patch, "y": patch, "x": patch, "c": 1},
            "4D(rgb)": {"t": 2, "z": patch, "y": patch, "x": patch, "c": 3}
        }
        
        for dims in dimensions.keys():
            
            image_size = list(dimensions[dims].values())
            patch_size = list(patches[dims].values())
            
            memory_per_image = profile_utils.data_memory_footprint(
                image_size=image_size,
                batch_size=1,
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
                    gflops_per_patch = np.round(flops_per_token / 1e9, 3)
    
                    model_inference_memory = profile_utils.transformer_inference_memory_footprint(
                        params=params,
                        dtype=dtype
                    )
    
                    model_training_memory = profile_utils.transformer_training_memory_footprint(
                        params=params,
                        dtype=dtype
                    )
    
                    inference_time_per_image = profile_utils.compute_time(flops=flops, gpu="H100", unit="seconds")
                    training_time_per_image = profile_utils.compute_time(flops=3*flops, gpu="H100", unit="seconds")
                    
                    patches_per_image = np.product([s // p for s, p in zip(image_size, patch_size)])
                    pixels_per_patch = np.product(patch_size)
                    images_per_h100 = 80 // memory_per_image
                    
                    transformer_configs[f"{dims} {c}/{patch} {transformer}"] = {
                        "data": dims,
                        "class": f"{c}/{patch}",
                        "transformer": "encoder",
                        "layers": layers,
                        "heads": heads,
                        "mlp": mlp,
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
                        "patches_per_image": patches_per_image,
                        "pixels_per_patch": pixels_per_patch,
                        "images_per_h100": images_per_h100,
                        "memory_per_image": memory_per_image,
                        "parameters": params,
                        "inference_gflops_per_image": gflops,
                        "training_glops_per_image": 3 * gflops,
                        "gflops_per_patch": gflops_per_patch,
                        "model_inference_memory": model_inference_memory,
                        "model_training_memory": model_training_memory,
                        "inference_time_per_image": inference_time_per_image,
                        "training_time_per_image": training_time_per_image,
                    }
    
    transformer_scaling = pd.DataFrame.from_dict(transformer_configs, orient='index')
    transformer_scaling = transformer_scaling.sort_values(['px', 'parameters', 'layers', 'heads'], ascending=[True, True, True, True])
    transformer_scaling.to_csv(outdir/"transformers.csv")
    return transformer_scaling
    
def scaling_vit(dtype = 'float16', outdir=Path("../scaling")):
    
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
                batch_size=1,
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
                
                model_inference_memory = profile_utils.transformer_inference_memory_footprint(
                    params=params,
                    dtype=dtype
                )
                
                model_training_memory = profile_utils.transformer_training_memory_footprint(
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
                
                """
                    ViT L/16: https://arxiv.org/pdf/2010.11929.pdf (table 6)
                    exaFLOPs = 783
                    epochs = 7
                    dataset = 303,000,000
                    TPUv3 peak FLOPS = 123 * 10**12
 
                    training_time_per_image = (783 * 10^18) / 7 / 303,000,000 / (123 * 10**12)
                    training_time_per_image = 0.00300134543
                """
                vit_configs[f"{dims} ViT {v}/{patch}"] = {
                    "data": dims,
                    "class": f"{v}/{patch}",
                    "transformer": "encoder",
                    "layers": layers,
                    "heads": heads,
                    "mlp": mlp,
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
                    "patches_per_image": patches_per_image,
                    "pixels_per_patch": pixels_per_patch,
                    "images_per_h100": images_per_h100,
                    "memory_per_image": memory_per_image,
                    "parameters": params,
                    "inference_gflops_per_image": gflops,
                    "training_gflops_per_image": 3 * gflops,
                    "gflops_per_patch": gflops_per_patch,
                    "model_inference_memory": model_inference_memory,
                    "model_training_memory": model_training_memory,
                    "inference_time_per_image": inference_time_per_image,
                    "training_time_per_image": training_time_per_image,
                }
    
    vit_scaling = pd.DataFrame.from_dict(vit_configs, orient='index')
    vit_scaling = vit_scaling.sort_values(['px', 'parameters', 'layers', 'heads'], ascending=[True, True, True, True])
    vit_scaling.to_csv(outdir/"vits.csv")
    return vit_scaling

def plot_parameter_scaling(df, outdir, dataset_size, batch_size):
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'xtick.major.pad': 10
    })
    
    df["number_h100_for_batch"] = np.ceil(df["model_training_memory"] + (df["memory_per_image"] * batch_size) / 80)
    df["training_h100_hours_per_step"] = batch_size * df["training_time_per_image"] / 3600
    df["training_h100_days_per_epoch"] = dataset_size * df["training_time_per_image"] / 3600 / 24
    df["multigpu_training_days_per_epoch"] = df["training_h100_days_per_epoch"] / df["number_h100_for_batch"]
    df["multigpu_256_training_days_per_epoch"] = df["training_h100_days_per_epoch"] / 256
    df["gpu_compute_cost_per_epoch"] = df["training_h100_days_per_epoch"] * 24 * 2
    
    fois = {
        "training_gflops_per_image": "Training GFLOPs per image",
        "training_time_per_image": "Training H100 seconds per image",
        "number_h100_for_batch": "Number of H100s needed for a batch (4096)",
        "training_h100_hours_per_step": "Training H100 hours per batch (4096)",
        "training_h100_days_per_epoch": "Training H100 days per epoch",
        "gpu_compute_cost_per_epoch": "H100 compute cost per epoch ($2/hr)",
    }
    for ff, ll in fois.items():
        fig, ax = plt.subplots(figsize=(8, 8))
        data = df.loc[df['data'].str.match(r'.*\(rgb\)')]
        g = sns.lineplot(
            data=data,
            x="parameters",
            y=ff,
            hue='data',
            style="px",
            hue_order=['4D(rgb)', '3D(rgb)', '2D(rgb)'],
            ax=ax,
            legend=True,
            markers=True
        )
        
        if ff in ["training_h100_days_per_epoch", "gpu_compute_cost_per_epoch"]:
            ax.set_title(f'Dataset: {dataset_size:,} images')
        
        ax.grid(True, which="major", axis='both', lw=.5, ls='--', zorder=0)
        ax.grid(True, which="minor", axis='both', lw=.25, ls='--', zorder=0)
        ax.set_ylabel(ll)
        ax.set_xlabel('Model size (non-embedding parameters)')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend(loc='lower left', ncol=1, title="", frameon=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend(loc='lower right', ncol=1, title="", frameon=False)
        legend_handles, _ = g.get_legend_handles_labels()
        ax.legend(
            legend_handles, [
                'Data', '4D(rgb)', '3D(rgb)', '2D(rgb)',
                'Patch (t, x, y, z, c)', f'(2, 14, 14, 14, 3)', f'(2, 16, 16, 16, 3)',
            ],
            loc='lower right', ncol=1, title="", frameon=False
        )
    
        savepath = Path(f'{outdir}/{ff}')
        plt.savefig(f'{savepath}.pdf', bbox_inches='tight', pad_inches=.25)
        plt.savefig(f'{savepath}.png', dpi=300, bbox_inches='tight', pad_inches=.25)
        plt.savefig(f'{savepath}.svg', dpi=300, bbox_inches='tight', pad_inches=.25)
    
    
def main():
    timeit = time.time()
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)
    
    dtype = 'float16'
    batch_size = 4096
    dataset_size = 303000000
    
    outdir = Path("../scaling")
    outdir.mkdir(parents=True, exist_ok=True)
    
    # transformer_scaling = scaling_transformers(dtype=dtype, outdir=outdir)
    vit_scaling = scaling_vit(dtype=dtype, outdir=outdir)
    plot_parameter_scaling(vit_scaling, outdir=outdir, dataset_size=dataset_size, batch_size=batch_size)
    
    logger.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
