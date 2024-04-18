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
        
        # "B196": {"layers": 12, "heads": 12, "embedding": 196, "mlp": 4*196},
        # "B588": {"layers": 12, "heads": 12, "embedding": 588, "mlp": 4*588},
        # "B2744": {"layers": 12, "heads": 12, "embedding": 2744, "mlp": 4*2744},
        # "B8232": {"layers": 12, "heads": 12, "embedding": 8232, "mlp": 4*8232},
        # "B5488": {"layers": 12, "heads": 12, "embedding": 5488, "mlp": 4*5488},
        # "B16464": {"layers": 12, "heads": 12, "embedding": 16464, "mlp": 4*16464},
        #
        # "L196": {"layers": 24, "heads": 16, "embedding": 196, "mlp": 4 * 196},
        # "L588": {"layers": 24, "heads": 16, "embedding": 588, "mlp": 4 * 588},
        # "L2744": {"layers": 24, "heads": 16, "embedding": 2744, "mlp": 4 * 2744},
        # "L8232": {"layers": 24, "heads": 16, "embedding": 8232, "mlp": 4 * 8232},
        # "L5488": {"layers": 24, "heads": 16, "embedding": 5488, "mlp": 4 * 5488},
        # "L16464": {"layers": 24, "heads": 16, "embedding": 16464, "mlp": 4 * 16464},
        #
        # "H196": {"layers": 32, "heads": 16, "embedding": 196, "mlp": 4 * 196},
        # "H588": {"layers": 32, "heads": 16, "embedding": 588, "mlp": 4 * 588},
        # "H2744": {"layers": 32, "heads": 16, "embedding": 2744, "mlp": 4 * 2744},
        # "H8232": {"layers": 32, "heads": 16, "embedding": 8232, "mlp": 4 * 8232},
        # "H5488": {"layers": 32, "heads": 16, "embedding": 5488, "mlp": 4 * 5488},
        # "H16464": {"layers": 32, "heads": 16, "embedding": 16464, "mlp": 4 * 16464},
        #
        # "G196": {"layers": 48, "heads": 16, "embedding": 196, "mlp": 4 * 196},
        # "G588": {"layers": 48, "heads": 16, "embedding": 588, "mlp": 4 * 588},
        # "G2744": {"layers": 48, "heads": 16, "embedding": 2744, "mlp": 4 * 2744},
        # "G8232": {"layers": 48, "heads": 16, "embedding": 8232, "mlp": 4 * 8232},
        # "G5488": {"layers": 48, "heads": 16, "embedding": 5488, "mlp": 4 * 5488},
        # "G16464": {"layers": 48, "heads": 16, "embedding": 16464, "mlp": 4 * 16464},
        #
        # "E196": {"layers": 56, "heads": 32, "embedding": 196, "mlp": 4 * 196},
        # "E588": {"layers": 56, "heads": 32, "embedding": 588, "mlp": 4 * 588},
        # "E2744": {"layers": 56, "heads": 32, "embedding": 2744, "mlp": 4 * 2744},
        # "E8232": {"layers": 56, "heads": 32, "embedding": 8232, "mlp": 4 * 8232},
        # "E5488": {"layers": 56, "heads": 32, "embedding": 5488, "mlp": 4 * 5488},
        # "E16464": {"layers": 56, "heads": 32, "embedding": 16464, "mlp": 4 * 16464},
        #
        # "T196": {"layers": 64, "heads": 48, "embedding": 196, "mlp": 4 * 196},
        # "T588": {"layers": 64, "heads": 48, "embedding": 588, "mlp": 4 * 588},
        # "T2744": {"layers": 64, "heads": 48, "embedding": 2744, "mlp": 4 * 2744},
        # "T8232": {"layers": 64, "heads": 48, "embedding": 8232, "mlp": 4 * 8232},
        # "T5488": {"layers": 64, "heads": 48, "embedding": 5488, "mlp": 4 * 5488},
        # "T16464": {"layers": 64, "heads": 48, "embedding": 16464, "mlp": 4 * 16464},
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
    
    vit_configs = {}
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

def plot_parameter_scaling(
    df,
    outdir,
    x="parameters",
    y="gflops",
    xlabel='Model size (non-embedding parameters)',
    ylabel='GFLOPS',
    dataset_size=None,
    palette='muted',
    published_models_only=False,
    xlog=True,
    ylog=True,
):
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
    
    if published_models_only:
        data = df.loc[df['data'].str.match(r'2D\(rgb\)')]
    else:
        data = df.loc[df['data'].str.match(r'.*\(rgb\)')]
    
    if published_models_only:
        g = sns.lineplot(
            data=data,
            x=x,
            y=y,
            hue='data',
            style="px",
            ax=ax,
            legend=True,
            markers=True,
            palette='Greys_r'
        )
    else:
        g = sns.lineplot(
            data=data,
            x=x,
            y=y,
            hue='data',
            hue_order=['4D(rgb)', '3D(rgb)', '2D(rgb)'],
            style="px",
            ax=ax,
            legend=True,
            markers=True,
            palette=palette
        )
    
    for dd, cc in zip(['4D(rgb)', '3D(rgb)', '2D(rgb)'], ['C0', 'C1', 'C2']):
        d = data[(data['data'] == dd) & (data['px'] == 14)]
        
        for line in range(0, d.shape[0]):
            plt.text(
                d[x][line], d[y][line], d['class'][line].strip('/14'),
                horizontalalignment='left', size='medium', color='dimgrey', weight='semibold'
            )

    ax.grid(True, which="major", axis='both', lw=.5, ls='--', zorder=0)
    ax.grid(True, which="minor", axis='both', lw=.25, ls='--', zorder=0)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    
    if xlog:
        ax.set_xscale('log')
    
    if ylog:
        ax.set_yscale('log')
        
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    legend_handles, _ = g.get_legend_handles_labels()
    
    if published_models_only:
        ax.legend(
            legend_handles, [
                'Data (x, y, c)', '2D (224, 224, 3)',
                'Patch (x, y, c)', f'(14, 14, 3)', f'(16, 16, 3)',
            ],
            loc='upper left', ncol=1, title="", frameon=False
        )
    else:
        ax.legend(
            legend_handles, [
                'Data (t, x, y, z, c)', '4D (8, 224, 224, 224, 3)', '3D (1, 224, 224, 224, 3)', '2D (1, 224, 224, 1, 3)',
                'Patch (t, x, y, z, c)', f'(2, 14, 14, 14, 3)', f'(2, 16, 16, 16, 3)',
            ],
            loc='upper left', ncol=1, title="", frameon=False
        )

    if dataset_size is not None:
        ax.set_title(f'Dataset: {dataset_size:,} images')
        savepath = Path(f'{outdir}/{y}_{dataset_size}')
    else:
        savepath = Path(f'{outdir}/{y}')
        
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
    
    outdir = Path("../scaling")
    outdir.mkdir(parents=True, exist_ok=True)
    
    # transformer_scaling = scaling_transformers(dtype=dtype, outdir=outdir)
    vit_scaling = scaling_vit(dtype=dtype, outdir=outdir)
    vit_scaling["number_h100_for_batch"] = np.ceil(vit_scaling["model_training_memory"] + (vit_scaling["memory_per_image"] * batch_size) / 80)
    vit_scaling["cost_h100_for_batch"] = vit_scaling["number_h100_for_batch"] * 37500
    vit_scaling["training_h100_hours_per_step"] = batch_size * vit_scaling["training_time_per_image"] / 3600
    vit_scaling["training_tflops_per_image"] = vit_scaling["training_gflops_per_image"] / 1000
    
    fois = {
        f"training_gflops_per_image": f"Training GFLOPs per image",
        f"training_time_per_image": f"Training H100 seconds per image",
        f"number_h100_for_batch": f"Minimum number of H100s needed for a batch ({batch_size})",
        f"cost_h100_for_batch": f"Cost of H100s needed for a batch ({batch_size}, $37,500 each)",
        f"training_h100_hours_per_step": f"Training H100 hours per batch ({batch_size})",

    }
    for y, ylabel in fois.items():
        plot_parameter_scaling(
            vit_scaling,
            outdir=outdir,
            x="parameters",
            xlabel="Model size (non-embedding parameters)",
            y=y,
            ylabel=ylabel,
        )
    
    fois ={
        f"training_h100_days_per_epoch": f"Training H100 days per epoch",
        f"gpu_compute_cost_per_epoch": f"H100 compute cost per epoch ($2/hr)",
    }
    for dataset_size in [1000000, 10000000, 100000000, 303000000]:
        vit_scaling["training_h100_days_per_epoch"] = dataset_size * vit_scaling["training_time_per_image"] / 3600 / 24
        vit_scaling["multigpu_training_days_per_epoch"] = vit_scaling["training_h100_days_per_epoch"] / vit_scaling["number_h100_for_batch"]
        vit_scaling["multigpu_256_training_days_per_epoch"] = vit_scaling["training_h100_days_per_epoch"] / 256
        vit_scaling["gpu_compute_cost_per_epoch"] = vit_scaling["training_h100_days_per_epoch"] * 24 * 2
        
        for y, ylabel in fois.items():
            plot_parameter_scaling(
                vit_scaling,
                outdir=outdir,
                x="parameters",
                xlabel="Model size (non-embedding parameters)",
                y=y,
                ylabel=ylabel,
                dataset_size=dataset_size
            )
    
    df = vit_scaling.loc[vit_scaling['data'].str.match(r'2D\(rgb\)')]
    datasets = {
        "S": {"dataset": "ImageNet-21K", "dataset_size": 14000000, "epochs": 7, "steps": 1000000, "batch_size": 4096},
        "B": {"dataset": "ImageNet-21K", "dataset_size": 14000000, "epochs": 7, "steps": 1000000, "batch_size": 4096},
        "L": {"dataset": "JFT-300M", "dataset_size": 300000000, "epochs": 14, "steps": 1000000, "batch_size": 4096},
        "H": {"dataset": "JFT-300M", "dataset_size": 300000000,  "epochs": 14, "steps": 1000000, "batch_size": 4096},
        "g": {"dataset": "JFT-1B", "dataset_size": 1000000000, "epochs": 5, "steps": 500000, "batch_size": 32768},
        "G": {"dataset": "JFT-3B", "dataset_size": 3000000000,  "epochs": 5, "steps": 500000, "batch_size": 32768},
        "e": {"dataset": "JFT-3B", "dataset_size": 3000000000,  "epochs": 5, "steps": 1000000, "batch_size": 16384},
        "22B": {"dataset": "JFT-4B", "dataset_size": 4000000000, "epochs": 3, "steps": 177000, "batch_size": 65000},
    }
    cols = list(datasets['S'].keys())
    df[cols] = np.nan
    for k in datasets.keys():
        idx = df.loc[df['class'].str.match(k)].index
        df.loc[idx, cols] =  datasets[k].values()
    
    df["dataset_size"] = df["dataset_size"] // 1000000
    df["training_compute"] = df[f"training_gflops_per_image"] * df["batch_size"] * df["steps"]
    df["training_time"] = df[f"training_time_per_image"] * df["batch_size"] * df["steps"] / 3600 / 24
    
    fois ={
        f"dataset_size": f"Training dataset size (millions of images)",
        f"training_time": f"Training H100 days",
        f"training_compute": f"Training GFLOPs",
    }
    for y, ylabel in fois.items():
        plot_parameter_scaling(
            df,
            outdir=outdir,
            x="parameters",
            xlabel="Model size (non-embedding parameters)",
            y=y,
            ylabel=ylabel,
            published_models_only=True,
            ylog=False if y == "dataset_size" else True,
        )
    
    logger.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
