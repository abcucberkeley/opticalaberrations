import matplotlib
matplotlib.use('Agg')

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torch.optim.lr_scheduler import OneCycleLR, LinearLR
from apex.optimizers import FusedLAMB

from ray import init
from ray.train import Checkpoint, CheckpointConfig, RunConfig, ScalingConfig, FailureConfig, report
from ray.train.torch import TorchTrainer
from ray.experimental.tqdm_ray import tqdm

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

import logging
import ujson
import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Optional

from otfnet import OTFNet
from convnext import ConvNeXtV2
import data_utils
import cli

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ray")
os.environ["RAY_DEDUP_LOGS"] = "0"


def summarize_model(model: nn.Module, inputs: tuple, batch_size: int, logdir: Path):
    model_logbook = {}
    model_stats = summary(
        model=model,
        input_size=(1, *inputs[1:]),
        depth=5,
        col_width=25,
        col_names=["kernel_size", "output_size", "num_params"],
        row_settings=["var_names"],
        verbose=0,
        mode='eval'
    )
    train_stats = summary(
        model=model,
        input_size=inputs,
        depth=5,
        col_width=25,
        col_names=["kernel_size", "output_size", "num_params"],
        row_settings=["var_names"],
        verbose=1,
        mode='train'
    )

    model_logbook['training_batch_size'] = batch_size
    model_logbook['input_bytes'] = model_stats.total_input
    model_logbook['total_params'] = model_stats.total_params
    model_logbook['trainable_params'] = model_stats.trainable_params
    model_logbook['param_bytes'] = model_stats.total_param_bytes

    model_logbook['eval_macs'] = model_stats.total_mult_adds
    model_logbook['training_macs'] = train_stats.total_mult_adds

    model_logbook['forward_pass_bytes'] = model_stats.total_output_bytes
    model_logbook['forward_backward_pass_bytes'] = train_stats.total_output_bytes

    model_logbook['eval_model_bytes'] = model_logbook['param_bytes'] + model_logbook['forward_pass_bytes']
    model_logbook['training_model_bytes'] = model_logbook['param_bytes'] + model_logbook['forward_backward_pass_bytes']

    model_logbook['eval_bytes'] = model_logbook['input_bytes'] + model_logbook['eval_model_bytes']
    model_logbook['training_bytes'] = model_logbook['input_bytes'] + model_logbook['training_model_bytes']

    model_logbook['layers'] = {}
    for layer in train_stats.summary_list:
        if layer.is_leaf_layer:
            model_logbook['layers'][f'{layer.class_name}_{layer.var_name}'] = {
                'macs': layer.macs,
                'params': max(layer.num_params, 0),
                'param_bytes': layer.param_bytes,
                'forward_pass_bytes': layer.output_bytes,
                'forward_backward_pass_bytes': layer.output_bytes * 2,  # x2 for gradients
                'output_shape': layer.output_size,
            }

    with Path(logdir / 'model_logbook.json').open('w') as f:
        ujson.dump(
            model_logbook,
            f,
            indent=4,
            sort_keys=False,
            ensure_ascii=False,
            escape_forward_slashes=False
        )


def train_worker(config: dict):

    train_dataloader = data_utils.collect_dataset(
        config['dataset'],
        metadata=False,
        modes=config['pmodes'],
        distribution=config['distribution'],
        embedding=config['embedding'],
        samplelimit=config['samplelimit'],
        max_amplitude=config['max_amplitude'],
        no_phase=config['no_phase'],
        lls_defocus=config['lls_defocus'],
        defocus_only=config['defocus_only'],
        photons_range=(config['min_photons'], config['max_photons']),
        cpu_workers=config['cpu_workers'],
        model_input_shape=config['inputs'],
        batch_size=config['batch_size']
    )
    steps_per_epoch = len(train_dataloader)

    loss_fn = nn.MSELoss(reduction='sum')
    mse_fn = nn.MSELoss(reduction='mean')

    try:  # check if model already exists
        checkpoints = [d for d in config['checkpointdir'].glob('*') if d.is_dir()]
        checkpoints.sort(key=os.path.getctime)
        latest_checkpoint = checkpoints[-1]

        logger.info(f"{config['logdir']/'logbook.csv'}: {Path(config['logdir']/'logbook.csv').exists()}")
        training_history = pd.read_csv(config['logdir']/'logbook.csv', header=0, index_col=0)
        logger.info(f"Training history\n{training_history}")

        starting_epoch = training_history.index.values[-1]
        overall_step = starting_epoch * steps_per_epoch
        best_loss = training_history.loc[starting_epoch, 'epoch_loss']
        starting_epoch += 1
        step_logbook = {}
        epoch_logbook = training_history.to_dict(orient='index')

        logger.info(f"Epochs left {config['epochs'] - starting_epoch}")
        restored = True

    except Exception as e:
        restored = False
        logger.warning(e)
        logger.warning(f"No model found in {config['checkpointdir']}")
        best_loss, overall_step, starting_epoch = np.inf, 0, 0
        step_logbook, epoch_logbook = {}, {}


    accelerator_config = ProjectConfiguration(
        project_dir=config['outdir'],
        logging_dir=config['logdir'],
    )

    accelerator = Accelerator(
        mixed_precision="fp16" if config['amp'] else None,
        log_with="tensorboard",
        project_dir=config['outdir'],
        project_config=accelerator_config,
        device_placement=True,
    )


    if config['network'] == 'otfnet':
        model = OTFNet(
            name='OTFNet',
            input_shape=config['inputs'],
            modes=config['pmodes']
        )
    elif config['network'] == 'convnext':
        model = ConvNeXtV2(
            name='ConvNeXtV2',
            input_shape=config['inputs'],
            modes=config['pmodes'],
            depths=config['repeats'],
            dims=config['heads'],
        )
    # elif network == 'prototype':
    #     model = OpticalTransformer(
    #         name='Prototype',
    #         input_shape=inputs,
    #         roi=roi,
    #         stem=stem,
    #         patches=patches,
    #         heads=heads,
    #         repeats=repeats,
    #         modes=pmodes,
    #         depth_scalar=depth_scalar,
    #         width_scalar=width_scalar,
    #         dropout_rate=dropout,
    #         activation=activation,
    #         mul=mul,
    #         no_phase=no_phase,
    #         positional_encoding_scheme=positional_encoding_scheme,
    #         radial_encoding_period=radial_encoding_period,
    #         radial_encoding_nth_order=radial_encoding_nth_order,
    #         fixed_dropout_depth=fixed_dropout_depth,
    #     )
    # elif network == 'vit':
    #     model = VIT(
    #         name='ViT',
    #         input_shape=inputs,
    #         hidden_size=hidden_size,
    #         roi=roi,
    #         stem=stem,
    #         patches=patches,
    #         heads=heads,
    #         repeats=repeats,
    #         modes=pmodes,
    #         depth_scalar=depth_scalar,
    #         width_scalar=width_scalar,
    #         dropout_rate=dropout,
    #         activation=activation,
    #         mul=mul,
    #         no_phase=no_phase,
    #         positional_encoding_scheme=positional_encoding_scheme,
    #         fixed_dropout_depth=fixed_dropout_depth,
    #     )
    # elif network == 'swin':
    #     model = Swin(
    #         name='Swin',
    #         input_shape=inputs,
    #         hidden_size=hidden_size,
    #         patches=patches,
    #         heads=heads,
    #         repeats=repeats,
    #         modes=pmodes,
    #     )
    #
    # elif network == 'opticalnet':
    #     model = OpticalTransformer(
    #         name='OpticalNet',
    #         input_shape=inputs,
    #         roi=roi,
    #         stem=stem,
    #         patches=patches,
    #         heads=heads,
    #         repeats=repeats,
    #         modes=pmodes,
    #         depth_scalar=depth_scalar,
    #         width_scalar=width_scalar,
    #         dropout_rate=dropout,
    #         activation=activation,
    #         mul=mul,
    #         no_phase=no_phase,
    #         positional_encoding_scheme=positional_encoding_scheme,
    #         radial_encoding_period=radial_encoding_period,
    #         radial_encoding_nth_order=radial_encoding_nth_order,
    #         fixed_dropout_depth=fixed_dropout_depth,
    #     )
    #
    # elif network == 'baseline':
    #     model = Baseline(
    #         name='Baseline',
    #         input_shape=inputs,
    #         modes=pmodes,
    #         repeats=repeats,
    #         projections=heads,
    #     )
    else:
        raise Exception(f'Network "{config["network"]}" is unknown.')

    summarize_model(
        model=model,
        inputs=config['inputs'],
        batch_size=config['batch_size'],
        logdir=config['logdir'],
    )

    if config['opt'] == 'lamb':
        opt = FusedLAMB(model.parameters(), lr=config['lr'], weight_decay=config['wd'], betas=(0.9, 0.99))
    elif config['opt'].lower() == 'adamw':
        opt = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['wd'], betas=(0.9, 0.99))
    else:
        opt = optim.Adam(model.parameters(), lr=config['lr'])

    if config['fixedlr']:
        scheduler = LinearLR(
            opt,
            start_factor=1.0,
            end_factor=1.0,
            total_iters=config['epochs'],
        )
        logger.info(f"Training steps: [{steps_per_epoch * config['epochs']}]")
    else:
        decay_epochs = config['epochs'] - config['warmup']
        total_steps = config['epochs'] * steps_per_epoch
        warmup_steps = config['warmup'] * steps_per_epoch
        decay_steps = total_steps - warmup_steps

        logger.info(
            f"Training [{config['epochs']=}: {total_steps=}] = "
            f"({config['warmup']=}: {warmup_steps=}) + ({decay_epochs=}: {decay_steps=})"
        )

        scheduler = OneCycleLR(
            optimizer=opt,
            max_lr=config['lr'],
            epochs=config['epochs'],
            steps_per_epoch=steps_per_epoch,
            anneal_strategy='cos',
            pct_start=warmup_steps/total_steps,
        )

    logger.info(accelerator.distributed_type)
    model, opt, train_dataloader, scheduler = accelerator.prepare(model, opt, train_dataloader, scheduler)
    accelerator.init_trackers(config['outdir'])
    accelerator.register_for_checkpointing(scheduler)

    if config['finetune'] is not None:
        logger.info(f"Finetuning pretrained model @ {config['finetune']}")
        accelerator.load_state(config['finetune'])
    elif restored:
        logger.info(f"Loading pretrained model @ {latest_checkpoint}")
        accelerator.load_state(latest_checkpoint)

    with accelerator.autocast():
        for epoch in range(starting_epoch, config['epochs'], 1):
            epoch_time = time.time()
            model.train()
            loss, mse = 0, 0

            step_times = []
            for step, (inputs, zernikes) in tqdm(
                enumerate(train_dataloader),
                total=len(train_dataloader),
                desc=f"Epoch {epoch}/{config['epochs']}",
            ):
                opt.zero_grad()
                step_time = time.time()
                lr = scheduler.get_lr()[0]

                outputs = model(inputs)
                step_loss = loss_fn(outputs, zernikes)

                accelerator.backward(step_loss)
                opt.step()
                opt.zero_grad()

                loss += step_loss.detach().float()
                step_mse = mse_fn(outputs, zernikes)
                mse += step_mse.detach().float()

                overall_step += 1
                step_timer = time.time() - step_time
                step_times.append(step_timer)

                step_logbook[overall_step] = {
                    "step_loss": step_loss,
                    "step_mse": step_mse,
                    "step_lr": lr,
                    "step_timer": step_timer,
                }
                accelerator.log(step_logbook[overall_step], step=overall_step)

            scheduler.step()
            loss = loss.item() / steps_per_epoch
            mse = mse.item() / steps_per_epoch
            step_timer = np.mean(step_times)
            epoch_timer = time.time() - epoch_time
            remaining_epochs = config['epochs'] - (epoch + 1)
            eta = epoch_timer * remaining_epochs / 3600

            logger.info(
                f"[Epochs left {remaining_epochs} | ETA {eta:.2f}hrs] ({epoch}): "
                f"{step_timer * 1000:.0f}ms/step - {epoch_timer:.0f}s/epoch - {loss=:.4g} - {mse=:.4g} - {lr=:.4g}",
            )
            epoch_logbook[epoch] = {
                "epoch_loss": loss,
                "epoch_mse": mse,
                "epoch_lr": lr,
                "epoch_timer": epoch_timer,
            }
            df = pd.DataFrame.from_dict(epoch_logbook, orient='index')
            df.to_csv(config['logdir']/'logbook.csv')

            with config['outdir']/'checkpoints' as checkpointdir:

                if accelerator.is_main_process:
                    if loss < best_loss:
                        best_loss = loss
                        accelerator.save_state(config['checkpointdir']/f'best_state')
                        unwrapped_model = accelerator.unwrap_model(model)
                        torch.save(unwrapped_model.state_dict(), config['checkpointdir'] / f"best_model.bin")

                checkpoint = Checkpoint.from_directory(checkpointdir)

            report(
                metrics={
                    "lr": lr,
                    "loss": loss,
                    "mse": mse,
                    "step_timer": step_timer,
                    "epoch_timer": epoch_timer,
                    "eta": eta,
                },
                checkpoint=checkpoint
            )

        accelerator.save_state(config['checkpointdir']/'last_state')
        accelerator.end_training()


def train_model(
    dataset: Path,
    outdir: Path,
    network: str = 'opticalnet',
    distribution: str = '/',
    embedding: str = 'spatial_planes',
    samplelimit: Optional[int] = None,
    max_amplitude: float = 1,
    input_shape: int = 64,
    batch_size: int = 32,
    fixedlr: bool = False,
    opt: str = 'AdamW',
    lr: float = 5e-4,
    wd: float = 5e-6,
    warmup: int = 2,
    epochs: int = 5,
    modes: int = 15,
    pmodes: Optional[int] = None,
    min_photons: int = 1,
    max_photons: int = 1000000,
    no_phase: bool = False,
    lls_defocus: bool = False,
    defocus_only: bool = False,
    finetune: Optional[Path] = None,
    fixed_precision: bool = False,
    roi: Any = None,
    hidden_size: int = 768,
    patches: list = [32, 16, 8, 8],
    heads: list = [2, 4, 8, 16],
    repeats: list = [2, 4, 6, 2],
    depth_scalar: float = 1.,
    width_scalar: float = 1.,
    activation: str = 'gelu',
    dropout: float = .1,
    radial_encoding_period: int = 16,
    radial_encoding_nth_order: int = 4,
    positional_encoding_scheme: str = 'rotational_symmetry',
    fixed_dropout_depth: bool = False,
    stem: bool = False,
    mul: bool = False,
    wavelength: float = .510,
    psf_type: str = '../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat',
    x_voxel_size: float = .125,
    y_voxel_size: float = .125,
    z_voxel_size: float = .2,
    refractive_index: float = 1.33,
    cpu_workers: int = -1,
    gpu_workers: int = 1,
):
    outdir.mkdir(exist_ok=True, parents=True)
    logdir = outdir / 'logs'
    logdir.mkdir(exist_ok=True, parents=True)
    checkpointdir = outdir / 'checkpoints'
    checkpointdir.mkdir(exist_ok=True, parents=True)

    network = network.lower()
    opt = opt.lower()

    if network == 'realspace':
        inputs = (batch_size, input_shape, input_shape, input_shape, 1)
    else:
        inputs = (batch_size, 3 if no_phase else 6, input_shape, input_shape, 1)

    if defocus_only:  # only predict LLS defocus offset
        pmodes = 1
    elif lls_defocus:  # add LLS defocus offset to predictions
        pmodes = modes + 1 if pmodes is None else pmodes + 1
    else:
        pmodes = modes if pmodes is None else pmodes

    amp = True if not fixed_precision else False

    if gpu_workers == -1:
        gpu_workers = torch.cuda.device_count()

    worker_batch_size = batch_size // gpu_workers

    train_loop_config = {
        "epochs": epochs,
        "opt": opt,
        "lr": lr,
        "wd": wd,
        "fixedlr": fixedlr,
        "warmup": warmup,
        "batch_size": worker_batch_size,
        "pmodes": pmodes,
        "inputs": inputs,
        "network": network,
        "distribution": distribution,
        "embedding": embedding,
        "max_amplitude": max_amplitude,
        "samplelimit": samplelimit,
        "min_photons": min_photons,
        "max_photons": max_photons,
        "lls_defocus": lls_defocus,
        "defocus_only": defocus_only,
        "no_phase": no_phase,
        "finetune": finetune,
        "dataset": dataset,
        "outdir": outdir,
        "logdir": logdir,
        "checkpointdir": checkpointdir,
        "cpu_workers": cpu_workers,
        "amp": amp,
        "repeats": repeats,
        "heads": heads,
        "patches": patches,
        "hidden_size": hidden_size,
    }
    scaling_config = ScalingConfig(
        num_workers=gpu_workers,
        use_gpu=True,
        placement_strategy="SPREAD",
    )

    checkpoint_config = CheckpointConfig(
        num_to_keep=3,
        checkpoint_score_attribute='loss',
        checkpoint_score_order='min',
    )

    run_config = RunConfig(
        log_to_file=str(logdir/"log.txt"),
        checkpoint_config=checkpoint_config,
        failure_config=FailureConfig(max_failures=0),
        storage_path=outdir,
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_worker,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    result = trainer.fit()

def eval_model(
    dataset: Path,
    network: str = 'opticalnet',
    distribution: str = '/',
    embedding: str = 'spatial_planes',
    samplelimit: Optional[int] = None,
    max_amplitude: float = 1,
    input_shape: int = 64,
    batch_size: int = 32,
    wavelength: float = .510,
    psf_type: str = '../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat',
    x_voxel_size: float = .125,
    y_voxel_size: float = .125,
    z_voxel_size: float = .2,
    modes: int = 15,
    min_photons: int = 1,
    max_photons: int = 1000000,
    refractive_index: float = 1.33,
    no_phase: bool = False,
    lls_defocus: bool = False,
):
    if network == 'baseline':
        inputs = (input_shape, input_shape, input_shape, 1)
    else:
        inputs = (3 if no_phase else 6, input_shape, input_shape, 1)

    eval_data = data_utils.collect_dataset(
        dataset,
        metadata=False,
        modes=modes,
        distribution=distribution,
        embedding=embedding,
        samplelimit=samplelimit,
        max_amplitude=max_amplitude,
        no_phase=no_phase,
        lls_defocus=lls_defocus,
        photons_range=(min_photons, max_photons),
        model_input_shape=inputs
    )
    

def parse_args(args):
    train_parser = cli.argparser()

    train_parser.add_argument(
        "--network", default='opticalnet', type=str, help="codename for target network to train"
    )

    train_parser.add_argument(
        "--dataset", type=Path, help="path to dataset directory"
    )

    train_parser.add_argument(
        "--outdir", default="../models", type=Path, help='path to save trained models'
    )

    train_parser.add_argument(
        "--batch_size", default=2048, type=int, help="number of images per batch"
    )
    
    train_parser.add_argument(
        "--hidden_size", default=768, type=int, help="hidden size of transformer block"
    )

    train_parser.add_argument(
        "--patches", default='32-16-8-8', help="patch size for transformer-based model"
    )
    
    train_parser.add_argument(
        "--heads", default='2-4-8-16', help="patch size for transformer-based model"
    )
        
    train_parser.add_argument(
        "--repeats", default='2-4-6-2', help="patch size for transformer-based model"
    )

    train_parser.add_argument(
        "--roi", default=None, help="region of interest to crop from the center of the input image"
    )

    train_parser.add_argument(
        "--psf_type", default='../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat',
        help="type of the desired PSF"
    )

    train_parser.add_argument(
        "--x_voxel_size", default=.125, type=float, help='lateral voxel size in microns for X'
    )

    train_parser.add_argument(
        "--y_voxel_size", default=.125, type=float, help='lateral voxel size in microns for Y'
    )

    train_parser.add_argument(
        "--z_voxel_size", default=.2, type=float, help='axial voxel size in microns for Z'
    )

    train_parser.add_argument(
        "--input_shape", default=64, type=int, help="PSF input shape"
    )

    train_parser.add_argument(
        "--modes", default=15, type=int, help="number of modes to describe aberration"
    )

    train_parser.add_argument(
        "--pmodes", default=None, type=int, help="number of modes to predict"
    )

    train_parser.add_argument(
        "--min_photons", default=1, type=int, help="minimum photons for training samples"
    )

    train_parser.add_argument(
        "--max_photons", default=10000000, type=int, help="maximum photons for training samples"
    )

    train_parser.add_argument(
        "--dist", default='/', type=str, help="distribution of the zernike amplitudes"
    )

    train_parser.add_argument(
        "--embedding", default='spatial_planes', type=str, help="embedding option to use for training"
    )

    train_parser.add_argument(
        "--samplelimit", default=None, type=int, help="max number of files to load from a dataset [per bin/class]"
    )

    train_parser.add_argument(
        "--max_amplitude", default=1., type=float, help="max amplitude for the zernike coefficients"
    )

    train_parser.add_argument(
        "--wavelength", default=.510, type=float, help='wavelength in microns'
    )

    train_parser.add_argument(
        "--depth_scalar", default=1., type=float, help='scale the number of blocks in the network'
    )

    train_parser.add_argument(
        "--width_scalar", default=1., type=float, help='scale the number of channels in each block'
    )

    train_parser.add_argument(
        '--fixedlr', action='store_true',
        help='toggle to use a fixed learning rate'
    )

    train_parser.add_argument(
        '--mul', action='store_true',
        help='toggle to multiply ratio (alpha) and phase (phi) in the STEM block'
    )

    train_parser.add_argument(
        "--lr", default=1e-3, type=float,
        help='initial learning rate; optimal config: 1e-3 for LAMB and 5e-4 for AdamW'
    )

    train_parser.add_argument(
        "--wd", default=1e-2, type=float, help='initial weight decay; optimal config: 1e-2 for LAMB and 5e-6 for AdamW'
    )

    train_parser.add_argument(
        "--dropout", default=0.1, type=float, help='initial dropout rate for stochastic depth'
    )

    train_parser.add_argument(
        "--opt", default='adamw', type=str, help='optimizer to use for training'
    )

    train_parser.add_argument(
        "--activation", default='gelu', type=str, help='activation function for the model'
    )

    train_parser.add_argument(
        "--warmup", default=25, type=int, help='number of epochs for the initial linear warmup'
    )

    train_parser.add_argument(
        "--epochs", default=500, type=int, help="number of training epochs"
    )

    train_parser.add_argument(
        "--steps_per_epoch", default=100, type=int, help="number of steps per epoch"
    )

    train_parser.add_argument(
        "--cpu_workers", default=8, type=int, help='number of CPU cores to use'
    )

    train_parser.add_argument(
        "--gpu_workers", default=1, type=int, help='number of GPUs to use'
    )

    train_parser.add_argument(
        '--multinode', action='store_true',
        help='toggle for multi-node/multi-gpu training on a slurm-based cluster'
    )

    train_parser.add_argument(
        '--no_phase', action='store_true',
        help='toggle to use exclude phase from the model embeddings'
    )

    train_parser.add_argument(
        '--lls_defocus', action='store_true',
        help='toggle to also predict the offset between the excitation and detection focal plan'
    )

    train_parser.add_argument(
        '--defocus_only', action='store_true',
        help='toggle to only predict the offset between the excitation and detection focal plan'
    )

    train_parser.add_argument(
        '--positional_encoding_scheme', default='rotational_symmetry', type=str,
        help='toggle to use different radial encoding types/schemes'
    )

    train_parser.add_argument(
        '--radial_encoding_period', default=16, type=int,
        help='toggle to add more periods for each sin/cos layer in the radial encodings'
    )

    train_parser.add_argument(
        '--radial_encoding_nth_order', default=4, type=int,
        help='toggle to define the max nth zernike order in the radial encodings'
    )

    train_parser.add_argument(
        '--stem', action='store_true',
        help='toggle to use a stem block'
    )

    train_parser.add_argument(
        '--fixed_dropout_depth', action='store_true',
        help='toggle to linearly increase dropout rate for deeper layers'
    )

    train_parser.add_argument(
        "--fixed_precision", action='store_true',
        help='optional toggle to disable automatic mixed precision training'
             '(https://www.tensorflow.org/guide/mixed_precision)'
    )

    train_parser.add_argument(
        "--eval", action='store_true',
        help='evaluate on validation set'
    )

    train_parser.add_argument(
        "--finetune", default=None, type=Path,
        help='evaluate on validation set'
    )

    return train_parser.parse_known_args(args)[0]


def main(args=None):
    timeit = time.time()
    args = parse_args(args)
    logger.info(args)

    # head_node = str(os.environ["RAY_HEAD"])
    # port = str(os.environ["RAY_PORT"])
    # dashboard_port = str(os.environ["RAY_DASHBOARD_PORT"])
    init(log_to_driver=True, address='auto')

    if args.eval:
        eval_model(
            dataset=args.dataset,
            network=args.network,
            embedding=args.embedding,
            input_shape=args.input_shape,
            batch_size=args.batch_size,
            psf_type=args.psf_type,
            x_voxel_size=args.x_voxel_size,
            y_voxel_size=args.y_voxel_size,
            z_voxel_size=args.z_voxel_size,
            modes=args.modes,
            min_photons=args.min_photons,
            max_photons=args.max_photons,
            max_amplitude=args.max_amplitude,
            distribution=args.dist,
            samplelimit=args.samplelimit,
            wavelength=args.wavelength,
            no_phase=args.no_phase,
            lls_defocus=args.lls_defocus,
        )
    else:
        train_model(
            dataset=args.dataset,
            embedding=args.embedding,
            outdir=args.outdir,
            network=args.network,
            input_shape=args.input_shape,
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
            patches=[int(i) for i in args.patches.split('-')],
            heads=[int(i) for i in args.heads.split('-')],
            repeats=[int(i) for i in args.repeats.split('-')],
            roi=[int(i) for i in args.roi.split('-')] if args.roi is not None else args.roi,
            psf_type=args.psf_type,
            x_voxel_size=args.x_voxel_size,
            y_voxel_size=args.y_voxel_size,
            z_voxel_size=args.z_voxel_size,
            modes=args.modes,
            activation=args.activation,
            mul=args.mul,
            opt=args.opt,
            lr=args.lr,
            wd=args.wd,
            dropout=args.dropout,
            fixedlr=args.fixedlr,
            warmup=args.warmup,
            epochs=args.epochs,
            pmodes=args.pmodes,
            min_photons=args.min_photons,
            max_photons=args.max_photons,
            max_amplitude=args.max_amplitude,
            distribution=args.dist,
            samplelimit=args.samplelimit,
            wavelength=args.wavelength,
            depth_scalar=args.depth_scalar,
            width_scalar=args.width_scalar,
            no_phase=args.no_phase,
            lls_defocus=args.lls_defocus,
            defocus_only=args.defocus_only,
            radial_encoding_period=args.radial_encoding_period,
            radial_encoding_nth_order=args.radial_encoding_nth_order,
            positional_encoding_scheme=args.positional_encoding_scheme,
            stem=args.stem,
            fixed_dropout_depth=args.fixed_dropout_depth,
            finetune=args.finetune,
            fixed_precision=args.fixed_precision,
            cpu_workers=args.cpu_workers,
            gpu_workers=args.gpu_workers,
        )

    logger.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")


if __name__ == "__main__":
    main()
