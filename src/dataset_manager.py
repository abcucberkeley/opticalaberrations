import logging
import time
from pathlib import Path
from subprocess import call

import cli
from utils import multiprocess
from functools import partial
from dataset import create_synthetic_sample
import numpy as np


def parse_args(args):
    parser = cli.argparser()

    subparsers = parser.add_subparsers(
        help="Arguments for specific action.", dest="dtype"
    )
    subparsers.required = True

    slurm = subparsers.add_parser("slurm", help='use SLURM to submit jobs')

    slurm.add_argument(
        "--script", type=str, default='dataset.py',
        help='path to script to run'
    )

    slurm.add_argument(
        "--python", default=f'{Path.home()}/anaconda3/envs/deep/bin/python', type=str,
        help='path to ext python to run program with'
    )

    slurm.add_argument(
        "--task", action='append',
        help='any additional flags you want to run the script with'
    )

    slurm.add_argument(
        "--taskname", action='append',
        help='allies name for each task'
    )

    slurm.add_argument(
        "--outdir", default='/clusterfs/fiona/thayer/opticalaberrations/dataset', type=str,
        help='output directory'
    )

    slurm.add_argument(
        "--partition", default='abc', type=str,
    )

    slurm.add_argument(
        "--qos", default='abc_normal', type=str,
        help='use `abc_high` for unlimited runtime',
    )

    slurm.add_argument(
        "--cpus", default=1, type=int,
        help='number of CPUs to use for this job'
    )

    slurm.add_argument(
        "--mem", default='20G', type=str,
        help='requested RAM to use for this job'
    )

    slurm.add_argument(
        "--time", default='1:00:00', type=str,
        help='walltime limit for this job'
    )

    slurm.add_argument(
        "--name", default='sample', type=str,
        help='allies name for this job'
    )

    slurm.add_argument(
        "--job", default='job.slm', type=str,
        help='path to slurm job template'
    )

    default = subparsers.add_parser("default", help='run a job using default python')

    default.add_argument(
        "--script", type=str, default='dataset.py',
        help='path to script to run'
    )

    default.add_argument(
        "--python", default=f'~/anaconda3/bin/python', type=str,
        help='path to ext python to run program with'
    )

    default.add_argument(
        "--flags", default='', type=str,
        help='any additional flags you want to run the script with'
    )

    default.add_argument(
        "--outdir", default='../dataset', type=str,
        help='output directory'
    )

    default.add_argument(
        "--name", default='dataset', type=str,
        help='allies name for this job'
    )

    threads = subparsers.add_parser("threads")
    threads.add_argument(
        "--outdir", default='../dataset', type=str,
        help='output directory'
    )

    threads.add_argument(
        "--dist", default='mixed', type=str,
        help='target distribution'
    )

    test = subparsers.add_parser("test")
    test.add_argument(
        "--outdir", default='../dataset', type=str,
        help='output directory'
    )

    test.add_argument(
        "--dist", default='mixed', type=str,
        help='target distribution'
    )

    return parser.parse_args(args)


def main(args=None):
    args = parse_args(args)

    outdir = Path(f"{args.outdir}/{args.name}")
    outdir.mkdir(exist_ok=True, parents=True)
    profiler = f"/usr/bin/time -v -o {outdir}/{args.script.split('.')[0]}_profile.log "

    if args.dtype == 'slurm':
        sjob = '/usr/bin/sbatch '
        sjob += f' --qos={args.qos} '
        sjob += f' --partition={args.partition} '
        sjob += f' --cpus-per-task={args.cpus} '
        sjob += f" --mem='{args.mem}' "
        sjob += f" --job-name={args.name} "
        sjob += f" --output={outdir}/{args.name.replace('/', '_')}.log"
        sjob += f" --time='{args.time}' "
        sjob += f" --export=ALL,"
        sjob += f"PROFILER='{profiler}',"
        sjob += f"SCRIPT='{args.script}',"
        sjob += f"PYTHON='{args.python}',"
        sjob += f"JOBS='{len(args.task)}',"

        for i, (t, n) in enumerate(zip(args.task, args.taskname)):
            sjob += f"TASK_{i + 1}='{profiler} {args.python} {args.script} --cpu_workers 1 --outdir {outdir} {t}'"
            sjob += ',' if i < len(args.task)-1 else ' '

        sjob += args.job
        call([sjob], shell=True)

    elif args.dtype == 'default':
        sjob = f"{args.python} "
        sjob += f"{args.script} "
        sjob += f" --outdir {args.outdir} {args.flags} 2>&1 | tee {outdir}/{args.name.replace('/', '_')}.log"

        print(sjob)
        print(args.script)
        print(args.flags)
        call([sjob], shell=True)

    elif args.dtype == 'threads':

        for amp1, amp2 in zip(np.arange(0, .3, .05), np.arange(.05, .35, .05)):

            worker = partial(
                create_synthetic_sample,
                outdir=Path(args.outdir),
                otf=True,
                modes=60,
                input_shape=64,
                distribution=args.dist,
                min_amplitude=amp1,
                max_amplitude=amp2,
                max_jitter=0,
                x_voxel_size=.15,
                y_voxel_size=.15,
                z_voxel_size=.6,
                min_psnr=20,
                max_psnr=40,
                lam_detection=.605,
                refractive_index=1.33,
                na_detection=1.0,
                cpu_workers=1,
            )
            jobs = list(map(str, range(10**3)))
            multiprocess(worker, jobs, desc=f"AMPS [{round(amp1, 2)} :: {round(amp2, 2)}]", cores=-1)

    elif args.dtype == 'test':

        for mpsnr, xpsnr in zip(range(1, 100, 10), range(10, 110, 10)):
            for amp1, amp2 in zip(np.arange(0, .3, .05), np.arange(.05, .35, .05)):

                worker = partial(
                    create_synthetic_sample,
                    outdir=Path(args.outdir),
                    otf=False,
                    modes=60,
                    input_shape=64,
                    distribution=args.dist,
                    min_amplitude=amp1,
                    max_amplitude=amp2,
                    max_jitter=1,
                    x_voxel_size=.15,
                    y_voxel_size=.15,
                    z_voxel_size=.6,
                    min_psnr=mpsnr,
                    max_psnr=xpsnr,
                    lam_detection=.605,
                    refractive_index=1.33,
                    na_detection=1.0,
                    cpu_workers=1,
                )
                jobs = list(map(str, range(100)))
                multiprocess(
                    worker,
                    jobs,
                    desc=f"PSNR[{mpsnr}::{xpsnr}] - AMPS[{round(amp1, 2)}::{round(amp2, 2)}]", cores=-1
                )

    else:
        logging.error('Unknown action')


if __name__ == "__main__":
    main()
