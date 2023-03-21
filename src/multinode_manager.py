import logging
from pathlib import Path
from subprocess import call

import cli


def parse_args(args):
    slurm = cli.argparser()

    slurm.add_argument(
        "script", type=str,
        help='path to script to run'
    )

    slurm.add_argument(
        "--python", default=f'{Path.home()}/anaconda3/envs/ml/bin/python', type=str,
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
        "--outdir", default='/clusterfs/nvme/thayer/opticalaberrations/models', type=str,
        help='output directory'
    )

    slurm.add_argument(
        "--partition", default='abc', type=str,
    )

    slurm.add_argument(
        "--qos", default='abc_high', type=str,
        help='using `abc_high` for unlimited runtime',
    )

    slurm.add_argument(
        "--nodes", default=1, type=int,
        help='number of nodes to use for this job'
    )

    slurm.add_argument(
        "--mem", default='500GB', type=str,
        help='requested RAM to use for this job'
    )

    slurm.add_argument(
        "--gpus", default=1, type=int,
        help='number of GPUs per node to use for this job'
    )

    slurm.add_argument(
        "--cpus", default=4, type=int,
        help='number of CPUs per node to use for this job'
    )

    slurm.add_argument(
        "--name", default='train', type=str,
        help='allies name for this job'
    )

    slurm.add_argument(
        "--job", default='job.slm', type=str,
        help='path to slurm job template'
    )

    slurm.add_argument(
        "--constraint", default=None, type=str,
        help='select a specific node type eg. titan'
    )

    return slurm.parse_args(args)


def main(args=None):
    args = parse_args(args)
    logging.info(args)

    outdir = Path(f"{args.outdir}/{args.name}")
    outdir.mkdir(exist_ok=True, parents=True)
    profiler = f"/usr/bin/time -v -o {outdir}/{args.script.split('.')[0]}_profile.log "

    sjob = '/usr/bin/sbatch '
    sjob += f' --qos={args.qos} '
    sjob += f' --partition={args.partition} '
    sjob += f' --nodes={args.nodes} '
    sjob += f' --ntasks-per-node=1 '

    if args.constraint is not None:
        sjob += f" -C '{args.constraint}' "

    if args.gpus > 0:
        sjob += f' --gres=gpu:{args.gpus} '

    sjob += f" --mem='{args.mem}' "
    sjob += f' --cpus-per-task={args.cpus} '
    sjob += f" --job-name={args.name} "
    sjob += f" --output={outdir}/{args.script.split('.')[0]}.log"
    sjob += f" --export=ALL,"
    sjob += f"PROFILER='{profiler}',"
    sjob += f"SCRIPT='{args.script}',"
    sjob += f"PYTHON='{args.python}',"
    sjob += f"JOBS='{len(args.task)}',"

    for i, (t, n) in enumerate(zip(args.task, args.taskname)):
        sjob += f"TASK_{i + 1}='srun {profiler} {args.python} {args.script} --cpu_workers -1 --gpu_workers -1 --outdir {outdir/n} {t}'"
        sjob += ',' if i < len(args.task)-1 else ' '

    sjob += args.job
    call([sjob], shell=True)


if __name__ == "__main__":
    main()
