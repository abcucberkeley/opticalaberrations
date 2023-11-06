import logging
from pathlib import Path
from subprocess import call
import cli


def parse_args(args):
    parser = cli.argparser()

    subparsers = parser.add_subparsers(
        help="Arguments for specific action.", dest="cmd"
    )
    subparsers.required = True

    slurm = subparsers.add_parser("slurm", help='use SLURM to submit jobs')

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
        "--gpus", default=1, type=int,
        help='number of GPUs to use for this job'
    )

    slurm.add_argument(
        "--mem", default='500GB', type=str,
        help='requested RAM to use for this job'
    )

    slurm.add_argument(
        "--cpus", default=5, type=int,
        help='number of CPUs to use for this job'
    )

    slurm.add_argument(
        "--nodelist", default=None, type=str,
        help='submit job to a specific node'
    )

    slurm.add_argument(
        "--dependency", default=None, type=str,
        help='submit job with a specific dependency'
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

    lsf = subparsers.add_parser("lsf", help='use LSF to submit jobs')

    lsf.add_argument(
        "script", type=str,
        help='path to script to run'
    )

    lsf.add_argument(
        "--python", default=f'{Path.home()}/anaconda3/envs/ml/bin/python', type=str,
        help='path to ext python to run program with'
    )

    lsf.add_argument(
        "--task", action='append',
        help='any additional flags you want to run the script with'
    )

    lsf.add_argument(
        "--taskname", action='append',
        help='allies name for each task'
    )

    lsf.add_argument(
        "--outdir", default='/groups/betzig/betziglab/thayer/opticalaberrations/models', type=str,
        help='output directory'
    )

    lsf.add_argument(
        "--partition", default='gpu_a100', type=str,
    )

    lsf.add_argument(
        "--gpus", default=1, type=int,
        help='number of GPUs to use for this job'
    )

    lsf.add_argument(
        "--cpus", default=5, type=int,
        help='number of CPUs to use for this job'
    )

    lsf.add_argument(
        "--name", default='train', type=str,
        help='allies name for this job'
    )

    lsf.add_argument(
        "--mem", default='500GB', type=str,
        help='requested RAM to use for this job'
    )

    lsf.add_argument(
        "--dependency", default=None, type=str,
        help='submit job with a specific dependency'
    )

    default = subparsers.add_parser("default", help='run a job using default python')

    default.add_argument(
        "script", type=str,
        help='path to script to run'
    )

    default.add_argument(
        "--python", default=f'{Path.home()}/anaconda3/envs/ml/bin/python', type=str,
        help='path to ext python to run program with'
    )

    default.add_argument(
        "--flags", default='', type=str,
        help='any additional flags you want to run the script with'
    )

    default.add_argument(
        "--outdir", default='/clusterfs/nvme/thayer/opticalaberrations/models', type=str,
        help='output directory'
    )

    default.add_argument(
        "--name", default='train', type=str,
        help='allies name for this job'
    )

    default.add_argument(
        "--exclusive", action='store_true',
        help='allies name for this job'
    )

    return parser.parse_args(args)


def main(args=None):
    args = parse_args(args)
    logging.info(args)

    outdir = Path(f"{args.outdir}/{args.name}").resolve()
    outdir.mkdir(exist_ok=True, parents=True)

    profiler = f"/usr/bin/time -v -o {outdir}/{args.script.split('.')[0]}_profile.log "

    if args.cmd == 'default':
        sjob = profiler
        sjob += f"{args.python} "
        sjob += f"{args.script} "
        sjob += f" --outdir {outdir} {args.flags} 2>&1 | tee {outdir}/{args.script.split('.')[0]}.log"

        logging.info(sjob)
        call([sjob], shell=True)

    elif args.cmd == 'slurm':
        sjob = '/usr/bin/sbatch '
        sjob += f' --qos={args.qos} '
        sjob += f' --partition={args.partition} '

        if args.constraint is not None:
            sjob += f" -C '{args.constraint}' "

        if args.exclusive:
            sjob += f"--exclusive "
        else:
            if args.gpus > 0:
                sjob += f' --gres=gpu:{args.gpus} '

            sjob += f' --cpus-per-task={args.cpus} '
            sjob += f" --mem='{args.mem}' "

        if args.nodelist is not None:
            sjob += f" --nodelist='{args.nodelist}' "

        if args.dependency is not None:
            sjob += f" --dependency={args.dependency} "

        sjob += f" --job-name={args.name} "
        sjob += f" --output={outdir}/{args.script.split('.')[0]}.log"
        sjob += f" --export=ALL,"
        sjob += f"PROFILER='{profiler}',"
        sjob += f"SCRIPT='{args.script}',"
        sjob += f"PYTHON='{args.python}',"
        sjob += f"JOBS='{len(args.task)}',"

        for i, (t, n) in enumerate(zip(args.task, args.taskname)):
            sjob += f"TASK_{i + 1}='{profiler} {args.python} {args.script} {t} --cpu_workers -1 --gpu_workers -1 --outdir {outdir/n}'"
            sjob += ',' if i < len(args.task)-1 else ' '

        sjob += args.job
        logging.info(sjob)
        call([sjob], shell=True)

    elif args.cmd == 'lsf':
        sjob = 'bsub'
        sjob += f' -q {args.partition}'

        if args.gpus > 0:
            if args.partition == 'gpu_a100':
                sjob += f' -gpu "num={args.gpus}:nvlink=yes"'
            else:
                sjob += f' -gpu "num={args.gpus}"'

        if args.dependency is not None:
            sjob += f'-w "done({args.name})"'

        sjob += f' -n {args.cpus}'
        sjob += f" -J {args.name}"
        sjob += f" -o {outdir}/{args.script.split('.')[0]}.log"

        tasks = ""
        for i, (t, n) in enumerate(zip(args.task, args.taskname)):
            tasks += f"{args.python} {args.script} {t} --cpu_workers -1 --gpu_workers -1 --outdir {outdir/n}"
            tasks += ' ; ' if i < len(args.task)-1 else ''

        sjob = f'{sjob} "{tasks}"'
        logging.info(sjob)
        call([sjob], shell=True)

    else:
        logging.error('Unknown action')


if __name__ == "__main__":
    main()
