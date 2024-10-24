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
        help='name for each task'
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
        "--gpus", default=0, type=int,
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
        "--nodes", default=1, type=int,
        help='number of node/host (s)'
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
        help='name for this job'
    )

    slurm.add_argument(
        "--job", default='job.slm', type=str,
        help='path to slurm job template'
    )

    slurm.add_argument(
        "--constraint", default=None, type=str,
        help='select a specific node type eg. titan'
    )

    slurm.add_argument(
        "--exclusive", action='store_true',
        help='exclusive access to all resources on the requested node'
    )

    slurm.add_argument(
        "--timelimit", default=None, type=str,
        help='execution timelimit string'
    )

    slurm.add_argument(
        "--apptainer", default=None, type=str,
        help='path to apptainer (*.sif) image to use for this job'
    )

    slurm.add_argument(
        "--ray", action='store_true',
        help='use a ray cluster'
    )

    slurm.add_argument(
        "--ray_template", default='ray_slurm_cluster.sh', type=str,
        help='path to bash script to start a ray cluster'
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
        help='name for each task'
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
        help='number of GPUs per node/host'
    )

    lsf.add_argument(
        "--cpus", default=2, type=int,
        help='number of CPUs to use for this job'
    )

    lsf.add_argument(
        "--nodes", default=1, type=int,
        help='number of node/host (s)'
    )

    lsf.add_argument(
        "--name", default='train', type=str,
        help='name for this job'
    )

    lsf.add_argument(
        "--mem", default='500GB', type=str,
        help='requested RAM to use for this job'
    )

    lsf.add_argument(
        "--dependency", default=None, type=str,
        help='submit job with a specific dependency'
    )

    lsf.add_argument(
        "--timelimit", default=None, type=str,
        help='execution timelimit string'
    )

    lsf.add_argument(
        "--exclusive", action='store_true',
        help='name for this job'
    )

    lsf.add_argument(
        "--apptainer", default=None, type=str,
        help='path to apptainer (*.sif) image to use for this job'
    )

    lsf.add_argument(
        "--ray", action='store_true',
        help='use a ray cluster'
    )

    lsf.add_argument(
        "--span", action='store_true',
        help='use span argument to allocate all nodes at once in a single job, otherwise allocate one node at a time'
    )

    lsf.add_argument(
        "--parallel", action='store_true',
        help='use parallel queue to multiple nodes for a single job'
    )

    lsf.add_argument(
        "--ray_template", default='ray_lsf_cluster.sh', type=str,
        help='path to bash script to start a ray cluster'
    )


    local = subparsers.add_parser("local", help='use docker to run jobs on your local machine')

    local.add_argument(
        "script", type=str,
        help='path to script to run'
    )

    local.add_argument(
        "--image", default='ghcr.io/abcucberkeley/opticalaberrations:develop_Torch_CUDA_12_3', type=str,
        help='docker image to use for this job'
    )

    local.add_argument(
        "--python", default=f'python', type=str,
        help='path to ext python to run program with'
    )

    local.add_argument(
        "--task", action='append',
        help='any additional flags you want to run the script with'
    )

    local.add_argument(
        "--taskname", action='append',
        help='name for each task'
    )

    local.add_argument(
        "--outdir", default='../models', type=str,
        help='output directory'
    )

    local.add_argument(
        "--name", default='train', type=str,
        help='name for this job'
    )

    return parser.parse_args(args)


def main(args=None):
    args = parse_args(args)

    outdir = Path(f"{args.outdir}/{args.name}").resolve()

    try:
        outdir.mkdir(exist_ok=True, parents=True)
    except PermissionError:
        pass

    profiler = f"/usr/bin/time -v -o {outdir}/{args.script.split('.')[0]}_profile.log "

    if args.cmd == 'local':
        outdir = f"/app/opticalaberrations/models/{args.name}"

        tasks = ""
        for i, (t, n) in enumerate(zip(args.task, args.taskname)):
            tasks = f" {args.python} {args.script} {t} --cpu_workers -1 --gpu_workers -1 --outdir {outdir}/{n}"
            tasks += ' ; ' if i < len(args.task)-1 else ''

        docker_job = "docker run --network host -u 1000 --privileged --rm -it --gpus all --ipc=host --env PYTHONUNBUFFERED=1 --pull missing"
        docker_job += f" -v {Path.cwd().parent}:/app/opticalaberrations -w /app/opticalaberrations/src "
        docker_job += f" {args.image} "
        docker_job += f" \"{tasks}\" "

        print(docker_job)
        call([docker_job], shell=True)

    elif args.cmd == 'slurm':
        app = f'apptainer exec --nv --bind /clusterfs:/clusterfs \"{args.apptainer}\"'

        if args.nodes > 1:
            gpu_workers = args.nodes * args.gpus
        else:
            gpu_workers = args.gpus

        if args.nodes > 1:
            cpu_workers = args.nodes * args.cpus
        else:
            cpu_workers = args.cpus

        env = 'python' if args.apptainer is not None else args.python

        sjob = "/usr/bin/sbatch"
        sjob += f" --qos={args.qos}"
        sjob += f" --partition={args.partition}"

        if args.constraint is not None:
            sjob += f" -C '{args.constraint}'"

        if args.exclusive:
            sjob += f" --exclusive"
        else:

            if args.nodes > 1:
                sjob += f" --nodes {args.nodes}"
                sjob += f" --cpus-per-task={args.cpus}"
            else:
                sjob += f" -n {cpu_workers}"

            if args.gpus > 0:
                sjob += f" --gres=gpu:{args.gpus}"

            sjob += f" --mem='{args.mem}'"

        if args.nodelist is not None:
            sjob += f" --nodelist='{args.nodelist}'"

        if args.dependency is not None:
            sjob += f" --dependency={args.dependency}"

        if args.timelimit is not None:
            sjob += f" --time={args.timelimit}"

        sjob += f" --job-name={args.name}"
        sjob += f" --output={outdir}/{args.script.split('.')[0]}.log"
        sjob += f" --export=ALL"

        tasks = ""
        for i, (t, n) in enumerate(zip(args.task, args.taskname)):
            tasks = f" {env} {args.script} {t} --cpu_workers {cpu_workers} --gpu_workers {gpu_workers} --outdir {outdir/n}"
            tasks += ' ; ' if i < len(args.task)-1 else ''

        if args.ray and args.apptainer is not None:
            sjob += f' --wrap=\" ./{args.ray_template} -w \" {app} {tasks} \" \" '
        elif args.apptainer is not None:
            sjob += f' --wrap=\" {app} {tasks} \"'
        else:
            sjob += f' --wrap=\" {tasks} \"'

        print(sjob)
        call([sjob], shell=True)

    elif args.cmd == 'lsf':
        app = f'apptainer exec --nv --bind /groups/betzig/betziglab:/groups/betzig/betziglab \"{args.apptainer}\"'
        env = 'python' if args.apptainer is not None else args.python

        sjob = 'bsub'
        sjob += f' -q {args.partition}'

        if args.partition == 'gpu_h100_parallel':
            args.gpus, args.cpus = 8, 96
            if args.nodes > 1:
                gpu_workers = args.nodes * args.gpus
                cpu_workers = args.nodes * args.cpus
            else:
                raise ValueError('Nodes must be greater than 1')

            sjob += f" -app parallel-96"
            sjob += f" -n {cpu_workers}"
            sjob += f' -gpu "num={args.gpus}:mode=shared"'
        else:
            if args.nodes > 1:
                gpu_workers = args.nodes * args.gpus
            else:
                gpu_workers = args.gpus

            if args.nodes > 1:
                cpu_workers = args.nodes * args.cpus
            else:
                cpu_workers = args.cpus

            if args.span:
                sjob += f" -n {cpu_workers}"
                sjob += f' -R "span[ptile={args.cpus}]"'
            else:
                sjob += f" -n {args.cpus}"

            if args.gpus > 0:
                if args.partition == 'gpu_a100':
                    sjob += f' -gpu "num={args.gpus}:nvlink=yes"'
                else:
                    sjob += f' -gpu "num={args.gpus}:mode=shared"'

        if args.dependency is not None:
            sjob += f' -w "done({args.name})"'

        if args.timelimit is not None:
            sjob += f" --We {args.timelimit} "

        sjob += f" -J {args.name}"
        sjob += f" -o {outdir}/{args.script.split('.')[0]}.log"

        tasks = ""
        for i, (t, n) in enumerate(zip(args.task, args.taskname)):
            tasks = f" {env} {args.script} {t} --cpu_workers {cpu_workers} --gpu_workers {gpu_workers} --outdir {outdir/n}"
            tasks += ' ; ' if i < len(args.task)-1 else ''

        if args.ray and args.apptainer is not None:
            sjob += f' ./{args.ray_template} -n {args.nodes} -c {args.cpus} -g {args.gpus} -o {outdir}  -w \" {app} {tasks} \" '
        elif args.apptainer is not None:
            sjob += f' {app} {tasks} '
        else:
            sjob += f' {tasks} '

        print(sjob)
        call([sjob], shell=True)

    else:
        print('Unknown action')


if __name__ == "__main__":
    main()
