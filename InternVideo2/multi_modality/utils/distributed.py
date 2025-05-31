import os
import datetime
import torch
import torch.distributed as dist
import logging
import sys
try:
    import deepspeed
except Exception as e:
    print(e)
    print("deepspeed is not installed!!!")
import datetime
from datetime import timedelta

logger = logging.getLogger(__name__)


def setup_for_distributed(is_master):
    import warnings

    builtin_warn = warnings.warn

    def warn(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_warn(*args, **kwargs)

    # Log warnings only once
    warnings.warn = warn
    warnings.simplefilter("once", UserWarning)

    if not is_master:
        logging.disable()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


# def init_distributed_mode(args):
#     sys.stderr.write(">>> init_distributed_mode() entry\n"); sys.stderr.flush()

#     if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#         sys.stderr.write(">>> init_distributed_mode(): Detected torch.distributed.launch (torchrun)\n"); sys.stderr.flush()
#         # job started by torch.distributed.launch
#         args.rank = int(os.environ["RANK"])
#         args.world_size = int(os.environ['WORLD_SIZE'])
#         args.gpu = int(os.environ['LOCAL_RANK'])
#         sys.stderr.write(f">>> init_distributed_mode(): torchrun: Rank {args.rank}, GPU {args.gpu}, World Size {args.world_size}\n"); sys.stderr.flush()

#     elif 'SLURM_PROCID' in os.environ:
#         sys.stderr.write(">>> init_distributed_mode(): Detected SLURM\n"); sys.stderr.flush()
#         # local rank on the current node / global rank
#         local_rank = int(os.environ['SLURM_LOCALID'])
#         global_rank = int(os.environ['SLURM_PROCID'])
#         # number of processes / GPUs per node
#         world_size = int(os.environ["SLURM_NNODES"]) * \
#             int(os.environ["SLURM_TASKS_PER_NODE"][0])

#         # print(world_size) # Change to stderr
#         sys.stderr.write(f">>> init_distributed_mode(): SLURM world_size={world_size}\n"); sys.stderr.flush()

#         args.rank = global_rank
#         args.gpu = local_rank
#         args.world_size = world_size
#         sys.stderr.write(f">>> init_distributed_mode(): SLURM: Rank {args.rank}, GPU {args.gpu}, World Size {args.world_size}\n"); sys.stderr.flush()

#     else:
#         sys.stderr.write(">>> init_distributed_mode(): Not using distributed mode\n"); sys.stderr.flush()
#         logger.info('Not using distributed mode')
#         args.distributed = False
#         return

#     args.distributed = True
#     sys.stderr.write(f">>> init_distributed_mode(): Distributed mode enabled\n"); sys.stderr.flush()


#     # NOTE: While setting device *before* init_process_group is common,
#     # the NCCL warning you see sometimes indicates a need to set it *after*
#     # or rely on the process group to handle device association correctly.
#     # Let's keep it here for now as it was, but add instrumentation.
#     sys.stderr.write(f">>> init_distributed_mode(): Before torch.cuda.set_device({args.gpu})\n"); sys.stderr.flush()
#     torch.cuda.set_device(args.gpu)
#     sys.stderr.write(f">>> init_distributed_mode(): After torch.cuda.set_device()\n"); sys.stderr.flush()


#     args.dist_backend = 'nccl'

#     # dist_url logic is likely skipped with env://, but add instrumentation just in case
#     sys.stderr.write(f">>> init_distributed_mode(): Checking dist_url: {args.dist_url}\n"); sys.stderr.flush()
#     if "tcp" in args.dist_url:
#          sys.stderr.write(f">>> init_distributed_mode(): Handling TCP dist_url: {args.dist_url}\n"); sys.stderr.flush()
#          # Make sure is_port_in_use is defined and accessible
#          # This loop could potentially hang if is_port_in_use is broken or port range is exhausted
#          dist_port = int(args.dist_url.split(":")[-1])
#          sys.stderr.write(f">>> init_distributed_mode(): TCP port check starting from {dist_port}\n"); sys.stderr.flush()
#          while is_port_in_use(dist_port): # is_port_in_use needs definition if not provided
#              dist_port += 10
#              sys.stderr.write(f">>> init_distributed_mode(): Port {dist_port - 10} in use, trying {dist_port}\n"); sys.stderr.flush()

#          args.dist_url = ":".join(args.dist_url.split(":")[:-1] + [str(dist_port)])
#          sys.stderr.write(f">>> init_distributed_mode(): Updated TCP dist_url: {args.dist_url}\n"); sys.stderr.flush()
#     else:
#          sys.stderr.write(f">>> init_distributed_mode(): dist_url is not TCP, using env:// or similar\n"); sys.stderr.flush()


#     logger.info('| distributed init (rank {}): {}'.format(
#         args.rank, args.dist_url))
#     if "SLURM_JOB_ID" in os.environ:
#         logger.info(f"SLURM_JOB_ID {os.environ['SLURM_JOB_ID']}")

#     sys.stderr.write(f">>> init_distributed_mode(): Before init_process_group or deepspeed. Rank {args.rank}\n"); sys.stderr.flush()

#     if hasattr(args, "deepspeed") and args.deepspeed.enable:
#         sys.stderr.write(f">>> init_distributed_mode(): Calling deepspeed.init_distributed with device_id={args.gpu}. Rank {args.rank}\n"); sys.stderr.flush()
#         # Explicitly pass the device_id to DeepSpeed init
#         deepspeed.init_distributed(
#             dist_backend=args.dist_backend,
#             init_method=args.dist_url,
#             world_size=args.world_size,
#             rank=args.rank,
#             device_id=args.gpu,
#             timeout=datetime.timedelta(seconds=7200)
#         )
#         sys.stderr.write(f">>> init_distributed_mode(): deepspeed.init_distributed returned. Rank {args.rank}\n"); sys.stderr.flush()
#     else:
#         sys.stderr.write(f">>> init_distributed_mode(): Calling torch.distributed.init_process_group. Rank {args.rank}\n"); sys.stderr.flush()
#         # This path is likely not taken based on your logs, but keep it correct
#         # If you were using this path, you might need pg_options={"device_id": args.gpu}
#         torch.distributed.init_process_group(
#             backend=args.dist_backend,
#             init_method=args.dist_url,
#             world_size=args.world_size,
#             rank=args.rank,
#             timeout=datetime.timedelta(minutes=60))
#         sys.stderr.write(f">>> init_distributed_mode(): torch.distributed.init_process_group returned. Rank {args.rank}\n"); sys.stderr.flush()

#     sys.stderr.write(f">>> init_distributed_mode(): Before barrier. Rank {args.rank}\n"); sys.stderr.flush()
#     torch.distributed.barrier() # They are hanging here
#     sys.stderr.write(f">>> init_distributed_mode(): After barrier. Rank {args.rank}\n"); sys.stderr.flush()

#     # Make sure setup_for_distributed is defined and accessible
#     setup_for_distributed(args.rank == 0)
#     sys.stderr.write(">>> init_distributed_mode() exit\n"); sys.stderr.flush()



def init_distributed_mode(args):
    """
    Sets up distributed training via either torchrun (torch.distributed.launch) or SLURM.
    Binds each process to its own GPU and initializes the NCCL process group with PGOptions.device_id.

    After calling this function, every rank will have:
      - args.distributed = True
      - args.rank: the global rank (0 .. world_size-1)
      - args.world_size: total number of processes
      - args.gpu: the local GPU index (0 .. gpus_per_node-1)
      - current CUDA device set to args.gpu
      - an active NCCL process group, so dist.barrier() will return.
    """

    sys.stderr.write(">>> init_distributed_mode() entry\n")
    sys.stderr.flush()

    #────── 1) Detect whether we're running under torchrun/torch.distributed.launch ──────
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # torchrun sets these automatically.
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ.get("LOCAL_RANK", 0))
        sys.stderr.write(
            f">>> init_distributed_mode(): Detected torchrun "
            f"→ rank={args.rank}, world_size={args.world_size}, gpu={args.gpu}\n"
        )
        sys.stderr.flush()

    #────── 2) Detect whether we're running under SLURM ──────
    elif "SLURM_PROCID" in os.environ:
        # SLURM_PROCID gives the global rank. SLURM_LOCALID gives the GPU slot on this node.
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = int(os.environ["SLURM_LOCALID"])

        # SLURM_TASKS_PER_NODE may look like "4(x2)" or simply "4".
        # We take the first integer as the per-node GPU count.
        raw_tasks = os.environ.get("SLURM_TASKS_PER_NODE", "1")
        # If it’s “4(x2)”, split off at “(” → “4”
        tasks_per_node = int(raw_tasks.split("(")[0])
        num_nodes = int(os.environ.get("SLURM_NNODES", "1"))
        args.world_size = num_nodes * tasks_per_node

        sys.stderr.write(
            f">>> init_distributed_mode(): Detected SLURM "
            f"→ rank={args.rank}, gpu={args.gpu}, world_size={args.world_size}\n"
        )
        sys.stderr.flush()

    #────── 3) If neither torchrun nor SLURM, disable distributed mode ──────
    else:
        sys.stderr.write(">>> init_distributed_mode(): Not using distributed mode\n")
        sys.stderr.flush()
        args.distributed = False
        return

    # From here on, we are guaranteed to be in distributed mode.
    args.distributed = True

    #────── 4) Bind this process to the designated GPU ──────
    # It’s critical to set this before init_process_group so NCCL knows which GPU to use.
    torch.cuda.set_device(args.gpu)
    sys.stderr.write(f">>> init_distributed_mode(): torch.cuda.set_device({args.gpu}) done\n")
    sys.stderr.flush()

    #────── 5) Build PGOptions and assign device_id ──────
    pg_options = PGOptions()
    # PGOptions.device_id expects a torch.device, not a raw integer.
    pg_options.device_id = torch.device(f"cuda:{args.gpu}")
    sys.stderr.write(f">>> init_distributed_mode(): PGOptions.device_id set to cuda:{args.gpu}\n")
    sys.stderr.flush()

    #────── 6) If using a TCP-based dist_url, possibly shift to an unused port ──────
    # (Optional: Only if you supplied args.dist_url="tcp://host:port")
    if "tcp" in getattr(args, "dist_url", ""):
        # Extract the integer port at the end of dist_url, e.g. "tcp://hostname:12345"
        url_parts = args.dist_url.split(":")
        try:
            base = ":".join(url_parts[:-1])
            port = int(url_parts[-1])
        except ValueError:
            raise ValueError(f"Invalid TCP dist_url format: {args.dist_url}")

        sys.stderr.write(f">>> init_distributed_mode(): Checking TCP port {port}\n")
        sys.stderr.flush()

        # You need to define is_port_in_use(port) elsewhere in your code.
        # It should return True if the port is occupied, False otherwise.
        while is_port_in_use(port):
            old = port
            port += 10
            sys.stderr.write(f">>> init_distributed_mode(): port {old} in use → trying {port}\n")
            sys.stderr.flush()

        args.dist_url = f"{base}:{port}"
        sys.stderr.write(f">>> init_distributed_mode(): Updated TCP dist_url → {args.dist_url}\n")
        sys.stderr.flush()
    else:
        sys.stderr.write(">>> init_distributed_mode(): dist_url is not TCP (using env:// or similar)\n")
        sys.stderr.flush()

    #────── 7) Finally, initialize the NCCL-backed process group ──────
    args.dist_backend = "nccl"

    if hasattr(args, "deepspeed") and getattr(args.deepspeed, "enable", False):
        # If you're using DeepSpeed, call its init_distributed WITHOUT any device_id keyword.
        sys.stderr.write(
            f">>> init_distributed_mode(): Calling deepspeed.init_distributed(...)\n"
            f"    → backend={args.dist_backend}, init_method={args.dist_url}, "
            f"world_size={args.world_size}, rank={args.rank}\n"
        )
        sys.stderr.flush()

        # DeepSpeed’s init_distributed respects PGOptions.device_id internally if launcher has set LOCAL_RANK.
        import deepspeed
        deepspeed.init_distributed(
            dist_backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
            # device_id is handled via PGOptions above, so do NOT pass `device_id=` here
            timeout=datetime.timedelta(seconds=7200)
        )
        sys.stderr.write(">>> init_distributed_mode(): deepspeed.init_distributed() returned\n")
        sys.stderr.flush()

    else:
        # Native PyTorch path:
        sys.stderr.write(
            f">>> init_distributed_mode(): Calling torch.distributed.init_process_group(...)\n"
            f"    → backend={args.dist_backend}, init_method={args.dist_url}, "
            f"world_size={args.world_size}, rank={args.rank}\n"
        )
        sys.stderr.flush()

        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
            timeout=datetime.timedelta(minutes=60),
            pg_options=pg_options,
        )
        sys.stderr.write(">>> init_distributed_mode(): torch.distributed.init_process_group() returned\n")
        sys.stderr.flush()

    #────── 8) Synchronize all ranks ──────
    # Every rank must reach this barrier. If one rank skipped init_process_group,
    # all others will hang here.
    sys.stderr.write(f">>> init_distributed_mode(): Reaching torch.distributed.barrier() ...\n")
    sys.stderr.flush()
    dist.barrier()
    sys.stderr.write(f">>> init_distributed_mode(): After barrier (rank {args.rank})\n")
    sys.stderr.flush()

    #────── 9) Any post‐barrier setup ──────
    # For example, only the master (rank 0) may want to do certain things:
    setup_for_distributed(is_master=(args.rank == 0))
    sys.stderr.write(">>> init_distributed_mode() exit\n")
    sys.stderr.flush()



# Copyright (c) Facebook, Inc. and its affiliates.
# copied from https://github.com/facebookresearch/vissl/blob/master/vissl/utils/distributed_gradients.py
class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


# copied from megavlt
def gather_tensor_along_batch_with_backward(tensor, dim=0):
    world_size = get_world_size()

    if world_size < 2:
        return tensor

    tensor_list = GatherLayer.apply(tensor)
    tensor_list = torch.cat(tensor_list, dim=dim)
    return tensor_list


@torch.no_grad()
def gather_tensor_along_batch(tensor, dim=0):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    world_size = get_world_size()

    if world_size < 2:
        return tensor

    with torch.no_grad():
        tensor_list = []

        for _ in range(world_size):
            tensor_list.append(torch.zeros_like(tensor))

        dist.all_gather(tensor_list, tensor)
        tensor_list = torch.cat(tensor_list, dim=dim)
    return tensor_list
