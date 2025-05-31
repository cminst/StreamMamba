import argparse
import os
import sys
import datetime
import torch
import torch.distributed as dist

def setup_for_distributed(is_master: bool):
    """Anything only rank 0 should do (mkdir, logging, etc.)."""
    if is_master:
        print(">>> [rank 0] is master—do any setup here (mkdir, logger, etc.)")

def init_distributed_mode(args):
    """
    Initializes NCCL-based distributed training using env:// as the init_method.
    Expects torchrun to have exported: RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT.
    """
    sys.stderr.write(">>> init_distributed_mode() entry\n"); sys.stderr.flush()

    # 1) Check if torchrun provided the necessary variables
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
        sys.stderr.write(
            f">>> init_distributed_mode(): Detected torchrun → "
            f"rank={args.rank}, world_size={args.world_size}, local_rank={args.local_rank}\n"
        )
        sys.stderr.flush()
    else:
        sys.stderr.write(">>> init_distributed_mode(): Not using distributed mode\n")
        sys.stderr.flush()
        args.distributed = False
        return

    args.distributed = True

    # 2) Bind to the local GPU
    torch.cuda.set_device(args.local_rank)
    sys.stderr.write(f">>> init_distributed_mode(): torch.cuda.set_device({args.local_rank}) done\n")
    sys.stderr.flush()

    # 3) Initialize the NCCL process group. We pass device_id= so NCCL can initialize on that GPU immediately.
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=args.rank,
        device_id=torch.device(f"cuda:{args.local_rank}"),
        timeout=datetime.timedelta(minutes=5),
    )
    sys.stderr.write(">>> init_distributed_mode(): init_process_group() returned\n")
    sys.stderr.flush()

    # 4) Barrier so every rank waits until all ranks are in the PG
    sys.stderr.write(">>> init_distributed_mode(): barrier()\n")
    sys.stderr.flush()
    dist.barrier()
    sys.stderr.write(f">>> init_distributed_mode(): after barrier (rank {args.rank})\n")
    sys.stderr.flush()

    # 5) Do any master/R0 setup
    setup_for_distributed(is_master=(args.rank == 0))
    sys.stderr.write(">>> init_distributed_mode() exit\n")
    sys.stderr.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_dir",
        help="(Dummy) path where we might save embeddings",
    )
    args = parser.parse_args()

    init_distributed_mode(args)

    if not args.distributed:
        print(">>> Running in non‐distributed mode (no torchrun variables found).")
        return

    # At this point, we know we are in distributed mode, NCCL PG is initialized,
    # and torch.cuda.current_device() == args.local_rank.
    print(f">>> [rank {args.rank}] Now {torch.cuda.get_device_name(args.local_rank)} is bound.")
    print(f">>> [rank {args.rank}] world_size: {torch.distributed.get_world_size()}")

    # Put any code here that actually gathers embeddings, runs training, etc.
    # For this minimal example, we’ll just sleep for a second and exit:
    import time
    time.sleep(2)

    dist.barrier()
    if args.rank == 0:
        print(">>>>> All ranks reached the final barrier. Exiting cleanly.")


if __name__ == "__main__":
    main()
