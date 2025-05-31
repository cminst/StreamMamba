import torch
import torch.distributed as dist
import os

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    print(f"Rank {rank} initialized successfully")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
