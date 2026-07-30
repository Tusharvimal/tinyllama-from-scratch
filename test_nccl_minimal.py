import torch
import torch.distributed as dist
import os

def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    print(f"[rank {rank}] before init_process_group", flush=True)
    dist.init_process_group(backend="nccl", device_id=device)
    print(f"[rank {rank}] after init_process_group", flush=True)

    dist.barrier(device_ids=[rank])
    print(f"[rank {rank}] after barrier", flush=True)

    x = torch.ones(4, device=device) * (rank + 1)
    print(f"[rank {rank}] before all_reduce: {x}", flush=True)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    print(f"[rank {rank}] after all_reduce: {x}", flush=True)

    dist.destroy_process_group()
    print(f"[rank {rank}] done", flush=True)

if __name__ == "__main__":
    main()