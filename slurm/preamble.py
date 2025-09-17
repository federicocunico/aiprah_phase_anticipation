def get_resources():
    import os
    if os.environ.get("RANK"): # launched with torchrun (python -m torch.distributed.run)
        rank = int(os.getenv("RANK"))
        local_rank = int(os.getenv("LOCAL_RANK"))
        world_size = int(os.getenv("WORLD_SIZE"))
    elif os.environ.get("OMPI_COMMAND"): # launched with mpirun
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    else: # launched with srun (SLURM)
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NPROCS"])
    return rank, local_rank, world_size
import torch

rank, local_rank, world_size = get_resources()
torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
torch.cuda.set_device(local_rank)
