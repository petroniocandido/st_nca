import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initializes a distributed environment for training.
# # On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
def setup(rank, world_size):
    torch.distributed.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )

def sync_weights(model, rank, world_size):
    for param in model.parameters():
        if rank == 0:
            # Rank 0 is sending it's own weight
            # to all it's siblings (1 to world_size)
            for sibling in range(1, world_size):
                dist.send(param.data, dst=sibling)
        else:
            # Siblings must recieve the parameters
            dist.recv(param.data, src=0)


def broadcast_global_state(rank, world_size, current_state = None):
    dist.broadcast_object_list(current_state, src=0)
    return current_state


def gather_partial_states(rank, world_size, current_state = None):
    if rank == 0:
        output = [None for _ in range(world_size)]
    else:
        output = None
    dist.all_gather_object(output, current_state)
    return output


def inference(model, initial_state, rank, world_size):
    setup(rank, world_size)
    torch.backends.cudnn.benchmark = True  # Optional performance optimization

    model = model().to(rank)
    # Converting each model TO DDP object
    model = DDP(model, device_ids=[rank])

    #sampler = torch.utils.data.distributed.DistributedSampler(
    #    dataset, num_replicas=world_size, rank=rank
    #)
    #dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(5):
        sampler.set_epoch(epoch)  # Ensure proper shuffling
        for batch, (data, labels) in enumerate(dataloader):
            data, labels = data.to(rank), labels.to(rank)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch % 10 == 0 and rank == 0:
                print(f"Rank {rank}, Epoch {epoch}, Batch {batch}, Loss {loss.item()}")

    torch.distributed.destroy_process_group()  # Graceful shutdown



if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)
