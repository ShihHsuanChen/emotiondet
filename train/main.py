
import os
import random
import time
import numpy as np
import torch
import torch.distributed as dist

try:
    from _model import create_model
    from _train import train, TrainConfig
    from _utils import read_data
except:
    pass


def setup(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    return


def main(rank: int = 0, world_size: int = 1, debug: bool = False, dry: bool = False):
    """
    Deberta:
    - max_length: 72, batch_size: 128
    - max_length: 512, batch_size: 8
    DebertaV2:
    - max_length: 72, batch_size: 80 (13.7GB)
    - max_length: 512, batch_size: 8 (11.5GB)
    """
    common = dict(
        seed=567,
        max_length=72,
        train_batch_size=80,
        valid_batch_size=80,
    )
    if debug:
        cfg = TrainConfig(ddp=True, num_epochs=2, max_train_steps=2, max_valid_steps=2, **common)
    else:
        cfg = TrainConfig(ddp=True, num_epochs=20, **common)

    if cfg.ddp:
        setup(rank, world_size)
        time.sleep(rank)
        _, total = torch.cuda.mem_get_info(device=rank)
        print(f"Rank: {rank}, World size: {world_size}, GPU memory: {total / 1024**3:.2f}GB", flush=True)
        time.sleep(world_size - rank)

    datadir = '/kaggle/input/setfit-emotion/'
    data_dict = read_data(
        os.path.join(datadir, 'train.jsonl'),
        os.path.join(datadir, 'validation.jsonl'),
        os.path.join(datadir, 'test.jsonl'),
    )
    df_train = data_dict['train']
    df_valid = data_dict['valid']
    df_test = data_dict['test']

    try:
        model, tokenizer = create_model()
        train(tokenizer, model, df_train, df_valid, rank=rank, world_size=world_size, cfg=cfg, dry=dry)
    finally:
        if cfg.ddp:
            cleanup()


def cleanup():
    dist.barrier()
    dist.destroy_process_group()
    return


if __name__ == "__main__":
    # GPU Specs
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Run
    KERNEL_TYPE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE','').upper()
    print(KERNEL_TYPE)
    # main(rank, world_size, debug=False, dry=True)
    main(rank, world_size, debug=KERNEL_TYPE != 'BATCH')
