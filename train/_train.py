
import os
import time
import json
import random
from dataclasses import dataclass, asdict
from typing import Optional

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from transformers import get_scheduler

try:
    from _dataset import Dataset, collate_fn
except:
    pass

@dataclass
class TrainConfig:
    seed: int = 0
    ddp: bool = False
    learning_rate: float = 5e-05
    train_batch_size: int = 80
    valid_batch_size: int = 80
    lr_scheduler_type: str = 'linear'
    num_epochs: int = 60
    num_warmup_steps: int = 2000//80*5
    max_train_steps: Optional[int] = None
    max_valid_steps: Optional[int] = None
    max_length: int = 512

    
def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def train(tokenizer, model, df_train, df_valid, rank=0, world_size=0, cfg: TrainConfig = TrainConfig(), output_dir: str = './out', dry: bool = False):
    print(f"[{rank}] Starting training process.")
    os.makedirs(output_dir, exist_ok=True)
    
    if rank == 0:
        with open(os.path.join(output_dir, 'train_config.json'), 'w') as fp:
            json.dump(asdict(cfg), fp)
            
    set_seed(cfg.seed+rank)
    torch.cuda.set_device(rank)

    ds_train = Dataset(df_train, tokenizer, max_length=cfg.max_length)
    ds_valid = Dataset(df_valid, tokenizer, max_length=cfg.max_length)

    if cfg.ddp:
        train_sampler = DistributedSampler(ds_train, num_replicas=world_size, rank=rank, shuffle=True)
        valid_sampler = DistributedSampler(ds_valid, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        valid_sampler = None
        
    train_dataloader = DataLoader(ds_train, sampler=train_sampler, batch_size=cfg.train_batch_size, collate_fn=collate_fn)
    valid_dataloader = DataLoader(ds_valid, sampler=valid_sampler, batch_size=cfg.valid_batch_size, collate_fn=collate_fn, drop_last=True)

    # Model
    model = model.to(rank)
    if cfg.ddp:
        model = DDP(model, device_ids=[rank])

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)
    lr_scheduler = get_scheduler(
        cfg.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.num_warmup_steps,
        num_training_steps=min(len(train_dataloader),cfg.max_train_steps or len(train_dataloader))*cfg.num_epochs+1
    )

    records = []
    best_loss = None
    
    for epoch in range(1, cfg.num_epochs+1):
        st = time.time()
        step = 1
        train_loss = 0

        model.train()
        for batch in tqdm(train_dataloader, disable=rank!=0):
            lr_scheduler.step()
            lr = lr_scheduler.get_last_lr()[0]
            if dry:
                continue
            batch = {k: v.to(rank) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss = loss.item()
            if cfg.max_train_steps is not None and step >= cfg.max_train_steps:
                break
            step += 1

        valid_loss = 0
        valid_accuracy = 0
        cnt = 0
        step = 1
        
        model.eval()
        with torch.no_grad():
            for batch in tqdm(valid_dataloader, disable=rank!=0):
                cnt += 1
                if dry:
                    continue
                batch = {k: v.to(rank) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                valid_loss += loss.item()
                # score: accuracy
                pred = torch.argmax(F.softmax(outputs.logits), dim=1)
                valid_accuracy += torch.mean((pred == batch['labels']).float()).item()
                step += 1
                if cfg.max_valid_steps is not None and step >= cfg.max_valid_steps:
                    break
            valid_loss /= cnt
            valid_accuracy /= cnt
            
        if rank == 0:
            dt = time.time() - st
            records.append({
                'epoch': epoch,
                'lr': lr,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'valid_accuracy': valid_accuracy
            })
            print(f"[{epoch}/{cfg.num_epochs}] Train Loss: {train_loss:.4f} Valid Loss: {valid_loss:.4f} Accuracy: {valid_accuracy:.4f} Cost: {dt} s")
            if best_loss is None or valid_loss < best_loss:
                best_loss = valid_loss
                # save model
                print('Save model')
                save_path = os.path.join(output_dir, 'best')
                if cfg.ddp:
                    model.module.save_pretrained(save_path)
                else:
                    model.save_pretrained(save_path)
    
    if rank == 0:
        # save last
        save_path = os.path.join(output_dir, 'last')
        if cfg.ddp:
            model.module.save_pretrained(save_path)
        else:
            model.save_pretrained(save_path)
        # save training log
        df_log = pd.DataFrame.from_records(records)
        df_log.to_csv(os.path.join(output_dir, 'train_log.csv'), index=False)
