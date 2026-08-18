import os
import time
import math
import yaml
import contextlib
import torch
import torch.nn as nn
import torch.distributed as dist
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import GradScaler, autocast
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer

from src.trainer.base_trainer import (
    load_config, get_lr, estimate_mfu, WikiTextDataset
)


def setup_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, local_rank, device


def cleanup_distributed():
    dist.destroy_process_group()


class DDPTrainer:
    def __init__(self, config_path: str):
        self.rank, self.world_size, self.local_rank, self.device = setup_distributed()
        self.is_master = (self.rank == 0)
        self.config = load_config(config_path)

        if self.is_master:
            print(f"Distributed Training | world_size={self.world_size}")
            print(f"Device: {self.device} | {torch.cuda.get_device_name(self.device)}")

        self._setup_model()
        self._setup_data()
        self._setup_optimizer()
        self.scaler = GradScaler(device="cuda")
        self.step = 0

    def _setup_model(self):
        cfg = self.config["model"]
        gpt_config = GPT2Config(
            vocab_size=cfg["vocab_size"],
            n_positions=cfg["n_positions"],
            n_embd=cfg["n_embd"],
            n_layer=cfg["n_layer"],
            n_head=cfg["n_head"],
        )
        model = GPT2LMHeadModel(gpt_config).to(self.device)
        self.model = DDP(
            model,
            device_ids=[self.local_rank],
            find_unused_parameters=False,
            bucket_cap_mb=25,
        )

        if self.is_master:
            num_params = sum(p.numel() for p in self.model.parameters()) / 1e6
            print(f"Model: GPT-2 | Params: {num_params:.1f}M | "
                  f"DDP world_size={self.world_size}")

    def _setup_data(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        seq_len = self.config["data"]["seq_length"]
        batch_size = self.config["training"]["batch_size"]

        train_ds = WikiTextDataset("train", seq_len, tokenizer)
        val_ds = WikiTextDataset("validation", seq_len, tokenizer)
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )
        val_sampler = DistributedSampler(
            val_ds,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
        )

        self.train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=0,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=0,
            pin_memory=True,
        )
        self.train_sampler = train_sampler

    def _setup_optimizer(self):
        cfg = self.config["training"]
        decay_params = [p for n, p in self.model.module.named_parameters()
                        if p.requires_grad and p.dim() >= 2]
        no_decay_params = [p for n, p in self.model.module.named_parameters()
                           if p.requires_grad and p.dim() < 2]
        self.optimizer = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": cfg["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=cfg["learning_rate"], betas=(0.9, 0.95), eps=1e-8)

    def add_powersgd_hook(self):
        from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as powerSGD
        state = powerSGD.PowerSGDState(
            process_group=None,
            matrix_approximation_rank=4,
            start_powerSGD_iter=100,
        )
        self.model.register_comm_hook(state, powerSGD.powerSGD_hook)

        if self.is_master:
            print("PowerSGD hook registered | rank=4 | starts at step 100")

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss, count = 0.0, 0
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            with autocast(device_type="cuda"):
                out = self.model(x, labels=y)
            total_loss += out.loss.item()
            count += 1
            if count >= 50:
                break
        self.model.train()
        loss_tensor = torch.tensor(total_loss / count, device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()
        return avg_loss, math.exp(avg_loss)

    def save_checkpoint(self, step: int, loss: float):
        if not self.is_master:
            return
        save_dir = Path(self.config["checkpoint"]["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"ddp_checkpoint_step_{step}.pt"
        torch.save({
            "step": step,
            "model_state_dict": self.model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "config": self.config,
        }, path)
        print(f"  Saved: {path}")

    def train(self, use_compression: bool = False):
        cfg = self.config["training"]
        grad_accum = cfg["gradient_accumulation_steps"]
        log_every = self.config["logging"]["log_every_n_steps"]
        save_every = self.config["checkpoint"]["save_every_n_steps"]

        if use_compression:
            self.add_powersgd_hook()

        if self.is_master:
            mode = "DDP + PowerSGD" if use_compression else "DDP"
            print(f"\n{mode} Training | {cfg['max_steps']} steps | "
                  f"effective batch = {cfg['batch_size'] * grad_accum * self.world_size}")

        self.model.train()
        data_iter = iter(self.train_loader)
        t0 = time.time()
        tokens_seen = 0

        for step in range(cfg["max_steps"]):
            self.train_sampler.set_epoch(step)

            lr = get_lr(step, cfg["warmup_steps"], cfg["max_steps"], cfg["learning_rate"])
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0

            for micro_step in range(grad_accum):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    x, y = next(data_iter)

                x, y = x.to(self.device), y.to(self.device)
                is_last_micro = (micro_step == grad_accum - 1)
                sync_ctx = self.model.no_sync() if not is_last_micro else contextlib.nullcontext()

                with sync_ctx:
                    with autocast(device_type="cuda"):
                        out = self.model(x, labels=y)
                        loss = out.loss / grad_accum
                    self.scaler.scale(loss).backward()
                step_loss += loss.item()

            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), cfg["max_grad_norm"]
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            tokens_seen += cfg["batch_size"] * grad_accum * self.config["data"]["seq_length"]
            elapsed = time.time() - t0
            tok_per_sec = tokens_seen / elapsed
            mfu = estimate_mfu(self.model.module, tok_per_sec, self.device)

            if self.is_master and step % log_every == 0:
                vram = torch.cuda.memory_allocated() / 1e9
                print(
                    f"step {step:5d} | loss {step_loss:.4f} | "
                    f"lr {lr:.2e} | norm {grad_norm:.3f} | "
                    f"tok/s {tok_per_sec:,.0f} | "
                    f"MFU {mfu*100:.1f}% | "
                    f"VRAM {vram:.2f}GB"
                )

            if self.is_master and step > 0 and step % save_every == 0:
                val_loss, ppl = self.evaluate()
                print(f"  [val] loss {val_loss:.4f} | ppl {ppl:.2f}")
                self.save_checkpoint(step, step_loss)

        if self.is_master:
            print("\nDone!")
            self.save_checkpoint(cfg["max_steps"], step_loss)

        cleanup_distributed()