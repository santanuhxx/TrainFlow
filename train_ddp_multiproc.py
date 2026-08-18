import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import GradScaler, autocast
import time
import math
import argparse
from pathlib import Path

from src.trainer.base_trainer import load_config, get_lr, estimate_mfu, WikiTextDataset
from src.checkpoint.checkpoint_manager import CheckpointManager
from src.trainer.config_validator import validate_config
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer


def setup():
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), local_rank


def cleanup():
    dist.destroy_process_group()


def train(config_path: str, use_compression: bool = False):
    rank, world_size, local_rank = setup()
    is_master = rank == 0
    device = torch.device(f"cuda:{local_rank}")

    config = load_config(config_path)
    if is_master:
        validate_config(config)
        print(f"DDP Training | world_size={world_size} | device={device}")

    # Model
    cfg = config["model"]
    gpt_config = GPT2Config(
        vocab_size=cfg["vocab_size"],
        n_positions=cfg["n_positions"],
        n_embd=cfg["n_embd"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
    )
    model = GPT2LMHeadModel(gpt_config).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False, bucket_cap_mb=25)

    if use_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as powerSGD
        state = powerSGD.PowerSGDState(
            process_group=None,
            matrix_approximation_rank=4,
            start_powerSGD_iter=100,
        )
        model.register_comm_hook(state, powerSGD.powerSGD_hook)
        if is_master:
            print("PowerSGD compression enabled | rank=4")

    # Data
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    seq_len = config["data"]["seq_length"]
    batch_size = config["training"]["batch_size"]

    train_ds = WikiTextDataset("train", seq_len, tokenizer)
    val_ds = WikiTextDataset("validation", seq_len, tokenizer)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
        num_workers=config["data"]["num_workers"], pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, sampler=val_sampler,
        num_workers=2, pin_memory=True
    )

    # Optimizer
    tcfg = config["training"]
    decay_params = [p for n, p in model.module.named_parameters()
                    if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.module.named_parameters()
                       if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": tcfg["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=tcfg["learning_rate"], betas=(0.9, 0.95), eps=1e-8)

    scaler = GradScaler(device="cuda")
    ckpt_manager = CheckpointManager(
        save_dir=config["checkpoint"]["save_dir"],
        keep_last_n=config["checkpoint"]["keep_last_n"],
    )

    # Resume
    start_step = 0
    if is_master:
        state = ckpt_manager.load_latest(device)
        if state:
            model.module.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            start_step = state["step"] + 1
            print(f"Resumed from step {state['step']}")

    # Broadcast start_step to all ranks
    start_step_tensor = torch.tensor(start_step, device=device)
    dist.broadcast(start_step_tensor, src=0)
    start_step = start_step_tensor.item()

    grad_accum = tcfg["gradient_accumulation_steps"]
    log_every = config["logging"]["log_every_n_steps"]
    save_every = config["checkpoint"]["save_every_n_steps"]

    if is_master:
        mode = "DDP + PowerSGD" if use_compression else "DDP"
        print(f"\n{mode} | steps {start_step} → {tcfg['max_steps']} | "
              f"effective batch = {batch_size * grad_accum * world_size}")

    model.train()
    data_iter = iter(train_loader)
    t0 = time.time()
    tokens_seen = 0

    for step in range(start_step, tcfg["max_steps"]):
        train_sampler.set_epoch(step)

        lr = get_lr(step, tcfg["warmup_steps"], tcfg["max_steps"], tcfg["learning_rate"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for micro_step in range(grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            x, y = x.to(device), y.to(device)

            # sync only on last micro step
            is_last = micro_step == grad_accum - 1
            context = model.no_sync() if not is_last else torch.nullcontext()

            with context:
                with autocast(device_type="cuda"):
                    out = model(x, labels=y)
                    loss = out.loss / grad_accum
                scaler.scale(loss).backward()

            step_loss += loss.item()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
        scaler.step(optimizer)
        scaler.update()

        tokens_seen += batch_size * grad_accum * seq_len * world_size
        elapsed = time.time() - t0
        tok_per_sec = tokens_seen / elapsed
        mfu = estimate_mfu(model.module, tok_per_sec, device)

        if is_master and step % log_every == 0:
            vram = torch.cuda.memory_allocated() / 1e9
            print(
                f"step {step:5d} | loss {step_loss:.4f} | "
                f"lr {lr:.2e} | norm {grad_norm:.3f} | "
                f"tok/s {tok_per_sec:,.0f} | "
                f"MFU {mfu*100:.1f}% | "
                f"VRAM {vram:.2f}GB"
            )

        if is_master and step > 0 and step % save_every == 0:
            model.eval()
            total_loss, count = 0.0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    with autocast(device_type="cuda"):
                        out = model(x, labels=y)
                    total_loss += out.loss.item()
                    count += 1
                    if count >= 50:
                        break
            val_loss = total_loss / count
            print(f"  [val] loss {val_loss:.4f} | ppl {math.exp(val_loss):.2f}")
            model.train()

            ckpt_manager.save({
                "step": step,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": step_loss,
            }, step=step, async_save=True)

    if is_master:
        print("\nDone!")
        ckpt_manager.wait_for_async()

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gpt2_wikitext.yaml")
    parser.add_argument("--compression", action="store_true")
    args = parser.parse_args()
    train(config_path=args.config, use_compression=args.compression)