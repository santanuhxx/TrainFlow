import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import tempfile
import argparse
import math
import time
from pathlib import Path
from torch.cuda.amp import GradScaler, autocast
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer
from torch.utils.data import DataLoader

from src.trainer.base_trainer import load_config, get_lr, estimate_mfu, WikiTextDataset


def setup(rank, world_size):
    store_path = os.path.join(tempfile.gettempdir(), "ddp_store_sim")
    if os.path.exists(store_path):
        os.remove(store_path)
    store = dist.FileStore(store_path, world_size)
    dist.init_process_group(
        backend="gloo",
        store=store,
        rank=rank,
        world_size=world_size,
    )


def cleanup():
    dist.destroy_process_group()


def train_ddp_simulated(config_path: str, use_compression: bool = False):
    setup(rank=0, world_size=1)

    config = load_config(config_path)
    device = torch.device("cuda")
    print(f"Device: {device} | {torch.cuda.get_device_name(device)}")
    cfg = config["model"]
    gpt_config = GPT2Config(
        vocab_size=cfg["vocab_size"],
        n_positions=cfg["n_positions"],
        n_embd=cfg["n_embd"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
    )
    model = GPT2LMHeadModel(gpt_config).to(device)
    model = DDP(model, device_ids=None, find_unused_parameters=False, bucket_cap_mb=25)

    if use_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as powerSGD
        state = powerSGD.PowerSGDState(
            process_group=None,
            matrix_approximation_rank=4,
            start_powerSGD_iter=50,
        )
        model.register_comm_hook(state, powerSGD.powerSGD_hook)
        print("PowerSGD compression enabled | rank=4")

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: GPT-2 | Params: {num_params:.1f}M | DDP wrapper: ON")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    seq_len = config["data"]["seq_length"]
    batch_size = config["training"]["batch_size"]

    train_ds = WikiTextDataset("train", seq_len, tokenizer)
    val_ds = WikiTextDataset("validation", seq_len, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    tcfg = config["training"]
    decay_params = [p for n, p in model.module.named_parameters()
                    if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.module.named_parameters()
                       if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": tcfg["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=tcfg["learning_rate"], betas=(0.9, 0.95), eps=1e-8)

    scaler = GradScaler()
    grad_accum = tcfg["gradient_accumulation_steps"]
    log_every = config["logging"]["log_every_n_steps"]
    save_every = config["checkpoint"]["save_every_n_steps"]

    mode = "DDP + PowerSGD" if use_compression else "DDP"
    print(f"\n{mode} | {tcfg['max_steps']} steps | "
          f"effective batch = {batch_size * grad_accum}")

    model.train()
    data_iter = iter(train_loader)
    t0 = time.time()
    tokens_seen = 0

    for step in range(tcfg["max_steps"]):
        lr = get_lr(step, tcfg["warmup_steps"], tcfg["max_steps"], tcfg["learning_rate"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for _ in range(grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            x, y = x.to(device), y.to(device)
            with autocast():
                out = model(x, labels=y)
                loss = out.loss / grad_accum

            scaler.scale(loss).backward()
            step_loss += loss.item()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), tcfg["max_grad_norm"]
        )
        scaler.step(optimizer)
        scaler.update()

        tokens_seen += batch_size * grad_accum * seq_len
        elapsed = time.time() - t0
        tok_per_sec = tokens_seen / elapsed
        mfu = estimate_mfu(model.module, tok_per_sec, device)

        if step % log_every == 0:
            vram = torch.cuda.memory_allocated() / 1e9
            print(
                f"step {step:5d} | loss {step_loss:.4f} | "
                f"lr {lr:.2e} | norm {grad_norm:.3f} | "
                f"tok/s {tok_per_sec:,.0f} | "
                f"MFU {mfu*100:.1f}% | "
                f"VRAM {vram:.2f}GB"
            )

        if step > 0 and step % save_every == 0:
            model.eval()
            total_loss, count = 0.0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    with autocast():
                        out = model(x, labels=y)
                    total_loss += out.loss.item()
                    count += 1
                    if count >= 50:
                        break
            val_loss = total_loss / count
            ppl = math.exp(val_loss)
            print(f"  [val] loss {val_loss:.4f} | ppl {ppl:.2f}")
            model.train()

            # Save
            save_dir = Path(config["checkpoint"]["save_dir"])
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "step": step,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": step_loss,
            }, save_dir / f"ddp_step_{step}.pt")
            print(f"  Checkpoint saved.")

    print("\nDone!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gpt2_wikitext.yaml")
    parser.add_argument("--compression", action="store_true")
    args = parser.parse_args()

    train_ddp_simulated(config_path=args.config, use_compression=args.compression)