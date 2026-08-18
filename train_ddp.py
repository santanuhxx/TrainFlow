import os
import argparse
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast

from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer
from torch.utils.data import DataLoader

from src.trainer.base_trainer import (
    load_config,
    get_lr,
    estimate_mfu,
    WikiTextDataset,
)


def setup(rank: int, world_size: int):
    """
    Initialize DDP using environment variables.

    For this simulated/single-process setup:
        rank = 0
        world_size = 1
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def train_ddp_simulated(
    config_path: str,
    use_compression: bool = False,
):
    # ---------------------------------------------------------
    # Single-process DDP simulation
    # ---------------------------------------------------------
    rank = 0
    world_size = 1

    setup(rank=rank, world_size=world_size)

    try:
        # -----------------------------------------------------
        # Config
        # -----------------------------------------------------
        config = load_config(config_path)

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for this training script, "
                "but torch.cuda.is_available() is False."
            )

        device = torch.device("cuda")

        print(
            f"Rank: {rank} | World size: {world_size} | "
            f"Device: {device} | "
            f"GPU: {torch.cuda.get_device_name(device)}"
        )

        cfg = config["model"]

        # -----------------------------------------------------
        # GPT-2 model
        # -----------------------------------------------------
        gpt_config = GPT2Config(
            vocab_size=cfg["vocab_size"],
            n_positions=cfg["n_positions"],
            n_embd=cfg["n_embd"],
            n_layer=cfg["n_layer"],
            n_head=cfg["n_head"],
        )

        model = GPT2LMHeadModel(gpt_config).to(device)

        # DDP wrapper
        #
        # device_ids=None is appropriate here because this is
        # a single-process simulation.
        model = DDP(
            model,
            device_ids=None,
            find_unused_parameters=False,
            bucket_cap_mb=25,
        )

        # -----------------------------------------------------
        # Optional PowerSGD
        # -----------------------------------------------------
        if use_compression:
            from torch.distributed.algorithms.ddp_comm_hooks import (
                powerSGD_hook as powerSGD,
            )

            state = powerSGD.PowerSGDState(
                process_group=dist.group.WORLD,
                matrix_approximation_rank=4,
                start_powerSGD_iter=50,
            )

            model.register_comm_hook(
                state,
                powerSGD.powerSGD_hook,
            )

            print(
                "PowerSGD compression enabled | "
                "matrix approximation rank=4"
            )

        # -----------------------------------------------------
        # Model information
        # -----------------------------------------------------
        num_params = (
            sum(p.numel() for p in model.module.parameters()) / 1e6
        )

        print(
            f"Model: GPT-2 | "
            f"Params: {num_params:.1f}M | "
            f"DDP wrapper: ON"
        )

        # -----------------------------------------------------
        # Tokenizer
        # -----------------------------------------------------
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        seq_len = config["data"]["seq_length"]
        batch_size = config["training"]["batch_size"]

        # -----------------------------------------------------
        # Dataset
        # -----------------------------------------------------
        train_ds = WikiTextDataset(
            "train",
            seq_len,
            tokenizer,
        )

        val_ds = WikiTextDataset(
            "validation",
            seq_len,
            tokenizer,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        # -----------------------------------------------------
        # Optimizer
        # -----------------------------------------------------
        tcfg = config["training"]

        decay_params = [
            p
            for _, p in model.module.named_parameters()
            if p.requires_grad and p.dim() >= 2
        ]

        no_decay_params = [
            p
            for _, p in model.module.named_parameters()
            if p.requires_grad and p.dim() < 2
        ]

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": decay_params,
                    "weight_decay": tcfg["weight_decay"],
                },
                {
                    "params": no_decay_params,
                    "weight_decay": 0.0,
                },
            ],
            lr=tcfg["learning_rate"],
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # -----------------------------------------------------
        # AMP
        # -----------------------------------------------------
        scaler = GradScaler(device="cuda")

        grad_accum = tcfg["gradient_accumulation_steps"]

        log_every = config["logging"]["log_every_n_steps"]
        save_every = config["checkpoint"]["save_every_n_steps"]

        mode = (
            "DDP + PowerSGD"
            if use_compression
            else "DDP"
        )

        print(
            f"\n{mode} | "
            f"{tcfg['max_steps']} steps | "
            f"micro batch = {batch_size} | "
            f"gradient accumulation = {grad_accum} | "
            f"effective batch = {batch_size * grad_accum}"
        )

        # -----------------------------------------------------
        # Training
        # -----------------------------------------------------
        model.train()

        data_iter = iter(train_loader)

        t0 = time.time()
        tokens_seen = 0

        for step in range(tcfg["max_steps"]):

            # -------------------------------------------------
            # Learning rate
            # -------------------------------------------------
            lr = get_lr(
                step,
                tcfg["warmup_steps"],
                tcfg["max_steps"],
                tcfg["learning_rate"],
            )

            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # -------------------------------------------------
            # Clear gradients ONCE per optimizer step
            # -------------------------------------------------
            optimizer.zero_grad(set_to_none=True)

            step_loss = 0.0

            # -------------------------------------------------
            # Gradient accumulation
            # -------------------------------------------------
            for _ in range(grad_accum):

                try:
                    x, y = next(data_iter)

                except StopIteration:
                    data_iter = iter(train_loader)
                    x, y = next(data_iter)

                x = x.to(
                    device,
                    non_blocking=True,
                )

                y = y.to(
                    device,
                    non_blocking=True,
                )

                # Divide loss before backward so that the
                # accumulated gradient is approximately the
                # average gradient over grad_accum batches.
                with autocast(device_type="cuda"):

                    out = model(
                        x,
                        labels=y,
                    )

                    loss = out.loss / grad_accum

                scaler.scale(loss).backward()

                step_loss += loss.item()

            # -------------------------------------------------
            # Unscale before gradient clipping
            # -------------------------------------------------
            scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                tcfg["max_grad_norm"],
            )

            # -------------------------------------------------
            # Parameter update
            # -------------------------------------------------
            scaler.step(optimizer)
            scaler.update()

            # -------------------------------------------------
            # Statistics
            # -------------------------------------------------
            tokens_seen += (
                batch_size
                * grad_accum
                * seq_len
            )

            elapsed = time.time() - t0

            tok_per_sec = (
                tokens_seen / elapsed
                if elapsed > 0
                else 0.0
            )

            mfu = estimate_mfu(
                model.module,
                tok_per_sec,
                device,
            )

            # -------------------------------------------------
            # Logging
            # -------------------------------------------------
            if step % log_every == 0:

                vram = (
                    torch.cuda.memory_allocated(device)
                    / 1e9
                )

                print(
                    f"step {step:5d} | "
                    f"loss {step_loss:.4f} | "
                    f"lr {lr:.2e} | "
                    f"norm {grad_norm:.3f} | "
                    f"tok/s {tok_per_sec:,.0f} | "
                    f"MFU {mfu * 100:.1f}% | "
                    f"VRAM {vram:.2f}GB"
                )

            # -------------------------------------------------
            # Validation + checkpoint
            # -------------------------------------------------
            if step > 0 and step % save_every == 0:

                model.eval()

                total_loss = 0.0
                count = 0

                with torch.no_grad():

                    for x, y in val_loader:

                        x = x.to(
                            device,
                            non_blocking=True,
                        )

                        y = y.to(
                            device,
                            non_blocking=True,
                        )

                        with autocast(device_type="cuda"):

                            out = model(
                                x,
                                labels=y,
                            )

                        total_loss += out.loss.item()
                        count += 1

                        if count >= 50:
                            break

                val_loss = (
                    total_loss / count
                    if count > 0
                    else float("inf")
                )

                ppl = (
                    math.exp(val_loss)
                    if val_loss < 20
                    else float("inf")
                )

                print(
                    f"  [val] loss {val_loss:.4f} | "
                    f"ppl {ppl:.2f}"
                )

                model.train()

                # -------------------------------------------------
                # Save checkpoint
                # -------------------------------------------------
                save_dir = Path(
                    config["checkpoint"]["save_dir"]
                )

                save_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                checkpoint_path = (
                    save_dir
                    / f"ddp_step_{step}.pt"
                )

                torch.save(
                    {
                        "step": step,
                        "model_state_dict":
                            model.module.state_dict(),
                        "optimizer_state_dict":
                            optimizer.state_dict(),
                        "loss": step_loss,
                    },
                    checkpoint_path,
                )

                print(
                    f"  Checkpoint saved: "
                    f"{checkpoint_path}"
                )

        print("\nDone!")

    finally:
        cleanup()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/gpt2_wikitext.yaml",
    )

    parser.add_argument(
        "--compression",
        action="store_true",
    )

    args = parser.parse_args()

    train_ddp_simulated(
        config_path=args.config,
        use_compression=args.compression,
    )