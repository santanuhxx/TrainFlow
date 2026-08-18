import os
import time
import math

import torch
import yaml
import torch.nn as nn

from pathlib import Path
from src.trainer.config_validator import validate_config

from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
    AutoTokenizer,
)

from datasets import load_dataset
from tqdm import tqdm


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
) -> float:
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    if step > max_steps:
        return max_lr * 0.1

    decay_ratio = (
        (step - warmup_steps)
        / (max_steps - warmup_steps)
    )

    coeff = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )

    return (
        max_lr * 0.1
        + coeff * (max_lr - max_lr * 0.1)
    )


def estimate_mfu(
    model: nn.Module,
    tokens_per_sec: float,
    device: torch.device,
) -> float:

    num_params = sum(
        p.numel()
        for p in model.parameters()
    )

    # Approximate training FLOPs/token.
    flops_per_token = 6 * num_params
    actual_flops = (
        flops_per_token * tokens_per_sec
    )

    gpu_flops = {
        "RTX 2050": 22e12,
        "RTX 3090": 71e12,
        "RTX 4090": 165e12,
        "A100": 312e12,
        "V100": 125e12,
        "T4": 65e12,
    }

    # CPU / unknown GPU fallback
    peak_flops = 22e12

    if device.type == "cuda":

        gpu_name = torch.cuda.get_device_name(device)

        for name, flops in gpu_flops.items():
            if name in gpu_name:
                peak_flops = flops
                break

    return actual_flops / peak_flops


class WikiTextDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        split: str,
        seq_length: int,
        tokenizer,
    ):
        self.seq_length = seq_length

        print(
            f"Loading WikiText-103 ({split})..."
        )

        # IMPORTANT:
        # Use the current Hugging Face repository.
        dataset = load_dataset(
            "Salesforce/wikitext",
            "wikitext-103-raw-v1",
            split=split,
        )

        all_tokens = []

        batch_size = 1000

        texts = [
            t
            for t in dataset["text"]
            if t.strip()
        ]

        print(
            f"  Non-empty documents: {len(texts):,}"
        )

        for i in tqdm(
            range(0, len(texts), batch_size),
            desc="Tokenizing",
        ):

            batch = texts[
                i : i + batch_size
            ]

            joined = "\n\n".join(batch)

            tokens = tokenizer.encode(
                joined,
                add_special_tokens=False,
            )

            all_tokens.extend(tokens)

            # Memory constraint for training.
            if (
                len(all_tokens) > 5_000_000
                and split == "train"
            ):
                print(
                    "  Using first 5M tokens "
                    "(memory constraint)"
                )
                break

        print(
            f"  Total tokens: "
            f"{len(all_tokens):,}"
        )

        self.chunks = []

        for i in range(
            0,
            len(all_tokens) - seq_length,
            seq_length,
        ):

            chunk = all_tokens[
                i : i + seq_length + 1
            ]

            if len(chunk) == seq_length + 1:

                self.chunks.append(
                    torch.tensor(
                        chunk,
                        dtype=torch.long,
                    )
                )

        print(
            f"  {len(self.chunks):,} chunks created"
        )

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):

        chunk = self.chunks[idx]

        return (
            chunk[:-1],
            chunk[1:],
        )


class BaseTrainer:

    def __init__(self, config_path: str):

        self.config = load_config(
            config_path
        )

        validate_config(self.config)

        # --------------------------------------------------
        # Device
        # --------------------------------------------------

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Device: {self.device}"
        )

        if self.device.type == "cuda":

            print(
                "GPU:",
                torch.cuda.get_device_name(
                    self.device
                ),
            )

            print(
                f"VRAM: "
                f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )

        # --------------------------------------------------
        # Setup
        # --------------------------------------------------

        self._setup_model()
        self._setup_data()
        self._setup_optimizer()

        self.scaler = GradScaler(device="cuda")

        self.step = 0

    # ======================================================
    # MODEL
    # ======================================================

    def _setup_model(self):

        cfg = self.config["model"]

        gpt_config = GPT2Config(
            vocab_size=cfg["vocab_size"],
            n_positions=cfg["n_positions"],
            n_embd=cfg["n_embd"],
            n_layer=cfg["n_layer"],
            n_head=cfg["n_head"],
        )

        self.model = GPT2LMHeadModel(
            gpt_config
        ).to(self.device)

        num_params = (
            sum(
                p.numel()
                for p in self.model.parameters()
            )
            / 1e6
        )

        print(
            f"Model: GPT-2 | "
            f"Params: {num_params:.1f}M"
        )

    # ======================================================
    # DATA
    # ======================================================

    def _setup_data(self):

        self.tokenizer = (
            AutoTokenizer.from_pretrained("gpt2")
        )

        self.tokenizer.pad_token = (
            self.tokenizer.eos_token
        )

        seq_len = (
            self.config["data"]["seq_length"]
        )

        batch_size = (
            self.config["training"]["batch_size"]
        )

        num_workers = (
            self.config["data"]["num_workers"]
        )

        # -------------------------------
        # Train
        # -------------------------------

        train_ds = WikiTextDataset(
            "train",
            seq_len,
            self.tokenizer,
        )

        # -------------------------------
        # Validation
        # -------------------------------

        val_ds = WikiTextDataset(
            "validation",
            seq_len,
            self.tokenizer,
        )

        self.train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(
                self.device.type == "cuda"
            ),
        )

        self.val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(
                self.device.type == "cuda"
            ),
        )

        print(
            f"Train batches: "
            f"{len(self.train_loader):,}"
        )

        print(
            f"Validation batches: "
            f"{len(self.val_loader):,}"
        )

    # ======================================================
    # OPTIMIZER
    # ======================================================

    def _setup_optimizer(self):

        cfg = self.config["training"]

        decay_params = [
            p
            for _, p in self.model.named_parameters()
            if p.requires_grad
            and p.dim() >= 2
        ]

        no_decay_params = [
            p
            for _, p in self.model.named_parameters()
            if p.requires_grad
            and p.dim() < 2
        ]

        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": decay_params,
                    "weight_decay": cfg[
                        "weight_decay"
                    ],
                },
                {
                    "params": no_decay_params,
                    "weight_decay": 0.0,
                },
            ],
            lr=cfg["learning_rate"],
            betas=(0.9, 0.95),
            eps=1e-8,
        )

    # ======================================================
    # EVALUATION
    # ======================================================

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float]:

        self.model.eval()

        total_loss = 0.0
        count = 0

        for x, y in self.val_loader:

            x = x.to(
                self.device,
                non_blocking=True,
            )

            y = y.to(
                self.device,
                non_blocking=True,
            )

            with autocast(
                device_type="cuda",
                enabled=self.device.type == "cuda",
            ):

                out = self.model(
                    x,
                    labels=y,
                )

            total_loss += out.loss.item()

            count += 1

            if count >= 50:
                break

        self.model.train()

        if count == 0:
            return float("inf"), float("inf")

        avg_loss = (
            total_loss / count
        )

        try:
            perplexity = math.exp(
                avg_loss
            )
        except OverflowError:
            perplexity = float("inf")

        return (
            avg_loss,
            perplexity,
        )

    # ======================================================
    # CHECKPOINT
    # ======================================================

    def save_checkpoint(
        self,
        step: int,
        loss: float,
    ):

        save_dir = Path(
            self.config["checkpoint"][
                "save_dir"
            ]
        )

        save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        path = (
            save_dir
            / f"checkpoint_step_{step}.pt"
        )

        torch.save(
            {
                "step": step,
                "model_state_dict":
                    self.model.state_dict(),
                "optimizer_state_dict":
                    self.optimizer.state_dict(),
                "loss": loss,
                "config": self.config,
            },
            path,
        )

        print(
            f"  Saved: {path}"
        )

    # ======================================================
    # TRAIN
    # ======================================================

    def train(self):

        cfg = self.config["training"]

        grad_accum = cfg[
            "gradient_accumulation_steps"
        ]

        log_every = (
            self.config["logging"][
                "log_every_n_steps"
            ]
        )

        save_every = (
            self.config["checkpoint"][
                "save_every_n_steps"
            ]
        )

        batch_size = cfg["batch_size"]

        effective_batch_size = (
            batch_size * grad_accum
        )

        print(
            f"\nTraining for "
            f"{cfg['max_steps']} steps | "
            f"micro batch = {batch_size} | "
            f"gradient accumulation = "
            f"{grad_accum} | "
            f"effective batch = "
            f"{effective_batch_size}"
        )

        self.model.train()

        data_iter = iter(
            self.train_loader
        )

        t0 = time.time()

        tokens_seen = 0

        last_loss = 0.0

        # ==================================================
        # TRAINING LOOP
        # ==================================================

        for step in range(
            cfg["max_steps"]
        ):

            self.step = step

            # ----------------------------------------------
            # Learning rate
            # ----------------------------------------------

            lr = get_lr(
                step,
                cfg["warmup_steps"],
                cfg["max_steps"],
                cfg["learning_rate"],
            )

            for pg in (
                self.optimizer.param_groups
            ):
                pg["lr"] = lr

            # ----------------------------------------------
            # Clear gradients
            # ----------------------------------------------

            self.optimizer.zero_grad(
                set_to_none=True
            )

            step_loss = 0.0

            # ==============================================
            # GRADIENT ACCUMULATION
            # ==============================================

            for _ in range(
                grad_accum
            ):

                try:

                    x, y = next(
                        data_iter
                    )

                except StopIteration:

                    data_iter = iter(
                        self.train_loader
                    )

                    x, y = next(
                        data_iter
                    )

                x = x.to(
                    self.device,
                    non_blocking=True,
                )

                y = y.to(
                    self.device,
                    non_blocking=True,
                )

                # ------------------------------------------
                # Forward
                # ------------------------------------------

                with autocast(
                    device_type="cuda",
                    enabled=self.device.type == "cuda",
                ):

                    out = self.model(
                        x,
                        labels=y,
                    )

                    loss = (
                        out.loss
                        / grad_accum
                    )

                # ------------------------------------------
                # Backward
                # ------------------------------------------

                self.scaler.scale(
                    loss
                ).backward()

                step_loss += loss.item()

            # ==============================================
            # GRADIENT CLIPPING
            # ==============================================

            self.scaler.unscale_(
                self.optimizer
            )

            grad_norm = (
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    cfg["max_grad_norm"],
                )
            )

            # ==============================================
            # OPTIMIZER UPDATE
            # ==============================================

            self.scaler.step(
                self.optimizer
            )

            self.scaler.update()

            # ==============================================
            # STATISTICS
            # ==============================================

            tokens_seen += (
                batch_size
                * grad_accum
                * self.config["data"][
                    "seq_length"
                ]
            )

            elapsed = (
                time.time() - t0
            )

            tok_per_sec = (
                tokens_seen / elapsed
                if elapsed > 0
                else 0.0
            )

            mfu = estimate_mfu(
                self.model,
                tok_per_sec,
                self.device,
            )

            last_loss = step_loss

            # ==============================================
            # LOGGING
            # ==============================================

            if (
                step % log_every == 0
            ):

                if self.device.type == "cuda":

                    vram_used = (
                        torch.cuda.memory_allocated()
                        / 1e9
                    )

                else:

                    vram_used = 0.0

                print(
                    f"step {step:5d} | "
                    f"loss {step_loss:.4f} | "
                    f"lr {lr:.2e} | "
                    f"norm {grad_norm:.3f} | "
                    f"tok/s {tok_per_sec:,.0f} | "
                    f"MFU {mfu * 100:.1f}% | "
                    f"VRAM {vram_used:.2f}GB"
                )

            # ==============================================
            # VALIDATION + CHECKPOINT
            # ==============================================

            if (
                step > 0
                and step % save_every == 0
            ):

                val_loss, ppl = (
                    self.evaluate()
                )

                print(
                    f"  [val] "
                    f"loss {val_loss:.4f} | "
                    f"ppl {ppl:.2f}"
                )

                self.save_checkpoint(
                    step,
                    step_loss,
                )

        # ==================================================
        # FINAL CHECKPOINT
        # ==================================================

        print("\nDone!")

        self.save_checkpoint(
            cfg["max_steps"],
            last_loss,
        )