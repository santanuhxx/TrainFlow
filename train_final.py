import torch
import argparse
import time
import sys
import signal
import math
from pathlib import Path
from torch.cuda.amp import autocast

from src.trainer.base_trainer import BaseTrainer, get_lr, estimate_mfu
from src.checkpoint.checkpoint_manager import CheckpointManager
from src.monitoring.wandb_logger import WandBLogger
from src.monitoring.spike_detector import GradientSpikeDetector


class FinalTrainer(BaseTrainer):
    def __init__(self, config_path: str, use_wandb: bool = True):
        super().__init__(config_path)
        self.ckpt_manager = CheckpointManager(
            save_dir=self.config["checkpoint"]["save_dir"],
            keep_last_n=self.config["checkpoint"]["keep_last_n"],
        )
        self.logger = WandBLogger(self.config, enabled=use_wandb)
        self.spike_detector = GradientSpikeDetector(
            spike_threshold=10.0,
            window_size=20,
            max_rollbacks=3,
        )
        self._interrupted = False
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        print("\n[!] Interrupt — saving checkpoint...")
        self._interrupted = True

    def _save(self, step: int, loss: float):
        state = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "loss": loss,
            "config": self.config,
        }
        self.ckpt_manager.save(state, step, async_save=True)

    def _resume(self) -> int:
        state = self.ckpt_manager.load_latest(self.device)
        if state is None:
            return 0
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if "scaler_state_dict" in state:
            self.scaler.load_state_dict(state["scaler_state_dict"])
        return state["step"] + 1

    def train(self):
        cfg = self.config["training"]
        grad_accum = cfg["gradient_accumulation_steps"]
        log_every = self.config["logging"]["log_every_n_steps"]
        save_every = self.config["checkpoint"]["save_every_n_steps"]

        start_step = self._resume()
        if start_step > 0:
            print(f"Resumed from step {start_step}")

        print(f"\nTraining | steps {start_step} → {cfg['max_steps']} | "
              f"effective batch = {cfg['batch_size'] * grad_accum}")

        self.model.train()
        data_iter = iter(self.train_loader)
        t0 = time.time()
        tokens_seen = 0
        best_val_loss = float("inf")

        for step in range(start_step, cfg["max_steps"]):

            if self._interrupted:
                self._save(step, 0.0)
                self.ckpt_manager.wait_for_async()
                self.logger.finish()
                sys.exit(0)

            lr = get_lr(step, cfg["warmup_steps"], cfg["max_steps"], cfg["learning_rate"])
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0

            for _ in range(grad_accum):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    x, y = next(data_iter)

                x, y = x.to(self.device), y.to(self.device)
                with autocast():
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

            is_spike = self.spike_detector.update(grad_norm.item(), step)
            if is_spike:
                self.logger.log({"train/grad_spike": 1}, step=step)
                resume_step = self.spike_detector.rollback(
                    self.model,
                    self.optimizer,
                    self.scaler,
                    self.ckpt_manager,
                    self.device,
                )
                if resume_step is not None:
                    data_iter = iter(self.train_loader)
                    continue

            tokens_seen += cfg["batch_size"] * grad_accum * self.config["data"]["seq_length"]
            elapsed = time.time() - t0
            tok_per_sec = tokens_seen / elapsed
            mfu = estimate_mfu(self.model, tok_per_sec, self.device)
            vram = torch.cuda.memory_allocated() / 1e9

            if step % log_every == 0:
                print(
                    f"step {step:5d} | loss {step_loss:.4f} | "
                    f"lr {lr:.2e} | norm {grad_norm:.3f} | "
                    f"tok/s {tok_per_sec:,.0f} | "
                    f"MFU {mfu*100:.1f}% | "
                    f"VRAM {vram:.2f}GB"
                )
                self.logger.log({
                    "train/loss": step_loss,
                    "train/learning_rate": lr,
                    "train/grad_norm": grad_norm.item(),
                    "perf/tok_per_sec": tok_per_sec,
                    "perf/mfu_pct": mfu * 100,
                    "perf/vram_gb": vram,
                }, step=step)
                self.logger.log_system(step=step)

            if step > 0 and step % save_every == 0:
                val_loss, ppl = self.evaluate()
                print(f"  [val] loss {val_loss:.4f} | ppl {ppl:.2f}")
                self.logger.log({
                    "val/loss": val_loss,
                    "val/perplexity": ppl,
                }, step=step)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"  New best val loss: {best_val_loss:.4f}")
                self._save(step, step_loss)

        self.spike_detector.summary()
        self.ckpt_manager.wait_for_async()
        self.logger.finish()
        print(f"\nDone! Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gpt2_wikitext.yaml")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    trainer = FinalTrainer(
        config_path=args.config,
        use_wandb=not args.no_wandb
    )
    trainer.train()