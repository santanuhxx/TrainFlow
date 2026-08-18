import argparse
import signal
import sys
from src.trainer.base_trainer import BaseTrainer
from src.checkpoint.checkpoint_manager import CheckpointManager


class FaultTolerantTrainer(BaseTrainer):
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.ckpt_manager = CheckpointManager(
            save_dir=self.config["checkpoint"]["save_dir"],
            keep_last_n=self.config["checkpoint"]["keep_last_n"],
        )
        self._interrupted = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        def handler(signum, frame):
            print("\n[!] Interrupt received. Saving checkpoint before exit...")
            self._interrupted = True
        signal.signal(signal.SIGINT, handler)

    def save_checkpoint(self, step: int, loss: float):
        state = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "loss": loss,
            "config": self.config,
        }
        self.ckpt_manager.save(state, step, async_save=True)

    def resume_from_checkpoint(self) -> int:
        state = self.ckpt_manager.load_latest(self.device)
        if state is None:
            return 0

        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if "scaler_state_dict" in state:
            self.scaler.load_state_dict(state["scaler_state_dict"])
        return state["step"] + 1

    def train(self):
        import time
        import math
        from src.trainer.base_trainer import get_lr, estimate_mfu

        cfg = self.config["training"]
        grad_accum = cfg["gradient_accumulation_steps"]
        log_every = self.config["logging"]["log_every_n_steps"]
        save_every = self.config["checkpoint"]["save_every_n_steps"]
        start_step = self.resume_from_checkpoint()
        if start_step > 0:
            print(f"Resuming from step {start_step}")

        print(f"\nFault-tolerant training | steps {start_step} → {cfg['max_steps']}")

        self.model.train()
        data_iter = iter(self.train_loader)
        t0 = time.time()
        tokens_seen = start_step * cfg["batch_size"] * grad_accum * self.config["data"]["seq_length"]

        from torch.amp import autocast
        import torch

        for step in range(start_step, cfg["max_steps"]):
            if self._interrupted:
                print(f"Saving emergency checkpoint at step {step}...")
                self.save_checkpoint(step, 0.0)
                self.ckpt_manager.wait_for_async()
                print("Checkpoint saved. Exiting.")
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
                with autocast(device_type="cuda", enabled=self.device.type == "cuda"):
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
            mfu = estimate_mfu(self.model, tok_per_sec, self.device)

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
                val_loss, ppl = self.evaluate()
                print(f"  [val] loss {val_loss:.4f} | ppl {ppl:.2f}")
                self.save_checkpoint(step, step_loss)

        self.ckpt_manager.wait_for_async()
        print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gpt2_wikitext.yaml")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    args = parser.parse_args()

    trainer = FaultTolerantTrainer(config_path=args.config)
    trainer.train()


if __name__ == "__main__":
    main()