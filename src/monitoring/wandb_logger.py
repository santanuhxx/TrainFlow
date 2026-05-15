import torch
import wandb
from typing import Optional


class WandBLogger:
    def __init__(self, config: dict, enabled: bool = True):
        self.enabled = enabled
        if not enabled:
            return

        wandb.init(
            project=config["logging"]["wandb_project"],
            config={
                "model": config["model"],
                "training": config["training"],
                "data": config["data"],
            },
            tags=["gpt2", "distributed", "wikitext103"],
        )
        print(f"W&B initialized: {wandb.run.url}")

    def log(self, metrics: dict, step: int):
        if not self.enabled:
            return
        wandb.log(metrics, step=step)

    def log_system(self, step: int):
        if not self.enabled:
            return
        metrics = {
            "system/vram_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "system/vram_reserved_gb": torch.cuda.memory_reserved() / 1e9,
        }
        wandb.log(metrics, step=step)

    def finish(self):
        if not self.enabled:
            return
        wandb.finish()