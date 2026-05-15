import torch
from pathlib import Path
from typing import Optional


class GradientSpikeDetector:
  
    def __init__(
        self,
        spike_threshold: float = 10.0,
        window_size: int = 20,
        max_rollbacks: int = 3,
    ):
        self.spike_threshold = spike_threshold
        self.window_size = window_size
        self.max_rollbacks = max_rollbacks

        self.grad_norm_history = []
        self.rollback_count = 0
        self.spike_steps = []

    def update(self, grad_norm: float, step: int) -> bool:
        self.grad_norm_history.append(grad_norm)

        if len(self.grad_norm_history) < self.window_size:
            return False

        window = self.grad_norm_history[-self.window_size:]
        avg_norm = sum(window) / len(window)

        is_spike = grad_norm > (avg_norm * self.spike_threshold)

        if is_spike:
            self.spike_steps.append(step)
            print(
                f"\n[SpikeDetector] SPIKE at step {step} | "
                f"grad_norm={grad_norm:.3f} | "
                f"avg={avg_norm:.3f} | "
                f"ratio={grad_norm/avg_norm:.1f}x"
            )

        return is_spike

    def should_rollback(self) -> bool:
        if self.rollback_count >= self.max_rollbacks:
            print(f"[SpikeDetector] Max rollbacks ({self.max_rollbacks}) reached. Continuing.")
            return False
        return True

    def rollback(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler,
        checkpoint_manager,
        device: torch.device,
    ) -> Optional[int]:
     
        if not self.should_rollback():
            return None

        print("[SpikeDetector] Rolling back to last checkpoint...")
        state = checkpoint_manager.load_latest(device)

        if state is None:
            print("[SpikeDetector] No checkpoint found. Cannot rollback.")
            return None

        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if "scaler_state_dict" in state:
            scaler.load_state_dict(state["scaler_state_dict"])

        self.rollback_count += 1
        resume_step = state["step"] + 1
        print(f"[SpikeDetector] Rolled back to step {state['step']} "
              f"(rollback #{self.rollback_count})")

        return resume_step

    def summary(self):
        print(f"\n[SpikeDetector] Summary:")
        print(f"  Total spikes detected: {len(self.spike_steps)}")
        print(f"  Spike at steps: {self.spike_steps}")
        print(f"  Total rollbacks: {self.rollback_count}")