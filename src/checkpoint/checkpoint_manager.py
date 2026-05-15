import os
import time
import shutil
import threading
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Optional


class CheckpointManager:
    def __init__(self, save_dir: str, keep_last_n: int = 3):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self._save_thread: Optional[threading.Thread] = None

    def save(self, state: dict, step: int, async_save: bool = True):     
        if async_save:
            if self._save_thread and self._save_thread.is_alive():
                self._save_thread.join()

            self._save_thread = threading.Thread(
                target=self._save_atomic,
                args=(state, step),
                daemon=True,
            )
            self._save_thread.start()
            print(f"  [checkpoint] Async save started for step {step}")
        else:
            self._save_atomic(state, step)

    def _save_atomic(self, state: dict, step: int):
        t0 = time.time()

        final_path = self.save_dir / f"checkpoint_step_{step}.pt"
        temp_path = self.save_dir / f"checkpoint_step_{step}.tmp"
        torch.save(state, temp_path)
        temp_path.rename(final_path)
        latest_path = self.save_dir / "latest.txt"
        latest_path.write_text(str(final_path))

        elapsed = time.time() - t0
        size_mb = final_path.stat().st_size / 1e6
        print(f"  [checkpoint] Saved step {step} | "
              f"{size_mb:.1f}MB | {elapsed:.2f}s")
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        checkpoints = sorted(
            self.save_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1])
        )
        for old_ckpt in checkpoints[:-self.keep_last_n]:
            old_ckpt.unlink()
            print(f"  [checkpoint] Deleted old: {old_ckpt.name}")

    def load_latest(self, device: torch.device) -> Optional[dict]:
        latest_path = self.save_dir / "latest.txt"
        if not latest_path.exists():
            print("  [checkpoint] No checkpoint found, starting fresh.")
            return None

        ckpt_path = Path(latest_path.read_text().strip())
        if not ckpt_path.exists():
            print(f"  [checkpoint] Checkpoint file missing: {ckpt_path}")
            return None

        print(f"  [checkpoint] Loading: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        print(f"  [checkpoint] Resumed from step {state['step']}")
        return state

    def wait_for_async(self):
        if self._save_thread and self._save_thread.is_alive():
            print("  [checkpoint] Waiting for async save to complete...")
            self._save_thread.join()