import torch
import torch.profiler
from pathlib import Path


class TrainingProfiler:
    def __init__(self, log_dir: str = "./profiler_logs", active_steps: int = 5):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.active_steps = active_steps
        self.profiler = None

    def setup(self):
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1, warmup=1, active=self.active_steps, repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(self.log_dir)
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        return self.profiler

    def print_summary(self, top_k: int = 15):
        if not self.profiler:
            return

        print("\n" + "="*60)
        print("PROFILER SUMMARY — Top operations by CPU time")
        print("="*60)

        events = self.profiler.key_averages()
        sorted_events = sorted(
            events,
            key=lambda e: e.self_cpu_time_total,
            reverse=True
        )[:top_k]

        print(f"{'Operation':<40} {'CPU ms':>10} {'Calls':>8}")
        print("-"*60)
        for e in sorted_events:
            if e.self_cpu_time_total > 0:
                print(
                    f"{e.key[:40]:<40} "
                    f"{e.self_cpu_time_total/1000:>10.2f} "
                    f"{e.count:>8}"
                )
        print("="*60)

    def step(self):
        if self.profiler:
            self.profiler.step()


class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.losses = []
        self.tok_per_sec_list = []
        self.mfu_list = []
        self.vram_list = []
        self.grad_norms = []

    def update(self, loss, tok_per_sec, mfu, vram, grad_norm):
        self.losses.append(loss)
        self.tok_per_sec_list.append(tok_per_sec)
        self.mfu_list.append(mfu)
        self.vram_list.append(vram)
        self.grad_norms.append(grad_norm)

    def summary(self) -> dict:
        if not self.losses:
            return {}
        import statistics
        stable_start = len(self.tok_per_sec_list) // 4
        return {
            "final_loss": self.losses[-1],
            "avg_tok_per_sec": statistics.mean(self.tok_per_sec_list[stable_start:]),
            "avg_mfu": statistics.mean(self.mfu_list[stable_start:]),
            "peak_vram_gb": max(self.vram_list),
            "avg_grad_norm": statistics.mean(self.grad_norms),
        }

    def print_summary(self, config_name: str = "baseline"):
        s = self.summary()
        if not s:
            return
        print("\n" + "="*50)
        print(f"METRICS SUMMARY — {config_name}")
        print("="*50)
        print(f"Final loss:      {s['final_loss']:.4f}")
        print(f"Avg throughput:  {s['avg_tok_per_sec']:,.0f} tok/s")
        print(f"Avg MFU:         {s['avg_mfu']*100:.1f}%")
        print(f"Peak VRAM:       {s['peak_vram_gb']:.2f} GB")
        print(f"Avg grad norm:   {s['avg_grad_norm']:.3f}")
        print("="*50)
        return s