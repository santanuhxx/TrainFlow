import torch
import argparse
import time
import csv
from pathlib import Path
from torch.cuda.amp import autocast
from src.trainer.base_trainer import BaseTrainer, get_lr, estimate_mfu
from src.monitoring.profiler import TrainingProfiler, MetricsTracker


def save_benchmark(summary: dict, config_name: str):
    bench_file = Path("benchmarks/results.csv")
    bench_file.parent.mkdir(exist_ok=True)
    file_exists = bench_file.exists()
    with open(bench_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "config", "final_loss", "avg_tok_per_sec",
            "avg_mfu_pct", "peak_vram_gb", "avg_grad_norm"
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "config": config_name,
            "final_loss": f"{summary['final_loss']:.4f}",
            "avg_tok_per_sec": f"{summary['avg_tok_per_sec']:.0f}",
            "avg_mfu_pct": f"{summary['avg_mfu']*100:.1f}",
            "peak_vram_gb": f"{summary['peak_vram_gb']:.2f}",
            "avg_grad_norm": f"{summary['avg_grad_norm']:.3f}",
        })
    print(f"Benchmark saved to {bench_file}")


def run_profiled_training(config_path: str, profile_steps: int = 100):
    trainer = BaseTrainer(config_path=config_path)
    cfg = trainer.config["training"]
    grad_accum = cfg["gradient_accumulation_steps"]
    log_every = trainer.config["logging"]["log_every_n_steps"]

    metrics = MetricsTracker()
    profiler = TrainingProfiler(log_dir="./profiler_logs", active_steps=5)

    print(f"\nProfiled training | {profile_steps} steps")

    trainer.model.train()
    data_iter = iter(trainer.train_loader)
    t0 = time.time()
    tokens_seen = 0

    prof = profiler.setup()

    with prof:
        for step in range(profile_steps):
            lr = get_lr(step, cfg["warmup_steps"], cfg["max_steps"], cfg["learning_rate"])
            for pg in trainer.optimizer.param_groups:
                pg["lr"] = lr

            trainer.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0

            for _ in range(grad_accum):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(trainer.train_loader)
                    x, y = next(data_iter)

                x, y = x.to(trainer.device), y.to(trainer.device)
                with autocast():
                    out = trainer.model(x, labels=y)
                    loss = out.loss / grad_accum

                trainer.scaler.scale(loss).backward()
                step_loss += loss.item()

            trainer.scaler.unscale_(trainer.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainer.model.parameters(), cfg["max_grad_norm"]
            )
            trainer.scaler.step(trainer.optimizer)
            trainer.scaler.update()

            tokens_seen += cfg["batch_size"] * grad_accum * trainer.config["data"]["seq_length"]
            elapsed = time.time() - t0
            tok_per_sec = tokens_seen / elapsed
            mfu = estimate_mfu(trainer.model, tok_per_sec, trainer.device)
            vram = torch.cuda.memory_allocated() / 1e9

            metrics.update(step_loss, tok_per_sec, mfu, vram, grad_norm.item())

            if step % log_every == 0:
                print(
                    f"step {step:4d} | loss {step_loss:.4f} | "
                    f"tok/s {tok_per_sec:,.0f} | "
                    f"MFU {mfu*100:.1f}% | "
                    f"VRAM {vram:.2f}GB"
                )
            prof.step()
    profiler.print_summary()
    summary = metrics.print_summary("Single GPU Baseline")
    save_benchmark(summary, "single_gpu_baseline")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gpt2_wikitext.yaml")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    run_profiled_training(config_path=args.config, profile_steps=args.steps)