import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_benchmarks():
    results_file = Path("benchmarks/results.csv")
    if not results_file.exists():
        print("No benchmark results found.")
        return

    with open(results_file) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No data in results.csv")
        return

    _plot_with_matplotlib(rows)


def _plot_with_matplotlib(rows):
    configs = [r["config"] for r in rows]
    tok_per_sec = [float(r["avg_tok_per_sec"]) for r in rows]
    mfu = [float(r["avg_mfu_pct"]) for r in rows]
    vram = [float(r["peak_vram_gb"]) for r in rows]
    loss = [float(r["final_loss"]) for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "TrainFlow — Benchmark Results\nGPT-2 124M | WikiText-103 | RTX 2050 4GB",
        fontsize=13,
        fontweight="bold",
    )

    colors = ["#4C9BE8", "#E87B4C", "#4CE87B", "#E84C9B", "#9B4CE8"][:len(configs)]

    # Throughput
    axes[0, 0].bar(configs, tok_per_sec, color=colors)
    axes[0, 0].set_title("Throughput (tok/s)", fontweight="bold")
    axes[0, 0].set_ylabel("tokens/sec")
    axes[0, 0].tick_params(axis="x", rotation=15)
    for i, v in enumerate(tok_per_sec):
        axes[0, 0].text(i, v + 10, f"{v:,.0f}", ha="center", fontsize=9)

    # MFU
    axes[0, 1].bar(configs, mfu, color=colors)
    axes[0, 1].set_title("Model FLOP Utilization (%)", fontweight="bold")
    axes[0, 1].set_ylabel("MFU %")
    axes[0, 1].tick_params(axis="x", rotation=15)
    for i, v in enumerate(mfu):
        axes[0, 1].text(i, v + 0.05, f"{v:.1f}%", ha="center", fontsize=9)

    # VRAM
    axes[1, 0].bar(configs, vram, color=colors)
    axes[1, 0].set_title("Peak VRAM Usage (GB)", fontweight="bold")
    axes[1, 0].set_ylabel("GB")
    axes[1, 0].tick_params(axis="x", rotation=15)
    axes[1, 0].axhline(y=4.0, color="red", linestyle="--", alpha=0.7, label="4GB limit")
    axes[1, 0].legend(fontsize=8)
    for i, v in enumerate(vram):
        axes[1, 0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    # Loss
    axes[1, 1].bar(configs, loss, color=colors)
    axes[1, 1].set_title("Final Training Loss", fontweight="bold")
    axes[1, 1].set_ylabel("loss")
    axes[1, 1].tick_params(axis="x", rotation=15)
    for i, v in enumerate(loss):
        axes[1, 1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    output_path = Path("benchmarks/benchmark_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


def _plot_ascii(rows):
    print("\nBENCHMARK RESULTS")
    print("="*60)
    for row in rows:
        bar_len = int(float(row["avg_tok_per_sec"]) / 50)
        bar = "█" * bar_len
        print(f"{row['config']:<25} {bar} {float(row['avg_tok_per_sec']):,.0f} tok/s")
    print("="*60)


if __name__ == "__main__":
    plot_benchmarks()