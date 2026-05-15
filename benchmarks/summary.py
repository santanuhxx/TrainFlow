import csv
from pathlib import Path


def print_benchmark_table():
    results_file = Path("benchmarks/results.csv")

    if not results_file.exists():
        print("No benchmark results found.")
        return

    with open(results_file) as f:
        rows = list(csv.DictReader(f))

    print("\n" + "="*80)
    print("BENCHMARK RESULTS — Distributed ML Training Orchestrator")
    print("="*80)
    print(f"{'Config':<30} {'Loss':>8} {'tok/s':>10} {'MFU%':>8} {'VRAM GB':>10}")
    print("-"*80)

    for row in rows:
        print(
            f"{row['config']:<30} "
            f"{row['final_loss']:>8} "
            f"{row['avg_tok_per_sec']:>10} "
            f"{row['avg_mfu_pct']:>8} "
            f"{row['peak_vram_gb']:>10}"
        )
    print("="*80)


if __name__ == "__main__":
    print_benchmark_table()