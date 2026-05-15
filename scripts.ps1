param([string]$command = "help")

switch ($command) {
    "help" {
        Write-Host "TrainFlow - Distributed ML Training Orchestrator"
        Write-Host ""
        Write-Host "Usage: .\scripts.ps1 <command>"
        Write-Host ""
        Write-Host "  train              Single GPU training"
        Write-Host "  train-ddp          DDP simulated training"
        Write-Host "  train-final        Full training with W&B"
        Write-Host "  train-no-wandb     Full training without W&B"
        Write-Host "  profile            Profiled training"
        Write-Host "  benchmark          Print benchmark table"
        Write-Host "  plot               Generate benchmark plot"
        Write-Host "  clean              Remove checkpoints and logs"
    }
    "train" { python train.py --config configs/gpt2_wikitext.yaml }
    "train-ddp" { python train_ddp.py --config configs/gpt2_wikitext.yaml }
    "train-ddp-compression" { python train_ddp.py --config configs/gpt2_wikitext.yaml --compression }
    "train-final" { python train_final.py --config configs/gpt2_wikitext.yaml }
    "train-no-wandb" { python train_final.py --config configs/gpt2_wikitext.yaml --no-wandb }
    "profile" { python train_profiled.py --config configs/gpt2_wikitext.yaml --steps 100 }
    "benchmark" { python benchmarks/summary.py }
    "plot" { python benchmarks/plot.py }
    "clean" {
        Remove-Item -Recurse -Force checkpoints/ -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force profiler_logs/ -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force wandb/ -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force logs/ -ErrorAction SilentlyContinue
        Write-Host "Cleaned."
    }
    default { Write-Host "Unknown command: $command. Run '.\scripts.ps1 help'" }
}