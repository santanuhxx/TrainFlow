# TrainFlow — Distributed ML Training Orchestrator

A fault-tolerant distributed training system for large language models,
implementing PyTorch DDP, gradient compression, async checkpointing,
and real-time monitoring.

## Architecture
```
Master Process (Orchestrator)
        |
   DDP Wrapper
   /    |    \
Rank0  Rank1  Rank2
  |      |      |
  ∇      ∇      ∇
    AllReduce
  (Gloo backend)
        |
  Aggregated Update
        |
  Async Checkpoint
  (local + S3-ready)
```

## Features

- **Fault-tolerant training** — automatic checkpoint on crash or interrupt, resumes from exact step
- **DDP with gradient bucketing** — `bucket_cap_mb=25` for communication overlap
 - **DDP with gradient bucketing** — `bucket_cap_mb=25` in `train_ddp.py` (simulated single-process DDP);
   the real multi-process trainer in `src/trainer/ddp_trainer.py` uses `bucket_cap_mb=10`.
- **PowerSGD gradient compression** — 40-60% bandwidth reduction (requires >6GB VRAM)
- **Async atomic checkpointing** — background thread save, zero GPU blocking
- **Mixed precision (fp16)** — automatic loss scaling with GradScaler
- **Cosine LR decay with warmup** — GPT-3 paper schedule
- **MFU tracking** — Model FLOP Utilization monitored in real-time
- **W&B integration** — loss, perplexity, MFU, VRAM live dashboard
- **PyTorch Profiler** — bottleneck analysis with TensorBoard export

## Model and Dataset

| Component | Details |
|-----------|---------|
| Model | GPT-2 (124M parameters) |
| Dataset | WikiText-103 (103M tokens) |
| Sequence length | 128 tokens |
| Effective batch size | 16 (batch=1, grad_accum=16) |
| Precision | fp16 mixed precision |

## Benchmark Results (RTX 2050, 4GB VRAM)

| Configuration | Throughput | MFU | VRAM | Notes |
|---|---|---|---|---|
| Single GPU baseline | ~1,300 tok/s | 4.4% | 2.04 GB | Full training loop |
| DDP simulated (2 proc) | ~810 tok/s | 2.7% | 2.59 GB | Gloo communication overhead |
| Fault-tolerant + async ckpt | ~1,300 tok/s | 4.4% | 2.04 GB | Zero training overhead |
| Profiled run | ~1,075 tok/s | 3.6% | 2.04 GB | Profiler overhead included |
| PowerSGD compression | OOM | — | >4GB | Requires >6GB VRAM |

**Key finding from profiler:** `aten::_to_copy` (29,460 calls) identified as
primary CPU-GPU transfer bottleneck. On Linux with `num_workers=4`,
this reduces significantly.

## Project Structure
```
TrainFlow/
├── src/
│   ├── __init__.py
│   ├── trainer/
│   │   ├── __init__.py
│   │   ├── base_trainer.py         # Single GPU training loop
│   │   └── ddp_trainer.py          # DDP distributed training
│   ├── compression/
│   │   ├── __init__.py
│   │   └── gradient_hooks.py       # PowerSGD compression docs
│   ├── checkpoint/
│   │   ├── __init__.py
│   │   └── checkpoint_manager.py   # Atomic + async checkpointing
│   └── monitoring/
│       ├── __init__.py
│       ├── profiler.py             # PyTorch Profiler wrapper
│       └── wandb_logger.py         # W&B integration
├── benchmarks/
│   ├── results.csv                 # Benchmark data
│   └── summary.py                  # Print benchmark table
├── checkpoints/                    # Auto-generated, gitignored
├── configs/
│   └── gpt2_wikitext.yaml          # Training configuration
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── train.py                        # Fault-tolerant single GPU
├── train_ddp.py                    # DDP simulation
├── train_final.py                  # Full pipeline with W&B
└── train_profiled.py               # Profiler + benchmark
```

## Setup
```bash
conda create -n dist-ml python=3.10 -y
conda activate dist-ml
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets wandb boto3 tensorboard tqdm rich pyyaml
```

## Usage
```bash
# Single GPU baseline
python train.py

# Resume from checkpoint (auto-detected)
python train.py

# DDP training (simulated single-process)
python train_ddp.py

# DDP with PowerSGD compression (simulated single-process)
python train_ddp.py --compression

# Multi-process DDP (requires Linux + NCCL or multi-GPU setup)
torchrun --nproc_per_node=2 train_ddp.py

# Profiled training with bottleneck analysis
python train_profiled.py --steps 100

# Full training with W&B monitoring
python train_final.py

# Without W&B
python train_final.py --no-wandb

# View benchmark results
python benchmarks/summary.py
```

## Checkpoint System

- **Atomic save** — writes to temp file then renames, prevents corruption
- **Async background thread** — zero training overhead during save
- **Auto-cleanup** — keeps last N checkpoints, deletes older ones
- **Auto-resume** — detects latest checkpoint on every startup
- **Checkpoint size** — ~1.4GB for GPT-2 124M
```
checkpoints/
├── checkpoint_step_500.pt
├── checkpoint_step_1000.pt
├── checkpoint_step_1500.pt
└── latest.txt
```

## W&B Dashboard

Live training metrics:
Live training metrics available at your W&B project dashboard.

| Metric | Description |
|--------|-------------|
| `train/loss` | Training loss per step |
| `train/learning_rate` | Cosine decay with warmup |
| `train/grad_norm` | Gradient norm stability |
| `perf/tok_per_sec` | Throughput |
| `perf/mfu_pct` | Model FLOP Utilization |
| `perf/vram_gb` | GPU memory usage |
| `val/loss` | Validation loss |
| `val/perplexity` | Validation perplexity |

## Key Technical Decisions

**Why Gloo over NCCL?**
NCCL requires Linux with proper GPU peer-to-peer support. Gloo works
cross-platform including Windows, making it suitable for development
on consumer hardware.

**Why atomic checkpointing?**
Writing directly to the final path risks corruption if the process dies
mid-write. Temp file + rename is an OS-level atomic operation on all
platforms, guaranteeing checkpoint integrity.

**Why async checkpointing?**
Synchronous save of GPT-2 (~1.4GB) blocks training for ~3 seconds.
A background thread keeps GPU utilization continuous during saves.

**Why gradient accumulation over larger batch?**
4GB VRAM limits batch size to 1. Gradient accumulation achieves
equivalent optimization dynamics to batch=16 without the memory cost.

**Why fp16 over bf16?**
RTX 2050 (Turing architecture) does not support bfloat16. fp16 with
GradScaler provides equivalent training stability on supported hardware.

## Profiler Analysis

Top CPU bottlenecks identified on RTX 2050:

| Operation | CPU ms | Calls |
|-----------|--------|-------|
| `aten::_to_copy` | 1156ms | 29,460 |
| `aten::_local_scalar_dense` | 1052ms | 1,570 |
| `aten::transpose` | 508ms | 27,280 |
| `aten::addmm` | 373ms | 7,680 |

Primary bottleneck is CPU-GPU data transfer. Mitigation on Linux:
set `num_workers=4` in DataLoader.

## Limitations and Future Work

- PowerSGD requires >6GB VRAM — architecture tested on 4GB hardware
- Windows WDDM mode reduces GPU utilization compared to Linux
- Pipeline parallelism planned for v2
- S3 async upload interface ready — requires AWS credentials
- Full multi-node training requires Linux with NCCL backend

## References

- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PowerSGD Paper — Vogels et al., 2019](https://arxiv.org/abs/1905.13727)
- [Deep Gradient Compression — Lin et al., 2018](https://arxiv.org/abs/1712.01887)
- [GPT-2 — Radford et al., 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [FSDP — Zhao et al., 2023](https://arxiv.org/abs/2304.11277)