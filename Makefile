.PHONY: train train-ddp train-final profile benchmark test clean help

help:
	@echo "TrainFlow - Distributed ML Training Orchestrator"
	@echo ""
	@echo "  make train              Single GPU training"
	@echo "  make train-ddp          Multi-process DDP (NCCL)"
	@echo "  make train-ddp-compress DDP with PowerSGD compression"
	@echo "  make train-final        Full training with W&B"
	@echo "  make train-no-wandb     Full training without W&B"
	@echo "  make profile            Profiled training"
	@echo "  make benchmark          Print benchmark table"
	@echo "  make plot               Generate benchmark plot"
	@echo "  make test               Run fault tolerance test"
	@echo "  make clean              Remove checkpoints and logs"

train:
	python train.py --config configs/gpt2_wikitext.yaml

train-ddp:
	torchrun --nproc_per_node=2 train_ddp.py --config configs/gpt2_wikitext.yaml

train-ddp-compress:
	torchrun --nproc_per_node=2 train_ddp.py --config configs/gpt2_wikitext.yaml --compression

train-final:
	python train_final.py --config configs/gpt2_wikitext.yaml

train-no-wandb:
	python train_final.py --config configs/gpt2_wikitext.yaml --no-wandb

profile:
	python train_profiled.py --config configs/gpt2_wikitext.yaml --steps 100

benchmark:
	python benchmarks/summary.py

plot:
	python benchmarks/plot.py

test:
	python tests/test_fault_tolerance.py

clean:
	rm -rf checkpoints/
	rm -rf profiler_logs/
	rm -rf wandb/
	rm -rf logs/
	find . -type d -name "__pycache__" -exec rm -rf {} +