.PHONY: train train-ddp train-final profile benchmark clean help

help:
	@echo "TrainFlow - Distributed ML Training Orchestrator"
	@echo ""
	@echo "Usage:"
	@echo "  make train          - Single GPU fault-tolerant training"
	@echo "  make train-ddp      - DDP simulated training"
	@echo "  make train-final    - Full training with W&B monitoring"
	@echo "  make train-no-wandb - Full training without W&B"
	@echo "  make profile        - Profiled training with bottleneck analysis"
	@echo "  make benchmark      - Print benchmark results table"
	@echo "  make plot           - Generate benchmark visualization"
	@echo "  make clean          - Remove checkpoints and logs"
	@echo "  make clean-all      - Remove all generated files"

train:
	python train.py --config configs/gpt2_wikitext.yaml

train-ddp:
	python train_ddp.py --config configs/gpt2_wikitext.yaml

train-ddp-compression:
	python train_ddp.py --config configs/gpt2_wikitext.yaml --compression

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

clean:
	rm -rf checkpoints/
	rm -rf profiler_logs/
	rm -rf wandb/
	rm -rf logs/

clean-all: clean
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete