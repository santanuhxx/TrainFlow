REQUIRED_FIELDS = {
    "model": ["vocab_size", "n_positions", "n_embd", "n_layer", "n_head"],
    "training": ["batch_size", "gradient_accumulation_steps", "learning_rate",
                 "weight_decay", "warmup_steps", "max_steps", "max_grad_norm",
                 "mixed_precision"],
    "data": ["dataset", "seq_length", "num_workers"],
    "checkpoint": ["save_dir", "save_every_n_steps", "keep_last_n"],
    "logging": ["log_every_n_steps", "wandb_project"],
}

VALID_PRECISIONS = ["fp16", "bf16", "fp32"]


def validate_config(config: dict) -> None:
  
    errors = []

    for section, fields in REQUIRED_FIELDS.items():
        if section not in config:
            errors.append(f"Missing section: '{section}'")
            continue
        for field in fields:
            if field not in config[section]:
                errors.append(f"Missing field: '{section}.{field}'")

    if errors:
        raise ValueError("Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    t = config["training"]

    if t["batch_size"] < 1:
        errors.append("training.batch_size must be >= 1")

    if t["learning_rate"] <= 0:
        errors.append("training.learning_rate must be > 0")

    if t["max_steps"] < t["warmup_steps"]:
        errors.append("training.max_steps must be > warmup_steps")

    if t["mixed_precision"] not in VALID_PRECISIONS:
        errors.append(f"training.mixed_precision must be one of {VALID_PRECISIONS}")

    if t["max_grad_norm"] <= 0:
        errors.append("training.max_grad_norm must be > 0")

    d = config["data"]
    if d["seq_length"] > config["model"]["n_positions"]:
        errors.append(
            f"data.seq_length ({d['seq_length']}) cannot exceed "
            f"model.n_positions ({config['model']['n_positions']})"
        )

    if d["num_workers"] < 0:
        errors.append("data.num_workers must be >= 0")

    c = config["checkpoint"]
    if c["keep_last_n"] < 1:
        errors.append("checkpoint.keep_last_n must be >= 1")

    if c["save_every_n_steps"] < 1:
        errors.append("checkpoint.save_every_n_steps must be >= 1")

    if errors:
        raise ValueError("Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    print("Config validation passed.")