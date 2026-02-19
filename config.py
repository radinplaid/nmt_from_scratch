from dataclasses import dataclass, fields


@dataclass
class ModelConfig:
    d_model: int = 768
    enc_layers: int = 12
    dec_layers: int = 2
    n_heads: int = 16
    ffn_dim: int = 4096
    max_len: int = 256
    dropout: float = 0.1
    vocab_size: int = 32000
    use_checkpoint: bool = False


@dataclass
class TrainConfig:
    experiment_name: str = "default"
    aim_repo: str = "~/.aim"
    max_tokens_per_batch: int = 4000
    buffer_size: int = 50000
    num_workers: int = 2
    lr: float = 8.5e-4
    weight_decay: float = 0.01
    adam_eps: float = 1e-6
    label_smoothing: float = 0.1
    scheduler_type: str = "inv_sqrt"  # "inv_sqrt" or "cosine"
    epochs: int = 20
    grad_clip: float = 1.0
    accum_steps: int = 30
    warmup_steps: int = 5000
    max_steps: int = 100000
    eval_steps: int = 2500
    max_checkpoints: int = 4
    checkpoint_dir: str = None  # type: ignore[assignment]

    # Data params
    src_lang: str = "fa"
    tgt_lang: str = "en"
    src_train_path: str = "data/train.fa"
    tgt_train_path: str = "data/train.en"
    src_dev_path: str = "data/dev.fa"
    tgt_dev_path: str = "data/dev.en"

    def __post_init__(self):
        # Dynamically generate checkpoint_dir from experiment_name if not provided
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.experiment_name


@dataclass
class ExportConfig:
    k: int = 1
    export_int8: bool = False
    calib_batches: int = 200
    model_path: str = "averaged_model.safetensors"
    output_dir: str = "ct2_model"
    src_vocab: str = "tokenizer_src.vocab"
    tgt_vocab: str = "tokenizer_tgt.vocab"
    quantization: str = "int8"

    def __post_init__(self):
        # Dynamically generate checkpoint_dir from experiment_name if not provided
        if self.output_prefix is None:
            self.output_prefix = "averaged_model"


def _from_dict(cls, d):
    valid_fields = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in valid_fields})


def load_config(path: str):
    import yaml

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}

    model_config = _from_dict(ModelConfig, cfg.get("model", {}))
    train_config = _from_dict(TrainConfig, cfg.get("train", {}))
    export_config = _from_dict(ExportConfig, cfg.get("export", {}))

    return model_config, train_config, export_config
