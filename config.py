from dataclasses import dataclass, fields
import os


@dataclass
class ModelConfig:
    """Configuration for the Transformer model architecture."""

    d_model: int = 768
    enc_layers: int = 12
    dec_layers: int = 2
    n_heads: int = 16
    ffn_dim: int = 4096
    max_len: int = 256
    dropout: float = 0.1
    vocab_size: int = 32000
    activation: str = "gelu"
    use_checkpoint: bool = False

    # Special Tokens
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3


@dataclass
class DataConfig:
    """Configuration for data loading, preprocessing, and tokenization."""

    # Experiment info (usually populated from TrainConfig)
    experiment_name: str = "default"

    # Languages
    src_lang: str = "fa"
    tgt_lang: str = "en"

    # Paths
    src_train_path: str = "data/train.fa"
    tgt_train_path: str = "data/train.en"
    src_dev_path: str = "data/dev.fa"
    tgt_dev_path: str = "data/dev.en"

    # Tokenizer
    char_coverage: float = 0.9999
    input_sentence_size: int = 5_000_000

    @property
    def tokenizer_prefix_src(self) -> str:
        return os.path.join(self.experiment_name, "tokenizer_src")

    @property
    def tokenizer_prefix_tgt(self) -> str:
        return os.path.join(self.experiment_name, "tokenizer_tgt")

    # Streaming & Batching
    max_tokens_per_batch: int = 4000
    buffer_size: int = 50000
    num_workers: int = 2
    prefetch_factor: int = 64
    pad_multiple: int = 16
    max_seq_len: int = 256  # Hard filter during data loading


@dataclass
class TrainConfig:
    """Configuration for the training loop and optimization."""

    experiment_name: str = "default"
    aim_repo: str = "~/.aim"

    # Optimizer
    lr: float = 8.5e-4
    weight_decay: float = 0.01
    adam_eps: float = 1e-6
    label_smoothing: float = 0.1

    # Scheduler
    scheduler_type: str = "inv_sqrt"  # "inv_sqrt" or "cosine"
    warmup_steps: int = 5000
    max_steps: int = 100000
    epochs: int = 20

    # Training Loop
    accum_steps: int = 30
    grad_clip: float = 1.0
    eval_steps: int = 2500
    max_checkpoints: int = 4

    # Hardware & Performance
    device: str = "cuda"  # "cuda", "cpu", or "auto"
    precision: str = "bf16"  # "bf16", "fp32"
    tf32: bool = True

    # Logging & Validation
    log_steps: int = 1000
    val_max_samples: int = 500
    quick_test_samples: int = 5

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(self.experiment_name, "checkpoints")


@dataclass
class ExportConfig:
    """Configuration for checkpoint averaging, quantization, and export."""

    # Averaging
    k: int = 1

    # Quantization
    export_int8: bool = False
    calib_batches: int = 200
    quantization: str = "int8"
    qconfig_backend: str = "fbgemm"  # "fbgemm" or "qnnpack"

    # Inference Defaults
    beam_size: int = 5
    max_len: int = 100
    batch_size: int = 32

    # CT2 specific
    add_source_bos: bool = True
    add_source_eos: bool = False

    # Experiment info (usually populated from TrainConfig)
    experiment_name: str = "default"

    @property
    def model_path(self) -> str:
        return os.path.join(self.experiment_name, "averaged_model.safetensors")

    @property
    def output_dir(self) -> str:
        return os.path.join(self.experiment_name, "exported_model")

    @property
    def src_vocab(self) -> str:
        return os.path.join(self.experiment_name, "tokenizer_src.vocab")

    @property
    def tgt_vocab(self) -> str:
        return os.path.join(self.experiment_name, "tokenizer_tgt.vocab")

    @property
    def output_prefix(self) -> str:
        return os.path.join(self.experiment_name, "averaged_model")


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
    data_config = _from_dict(DataConfig, cfg.get("data", {}))
    train_config = _from_dict(TrainConfig, cfg.get("train", {}))
    export_config = _from_dict(ExportConfig, cfg.get("export", {}))

    # Link experiment name across configs
    export_config.experiment_name = train_config.experiment_name

    # Link experiment name across configs
    data_config.experiment_name = train_config.experiment_name

    return model_config, data_config, train_config, export_config
