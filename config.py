from dataclasses import dataclass


@dataclass
class ModelConfig:
    d_model: int = 384
    enc_layers: int = 10
    dec_layers: int = 2
    n_heads: int = 8
    ffn_dim: int = 2048
    max_len: int = 256
    dropout: float = 0.1
    vocab_size: int = 20000
    use_checkpoint: bool = False


@dataclass
class TrainConfig:
    experiment_name: str = "v28"
    aim_repo: str = "/home/mark/mt/.aim"
    batch_size: int = 32
    max_tokens_per_batch: int = 10000
    buffer_size: int = 150000
    num_workers: int = 6
    lr: float = 4e-4
    weight_decay: float = 0.01
    adam_eps: float = 1e-6
    label_smoothing: float = 0.1
    scheduler_type: str = "inv_sqrt"  # "inv_sqrt" or "cosine"
    epochs: int = 20
    grad_clip: float = 1.0
    accum_steps: int = 3
    warmup_steps: int = 10000
    max_steps: int = 40000
    eval_steps: int = 1000
    max_checkpoints: int = 4
    checkpoint_dir: str = "checkpoints"
    use_qat: bool = True
    qat_start_step: int = 20000

    # Data params
    src_lang: str = "fa"
    tgt_lang: str = "en"
    src_train_path: str = "data/train.fa"
    tgt_train_path: str = "data/train.en"
    src_dev_path: str = "data/dev.fa"
    tgt_dev_path: str = "data/dev.en"


@dataclass
class DataConfig:
    vocab_size: int
    max_len: int
    batch_size: int
    max_tokens_per_batch: int
    src_train_path: str
    tgt_train_path: str
    src_dev_path: str
    tgt_dev_path: str
    buffer_size: int
    num_workers: int

    @classmethod
    def from_configs(cls, model_cfg: ModelConfig, train_cfg: TrainConfig):
        return cls(
            vocab_size=model_cfg.vocab_size,
            max_len=model_cfg.max_len,
            batch_size=train_cfg.batch_size,
            max_tokens_per_batch=train_cfg.max_tokens_per_batch,
            src_train_path=train_cfg.src_train_path,
            tgt_train_path=train_cfg.tgt_train_path,
            src_dev_path=train_cfg.src_dev_path,
            tgt_dev_path=train_cfg.tgt_dev_path,
            buffer_size=train_cfg.buffer_size,
            num_workers=train_cfg.num_workers,
        )
