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
    experiment_name: str = "v30"
    aim_repo: str = "/home/mark/mt/.aim"
    batch_size: int = 32
    max_tokens_per_batch: int = 10000
    buffer_size: int = 150000
    num_workers: int = 4
    lr: float = 8.5e-4
    weight_decay: float = 0.01
    adam_eps: float = 1e-6
    label_smoothing: float = 0.1
    scheduler_type: str = "inv_sqrt"  # "inv_sqrt" or "cosine"
    epochs: int = 20
    grad_clip: float = 1.0
    accum_steps: int = 12
    warmup_steps: int = 8000
    max_steps: int = 20000
    eval_steps: int = 1000
    max_checkpoints: int = 4
    checkpoint_dir: str = "checkpoints"

    # Data params
    src_lang: str = "fa"
    tgt_lang: str = "en"
    src_train_path: str = "data/train.fa"
    tgt_train_path: str = "data/train.en"
    src_dev_path: str = "data/dev.fa"
    tgt_dev_path: str = "data/dev.en"


