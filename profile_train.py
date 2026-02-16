import torch
import time
import sys
sys.path.insert(0, '.')
from train import train
from config import TrainConfig, ModelConfig
from data import PrepareData
import torch.utils.data

def profile_data_loading():
    print("Profiling data loading...")
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    # Create DataConfig as in train.py
    from dataclasses import dataclass
    @dataclass
    class DataConfig:
        vocab_size: int = model_cfg.vocab_size
        max_len: int = model_cfg.max_len
        batch_size: int = train_cfg.batch_size
        max_tokens_per_batch: int = train_cfg.max_tokens_per_batch
        src_train_path: str = train_cfg.src_train_path
        tgt_train_path: str = train_cfg.tgt_train_path
        src_dev_path: str = train_cfg.src_dev_path
        tgt_dev_path: str = train_cfg.tgt_dev_path
        buffer_size: int = train_cfg.buffer_size
        num_workers: int = train_cfg.num_workers
    data_cfg = DataConfig()
    start = time.time()
    train_loader, dev_loader, src_sp, tgt_sp = PrepareData(data_cfg)
    print(f"Data preparation took {time.time() - start:.2f}s")
    # Iterate through a few batches
    batch_times = []
    data_start = time.time()
    for i, (src, tgt) in enumerate(train_loader):
        if i >= 10:
            break
        batch_times.append(time.time())
    if batch_times:
        avg = (batch_times[-1] - data_start) / len(batch_times)
        print(f"Average time per batch: {avg:.4f}s")
        print(f"Batch shapes: src {src.shape}, tgt {tgt.shape}")
    # GPU memory
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

def profile_training_step():
    print("\nProfiling training step...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    # Build model
    from model import Seq2SeqTransformer
    model = Seq2SeqTransformer(model_cfg).to(device)
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)
    # Prepare a dummy batch
    src = torch.randint(0, model_cfg.vocab_size, (32, 50), device=device)
    tgt = torch.randint(0, model_cfg.vocab_size, (32, 55), device=device)
    # Warmup
    for _ in range(5):
        loss = model(src, tgt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    # Measure
    steps = 20
    start = time.time()
    for _ in range(steps):
        loss = model(src, tgt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"Time per step: {elapsed/steps:.4f}s")
    print(f"Throughput: {steps/elapsed:.2f} steps/sec")
    print(f"Loss: {loss.item():.4f}")

if __name__ == "__main__":
    profile_data_loading()
    profile_training_step()