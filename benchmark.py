import torch
import time
import sys
sys.path.insert(0, '.')
from train import train
from config import TrainConfig, ModelConfig
from data import PrepareData
import torch.utils.data

def benchmark():
    print("=== Training Speed Benchmark ===")
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
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
    train_loader, _, src_sp, tgt_sp = PrepareData(data_cfg)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from model import Seq2SeqTransformer
    model = Seq2SeqTransformer(model_cfg).to(device)
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)
    
    # Warmup
    for _ in range(5):
        src, tgt = next(iter(train_loader))
        src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
        loss = model(src, tgt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    
    # Measure
    steps = 50
    total_tokens = 0
    start = time.time()
    for _ in range(steps):
        src, tgt = next(iter(train_loader))
        src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
        loss = model(src, tgt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_tokens += src.numel() + tgt.numel()
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"Steps: {steps}")
    print(f"Time elapsed: {elapsed:.2f}s")
    print(f"Time per step: {elapsed/steps:.4f}s")
    print(f"Throughput: {steps/elapsed:.2f} steps/sec")
    print(f"Tokens processed: {total_tokens}")
    print(f"Tokens per second: {total_tokens/elapsed:.0f}")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    print("=== End Benchmark ===")

if __name__ == "__main__":
    benchmark()