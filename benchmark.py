import torch
import time
import sys
import argparse

sys.path.insert(0, ".")
from config import TrainConfig, ModelConfig, DataConfig, load_config
from data import PrepareData
import torch.utils.data


def benchmark(model_cfg=None, data_cfg=None, train_cfg=None):
    print("=== Training Speed Benchmark ===")

    if model_cfg is None:
        model_cfg = ModelConfig()
    if data_cfg is None:
        data_cfg = DataConfig()
    if train_cfg is None:
        train_cfg = TrainConfig()

    train_loader, _, src_sp, tgt_sp = PrepareData(model_cfg, data_cfg, train_cfg)

    if train_cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(train_cfg.device)

    from model import Seq2SeqTransformer

    model = Seq2SeqTransformer(model_cfg).to(device)
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)

    # ... benchmark logic ...
    # Warmup
    for _ in range(5):
        src, tgt = next(iter(train_loader))
        src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
        loss, _ = model(src, tgt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure
    steps = 50
    total_tokens = 0
    start = time.time()
    for _ in range(steps):
        src, tgt = next(iter(train_loader))
        src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
        loss, _ = model(src, tgt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_tokens += src.numel() + tgt.numel()

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"Steps: {steps}")
    print(f"Time elapsed: {elapsed:.2f}s")
    print(f"Time per step: {elapsed / steps:.4f}s")
    print(f"Throughput: {steps / elapsed:.2f} steps/sec")
    print(f"Tokens processed: {total_tokens}")
    print(f"Tokens per second: {total_tokens / elapsed:.0f}")
    if device.type == "cuda":
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print("=== End Benchmark ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()

    model_cfg = None
    data_cfg = None
    train_cfg = None
    if args.config:
        model_cfg, data_cfg, train_cfg, _ = load_config(args.config)

    benchmark(model_cfg, data_cfg, train_cfg)
