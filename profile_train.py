import torch
import time
import sys
import argparse

sys.path.insert(0, ".")
from train import train
from config import TrainConfig, ModelConfig, load_config
from data import PrepareData
import torch.utils.data


def profile_data_loading(model_cfg=None, train_cfg=None):
    print("Profiling data loading...")
    if model_cfg is None:
        model_cfg = ModelConfig()
    if train_cfg is None:
        train_cfg = TrainConfig()

    start = time.time()
    train_loader, dev_loader, src_sp, tgt_sp = PrepareData(model_cfg, train_cfg)
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
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")


def profile_training_step(model_cfg=None, train_cfg=None):
    print("\nProfiling training step...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_cfg is None:
        model_cfg = ModelConfig()
    if train_cfg is None:
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
        loss, _ = model(src, tgt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Measure
    steps = 20
    start = time.time()
    for _ in range(steps):
        loss, _ = model(src, tgt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.time() - start
    print(f"Time per step: {elapsed / steps:.4f}s")
    print(f"Throughput: {steps / elapsed:.2f} steps/sec")
    print(f"Loss: {loss.item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()

    model_cfg = None
    train_cfg = None
    if args.config:
        model_cfg, train_cfg = load_config(args.config)

    profile_data_loading(model_cfg, train_cfg)
    profile_training_step(model_cfg, train_cfg)
