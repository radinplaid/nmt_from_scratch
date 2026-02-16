import torch
import torch.optim as optim
from dataclasses import dataclass
from config import ModelConfig, TrainConfig

from model import Seq2SeqTransformer
from data import PrepareData
import time
import os
import sacrebleu
import math
from aim import Run


def train():
    # Configs
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Performance optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on matmul (Ampere+)
        torch.backends.cudnn.allow_tf32 = True  # Allow TF32 on cudnn
        # Do NOT call empty_cache at start - it's unnecessary and may slow down
        # Do NOT set memory fraction unless you have multiple processes

    run = Run(repo=train_cfg.aim_repo, experiment=train_cfg.experiment_name)
    run["hparams"] = {
        **{k: v for k, v in model_cfg.__dict__.items()},
        **{k: v for k, v in train_cfg.__dict__.items()},
    }

    # Data
    print("Preparing data...")

    # PrepareData needs vocab_size/max_len from model_cfg AND batch_size/tokens from train_cfg
    # Let's pass a merged object or just both
    # Updating PrepareData signature to accept 'config' which has what we need.
    # The simplest fix without changing PrepareData signature (which takes 'config')
    # is to create a merged config object or MonkeyPatch.
    # Let's make a merged config for data prep.
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
    train_loader, dev_loader, src_sp, tgt_sp = PrepareData(data_cfg)

    # Model
    print("Initializing model...")
    # Enable TF32 for speed on Ampere+
    torch.set_float32_matmul_precision("high")

    model = Seq2SeqTransformer(model_cfg).to(device)
    model = torch.compile(model)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        eps=train_cfg.adam_eps,
    )

    # Scheduler
    def lr_lambda(current_step):
        # current_step is the number of scheduler.step() calls made so far (0-indexed)
        # We want to treat the first step as 1
        step = current_step + 1
        if train_cfg.scheduler_type == "cosine":
            if step < train_cfg.warmup_steps:
                return float(step) / float(max(1, train_cfg.warmup_steps))
            progress = float(step - train_cfg.warmup_steps) / float(
                max(1, train_cfg.max_steps - train_cfg.warmup_steps)
            )
            return 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)).item())
        else:
            # Inverse Square Root scheduler
            if step < train_cfg.warmup_steps:
                return float(step) / float(max(1, train_cfg.warmup_steps))
            else:
                # Scale so that at warmup_steps, factor is 1.0
                return (train_cfg.warmup_steps**0.5) * (step**-0.5)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def save_checkpoint(step, model, optimizer, scheduler, config):
        if not os.path.exists(config.checkpoint_dir):
            os.makedirs(config.checkpoint_dir)

        path = os.path.join(config.checkpoint_dir, f"checkpoint_{step}.pt")
        torch.save(
            {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            path,
        )
        print(f"Checkpoint saved: {path}")

        # Rotation
        checkpoints = sorted(
            [
                f
                for f in os.listdir(config.checkpoint_dir)
                if f.startswith("checkpoint_")
            ],
            key=lambda x: int(x.split("_")[1].split(".")[0]),
        )
        if len(checkpoints) > config.max_checkpoints:
            to_remove = os.path.join(config.checkpoint_dir, checkpoints[0])
            os.remove(to_remove)
            print(f"Removed old checkpoint: {to_remove}")

    def validate(model, loader, src_sp, tgt_sp, device, use_autoregressive=False):
        """
        Validate the model.

        Args:
            model: The seq2seq model
            loader: Validation data loader
            src_sp: Source sentencepiece tokenizer
            tgt_sp: Target sentencepiece tokenizer
            device: Device to run on
            use_autoregressive: If True, use true autoregressive generation with pre-computed encoder.
                               If False (default), use teacher-forced predictions (faster, less memory).
        """
        model.eval()
        total_loss = 0
        total_tokens = 0
        correct_tokens = 0

        # Limit samples for BLEU calculation to reduce memory
        max_samples = 500
        hypotheses = []
        references = []
        sample_count = 0

        val_steps = 0
        # Use inference_mode instead of no_grad for better performance
        with torch.inference_mode():
            for batch_idx, (src, tgt) in enumerate(loader):
                src, tgt = (
                    src.to(device, non_blocking=True),
                    tgt.to(device, non_blocking=True),
                )

                # Forward pass for loss and logits (calculates loss internally)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss, (logits, _) = model(src, tgt, return_outputs=True)

                # Accumulate loss
                total_loss += loss.item()
                val_steps += 1

                # Accuracy calculation
                tgt_labels = tgt[:, 1:]
                # logits matches tgt_labels shape (batch, seq_len-1, vocab)
                preds = logits.argmax(dim=-1)
                mask = tgt_labels != 0
                correct_tokens += ((preds == tgt_labels) & mask).sum().item()
                total_tokens += mask.sum().item()

                # Generation for BLEU/ChrF - only process if we still need samples
                if sample_count < max_samples:
                    if use_autoregressive:
                        # True autoregressive generation including encoding
                        # We can manually encode to match previous behavior of reusing encoder output
                        enc = model.encode(src)
                        generated_ids = model.generate(src, max_len=256, enc_output=enc)
                    else:
                        # Teacher-forced predictions (fastest, uses existing logits)
                        generated_ids = preds

                    for i in range(src.size(0)):
                        if sample_count >= max_samples:
                            break
                        hyp = tgt_sp.decode(generated_ids[i].tolist())
                        ref = tgt_sp.decode(tgt[i].tolist())
                        hypotheses.append(hyp)
                        references.append(ref)
                        sample_count += 1

                # No need to delete tensors; they will be freed automatically
                # Avoid calling empty_cache frequently (slows down)

        avg_loss = total_loss / max(1, val_steps)
        ppl = math.exp(min(avg_loss, 100))
        acc = correct_tokens / max(1, total_tokens)

        bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
        chrf = sacrebleu.corpus_chrf(hypotheses, [references]).score

        metrics = {"loss": avg_loss, "ppl": ppl, "acc": acc, "bleu": bleu, "chrf": chrf}

        print(
            f"\n[Validation] Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | Acc: {acc:.4f} | BLEU: {bleu:.2f} | ChrF: {chrf:.2f}"
        )
        for i in range(min(10, len(hypotheses))):
            print(f"Sample {i}:")
            print(f"  Ref: {references[i]}")
            print(f"  Hyp: {hypotheses[i]}")
        print("-" * 30)

        model.train()
        return metrics

    # Loop
    model.train()
    global_step = 0
    optimizer.zero_grad()
    for epoch in range(train_cfg.epochs):
        start_time = time.time()
        total_loss = 0
        batch_src_tokens = 0
        batch_tgt_tokens = 0
        last_log_time = time.time()

        for batch_idx, (src, tgt) in enumerate(train_loader):
            # Use non_blocking for async data transfer
            src, tgt = (
                src.to(device, non_blocking=True),
                tgt.to(device, non_blocking=True),
            )

            # Allow any efficient backend (Flash or MemEfficient). Math is fallback.
            # We don't strictly enforce Flash because it might fail with certain masks/dtypes.
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(src, tgt, label_smoothing=train_cfg.label_smoothing)

            loss = loss / train_cfg.accum_steps
            loss.backward()
            total_loss += loss.item() * train_cfg.accum_steps

            # Throughput tracking
            batch_src_tokens += (src != 0).sum().item()
            batch_tgt_tokens += (tgt != 0).sum().item()

            if (batch_idx + 1) % train_cfg.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Validation and Checkpointing
                if global_step % train_cfg.eval_steps == 0:
                    val_metrics = validate(model, dev_loader, src_sp, tgt_sp, device)
                    for k, v in val_metrics.items():
                        run.track(
                            v,
                            name=f"val_{k}",
                            step=global_step,
                            context={"subset": "dev"},
                        )
                    save_checkpoint(global_step, model, optimizer, scheduler, train_cfg)

            # Progress Print
            if batch_idx % 1000 == 0:
                curr_lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - last_log_time
                in_tok_s = batch_src_tokens / max(1e-6, elapsed)
                out_tok_s = batch_tgt_tokens / max(1e-6, elapsed)

                print(
                    f"Epoch {epoch + 1} | Batch {batch_idx} | Step {global_step} | "
                    f"Loss: {loss.item() * train_cfg.accum_steps:.4f} | LR: {curr_lr:.6f} | "
                    f"In: {in_tok_s:.0f} tok/s | Out: {out_tok_s:.0f} tok/s"
                )

                # Aim tracking
                run.track(
                    loss.item() * train_cfg.accum_steps,
                    name="loss",
                    step=global_step,
                    context={"subset": "train"},
                )
                run.track(curr_lr, name="lr", step=global_step)
                run.track(in_tok_s, name="input_tokens_per_sec", step=global_step)
                run.track(out_tok_s, name="output_tokens_per_sec", step=global_step)

                # Reset throughput counters
                batch_src_tokens = 0
                batch_tgt_tokens = 0
                last_log_time = time.time()

        avg_loss = total_loss / (batch_idx + 1)
        print(
            f"Epoch {epoch + 1}/{train_cfg.epochs} Completed | Avg Loss: {avg_loss:.4f} | Time: {time.time() - start_time:.2f}s"
        )

    print("Training complete.")

    # Quick Test
    model.eval()
    test_src = ["اعضای مجلس ولز نگران هستند که «همانند عروسک خیمه‌ شب بازی» دیده شوند"]
    src_ids = src_sp.encode(test_src, out_type=int, add_bos=True, add_eos=True)
    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)

    # Greedy Decode
    print(f"Input: {test_src[0]}")
    print("Output: ", end="")

    with torch.no_grad():
        generated_ids = model.generate(src_tensor, max_len=20)

    decoded = tgt_sp.decode(generated_ids[0].tolist())
    print(decoded)


if __name__ == "__main__":
    train()
