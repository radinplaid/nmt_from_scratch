import torch
import torch.optim as optim
from dataclasses import dataclass
from config import ModelConfig, TrainConfig

from model import Seq2SeqTransformer
from data import PrepareData
import time
from datetime import datetime, timedelta
import os
import sacrebleu
import math
from aim import Run
from safetensors.torch import save_file


def train():
    training_start = time.time()

    def get_time_info():
        elapsed = time.time() - training_start
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        curr_time = datetime.now().strftime("%H:%M:%S")
        return f"[{curr_time}] [{elapsed_str}]"

    # Configs
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{get_time_info()} Using device: {device}")

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
    print(f"{get_time_info()} Preparing data...")

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
    print(f"{get_time_info()} Initializing model...")
    # Enable TF32 for speed on Ampere+
    torch.set_float32_matmul_precision("high")

    model = Seq2SeqTransformer(model_cfg).to(device)
    model = torch.compile(model)
    print(
        f"{get_time_info()} Model parameters: {sum(p.numel() for p in model.parameters())}"
    )

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

        path = os.path.join(config.checkpoint_dir, f"model_{step}.safetensors")
        # Handle torch.compile wrapper if present
        state_dict = (
            model.module.state_dict()
            if hasattr(model, "module")
            else model.state_dict()
        )
        # Remove prefix from torch.compile (_orig_mod.)
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        save_file(state_dict, path)
        print(f"{get_time_info()} Model weights saved: {path}")

        # Save full state (optimizer, scheduler) in .pt for resuming
        path_pt = os.path.join(config.checkpoint_dir, f"checkpoint_{step}.pt")
        torch.save(
            {
                "step": step,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            path_pt,
        )
        print(f"{get_time_info()} Training state saved: {path_pt}")

        # Rotation
        def get_step(f):
            try:
                # model_1000.safetensors or checkpoint_1000.pt
                return int(f.split("_")[1].split(".")[0])
            except (ValueError, IndexError):
                return -1

        all_files = os.listdir(config.checkpoint_dir)
        checkpoints_pt = sorted(
            [f for f in all_files if f.startswith("checkpoint_")], key=get_step
        )
        models_st = sorted(
            [f for f in all_files if f.startswith("model_")], key=get_step
        )

        if len(checkpoints_pt) > config.max_checkpoints:
            os.remove(os.path.join(config.checkpoint_dir, checkpoints_pt[0]))
            print(f"{get_time_info()} Removed old state: {checkpoints_pt[0]}")
        if len(models_st) > config.max_checkpoints:
            os.remove(os.path.join(config.checkpoint_dir, models_st[0]))
            print(f"{get_time_info()} Removed old weights: {models_st[0]}")

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
        total_loss_sum = 0
        total_tokens = 0
        correct_tokens = 0

        # Limit samples for BLEU calculation to reduce memory
        max_samples = 500
        hypotheses = []
        references = []
        sample_count = 0

        # Use inference_mode instead of no_grad for better performance
        with torch.inference_mode():
            for batch_idx, (src, tgt) in enumerate(loader):
                src, tgt = (
                    src.to(device, non_blocking=True),
                    tgt.to(device, non_blocking=True),
                )

                # Forward pass for loss and logits (calculates loss internally)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss_sum, (logits, num_tokens_batch) = model(
                        src, tgt, return_outputs=True
                    )

                # Accumulate loss and tokens
                total_loss_sum += loss_sum.item()
                total_tokens += num_tokens_batch.item()

                # Accuracy calculation
                tgt_labels = tgt[:, 1:]
                preds = logits.argmax(dim=-1)
                mask_acc = tgt_labels != 0
                correct_tokens += ((preds == tgt_labels) & mask_acc).sum().item()

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

        avg_loss = total_loss_sum / max(1, total_tokens)
        ppl = math.exp(min(avg_loss, 100))
        acc = correct_tokens / max(1, total_tokens)

        bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
        chrf = sacrebleu.corpus_chrf(hypotheses, [references]).score

        metrics = {"loss": avg_loss, "ppl": ppl, "acc": acc, "bleu": bleu, "chrf": chrf}

        print(
            f"\n{get_time_info()} [Validation] Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | Acc: {acc:.4f} | BLEU: {bleu:.2f} | ChrF: {chrf:.2f}"
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
        total_loss_sum = 0
        total_tokens_epoch = 0
        batch_src_tokens = 0
        batch_tgt_tokens = 0
        last_log_time = time.time()

        # Token-based accumulation state
        accum_loss = 0
        accum_tokens = 0
        last_batch_loss = 0.0

        for batch_idx, (src, tgt) in enumerate(train_loader):
            # Use non_blocking for async data transfer
            src, tgt = (
                src.to(device, non_blocking=True),
                tgt.to(device, non_blocking=True),
            )

            # Allow any efficient backend (Flash or MemEfficient). Math is fallback.
            # We don't strictly enforce Flash because it might fail with certain masks/dtypes.
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, num_tokens = model(
                    src, tgt, label_smoothing=train_cfg.label_smoothing
                )

            loss.backward()
            accum_loss += loss.item()
            accum_tokens += num_tokens.item()

            total_loss_sum += loss.item()
            total_tokens_epoch += num_tokens.item()

            # Throughput tracking
            batch_src_tokens += (src != 0).sum().item()
            batch_tgt_tokens += (tgt != 0).sum().item()

            if (batch_idx + 1) % train_cfg.accum_steps == 0:
                # Scale gradients by total number of tokens in the accumulation bucket
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.data.div_(max(1, accum_tokens))

                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                last_batch_loss = accum_loss / max(1, accum_tokens)
                accum_loss = 0
                accum_tokens = 0
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
                    f"{get_time_info()} Epoch {epoch + 1} | Batch {batch_idx} | Step {global_step} | "
                    f"Loss: {last_batch_loss:.4f} | LR: {curr_lr:.6f} | "
                    f"In: {in_tok_s:.0f} tok/s | Out: {out_tok_s:.0f} tok/s"
                )

                # Aim tracking
                run.track(
                    last_batch_loss,
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

        avg_loss = total_loss_sum / max(1, total_tokens_epoch)
        print(
            f"{get_time_info()} Epoch {epoch + 1}/{train_cfg.epochs} Completed | Avg Loss: {avg_loss:.4f} | Epoch Time: {time.time() - start_time:.2f}s"
        )

    print(f"{get_time_info()} Training complete.")

    # Quick Test with 5 examples from dev data
    print(f"\n{get_time_info()} Running final quick test on 5 dev samples:")
    model.eval()

    samples_found = 0
    with torch.inference_mode():
        for src, tgt in dev_loader:
            src, tgt = src.to(device), tgt.to(device)
            # Process up to 5 samples from this batch
            n = min(5 - samples_found, src.size(0))

            for i in range(n):
                s_tensor = src[i : i + 1]
                t_tensor = tgt[i : i + 1]

                # Generate
                generated_ids = model.generate(s_tensor, max_len=model_cfg.max_len)

                # Decoding
                # Helper to remove padding and decode
                def cleanup_and_decode(ids_tensor, sp):
                    ids = ids_tensor[0].tolist()
                    # Remove padding (0)
                    ids = [idx for idx in ids if idx != 0]
                    return sp.decode(ids)

                s_text = cleanup_and_decode(s_tensor, src_sp)
                t_ref = cleanup_and_decode(t_tensor, tgt_sp)
                t_hyp = cleanup_and_decode(generated_ids, tgt_sp)

                print(f"Example {samples_found + 1}:")
                print(f"  Input:  {s_text}")
                print(f"  Ref:    {t_ref}")
                print(f"  Output: {t_hyp}")
                print()

                samples_found += 1

            if samples_found >= 5:
                break


if __name__ == "__main__":
    train()
