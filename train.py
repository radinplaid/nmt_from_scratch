import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
import contextlib
from torch.nn.parallel import DistributedDataParallel as DDP
from config import ModelConfig, DataConfig, TrainConfig

from model import Seq2SeqTransformer
from data import PrepareData
import time
from datetime import datetime, timedelta
import os
import sacrebleu
import math
from aim import Run
from safetensors.torch import save_file


class EMA:
    """
    Exponential Moving Average for model parameters.
    """

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def _get_raw_model(self):
        # Handle torch.compile wrapper if present
        model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        # Handle DDP wrapper if present
        model = model.module if hasattr(model, "module") else model
        return model

    def register(self):
        raw_model = self._get_raw_model()
        for name, param in raw_model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self):
        raw_model = self._get_raw_model()
        for name, param in raw_model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                # v_new = (1 - alpha) * v_old + alpha * v_current
                # We use in-place operations to save memory
                self.shadow[name].mul_(1.0 - self.decay).add_(param.detach(), alpha=self.decay)

    @torch.no_grad()
    def apply_shadow(self):
        raw_model = self._get_raw_model()
        for name, param in raw_model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.detach().clone()
                param.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self):
        raw_model = self._get_raw_model()
        for name, param in raw_model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.copy_(self.backup[name])
        self.backup = {}


def train(model_cfg=None, data_cfg=None, train_cfg=None):
    training_start = time.time()

    def get_time_info():
        elapsed = time.time() - training_start
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        curr_time = datetime.now().strftime("%H:%M:%S")
        return f"[{curr_time}] [{elapsed_str}]"

    # Configs
    if model_cfg is None:
        model_cfg = ModelConfig()
    if data_cfg is None:
        data_cfg = DataConfig()
    if train_cfg is None:
        train_cfg = TrainConfig()

    # Device selection
    if train_cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(train_cfg.device)

    # DDP Setup
    is_distributed = False
    rank = 0
    world_size = 1
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"{device.type}:{rank}")
        is_distributed = True
        if rank != 0:
            # Suppress printing on non-zero ranks
            import builtins

            builtins.print = lambda *args, **kwargs: None

    print(f"{get_time_info()} Using device: {device}")

    # Performance optimizations
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
        torch.backends.cuda.matmul.allow_tf32 = train_cfg.tf32  # Allow TF32 on matmul
        torch.backends.cudnn.allow_tf32 = train_cfg.tf32  # Allow TF32 on cudnn
        if train_cfg.tf32:
            torch.set_float32_matmul_precision("high")

    run = Run(repo=train_cfg.aim_repo, experiment=train_cfg.experiment_name)
    run["hparams"] = {
        **{f"model_{k}": v for k, v in model_cfg.__dict__.items()},
        **{f"data_{k}": v for k, v in data_cfg.__dict__.items()},
        **{f"train_{k}": v for k, v in train_cfg.__dict__.items()},
    }

    # Data
    print(f"{get_time_info()} Preparing data...")

    train_loader, dev_loader, src_sp, tgt_sp = PrepareData(
        model_cfg, data_cfg, train_cfg
    )

    # Model
    print(f"{get_time_info()} Initializing model...")

    model = Seq2SeqTransformer(model_cfg).to(device)

    # Convert model to precision for reduced memory footprint
    if device.type == "cuda" and train_cfg.precision == "bf16":
        model = model.to(dtype=torch.bfloat16)

    if is_distributed:
        model = DDP(model, device_ids=[rank] if device.type == "cuda" else None)
    elif torch.cuda.device_count() > 1 and train_cfg.device in ["cuda", "auto"]:
        print(
            f"{get_time_info()} Detected {torch.cuda.device_count()} GPUs. Using DataParallel."
        )
        model = nn.DataParallel(model)

    model = torch.compile(model)

    print(
        f"{get_time_info()} Model parameters: {sum(p.numel() for p in model.parameters())}"
    )

    print(
        f"{get_time_info()} Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    # Print model architecture
    print(f"\n{get_time_info()} Model Architecture:")
    print("-" * 60)
    print(model)
    print("-" * 60)

    # Print configs
    print(f"\n{get_time_info()} Configuration:")
    print("-" * 60)
    print("Model Config:")
    for key, value in model_cfg.__dict__.items():
        print(f"  {key}: {value}")
    print("\nData Config:")
    for key, value in data_cfg.__dict__.items():
        print(f"  {key}: {value}")
    print("\nTrain Config:")
    for key, value in train_cfg.__dict__.items():
        print(f"  {key}: {value}")
    print("-" * 60)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        eps=train_cfg.adam_eps,
        betas=train_cfg.adam_betas
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
    global_step = 0

    # EMA
    ema = None
    if train_cfg.ema_decay > 0 and train_cfg.ema_start_step == 0:
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        ema = EMA(raw_model, train_cfg.ema_decay)
        ema.register()
        print(f"{get_time_info()} Initialized EMA with decay {train_cfg.ema_decay}")

    def save_checkpoint(step, model, optimizer, scheduler, config, ema=None):
        # Ensure experiment directory exists
        os.makedirs(config.experiment_name, exist_ok=True)

        if not os.path.exists(config.checkpoint_dir):
            os.makedirs(config.checkpoint_dir)

        path = os.path.join(config.checkpoint_dir, f"model_{step}.safetensors")
        # Handle torch.compile wrapper if present
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        # Handle DDP wrapper if present
        raw_model = raw_model.module if hasattr(raw_model, "module") else raw_model

        from safetensors.torch import save_model

        save_model(raw_model, path)
        print(f"{get_time_info()} Model weights saved: {path}")

        if ema is not None:
            ema.apply_shadow()
            try:
                ema_path = os.path.join(config.checkpoint_dir, f"model_{step}_ema.safetensors")
                save_model(raw_model, ema_path)
                print(f"{get_time_info()} EMA weights saved: {ema_path}")
            finally:
                ema.restore()

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

        # If it's a quantized model, also save a converted version for inference
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        if hasattr(raw_model, "qconfig") and raw_model.qconfig is not None:
            import copy

            try:
                quant_model = copy.deepcopy(raw_model)
                quant_model.convert_to_int8()
                quant_path = os.path.join(
                    config.checkpoint_dir, f"model_{step}_int8.pt"
                )
                torch.save(quant_model.state_dict(), quant_path)
                print(f"{get_time_info()} Exported INT8 model: {quant_path}")
            except Exception as e:
                print(f"{get_time_info()} Could not export INT8 model: {e}")

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

    def validate(
        model,
        loader,
        src_sp,
        tgt_sp,
        device,
        train_cfg,
        data_cfg,
        model_cfg,
        use_autoregressive=True,
        ema=None,
    ):
        """
        Validate the model.
        """
        if ema is not None:
            ema.apply_shadow()
            print(f"{get_time_info()} [Validation] Using EMA weights")

        try:
            model.eval()
            total_loss_sum = 0
            total_tokens = 0
            correct_tokens = 0

            hypotheses = []
            references = []

            # Use inference_mode instead of no_grad for better performance
            autocast_dtype = (
                torch.bfloat16 if train_cfg.precision == "bf16" else torch.float32
            )

            with torch.inference_mode():
                for batch_idx, (src, tgt) in enumerate(loader):
                    src, tgt = (
                        src.to(device, non_blocking=True),
                        tgt.to(device, non_blocking=True),
                    )

                    # Forward pass for loss and logits (calculates loss internally)
                    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                        loss_sum, (logits, num_tokens_batch) = model(
                            src, tgt, return_outputs=True
                        )

                    # Handle DataParallel output (vectors per GPU)
                    if loss_sum.ndim > 0:
                        loss_sum = loss_sum.sum()
                    if num_tokens_batch.ndim > 0:
                        num_tokens_batch = num_tokens_batch.sum()

                    # Accumulate loss and tokens
                    total_loss_sum += loss_sum.item()
                    total_tokens += num_tokens_batch.item()

                    # Accuracy calculation
                    tgt_labels = tgt[:, 1:]
                    preds = logits.argmax(dim=-1)
                    mask_acc = tgt_labels != model_cfg.pad_id
                    correct_tokens += ((preds == tgt_labels) & mask_acc).sum().item()

                    # Generation for BLEU/ChrF
                    if use_autoregressive:
                        # True autoregressive generation including encoding
                        raw_model = model.module if hasattr(model, "module") else model
                        generated_ids = raw_model.generate(
                            src,
                            max_len=model_cfg.max_len,
                            bos_id=model_cfg.bos_id,
                            eos_id=model_cfg.eos_id,
                        )
                    else:
                        # Teacher-forced predictions (fastest, uses existing logits)
                        generated_ids = preds

                    for i in range(src.size(0)):
                        # Post-process: stop at EOS or PAD tokens
                        ids = generated_ids[i].tolist()
                        # Find first EOS or PAD token and truncate
                        for idx, token_id in enumerate(ids):
                            if (
                                token_id == model_cfg.eos_id
                                or token_id == model_cfg.pad_id
                            ):
                                ids = ids[:idx]
                                break
                        hyp = tgt_sp.decode(ids)
                        
                        # Reference decoding: also stop at EOS/PAD
                        ref_ids = tgt[i].tolist()
                        # Skip BOS (usually at index 0)
                        if len(ref_ids) > 0 and ref_ids[0] == model_cfg.bos_id:
                            ref_ids = ref_ids[1:]
                        for idx, token_id in enumerate(ref_ids):
                            if (
                                token_id == model_cfg.eos_id
                                or token_id == model_cfg.pad_id
                            ):
                                ref_ids = ref_ids[:idx]
                                break
                        ref = tgt_sp.decode(ref_ids)
                        
                        hypotheses.append(hyp)
                        references.append(ref)

            avg_loss = total_loss_sum / max(1, total_tokens)
            ppl = math.exp(min(avg_loss, 100))
            acc = correct_tokens / max(1, total_tokens)

            # sacrebleu expects a list of lists for references (multiple references per hypothesis)
            bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
            chrf = sacrebleu.corpus_chrf(hypotheses, [references]).score

            metrics = {"loss": avg_loss, "ppl": ppl, "acc": acc, "bleu": bleu, "chrf": chrf}

            print(
                f"\n{get_time_info()} [Validation] Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | Acc: {acc:.4f} | BLEU: {bleu:.2f} | ChrF: {chrf:.2f}"
            )
            for i in range(min(5, len(hypotheses))):
                print(f"Sample {i}:")
                print(f"  Ref: {references[i]}")
                print(f"  Hyp: {hypotheses[i]}")
            print("-" * 30)
        finally:
            if ema is not None:
                ema.restore()
                print(f"{get_time_info()} [Validation] Restored original weights")

            model.train()
        
        return metrics

    # Loop
    model.train()
    optimizer.zero_grad()
    autocast_dtype = torch.bfloat16 if train_cfg.precision == "bf16" else torch.float32

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

            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                # Disable gradient synchronization during accumulation steps
                context = (
                    model.no_sync()
                    if is_distributed
                    and (batch_idx + 1) % train_cfg.accum_steps != 0
                    else contextlib.nullcontext()
                )
                with context:
                    loss, num_tokens = model(
                        src, tgt, label_smoothing=train_cfg.label_smoothing
                    )

                    # Handle DataParallel/DDP output (vectors per GPU)
                    if loss.ndim > 0:
                        loss = loss.sum()
                    if num_tokens.ndim > 0:
                        num_tokens = num_tokens.sum()

                    # Scale loss by accumulation steps to simulate larger batch size
                    # This is the standard way to do accumulation in PyTorch
                    loss = loss / train_cfg.accum_steps

                loss.backward()
            accum_loss += loss.item()
            accum_tokens += num_tokens.item()

            total_loss_sum += loss.item()
            total_tokens_epoch += num_tokens.item()

            # Throughput tracking
            batch_src_tokens += (src != model_cfg.pad_id).sum().item()
            batch_tgt_tokens += (tgt != model_cfg.pad_id).sum().item()

            if (batch_idx + 1) % train_cfg.accum_steps == 0:
                if is_distributed:
                    # Average gradients across all processes
                    # We also need to sync the token count to ensure correct scaling
                    tokens_tensor = torch.tensor([accum_tokens], device=device)
                    dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
                    total_accum_tokens = tokens_tensor.item()

                    for p in model.parameters():
                        if p.grad is not None:
                            dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                            # Scale by total tokens across all GPUs
                            # We multiply by accum_steps because we divided the loss by it earlier
                            p.grad.data.mul_(train_cfg.accum_steps).div_(max(1, total_accum_tokens))
                else:
                    # Scale gradients by total number of tokens in the accumulation bucket
                    for p in model.parameters():
                        if p.grad is not None:
                            # We multiply by accum_steps because we divided the loss by it earlier
                            p.grad.data.mul_(train_cfg.accum_steps).div_(max(1, accum_tokens))

                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                optimizer.step()
                
                # EMA Update
                if train_cfg.ema_decay > 0:
                    if global_step + 1 == train_cfg.ema_start_step:
                        # Get the raw model to avoid any torch.compile issues
                        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                        ema = EMA(raw_model, train_cfg.ema_decay)
                        ema.register()
                        print(f"{get_time_info()} Initialized EMA at step {global_step + 1}")
                    elif global_step + 1 > train_cfg.ema_start_step:
                        if ema is not None:
                            ema.update()

                scheduler.step()
                optimizer.zero_grad()

                last_batch_loss = accum_loss / max(1, accum_tokens)
                accum_loss = 0
                accum_tokens = 0
                global_step += 1

                if global_step >= train_cfg.max_steps:
                    # Final validation and checkpoint
                    val_metrics = validate(
                        model,
                        dev_loader,
                        src_sp,
                        tgt_sp,
                        device,
                        train_cfg,
                        data_cfg,
                        model_cfg,
                        ema=ema,
                    )
                    save_checkpoint(global_step, model, optimizer, scheduler, train_cfg, ema=ema)
                    break

                # Validation and Checkpointing
                if global_step % train_cfg.eval_steps == 0:
                    val_metrics = validate(
                        model,
                        dev_loader,
                        src_sp,
                        tgt_sp,
                        device,
                        train_cfg,
                        data_cfg,
                        model_cfg,
                        ema=ema,
                    )
                    for k, v in val_metrics.items():
                        run.track(
                            v,
                            name=f"val_{k}",
                            step=global_step,
                            context={"subset": "dev"},
                        )
                    save_checkpoint(global_step, model, optimizer, scheduler, train_cfg, ema=ema)

            # Progress Print
            if batch_idx % train_cfg.log_steps == 0:
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

            if global_step >= train_cfg.max_steps:
                break

        avg_loss = total_loss_sum / max(1, total_tokens_epoch)
        print(
            f"{get_time_info()} Epoch {epoch + 1}/{train_cfg.epochs} Completed | Avg Loss: {avg_loss:.4f} | Epoch Time: {time.time() - start_time:.2f}s"
        )

    print(f"{get_time_info()} Training complete.")

    # Quick Test with examples from dev data
    print(
        f"\n{get_time_info()} Running final quick test on {train_cfg.quick_test_samples} dev samples:"
    )
    if ema is not None:
        ema.apply_shadow()
        print(f"{get_time_info()} Using EMA weights for final test")

    model.eval()

    samples_found = 0
    with torch.inference_mode():
        for src, tgt in dev_loader:
            src, tgt = src.to(device), tgt.to(device)
            # Process up to n samples from this batch
            n = min(train_cfg.quick_test_samples - samples_found, src.size(0))

            for i in range(n):
                s_tensor = src[i : i + 1]
                t_tensor = tgt[i : i + 1]

                # Generate
                raw_model = model.module if hasattr(model, "module") else model
                generated_ids = raw_model.generate(
                    s_tensor,
                    max_len=model_cfg.max_len,
                    bos_id=model_cfg.bos_id,
                    eos_id=model_cfg.eos_id,
                )

                # Decoding
                # Helper to remove padding and decode
                def cleanup_and_decode(ids_tensor, sp, pad_id, eos_id):
                    ids = ids_tensor[0].tolist()
                    # Stop at EOS or PAD tokens
                    for idx, token_id in enumerate(ids):
                        if token_id == eos_id or token_id == pad_id:
                            ids = ids[:idx]
                            break
                    return sp.decode(ids)

                s_text = cleanup_and_decode(
                    s_tensor, src_sp, model_cfg.pad_id, model_cfg.eos_id
                )
                t_ref = cleanup_and_decode(
                    t_tensor, tgt_sp, model_cfg.pad_id, model_cfg.eos_id
                )
                t_hyp = cleanup_and_decode(
                    generated_ids, tgt_sp, model_cfg.pad_id, model_cfg.eos_id
                )

                print(f"Example {samples_found + 1}:")
                print(f"  Input:  {s_text}")
                print(f"  Ref:    {t_ref}")
                print(f"  Output: {t_hyp}")
                print()

                samples_found += 1

            if samples_found >= train_cfg.quick_test_samples:
                break


if __name__ == "__main__":
    import argparse
    from config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()

    model_cfg = None
    data_cfg = None
    train_cfg = None
    if args.config:
        model_cfg, data_cfg, train_cfg, _ = load_config(args.config)

    train(model_cfg, data_cfg, train_cfg)
