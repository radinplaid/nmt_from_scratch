import math
import os
import time
import json
from datetime import datetime, timedelta

import sacrebleu
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from aim import Run
from safetensors.torch import load_file, save_model
from shutil import copyfile

from config import DataConfig, ModelConfig, TrainConfig
from data import PrepareData
from model import Seq2SeqTransformer


def train(model_cfg=None, data_cfg=None, train_cfg=None, trial=None):
    training_start = time.time()
    best_ppl = float("inf")

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

    # Remove metrics file if exists
    metrics_path = os.path.join(train_cfg.experiment_name, "metrics.jsonl")
    if os.path.exists(metrics_path):
        os.remove(metrics_path)

    # Device selection
    if train_cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(train_cfg.device)
    print(f"{get_time_info()} Using device: {device}")

    # Performance optimizations
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
        torch.backends.cuda.matmul.allow_tf32 = train_cfg.tf32  # Allow TF32 on matmul
        torch.backends.cudnn.allow_tf32 = train_cfg.tf32  # Allow TF32 on cudnn
        if train_cfg.tf32:
            torch.set_float32_matmul_precision("high")

    run = Run(repo=train_cfg.aim_repo, experiment=train_cfg.experiment_name)
    import dataclasses

    run["hparams"] = {
        **{f"model_{k}": v for k, v in dataclasses.asdict(model_cfg).items()},
        **{f"data_{k}": v for k, v in dataclasses.asdict(data_cfg).items()},
        **{f"train_{k}": v for k, v in dataclasses.asdict(train_cfg).items()},
    }

    # Data
    print(f"{get_time_info()} Preparing data...")

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    global_step_value = ctx.Value("i", 0)

    train_loader, dev_loader, src_sp, tgt_sp = PrepareData(
        model_cfg, data_cfg, train_cfg, global_step_value=global_step_value
    )

    # Model
    print(f"{get_time_info()} Initializing model...")

    model = Seq2SeqTransformer(model_cfg).to(device)

    # Convert model to precision for reduced memory footprint
    if device.type == "cuda" and train_cfg.precision in ("bf16", "bfloat16"):
        model = model.to(dtype=torch.bfloat16)

    # Checkpoint loading (weights)
    if train_cfg.resume_from:
        checkpoint_path = train_cfg.resume_from
        weights_path = None

        if checkpoint_path.endswith(".safetensors"):
            weights_path = checkpoint_path
        elif checkpoint_path.endswith(".pt"):
            # Could be a full checkpoint or just weights
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if (
                isinstance(checkpoint, dict)
                and "optimizer_state_dict" not in checkpoint
            ):
                # Likely just weights in .pt
                model.load_state_dict(checkpoint)
                print(f"{get_time_info()} Loaded weights from {checkpoint_path}")
            elif isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
                # Full checkpoint state, find weights
                step = checkpoint.get("step", 0)
                # Try to find model_{step}.safetensors in the same directory
                weights_path = os.path.join(
                    os.path.dirname(checkpoint_path), f"model_{step}.safetensors"
                )
                if not os.path.exists(weights_path):
                    print(
                        f"{get_time_info()} Warning: weights file not found for checkpoint at {weights_path}"
                    )
                    weights_path = None

        if weights_path:
            print(f"{get_time_info()} Loading weights from {weights_path}")
            state_dict = load_file(weights_path, device=device.type)
            # Remove _orig_mod. prefix if present in state_dict (shouldn't be, but safe)
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)

    model = torch.compile(model)

    if torch.cuda.device_count() > 1 and train_cfg.device in ["cuda", "auto"]:
        print(
            f"{get_time_info()} Detected {torch.cuda.device_count()} GPUs. Using DataParallel."
        )
        model = nn.DataParallel(model)
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
        betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
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

    # Checkpoint loading (state)
    if (
        train_cfg.resume_from
        and train_cfg.resume_from.endswith(".pt")
        and not train_cfg.reset_optimizer
    ):
        checkpoint = torch.load(train_cfg.resume_from, map_location=device)
        if isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
            print(
                f"{get_time_info()} Resuming optimizer and scheduler state from {train_cfg.resume_from}"
            )
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            global_step = checkpoint.get("step", 0)
            global_step_value.value = global_step
            print(f"{get_time_info()} Resumed from step {global_step}")

    def save_checkpoint(step, model, optimizer, scheduler, config, val_metrics=None):
        # Ensure experiment directory exists
        os.makedirs(config.experiment_name, exist_ok=True)

        # Save validation metrics to jsonl
        if val_metrics is not None:
            metrics_path = os.path.join(config.experiment_name, "metrics.jsonl")
            with open(metrics_path, "a") as f:
                metric_entry = {"step": step, **val_metrics}
                f.write(json.dumps(metric_entry) + "\n")

        if not os.path.exists(config.checkpoint_dir):
            os.makedirs(config.checkpoint_dir)

        # Use save_model instead of save_file to handle shared tensors (tied embeddings)
        # We need to unwrap the model to get the underlying structure for save_model
        raw_model = model.module if hasattr(model, "module") else model
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod

        path = os.path.join(config.checkpoint_dir, f"model_{step}.safetensors")
        save_model(raw_model, path)
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
        use_autoregressive=False,
    ):
        """
        Validate the model.
        """
        model.eval()
        total_loss_sum = 0
        total_tokens = 0
        correct_tokens = 0

        # Limit samples for BLEU calculation to reduce memory
        max_samples = train_cfg.val_max_samples
        hypotheses = []
        references = []
        sample_count = 0

        # Use inference_mode instead of no_grad for better performance
        autocast_dtype = (
            torch.bfloat16
            if train_cfg.precision in ("bf16", "bfloat16")
            else torch.float32
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

                # Generation for BLEU/ChrF - only process if we still need samples
                if sample_count < max_samples:
                    if use_autoregressive:
                        # True autoregressive generation including encoding
                        raw_model = model.module if hasattr(model, "module") else model
                        enc = raw_model.encode(src)
                        generated_ids = raw_model.generate(
                            src,
                            max_len=model_cfg.max_len,
                            enc_output=enc,
                            bos_id=model_cfg.bos_id,
                            eos_id=model_cfg.eos_id,
                        )
                    else:
                        # Teacher-forced predictions (fastest, uses existing logits)
                        generated_ids = preds

                    for i in range(src.size(0)):
                        if sample_count >= max_samples:
                            break
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
                        ref = tgt_sp.decode(tgt[i].tolist())
                        hypotheses.append(hyp)
                        references.append(ref)
                        sample_count += 1

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
    optimizer.zero_grad()
    autocast_dtype = (
        torch.bfloat16 if train_cfg.precision in ("bf16", "bfloat16") else torch.float32
    )

    start_time = time.time()
    total_loss_sum = 0
    total_tokens_trained = 0
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
            loss, num_tokens = model(
                src, tgt, label_smoothing=train_cfg.label_smoothing
            )

            # Handle DataParallel output (vectors per GPU)
            if loss.ndim > 0:
                loss = loss.sum()
            if num_tokens.ndim > 0:
                num_tokens = num_tokens.sum()

        loss.backward()
        accum_loss += loss.item()
        accum_tokens += num_tokens.item()

        total_loss_sum += loss.item()
        total_tokens_trained += num_tokens.item()

        # Throughput tracking
        batch_src_tokens += (src != model_cfg.pad_id).sum().item()
        batch_tgt_tokens += (tgt != model_cfg.pad_id).sum().item()

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
            global_step_value.value = global_step

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
                )
                for k, v in val_metrics.items():
                    run.track(
                        v,
                        name=f"val_{k}",
                        step=global_step,
                        context={"subset": "dev"},
                    )
                
                # Update best perplexity
                if val_metrics["ppl"] < best_ppl:
                    best_ppl = val_metrics["ppl"]

                # Report to Optuna
                if trial is not None:
                    trial.report(val_metrics["ppl"], global_step)
                    if trial.should_prune():
                        run.close()
                        raise optuna.exceptions.TrialPruned()

                save_checkpoint(
                    global_step,
                    model,
                    optimizer,
                    scheduler,
                    train_cfg,
                    val_metrics=val_metrics,
                )

            if global_step >= train_cfg.max_steps:
                break

        # Progress Print
        if batch_idx % train_cfg.log_steps == 0:
            curr_lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - last_log_time
            in_tok_s = batch_src_tokens / max(1e-6, elapsed)
            out_tok_s = batch_tgt_tokens / max(1e-6, elapsed)

            print(
                f"{get_time_info()} Step {global_step}/{train_cfg.max_steps} | Batch {batch_idx} | "
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

    avg_loss = total_loss_sum / max(1, total_tokens_trained)
    print(
        f"{get_time_info()} Training Completed | Avg Loss: {avg_loss:.4f} | Total Time: {time.time() - start_time:.2f}s"
    )

    print(f"{get_time_info()} Training complete.")
    run.close()

    # Quick Test with examples from dev data
    print(
        f"\n{get_time_info()} Running final quick test on {train_cfg.quick_test_samples} dev samples:"
    )
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
    
    return best_ppl


if __name__ == "__main__":
    import argparse
    import copy
    from config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("--study_name", type=str, default="nmt_optimization", help="Optuna study name")
    args = parser.parse_args()

    base_model_cfg, base_data_cfg, base_train_cfg, _ = load_config(args.config)

    def objective(trial):
        # Sample hyperparameters
        # n_heads must divide d_model
        lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
        n_heads = trial.suggest_categorical("n_heads", [2, 4, 8, 16])
        activation = trial.suggest_categorical("activation", ["gelu", "silu", "relu"])
        mlp_type = trial.suggest_categorical("mlp_type", ['gated','standard'])
        ff_bias = trial.suggest_categorical("ff_bias", [True, False])
        dropout = trial.suggest_float("dropout", 0.0, 0.3, log=False)
        layernorm_eps = trial.suggest_float("layernorm_eps", 1e-6, 1e-4, log=True)
        norm_type = trial.suggest_categorical("norm_type", ["layernorm", "rmsnorm"])
        tie_decoder_embeddings = trial.suggest_categorical("tie_decoder_embeddings", [True, False])
        grad_clip = trial.suggest_float("grad_clip", 0.0, 2.0, log=False)

        # Create trial-specific configs
        model_cfg = copy.deepcopy(base_model_cfg)
        data_cfg = copy.deepcopy(base_data_cfg)
        train_cfg = copy.deepcopy(base_train_cfg)

        train_cfg.lr = lr
        train_cfg.grad_clip = grad_clip 
        model_cfg.n_heads = n_heads
        model_cfg.activation = activation
        model_cfg.mlp_type = mlp_type
        model_cfg.ff_bias = ff_bias
        model_cfg.dropout = dropout
        model_cfg.layernorm_eps = layernorm_eps
        model_cfg.norm_type = norm_type
        model_cfg.tie_decoder_embeddings = tie_decoder_embeddings
        
        # Unique experiment name for each trial
        # Keep commented out to re-use tokenizer etc; speeds up trials considerably
        # train_cfg.experiment_name = f"{base_train_cfg.experiment_name}_trial_{trial.number}"
        # data_cfg.experiment_name = train_cfg.experiment_name
        
        # Make experiment folder
        os.makedirs(train_cfg.experiment_name, exist_ok=True)

        try:
            return train(model_cfg, data_cfg, train_cfg, trial=trial)
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            return float("inf")

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        storage=f"sqlite:///{args.study_name}.db",
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=args.trials)

    print("\nOptimization finished!")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (PPL): {study.best_value:.4f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
