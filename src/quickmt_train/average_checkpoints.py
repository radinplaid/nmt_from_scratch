import torch
import os
import fire
import json
from safetensors.torch import load_file, save_model
from .config import load_config
from .model import Seq2SeqTransformer
from .data import PrepareData


def main():
    import fire
    fire.Fire(average_checkpoints_cli)


def average_checkpoints_cli(experiment_dir: str):
    """
    Average the last k checkpoints and save as safetensors/INT8.

    Args:
        experiment_dir: Path to experiment directory
    """
    model_cfg, data_cfg, train_cfg, export_cfg = load_config(
        os.path.join(experiment_dir, "config.yaml")
    )

    # 1. Find the best k models based on validation perplexity
    metrics_path = os.path.join(experiment_dir, "metrics.jsonl")
    if os.path.exists(metrics_path):
        print(f"Reading metrics from {metrics_path}")
        metrics = []
        with open(metrics_path, "r") as f:
            for line in f:
                metrics.append(json.loads(line))

        # Sort by perplexity (ppl) ascending (lower is better)
        metrics.sort(key=lambda x: x.get("ppl", float("inf")))

        best_steps = [m["step"] for m in metrics]
        selected = [f"model_{step}.safetensors" for step in best_steps]

        # Verify files exist
        selected = [
            f
            for f in selected
            if os.path.exists(os.path.join(train_cfg.checkpoint_dir, f))
        ]
        selected = selected[: export_cfg.k]
        print(f"Selected {len(selected)} best checkpoints based on PPL.")
    else:
        print(
            f"Metrics file {metrics_path} not found. Falling back to last k checkpoints."
        )
        if not os.path.exists(train_cfg.checkpoint_dir):
            print(f"Directory {train_cfg.checkpoint_dir} not found.")
            return

        checkpoints = [
            f
            for f in os.listdir(train_cfg.checkpoint_dir)
            if f.startswith("model_")
            and f.endswith(".safetensors")
            and "_int8" not in f
        ]

        # Sort by step number
        checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]), reverse=True)
        selected = checkpoints[: export_cfg.k]

    if not selected:
        print("No model files found.")
        return

    print(f"Averaging {len(selected)} model checkpoints:")
    for c in selected:
        print(f" - {c}")

    # 2. Load and average state dicts
    avg_state_dict: dict[str, torch.Tensor] = {}
    count = len(selected)

    for i, ckpt_name in enumerate(selected):
        ckpt_path = os.path.join(train_cfg.checkpoint_dir, ckpt_name)
        clean_state_dict = load_file(ckpt_path, device="cpu")

        if not avg_state_dict:
            avg_state_dict = clean_state_dict
        else:
            for k in clean_state_dict:
                if k in avg_state_dict:
                    avg_state_dict[k] += clean_state_dict[k]
                else:
                    # This might happen if mixing different architectures
                    print(f"Warning: Key {k} not found in first checkpoint. Skipping.")

    # Divide by count
    for k in avg_state_dict:
        # Only divide floating point tensors
        if avg_state_dict[k].is_floating_point():
            avg_state_dict[k] = avg_state_dict[k] / count
        else:
            avg_state_dict[k] = torch.div(
                avg_state_dict[k], count, rounding_mode="floor"
            )

    # 3. Save as .pt and .safetensors (FP32/Averaged weights)
    # Ensure experiment directory exists
    os.makedirs(train_cfg.experiment_name, exist_ok=True)

    pt_output = f"{export_cfg.output_prefix}.pt"
    torch.save({"model_state_dict": avg_state_dict}, pt_output)

    st_output = f"{export_cfg.output_prefix}.safetensors"
    # Create model to handle shared tensors correctly during save
    model = Seq2SeqTransformer(model_cfg).to("cpu")
    model.load_state_dict(avg_state_dict, strict=False)
    save_model(model, st_output)
    print(f"Saved averaged model to {pt_output} and {st_output}")

    # 4. Calibration and INT8 Export
    if export_cfg.export_int8:
        print("\nStarting re-calibration for INT8 export...")

        # Override settings for calibration
        data_cfg.max_tokens_per_batch = 2048
        data_cfg.buffer_size = 10000
        data_cfg.num_workers = 0
        _, dev_loader, _, _ = PrepareData(model_cfg, data_cfg, train_cfg)

        model = Seq2SeqTransformer(model_cfg).to("cpu")

        # Load averaged weights BEFORE preparing for quantization
        print("Loading averaged weights...")
        model.load_state_dict(avg_state_dict, strict=False)
        model.eval()

        # Prepare model for Post-Training Quantization (PTQ)
        print("Preparing model for Post-Training Quantization (PTQ)...")
        # Set quantization config
        if export_cfg.qconfig_backend == "fbgemm":
            # Use a more robust qconfig for x86
            model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
            # Histogram observer is generally better for activations in Transformers
            model.qconfig.activation = (
                torch.ao.quantization.HistogramObserver.with_args(reduce_range=True)
            )
        else:
            model.qconfig = torch.ao.quantization.get_default_qconfig(
                export_cfg.qconfig_backend
            )

        # Disable quantization for Embedding
        for name, module in model.named_modules():
            if any(k in name for k in ["self_attn", "multihead_attn", "emb"]):
                module.qconfig = None  # type: ignore

        # Prepare the model (inserts observers)
        torch.ao.quantization.prepare(model, inplace=True)

        # Calibrate
        model.calibrate(dev_loader, num_batches=export_cfg.calib_batches)

        # Convert and Save
        model.convert_to_int8()
        int8_state_dict = model.state_dict()
        int8_output = f"{export_cfg.output_prefix}_int8.pt"
        torch.save({"model_state_dict": int8_state_dict}, int8_output)
        print(f"Saved calibrated INT8 model to {int8_output}")

        # Also save as safetensors (dequantized) for easier loading
        st_int8_output = f"{export_cfg.output_prefix}_int8.safetensors"
        dequantized_state_dict = {}
        for k, v in int8_state_dict.items():
            # Handle packed params
            if k.endswith("._packed_params._packed_params") and isinstance(v, tuple):
                # This is a bit tricky for safetensors as we need to map it back to .weight and .bias
                prefix = k.replace("._packed_params._packed_params", "")
                qweight, bias = v
                dequantized_state_dict[f"{prefix}.weight"] = (
                    qweight.dequantize() if hasattr(qweight, "dequantize") else qweight
                )
                if bias is not None:
                    dequantized_state_dict[f"{prefix}.bias"] = bias
            elif hasattr(v, "dequantize"):
                dequantized_state_dict[k] = v.dequantize()
            else:
                dequantized_state_dict[k] = v

        # Remove scale and zero_point if they exist as they are now baked into the dequantized weights
        # Also remove any non-tensor values (like .dtype) which safetensors doesn't support
        keys_to_remove = [
            k
            for k, v in dequantized_state_dict.items()
            if k.endswith(".scale")
            or k.endswith(".zero_point")
            or not isinstance(v, torch.Tensor)
        ]
        for k in keys_to_remove:
            del dequantized_state_dict[k]

        temp_model = Seq2SeqTransformer(model_cfg).to("cpu")
        temp_model.load_state_dict(dequantized_state_dict, strict=False)
        save_model(temp_model, st_int8_output)
        print(f"Saved dequantized INT8 model to {st_int8_output}")


if __name__ == "__main__":
    fire.Fire(main)
