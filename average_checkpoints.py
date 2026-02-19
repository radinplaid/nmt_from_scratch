import torch
import os
import argparse
from safetensors.torch import load_file, save_file
from config import load_config
from model import Seq2SeqTransformer
from data import PrepareData


def main():
    parser = argparse.ArgumentParser(
        description="Average the last k checkpoints and save as safetensors/INT8"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    model_cfg, train_cfg, export_cfg = load_config(args.config)

    # 1. Find the last k models
    if not os.path.exists(train_cfg.checkpoint_dir):
        print(f"Directory {train_cfg.checkpoint_dir} not found.")
        return

    checkpoints = [
        f
        for f in os.listdir(train_cfg.checkpoint_dir)
        if f.startswith("model_") and f.endswith(".safetensors") and "_int8" not in f
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
        # Only divide floating point tensors (not scale/zero_point which might be int but are usually float in fake_quant)
        if avg_state_dict[k].is_floating_point():
            avg_state_dict[k] = avg_state_dict[k] / count
        else:
            # For integer buffers (like zero_point), use integer division or just keep last?
            # Usually zero_point in fake_quant is a float tensor in observers.
            avg_state_dict[k] = torch.div(
                avg_state_dict[k], count, rounding_mode="floor"
            )

    # 3. Save as .pt and .safetensors (FP32/Averaged weights)
    pt_output = f"{export_cfg.output_prefix}.pt"
    torch.save({"model_state_dict": avg_state_dict}, pt_output)

    st_output = f"{export_cfg.output_prefix}.safetensors"
    save_file(avg_state_dict, st_output)
    print(f"Saved averaged model to {pt_output} and {st_output}")

    # 4. Calibration and INT8 Export
    # The calibration seems to help a very slight amount compared to int8 quantization with ctranslate2
    # It also enables smaller pt/safetensors model files
    if export_cfg.export_int8:
        print("\nStarting re-calibration for INT8 export...")

        # Override settings for calibration
        train_cfg.max_tokens_per_batch = 2048
        train_cfg.buffer_size = 10000
        train_cfg.num_workers = 0
        _, dev_loader, _, _ = PrepareData(model_cfg, train_cfg)

        model = Seq2SeqTransformer(model_cfg).to("cpu")

        # Load averaged weights BEFORE preparing for quantization
        print("Loading averaged weights...")
        model.load_state_dict(avg_state_dict)
        model.eval()

        # Prepare model for Post-Training Quantization (PTQ)
        print("Preparing model for Post-Training Quantization (PTQ)...")
        # Set quantization config
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")

        # Disable quantization for Embedding as it requires special qconfig (e.g. float16 or quint8)
        # MultiheadAttention is now kept enabled as we've ensured boolean masks in model.py
        for name, module in model.named_modules():
            if any(k in name for k in ["self_attn", "multihead_attn", "emb"]):
                module.qconfig = None  # type: ignore

        # Prepare the model (inserts observers)
        torch.ao.quantization.prepare(model, inplace=True)

        # Calibrate
        model.calibrate(dev_loader, num_batches=export_cfg.calib_batches)

        # Convert and Save
        model.convert_to_int8()
        int8_output = f"{export_cfg.output_prefix}_int8.pt"
        torch.save({"model_state_dict": model.state_dict()}, int8_output)
        print(f"Saved calibrated INT8 model to {int8_output}")


if __name__ == "__main__":
    main()
