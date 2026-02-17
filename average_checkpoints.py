import torch
import os
import argparse
from safetensors.torch import load_file, save_file
from config import ModelConfig, TrainConfig
from model import Seq2SeqTransformer
from data import PrepareData


def main():
    parser = argparse.ArgumentParser(
        description="Average the last k checkpoints and save as safetensors/INT8"
    )
    parser.add_argument(
        "--k", type=int, default=4, help="Number of checkpoints to average"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory containing checkpoints",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="averaged_model",
        help="Prefix for output files",
    )
    parser.add_argument(
        "--export_int8",
        action="store_true",
        help="If set, calibrate and export as INT8 model",
    )
    parser.add_argument(
        "--calib_batches",
        type=int,
        default=500,
        help="Number of batches for calibration",
    )
    args = parser.parse_args()

    # 1. Find the last k models
    if not os.path.exists(args.checkpoint_dir):
        print(f"Directory {args.checkpoint_dir} not found.")
        return

    checkpoints = [
        f
        for f in os.listdir(args.checkpoint_dir)
        if f.startswith("model_") and f.endswith(".safetensors") and "_int8" not in f
    ]

    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]), reverse=True)

    selected = checkpoints[: args.k]

    if not selected:
        print("No model files found.")
        return

    print(f"Averaging {len(selected)} model checkpoints:")
    for c in selected:
        print(f" - {c}")

    # 2. Load and average state dicts
    avg_state_dict = None
    count = len(selected)

    for i, ckpt_name in enumerate(selected):
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
        clean_state_dict = load_file(ckpt_path, device="cpu")

        if avg_state_dict is None:
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
    pt_output = f"{args.output_prefix}.pt"
    torch.save({"model_state_dict": avg_state_dict}, pt_output)

    st_output = f"{args.output_prefix}.safetensors"
    save_file(avg_state_dict, st_output)
    print(f"Saved averaged model to {pt_output} and {st_output}")

    # 4. Calibration and INT8 Export
    if args.export_int8:
        print("\nStarting re-calibration for INT8 export...")
        model_cfg = ModelConfig()
        train_cfg = TrainConfig()

        # Override settings for calibration
        train_cfg.batch_size = 8
        train_cfg.max_tokens_per_batch = 2048
        train_cfg.buffer_size = 10000
        train_cfg.num_workers = 0
        _, dev_loader, _, _ = PrepareData(model_cfg, train_cfg)

        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = "cpu"
        model = Seq2SeqTransformer(model_cfg).to(device)

        # Load averaged weights BEFORE preparing for quantization
        print("Loading averaged weights...")
        model.load_state_dict(avg_state_dict)

        # Prepare model for Post-Training Quantization (PTQ)
        print("Preparing model for Post-Training Quantization (PTQ)...")
        # Set quantization config
        model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        
        # Disable quantization for MultiheadAttention to avoid masked_fill dtype bug
        # Also disable for Embedding as it requires special qconfig
        for name, module in model.named_modules():
            if any(k in name for k in ["self_attn", "multihead_attn", "emb"]):
                module.qconfig = None

        # Prepare the model (inserts observers)
        torch.ao.quantization.prepare(model, inplace=True)

        # Calibrate
        model.calibrate(dev_loader, num_batches=args.calib_batches)

        # Convert and Save
        model.convert_to_int8()
        int8_output = f"{args.output_prefix}_int8.pt"
        torch.save(model.state_dict(), int8_output)
        print(f"Saved calibrated INT8 model to {int8_output}")


if __name__ == "__main__":
    main()
