import torch
import os
import argparse
from safetensors.torch import save_file
from collections import OrderedDict


def main():
    parser = argparse.ArgumentParser(
        description="Average the last k checkpoints and save as safetensors"
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
    args = parser.parse_args()

    # 1. Find the last k checkpoints
    if not os.path.exists(args.checkpoint_dir):
        print(f"Directory {args.checkpoint_dir} not found.")
        return

    checkpoints = [
        f
        for f in os.listdir(args.checkpoint_dir)
        if f.startswith("checkpoint_") and f.endswith(".pt")
    ]

    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]), reverse=True)

    selected = checkpoints[: args.k]

    if not selected:
        print("No checkpoints found.")
        return

    print(f"Averaging {len(selected)} checkpoints:")
    for c in selected:
        print(f" - {c}")

    # 2. Load and average state dicts
    avg_state_dict = None
    count = len(selected)

    for i, ckpt_name in enumerate(selected):
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"]

        # Strip _orig_mod. prefix if present (from torch.compile)
        clean_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "")
            clean_state_dict[new_key] = v

        if avg_state_dict is None:
            avg_state_dict = clean_state_dict
        else:
            for k in avg_state_dict:
                avg_state_dict[k] += clean_state_dict[k]

    # Divide by count
    for k in avg_state_dict:
        avg_state_dict[k] = avg_state_dict[k] / count

    # 3. Save as .pt
    pt_output = f"{args.output_prefix}.pt"
    torch.save({"model_state_dict": avg_state_dict}, pt_output)
    print(f"Saved averaged model to {pt_output}")

    # 4. Save as .safetensors (only weights)
    st_output = f"{args.output_prefix}.safetensors"
    save_file(avg_state_dict, st_output)
    print(f"Saved averaged model weights to {st_output}")


if __name__ == "__main__":
    main()
