import argparse
import torch
import os
from model import Seq2SeqTransformer
from data import load_tokenizers
from config import load_config
from safetensors.torch import load_file
import sacrebleu
from tqdm import tqdm
from collections import OrderedDict


class PyTorchTranslator:
    def __init__(self, model_cfg, data_cfg, model_path, device="cuda"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading model to {self.device}...")
        self.model = Seq2SeqTransformer(model_cfg).to(self.device)

        if model_path.endswith(".safetensors"):
            state_dict = load_file(model_path, device=self.device)
        else:
            state_dict = torch.load(model_path, map_location=self.device)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]

        # Strip _orig_mod. prefix if present (from torch.compile)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "")
            new_state_dict[new_key] = v

        self.model.load_state_dict(new_state_dict)
        self.model.eval()

        print(f"Loading tokenizers from {data_cfg.tokenizer_prefix_src}...")
        self.src_sp, self.tgt_sp = load_tokenizers(
            data_cfg.tokenizer_prefix_src,
            data_cfg.tokenizer_prefix_tgt,
            expected_vocab_size=model_cfg.vocab_size,
        )
        self.config = model_cfg

    def translate_batch(self, src_texts, beam_size=5, max_len=100):
        # Tokenize
        src_ids = [
            self.src_sp.encode(t, out_type=int, add_bos=True, add_eos=True)
            for t in src_texts
        ]

        # Pad
        max_batch_len = max(len(ids) for ids in src_ids)
        padded_src = [
            ids + [self.config.pad_id] * (max_batch_len - len(ids)) for ids in src_ids
        ]
        src_tensor = torch.tensor(padded_src, device=self.device)

        with torch.no_grad():
            if beam_size > 1:
                translated_ids = self.model.beam_search(
                    src_tensor, beam_size=beam_size, max_len=max_len
                )
            else:
                translated_ids = self.model.generate(src_tensor, max_len=max_len)

        # Decode
        results = []
        for ids in translated_ids:
            # Remove EOS if present
            ids_list = ids.tolist()
            if self.config.eos_id in ids_list:
                ids_list = ids_list[: ids_list.index(self.config.eos_id)]
            results.append(self.tgt_sp.decode(ids_list))

        return results

    def __call__(self, src_lines, beam_size=5, max_batch_size=32, max_len=100):
        hypotheses = []
        for i in tqdm(range(0, len(src_lines), max_batch_size), desc="Translating"):
            batch = src_lines[i : i + max_batch_size]
            hypotheses.extend(
                self.translate_batch(batch, beam_size=beam_size, max_len=max_len)
            )
        return hypotheses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--model",
        type=str,
        help="Path to pytorch model (.pt or .safetensors). If omitted, uses default from config.",
    )
    parser.add_argument("--src_file", type=str, help="Path to source text file")
    parser.add_argument("--ref_file", type=str, help="Path to reference text file")
    parser.add_argument(
        "--beam_size", type=int, default=5, help="Beam size for decoding"
    )
    parser.add_argument("--max_len", type=int, default=100, help="Max sequence length")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for translation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Auto detect GPU and use if available (or cuda, cpu)",
    )
    args = parser.parse_args()

    # Load defaults from config
    model_cfg, data_cfg, train_cfg, export_cfg = load_config(args.config)

    if args.src_file is None:
        args.src_file = data_cfg.src_dev_path
    if args.ref_file is None:
        args.ref_file = data_cfg.tgt_dev_path
    if args.model is None:
        args.model = export_cfg.model_path

    # Pull defaults from export_cfg if not specified on CLI
    if args.beam_size == 5:
        args.beam_size = export_cfg.beam_size
    if args.max_len == 100:
        args.max_len = export_cfg.max_len
    if args.batch_size == 32:
        args.batch_size = export_cfg.batch_size
    if args.device == "auto":
        args.device = train_cfg.device

    if not os.path.exists(args.src_file):
        parser.error(f"Source file not found: {args.src_file}")
    if not os.path.exists(args.ref_file):
        parser.error(f"Reference file not found: {args.ref_file}")
    if not os.path.exists(args.model):
        parser.error(f"Model file not found: {args.model}")

    # Load data
    with open(args.src_file, "r", encoding="utf-8") as f:
        src_lines = [line.strip() for line in f.readlines()]
    with open(args.ref_file, "r", encoding="utf-8") as f:
        ref_lines = [line.strip() for line in f.readlines()][: len(src_lines)]

    if len(src_lines) != len(ref_lines):
        print(
            f"Warning: Source ({len(src_lines)}) and Reference ({len(ref_lines)}) line counts differ."
        )

    # Initialize Translator
    translator = PyTorchTranslator(model_cfg, data_cfg, args.model, device=args.device)

    print(f"Translating {len(src_lines)} lines...")
    hypotheses = translator(
        src_lines,
        beam_size=args.beam_size,
        max_batch_size=args.batch_size,
        max_len=args.max_len,
    )

    # Metrics
    bleu = sacrebleu.corpus_bleu(hypotheses, [ref_lines])
    chrf = sacrebleu.corpus_chrf(hypotheses, [ref_lines])

    print("\n" + "=" * 30)
    print(f"Results for {args.model}:")
    print(f"BLEU: {bleu.score:.2f}")
    print(f"ChrF: {chrf.score:.2f}")
    print("=" * 30)

    # Show some samples
    print("\nSamples:")
    for i in range(min(5, len(hypotheses))):
        print(f"Source: {src_lines[i]}")
        print(f"Ref:    {ref_lines[i]}")
        print(f"Hyp:    {hypotheses[i]}")
        print("-" * 15)


if __name__ == "__main__":
    main()
