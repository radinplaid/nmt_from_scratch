import argparse
from quickmt import Translator
from config import load_config
import sacrebleu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to ctranslate2 model folder"
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
    parser.add_argument(
        "--compute_type", type=str, default="auto", help="CTranslate2 compute type"
    )
    args = parser.parse_args()

    # Load defaults from config if available
    if args.config:
        model_cfg, data_cfg, train_cfg, export_cfg = load_config(args.config)

        if args.src_file is None:
            args.src_file = data_cfg.src_dev_path
        if args.ref_file is None:
            args.ref_file = data_cfg.tgt_dev_path

        # Pull defaults from export_cfg if not specified on CLI
        # (Assuming argparse defaults are already set, we only override if user didn't provide)
        if args.beam_size == 5:  # default in parser
            args.beam_size = export_cfg.beam_size
        if args.max_len == 100:  # default in parser
            args.max_len = export_cfg.max_len
        if args.batch_size == 32:  # default in parser
            args.batch_size = export_cfg.batch_size
        if args.device == "auto":
            args.device = train_cfg.device

    if args.src_file is None or args.ref_file is None:
        parser.error("src_file and ref_file are required (or valid config file)")

    print(f"Using device: {args.device}")

    # Load config and model
    translator = Translator(
        model_path=args.model, device=args.device, compute_type=args.compute_type
    )

    # Load data
    with open(args.src_file, "r", encoding="utf-8") as f:
        src_lines = [l.strip() for l in f.readlines()]
    with open(args.ref_file, "r", encoding="utf-8") as f:
        ref_lines = [l.strip() for l in f.readlines()][: len(src_lines)]

    if len(src_lines) != len(ref_lines):
        print(
            f"Warning: Source ({len(src_lines)}) and Reference ({len(ref_lines)}) line counts differ."
        )

    print(f"Translating {len(src_lines)} lines...")
    hypotheses = translator(
        src_lines, beam_size=args.beam_size, max_batch_size=args.batch_size
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
