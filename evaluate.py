import torch
import argparse
from model import Seq2SeqTransformer
from config import ModelConfig
import sentencepiece as spm
import sacrebleu
from tqdm import tqdm
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--src_file", type=str, required=True, help="Path to source text file"
    )
    parser.add_argument(
        "--ref_file", type=str, required=True, help="Path to reference text file"
    )
    parser.add_argument(
        "--src_sp",
        type=str,
        default="tokenizer_src.model",
        help="Source SentencePiece model",
    )
    parser.add_argument(
        "--tgt_sp",
        type=str,
        default="tokenizer_tgt.model",
        help="Target SentencePiece model",
    )
    parser.add_argument(
        "--beam_size", type=int, default=5, help="Beam size for decoding"
    )
    parser.add_argument("--max_len", type=int, default=100, help="Max sequence length")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for translation"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Auto detect GPU and use if available (or cuda, cpu)"
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load config and model
    config = ModelConfig()
    model = Seq2SeqTransformer(config).to(device)

    print(f"Loading checkpoint: {args.checkpoint}")
    if args.checkpoint.endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(args.checkpoint, device=str(device))
    else:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint["model_state_dict"]

    # Handle compiled model prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("_orig_mod.", "")
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    model = torch.compile(model)

    # Load tokenizers
    src_sp = spm.SentencePieceProcessor(model_file=args.src_sp)
    tgt_sp = spm.SentencePieceProcessor(model_file=args.tgt_sp)

    # Load data
    with open(args.src_file, "r", encoding="utf-8") as f:
        src_lines = [l.strip() for l in f.readlines()]
    with open(args.ref_file, "r", encoding="utf-8") as f:
        ref_lines = [l.strip() for l in f.readlines()][: len(src_lines)]

    if len(src_lines) != len(ref_lines):
        print(
            f"Warning: Source ({len(src_lines)}) and Reference ({len(ref_lines)}) line counts differ."
        )

    hypotheses = []

    print(f"Translating {len(src_lines)} lines...")

    # Batch processing for efficiency
    for i in tqdm(range(0, len(src_lines), args.batch_size)):
        batch_lines = src_lines[i : i + args.batch_size]

        # Tokenize
        batch_ids = [
            torch.tensor(src_sp.encode(l, out_type=int, add_bos=True, add_eos=True))
            for l in batch_lines
        ]

        # Pad with pad_sequence
        src_tensor = torch.nn.utils.rnn.pad_sequence(
            batch_ids, batch_first=True, padding_value=0
        ).to(device)

        with torch.no_grad():
            if args.beam_size > 1:
                generated_ids = model.beam_search(
                    src_tensor, max_len=args.max_len, beam_size=args.beam_size
                )
            else:
                generated_ids = model.generate(src_tensor, max_len=args.max_len)

        for g_ids in generated_ids:
            ids_list = g_ids.tolist()
            # Truncate at EOS (id 3)
            if 3 in ids_list:
                ids_list = ids_list[: ids_list.index(3)]

            decoded = tgt_sp.decode(ids_list)
            hypotheses.append(decoded)

    # Metrics
    bleu = sacrebleu.corpus_bleu(hypotheses, [ref_lines])
    chrf = sacrebleu.corpus_chrf(hypotheses, [ref_lines])

    print("\n" + "=" * 30)
    print(f"Results for {args.checkpoint}:")
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
