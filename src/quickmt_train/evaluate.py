import fire
from time import time
from quickmt import Translator
from .config import load_config
import sacrebleu


def main():
    import fire
    fire.Fire(evaluate_cli)


def evaluate_cli(
    model: str,
    config: str | None = None,
    src_file: str | None = None,
    ref_file: str | None = None,
    beam_size: int = 5,
    max_len: int = 100,
    batch_size: int = 32,
    device: str = "auto",
    compute_type: str = "auto",
):
    """
    Evaluate a CTranslate2 model.

    Args:
        model: Path to ctranslate2 model folder
        config: Path to config file
        src_file: Path to source text file
        ref_file: Path to reference text file
        beam_size: Beam size for decoding
        max_len: Max sequence length
        batch_size: Batch size for translation
        device: Auto detect GPU and use if available (or cuda, cpu)
        compute_type: CTranslate2 compute type
    """
    # Load defaults from config if available
    if config:
        model_cfg, data_cfg, train_cfg, export_cfg = load_config(config)

        if src_file is None:
            src_file = data_cfg.src_dev_path
        if ref_file is None:
            ref_file = data_cfg.tgt_dev_path

        # Pull defaults from export_cfg if not specified on CLI
        # (Assuming fire defaults are already set, we only override if user didn't provide)
        if beam_size == 5:  # default in function signature
            beam_size = export_cfg.beam_size
        if max_len == 100:  # default in parser
            max_len = export_cfg.max_len
        if batch_size == 32:  # default in parser
            batch_size = export_cfg.batch_size
        if device == "auto":
            device = train_cfg.device

    if src_file is None or ref_file is None:
        raise ValueError("src_file and ref_file are required (or valid config file)")

    print(f"Using device: {device}")

    # Load config and model
    translator = Translator(
        model_path=model, device=device, compute_type=compute_type
    )

    # Load data
    with open(src_file, "r", encoding="utf-8") as f:
        src_lines = [l.strip() for l in f.readlines()]
    with open(ref_file, "r", encoding="utf-8") as f:
        ref_lines = [l.strip() for l in f.readlines()][: len(src_lines)]

    if len(src_lines) != len(ref_lines):
        print(
            f"Warning: Source ({len(src_lines)}) and Reference ({len(ref_lines)}) line counts differ."
        )

    print(f"Translating {len(src_lines)} lines...")
    t1 = time()
    hypotheses = translator(
        src_lines, beam_size=beam_size, max_batch_size=batch_size
    )
    t2 = time()
    print(f"Translation time: {(t2 - t1):.2f} seconds")

    # Metrics
    bleu = sacrebleu.corpus_bleu(hypotheses, [ref_lines])
    chrf = sacrebleu.corpus_chrf(hypotheses, [ref_lines])

    print("\n" + "=" * 30)
    print(f"Results for {model}:")
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
    fire.Fire(main)
