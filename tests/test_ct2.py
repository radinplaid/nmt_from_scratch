import ctranslate2
import sentencepiece as spm


def main():
    model_path = "ct2_model"
    sp_src_path = "tokenizer_src.model"
    sp_tgt_path = "tokenizer_tgt.model"

    # Load tokenizer
    sp_src = spm.SentencePieceProcessor()
    sp_src.load(sp_src_path)

    sp_tgt = spm.SentencePieceProcessor()
    sp_tgt.load(sp_tgt_path)

    # Load translator
    translator = ctranslate2.Translator(model_path, device="cpu")

    # Test sentence (Persian)
    text = "دکتر ایهود اور، استاد پزشکی دانشگاه دالهاوزی در هلیفکس، نوااسکوشیا و رئیس بخش کلینیکی و علمی انجمن دیابت کانادا هشدار داد که این تحقیق هنوز در روزهای آغازین خود می‌باشد."
    print(f"Input: {text}")

    # Tokenize
    tokens = sp_src.encode_as_pieces(text) + ["</s>"]
    print(f"Tokens: {tokens}")

    # Translate
    results = translator.translate_batch(
        [tokens],
        beam_size=5,
        max_decoding_length=256,
        num_hypotheses=1,
        length_penalty=1.05,
        repetition_penalty=1.05,
    )
    output_tokens = results[0].hypotheses[0]
    print(f"Output tokens: {output_tokens}")

    # Decode
    output_text = sp_tgt.decode_pieces(output_tokens)
    print(f"Output: {output_text}")


if __name__ == "__main__":
    main()
