import ctranslate2
import sentencepiece as spm

src_sp = spm.SentencePieceProcessor("faen-small/tokenizer_src.model")
tgt_sp = spm.SentencePieceProcessor("faen-small/tokenizer_tgt.model")

translator = ctranslate2.Translator("faen-small/exported_model", device="cuda")

text = "اعضای مجلس ولز نگران هستند که «همانند عروسک خیمه‌ شب بازی» دیده شوند"
# Note CTranslate2 automatically adds BOS/EOS if configured!
# Just provide the raw tokens.
tokens = src_sp.encode(text, out_type=str)
print("Input tokens:", tokens)

results = translator.translate_batch([tokens], beam_size=5, max_decoding_length=100)
out_tokens = results[0].hypotheses[0]
print("Wait, raw out tokens:", out_tokens)
out_text = tgt_sp.decode(out_tokens)
print("Translation:", out_text)
