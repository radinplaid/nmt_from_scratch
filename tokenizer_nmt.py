import os
from typing import List, Optional
import sentencepiece as spm
from transformers import PreTrainedTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "src_vocab_file": "src.spm.model",
    "tgt_vocab_file": "tgt.spm.model",
}


class NMTTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        src_vocab_file,
        tgt_vocab_file,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        sp_model_kwargs=None,
        **kwargs,
    ):
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        self.src_vocab_file = src_vocab_file
        self.tgt_vocab_file = tgt_vocab_file

        self.src_spm = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.src_spm.Load(src_vocab_file)

        self.tgt_spm = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.tgt_spm.Load(tgt_vocab_file)

        super().__init__(
            pad_token=pad_token,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            src_vocab_file=src_vocab_file,
            tgt_vocab_file=tgt_vocab_file,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

    def get_vocab(self):
        vocab = {
            self.src_spm.id_to_piece(i): i for i in range(self.src_spm.get_piece_size())
        }
        return vocab

    def __getstate__(self):
        state = self.__dict__.copy()
        state["src_spm"] = None
        state["tgt_spm"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        self.src_spm = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.src_spm.Load(self.src_vocab_file)
        self.tgt_spm = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.tgt_spm.Load(self.tgt_vocab_file)

    @property
    def vocab_size(self):
        return self.src_spm.get_piece_size()

    def _tokenize(self, text, is_target=False):
        if is_target:
            return self.tgt_spm.encode_as_pieces(text)
        return self.src_spm.encode_as_pieces(text)

    def _convert_token_to_id(self, token, is_target=False):
        if is_target:
            return self.tgt_spm.piece_to_id(token)
        return self.src_spm.piece_to_id(token)

    def _convert_id_to_token(self, index, is_target=False):
        if is_target:
            return self.tgt_spm.id_to_piece(index)
        return self.src_spm.id_to_piece(index)

    def convert_tokens_to_string(self, tokens, is_target=False):
        if is_target:
            return self.tgt_spm.decode_pieces(tokens)
        return self.src_spm.decode_pieces(tokens)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        src_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "")
            + VOCAB_FILES_NAMES["src_vocab_file"],
        )
        tgt_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "")
            + VOCAB_FILES_NAMES["tgt_vocab_file"],
        )

        with open(src_vocab_file, "wb") as f_src:
            with open(self.src_vocab_file, "rb") as spm_file:
                f_src.write(spm_file.read())

        with open(tgt_vocab_file, "wb") as f_tgt:
            with open(self.tgt_vocab_file, "rb") as spm_file:
                f_tgt.write(spm_file.read())

        return (src_vocab_file, tgt_vocab_file)

    # Override standard methods since they are not target-aware by default in huggingface natively
    # Best effort to allow text_target mapping
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if self.bos_token_id is not None:
            token_ids_0 = [self.bos_token_id] + token_ids_0
        if self.eos_token_id is not None:
            token_ids_0 = token_ids_0 + [self.eos_token_id]
        return token_ids_0

    def encode(self, text, text_target=None, **kwargs):
        # basic monkey patch for encoding text natively
        if text_target is not None:
            return super().encode(
                text_target, **kwargs
            )  # Wait, it just drops src if text_target used this way
        return super().encode(text, **kwargs)

    def __call__(self, text=None, text_target=None, **kwargs):
        if text_target is not None:
            # Tokenize targets specifically
            target_out = super().__call__(text_target, **kwargs)
            # if we also have text, we should tokenize that as inputs
            if text is not None:
                src_out = super().__call__(text, **kwargs)
                src_out["labels"] = target_out["input_ids"]
                return src_out
            return target_out

        return super().__call__(text, **kwargs)

    def decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=True,
        **kwargs,
    ):
        # We always assume decode is for the target language!
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        elif hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()

        tokens = []
        for id in token_ids:
            if skip_special_tokens and id in self.all_special_ids:
                continue
            tokens.append(self._convert_id_to_token(id, is_target=True))

        text = self.convert_tokens_to_string(tokens, is_target=True)
        if clean_up_tokenization_spaces:
            text = (
                text.replace(" .", ".")
                .replace(" ,", ",")
                .replace(" ?", "?")
                .replace(" !", "!")
            )
        return text


NMTTokenizer.register_for_auto_class()
