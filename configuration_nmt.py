import os
import torch

for attr in [
    "int1",
    "int2",
    "int3",
    "int4",
    "int5",
    "int6",
    "int7",
    "uint1",
    "uint2",
    "uint3",
    "uint4",
    "uint5",
    "uint6",
    "uint7",
    "float8_e4m3fn",
    "float8_e5m2",
]:
    if not hasattr(torch, attr):
        setattr(torch, attr, torch.int8)
from transformers import PretrainedConfig


class NMTConfig(PretrainedConfig):
    model_type = "nmt"

    def __init__(
        self,
        d_model: int = 256,
        enc_layers: int = 4,
        dec_layers: int = 2,
        n_heads: int = 8,
        ffn_dim: int = 1024,
        max_len: int = 512,
        dropout: float = 0.1,
        vocab_size_src: int = 10000,
        vocab_size_tgt: int = 10000,
        vocab_size: int = 10000,  # Hugging Face standard base vocab size, maps to tgt typicaly
        activation: str = "gelu",
        ff_bias: bool = False,
        mlp_type: str = "standard",
        layernorm_eps: float = 1e-5,
        norm_type: str = "layernorm",
        tie_decoder_embeddings: bool = False,
        pad_token_id: int = 0,
        unk_token_id: int = 1,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        decoder_start_token_id: int = 2,
        is_encoder_decoder: bool = True,
        **kwargs,
    ):
        self.d_model = d_model
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.n_heads = n_heads
        self.ffn_dim = ffn_dim
        self.max_len = max_len
        self.dropout = dropout
        self.vocab_size_src = vocab_size_src
        self.vocab_size_tgt = vocab_size_tgt
        self.activation = activation
        self.ff_bias = ff_bias
        self.mlp_type = mlp_type
        self.layernorm_eps = layernorm_eps
        self.norm_type = norm_type
        self.tie_decoder_embeddings = tie_decoder_embeddings
        self.num_hidden_layers = dec_layers

        # Compatibility for AutoModel loading
        self.vocab_size = vocab_size_tgt

        super().__init__(
            pad_token_id=pad_token_id,
            unk_token_id=unk_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )
