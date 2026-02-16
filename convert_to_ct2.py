import numpy as np
import ctranslate2
import os
import argparse
import torch
from safetensors.torch import load_file
from config import ModelConfig
from collections import OrderedDict
import shutil
from pathlib import Path

from config import ModelConfig, TrainConfig

model_config = ModelConfig()
train_config = TrainConfig()


def get_layer_weights(state_dict, prefix):
    """Extract weights and biases for a layer with a given prefix."""
    weights = state_dict.get(f"{prefix}.weight")
    bias = state_dict.get(f"{prefix}.bias")
    if weights is not None:
        weights = weights.numpy()
    if bias is not None:
        bias = bias.numpy()
    return weights, bias


def set_linear(spec, state_dict, prefix):
    """Set weights and bias for a CT2 LinearSpec."""
    spec.weight, spec.bias = get_layer_weights(state_dict, prefix)


def set_layer_norm(spec, state_dict, prefix):
    """Set gamma and beta for a CT2 LayerNormSpec."""
    spec.gamma = state_dict.get(f"{prefix}.weight").numpy()
    spec.beta = state_dict.get(f"{prefix}.bias").numpy()


def _make_sinusoidal_position_encodings(max_len, d_model):
    """Generate sinusoidal position encodings as a numpy array."""
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


def set_multihead_attention(spec, state_dict, prefix, self_attention=True):
    """Set weights for a CT2 MultiHeadAttentionSpec from PyTorch MultiheadAttention."""
    in_proj_weight = state_dict.get(f"{prefix}.in_proj_weight").numpy()
    in_proj_bias = state_dict.get(f"{prefix}.in_proj_bias").numpy()

    if self_attention:
        # linear[0] is in_proj
        spec.linear[0].weight = in_proj_weight
        spec.linear[0].bias = in_proj_bias
        # linear[1] is out_proj
        spec.linear[1].weight = state_dict.get(f"{prefix}.out_proj.weight").numpy()
        spec.linear[1].bias = state_dict.get(f"{prefix}.out_proj.bias").numpy()
    else:
        # linear[0] is query_proj
        # linear[1] is kv_proj (fused)
        # linear[2] is out_proj
        q, k, v = np.split(in_proj_weight, 3)
        qb, kb, vb = np.split(in_proj_bias, 3)

        spec.linear[0].weight = q
        spec.linear[0].bias = qb

        spec.linear[1].weight = np.concatenate([k, v], axis=0)
        spec.linear[1].bias = np.concatenate([kb, vb], axis=0)

        spec.linear[2].weight = state_dict.get(f"{prefix}.out_proj.weight").numpy()
        spec.linear[2].bias = state_dict.get(f"{prefix}.out_proj.bias").numpy()


def convert_vocab(sp_vocab_path):
    """Load SentencePiece vocab file and return tokens list."""
    tokens = []
    with open(sp_vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                tokens.append(parts[0])
    return tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model_avg.safetensors")
    parser.add_argument("--output_dir", type=str, default="ct2_model")
    parser.add_argument("--src_vocab", type=str, default="tokenizer_src.vocab")
    parser.add_argument("--tgt_vocab", type=str, default="tokenizer_tgt.vocab")
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=[
            "int8",
            "int8_float32",
            "int8_float16",
            "int8_bfloat16",
            "int16",
            "float16",
            "bfloat16",
            "float32",
        ],
    )
    args = parser.parse_args()

    # 1. Load config and weights
    cfg = ModelConfig()
    if args.model_path.endswith(".safetensors"):
        state_dict = load_file(args.model_path, device="cpu")
    else:
        state_dict = torch.load(args.model_path, map_location="cpu")
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]

    # Strip _orig_mod. prefix if present (from torch.compile)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        new_state_dict[new_key] = v
    state_dict = new_state_dict

    # 2. Initialize Specs
    encoder_spec = ctranslate2.specs.TransformerEncoderSpec(
        num_layers=cfg.enc_layers,
        num_heads=cfg.n_heads,
        pre_norm=True,
        activation=ctranslate2.specs.Activation.GELU,
    )
    decoder_spec = ctranslate2.specs.TransformerDecoderSpec(
        num_layers=cfg.dec_layers,
        num_heads=cfg.n_heads,
        pre_norm=True,
        activation=ctranslate2.specs.Activation.GELU,
    )

    # 3. Map weights

    # Embeddings
    encoder_spec.embeddings[0].weight = state_dict.get(
        "src_tok_emb.embedding.weight"
    ).numpy()
    decoder_spec.embeddings.weight = state_dict.get(
        "tgt_tok_emb.embedding.weight"
    ).numpy()

    # Position Encodings
    pe = state_dict.get("positional_encoding.pe")[0].numpy()
    encoder_spec.position_encodings.encodings = pe
    decoder_spec.position_encodings.encodings = pe

    # Generator (Projection)
    set_linear(decoder_spec.projection, state_dict, "generator")

    # 4. Encoder Layers
    for i in range(cfg.enc_layers):
        prefix = f"encoder.layers.{i}"
        layer_spec = encoder_spec.layer[i]

        set_multihead_attention(
            layer_spec.self_attention,
            state_dict,
            f"{prefix}.self_attn",
            self_attention=True,
        )
        set_layer_norm(
            layer_spec.self_attention.layer_norm, state_dict, f"{prefix}.norm1"
        )

        set_linear(layer_spec.ffn.linear_0, state_dict, f"{prefix}.linear1")
        set_linear(layer_spec.ffn.linear_1, state_dict, f"{prefix}.linear2")
        set_layer_norm(layer_spec.ffn.layer_norm, state_dict, f"{prefix}.norm2")

    # Final Encoder Norm
    set_layer_norm(encoder_spec.layer_norm, state_dict, "encoder.norm")

    # 5. Decoder Layers
    for i in range(cfg.dec_layers):
        prefix = f"decoder.layers.{i}"
        layer_spec = decoder_spec.layer[i]

        set_multihead_attention(
            layer_spec.self_attention,
            state_dict,
            f"{prefix}.self_attn",
            self_attention=True,
        )
        set_layer_norm(
            layer_spec.self_attention.layer_norm, state_dict, f"{prefix}.norm1"
        )

        set_multihead_attention(
            layer_spec.attention,
            state_dict,
            f"{prefix}.multihead_attn",
            self_attention=False,
        )
        set_layer_norm(layer_spec.attention.layer_norm, state_dict, f"{prefix}.norm2")

        set_linear(layer_spec.ffn.linear_0, state_dict, f"{prefix}.linear1")
        set_linear(layer_spec.ffn.linear_1, state_dict, f"{prefix}.linear2")
        set_layer_norm(layer_spec.ffn.layer_norm, state_dict, f"{prefix}.norm3")

    # Final Decoder Norm
    set_layer_norm(decoder_spec.layer_norm, state_dict, "decoder.norm")

    # 6. Save model
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    spec = ctranslate2.specs.TransformerSpec(encoder_spec, decoder_spec)
    print(spec.config.__dict__)
    spec.config.add_source_bos = True
    spec.config.add_source_eos = False
    #'add_source_bos': False, 'add_source_eos': False,
    # ..decoder_start_token = "<s>"

    # Register vocabularies
    spec.register_source_vocabulary(convert_vocab(args.src_vocab))
    spec.register_target_vocabulary(convert_vocab(args.tgt_vocab))

    # Debug: Check variable types
    for name, value in spec.variables().items():
        if not isinstance(value, np.ndarray):
            print(f"Variable {name} is NOT a numpy array: {type(value)}")
        elif value.dtype == np.object_:
            print(f"Variable {name} has object dtype!")

    try:
        spec.validate()
        print("Model validation successful.")
    except Exception as e:
        print(f"Model validation failed: {e}")

    spec.optimize(quantization=args.quantization)
    spec.save(args.output_dir)
    print(f"Model saved to {args.output_dir}")

    # Copy Tokenizers to output directory
    shutil.copy("tokenizer_src.model", Path(args.output_dir) / "src.spm.model")
    shutil.copy("tokenizer_tgt.model", Path(args.output_dir) / "tgt.spm.model")


if __name__ == "__main__":
    main()
