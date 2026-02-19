import numpy as np
import ctranslate2
import os
import argparse
import torch
from safetensors.torch import load_file
from config import load_config
from collections import OrderedDict
import shutil
from pathlib import Path


def get_layer_weights(state_dict, prefix):
    """Extract weights and biases for a layer with a given prefix."""
    weights = state_dict.get(f"{prefix}.weight")
    bias = state_dict.get(f"{prefix}.bias")

    # Handle quantized linear layers
    if weights is None and f"{prefix}._packed_params._packed_params" in state_dict:
        packed_params = state_dict.get(f"{prefix}._packed_params._packed_params")
        if isinstance(packed_params, tuple) and len(packed_params) >= 2:
            qweight, bias = packed_params
            if hasattr(qweight, "dequantize"):
                weights = qweight.dequantize()
            else:
                weights = qweight

    if weights is not None:
        if hasattr(weights, "detach"):
            weights = weights.detach().float().cpu().numpy()
        elif hasattr(weights, "numpy"):
            weights = weights.numpy()
        else:
            weights = np.array(weights)
    if bias is not None:
        if hasattr(bias, "detach"):
            bias = bias.detach().float().cpu().numpy()
        elif hasattr(bias, "numpy"):
            bias = bias.numpy()
        else:
            bias = np.array(bias)
    return weights, bias


def set_linear(spec, state_dict, prefix):
    """Set weights and bias for a CT2 LinearSpec."""
    spec.weight, spec.bias = get_layer_weights(state_dict, prefix)


def set_layer_norm(spec, state_dict, prefix):
    """Set gamma and beta for a CT2 LayerNormSpec."""
    weight = state_dict.get(f"{prefix}.weight")
    bias = state_dict.get(f"{prefix}.bias")

    if weight is None:
        # Fallback for quantized LayerNorm which might use 'scale' instead of 'weight'
        weight = state_dict.get(f"{prefix}.scale")

    if weight is not None:
        if hasattr(weight, "detach"):
            spec.gamma = weight.detach().float().cpu().numpy()
        else:
            spec.gamma = weight.numpy()
    if bias is not None:
        if hasattr(bias, "detach"):
            spec.beta = bias.detach().float().cpu().numpy()
        else:
            spec.beta = bias.numpy()


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
    in_proj_weight = state_dict.get(f"{prefix}.in_proj_weight")
    in_proj_bias = state_dict.get(f"{prefix}.in_proj_bias")
    out_proj_weight = state_dict.get(f"{prefix}.out_proj.weight")
    out_proj_bias = state_dict.get(f"{prefix}.out_proj.bias")

    if in_proj_weight is not None:
        in_proj_weight = (
            in_proj_weight.detach().float().cpu().numpy()
            if hasattr(in_proj_weight, "detach")
            else in_proj_weight.numpy()
        )
    if in_proj_bias is not None:
        in_proj_bias = (
            in_proj_bias.detach().float().cpu().numpy()
            if hasattr(in_proj_bias, "detach")
            else in_proj_bias.numpy()
        )
    if out_proj_weight is not None:
        out_proj_weight = (
            out_proj_weight.detach().float().cpu().numpy()
            if hasattr(out_proj_weight, "detach")
            else out_proj_weight.numpy()
        )
    if out_proj_bias is not None:
        out_proj_bias = (
            out_proj_bias.detach().float().cpu().numpy()
            if hasattr(out_proj_bias, "detach")
            else out_proj_bias.numpy()
        )

    if self_attention:
        # linear[0] is in_proj
        spec.linear[0].weight = in_proj_weight
        spec.linear[0].bias = in_proj_bias
        # linear[1] is out_proj
        spec.linear[1].weight = out_proj_weight
        spec.linear[1].bias = out_proj_bias
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

        spec.linear[2].weight = out_proj_weight
        spec.linear[2].bias = out_proj_bias


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
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    # 1. Load config and weights
    model_cfg, _, export_cfg = load_config(args.config)

    if export_cfg.model_path.endswith(".safetensors"):
        state_dict = load_file(export_cfg.model_path, device="cpu")
    else:
        state_dict = torch.load(export_cfg.model_path, map_location="cpu")
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
        num_layers=model_cfg.enc_layers,
        num_heads=model_cfg.n_heads,
        pre_norm=True,
        activation=ctranslate2.specs.Activation.GELU,
    )
    decoder_spec = ctranslate2.specs.TransformerDecoderSpec(
        num_layers=model_cfg.dec_layers,
        num_heads=model_cfg.n_heads,
        pre_norm=True,
        activation=ctranslate2.specs.Activation.GELU,
    )

    # 3. Map weights

    # Embeddings
    src_emb = state_dict.get("src_tok_emb.embedding.weight")
    if src_emb is not None:
        encoder_spec.embeddings[0].weight = (
            src_emb.detach().float().cpu().numpy()
            if hasattr(src_emb, "detach")
            else src_emb.numpy()
        )

    tgt_emb = state_dict.get("tgt_tok_emb.embedding.weight")
    if tgt_emb is not None:
        decoder_spec.embeddings.weight = (
            tgt_emb.detach().float().cpu().numpy()
            if hasattr(tgt_emb, "detach")
            else tgt_emb.numpy()
        )

    # Position Encodings
    pe_tensor = state_dict.get("positional_encoding.pe")
    if pe_tensor is not None:
        pe = (
            pe_tensor[0].detach().float().cpu().numpy()
            if hasattr(pe_tensor, "detach")
            else pe_tensor[0].numpy()
        )
        encoder_spec.position_encodings.encodings = pe
        decoder_spec.position_encodings.encodings = pe

    # Generator (Projection)
    set_linear(decoder_spec.projection, state_dict, "generator")

    # 4. Encoder Layers
    for i in range(model_cfg.enc_layers):
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
    for i in range(model_cfg.dec_layers):
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
    if not os.path.exists(export_cfg.output_dir):
        os.makedirs(export_cfg.output_dir)

    spec = ctranslate2.specs.TransformerSpec(encoder_spec, decoder_spec)
    print(spec.config.__dict__)
    spec.config.add_source_bos = True # type: ignore
    spec.config.add_source_eos = False # type: ignore
    #'add_source_bos': False, 'add_source_eos': False,
    # ..decoder_start_token = "<s>"

    # Register vocabularies
    spec.register_source_vocabulary(convert_vocab(export_cfg.src_vocab))
    spec.register_target_vocabulary(convert_vocab(export_cfg.tgt_vocab))

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

    spec.optimize(quantization=export_cfg.quantization)
    spec.save(export_cfg.output_dir)
    print(f"Model saved to {export_cfg.output_dir}")

    # Copy Tokenizers to output directory
    shutil.copy("tokenizer_src.model", Path(export_cfg.output_dir) / "src.spm.model")
    shutil.copy("tokenizer_tgt.model", Path(export_cfg.output_dir) / "tgt.spm.model")


if __name__ == "__main__":
    main()
