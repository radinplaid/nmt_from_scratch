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
    weight, bias = get_layer_weights(state_dict, prefix)
    spec.weight = weight
    if bias is not None:
        spec.bias = bias
    elif weight is not None:
        spec.bias = np.zeros(weight.shape[0], dtype=np.float32)


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
    elif weight is not None:
        # Fill with zeros if bias is missing
        if hasattr(weight, "detach"):
            w = weight.detach().float().cpu().numpy()
        else:
            w = weight.numpy()
        spec.beta = np.zeros(w.shape[0], dtype=np.float32)


def _make_sinusoidal_position_encodings(max_len, d_model):
    """Generate sinusoidal position encodings as a numpy array."""
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


def set_multihead_attention(
    spec, state_dict, prefix, self_attention=True, n_kv_heads=None, n_heads=None
):
    """Map a ``GroupedQueryAttention`` module's weights onto a CT2 attention spec.

    Weight keys expected in ``state_dict`` (relative to ``prefix``):
        ``q_proj.weight``  / ``q_proj.bias``
        ``kv_proj.weight`` / ``kv_proj.bias``   (fused K and V)
        ``out_proj.weight`` / ``out_proj.bias``

    CTranslate2 self-attention layout when ``n_kv_heads == n_heads`` (standard MHA):
        ``linear[0]`` = fused QKV  ``(3 * d, d)``
        ``linear[1]`` = out_proj   ``(d, d)``

    CTranslate2 self-attention layout when ``n_kv_heads < n_heads`` (GQA / MQA),
    and *all* cross-attention regardless of group count:
        ``linear[0]`` = q_proj    ``(n_heads    * head_dim, d)``
        ``linear[1]`` = kv_proj   ``(2 * n_kv_heads * head_dim, d)``  – fused K+V
        ``linear[2]`` = out_proj  ``(d, n_heads * head_dim)``
    """

    def to_np(t):
        if t is None:
            return None
        return t.detach().float().cpu().numpy() if hasattr(t, "detach") else np.array(t)

    q_w = to_np(state_dict.get(f"{prefix}.q_proj.weight"))
    kv_w = to_np(state_dict.get(f"{prefix}.kv_proj.weight"))
    o_w = to_np(state_dict.get(f"{prefix}.out_proj.weight"))
    q_b = to_np(state_dict.get(f"{prefix}.q_proj.bias"))
    kv_b = to_np(state_dict.get(f"{prefix}.kv_proj.bias"))
    o_b = to_np(state_dict.get(f"{prefix}.out_proj.bias"))

    def zeros(arr):
        return np.zeros(arr.shape[0], dtype=np.float32)

    if self_attention:
        # ---------- Self-attention: always 2 CT2 linears ----------
        # CT2 self-attention layout is the same whether MHA or GQA:
        #   linear[0] = fused [Q ; K+V]
        #     MHA shape: (3 * n_heads * head_dim, d)
        #     GQA shape: ((n_heads + 2 * n_kv_heads) * head_dim, d)
        #   linear[1] = out_proj
        #
        # q_w:  (n_heads * hd, d)
        # kv_w: (2 * n_kv_heads * hd, d)
        qkv_w = np.concatenate([q_w, kv_w], axis=0)
        spec.linear[0].weight = qkv_w
        if q_b is not None or kv_b is not None:
            spec.linear[0].bias = np.concatenate([
                q_b  if q_b  is not None else np.zeros(q_w.shape[0],  dtype=np.float32),
                kv_b if kv_b is not None else np.zeros(kv_w.shape[0], dtype=np.float32),
            ])
        else:
            spec.linear[0].bias = np.zeros(qkv_w.shape[0], dtype=np.float32)

        spec.linear[1].weight = o_w
        spec.linear[1].bias = o_b if o_b is not None else zeros(o_w)

    else:
        # ---------- Cross-attention: always 3 CT2 linears ----------
        #   linear[0] = q_proj   (n_heads * head_dim, d)
        #   linear[1] = kv_proj  (2 * n_kv_heads * head_dim, d)  – fused K+V
        #   linear[2] = out_proj
        spec.linear[0].weight = q_w
        spec.linear[0].bias = q_b if q_b is not None else zeros(q_w)

        spec.linear[1].weight = kv_w
        spec.linear[1].bias = kv_b if kv_b is not None else zeros(kv_w)

        spec.linear[2].weight = o_w
        spec.linear[2].bias = o_b if o_b is not None else zeros(o_w)


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
    model_cfg, data_cfg, train_cfg, export_cfg = load_config(args.config)

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

    activation_map = {
        "gelu": ctranslate2.specs.Activation.GELU,
        "relu": ctranslate2.specs.Activation.RELU,
        "swish": ctranslate2.specs.Activation.SWISH,
        "silu": ctranslate2.specs.Activation.SWISH,
    }
    ct2_activation = activation_map.get(
        model_cfg.activation, ctranslate2.specs.Activation.GELU
    )

    is_gated = getattr(model_cfg, "mlp_type", "standard") == "gated"

    # Resolve effective n_kv_heads (0 means same as n_heads → standard MHA)
    n_kv_heads_cfg = getattr(model_cfg, "n_kv_heads", 0)
    effective_kv_heads = n_kv_heads_cfg if n_kv_heads_cfg > 0 else model_cfg.n_heads
    is_gqa = effective_kv_heads < model_cfg.n_heads

    enc_kwargs: dict = dict(
        num_layers=model_cfg.enc_layers,
        num_heads=model_cfg.n_heads,
        pre_norm=True,
        activation=ct2_activation,
        ffn_glu=is_gated,
    )
    dec_kwargs: dict = dict(
        num_layers=model_cfg.dec_layers,
        num_heads=model_cfg.n_heads,
        pre_norm=True,
        activation=ct2_activation,
        ffn_glu=is_gated,
    )
    # num_heads_kv activates GQA/MQA in CTranslate2; only pass when actually using GQA
    if is_gqa:
        enc_kwargs["num_heads_kv"] = effective_kv_heads
        dec_kwargs["num_heads_kv"] = effective_kv_heads

    encoder_spec = ctranslate2.specs.TransformerEncoderSpec(**enc_kwargs)
    decoder_spec = ctranslate2.specs.TransformerDecoderSpec(**dec_kwargs)

    # ... mapping ...
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
            n_kv_heads=effective_kv_heads,
            n_heads=model_cfg.n_heads,
        )
        set_layer_norm(
            layer_spec.self_attention.layer_norm, state_dict, f"{prefix}.norm1"
        )

        if is_gated:
            # gate_up_proj is fused [gate, up]
            weight, bias = get_layer_weights(state_dict, f"{prefix}.ffn.gate_up_proj")
            gate_w, up_w = np.split(weight, 2, axis=0)
            layer_spec.ffn.linear_0.weight = gate_w
            layer_spec.ffn.linear_0_noact.weight = up_w

            if bias is not None:
                gate_b, up_b = np.split(bias, 2)
                layer_spec.ffn.linear_0.bias = gate_b
                layer_spec.ffn.linear_0_noact.bias = up_b
            else:
                layer_spec.ffn.linear_0.bias = np.zeros(
                    gate_w.shape[0], dtype=np.float32
                )
                layer_spec.ffn.linear_0_noact.bias = np.zeros(
                    up_w.shape[0], dtype=np.float32
                )

            set_linear(layer_spec.ffn.linear_1, state_dict, f"{prefix}.ffn.down_proj")
        else:
            set_linear(layer_spec.ffn.linear_0, state_dict, f"{prefix}.ffn.linear1")
            set_linear(layer_spec.ffn.linear_1, state_dict, f"{prefix}.ffn.linear2")
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
            n_kv_heads=effective_kv_heads,
            n_heads=model_cfg.n_heads,
        )
        set_layer_norm(
            layer_spec.self_attention.layer_norm, state_dict, f"{prefix}.norm1"
        )

        set_multihead_attention(
            layer_spec.attention,
            state_dict,
            f"{prefix}.multihead_attn",
            self_attention=False,
            n_kv_heads=effective_kv_heads,
            n_heads=model_cfg.n_heads,
        )
        set_layer_norm(layer_spec.attention.layer_norm, state_dict, f"{prefix}.norm2")

        if is_gated:
            # gate_up_proj is fused [gate, up]
            weight, bias = get_layer_weights(state_dict, f"{prefix}.ffn.gate_up_proj")
            gate_w, up_w = np.split(weight, 2, axis=0)
            layer_spec.ffn.linear_0.weight = gate_w
            layer_spec.ffn.linear_0_noact.weight = up_w

            if bias is not None:
                gate_b, up_b = np.split(bias, 2)
                layer_spec.ffn.linear_0.bias = gate_b
                layer_spec.ffn.linear_0_noact.bias = up_b
            else:
                layer_spec.ffn.linear_0.bias = np.zeros(
                    gate_w.shape[0], dtype=np.float32
                )
                layer_spec.ffn.linear_0_noact.bias = np.zeros(
                    up_w.shape[0], dtype=np.float32
                )

            set_linear(layer_spec.ffn.linear_1, state_dict, f"{prefix}.ffn.down_proj")
        else:
            set_linear(layer_spec.ffn.linear_0, state_dict, f"{prefix}.ffn.linear1")
            set_linear(layer_spec.ffn.linear_1, state_dict, f"{prefix}.ffn.linear2")
        set_layer_norm(layer_spec.ffn.layer_norm, state_dict, f"{prefix}.norm3")

    # Final Decoder Norm
    set_layer_norm(decoder_spec.layer_norm, state_dict, "decoder.norm")

    # 6. Save model
    if not os.path.exists(export_cfg.output_dir):
        os.makedirs(export_cfg.output_dir)

    spec = ctranslate2.specs.TransformerSpec(encoder_spec, decoder_spec)
    spec.config.add_source_bos = export_cfg.add_source_bos  # type: ignore
    spec.config.add_source_eos = export_cfg.add_source_eos  # type: ignore

    # Register vocabularies
    spec.register_source_vocabulary(
        convert_vocab(f"{data_cfg.tokenizer_prefix_src}.vocab")
    )
    spec.register_target_vocabulary(
        convert_vocab(f"{data_cfg.tokenizer_prefix_tgt}.vocab")
    )

    spec.validate()
    spec.optimize(quantization=export_cfg.quantization)
    spec.save(export_cfg.output_dir)
    print(f"Model saved to {export_cfg.output_dir}")

    # Copy Tokenizers to output directory
    shutil.copy(
        f"{data_cfg.tokenizer_prefix_src}.model",
        Path(export_cfg.output_dir) / "src.spm.model",
    )
    shutil.copy(
        f"{data_cfg.tokenizer_prefix_tgt}.model",
        Path(export_cfg.output_dir) / "tgt.spm.model",
    )


if __name__ == "__main__":
    main()
