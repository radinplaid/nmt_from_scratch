import os
import argparse
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
from safetensors.torch import load_file
import shutil

from config import load_config
from configuration_nmt import NMTConfig
from modeling_nmt import NMTForConditionalGeneration, NMTModel
from tokenizer_nmt import NMTTokenizer


def map_weights(hf_model, state_dict, config):
    hf_state = {}

    # 1. Embeddings
    hf_state["model.encoder.src_tok_emb.embedding.weight"] = state_dict[
        "src_tok_emb.embedding.weight"
    ]
    if "tgt_tok_emb.embedding.weight" in state_dict:
        hf_state["model.decoder.tgt_tok_emb.embedding.weight"] = state_dict[
            "tgt_tok_emb.embedding.weight"
        ]
    elif config.tie_decoder_embeddings:
        hf_state["model.decoder.tgt_tok_emb.embedding.weight"] = state_dict[
            "generator.weight"
        ]

    # 2. Positional Encodings
    hf_state["model.encoder.positional_encoding.pe"] = state_dict[
        "positional_encoding.pe"
    ]
    hf_state["model.decoder.positional_encoding.pe"] = state_dict[
        "positional_encoding.pe"
    ]

    # 3. Encoder Labels
    for i in range(config.enc_layers):
        p_src = f"encoder.layers.{i}."
        p_dst = f"model.encoder.layers.{i}."

        hf_state[f"{p_dst}self_attn.in_proj_weight"] = state_dict[
            f"{p_src}self_attn.in_proj_weight"
        ]
        if config.ff_bias:
            hf_state[f"{p_dst}self_attn.in_proj_bias"] = state_dict[
                f"{p_src}self_attn.in_proj_bias"
            ]
            hf_state[f"{p_dst}self_attn.out_proj.bias"] = state_dict[
                f"{p_src}self_attn.out_proj.bias"
            ]
        hf_state[f"{p_dst}self_attn.out_proj.weight"] = state_dict[
            f"{p_src}self_attn.out_proj.weight"
        ]

        hf_state[f"{p_dst}norm1.weight"] = state_dict[f"{p_src}norm1.weight"]
        if config.norm_type != "rmsnorm" and config.ff_bias:
            hf_state[f"{p_dst}norm1.bias"] = state_dict[f"{p_src}norm1.bias"]

        hf_state[f"{p_dst}norm2.weight"] = state_dict[f"{p_src}norm2.weight"]
        if config.norm_type != "rmsnorm" and config.ff_bias:
            hf_state[f"{p_dst}norm2.bias"] = state_dict[f"{p_src}norm2.bias"]

        if config.mlp_type == "gated":
            hf_state[f"{p_dst}ffn.gate_up_proj.weight"] = state_dict[
                f"{p_src}ffn.gate_up_proj.weight"
            ]
            hf_state[f"{p_dst}ffn.down_proj.weight"] = state_dict[
                f"{p_src}ffn.down_proj.weight"
            ]
            if config.ff_bias:
                hf_state[f"{p_dst}ffn.gate_up_proj.bias"] = state_dict[
                    f"{p_src}ffn.gate_up_proj.bias"
                ]
                hf_state[f"{p_dst}ffn.down_proj.bias"] = state_dict[
                    f"{p_src}ffn.down_proj.bias"
                ]
        else:
            hf_state[f"{p_dst}ffn.linear1.weight"] = state_dict[
                f"{p_src}ffn.linear1.weight"
            ]
            hf_state[f"{p_dst}ffn.linear2.weight"] = state_dict[
                f"{p_src}ffn.linear2.weight"
            ]
            if config.ff_bias:
                hf_state[f"{p_dst}ffn.linear1.bias"] = state_dict[
                    f"{p_src}ffn.linear1.bias"
                ]
                hf_state[f"{p_dst}ffn.linear2.bias"] = state_dict[
                    f"{p_src}ffn.linear2.bias"
                ]

    hf_state["model.encoder.norm.weight"] = state_dict["encoder.norm.weight"]
    if config.norm_type != "rmsnorm" and config.ff_bias:
        hf_state["model.encoder.norm.bias"] = state_dict["encoder.norm.bias"]

    # 4. Decoder Layers
    for i in range(config.dec_layers):
        p_src = f"decoder.layers.{i}."
        p_dst = f"model.decoder.layers.{i}."

        hf_state[f"{p_dst}self_attn.in_proj_weight"] = state_dict[
            f"{p_src}self_attn.in_proj_weight"
        ]
        if config.ff_bias:
            hf_state[f"{p_dst}self_attn.in_proj_bias"] = state_dict[
                f"{p_src}self_attn.in_proj_bias"
            ]
            hf_state[f"{p_dst}self_attn.out_proj.bias"] = state_dict[
                f"{p_src}self_attn.out_proj.bias"
            ]
        hf_state[f"{p_dst}self_attn.out_proj.weight"] = state_dict[
            f"{p_src}self_attn.out_proj.weight"
        ]

        hf_state[f"{p_dst}multihead_attn.in_proj_weight"] = state_dict[
            f"{p_src}multihead_attn.in_proj_weight"
        ]
        if config.ff_bias:
            hf_state[f"{p_dst}multihead_attn.in_proj_bias"] = state_dict[
                f"{p_src}multihead_attn.in_proj_bias"
            ]
            hf_state[f"{p_dst}multihead_attn.out_proj.bias"] = state_dict[
                f"{p_src}multihead_attn.out_proj.bias"
            ]
        hf_state[f"{p_dst}multihead_attn.out_proj.weight"] = state_dict[
            f"{p_src}multihead_attn.out_proj.weight"
        ]

        hf_state[f"{p_dst}norm1.weight"] = state_dict[f"{p_src}norm1.weight"]
        hf_state[f"{p_dst}norm2.weight"] = state_dict[f"{p_src}norm2.weight"]
        hf_state[f"{p_dst}norm3.weight"] = state_dict[f"{p_src}norm3.weight"]
        if config.norm_type != "rmsnorm" and config.ff_bias:
            hf_state[f"{p_dst}norm1.bias"] = state_dict[f"{p_src}norm1.bias"]
            hf_state[f"{p_dst}norm2.bias"] = state_dict[f"{p_src}norm2.bias"]
            hf_state[f"{p_dst}norm3.bias"] = state_dict[f"{p_src}norm3.bias"]

        if config.mlp_type == "gated":
            hf_state[f"{p_dst}ffn.gate_up_proj.weight"] = state_dict[
                f"{p_src}ffn.gate_up_proj.weight"
            ]
            hf_state[f"{p_dst}ffn.down_proj.weight"] = state_dict[
                f"{p_src}ffn.down_proj.weight"
            ]
            if config.ff_bias:
                hf_state[f"{p_dst}ffn.gate_up_proj.bias"] = state_dict[
                    f"{p_src}ffn.gate_up_proj.bias"
                ]
                hf_state[f"{p_dst}ffn.down_proj.bias"] = state_dict[
                    f"{p_src}ffn.down_proj.bias"
                ]
        else:
            hf_state[f"{p_dst}ffn.linear1.weight"] = state_dict[
                f"{p_src}ffn.linear1.weight"
            ]
            hf_state[f"{p_dst}ffn.linear2.weight"] = state_dict[
                f"{p_src}ffn.linear2.weight"
            ]
            if config.ff_bias:
                hf_state[f"{p_dst}ffn.linear1.bias"] = state_dict[
                    f"{p_src}ffn.linear1.bias"
                ]
                hf_state[f"{p_dst}ffn.linear2.bias"] = state_dict[
                    f"{p_src}ffn.linear2.bias"
                ]

    hf_state["model.decoder.norm.weight"] = state_dict["decoder.norm.weight"]
    if config.norm_type != "rmsnorm" and config.ff_bias:
        hf_state["model.decoder.norm.bias"] = state_dict["decoder.norm.bias"]

    # Generator (LM Head)
    hf_state["lm_head.weight"] = state_dict["generator.weight"]
    if "generator.bias" in state_dict:
        hf_state["lm_head.bias"] = state_dict["generator.bias"]
    else:
        # Some HF components expect bias
        hf_state["lm_head.bias"] = torch.zeros(
            config.vocab_size_tgt, dtype=hf_state["lm_head.weight"].dtype
        )

    hf_model.load_state_dict(hf_state, strict=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_dir", type=str, required=True, help="Path to experiment directory"
    )
    args = parser.parse_args()

    model_cfg, data_cfg, train_cfg, export_cfg = load_config(
        os.path.join(args.experiment_dir, "config.yaml")
    )

    model_file = os.path.join(args.experiment_dir, "averaged_model.safetensors")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found at {model_file}")

    state_dict = load_file(model_file, device="cpu")

    # Strip prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace("_orig_mod.", "")] = v
    state_dict = new_state_dict

    config = NMTConfig(
        d_model=model_cfg.d_model,
        enc_layers=model_cfg.enc_layers,
        dec_layers=model_cfg.dec_layers,
        n_heads=model_cfg.n_heads,
        ffn_dim=model_cfg.ffn_dim,
        max_len=model_cfg.max_len,
        dropout=model_cfg.dropout,
        vocab_size_src=model_cfg.vocab_size_src,
        vocab_size_tgt=model_cfg.vocab_size_tgt,
        vocab_size=model_cfg.vocab_size_tgt,
        activation=model_cfg.activation,
        ff_bias=model_cfg.ff_bias,
        mlp_type=model_cfg.mlp_type,
        layernorm_eps=model_cfg.layernorm_eps,
        norm_type=model_cfg.norm_type,
        tie_decoder_embeddings=model_cfg.tie_decoder_embeddings,
        pad_token_id=model_cfg.pad_id,
        unk_token_id=model_cfg.unk_id,
        bos_token_id=model_cfg.bos_id,
        eos_token_id=model_cfg.eos_id,
        decoder_start_token_id=model_cfg.bos_id,
        is_encoder_decoder=True,
    )

    hf_model = NMTForConditionalGeneration(config)

    map_weights(hf_model, state_dict, config)

    output_dir = os.path.join(args.experiment_dir, "exported_model_huggingface")
    hf_model.save_pretrained(output_dir)

    src_spm = f"{data_cfg.tokenizer_prefix_src}.model"
    tgt_spm = f"{data_cfg.tokenizer_prefix_tgt}.model"
    tokenizer = NMTTokenizer(src_vocab_file=src_spm, tgt_vocab_file=tgt_spm)
    tokenizer.save_pretrained(output_dir)

    print(f"Hugging Face model successfully exported to {output_dir}")


if __name__ == "__main__":
    main()
