import math
from typing import Optional, Tuple, Union

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
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from configuration_nmt import NMTConfig


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm_x = torch.mean(x**2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x, start_pos=0):
        # We need to support start_pos for past_key_values (generation)
        seq_len = x.size(1)
        x = x + self.pe[:, start_pos : start_pos + seq_len, :]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.d_model)


class FeedForward(nn.Module):
    def __init__(
        self,
        d_model,
        ffn_dim,
        dropout=0.1,
        activation="gelu",
        bias=False,
        mlp_type="standard",
    ):
        super().__init__()
        self.mlp_type = mlp_type
        if mlp_type == "gated":
            self.gate_up_proj = nn.Linear(d_model, 2 * ffn_dim, bias=bias)
            self.down_proj = nn.Linear(ffn_dim, d_model, bias=bias)
        else:
            self.linear1 = nn.Linear(d_model, ffn_dim, bias=bias)
            self.linear2 = nn.Linear(ffn_dim, d_model, bias=bias)

        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "silu" or activation == "swish":
            self.act = nn.SiLU()
        else:
            self.act = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.mlp_type == "gated":
            gate_up = self.gate_up_proj(x)
            gate, up = gate_up.chunk(2, dim=-1)
            x = self.act(gate) * up
            x = self.dropout(x)
            x = self.down_proj(x)
        else:
            x = self.act(self.linear1(x))
            x = self.dropout(x)
            x = self.linear2(x)
        return x


def get_norm(d_model, eps, bias, norm_type):
    if norm_type == "rmsnorm":
        return RMSNorm(d_model, eps=eps)
    else:
        return nn.LayerNorm(d_model, eps=eps, bias=bias)


class NMTPreTrainedModel(PreTrainedModel):
    config_class = NMTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.d_model**-0.5)


class EncoderLayer(nn.Module):
    def __init__(self, config: NMTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
            bias=config.ff_bias,
        )
        self.ffn = FeedForward(
            config.d_model,
            config.ffn_dim,
            config.dropout,
            config.activation,
            config.ff_bias,
            config.mlp_type,
        )
        self.norm1 = get_norm(
            config.d_model, config.layernorm_eps, config.ff_bias, config.norm_type
        )
        self.norm2 = get_norm(
            config.d_model, config.layernorm_eps, config.ff_bias, config.norm_type
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, src, src_key_padding_mask=None):
        x = self.norm1(src)
        x = self.self_attn(
            x,
            x,
            x,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )[0]
        src = src + self.dropout(x)

        x = self.norm2(src)
        x = self.ffn(x)
        src = src + self.dropout(x)
        return src


class NMTEncoder(NMTPreTrainedModel):
    def __init__(self, config: NMTConfig):
        super().__init__(config)
        self.config = config
        self.src_tok_emb = TokenEmbedding(config.vocab_size_src, config.d_model)
        self.positional_encoding = PositionalEncoding(
            config.d_model, dropout=config.dropout, max_len=config.max_len
        )
        self.layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.enc_layers)]
        )
        self.norm = get_norm(
            config.d_model, config.layernorm_eps, config.ff_bias, config.norm_type
        )

    def forward(self, input_ids=None, attention_mask=None, return_dict=True, **kwargs):
        src_emb = self.positional_encoding(self.src_tok_emb(input_ids))

        # Hugging Face attention_mask is 1 for non-padding, 0 for padding.
        # PyTorch src_key_padding_mask is True for padding.
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0

        hidden_states = src_emb
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, src_key_padding_mask=src_key_padding_mask
            )

        hidden_states = self.norm(hidden_states)

        if not return_dict:
            return (hidden_states,)
        return BaseModelOutput(last_hidden_state=hidden_states)


class DecoderLayer(nn.Module):
    def __init__(self, config: NMTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
            bias=config.ff_bias,
        )
        self.multihead_attn = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
            bias=config.ff_bias,
        )
        self.ffn = FeedForward(
            config.d_model,
            config.ffn_dim,
            config.dropout,
            config.activation,
            config.ff_bias,
            config.mlp_type,
        )
        self.norm1 = get_norm(
            config.d_model, config.layernorm_eps, config.ff_bias, config.norm_type
        )
        self.norm2 = get_norm(
            config.d_model, config.layernorm_eps, config.ff_bias, config.norm_type
        )
        self.norm3 = get_norm(
            config.d_model, config.layernorm_eps, config.ff_bias, config.norm_type
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        past_key_value=None,
    ):
        # Handle caching (past_key_value) not fully implemented for training brevity but needed for efficient generation
        # We will use simple approach: if past_key_value is provided, we just rely on standard HF decoding where we pass the whole sequence if we can't do fast caching
        # But wait, PyTorch MultiheadAttention does not natively support caching easily without internal hackery.
        # We'll re-compute for now, generation will be slightly slower but correct.

        x = self.norm1(tgt)
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )[0]
        tgt = tgt + self.dropout(x)

        x = self.norm2(tgt)
        x = self.multihead_attn(
            x,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )[0]
        tgt = tgt + self.dropout(x)

        x = self.norm3(tgt)
        x = self.ffn(x)
        tgt = tgt + self.dropout(x)
        return tgt, None


class NMTDecoder(NMTPreTrainedModel):
    def __init__(self, config: NMTConfig):
        super().__init__(config)
        self.config = config
        self.tgt_tok_emb = TokenEmbedding(config.vocab_size_tgt, config.d_model)
        self.positional_encoding = PositionalEncoding(
            config.d_model, dropout=config.dropout, max_len=config.max_len
        )
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.dec_layers)]
        )
        self.norm = get_norm(
            config.d_model, config.layernorm_eps, config.ff_bias, config.norm_type
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        return_dict=True,
        **kwargs,
    ):
        seq_len = input_ids.size(1)

        tgt_emb = self.positional_encoding(self.tgt_tok_emb(input_ids), start_pos=0)

        tgt_key_padding_mask = None
        if attention_mask is not None:
            tgt_key_padding_mask = attention_mask == 0

        memory_key_padding_mask = None
        if encoder_attention_mask is not None:
            memory_key_padding_mask = encoder_attention_mask == 0

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(
            input_ids.device
        )
        if tgt_mask.dtype != torch.bool:
            tgt_mask = tgt_mask < 0

        hidden_states = tgt_emb
        next_decoder_cache = () if use_cache else None

        for layer in self.layers:
            hidden_states, layer_past = layer(
                hidden_states,
                encoder_hidden_states,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                past_key_value=None,  # Caching simplified out
            )
            if use_cache:
                next_decoder_cache += (layer_past,)

        hidden_states = self.norm(hidden_states)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_decoder_cache] if v is not None
            )

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=None,
            attentions=None,
        )


class NMTModel(NMTPreTrainedModel):
    def __init__(self, config: NMTConfig):
        super().__init__(config)
        self.encoder = NMTEncoder(config)
        self.decoder = NMTDecoder(config)
        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        use_cache=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=return_dict,
            )

        encoder_hidden_states = (
            encoder_outputs[0] if not return_dict else encoder_outputs.last_hidden_state
        )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=None,  # Simplifying caching
            encoder_last_hidden_state=encoder_hidden_states,
        )


class NMTForConditionalGeneration(NMTPreTrainedModel):
    def __init__(self, config: NMTConfig):
        super().__init__(config)
        self.model = NMTModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size_tgt, bias=True)
        if config.tie_decoder_embeddings:
            self.lm_head.weight = self.model.decoder.tgt_tok_emb.embedding.weight
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # Very simple generation setup without fast caching (past_key_values)
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # Will just pass forward without real caching logic
        }

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if labels is not None and decoder_input_ids is None:
            # Shift labels to create decoder_input_ids
            decoder_input_ids = torch.cat(
                [
                    torch.full(
                        (labels.shape[0], 1),
                        self.config.decoder_start_token_id,
                        device=labels.device,
                    ),
                    labels[:, :-1],
                ],
                dim=-1,
            )
            decoder_attention_mask = (
                decoder_input_ids != self.config.pad_token_id
            ).long()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
        )

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(
                lm_logits.view(-1, self.config.vocab_size_tgt), labels.view(-1)
            )

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=None,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state
            if hasattr(outputs, "encoder_last_hidden_state")
            else outputs[1],
        )


NMTConfig.register_for_auto_class()
NMTModel.register_for_auto_class("AutoModel")
NMTForConditionalGeneration.register_for_auto_class("AutoModelForSeq2SeqLM")
