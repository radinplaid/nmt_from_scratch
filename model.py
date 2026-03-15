import torch
import torch.nn as nn
import math
import torch.ao.quantization


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # Calculate RMS
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

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
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


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        ffn_dim,
        layernorm_eps,
        dropout=0.1,
        activation="gelu",
        bias=False,
        mlp_type="standard",
        norm_type="layernorm",
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True, bias=bias
        )
        self.ffn = FeedForward(d_model, ffn_dim, dropout, activation, bias, mlp_type)
        self.norm1 = get_norm(d_model, layernorm_eps, bias, norm_type)
        self.norm2 = get_norm(d_model, layernorm_eps, bias, norm_type)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # Pre-norm
        x = self.norm1(src)
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )[0]
        src = src + self.dropout(x)

        x = self.norm2(src)
        x = self.ffn(x)
        src = src + self.dropout(x)
        return src


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        ffn_dim,
        layernorm_eps,
        dropout=0.1,
        activation="gelu",
        bias=False,
        mlp_type="standard",
        norm_type="layernorm",
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True, bias=bias
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True, bias=bias
        )
        self.ffn = FeedForward(d_model, ffn_dim, dropout, activation, bias, mlp_type)
        self.norm1 = get_norm(d_model, layernorm_eps, bias, norm_type)
        self.norm2 = get_norm(d_model, layernorm_eps, bias, norm_type)
        self.norm3 = get_norm(d_model, layernorm_eps, bias, norm_type)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        tgt_is_causal=False,
        memory_is_causal=False,
    ):
        # Pre-norm
        x = self.norm1(tgt)
        # Self attention
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )[0]
        tgt = tgt + self.dropout(x)

        # Cross attention
        x = self.norm2(tgt)
        x = self.multihead_attn(
            x,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )[0]
        tgt = tgt + self.dropout(x)

        # FFN
        x = self.norm3(tgt)
        x = self.ffn(x)
        tgt = tgt + self.dropout(x)
        return tgt


class Seq2SeqTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.src_tok_emb = TokenEmbedding(config.vocab_size_src, config.d_model)
        self.tgt_tok_emb = TokenEmbedding(config.vocab_size_tgt, config.d_model)
        self.positional_encoding = PositionalEncoding(
            config.d_model, dropout=config.dropout, max_len=config.max_len
        )

        encoder_layer = EncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            ffn_dim=config.ffn_dim,
            layernorm_eps=config.layernorm_eps,
            dropout=config.dropout,
            activation=config.activation,
            bias=config.ff_bias,
            mlp_type=config.mlp_type,
            norm_type=config.norm_type,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.enc_layers,
            norm=get_norm(
                config.d_model, config.layernorm_eps, config.ff_bias, config.norm_type
            ),
        )

        decoder_layer = DecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            ffn_dim=config.ffn_dim,
            layernorm_eps=config.layernorm_eps,
            dropout=config.dropout,
            activation=config.activation,
            bias=config.ff_bias,
            mlp_type=config.mlp_type,
            norm_type=config.norm_type,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.dec_layers,
            norm=get_norm(
                config.d_model, config.layernorm_eps, config.ff_bias, config.norm_type
            ),
        )

        # Always use bias for the generator
        self.generator = nn.Linear(config.d_model, config.vocab_size_tgt, bias=True)
        if config.tie_decoder_embeddings:
            self.generator.weight = self.tgt_tok_emb.embedding.weight

        # Initialize parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        # Create padding mask: True where padding tokens exist
        src_padding_mask = (src == self.config.pad_id).to(torch.bool)

        # If src_mask is provided (e.g. for specific attention patterns), ensure it's bool
        if src_mask is not None and src_mask.dtype != torch.bool:
            src_mask = (
                (src_mask < 0)
                if src_mask.is_floating_point()
                else src_mask.to(torch.bool)
            )

        # Ensure the encoder itself uses boolean masks internally
        # This is a workaround for quantizable MultiheadAttention
        memory = self.encoder(
            src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask
        )
        return memory

    def decode(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))

        # Ensure all masks are boolean for quantizable MultiheadAttention
        if tgt_mask is not None and tgt_mask.dtype != torch.bool:
            tgt_mask = (
                (tgt_mask < 0)
                if tgt_mask.is_floating_point()
                else tgt_mask.to(torch.bool)
            )
        if memory_mask is not None and memory_mask.dtype != torch.bool:
            memory_mask = (
                (memory_mask < 0)
                if memory_mask.is_floating_point()
                else memory_mask.to(torch.bool)
            )
        if (
            tgt_key_padding_mask is not None
            and tgt_key_padding_mask.dtype != torch.bool
        ):
            tgt_key_padding_mask = tgt_key_padding_mask.to(torch.bool)
        if (
            memory_key_padding_mask is not None
            and memory_key_padding_mask.dtype != torch.bool
        ):
            memory_key_padding_mask = memory_key_padding_mask.to(torch.bool)

        out = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return out

    def project(self, x):
        return self.generator(x)

    def forward(self, src, tgt, return_outputs=False, label_smoothing=0.0):
        # src: (batch, src_len)
        # tgt: (batch, tgt_len) - contains BOS and EOS

        # Create masks
        src_padding_mask = (src == self.config.pad_id).to(torch.bool)

        # For training, we align input and target
        # Input to decoder: tgt[:, :-1] (BOS ... last_token)
        # Target for loss: tgt[:, 1:] (first_token ... EOS)

        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        tgt_padding_mask = (tgt_input == self.config.pad_id).to(torch.bool)

        # Causal mask for decoder autogression
        tgt_len = tgt_input.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(
            src.device
        )
        # Convert to bool
        if tgt_mask.dtype != torch.bool:
            tgt_mask = tgt_mask < 0

        # 1. Encode
        memory = self.encode(src)

        # 2. Decode
        outs = self.decode(
            tgt_input,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )

        # 3. Project
        logits = self.project(outs)

        # 4. Loss
        mask = tgt_out != self.config.pad_id
        num_tokens = mask.sum()

        loss = nn.functional.cross_entropy(
            logits.reshape(-1, self.config.vocab_size_tgt),
            tgt_out.reshape(-1),
            ignore_index=self.config.pad_id,
            label_smoothing=label_smoothing,
            reduction="sum",
        )

        if return_outputs:
            return loss, (logits, num_tokens)

        return loss, num_tokens

    @torch.no_grad()
    def generate(self, src, max_len=None, bos_id=None, eos_id=None, enc_output=None):
        max_len = max_len or self.config.max_len
        bos_id = bos_id if bos_id is not None else self.config.bos_id
        eos_id = eos_id if eos_id is not None else self.config.eos_id
        pad_id = self.config.pad_id

        src_padding_mask = (src == pad_id).to(torch.bool)
        bs = src.size(0)
        device = src.device

        if enc_output is None:
            memory = self.encode(src)
        else:
            memory = enc_output

        # Start with BOS
        ys = torch.full((bs, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(bs, dtype=torch.bool, device=device)

        for i in range(max_len):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(1)).to(
                device
            )
            out = self.decode(
                ys, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_padding_mask
            )

            # Project last token
            prob = self.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)

            # Update sequences
            next_word = next_word.clone()
            next_word[finished] = pad_id
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

            # Track finished
            finished = finished | (next_word == eos_id)

            if finished.all():
                break

        return ys[:, 1:]

    @torch.no_grad()
    def beam_search(self, src, max_len=None, beam_size=5, bos_id=None, eos_id=None):
        max_len = max_len or self.config.max_len
        bos_id = bos_id if bos_id is not None else self.config.bos_id
        eos_id = eos_id if eos_id is not None else self.config.eos_id
        pad_id = self.config.pad_id

        # src: (bs, seq_len)
        bs = src.size(0)
        device = src.device

        # Encode once
        memory = self.encode(src)

        src_padding_mask = (src == pad_id).to(torch.bool)

        # Tile memory and mask
        memory = memory.repeat_interleave(beam_size, dim=0)
        src_padding_mask = src_padding_mask.repeat_interleave(beam_size, dim=0)

        # Initialize
        scores = torch.zeros(bs, beam_size, device=device)
        scores[:, 1:] = -1e9

        # inputs: (bs, beam_size, seq_len)
        inputs = torch.full((bs, beam_size, 1), bos_id, dtype=torch.long, device=device)

        vocab_size = self.config.vocab_size_tgt

        for i in range(max_len):
            curr_seq_len = inputs.size(2)
            flat_inputs = inputs.view(bs * beam_size, curr_seq_len)

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(curr_seq_len).to(
                device
            )

            # Decode
            out = self.decode(
                flat_inputs,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_padding_mask,
            )

            # Logits for last token
            logits = self.project(out[:, -1])
            log_probs = torch.log_softmax(logits, dim=-1)

            # Reshape back to (bs, beam, vocab)
            log_probs = log_probs.view(bs, beam_size, vocab_size)

            # Add to previous scores
            total_scores = scores.unsqueeze(-1) + log_probs

            # Flatten to find top-k across all (beam * vocab) options
            total_scores_flat = total_scores.view(bs, -1)

            # Get top k
            top_acc_scores, top_indices = total_scores_flat.topk(beam_size, dim=-1)

            # Convert indices back
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size

            # Update scores
            scores = top_acc_scores

            # Construct new inputs
            new_inputs = []
            for b in range(bs):
                prev_beams = inputs[b]
                selected_beam_indices = beam_indices[b]
                selected_tokens = token_indices[b]

                selected_sequences = prev_beams[selected_beam_indices]
                new_seq = torch.cat(
                    [selected_sequences, selected_tokens.unsqueeze(-1)], dim=-1
                )
                new_inputs.append(new_seq)

            inputs = torch.stack(new_inputs)

        # Return best beam
        return inputs[:, 0, 1:]  # Skip BOS

    def convert_to_int8(self):
        """
        Convert the PTQ-calibrated model to a quantized INT8 model.
        This should be called after calibrate().
        """
        self.eval()
        torch.ao.quantization.convert(self, inplace=True)
        print("Model converted to INT8")

    def calibrate(self, loader, num_batches=10):
        """
        Run a few batches of data through the model to update quantization observers.
        This is useful after averaging weights.
        """
        device = next(self.parameters()).device
        self.eval()
        # Ensure observers are enabled and Dropout is disabled
        with torch.no_grad():
            for i, (src, tgt) in enumerate(loader):
                if i >= num_batches:
                    break
                # Run forward pass (updates observers)
                self.forward(src.to(device), tgt.to(device))
        print(f"Calibration completed on {num_batches} batches")
