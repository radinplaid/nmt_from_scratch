import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.ao.quantization
from torch.utils.checkpoint import checkpoint


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


class GroupedQueryAttention(nn.Module):
    """Multi-head attention with optional Grouped Query / Multi-Query support.

    Compatible with CTranslate2's ``num_heads_kv`` parameter.

    * ``n_kv_heads = 0`` or ``n_kv_heads = n_heads`` → standard MHA
    * ``n_kv_heads = 1``                              → Multi-Query Attention (MQA)
    * ``1 < n_kv_heads < n_heads``                   → Grouped Query Attention (GQA)

    Uses ``F.scaled_dot_product_attention``, which dispatches to Flash Attention /
    memory-efficient attention automatically when available.

    CTranslate2-compatible weight layout (separate Q and KV projections):
        q_proj   : ``(n_heads * head_dim, d_model)``
        kv_proj  : ``(2 * n_kv_heads * head_dim, d_model)``  – fused K and V
        out_proj : ``(d_model, n_heads * head_dim)``
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int = 0,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()
        n_kv_heads = n_kv_heads if n_kv_heads > 0 else n_heads
        assert n_heads % n_kv_heads == 0, (
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        )
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=bias)
        self.kv_proj = nn.Linear(d_model, 2 * n_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=bias)
        self.attn_dropout = dropout
        # nn.TransformerEncoder / nn.TransformerDecoder inspect this attribute
        # to determine tensor layout.  We always use batch-first (B, T, d).
        self.batch_first = True

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attn_mask=None,
        key_padding_mask=None,
        need_weights: bool = False,
    ):
        """
        Args:
            query:            ``(B, T, d_model)``
            key_value:        ``(B, S, d_model)`` – pass *query* for self-attention
            attn_mask:        optional ``(T, S)`` or ``(B, T, S)`` float **or** bool mask.
                              Bool convention: ``True`` = mask out (same as
                              ``nn.MultiheadAttention``).
            key_padding_mask: optional ``(B, S)`` bool mask, ``True`` = padding position.

        Returns:
            ``(output, None)`` where output has shape ``(B, T, d_model)``.
            The ``None`` placeholder keeps the interface identical to
            ``nn.MultiheadAttention``.
        """
        B, T, _ = query.shape
        S = key_value.size(1)

        q = self.q_proj(query)        # (B, T, H * hd)
        kv = self.kv_proj(key_value)  # (B, S, 2 * Hkv * hd)
        k, v = kv.chunk(2, dim=-1)    # each (B, S, Hkv * hd)

        q = q.view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)  # (B, H,   T, hd)
        k = k.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B, Hkv, S, hd)
        v = v.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B, Hkv, S, hd)

        # Expand KV heads for GQA (no-op when n_groups == 1)
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)  # (B, H, S, hd)
            v = v.repeat_interleave(self.n_groups, dim=1)  # (B, H, S, hd)

        # Build a merged float mask from the optional attn_mask + key_padding_mask.
        # Both follow the nn.MultiheadAttention convention: True = mask out.
        merged_mask = None
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                merged_mask = torch.zeros(
                    attn_mask.shape, dtype=q.dtype, device=q.device
                ).masked_fill_(attn_mask, float("-inf"))
            else:
                merged_mask = attn_mask.to(dtype=q.dtype)
            # Promote to 4-D so it broadcasts over (B, H, T, S)
            if merged_mask.dim() == 2:
                merged_mask = merged_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, S)
            elif merged_mask.dim() == 3:
                merged_mask = merged_mask.unsqueeze(1)               # (B, 1, T, S)

        if key_padding_mask is not None:
            # nn.TransformerEncoder (PyTorch ≥ 2.0) may pre-convert the bool mask
            # to a float additive mask (shape B×S, -inf where padding).
            # We must handle both bool and float inputs.
            if key_padding_mask.dtype == torch.bool:
                kpm = torch.zeros(
                    B, 1, 1, S, dtype=q.dtype, device=q.device
                ).masked_fill_(key_padding_mask.view(B, 1, 1, S), float("-inf"))
            else:
                # Already a float additive mask; just reshape for broadcasting
                kpm = key_padding_mask.to(dtype=q.dtype).view(B, 1, 1, S)
            merged_mask = kpm if merged_mask is None else merged_mask + kpm

        # F.scaled_dot_product_attention dispatches to Flash Attention when available
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=merged_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )  # (B, H, T, hd)

        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.out_proj(out), None  # None: no attention weights returned


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        ffn_dim,
        dropout=0.1,
        activation="gelu",
        bias=False,
        mlp_type="standard",
        n_kv_heads=0,
    ):
        super().__init__()
        self.self_attn = GroupedQueryAttention(
            d_model, nhead, n_kv_heads=n_kv_heads, dropout=dropout, bias=bias
        )
        self.ffn = FeedForward(d_model, ffn_dim, dropout, activation, bias, mlp_type)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6, bias=bias)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # Pre-norm
        x = self.norm1(src)
        x = self.self_attn(
            x, x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
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
        dropout=0.1,
        activation="gelu",
        bias=False,
        mlp_type="standard",
        n_kv_heads=0,
    ):
        super().__init__()
        self.self_attn = GroupedQueryAttention(
            d_model, nhead, n_kv_heads=n_kv_heads, dropout=dropout, bias=bias
        )
        # "multihead_attn" is kept as the attribute name to preserve state-dict keys
        # that the CTranslate2 converter already knows about.
        self.multihead_attn = GroupedQueryAttention(
            d_model, nhead, n_kv_heads=n_kv_heads, dropout=dropout, bias=bias
        )
        self.ffn = FeedForward(d_model, ffn_dim, dropout, activation, bias, mlp_type)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6, bias=bias)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6, bias=bias)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-6, bias=bias)
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
        # Pre-norm self-attention (causal)
        x = self.norm1(tgt)
        x = self.self_attn(
            x, x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(x)

        # Cross-attention (query from decoder, key/value from encoder memory)
        x = self.norm2(tgt)
        x = self.multihead_attn(
            x, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
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

        self.src_tok_emb = TokenEmbedding(config.vocab_size, config.d_model)
        self.tgt_tok_emb = TokenEmbedding(config.vocab_size, config.d_model)
        self.positional_encoding = PositionalEncoding(
            config.d_model, dropout=config.dropout, max_len=config.max_len
        )

        n_kv_heads = getattr(config, "n_kv_heads", 0)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.ffn_dim,
                dropout=config.dropout,
                activation=config.activation,
                batch_first=True,
                bias=config.ff_bias,
            ),
            num_layers=config.enc_layers,
            norm=nn.LayerNorm(config.d_model, eps=1e-6, bias=config.ff_bias),
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.ffn_dim,
                dropout=config.dropout,
                activation=config.activation,
                batch_first=True,
                bias=config.ff_bias,
            ),
            num_layers=config.dec_layers,
            norm=nn.LayerNorm(config.d_model, eps=1e-6, bias=config.ff_bias),
        )

        self.generator = nn.Linear(
            config.d_model, config.vocab_size, bias=config.ff_bias
        )

        # Initialize parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Tie decoder output-projection weights to the target token embeddings.
        # This reduces the parameter count by vocab_size × d_model and consistently
        # improves translation quality (Press & Wolf, 2017).
        if getattr(config, "tie_embeddings", False):
            self.generator.weight = self.tgt_tok_emb.embedding.weight

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
        if getattr(self.config, "use_checkpoint", False) and self.training:
            memory = checkpoint(
                self.encoder,
                src_emb,
                src_mask,
                src_padding_mask,
                use_reentrant=False,
            )
        else:
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

        if getattr(self.config, "use_checkpoint", False) and self.training:
            out = checkpoint(
                self.decoder,
                tgt_emb,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                use_reentrant=False,
            )
        else:
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
            logits.reshape(-1, self.config.vocab_size),
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

        vocab_size = self.config.vocab_size

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
