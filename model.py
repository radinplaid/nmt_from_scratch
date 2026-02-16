import torch
import torch.nn as nn
import math


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


class Seq2SeqTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.src_tok_emb = TokenEmbedding(config.vocab_size, config.d_model)
        self.tgt_tok_emb = TokenEmbedding(config.vocab_size, config.d_model)
        self.positional_encoding = PositionalEncoding(
            config.d_model, dropout=config.dropout, max_len=config.max_len
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm performs better
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.enc_layers
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.dec_layers
        )

        self.generator = nn.Linear(config.d_model, config.vocab_size)

        # Initialize parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        return self.encoder(src_emb, src_key_padding_mask=(src == 0))

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
        return self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

    def forward(self, src, tgt, return_outputs=False, label_smoothing=0.0):
        # src: (batch, src_len)
        # tgt: (batch, tgt_len) - contains BOS and EOS

        # Create masks
        src_padding_mask = src == 0  # (batch, src_len)

        # For training, we align input and target
        # Input to decoder: tgt[:, :-1] (BOS ... last_token)
        # Target for loss: tgt[:, 1:] (first_token ... EOS)

        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        tgt_padding_mask = tgt_input == 0  # (batch, tgt_len-1)

        # Causal mask for decoder autogression
        tgt_len = tgt_input.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(
            src.device
        )

        # 1. Encode
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        memory = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)

        # 2. Decode
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt_input))
        outs = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )

        # 3. Project
        logits = self.generator(outs)  # (batch, tgt_len-1, vocab)

        if return_outputs:
            # Calculate loss but also return logits
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                tgt_out.reshape(-1),
                ignore_index=0,
                label_smoothing=label_smoothing,
            )
            return loss, (logits, None)

        # 4. Loss
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            tgt_out.reshape(-1),
            ignore_index=0,
            label_smoothing=label_smoothing,
        )

        return loss

    @torch.no_grad()
    def generate(self, src, max_len=256, bos_id=2, eos_id=3, enc_output=None):
        src_padding_mask = src == 0
        bs = src.size(0)
        device = src.device

        if enc_output is None:
            memory = self.encode(src)
        else:
            memory = enc_output

        # Start with BOS
        ys = torch.full((bs, 1), bos_id, dtype=torch.long, device=device)

        for i in range(max_len):
            # Decode
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(1)).to(
                device
            )
            out = self.decode(
                ys, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_padding_mask
            )

            # Project last token
            prob = self.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)

            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

            # Check for EOS (simple check: if all have generated EOS at some point)
            # For simplicity in this greedy loop, we just run to max_len or check if current all are EOS (unlikely for batch)
            # Better: track finished status per sample.
            # Here keeping consistent with previous simple implementation behavior?
            # Previous one: "if (next_token == eos_id).all(): break"
            if (next_word == eos_id).all():
                break

        return ys[:, 1:]

    @torch.no_grad()
    def beam_search(self, src, max_len=256, beam_size=5, bos_id=2, eos_id=3):
        # src: (bs, seq_len)
        bs = src.size(0)
        device = src.device

        # Encode once
        # memory: (bs, seq_len, d_model)
        memory = self.encode(src)

        # We need to tile memory for beam search: (bs * beam, seq_len, d_model)
        # But standard beam search implementation usually keeps batch dimension separate until usually not efficient.
        # Let's do simple batched beam search where we flatten bs*beam

        src_padding_mask = src == 0

        # Tile memory and mask
        # memory: (bs, seq, dim) -> (bs*beam, seq, dim)
        memory = memory.repeat_interleave(beam_size, dim=0)
        src_padding_mask = src_padding_mask.repeat_interleave(beam_size, dim=0)

        # Initial input: BOS
        # beams: (bs*beam, 1) - but initially just (bs, 1) then expanded?
        # Let's keep beams as (bs, beam_size, seq_len)

        # Initialize
        # scores: (bs, beam_size)
        scores = torch.zeros(bs, beam_size, device=device)
        scores[:, 1:] = -1e9  # Only first beam is active initially for each batch

        # inputs: (bs, beam_size, seq_len)
        inputs = torch.full((bs, beam_size, 1), bos_id, dtype=torch.long, device=device)

        # To run explicitly in batch with Transformer, we flatten:
        # current_inputs: (bs * beam_size, seq_len)

        vocab_size = self.config.vocab_size

        for i in range(max_len):
            # Flatten inputs
            curr_seq_len = inputs.size(2)
            flat_inputs = inputs.view(bs * beam_size, curr_seq_len)

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(curr_seq_len).to(
                device
            )

            # Decode
            # out: (bs*beam, seq, dim)
            out = self.decode(
                flat_inputs,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_padding_mask,
            )

            # Logits for last token: (bs*beam, vocab)
            logits = self.generator(out[:, -1])
            log_probs = torch.log_softmax(logits, dim=-1)

            # Reshape back to (bs, beam, vocab)
            log_probs = log_probs.view(bs, beam_size, vocab_size)

            # Add to previous scores
            # scores: (bs, beam) -> (bs, beam, 1)
            # total_scores: (bs, beam, vocab)
            total_scores = scores.unsqueeze(-1) + log_probs

            # Flatten to find top-k across all (beam * vocab) options
            # (bs, beam * vocab)
            total_scores_flat = total_scores.view(bs, -1)

            # Get top k
            top_acc_scores, top_indices = total_scores_flat.topk(beam_size, dim=-1)

            # Convert indices back to (beam_idx, vocab_idx)
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size

            # Update scores
            scores = top_acc_scores

            # Construct new inputs
            # inputs: (bs, beam, seq)
            new_inputs = []
            for b in range(bs):
                # source beams for this batch element
                # inputs[b]: (beam, seq)
                prev_beams = inputs[b]

                # selected beams
                selected_beam_indices = beam_indices[b]  # (beam_size,)
                selected_tokens = token_indices[b]  # (beam_size,)

                # gather previous sequences
                selected_sequences = prev_beams[
                    selected_beam_indices
                ]  # (beam_size, seq)

                # append new tokens
                new_seq = torch.cat(
                    [selected_sequences, selected_tokens.unsqueeze(-1)], dim=-1
                )
                new_inputs.append(new_seq)

            inputs = torch.stack(new_inputs)

            # Check EOS? (Optional optimization: stop if all top beams are EOS)

        # Return best beam
        # inputs: (bs, beam, seq) -> return (bs, seq) of best beam (index 0)
        return inputs[:, 0, 1:]  # Skip BOS
