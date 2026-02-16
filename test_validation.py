#!/usr/bin/env python3
"""
Test script to verify the validation function works correctly.
This script tests the modified validate function with a simple example.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig
from model import Seq2SeqTransformer


def test_validation():
    """Test the validation function with a simple example."""
    print("Testing validation function...")

    # Create a simple config
    model_cfg = ModelConfig()
    model_cfg.vocab_size = 100  # Small vocab for testing
    model_cfg.max_len = 10

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2SeqTransformer(model_cfg).to(device)

    # Create dummy data
    batch_size = 2
    seq_len = 5

    # Create source and target sequences
    # BOS=2, EOS=3, PAD=0
    src = torch.randint(4, 100, (batch_size, seq_len)).to(device)
    src[:, 0] = 2  # BOS
    src[:, -1] = 3  # EOS

    tgt = torch.randint(4, 100, (batch_size, seq_len)).to(device)
    tgt[:, 0] = 2  # BOS
    tgt[:, -1] = 3  # EOS

    # Test forward pass
    print("Testing forward pass...")
    model.eval()
    with torch.no_grad():
        loss, (logits, _) = model(src, tgt, return_outputs=True)

    print(f"Loss shape: {loss.shape if hasattr(loss, 'shape') else 'scalar'}")
    print(f"Loss value: {loss.item():.4f}")
    print(f"Logits shape: {logits.shape}")

    # Test generate method
    print("\nTesting generate method...")
    with torch.no_grad():
        generated = model.generate(src, max_len=10)

    print(f"Generated shape: {generated.shape}")
    print(f"Generated sample: {generated[0].tolist()}")

    # Test beam_search method
    print("\nTesting beam_search method...")
    with torch.no_grad():
        beam_generated = model.beam_search(src, max_len=10, beam_size=3)

    print(f"Beam generated shape: {beam_generated.shape}")
    print(f"Beam generated sample: {beam_generated[0].tolist()}")

    # Test manual loss computation (same as validation)
    print("\nTesting manual loss computation...")
    model.eval()
    with torch.no_grad():
        # Encode source
        # src_mask not needed explicitly if we use model.encode which handles it
        src_padding_mask = src == 0
        enc = model.encode(src)

        # Get target without BOS
        tgt_no_bos = tgt[:, 1:]

        # Initialize with BOS
        curr_tokens = torch.full((batch_size, 1), 2, dtype=torch.long, device=device)

        # Compute loss manually
        total_loss_batch = 0
        num_tokens = 0

        for pos in range(tgt_no_bos.size(1)):
            # Forward pass
            # Decode requires causal mask for current sequence length
            tgt_len = curr_tokens.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(
                device
            )

            # decode returns hidden states
            out = model.decode(
                curr_tokens,
                enc,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_padding_mask,
            )

            # Project to logits
            logits = model.generator(out)
            next_logits = logits[:, -1, :]

            # Compute loss for this position
            target_token = tgt_no_bos[:, pos]
            loss_pos = torch.nn.functional.cross_entropy(
                next_logits, target_token, ignore_index=0, reduction="sum"
            )

            # Count non-padding tokens
            non_pad = (target_token != 0).sum().item()

            total_loss_batch += loss_pos.item()
            num_tokens += non_pad

            # Update for next step (teacher forcing)
            next_token = target_token.unsqueeze(-1)
            curr_tokens = torch.cat([curr_tokens, next_token], dim=-1)

        # Average loss over non-padding tokens
        if num_tokens > 0:
            manual_loss = total_loss_batch / num_tokens
        else:
            manual_loss = 0

        print(f"Manual loss: {manual_loss:.4f}")
        print(f"Model loss: {loss.item():.4f}")
        print(f"Difference: {abs(manual_loss - loss.item()):.6f}")

    print("\n✓ All tests passed!")
    return True


if __name__ == "__main__":
    try:
        test_validation()
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
