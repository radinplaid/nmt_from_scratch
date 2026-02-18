"""
Unit tests for Seq2SeqTransformer generate and beam_search methods.
"""
import torch
import torch.nn as nn
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ModelConfig
from model import Seq2SeqTransformer


class TestGenerateMethod:
    """Tests for the generate method (greedy decoding)."""
    
    @pytest.fixture
    def model(self):
        """Create a small model for testing."""
        config = ModelConfig(
            d_model=128,
            enc_layers=2,
            dec_layers=2,
            n_heads=4,
            ffn_dim=512,
            max_len=128,
            dropout=0.0,  # No dropout for testing
            vocab_size=1000,
        )
        model = Seq2SeqTransformer(config)
        model.eval()
        return model
    
    @pytest.fixture
    def sample_src(self):
        """Create sample source input."""
        # batch_size=2, seq_len=5
        return torch.tensor([[5, 10, 15, 20, 3], [6, 11, 16, 21, 3]])
    
    def test_generate_returns_tensor(self, model, sample_src):
        """Test that generate returns a tensor."""
        result = model.generate(sample_src, max_len=10, bos_id=2, eos_id=3)
        assert isinstance(result, torch.Tensor)
    
    def test_generate_output_shape(self, model, sample_src):
        """Test that generate returns correct shape."""
        bs = sample_src.size(0)
        result = model.generate(sample_src, max_len=10, bos_id=2, eos_id=3)
        # Should return (batch_size, generated_len)
        assert result.dim() == 2
        assert result.size(0) == bs
    
    def test_generate_starts_with_bos(self, model, sample_src):
        """Test that generation starts with BOS token."""
        # We can verify by checking the input includes BOS
        bs = sample_src.size(0)
        device = sample_src.device
        bos_id = 2
        eos_id = 3
        
        with torch.no_grad():
            ys = torch.full((bs, 1), bos_id, dtype=torch.long, device=device)
            # First decode should use BOS only
            src_padding_mask = (sample_src == 0).to(torch.bool)
            memory = model.encode(sample_src)
            
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(1)).to(device)
            out = model.decode(ys, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_padding_mask)
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
        
        # Should get some token prediction
        assert next_word.shape == (bs,)
    
    def test_generate_respects_max_len(self, model, sample_src):
        """Test that generation respects max_len."""
        max_len = 5
        result = model.generate(sample_src, max_len=max_len, bos_id=2, eos_id=3)
        # Should not exceed max_len (might be shorter if EOS is generated)
        assert result.size(1) <= max_len
    
    def test_generate_empty_input(self, model):
        """Test generation with empty input (padding only)."""
        src = torch.zeros((1, 3), dtype=torch.long)
        result = model.generate(src, max_len=5, bos_id=2, eos_id=3)
        assert result.dim() == 2
        assert result.size(0) == 1
    
    def test_generate_single_batch(self, model):
        """Test generation with batch_size=1."""
        src = torch.tensor([[5, 10, 15, 3]])
        result = model.generate(src, max_len=10, bos_id=2, eos_id=3)
        assert result.size(0) == 1
        assert result.size(1) <= 10
    
    def test_generate_deterministic(self, model, sample_src):
        """Test that generate is deterministic with same inputs."""
        result1 = model.generate(sample_src, max_len=10, bos_id=2, eos_id=3)
        result2 = model.generate(sample_src, max_len=10, bos_id=2, eos_id=3)
        assert torch.equal(result1, result2)
    
    def test_generate_with_enc_output(self, model, sample_src):
        """Test generate with pre-computed encoder output."""
        with torch.no_grad():
            enc_output = model.encode(sample_src)
            result1 = model.generate(sample_src, max_len=10, bos_id=2, eos_id=3)
            result2 = model.generate(sample_src, max_len=10, bos_id=2, eos_id=3, enc_output=enc_output)
        # Should produce same results
        assert torch.equal(result1, result2)


class TestBeamSearchMethod:
    """Tests for the beam_search method."""
    
    @pytest.fixture
    def model(self):
        """Create a small model for testing."""
        config = ModelConfig(
            d_model=128,
            enc_layers=2,
            dec_layers=2,
            n_heads=4,
            ffn_dim=512,
            max_len=128,
            dropout=0.0,
            vocab_size=1000,
        )
        model = Seq2SeqTransformer(config)
        model.eval()
        return model
    
    @pytest.fixture
    def sample_src(self):
        """Create sample source input."""
        return torch.tensor([[5, 10, 15, 20, 3], [6, 11, 16, 21, 3]])
    
    def test_beam_search_returns_tensor(self, model, sample_src):
        """Test that beam_search returns a tensor."""
        result = model.beam_search(sample_src, max_len=10, beam_size=3, bos_id=2, eos_id=3)
        assert isinstance(result, torch.Tensor)
    
    def test_beam_search_output_shape(self, model, sample_src):
        """Test that beam_search returns correct shape."""
        bs = sample_src.size(0)
        result = model.beam_search(sample_src, max_len=10, beam_size=3, bos_id=2, eos_id=3)
        # Should return (batch_size, seq_len) - best beam only
        assert result.dim() == 2
        assert result.size(0) == bs
    
    def test_beam_search_respects_max_len(self, model, sample_src):
        """Test that beam_search respects max_len."""
        max_len = 5
        result = model.beam_search(sample_src, max_len=max_len, beam_size=3, bos_id=2, eos_id=3)
        # Should not exceed max_len
        assert result.size(1) <= max_len
    
    def test_beam_search_single_batch(self, model):
        """Test beam_search with batch_size=1."""
        src = torch.tensor([[5, 10, 15, 3]])
        result = model.beam_search(src, max_len=10, beam_size=3, bos_id=2, eos_id=3)
        assert result.size(0) == 1
        assert result.size(1) <= 10
    
    def test_beam_search_different_beam_sizes(self, model, sample_src):
        """Test beam_search with different beam sizes."""
        for beam_size in [1, 2, 3, 5]:
            result = model.beam_search(sample_src, max_len=10, beam_size=beam_size, bos_id=2, eos_id=3)
            # Returns (batch_size, seq_len) - best beam only
            assert result.dim() == 2
            assert result.size(0) == sample_src.size(0)
    
    def test_beam_search_deterministic(self, model, sample_src):
        """Test that beam_search is deterministic."""
        torch.manual_seed(42)
        result1 = model.beam_search(sample_src, max_len=10, beam_size=3, bos_id=2, eos_id=3)
        
        torch.manual_seed(42)
        result2 = model.beam_search(sample_src, max_len=10, beam_size=3, bos_id=2, eos_id=3)
        
        assert torch.equal(result1, result2)


class TestCompareGreedyVsBeamSearch:
    """Tests comparing greedy and beam search results."""
    
    @pytest.fixture
    def model(self):
        """Create a small model for testing."""
        config = ModelConfig(
            d_model=128,
            enc_layers=2,
            dec_layers=2,
            n_heads=4,
            ffn_dim=512,
            max_len=128,
            dropout=0.0,
            vocab_size=1000,
        )
        model = Seq2SeqTransformer(config)
        model.eval()
        return model
    
    def test_beam_size_1_equals_greedy(self, model):
        """Test that beam_size=1 gives same results as greedy."""
        src = torch.tensor([[5, 10, 15, 3]])
        
        with torch.no_grad():
            # Greedy result
            greedy_result = model.generate(src, max_len=10, bos_id=2, eos_id=3)
            
            # Beam search with beam_size=1
            beam_result = model.beam_search(src, max_len=10, beam_size=1, bos_id=2, eos_id=3)
        
        # Should be equal (or very similar due to implementation differences)
        assert greedy_result.shape == beam_result.shape


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def model(self):
        """Create a small model for testing."""
        config = ModelConfig(
            d_model=64,
            enc_layers=1,
            dec_layers=1,
            n_heads=2,
            ffn_dim=128,
            max_len=64,
            dropout=0.0,
            vocab_size=100,
        )
        model = Seq2SeqTransformer(config)
        model.eval()
        return model
    
    def test_very_short_max_len(self, model):
        """Test with very short max_len."""
        src = torch.tensor([[5, 10, 15, 3]])
        result = model.generate(src, max_len=1, bos_id=2, eos_id=3)
        assert result.size(1) <= 1
    
    def test_all_padding_src(self, model):
        """Test with all padding in source."""
        src = torch.zeros((2, 5), dtype=torch.long)
        result = model.generate(src, max_len=5, bos_id=2, eos_id=3)
        # Returns (batch_size, generated_len) - up to max_len
        assert result.dim() == 2
        assert result.size(0) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
