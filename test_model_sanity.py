import torch
from config import ModelConfig
from model import Seq2SeqTransformer

config = ModelConfig(
    d_model=128,
    enc_layers=2,
    dec_layers=2,
    n_heads=4,
    n_kv_heads=2,
    ffn_dim=256,
    vocab_size=1000
)
model = Seq2SeqTransformer(config)
print("Model initialized successfully")

src = torch.randint(0, 1000, (2, 10))
tgt = torch.randint(0, 1000, (2, 8))

loss, n_tokens = model(src, tgt)
print(f"Forward pass successful. Loss: {loss.item()}, Tokens: {n_tokens}")

# Test generation
gen = model.generate(src, max_len=5)
print(f"Generation successful. Shape: {gen.shape}")
