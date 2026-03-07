import ctranslate2
import numpy as np

decoder_spec = ctranslate2.specs.TransformerDecoderSpec(
    num_layers=1,
    num_heads=8,
    pre_norm=True,
    num_heads_kv=2,
)

layer = decoder_spec.layer[0]
layer.attention.linear[0].weight = np.zeros((8 * 64, 512), dtype=np.float32)
layer.attention.linear[1].weight = np.zeros((2 * 2 * 64, 512), dtype=np.float32) # GQA
layer.attention.linear[2].weight = np.zeros((8 * 64, 512), dtype=np.float32)

print("Setup complete")
spec = ctranslate2.specs.TransformerSpec(
    ctranslate2.specs.TransformerEncoderSpec(1, 8),
    decoder_spec
)
spec.config.add_source_bos = False
spec.config.add_source_eos = False
spec.register_source_vocabulary(["a", "b", "c"])
spec.register_target_vocabulary(["a", "b", "c"])
try:
    spec.validate()
    print("Validation passed!")
except Exception as e:
    print("Validation failed:", e)
