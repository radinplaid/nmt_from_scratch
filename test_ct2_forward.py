import torch
import numpy as np
import ctranslate2
from convert_to_ct2 import get_layer_weights, set_linear
from safetensors.torch import load_file
from config import load_config
from model import Seq2SeqTransformer

model_cfg, _, _, _ = load_config("configs/faen-small.yaml")
model = Seq2SeqTransformer(model_cfg)
state_dict = load_file("faen-small/averaged_model.safetensors")
model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}, strict=False)
model.eval()

# Let's extract the first layer encoder output
src = torch.tensor([[2, 45, 67, 3]], dtype=torch.long)
with torch.no_grad():
    enc_out_pt = model.encode(src)
print("PyTorch Enc Out shape:", enc_out_pt.shape)
print("PyTorch Enc Out mean:", enc_out_pt.mean().item(), "std:", enc_out_pt.std().item())
print("PyTorch Enc Out first elements:", enc_out_pt[0, 0, :5].numpy())
