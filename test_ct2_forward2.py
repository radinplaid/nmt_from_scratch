import ctranslate2
import numpy as np

translator = ctranslate2.Translator("faen-small/exported_model", device="cpu", compute_type="float32")
# CTranslate2 does not have a direct forward_encoder method exposed to python API in Translator
# Wait, Generator has forward_encoder. Let's see if Translator has it.
print(dir(translator))
