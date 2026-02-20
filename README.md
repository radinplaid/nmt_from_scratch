# NMT Model Training From Scratch 

Experimenting with training Neural Machine Translation (NMT) models from scratch using PyTorch.

## Key Features

### 🏛️ Model Architecture
- **Vanilla Transformer**: Implementation based on `nn.Transformer` with configurable layers, heads, and dimensions
- **Pre-Norm Configuration**: Uses `norm_first=True` for better training stability and performance
- **GELU Activation**: Employs Gaussian Error Linear Units for non-linearities
- **Positional Encoding**: Sinusoidal positional embeddings for sequence awareness
- **Weight Sharing**: Support for tying token embeddings and generator weights

### 🚀 Performance & Optimization

- **`torch.compile()`**: Leverages the latest PyTorch compiler for training speedups
- **Mixed Precision (AMP)**: Uses `torch.autocast` with `bfloat16` or `float16` for faster training and reduced memory usage
- **Gradient Accumulation & Clipping**: Support for large effective batch sizes and stable training via gradient norm scaling

### 📊 Data Processing

- **Streaming Dataset**: `IterableDataset` implementation for handling datasets larger than RAM
- **Token-Based Batching**: Dynamic batching with bucket sorting to minimize padding and maximize throughput
- **SentencePiece Tokenization**: Integrated support for training and using SentencePiece (unigram/BPE) models
- **Multi-worker Sharding**: Efficient data loading with automatic sharding across multiple CPU workers

### 📈 Evaluation & Monitoring

- **Real-time Metrics**: Tracking of Loss, Perplexity (PPL), and Token Accuracy
- **Translation Quality**: In-training evaluation using **BLEU** and **ChrF** scores via `sacrebleu`
- **Aim Tracking**: Full integration with `aim` for experiment tracking and visualization

### 🛠️ Inference & Deployment

- **CTranslate2 Export**: Script to convert PyTorch models to highly optimized CTranslate2 format for production deployment
- **Model Averaging**: Tool for stochastic weight averaging of multiple checkpoints to improve generalization

## Dependencies

- torch
- sentencepiece
- sacrebleu
- aim
- ctranslate2
- quickmt


## TODO

This is a start but there is still some work to be done:

* Validation metrics do not seem to be calculated correctly
* The `generate` and `beam_search` methods in `model.py` does not seem to be implemented correctly
* All model files (tokenizers vocab files etc) should be stored in a single model run directory (maybe named after the experiment)
* Rather than truncating inputs that are too long, the dataloader should probably drop/ignore them
* Add code to resume training from checkpoint
* ... etc


## Usage

```bash
# Edit config to your liking
vim configs/faen-tiny.yaml

# Train
python train.py --config configs/faen-tiny.yaml 

# Average checkpoints and quantize the model
python average_checkpoints.py --config configs/faen-tiny.yaml

# Convert to CTranslate2 format
python convert_to_ct2.py --config configs/faen-tiny.yaml

# Evaluate (uses quickmt library, https://github.com/quickmt/quickmt)
python evaluate.py --src_file data/flores.fa --ref_file data/flores.en --device cuda --batch_size 8 --beam_size 5 --model ./faen-small/exported_model
```
