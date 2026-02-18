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
* The `generate` method in `model.py` does not seem to be implemented correctly, although the beam search seems to work fine
* The loss scaling in the training loop should be scaled by the number of tokens in the batch, not the number of batches
* Lots of things are hard-coded and should be configurable
* The `train.py` script should be refactored to use a more modular design
* ... etc


## Usage

```
# Edit Config
vim config.py

# Train
python train.py 

# Average checkpoints and quantize the model
python average_checkpoints.py --k 4 --checkpoint_dir checkpoints --output_prefix model_avg --export_int8 --calib_batches 200

# Convert to CTranslate2 format
python convert_to_ct2.py --model_path model_avg_int8.pt --output_dir ct2_model --quantization int8

# Evaluate (uses quickmt library, https://github.com/quickmt/quickmt)
python evaluate.py --src_file data/flores.fa --ref_file data/flores.en --device cuda --batch_size 16 --beam_size 5 --model ./ct2_model
```
