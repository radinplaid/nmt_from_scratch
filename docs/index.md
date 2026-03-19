# Welcome to quickmt-train

Experimenting with training Neural Machine Translation (NMT) models from scratch using PyTorch.

!!! note
    This project is a work in progress and is intended for experimentation and learning.

## Key Features

### 🚀 Performance & Optimization

- **`torch.compile`**: for faster training
- **Mixed Precision (AMP)**: Uses `torch.autocast` with `bfloat16` or `float16` for faster training and reduced memory usage
- **Gradient Accumulation & Clipping**: Support for large effective batch sizes and stable training via gradient norm scaling

### 📊 Data Processing

- **Streaming Dataset**: `IterableDataset` implementation for handling datasets larger than RAM
- **Token-Based Batching**: Dynamic batching with bucket sorting to minimize padding and maximize throughput
- **SentencePiece Tokenization**: Integrated support for training and on-the-fly SentencePiece (unigram/BPE) tokenization
- **Multi-worker Sharding**: Efficient data loading with automatic sharding across multiple CPU workers
- **Multi-dataset training**: Train on multiple datasets at once starting/stopping at specific steps

### 📈 Evaluation & Monitoring

- **Real-time Logging**: Tracking of Loss, Perplexity (PPL), Token Accuracy etc.
- **Translation Quality**: In-training evaluation using **BLEU** and **ChrF** scores via `sacrebleu`
- **Aim Tracking**: Integration with `aim` for experiment tracking and visualization
- **Hyperparameter Optimization**: Integration with `optuna` for hyperparameter optimization

### 🛠️ Inference & Deployment

- **Model Averaging**: Tool for stochastic weight averaging of multiple checkpoints to improve generalization
- **CTranslate2 Export**: Script to convert PyTorch models to CTranslate2 format for production deployment
- **quickmt compatible**: Models can be used with the quickmt library for inference
