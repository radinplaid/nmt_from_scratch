import os
import torch
import itertools
import random
from torch.utils.data import DataLoader, IterableDataset


class StreamingTextDataset(IterableDataset):
    """
    Streaming dataset using IterableDataset to handle large files.
    Includes bucketing logic to support dynamic batching (token-based).
    """

    def __init__(
        self,
        src_file: str,
        tgt_file: str,
        src_sp,
        tgt_sp,
        max_tokens: int,
        buffer_size: int = 10000,
    ):
        self.src_file = src_file
        self.tgt_file = tgt_file
        self.src_sp = src_sp
        self.tgt_sp = tgt_sp
        self.max_tokens = max_tokens
        self.buffer_size = max(
            buffer_size, 20000
        )  # Ensure minimum buffer size for better shuffling

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        with (
            open(self.src_file, "r", encoding="utf-8") as f_src,
            open(self.tgt_file, "r", encoding="utf-8") as f_tgt,
        ):
            pairs = zip(f_src, f_tgt)

            if worker_info is not None:
                # Sharding: each worker reads unique lines
                pairs = itertools.islice(
                    pairs, worker_info.id, None, worker_info.num_workers
                )

            # Generator for tokenized samples
            def get_samples():
                for s, t in pairs:
                    # Limit to prevent Positional Encoding overflow or OOM
                    s_ids = self.src_sp.encode(
                        s.strip(), out_type=int, add_bos=True, add_eos=True
                    )
                    t_ids = self.tgt_sp.encode(
                        t.strip(), out_type=int, add_bos=True, add_eos=True
                    )
                    if len(s_ids) <= 256 and len(t_ids) <= 256:
                        yield torch.tensor(s_ids), torch.tensor(t_ids)

            samples = get_samples()

            while True:
                # 1. Fill buffer
                buffer = list(itertools.islice(samples, self.buffer_size))
                if not buffer:
                    break

                # 2. Local shuffle for randomness
                random.shuffle(buffer)

                # 3. Sort by length to minimize padding
                buffer.sort(key=lambda x: max(len(x[0]), len(x[1])))

                # 4. Create batches based on token budget
                batches = []
                batch_srcs, batch_tgts = [], []
                max_len_in_batch = 0

                for s, t in buffer:
                    length = max(len(s), len(t))
                    new_max_len = max(max_len_in_batch, length)
                    new_cost = (len(batch_srcs) + 1) * new_max_len

                    if new_cost > self.max_tokens and batch_srcs:
                        batches.append(self._collate(batch_srcs, batch_tgts))
                        batch_srcs, batch_tgts = [], []
                        max_len_in_batch = 0

                    batch_srcs.append(s)
                    batch_tgts.append(t)
                    max_len_in_batch = max(max_len_in_batch, length)

                if batch_srcs:
                    batches.append(self._collate(batch_srcs, batch_tgts))

                # 5. Shuffle the created batches to eliminate length bias
                random.shuffle(batches)
                for b_src, b_tgt in batches:
                    yield b_src, b_tgt

    def _collate(self, srcs, tgts):
        src_padded = torch.nn.utils.rnn.pad_sequence(
            srcs, batch_first=True, padding_value=0
        )
        tgt_padded = torch.nn.utils.rnn.pad_sequence(
            tgts, batch_first=True, padding_value=0
        )

        # Pad to multiple of 16 for Tensor Core efficiency
        # For src: straight multiple of 16
        def pad_to_multiple(tensor, multiple=16, extra=0):
            seq_len = tensor.size(1)
            target_len = ((seq_len + multiple - 1) // multiple) * multiple + extra
            padding = target_len - seq_len
            if padding > 0:
                tensor = torch.nn.functional.pad(tensor, (0, padding))
            return tensor

        src_padded = pad_to_multiple(src_padded, 16, extra=0)
        # For tgt: multiple of 16. Slice will be 8N-1. Safe and standard.
        tgt_padded = pad_to_multiple(tgt_padded, 16, extra=0)

        return src_padded, tgt_padded


def collate_fn(batch):
    """
    Custom collate to pad to the max length *in this batch*.
    Ensures length is a multiple of 16 for Tensor Core efficiency.
    batch is list of (src_tensor, tgt_tensor)
    """
    srcs, tgts = zip(*batch)

    # Pad sequences
    src_padded = torch.nn.utils.rnn.pad_sequence(
        list(srcs), batch_first=True, padding_value=0
    )
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        list(tgts), batch_first=True, padding_value=0
    )

    # Pad to multiple of 16
    def pad_to_multiple(tensor, multiple=16):
        seq_len = tensor.size(1)
        remainder = seq_len % multiple
        if remainder != 0:
            padding = multiple - remainder
            tensor = torch.nn.functional.pad(tensor, (0, padding))
        return tensor

    src_padded = pad_to_multiple(src_padded)
    tgt_padded = pad_to_multiple(tgt_padded)

    return src_padded, tgt_padded


# ... tokenizer code ...
def train_tokenizer(text_file: str, model_prefix: str, vocab_size: int):
    import sentencepiece as spm

    # Inspect data size to prevent SentencePiece error on tiny data
    with open(text_file, "r") as f:
        line_count = sum(1 for _ in f)

    real_vocab_size = vocab_size
    if line_count < vocab_size:
        real_vocab_size = max(10, line_count * 5)
        if real_vocab_size > vocab_size:
            real_vocab_size = vocab_size

    spm.SentencePieceTrainer.train(
        input=text_file,
        model_prefix=model_prefix,
        vocab_size=real_vocab_size,
        character_coverage=0.9999,
        model_type="unigram",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        byte_fallback=True,
        input_sentence_size=5_000_000,
        shuffle_input_sentence=True,
    )
    print(f"Tokenizer trained: {model_prefix}.model")


def load_tokenizers(src_prefix: str, tgt_prefix: str):
    import sentencepiece as spm

    src_sp = spm.SentencePieceProcessor()
    src_sp.load(f"{src_prefix}.model")
    tgt_sp = spm.SentencePieceProcessor()
    tgt_sp.load(f"{tgt_prefix}.model")
    return src_sp, tgt_sp


def get_dummy_data():
    return (
        [
            "Das ist ein Haus.",
            "Dies ist ein Test.",
            "Deep Learning ist mächtig.",
            "Ich liebe Programmieren.",
            "Die Katze schläft auf dem Sofa.",
            "Guten Morgen, wie geht es Ihnen?",
            "Eins, zwei, drei, vier.",
            "Das Wetter ist heute schön.",
            "Maschinelles Lernen verändert die Welt.",
            "Bitte übersetzen Sie diesen Satz.",
        ],
        [
            "That is a house.",
            "This is a test.",
            "Deep Learning is powerful.",
            "I love programming.",
            "The cat is sleeping on the sofa.",
            "Good morning, how are you?",
            "One, two, three, four.",
            "The weather is nice today.",
            "Machine learning is changing the world.",
            "Please translate this sentence.",
        ],
    )


def load_file_lines(path, limit=None):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            lines.append(line.strip())
    return lines


def PrepareData(config):
    # 1. Train Tokenizers (if not exists)
    vocab_size = config.vocab_size
    model_prefix_src = "tokenizer_src"
    model_prefix_tgt = "tokenizer_tgt"

    if not os.path.exists(f"{model_prefix_src}.model"):
        print("Training Source Tokenizer...")
        train_tokenizer(config.src_train_path, model_prefix_src, vocab_size)

    if not os.path.exists(f"{model_prefix_tgt}.model"):
        print("Training Target Tokenizer...")
        train_tokenizer(config.tgt_train_path, model_prefix_tgt, vocab_size)

    # 2. Load Tokenizers
    src_sp, tgt_sp = load_tokenizers(model_prefix_src, model_prefix_tgt)

    # 3. Create Streaming Datasets
    print("Initializing Streaming Datasets...")
    max_tokens = getattr(config, "max_tokens_per_batch", 1024)

    train_dataset = StreamingTextDataset(
        config.src_train_path,
        config.tgt_train_path,
        src_sp,
        tgt_sp,
        max_tokens,
        buffer_size=config.buffer_size,
    )

    dev_dataset = StreamingTextDataset(
        config.src_dev_path,
        config.tgt_dev_path,
        src_sp,
        tgt_sp,
        max_tokens,
        buffer_size=config.buffer_size // 10,  # Smaller buffer for dev
    )

    # 4. Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=8
        if config.num_workers > 0
        else None,  # Increase prefetch for higher throughput
        persistent_workers=True if config.num_workers > 0 else False,
        multiprocessing_context="spawn" if config.num_workers > 0 else None,
    )

    # We use num_workers=0 for dev for simplicity/stability in metrics calculation
    # Disable pin_memory for validation to reduce memory usage
    dev_loader = DataLoader(
        dev_dataset, batch_size=None, num_workers=0, pin_memory=False
    )

    return train_loader, dev_loader, src_sp, tgt_sp
