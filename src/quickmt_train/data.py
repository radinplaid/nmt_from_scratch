import os
import torch
import itertools
import random
from torch.utils.data import DataLoader, IterableDataset
import sentencepiece as spm


class StreamingTextDataset(IterableDataset):
    """
    Streaming dataset using IterableDataset to handle large files.
    Includes bucketing logic to support dynamic batching (token-based).
    """

    def __init__(
        self,
        corpora,
        src_sp,
        tgt_sp,
        max_tokens: int,
        buffer_size: int = 10000,
        max_seq_len: int = 512,
        pad_id: int = 0,
        pad_multiple: int = 16,
        global_step_value=None,
        infinite: bool = True,
        src_spm_alpha: float = 1.0,
        tgt_spm_alpha: float = 1.0,
        src_spm_nbest_size: int = 0,
        tgt_spm_nbest_size: int = 0,
    ):
        self.corpora = corpora
        self.src_sp = src_sp
        self.tgt_sp = tgt_sp
        self.max_tokens = max_tokens
        self.buffer_size = max(
            buffer_size, 20000
        )  # Ensure minimum buffer size for better shuffling
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        self.pad_multiple = pad_multiple
        self.global_step_value = global_step_value
        self.infinite = infinite
        self.src_spm_alpha = src_spm_alpha
        self.tgt_spm_alpha = tgt_spm_alpha
        self.src_spm_nbest_size = src_spm_nbest_size
        self.tgt_spm_nbest_size = tgt_spm_nbest_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        # Generator for tokenized samples
        def get_samples():
            iters = []
            files_list = []

            def init_iter(c):
                f_src = open(c.src_file, "r", encoding="utf-8")
                f_tgt = open(c.tgt_file, "r", encoding="utf-8")
                pair_iter = zip(f_src, f_tgt)
                if worker_info is not None:
                    pair_iter = itertools.islice(
                        pair_iter, worker_info.id, None, worker_info.num_workers
                    )
                return f_src, f_tgt, pair_iter

            for c in self.corpora:
                f_src, f_tgt, p_iter = init_iter(c)
                iters.append(p_iter)
                files_list.append((f_src, f_tgt))

            try:
                logged_corpora = set()
                active_corpora = set(range(len(self.corpora)))
                corpus_epochs = {i: 0 for i in range(len(self.corpora))}

                while active_corpora:
                    current_step = (
                        self.global_step_value.value if self.global_step_value else 0
                    )

                    available_indices = []
                    to_remove = []
                    for i in active_corpora:
                        if current_step >= self.corpora[i].stop_step:
                            to_remove.append(i)
                        elif current_step >= self.corpora[i].start_step:
                            available_indices.append(i)

                    if to_remove:
                        for i in to_remove:
                            if i in active_corpora:
                                active_corpora.remove(i)
                                is_main_worker = (
                                    True
                                    if worker_info is None or worker_info.id == 0
                                    else False
                                )
                                if is_main_worker:
                                    import datetime

                                    timestamp = datetime.datetime.now().strftime(
                                        "%H:%M:%S"
                                    )
                                    print(
                                        f"[{timestamp}] Corpus stopped (reached stop_step): {self.corpora[i].src_file} (Stop step: {self.corpora[i].stop_step})"
                                    )
                                # Close files
                                try:
                                    f_src, f_tgt = files_list[i]
                                    f_src.close()
                                    f_tgt.close()
                                except Exception:
                                    pass

                    if not available_indices:
                        if not self.infinite or not active_corpora:
                            break
                        import time

                        time.sleep(1)
                        continue

                    for i in available_indices:
                        if i not in logged_corpora:
                            is_main_worker = (
                                True
                                if worker_info is None or worker_info.id == 0
                                else False
                            )
                            if is_main_worker:
                                import datetime

                                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                                print(
                                    f"[{timestamp}] Active corpus: {self.corpora[i].src_file} (Weight: {self.corpora[i].weight}, Start step: {self.corpora[i].start_step}, Stop step: {self.corpora[i].stop_step})"
                                )
                            logged_corpora.add(i)

                    schedule = []
                    for i in available_indices:
                        schedule.extend([i] * self.corpora[i].weight)
                    random.shuffle(schedule)

                    for c_idx in schedule:
                        if c_idx not in active_corpora:
                            continue

                        try:
                            s, t = next(iters[c_idx])
                        except StopIteration:
                            if not self.infinite:
                                active_corpora.remove(c_idx)
                                continue

                            corpus_epochs[c_idx] += 1
                            is_main_worker = (
                                True
                                if worker_info is None or worker_info.id == 0
                                else False
                            )
                            if is_main_worker:
                                import datetime

                                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                                print(
                                    f"[{timestamp}] Corpus {self.corpora[c_idx].src_file} completed epoch {corpus_epochs[c_idx]} (Weight: {self.corpora[c_idx].weight}, loop restarting)"
                                )

                            # Close previous iter files
                            old_f_src, old_f_tgt = files_list[c_idx]
                            try:
                                old_f_src.close()
                                old_f_tgt.close()
                            except Exception:
                                pass

                            f_src, f_tgt, p_iter = init_iter(self.corpora[c_idx])
                            iters[c_idx] = p_iter
                            files_list[c_idx] = (f_src, f_tgt)

                            try:
                                s, t = next(iters[c_idx])
                            except StopIteration:
                                active_corpora.remove(c_idx)
                                continue

                        s_ids = self.src_sp.encode(
                            s.strip(),
                            out_type=int,
                            add_bos=True,
                            add_eos=True,
                            alpha=self.src_spm_alpha,
                            nbest_size=self.src_spm_nbest_size,
                        )
                        t_ids = self.tgt_sp.encode(
                            t.strip(),
                            out_type=int,
                            add_bos=True,
                            add_eos=True,
                            alpha=self.tgt_spm_alpha,
                            nbest_size=self.tgt_spm_nbest_size,
                        )

                        if (
                            len(s_ids) <= self.max_seq_len
                            and len(t_ids) <= self.max_seq_len
                        ):
                            yield torch.tensor(s_ids), torch.tensor(t_ids)
            finally:
                for f_src, f_tgt in files_list:
                    try:
                        f_src.close()
                        f_tgt.close()
                    except Exception:
                        pass

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
            srcs, batch_first=True, padding_value=self.pad_id
        )
        tgt_padded = torch.nn.utils.rnn.pad_sequence(
            tgts, batch_first=True, padding_value=self.pad_id
        )

        # Pad to multiple for Tensor Core efficiency
        def pad_to_multiple(tensor, multiple=16, extra=0):
            seq_len = tensor.size(1)
            target_len = ((seq_len + multiple - 1) // multiple) * multiple + extra
            padding = target_len - seq_len
            if padding > 0:
                tensor = torch.nn.functional.pad(
                    tensor, (0, padding), value=self.pad_id
                )
            return tensor

        src_padded = pad_to_multiple(src_padded, self.pad_multiple, extra=0)
        tgt_padded = pad_to_multiple(tgt_padded, self.pad_multiple, extra=0)

        return src_padded, tgt_padded


def collate_fn(batch):
    """
    Custom collate to pad to the max length *in this batch*.
    Ensures length is a multiple of 16 for efficiency.
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


def train_tokenizer(
    text_file: str,
    model_prefix: str,
    vocab_size: int,
    char_coverage: float,
    input_sentence_size: int,
    pad_id: int = 0,
    unk_id: int = 1,
    bos_id: int = 2,
    eos_id: int = 3,
):

    spm.SentencePieceTrainer.train(
        input=text_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=char_coverage,
        model_type="unigram",
        pad_id=pad_id,
        unk_id=unk_id,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        byte_fallback=True,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
    )
    print(f"Tokenizer trained: {model_prefix}.model")


def load_tokenizers(
    src_prefix: str,
    tgt_prefix: str,
    expected_src_vocab_size: int = None,
    expected_tgt_vocab_size: int = None,
):

    src_sp = spm.SentencePieceProcessor()
    src_sp.load(f"{src_prefix}.model")
    tgt_sp = spm.SentencePieceProcessor()
    tgt_sp.load(f"{tgt_prefix}.model")

    if expected_src_vocab_size is not None:
        if src_sp.get_piece_size() != expected_src_vocab_size:
            raise ValueError(
                f"Source Vocabulary size mismatch: expected {expected_src_vocab_size}, got {src_sp.get_piece_size()}"
            )
    if expected_tgt_vocab_size is not None:
        if tgt_sp.get_piece_size() != expected_tgt_vocab_size:
            raise ValueError(
                f"Target Vocabulary size mismatch: expected {expected_tgt_vocab_size}, got {tgt_sp.get_piece_size()}"
            )

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


def PrepareData(model_cfg, data_cfg, train_cfg, global_step_value=None):
    from .config import CorpusConfig

    # 1. Train Tokenizers (if not exists)
    vocab_size_src = model_cfg.vocab_size_src
    vocab_size_tgt = model_cfg.vocab_size_tgt
    model_prefix_src = data_cfg.tokenizer_prefix_src
    model_prefix_tgt = data_cfg.tokenizer_prefix_tgt

    os.makedirs(data_cfg.experiment_name, exist_ok=True)

    if not data_cfg.corpora:
        raise ValueError("No corpora provided in data_cfg")

    tokenizer_train_src = data_cfg.corpora[0].src_file
    tokenizer_train_tgt = data_cfg.corpora[0].tgt_file

    if not os.path.exists(f"{model_prefix_src}.model"):
        print("Training Source Tokenizer...")
        train_tokenizer(
            tokenizer_train_src,
            model_prefix_src,
            vocab_size_src,
            char_coverage=data_cfg.char_coverage,
            input_sentence_size=data_cfg.input_sentence_size,
            pad_id=model_cfg.pad_id,
            unk_id=model_cfg.unk_id,
            bos_id=model_cfg.bos_id,
            eos_id=model_cfg.eos_id,
        )

    if not os.path.exists(f"{model_prefix_tgt}.model"):
        print("Training Target Tokenizer...")
        train_tokenizer(
            tokenizer_train_tgt,
            model_prefix_tgt,
            vocab_size_tgt,
            char_coverage=data_cfg.char_coverage,
            input_sentence_size=data_cfg.input_sentence_size,
            pad_id=model_cfg.pad_id,
            unk_id=model_cfg.unk_id,
            bos_id=model_cfg.bos_id,
            eos_id=model_cfg.eos_id,
        )

    # 2. Load Tokenizers
    src_sp, tgt_sp = load_tokenizers(
        model_prefix_src,
        model_prefix_tgt,
        expected_src_vocab_size=vocab_size_src,
        expected_tgt_vocab_size=vocab_size_tgt,
    )

    # 3. Create Streaming Datasets
    print("Initializing Streaming Datasets...")
    max_tokens = data_cfg.max_tokens_per_batch

    train_dataset = StreamingTextDataset(
        data_cfg.corpora,
        src_sp,
        tgt_sp,
        max_tokens,
        buffer_size=data_cfg.buffer_size,
        max_seq_len=model_cfg.max_len,
        pad_id=model_cfg.pad_id,
        pad_multiple=data_cfg.pad_multiple,
        global_step_value=global_step_value,
        src_spm_nbest_size=data_cfg.src_spm_nbest_size,
        tgt_spm_nbest_size=data_cfg.tgt_spm_nbest_size,
        src_spm_alpha=data_cfg.src_spm_alpha,
        tgt_spm_alpha=data_cfg.tgt_spm_alpha,
    )

    dev_corpora = [
        CorpusConfig(src_file=data_cfg.src_dev_path, tgt_file=data_cfg.tgt_dev_path)
    ]
    dev_dataset = StreamingTextDataset(
        dev_corpora,
        src_sp,
        tgt_sp,
        max_tokens,
        buffer_size=data_cfg.buffer_size // 10,  # Smaller buffer for dev
        max_seq_len=model_cfg.max_len,
        pad_id=model_cfg.pad_id,
        pad_multiple=data_cfg.pad_multiple,
        infinite=False,
    )

    # 4. Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
        prefetch_factor=data_cfg.prefetch_factor if data_cfg.num_workers > 0 else None,
        persistent_workers=True if data_cfg.num_workers > 0 else False,
        multiprocessing_context="spawn" if data_cfg.num_workers > 0 else None,
    )

    dev_loader = DataLoader(
        dev_dataset, batch_size=None, num_workers=0, pin_memory=False
    )

    return train_loader, dev_loader, src_sp, tgt_sp
