"""Language modeling datasets: PTB, WikiText-2, and WikiText-103."""

import os
import pickle
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from tonic.io import make_structured_array

from .collate import Data, DataLoader, ptb_collate_fn, event_stream_dataloader


# ============================================================
# Shared text processing classes
# ============================================================

class Dictionary:
    """Word-to-index dictionary for language modeling."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus:
    """Tokenized text corpus with caching support."""

    def __init__(self, path, cache_dir=None):
        self.dictionary = Dictionary()
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir and (self.cache_dir / 'corpus_cache.pkl').exists():
            with open(self.cache_dir / 'corpus_cache.pkl', 'rb') as f:
                cached_data = pickle.load(f)
                self.dictionary = cached_data['dictionary']
                self.train = cached_data['train']
                self.valid = cached_data['valid']
                self.test = cached_data['test']
        else:
            self.train = self.tokenize(os.path.join(path, 'train.txt'))
            self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
            self.test = self.tokenize(os.path.join(path, 'test.txt'))
            if self.cache_dir:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                with open(self.cache_dir / 'corpus_cache.pkl', 'wb') as f:
                    pickle.dump({
                        'dictionary': self.dictionary,
                        'train': self.train, 'valid': self.valid, 'test': self.test,
                    }, f)

    def tokenize(self, path):
        assert os.path.exists(path)
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        return ids


# ============================================================
# Tonic wrapper classes
# ============================================================

class PTBTonic(Dataset):
    """Penn Treebank dataset for next-token prediction."""
    sensor_size = (1,)

    def __init__(self, data_dir, split="train", cache_dir=None, transform=None, target_transform=None, seq_len=70):
        assert split in ['train', 'val', 'test']
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.seq_len = seq_len

        corpus = Corpus(data_dir, data_dir)
        if split == 'train':
            self.data = corpus.train
        elif split == 'val':
            self.data = corpus.valid
        elif split == 'test':
            self.data = corpus.test
        else:
            raise ValueError

    def __getitem__(self, index):
        bptt = self.seq_len if np.random.random() < 0.95 else self.seq_len / 2
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        seq_len = min(seq_len, len(self.data) - 1 - index)

        input_ids = self.data[index: index + seq_len]
        target = self.data[index + 1: index + 1 + seq_len]

        times = np.arange(0, len(input_ids))
        dtype = np.dtype([("t", int), ("x", int), ("p", int)])
        events = make_structured_array(times, input_ids.numpy(), 1, dtype=dtype)
        target = make_structured_array(times, target.numpy(), 1, dtype=dtype)

        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        return (self.data.size(0) - 1) // self.seq_len


class Wikitext2Tonic(Dataset):
    """WikiText-2 dataset for next-token prediction."""
    sensor_size = (1,)

    def __init__(self, data_dir, split="train", cache_dir=None, transform=None, target_transform=None, seq_len=70):
        assert split in ['train', 'val', 'test']
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.seq_len = seq_len

        corpus = Corpus(data_dir, data_dir)
        if split == 'train':
            self.data = corpus.train
        elif split == 'val':
            self.data = corpus.valid
        elif split == 'test':
            self.data = corpus.test
        else:
            raise ValueError

    def __getitem__(self, index):
        bptt = self.seq_len if np.random.random() < 0.95 else self.seq_len / 2
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        seq_len = min(seq_len, self.seq_len + 20)

        input_ids = self.data[index: index + seq_len]
        target = self.data[index + 1: index + 1 + seq_len]

        times = np.arange(0, len(input_ids))
        dtype = np.dtype([("t", int), ("x", int), ("p", int)])
        events = make_structured_array(times, input_ids.numpy(), 1, dtype=dtype)
        target = make_structured_array(times, target.numpy(), 1, dtype=dtype)

        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        return (self.data.size(0) - 1) // self.seq_len


class Wikitext103Tonic(Dataset):
    """WikiText-103 dataset for next-token prediction."""
    sensor_size = (1,)

    def __init__(self, data_dir, split="train", cache_dir=None, transform=None, target_transform=None, seq_len=140):
        assert split in ['train', 'val', 'test']
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.seq_len = seq_len

        corpus = Corpus(data_dir, data_dir)
        if split == 'train':
            self.data = corpus.train
        elif split == 'val':
            self.data = corpus.valid
        elif split == 'test':
            self.data = corpus.test
        else:
            raise ValueError

    def __getitem__(self, index):
        bptt = self.seq_len if np.random.random() < 0.95 else self.seq_len / 2
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        seq_len = min(seq_len, len(self.data) - 1 - index)

        input_ids = self.data[index: index + seq_len]
        target = self.data[index + 1: index + 1 + seq_len]

        times = np.arange(0, len(input_ids))
        dtype = np.dtype([("t", int), ("x", int), ("p", int)])
        events = make_structured_array(times, input_ids.numpy(), 1, dtype=dtype)
        target = make_structured_array(times, target.numpy(), 1, dtype=dtype)

        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        return (self.data.size(0) - 1) // self.seq_len


# ============================================================
# Factory functions
# ============================================================

def create_ptb_tonic_classification_dataset(
    data_dir=None, cache_dir=None,
    per_device_batch_size=32, per_device_eval_batch_size=64,
    world_size=1, num_workers=0, seed=42,
    pad_unit=32, cut_mix=0.0, no_time_information=False, **kwargs,
):
    """Create dataloaders for the Penn Treebank language modeling task."""
    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    cache_dir = data_dir = Path(cache_dir).parent / "ptb"

    TrainData = partial(PTBTonic, data_dir=data_dir, cache_dir=cache_dir, split='train')
    ValData = partial(PTBTonic, data_dir=data_dir, cache_dir=cache_dir, split='val')
    TestData = partial(PTBTonic, data_dir=data_dir, cache_dir=cache_dir, split='test')

    train_data, val_data, test_data = TrainData(), ValData(), TestData()
    assert len(train_data) > 0 and len(val_data) > 0 and len(test_data) > 0

    collate_fn = partial(ptb_collate_fn, resolution=(10000,), pad_unit=pad_unit, no_time_information=no_time_information)
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=partial(collate_fn, cut_mix=cut_mix), eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True,
    )
    data = Data(n_classes=10000, num_embeddings=10000, train_size=len(train_data))
    return train_loader, val_loader, test_loader, data


def create_wikitext2_tonic_classification_dataset(
    data_dir=None, cache_dir=None,
    per_device_batch_size=32, per_device_eval_batch_size=64,
    world_size=1, num_workers=0, seed=42,
    pad_unit=32, cut_mix=0.0, no_time_information=False, **kwargs,
):
    """Create dataloaders for the WikiText-2 language modeling task."""
    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    cache_dir = data_dir = Path(cache_dir).parent / "wikitext2"

    TrainData = partial(Wikitext2Tonic, data_dir=data_dir, cache_dir=cache_dir, split='train')
    ValData = partial(Wikitext2Tonic, data_dir=data_dir, cache_dir=cache_dir, split='val')
    TestData = partial(Wikitext2Tonic, data_dir=data_dir, cache_dir=cache_dir, split='test')

    train_data, val_data, test_data = TrainData(), ValData(), TestData()
    assert len(train_data) > 0 and len(val_data) > 0 and len(test_data) > 0

    collate_fn = partial(ptb_collate_fn, resolution=(84608,), pad_unit=pad_unit, no_time_information=no_time_information)
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=partial(collate_fn, cut_mix=cut_mix), eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True,
    )
    data = Data(n_classes=84608, num_embeddings=84608, train_size=len(train_data))
    return train_loader, val_loader, test_loader, data


def create_wikitext103_tonic_classification_dataset(
    data_dir=None, cache_dir=None,
    per_device_batch_size=32, per_device_eval_batch_size=64,
    world_size=1, num_workers=0, seed=42,
    pad_unit=32, cut_mix=0.0, no_time_information=False, **kwargs,
):
    """Create dataloaders for the WikiText-103 language modeling task."""
    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    cache_dir = data_dir = Path(cache_dir).parent / "wikitext103"

    TrainData = partial(Wikitext103Tonic, data_dir=data_dir, cache_dir=cache_dir, split='train')
    ValData = partial(Wikitext103Tonic, data_dir=data_dir, cache_dir=cache_dir, split='val')
    TestData = partial(Wikitext103Tonic, data_dir=data_dir, cache_dir=cache_dir, split='test')

    train_data, val_data, test_data = TrainData(), ValData(), TestData()
    assert len(train_data) > 0 and len(val_data) > 0 and len(test_data) > 0

    collate_fn = partial(ptb_collate_fn, resolution=(267735,), pad_unit=pad_unit, no_time_information=no_time_information)
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=partial(collate_fn, cut_mix=cut_mix), eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True,
    )
    data = Data(n_classes=267735, num_embeddings=267735, train_size=len(train_data))
    return train_loader, val_loader, test_loader, data
