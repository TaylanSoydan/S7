import os
import torch
import numpy as np
from collections import Counter
import os
import torch
import pickle
import torch
from pathlib import Path

from collections import Counter


class Dictionary(object):
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


class Corpus(object):
    def __init__(self, path, cache_dir=None):
        self.dictionary = Dictionary()
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Check for cached corpus
        if self.cache_dir and (self.cache_dir / 'corpus_cache.pkl').exists():
            with open(self.cache_dir / 'corpus_cache.pkl', 'rb') as f:
                cached_data = pickle.load(f)
                self.dictionary = cached_data['dictionary']
                self.train = cached_data['train']
                self.valid = cached_data['valid']
                self.test = cached_data['test']
        else:
            # Process and cache corpus if not available
            self.train = self.tokenize(os.path.join(path, 'train.txt'))
            self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
            self.test = self.tokenize(os.path.join(path, 'test.txt'))
            if self.cache_dir:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                with open(self.cache_dir / 'corpus_cache.pkl', 'wb') as f:
                    pickle.dump({
                        'dictionary': self.dictionary,
                        'train': self.train,
                        'valid': self.valid,
                        'test': self.test
                    }, f)

    def tokenize(self, path):
        """Tokenizes a text file."""
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


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data
    
def prepare_sequences(data, seq_len):
    """Generate input-target sequence pairs for each batch."""
    num_sequences = (data.size(0) - 1) // seq_len
    sequences = []
    for i in range(num_sequences):
        start_idx = i * seq_len
        input_ids = data[start_idx : start_idx + seq_len]
        target = data[start_idx + 1 : start_idx + 1 + seq_len]
        sequences.append((input_ids, target))
    return sequences

#corpus = Corpus("/data/storage/tsoydan/data/ptb", "/data/storage/tsoydan/data/ptb")
#train_data = corpus.train
#print(train_data)
#print(train_data.shape)

#batch_train_data = batchify(train_data,10)

#print(batch_train_data.shape)

import numpy as np

from tonic.io import make_structured_array
from torch.utils.data import Dataset
from typing import Callable, Optional, TypeVar, Dict, Tuple, List, Union
from pathlib import Path


class PTBTonic(Dataset):
    """PTB dataset for Tonic framework using the Dictionary and Corpus classes for data processing."""

    sensor_size = (1,)  # Adjust based on PTB token representation requirements

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        cache_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        batch_size: int = 32,
        seq_len: int = 70,
    ):
        assert split in ['train', 'val', 'test'], "split must be 'train', 'val', or 'test'"
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.batch_size = batch_size
        self.seq_len = seq_len

        # Load corpus and split-specific data
        corpus = Corpus(data_dir)
        if split == 'train':
            self.data = corpus.train
        elif split == 'val':
            self.data = corpus.valid
        else:
            self.data = corpus.test

    def __getitem__(self, index):
        # Probabilistically determine sequence length
        # Sequence length varies around `self.seq_len` with some randomness
        bptt = self.seq_len if np.random.random() < 0.95 else self.seq_len / 2
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        seq_len = min(seq_len, len(self.data) - 1 - index)

        # Get input and target sequences
        input_ids = self.data[index : index + seq_len]
        target = self.data[index + 1 : index + 1 + seq_len]  # Shifted by one for next-token prediction
        
        # Generate artificial time information (or other metadata if needed)
        times = np.arange(0, len(input_ids))

        # Using a structured array format similar to ListOps
        events = make_structured_array(times, input_ids.numpy(), 1, dtype=np.dtype([("t", int), ("x", int), ("p", int)]))
        target = make_structured_array(times, target.numpy(), 1, dtype=np.dtype([("t", int), ("x", int), ("p", int)]))
        
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target.numpy())
        return events, target

    def __len__(self):
        # Adjust length based on batchified data and average sequence length
        return (self.data.size(0) - 1) // self.seq_len
    
#ptb = PTBTonic("/data/storage/tsoydan/data/ptb","train")
#for i in range(5):
#    events, target = ptb.__getitem__(i)
#    print(events.shape, target.shape)

#print(len(corpus.dictionary))

class Wikitext2Tonic(Dataset):

    """PTB dataset for Tonic framework using the Dictionary and Corpus classes for data processing."""

    sensor_size = (1,)  # Adjust based on PTB token representation requirements

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        cache_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        seq_len: int = 140,
    ):
        assert split in ['train', 'val', 'test'], "split must be 'train', 'val', or 'test'"
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.seq_len = seq_len

        # Load corpus and split-specific data
        corpus = Corpus(data_dir)
        if split == 'train':
            self.data = corpus.train
        elif split == 'val':
            self.data = corpus.valid
        elif split == 'test':
            self.data = corpus.test
        else:
            raise ValueError

    def __getitem__(self, index):
        # Probabilistically determine sequence length
        # Sequence length varies around `self.seq_len` with some randomness
        bptt = self.seq_len if np.random.random() < 0.95 else self.seq_len / 2
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        seq_len = min(seq_len, len(self.data) - 1 - index)

        # Get input and target sequences
        input_ids = self.data[index : index + seq_len]
        target = self.data[index + 1 : index + 1 + seq_len]  # Shifted by one for next-token prediction
        
        # Generate artificial time information (or other metadata if needed)
        times = np.arange(0, len(input_ids))

        # Using a structured array format similar to ListOps
        events = make_structured_array(times, input_ids.numpy(), 1, dtype=np.dtype([("t", int), ("x", int), ("p", int)]))
        target = make_structured_array(times, target.numpy(), 1, dtype=np.dtype([("t", int), ("x", int), ("p", int)]))

        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)


        return events, target

    def __len__(self):
        # Adjust length based on batchified data and average sequence length
        return (self.data.size(0) - 1) // self.seq_len
    
#wt = Wikitext2Tonic("/data/storage/tsoydan/data/wikitext2","train")

corpus = Corpus("/data/storage/tsoydan/data/wikitext2", "/data/storage/tsoydan/data/wikitext2")

print(len(corpus.dictionary))