"""Collate functions and dataloader utilities for all dataset types."""

import numpy as np
import torch
from typing import TypeVar

from src.transform import cut_mix_augmentation

DataLoader = TypeVar('DataLoader')


class Data:
    """Container for dataset-specific metadata."""

    def __init__(self, n_classes: int, num_embeddings: int, train_size: int):
        self.n_classes = n_classes
        self.num_embeddings = num_embeddings
        self.train_size = train_size

def event_stream_collate_fn(
    batch, resolution, pad_unit, cut_mix=0.0,
    no_time_information=False, tokenize="unique",
):
    """
    Collate function for event stream data (e.g., SHD, SSC, DVS).

    Converts raw event data into padded token sequences ready for the JAX model.

    Args:
        batch: List of (events, target) tuples.
        resolution: Spatial resolution of the event stream.
        pad_unit: Sequences are padded to multiples of this value.
        cut_mix: Probability of applying CutMix augmentation.
        no_time_information: If True, ignore timestamps (ablation mode).
        tokenize: Tokenization strategy ("unique" or positional).
    """
    x, y, *z = zip(*batch)
    assert len(z) == 0
    batch_size_one = len(x) == 1

    if np.random.rand() < cut_mix:
        x, y = cut_mix_augmentation(x, y)

    y = np.stack(y)

    if no_time_information:
        timesteps = [np.ones_like(e['t'][:-1]) for e in x]
    else:
        timesteps = [np.diff(e['t']) for e in x]

    if len(resolution) == 1:
        tokens = [e['x'][:-1].astype(np.int32) for e in x]
    elif len(resolution) == 2:
        if tokenize == "unique":
            tokens = [
                (e['x'][:-1].astype(np.int32) * resolution[0] * 2
                 + e['y'][:-1].astype(np.int32) * 2
                 + e['p'][:-1].astype(np.int32))
                for e in x
            ]
        else:
            tokens = [
                (e['x'][:-1] * e['y'][:-1]
                 + np.prod(resolution) * e['p'][:-1].astype(np.int32)).astype(np.int32)
                for e in x
            ]
    else:
        raise ValueError('resolution must contain 1 or 2 elements')

    lengths = np.array([len(e) for e in timesteps], dtype=np.int32)
    pad_length = (lengths.max() // pad_unit) * pad_unit + pad_unit

    tokens = np.stack(
        [np.pad(e, (0, pad_length - len(e)), mode='constant', constant_values=-1) for e in tokens])
    timesteps = np.stack(
        [np.pad(e, (0, pad_length - len(e)), mode='constant', constant_values=0) for e in timesteps])

    timesteps = timesteps / 1000

    if batch_size_one:
        lengths = lengths[None, ...]

    return tokens, y, timesteps, lengths


def lra_text_collate_fn(
    batch, resolution, pad_unit, cut_mix=0.0,
    no_time_information=False, tokenize="unique",
):
    """Collate function for LRA text tasks (IMDB)."""
    x, y, *z = zip(*batch)
    assert len(z) == 0
    batch_size_one = len(x) == 1

    y = np.stack(y)
    timesteps = [np.ones_like(e['t']) for e in x]
    tokens = [e['x'].astype(np.int32) for e in x]

    lengths = np.array([len(e) for e in timesteps], dtype=np.int32)
    pad_length = (lengths.max() // pad_unit) * pad_unit + pad_unit

    tokens = np.stack(
        [np.pad(e, (0, pad_length - len(e)), mode='constant', constant_values=-1) for e in tokens])
    timesteps = np.stack(
        [np.pad(e, (0, pad_length - len(e)), mode='constant', constant_values=0) for e in timesteps])

    timesteps = timesteps / 1

    if batch_size_one:
        lengths = lengths[None, ...]

    return tokens, y, timesteps, lengths


def person_activity_collate_fn(
    batch, resolution, pad_unit, cut_mix=0.0,
    no_time_information=False, tokenize="unique",
):
    """Collate function for person activity and walker datasets."""
    x, y, *z = zip(*batch)
    assert len(z) == 0
    batch_size_one = len(x) == 1

    y = np.stack(y)
    timesteps = np.stack([e['t'] for e in x])
    tokens = np.stack([e['x'] for e in x])

    lengths = np.array([len(e) for e in timesteps], dtype=np.int32)
    timesteps = timesteps / 1

    if batch_size_one:
        lengths = lengths[None, ...]

    return tokens, y, timesteps, lengths


def lra_image_collate_fn(
    batch, resolution, pad_unit, cut_mix=0.0,
    no_time_information=False, tokenize="unique",
):
    """Collate function for LRA image classification (CIFAR10)."""
    x, y, *z = zip(*batch)
    assert len(z) == 0
    batch_size_one = len(x) == 1

    y = np.stack(y)
    timesteps = np.stack([np.ones_like(e['t']) for e in x])
    tokens = np.stack([e['x'].astype(np.int32) for e in x])
    lengths = np.array([len(e) for e in timesteps], dtype=np.int32)

    if batch_size_one:
        lengths = lengths[None, ...]

    return tokens, y, timesteps, lengths


def eigenworms_collate_fn(
    batch, resolution, pad_unit, cut_mix=0.0,
    no_time_information=False, tokenize="unique",
):
    """Collate function for EigenWorms dataset."""
    x, y, *z = zip(*batch)
    assert len(z) == 0
    batch_size_one = len(x) == 1

    y = np.stack(y)
    timesteps = np.stack([np.ones_like(e['t']) for e in x])
    tokens = np.stack([e['x'] for e in x])
    lengths = np.array([len(e) for e in timesteps], dtype=np.int32)

    if batch_size_one:
        lengths = lengths[None, ...]

    return tokens, y, timesteps, lengths


def lra_pathfinder_collate_fn(
    batch, resolution, pad_unit, cut_mix=0.0,
    no_time_information=False, tokenize="unique",
):
    """Collate function for PathFinder / PathX datasets."""
    x, y, *z = zip(*batch)
    assert len(z) == 0
    batch_size_one = len(x) == 1

    y = np.stack(y)
    timesteps = np.stack([np.ones_like(e['t']) for e in x])
    tokens = np.stack([e['x'] for e in x])
    lengths = np.array([len(e) for e in timesteps], dtype=np.int32)

    if batch_size_one:
        lengths = lengths[None, ...]

    return tokens, y, timesteps, lengths


def retrieval_collate_fn(
    batch, resolution, pad_unit, cut_mix=0.0,
    no_time_information=True, tokenize="unique",
):
    """Collate function for AAN retrieval task (dual-input)."""
    xs, y, *z = zip(*batch)
    x1, x2 = zip(*xs)

    assert len(z) == 0
    assert len(x1) == len(x2)
    batch_size_one = len(x1) == 1

    y = np.stack(y)

    if no_time_information:
        timesteps1 = [np.ones_like(e['t'][:-1]) for e in x1]
        timesteps2 = [np.ones_like(e['t'][:-1]) for e in x2]
    else:
        timesteps1 = [np.diff(e['t']) for e in x1]
        timesteps2 = [np.diff(e['t']) for e in x2]

    tokens1 = [e['x'][:-1].astype(np.int32) for e in x1]
    tokens2 = [e['x'][:-1].astype(np.int32) for e in x2]

    lengths1 = np.array([len(e) for e in timesteps1], dtype=np.int32)
    lengths2 = np.array([len(e) for e in timesteps2], dtype=np.int32)
    pad_length = max(
        (lengths1.max() // pad_unit) * pad_unit + pad_unit,
        (lengths2.max() // pad_unit) * pad_unit + pad_unit,
    )

    tokens1 = np.stack(
        [np.pad(e, (0, pad_length - len(e)), mode='constant', constant_values=-1) for e in tokens1])
    tokens2 = np.stack(
        [np.pad(e, (0, pad_length - len(e)), mode='constant', constant_values=-1) for e in tokens2])
    timesteps1 = np.stack(
        [np.pad(e, (0, pad_length - len(e)), mode='constant', constant_values=0) for e in timesteps1])
    timesteps2 = np.stack(
        [np.pad(e, (0, pad_length - len(e)), mode='constant', constant_values=0) for e in timesteps2])

    tokens = np.concatenate((tokens1, tokens2), axis=0)
    timesteps = np.concatenate((timesteps1, timesteps2), axis=0)
    lengths = np.concatenate((lengths1, lengths2), axis=0)

    if batch_size_one:
        lengths = lengths[None, ...]

    return tokens, y, timesteps, lengths


def ptb_collate_fn(
    batch, resolution, pad_unit, cut_mix=0.0,
    no_time_information=False, tokenize="unique",
):
    """Collate function for language modeling tasks (PTB, WikiText)."""
    x, y, *z = zip(*batch)
    assert len(z) == 0
    batch_size_one = len(x) == 1

    timesteps = [np.ones_like(e['t']) for e in x]
    tokens = [e['x'].astype(np.int32) for e in x]
    targets = [e['x'].astype(np.int32) for e in y]

    tokens_lengths = np.array([len(e) for e in tokens], dtype=np.int32)
    targets_lengths = np.array([len(t) for t in targets], dtype=np.int32)

    tokens_pad_length = (tokens_lengths.max() // pad_unit) * pad_unit + pad_unit
    targets_pad_length = (targets_lengths.max() // pad_unit) * pad_unit + pad_unit
    pad_length = max(tokens_pad_length, targets_pad_length)

    tokens = np.stack(
        [np.pad(e, (0, pad_length - len(e)), mode='constant', constant_values=-1) for e in tokens])
    targets = np.stack(
        [np.pad(t, (0, pad_length - len(t)), mode='constant', constant_values=-1) for t in targets])
    timesteps = np.stack(
        [np.pad(e, (0, pad_length - len(e)), mode='constant', constant_values=0) for e in timesteps])

    timesteps = timesteps / 1

    if batch_size_one:
        tokens_lengths = tokens_lengths[None, ...]

    return tokens, targets, timesteps, tokens_lengths


# --- Dataloader Creation Utilities ---

def event_stream_dataloader(
    train_data, val_data, test_data,
    batch_size, eval_batch_size,
    train_collate_fn, eval_collate_fn,
    rng, num_workers=0, shuffle_training=True,
):
    """Create train/val/test DataLoaders with given collate functions."""
    def dataloader(dset, bsz, collate_fn, shuffle, drop_last):
        return torch.utils.data.DataLoader(
            dset, batch_size=bsz, drop_last=drop_last,
            collate_fn=collate_fn, shuffle=shuffle,
            generator=rng, num_workers=num_workers,
        )

    train_loader = dataloader(train_data, batch_size, train_collate_fn, shuffle=shuffle_training, drop_last=True)
    val_loader = dataloader(val_data, eval_batch_size, eval_collate_fn, shuffle=False, drop_last=True)
    test_loader = dataloader(test_data, eval_batch_size, eval_collate_fn, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader


def event_stream_dataloader_image(
    train_data, val_data, test_data,
    batch_size, eval_batch_size,
    train_collate_fn, eval_collate_fn,
    rng, num_workers=0, shuffle_training=True,
):
    """Create train/val/test DataLoaders for image tasks (no drop_last on eval)."""
    def dataloader(dset, bsz, collate_fn, shuffle, drop_last):
        return torch.utils.data.DataLoader(
            dset, batch_size=bsz, drop_last=drop_last,
            collate_fn=collate_fn, shuffle=shuffle,
            generator=rng, num_workers=num_workers,
        )

    train_loader = dataloader(train_data, batch_size, train_collate_fn, shuffle=shuffle_training, drop_last=True)
    val_loader = dataloader(val_data, eval_batch_size, eval_collate_fn, shuffle=False, drop_last=False)
    test_loader = dataloader(test_data, eval_batch_size, eval_collate_fn, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader


def event_stream_dataloader_parallel(
    train_data, val_data, test_data,
    batch_size, eval_batch_size,
    train_collate_fn, eval_collate_fn,
    rng, num_workers=0, shuffle_training=True,
):
    """Create train/val/test DataLoaders for multi-GPU training (drop_last on all)."""
    def dataloader(dset, bsz, collate_fn, shuffle, drop_last):
        return torch.utils.data.DataLoader(
            dset, batch_size=bsz, drop_last=drop_last,
            collate_fn=collate_fn, shuffle=shuffle,
            generator=rng, num_workers=num_workers,
        )

    train_loader = dataloader(train_data, batch_size, train_collate_fn, shuffle=shuffle_training, drop_last=True)
    val_loader = dataloader(val_data, eval_batch_size, eval_collate_fn, shuffle=False, drop_last=True)
    test_loader = dataloader(test_data, eval_batch_size, eval_collate_fn, shuffle=False, drop_last=True)
    return train_loader, val_loader, test_loader
