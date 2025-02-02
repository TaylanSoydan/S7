import os, sys
from pathlib import Path
import torch
from typing import Callable, Optional, TypeVar, Dict, Tuple, List, Union
import tonic
from functools import partial
import numpy as np
from event_ssm.transform import Identity, Roll, Rotate, Scale, DropEventChunk, Jitter1D, OneHotLabels, cut_mix_augmentation

DEFAULT_CACHE_DIR_ROOT = Path('./cache_dir/')

DataLoader = TypeVar('DataLoader')
InputType = [str, Optional[int], Optional[int]]


class Data:
    """
    Data class for storing dataset specific information
    """
    def __init__(
            self,
            n_classes: int,
            num_embeddings: int,
            train_size: int
):
        self.n_classes = n_classes
        self.num_embeddings = num_embeddings
        self.train_size = train_size


def event_stream_collate_fn(batch, resolution, pad_unit, cut_mix=0.0, no_time_information=False, tokenize="unique"):
    """
    Collate function to turn event stream data into tokens ready for the JAX model

    :param batch: list of tuples of (events, target)
    :param resolution: resolution of the event stream
    :param pad_unit: padding unit for the tokens. All sequences will be padded to integer multiples of this unit.
                     This option results in JAX compiling multiple GPU kernels for different sequence lengths,
                     which might slow down compilation time, but improves throughput for the rest of the training process.
    :param cut_mix: probability of applying cut mix augmentation
    :param no_time_information: if True, the time information is ignored and all events are treated as if they were
                                recorded sampled at uniform time intervals.
                                This option is only used for ablation studies.
    """
    # x are inputs, y are targets, z are aux data
    x, y, *z = zip(*batch)
    assert len(z) == 0
    batch_size_one = len(x) == 1

    # apply cut mix augmentation
    if np.random.rand() < cut_mix:
        x, y = cut_mix_augmentation(x, y)

    # set labels to numpy array
    y = np.stack(y)

    # integration time steps are the difference between two consequtive time stamps
    if no_time_information:
        timesteps = [np.ones_like(e['t'][:-1]) for e in x]
    else:
        timesteps = [np.diff(e['t']) for e in x]

    # NOTE: since timesteps are deltas, their length is L - 1, and we have to remove the last token in the following

    # process tokens for single input dim (e.g. audio)
    if len(resolution) == 1:
        tokens = [e['x'][:-1].astype(np.int32) for e in x]
    elif len(resolution) == 2:
        if tokenize == "unique":
            tokens = [(e['x'][:-1].astype(np.int32) * resolution[0] * 2 + e['y'][:-1].astype(np.int32) * 2 + e['p'][:-1].astype(np.int32)) for e in x]
        else:  
            tokens = [(e['x'][:-1] * e['y'][:-1] + np.prod(resolution) * e['p'][:-1].astype(np.int32)).astype(np.int32) for e in x]
            #assert 1==2
    else:
        raise ValueError('resolution must contain 1 or 2 elements')

    # get padding lengths
    lengths = np.array([len(e) for e in timesteps], dtype=np.int32)
    pad_length = (lengths.max() // pad_unit) * pad_unit + pad_unit


    # pad tokens with -1, which results in a zero vector with embedding look-ups
    tokens = np.stack(
        [np.pad(e, (0, pad_length - len(e)), mode='constant', constant_values=-1) for e in tokens])
    timesteps = np.stack(
        [np.pad(e, (0, pad_length - len(e)), mode='constant', constant_values=0) for e in timesteps])

    # timesteps are in micro seconds... transform to milliseconds
    timesteps = timesteps / 1000

    if batch_size_one:
        lengths = lengths[None, ...]

    return tokens, y, timesteps, lengths

def lra_text_collate_fn(batch, resolution, pad_unit, cut_mix=0.0, no_time_information=False, tokenize="unique"):
    # x are inputs, y are targets, z are aux data
    x, y, *z = zip(*batch)
    assert len(z) == 0
    batch_size_one = len(x) == 1

    # set labels to numpy array
    y = np.stack(y)

    # integration time steps are the difference between two consequtive time stamps
    timesteps = [np.ones_like(e['t']) for e in x]


    # NOTE: since timesteps are deltas, their length is L - 1, and we have to remove the last token in the following

    # process tokens for single input dim (e.g. audio)
    tokens = [e['x'].astype(np.int32) for e in x]

    # get padding lengths
    lengths = np.array([len(e) for e in timesteps], dtype=np.int32)
    pad_length = (lengths.max() // pad_unit) * pad_unit + pad_unit

    # pad tokens with -1, which results in a zero vector with embedding look-ups
    tokens = np.stack(
        [np.pad(e, (0, pad_length - len(e)), mode='constant', constant_values=-1) for e in tokens])
    timesteps = np.stack(
        [np.pad(e, (0, pad_length - len(e)), mode='constant', constant_values=0) for e in timesteps])

    timesteps = timesteps / 1

    if batch_size_one:
        lengths = lengths[None, ...]

    return tokens, y, timesteps, lengths

def person_activity_collate_fn(batch, resolution, pad_unit, cut_mix=0.0, no_time_information=False, tokenize="unique"):
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

def lra_image_collate_fn(batch, resolution, pad_unit, cut_mix=0.0, no_time_information=False, tokenize="unique"):
    """
    Collate function to turn event stream data into tokens ready for the JAX model

    :param batch: list of tuples of (events, target)
    :param resolution: resolution of the event stream
    :param pad_unit: padding unit for the tokens. All sequences will be padded to integer multiples of this unit.
                     This option results in JAX compiling multiple GPU kernels for different sequence lengths,
                     which might slow down compilation time, but improves throughput for the rest of the training process.
    :param cut_mix: probability of applying cut mix augmentation
    :param no_time_information: if True, the time information is ignored and all events are treated as if they were
                                recorded sampled at uniform time intervals.
                                This option is only used for ablation studies.
    """

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

def eigenworms_collate_fn(batch, resolution, pad_unit, cut_mix=0.0, no_time_information=False, tokenize="unique"):
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


def lra_pathfinder_collate_fn(batch, resolution, pad_unit, cut_mix=0.0, no_time_information=False, tokenize="unique"):

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


def retrieval_collate_fn(batch, resolution, pad_unit, cut_mix=0.0, no_time_information=True, tokenize="unique"):
    """
    Collate function to turn event stream data into tokens ready for the JAX model

    :param batch: list of tuples of (events, target)
    :param resolution: resolution of the event stream
    :param pad_unit: padding unit for the tokens. All sequences will be padded to integer multiples of this unit.
                     This option results in JAX compiling multiple GPU kernels for different sequence lengths,
                     which might slow down compilation time, but improves throughput for the rest of the training process.
    :param cut_mix: probability of applying cut mix augmentation
    :param no_time_information: if True, the time information is ignored and all events are treated as if they were
                                recorded sampled at uniform time intervals.
                                This option is only used for ablation studies.
    """

    xs, y, *z = zip(*batch)
    x1, x2 = zip(*xs)

    assert len(z) == 0
    assert len(x1) == len(x2)
    batch_size_one = len(x1) == 1

    # set labels to numpy array
    y = np.stack(y)

    # integration time steps are the difference between two consecutive time stamps
    if no_time_information:
        timesteps1 = [np.ones_like(e['t'][:-1]) for e in x1]
        timesteps2 = [np.ones_like(e['t'][:-1]) for e in x2]
    else:
        timesteps1 = [np.diff(e['t']) for e in x1]
        timesteps2 = [np.diff(e['t']) for e in x2]

    # NOTE: since timesteps are deltas, their length is L - 1, and we have to remove the last token in the following

    # process tokens for single input dim (e.g. audio)
    tokens1 = [e['x'][:-1].astype(np.int32) for e in x1]
    tokens2 = [e['x'][:-1].astype(np.int32) for e in x2]

    # get padding lengths
    lengths1 = np.array([len(e) for e in timesteps1], dtype=np.int32)
    lengths2 = np.array([len(e) for e in timesteps2], dtype=np.int32)
    pad_length = max((lengths1.max() // pad_unit) * pad_unit + pad_unit, (lengths2.max() // pad_unit) * pad_unit + pad_unit)

    tokens1 = np.stack(
        [np.pad(e, (0, pad_length - len(e)), mode='constant', constant_values=-1) for e in tokens1])
    tokens2 = np.stack(
        [np.pad(e, (0, pad_length - len(e)), mode='constant', constant_values=-1) for e in tokens2])
    timesteps1 = np.stack(
        [np.pad(e, (0, pad_length - len(e)), mode='constant', constant_values=0) for e in timesteps1])
    timesteps2 = np.stack(
        [np.pad(e, (0, pad_length - len(e)), mode='constant', constant_values=0) for e in timesteps2])
 
    # Before concatenation

    # Perform concatenation
    tokens = np.concatenate((tokens1, tokens2), axis=0)
    timesteps = np.concatenate((timesteps1, timesteps2), axis=0)
    lengths = np.concatenate((lengths1, lengths2), axis=0)
    
    #lengths = lengths1 + lengths2

    # After concatenation

    if batch_size_one:
        lengths = lengths[None, ...]

    return tokens, y, timesteps, lengths


def ptb_collate_fn(batch, resolution, pad_unit, cut_mix=0.0, no_time_information=False, tokenize="unique"):
    """
    Collate function to turn PTB batch data into padded tokens for the model.
    
    :param batch: list of tuples of (events, target) where both events and target are sequences
    :param resolution: resolution of the event stream
    :param pad_unit: padding unit for the tokens. All sequences will be padded to integer multiples of this unit.
    :param cut_mix: probability of applying cut mix augmentation
    :param no_time_information: if True, ignores time information and assumes uniform sampling.
    """
    # x are inputs, y are targets
    x, y, *z = zip(*batch)
    assert len(z) == 0
    batch_size_one = len(x) == 1

    # Prepare timesteps and tokens based on input
    timesteps = [np.ones_like(e['t']) for e in x]
    tokens = [e['x'].astype(np.int32) for e in x]
    targets = [e['x'].astype(np.int32) for e in y] #y  # `y` is treated as a sequence and padded like `x`
    
    # Determine padding lengths for tokens and targets
    tokens_lengths = np.array([len(e) for e in tokens], dtype=np.int32)
    targets_lengths = np.array([len(t) for t in targets], dtype=np.int32)
    
    tokens_pad_length = (tokens_lengths.max() // pad_unit) * pad_unit + pad_unit
    targets_pad_length = (targets_lengths.max() // pad_unit) * pad_unit + pad_unit

    # Set pad_length as the maximum of tokens_pad_length and targets_pad_length
    pad_length = max(tokens_pad_length, targets_pad_length)

    # Pad `tokens`, `targets`, and `timesteps` to the computed pad_length
    tokens = np.stack([np.pad(e, (0, pad_length - len(e)), mode='constant', constant_values=-1) for e in tokens])
    targets = np.stack([np.pad(t, (0, pad_length - len(t)), mode='constant', constant_values=-1) for t in targets])    
    timesteps = np.stack([np.pad(e, (0, pad_length - len(e)), mode='constant', constant_values=0) for e in timesteps])

    # Normalize timesteps if needed
    timesteps = timesteps / 1

    # Ensure consistent dimensions when batch_size is 1
    if batch_size_one:
        tokens_lengths = tokens_lengths[None, ...]

    return tokens, targets, timesteps, tokens_lengths


def event_stream_dataloader(
        train_data,
        val_data,
        test_data,
        batch_size,
        eval_batch_size,
        train_collate_fn,
        eval_collate_fn,
        rng,
        num_workers=0,
        shuffle_training=True
):
    """
    Create dataloaders for training, validation and testing

    :param train_data: training dataset
    :param val_data: validation dataset
    :param test_data: test dataset
    :param batch_size: batch size for training
    :param eval_batch_size: batch size for evaluation
    :param train_collate_fn: collate function for training
    :param eval_collate_fn: collate function for evaluation
    :param rng: random number generator
    :param num_workers: number of workers for data loading
    :param shuffle_training: whether to shuffle the training data

    :return: train_loader, val_loader, test_loader
    """
    def dataloader(dset, bsz, collate_fn, shuffle, drop_last):
        return torch.utils.data.DataLoader(
            dset,
            batch_size=bsz,
            drop_last=drop_last,
            collate_fn=collate_fn,
            shuffle=shuffle,
            generator=rng,
            num_workers=num_workers
        )
    train_loader = dataloader(train_data, batch_size, train_collate_fn, shuffle=shuffle_training, drop_last=True)
    val_loader = dataloader(val_data, eval_batch_size, eval_collate_fn, shuffle=False, drop_last=True)
    test_loader = dataloader(test_data, eval_batch_size, eval_collate_fn, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader

def event_stream_dataloader_image(
        train_data,
        val_data,
        test_data,
        batch_size,
        eval_batch_size,
        train_collate_fn,
        eval_collate_fn,
        rng,
        num_workers=0,
        shuffle_training=True
):
    def dataloader(dset, bsz, collate_fn, shuffle, drop_last):
        return torch.utils.data.DataLoader(
            dset,
            batch_size=bsz,
            drop_last=drop_last,
            collate_fn=collate_fn,
            shuffle=shuffle,
            generator=rng,
            num_workers=num_workers
        )
    train_loader = dataloader(train_data, batch_size, train_collate_fn, shuffle=shuffle_training, drop_last=True)
    val_loader = dataloader(val_data, eval_batch_size, eval_collate_fn, shuffle=False, drop_last=False)
    test_loader = dataloader(test_data, eval_batch_size, eval_collate_fn, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader

def event_stream_dataloader_parallel(
        train_data,
        val_data,
        test_data,
        batch_size,
        eval_batch_size,
        train_collate_fn,
        eval_collate_fn,
        rng,
        num_workers=0,
        shuffle_training=True
):
    """
    Create dataloaders for training, validation and testing

    :param train_data: training dataset
    :param val_data: validation dataset
    :param test_data: test dataset
    :param batch_size: batch size for training
    :param eval_batch_size: batch size for evaluation
    :param train_collate_fn: collate function for training
    :param eval_collate_fn: collate function for evaluation
    :param rng: random number generator
    :param num_workers: number of workers for data loading
    :param shuffle_training: whether to shuffle the training data

    :return: train_loader, val_loader, test_loader
    """
    def dataloader(dset, bsz, collate_fn, shuffle, drop_last):
        return torch.utils.data.DataLoader(
            dset,
            batch_size=bsz,
            drop_last=drop_last,
            collate_fn=collate_fn,
            shuffle=shuffle,
            generator=rng,
            num_workers=num_workers
        )
    train_loader = dataloader(train_data, batch_size, train_collate_fn, shuffle=shuffle_training, drop_last=True)
    val_loader = dataloader(val_data, eval_batch_size, eval_collate_fn, shuffle=False, drop_last=True)
    test_loader = dataloader(test_data, eval_batch_size, eval_collate_fn, shuffle=False, drop_last=True)
    return train_loader, val_loader, test_loader

def create_events_shd_classification_dataset(
        cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
        per_device_batch_size: int = 32,
        per_device_eval_batch_size: int = 64,
        world_size: int = 1,
        num_workers: int = 0,
        seed: int = 42,
        time_jitter: float = 100,
        spatial_jitter: float = 1.0,
        max_drop_chunk: float = 0.1,
        noise: int = 100,
        drop_event: float = 0.1,
        time_skew: float = 1.1,
        cut_mix: float = 0.5,
        pad_unit: int = 8192,
        validate_on_test: bool = False,
        no_time_information: bool = False,
        **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Data]:
    """
    creates a view of the spiking heidelberg digits dataset

    :param cache_dir:		    (str):		where to store the dataset
    :param bsz:				    (int):		Batch size.
    :param seed:			    (int)		Seed for shuffling data.
    :param time_jitter:		    (float)		Standard deviation of the time jitter.
    :param spatial_jitter:	    (float)		Standard deviation of the spatial jitter.
    :param max_drop_chunk:	    (float)		Maximum fraction of events to drop in a single chunk.
    :param noise:			    (int)		Number of noise events to add.
    :param drop_event:		    (float)		Probability of dropping an event.
    :param time_skew:		    (float)		Time skew factor.
    :param cut_mix:			    (float)		Probability of applying cut mix augmentation.
    :param pad_unit:		    (int)		Padding unit for the tokens. See collate function for more details
    :param validate_on_test:	(bool)		If True, use the test set for validation.
                                            Else use a random validation split from the test set.
    :param no_time_information:	(bool)		Whether to ignore the time information in the events.

    :return: train_loader, val_loader, test_loader, data
    """

    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    sensor_size = (700, 1, 1)

    transforms = tonic.transforms.Compose([
        tonic.transforms.DropEvent(p=drop_event),
        DropEventChunk(p=0.3, max_drop_size=max_drop_chunk),
        Jitter1D(sensor_size=sensor_size, var=spatial_jitter),
        tonic.transforms.TimeSkew(coefficient=(1 / time_skew, time_skew), offset=0),
        tonic.transforms.TimeJitter(std=time_jitter, clip_negative=False, sort_timestamps=True),
        tonic.transforms.UniformNoise(sensor_size=sensor_size, n=(0, noise))
    ])
    target_transforms = OneHotLabels(num_classes=20)

    train_data = tonic.datasets.SHD(save_to=cache_dir, train=True, transform=transforms, target_transform=target_transforms)
    val_data = tonic.datasets.SHD(save_to=cache_dir, train=True, target_transform=target_transforms)
    test_data = tonic.datasets.SHD(save_to=cache_dir, train=False, target_transform=target_transforms)

    # create validation set
    if validate_on_test:
        val_data = tonic.datasets.SHD(save_to=cache_dir, train=False, target_transform=target_transforms)
    else:
        val_length = int(0.1 * len(train_data))
        indices = torch.randperm(len(train_data), generator=rng)
        train_data = torch.utils.data.Subset(train_data, indices[:-val_length])
        val_data = torch.utils.data.Subset(val_data, indices[-val_length:])

    collate_fn = partial(event_stream_collate_fn, resolution=(700,), pad_unit=pad_unit, no_time_information=no_time_information)
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=partial(collate_fn, cut_mix=cut_mix),
        eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True
    )
    data = Data(
        n_classes=20, num_embeddings=700, train_size=len(train_data)
    )
    return train_loader, val_loader, test_loader, data


def create_events_ssc_classification_dataset(
        cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
        per_device_batch_size: int = 32,
        per_device_eval_batch_size: int = 64,
        world_size: int = 1,
        num_workers: int = 0,
        seed: int = 42,
        time_jitter: float = 100,
        spatial_jitter: float = 1.0,
        max_drop_chunk: float = 0.1,
        noise: int = 100,
        drop_event: float = 0.1,
        time_skew: float = 1.1,
        cut_mix: float = 0.5,
        pad_unit: int = 8192,
        no_time_information: bool = False,
        **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Data]:
    """
    creates a view of the spiking speech commands dataset

    :param cache_dir:		    (str):		where to store the dataset
    :param bsz:				    (int):		Batch size.
    :param seed:			    (int)		Seed for shuffling data.
    :param time_jitter:		    (float)		Standard deviation of the time jitter.
    :param spatial_jitter:	    (float)		Standard deviation of the spatial jitter.
    :param max_drop_chunk:	    (float)		Maximum fraction of events to drop in a single chunk.
    :param noise:			    (int)		Number of noise events to add.
    :param drop_event:		    (float)		Probability of dropping an event.
    :param time_skew:		    (float)		Time skew factor.
    :param cut_mix:			    (float)		Probability of applying cut mix augmentation.
    :param pad_unit:		    (int)		Padding unit for the tokens. See collate function for more details
    :param no_time_information:	(bool)		Whether to ignore the time information in the events.

    :return: train_loader, val_loader, test_loader, data
    """

    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    sensor_size = (700, 1, 1)

    transforms = tonic.transforms.Compose([
        tonic.transforms.DropEvent(p=drop_event),
        DropEventChunk(p=0.3, max_drop_size=max_drop_chunk),
        Jitter1D(sensor_size=sensor_size, var=spatial_jitter),
        tonic.transforms.TimeSkew(coefficient=(1 / time_skew, time_skew), offset=0),
        tonic.transforms.TimeJitter(std=time_jitter, clip_negative=False, sort_timestamps=True),
        tonic.transforms.UniformNoise(sensor_size=sensor_size, n=(0, noise))
    ])
    target_transforms = OneHotLabels(num_classes=35)

    train_data = tonic.datasets.SSC(save_to=cache_dir, split='train', transform=transforms, target_transform=target_transforms)
    val_data = tonic.datasets.SSC(save_to=cache_dir, split='valid', target_transform=target_transforms)
    test_data = tonic.datasets.SSC(save_to=cache_dir, split='test', target_transform=target_transforms)

    collate_fn = partial(event_stream_collate_fn, resolution=(700,), pad_unit=pad_unit, no_time_information=no_time_information)
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=partial(collate_fn, cut_mix=cut_mix),
        eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True
    )

    data = Data(
        n_classes=35, num_embeddings=700, train_size=len(train_data)
    )
    return train_loader, val_loader, test_loader, data


def create_events_dvs_gesture_classification_dataset(
        cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
        per_device_batch_size: int = 32,
        per_device_eval_batch_size: int = 64,
        world_size: int = 1,
        num_workers: int = 0,
        seed: int = 42,
        slice_events: int = 0,
        pad_unit: int = 2 ** 19,
        # Augmentation parameters
        time_jitter: float = 100,
        spatial_jitter: float = 1.0,
        noise: int = 100,
        drop_event: float = 0.1,
        time_skew: float = 1.1,
        cut_mix: float = 0.5,
        downsampling: int = 1,
        max_roll: int = 4,
        max_angle: float = 10,
        max_scale: float = 1.5,
        max_drop_chunk: float = 0.1,
        validate_on_test: bool = False,
        slice_val_set: bool = False,
        **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Data]:
    """
    creates a view of the DVS Gesture dataset

    :param cache_dir:		    (str):		where to store the dataset
    :param bsz:				    (int):		Batch size.
    :param seed:			    (int)		Seed for shuffling data.
    :param slice_events:	    (int)		Number of events per slice.
    :param pad_unit:		    (int)		Padding unit for the tokens. See collate function for more details
    :param time_jitter:		    (float)		Standard deviation of the time jitter.
    :param spatial_jitter:	    (float)		Standard deviation of the spatial jitter.
    :param noise:			    (int)		Number of noise events to add.
    :param drop_event:		    (float)		Probability of dropping an event.
    :param time_skew:		    (float)		Time skew factor.
    :param cut_mix:			    (float)		Probability of applying cut mix augmentation.
    :param downsampling:	    (int)		Downsampling factor.
    :param max_roll:		    (int)		Maximum number of pixels to roll the events.
    :param max_angle:		    (float)		Maximum angle to rotate the events.
    :param max_scale:		    (float)		Maximum scale factor to scale the events.
    :param max_drop_chunk:	    (float)		Maximum fraction of events to drop in a single chunk.
    :param validate_on_test:	(bool)		If True, use the test set for validation.
                                            Else use a random validation split from the test set.

    :return: train_loader, val_loader, test_loader, data
    """

    assert time_skew > 1, "time_skew must be greater than 1"

    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    orig_sensor_size = (128, 128, 2)
    new_sensor_size = (128 // downsampling, 128 // downsampling, 2)
    train_transforms = [
        # Event transformations
        DropEventChunk(p=0.3, max_drop_size=max_drop_chunk),
        tonic.transforms.DropEvent(p=drop_event),
        tonic.transforms.UniformNoise(sensor_size=new_sensor_size, n=(0, noise)),
        # Time tranformations
        tonic.transforms.TimeSkew(coefficient=(1 / time_skew, time_skew), offset=0),
        tonic.transforms.TimeJitter(std=time_jitter, clip_negative=False, sort_timestamps=True),
        # Spatial transformations
        tonic.transforms.SpatialJitter(sensor_size=orig_sensor_size, var_x=spatial_jitter, var_y=spatial_jitter, clip_outliers=True),
        tonic.transforms.Downsample(sensor_size=orig_sensor_size, target_size=new_sensor_size[:2]) if downsampling > 1 else Identity(),
        # Geometric tranformations
        Roll(sensor_size=new_sensor_size, p=0.3, max_roll=max_roll),
        Rotate(sensor_size=new_sensor_size, p=0.3, max_angle=max_angle),
        Scale(sensor_size=new_sensor_size, p=0.3, max_scale=max_scale),
    ]

    train_transforms = tonic.transforms.Compose(train_transforms)
    test_transforms = tonic.transforms.Compose([
        tonic.transforms.Downsample(sensor_size=orig_sensor_size, target_size=new_sensor_size[:2]) if downsampling > 1 else Identity(),
    ])
    target_transforms = OneHotLabels(num_classes=11)

    TrainData = partial(tonic.datasets.DVSGesture, save_to=cache_dir, train=True)
    TestData = partial(tonic.datasets.DVSGesture, save_to=cache_dir, train=False)

    # create validation set
    if validate_on_test:
        val_data = TestData(transform=test_transforms, target_transform=target_transforms)
    else:
        # create train validation split
        val_data = TrainData(transform=test_transforms, target_transform=target_transforms)
        val_length = int(0.2 * len(val_data))
        indices = torch.randperm(len(val_data), generator=rng)
        val_data = torch.utils.data.Subset(val_data, indices[-val_length:])

    # if slice event count is given, train on slices of the training data
    if slice_events > 0:
        slicer = tonic.slicers.SliceByEventCount(event_count=slice_events, overlap=slice_events // 2, include_incomplete=True)
        train_subset = torch.utils.data.Subset(TrainData(), indices[:-val_length]) if not validate_on_test else TrainData()
        train_data = tonic.sliced_dataset.SlicedDataset(
            dataset=train_subset,
            slicer=slicer,
            transform=train_transforms,
            target_transform=target_transforms,
            metadata_path=None
        )
        if slice_val_set:
            val_subset = TestData()
            val_data = tonic.sliced_dataset.SlicedDataset(
                dataset=val_subset,
                slicer=slicer,
                transform=test_transforms,
                target_transform=target_transforms,
                metadata_path=None
        )
    else:
        train_data = torch.utils.data.Subset(
            TrainData(transform=train_transforms, target_transform=target_transforms),
            indices[:-val_length]
        ) if not validate_on_test else TrainData(transform=train_transforms)

    # Always evaluate on the full sequences
    test_data = TestData(transform=test_transforms, target_transform=target_transforms)

    # define collate functions
    train_collate_fn = partial(
            event_stream_collate_fn,
            resolution=new_sensor_size[:2],
            pad_unit=slice_events if (slice_events != 0 and slice_events < pad_unit) else pad_unit,
            cut_mix=cut_mix
        )
    eval_collate_fn = partial(
            event_stream_collate_fn,
            resolution=new_sensor_size[:2],
            pad_unit=pad_unit,
        )
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=train_collate_fn,
        eval_collate_fn=eval_collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True
    )

    data = Data(
        n_classes=11, num_embeddings=np.prod(new_sensor_size), train_size=len(train_data)
    )
    return train_loader, val_loader, test_loader, data



################################################## Long Range Arena ######################################################

from torch.utils.data import Dataset
from tonic.io import make_structured_array
from S5.s5.dataloaders.lra import *

# ListOps

class ListOpsTonic(Dataset):
    """ListOps dataset for Tonic framework using the ListOps class for data processing."""

    base_url = "https://github.com/google-research/long-range-arena"
    sensor_size = (18, 1, 1)
    def __init__(
        self,
        data_dir: str,
        cache_dir: Optional[str] = None,
        split: str = "None",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        assert split in ['train', 'val', 'test'], "split must be 'train', 'val', or 'test'"
        self.sensor_size = (18, 1, 1)
        self.dtype = np.dtype([("t", int), ("x", int), ("p", int)])
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.listops = ListOps(_name_="listops",data_dir=data_dir, cache_dir=cache_dir, train=(split=="train"))
        self.listops.prepare_data()
        self.listops.setup(stage=split)
        if split == 'train':
            self.data = self.listops.dataset_train
        elif split == 'val':
            self.data = self.listops.dataset_val
        else:
            self.data = self.listops.dataset_test

    def __getitem__(self, index):
        data = self.data[index]
        input_ids = data["input_ids"]
        target = data["Target"]
        
        # Generating artificial time info as an array from 1 to n
        times = np.arange(0, len(input_ids))
        
        # Simulating the polarity as 1 and converting to structured array
        events = make_structured_array(times, input_ids, 1, dtype = self.dtype)
        
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        if self.data is None:
            raise ValueError("Data not loaded properly; self.data is None.")
        return len(self.data)
    
def create_listops_tonic_classification_dataset(
        cache_dir: Union[str, Path] = None, 
        data_dir: str = None, 
        per_device_batch_size: int = 32,
        per_device_eval_batch_size: int = 64,
        world_size: int = 1,
        num_workers: int = 0,
        seed: int = 42,
        pad_unit: int = 2048,
        cut_mix: float = 0.0,
        no_time_information: bool = False,
        **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Data]:
    """
    Creates a view of the ListOps dataset for Tonic framework

    :param cache_dir:		    (str):		where to store the dataset
    :param data_dir:		    (str):		where the dataset is located
    :param per_device_batch_size:(int):		Batch size for training.
    :param per_device_eval_batch_size: (int): Batch size for evaluation.
    :param world_size:           (int):       Number of devices for training.
    :param num_workers:          (int):       Number of worker threads for data loading.
    :param seed:                 (int):       Seed for shuffling data.
    :param validate_on_test:     (bool):      If True, use the test set for validation.
                                             Else use a random validation split from the test set.

    :return: train_loader, val_loader, test_loader, data
    """

    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

 
    data_dir = Path(cache_dir).parent / "long-range-arena/lra_release/lra_release/listops-1000"
    cache_dir = data_dir / "l_max-2048-append_bos-False-append_eos-True"

    target_transforms = OneHotLabels(num_classes=10)
    TrainData = partial(ListOpsTonic, data_dir=data_dir, cache_dir=cache_dir, split='train', target_transform=target_transforms)
    ValData = partial(ListOpsTonic, data_dir=data_dir, cache_dir=cache_dir, split='val', target_transform=target_transforms)
    TestData = partial(ListOpsTonic, data_dir=data_dir, cache_dir=cache_dir, split='test', target_transform=target_transforms)
    
    train_data = TrainData()
    val_data = ValData()
    test_data = TestData()

    assert len(train_data) > 0
    assert len(val_data) > 0
    assert len(test_data) > 0

    collate_fn = partial(event_stream_collate_fn, resolution=(18,), pad_unit=pad_unit, no_time_information=no_time_information)
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=partial(collate_fn, cut_mix=cut_mix),
        eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True
    )
    
    data = Data(
        n_classes=10, num_embeddings=18, train_size=len(train_data)
    )

    return train_loader, val_loader, test_loader, data

# Text

class IMDBTonic(Dataset):
    """IMDB dataset for Tonic framework using the IMDB class for data processing."""

    def __init__(
        self,
        data_dir: str,
        cache_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        assert split in ['train', 'test'], "split must be 'train' or 'test'"
        self.sensor_size = (135, 1, 1)
        self.dtype = np.dtype([("t", int), ("x", int), ("p", int)])
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.imdb = IMDB(_name_="imdb", data_dir=data_dir, cache_dir=self.cache_dir)
        self.imdb.prepare_data()
        self.imdb.setup(stage=split)
        if split == 'train':
            self.data = self.imdb.dataset_train
        else:
            self.data = self.imdb.dataset_test

    def __getitem__(self, index):
        data = self.data[index]
        input_ids = data["input_ids"]
        label = data["label"]
        
        # Generating artificial time info as an array from 1 to n
        times = np.arange(0, len(input_ids))
        
        # Simulating the polarity as 1 and converting to structured array
        events = make_structured_array(times, input_ids, 1, dtype=self.dtype)
        
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return events, label

    def __len__(self):
        return len(self.data)


def create_imdb_tonic_classification_dataset(
        cache_dir: Union[str, Path] = None,
        data_dir: str = None,
        per_device_batch_size: int = 32,
        per_device_eval_batch_size: int = 32,
        world_size: int = 1,
        num_workers: int = 16,
        seed: int = 42,
        pad_unit: int = 4096,
        cut_mix: float = 0.0,
        no_time_information: bool = False,
        **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates a view of the IMDB dataset for Tonic framework

    :param cache_dir:		    (str):		where to store the dataset
    :param data_dir:		    (str):		where the dataset is located
    :param per_device_batch_size:(int):		Batch size for training.
    :param per_device_eval_batch_size: (int): Batch size for evaluation.
    :param world_size:           (int):       Number of devices for training.
    :param num_workers:          (int):       Number of worker threads for data loading.
    :param seed:                 (int):       Seed for shuffling data.
    :param val_split:            (float):     Proportion of training data to use for validation.
    
    :return: train_loader, val_loader, test_loader
    """

    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    data_dir = Path(cache_dir).parent / "long-range-arena/text/"
    cache_dir = data_dir 
    
    target_transforms = OneHotLabels(num_classes=2)
    TrainData = partial(IMDBTonic, data_dir=data_dir, cache_dir=cache_dir, split='train', target_transform=target_transforms)
    TestData = partial(IMDBTonic, data_dir=data_dir, cache_dir=cache_dir, split='test', target_transform=target_transforms)
    
    train_data = TrainData()
    test_data = TestData()
    val_data = test_data  # Use test set as validation set 
    collate_fn = partial(lra_text_collate_fn, resolution=(135,), pad_unit=pad_unit, no_time_information=no_time_information)
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=partial(collate_fn, cut_mix=cut_mix),
        eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True
    )


    data = Data(
        n_classes=2, num_embeddings=135, train_size=len(train_data)
    )

    return train_loader, val_loader, test_loader, data

# Retrieval

class AANTonic(Dataset):
    """AAN dataset for Tonic framework using the AAN class for data processing."""

    def __init__(
        self,
        data_dir: str,
        cache_dir: Optional[str] = None,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        assert split in ['train', 'val', 'test'], "split must be 'train', 'val', or 'test'"
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.aan = AAN(_name_="aan",data_dir=data_dir, cache_dir=cache_dir, train=(split=='train'))
        self.dtype = np.dtype([("t", int), ("x", int), ("p", int)])
        self.aan.prepare_data()
        self.aan.setup(stage=split)
        if split == 'train':
            self.data = self.aan.dataset_train
        elif split == 'val':
            self.data = self.aan.dataset_val
        elif split == "test":
            self.data = self.aan.dataset_test

    def __getitem__(self, index):
        data = self.data[index]
        input_ids1 = data["input_ids1"]
        input_ids2 = data["input_ids2"]
        target = data["label"]

        # Generating artificial time info as an array from 1 to n
        times1 = np.arange(0, len(input_ids1))
        times2 = np.arange(0, len(input_ids2))

        # Simulating the polarity as 1 and converting to structured array
        events1 = make_structured_array(times1, input_ids1, 1, dtype=self.dtype)
        events2 = make_structured_array(times2, input_ids2, 1, dtype=self.dtype)

        if self.transform is not None:
            events1 = self.transform(events1)
            events2 = self.transform(events2)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return (events1, events2), target

    def __len__(self):
        return len(self.data)
    
def create_retrieval_tonic_classification_dataset(
        cache_dir: Union[str, Path] = None,
        data_dir: str = None,
        per_device_batch_size: int = 256,
        per_device_eval_batch_size: int = 256,
        world_size: int = 1,
        num_workers: int = 40,
        seed: int = 42,
        pad_unit: int = 4000,
        cut_mix: float = 0.0,
        no_time_information: bool = True,
        **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Data]:
    """
    Creates a view of the AAN dataset for Tonic framework

    :param cache_dir:		    (str):		where to store the dataset
    :param data_dir:		    (str):		where the dataset is located
    :param per_device_batch_size:(int):		Batch size for training.
    :param per_device_eval_batch_size: (int): Batch size for evaluation.
    :param world_size:           (int):       Number of devices for training.
    :param num_workers:          (int):       Number of worker threads for data loading.
    :param seed:                 (int):       Seed for shuffling data.
    :param validate_on_test:     (bool):      If True, use the test set for validation.
                                             Else use a random validation split from the test set.

    :return: train_loader, val_loader, test_loader, data
    """

    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None


    data_dir = Path(cache_dir).parent / "long-range-arena/retrieval/"
    cache_dir = data_dir 

    target_transforms = OneHotLabels(num_classes=2)
    TrainData = partial(AANTonic, data_dir=data_dir, cache_dir=cache_dir, split='train', target_transform=target_transforms)
    ValData = partial(AANTonic, data_dir=data_dir, cache_dir=cache_dir, split='val', target_transform=target_transforms)
    TestData = partial(AANTonic, data_dir=data_dir, cache_dir=cache_dir, split='test', target_transform=target_transforms)
    
    train_data = TrainData()
    val_data = ValData()
    test_data = TestData()

    collate_fn = partial(retrieval_collate_fn, resolution=(98,), pad_unit=pad_unit, no_time_information=no_time_information)
    if world_size > 1:
        train_loader, val_loader, test_loader = event_stream_dataloader_parallel(
            train_data, val_data, test_data,
            train_collate_fn=collate_fn,
            eval_collate_fn=collate_fn,
            batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
            rng=rng, num_workers=num_workers, shuffle_training=True
        )
    else:
        train_loader, val_loader, test_loader = event_stream_dataloader(
            train_data, val_data, test_data,
            train_collate_fn=collate_fn,
            eval_collate_fn=collate_fn,
            batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
            rng=rng, num_workers=num_workers, shuffle_training=True
        )
    
    data = Data(
        n_classes=2, num_embeddings=98, train_size=len(train_data)
    )

    return train_loader, val_loader, test_loader, data


## Image

from S5.s5.dataloaders.basic import CIFAR10

class ImageTonic(Dataset):
    """Image dataset for Tonic framework using the CIFAR10 class for data processing."""

    base_url = "https://github.com/google-research/long-range-arena"
    sensor_size = (256, 1, 1)
    def __init__(
        self,
        data_dir: str,
        cache_dir: Optional[str] = None,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        assert split in ['train', 'val', 'test'], "split must be 'train', 'val', or 'test'"
        self.sensor_size = (256, 1, 1)
        self.dtype = np.dtype([("t", int), ("x", int), ("p", int)])
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        kwargs = {
		"grayscale": True,  # LRA uses a grayscale CIFAR image.
        "tokenize": True,
	    }
        self.image = CIFAR10(_name_="cifar",data_dir=data_dir, cache_dir=cache_dir, **kwargs)
        self.image.setup()
        if split == 'train':
            self.data = self.image.dataset_train
        elif split == 'val':
            self.data = self.image.dataset_val
        elif split == "test":
            self.data = self.image.dataset_test
        else:
            raise ValueError
        
        
        

    def __getitem__(self, index):
        data = self.data[index]

        input_ids,target = data

        # Generating artificial time info as an array from 1 to n
        times = np.arange(0, len(input_ids))
        

        input_ids = np.squeeze(input_ids)
        # Simulating the polarity as 1 and converting to structured array
        events = make_structured_array(times, input_ids, 1, dtype = self.dtype)
        
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        return len(self.data)
    
def create_image_tonic_classification_dataset(
        cache_dir: Union[str, Path] = None,
        data_dir: str = None,
        per_device_batch_size: int = 32,
        per_device_eval_batch_size: int = 64,
        world_size: int = 1,
        num_workers: int = 40,
        seed: int = 42,
        pad_unit: int = 1024,
        cut_mix: float = 0.0,
        no_time_information: bool = False,
        **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Data]:
    """
    Creates a view of the image dataset for Tonic framework

    :param cache_dir:		    (str):		where to store the dataset
    :param data_dir:		    (str):		where the dataset is located
    :param per_device_batch_size:(int):		Batch size for training.
    :param per_device_eval_batch_size: (int): Batch size for evaluation.
    :param world_size:           (int):       Number of devices for training.
    :param num_workers:          (int):       Number of worker threads for data loading.
    :param seed:                 (int):       Seed for shuffling data.
    :param validate_on_test:     (bool):      If True, use the test set for validation.
                                             Else use a random validation split from the test set.

    :return: train_loader, val_loader, test_loader, data
    """

    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    data_dir = Path(cache_dir).parent / "long-range-arena/image/"
    cache_dir = data_dir 

    target_transforms = OneHotLabels(num_classes=10)
    TrainData = partial(ImageTonic, data_dir=data_dir, cache_dir=cache_dir, split='train', target_transform=target_transforms)
    ValData = partial(ImageTonic, data_dir=data_dir, cache_dir=cache_dir, split='val', target_transform=target_transforms)
    TestData = partial(ImageTonic, data_dir=data_dir, cache_dir=cache_dir, split='test', target_transform=target_transforms)
    
    train_data = TrainData()
    val_data = ValData()
    test_data = TestData()
    

    collate_fn = partial(lra_image_collate_fn, resolution=(256,), pad_unit=pad_unit, no_time_information=no_time_information)
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=partial(collate_fn, cut_mix=cut_mix),
        eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True
    )
    
    data = Data(
        n_classes=10, num_embeddings=256, train_size=len(train_data)
    )

    return train_loader, val_loader, test_loader, data

## PathFinder

class PathFinderTonic(Dataset):
    """PathFinder dataset for Tonic framework using the PathFinder class for data processing."""

    base_url = "https://github.com/google-research/long-range-arena"
    sensor_size = (32, 32, 1)

    def __init__(
        self,
        data_dir: str,
        tokenize: bool,
        cache_dir: Optional[str] = None,
        resolution: int = 32,
        split: str = "None",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        
    ):
        assert split in ['train', 'val', 'test'], "split must be 'train', 'val', or 'test'"
        self.sensor_size = (32, 32, 1)
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.split = split
        self.transform = transform
        if tokenize:
            self.dtype = np.dtype([("t", int), ("x", int), ("p", int)])
        else:
            self.dtype = np.dtype([("t", int), ("x", float), ("p", int)])
        self.target_transform = target_transform
        kwargs = {"tokenize":tokenize}
        self.pathfinder = PathFinder(_name_="pathfinder", data_dir=data_dir, cache_dir=self.cache_dir,resolution=resolution,**kwargs)
        self.pathfinder.setup()

        if split == 'train':
            self.data = self.pathfinder.dataset_train
        elif split == 'val':
            self.data = self.pathfinder.dataset_val
        elif split == "test":
            self.data = self.pathfinder.dataset_test
        else:
            raise ValueError 

    def __getitem__(self, index):
        data = self.data[index]
        input_ids, target = data
        times = np.arange(0, len(input_ids))
        input_ids = np.squeeze(input_ids)
        events = make_structured_array(times, input_ids, 1, dtype = self.dtype)
        
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return events, target

    def __len__(self):
        if self.data is None:
            raise ValueError("Data not loaded properly; self.data is None.")
        return len(self.data)

def create_pathfinder_tonic_classification_dataset(
        cache_dir: Union[str, Path] = None,
        data_dir: str = None,
        resolution: int = 32,
        per_device_batch_size: int = 32,
        per_device_eval_batch_size: int = 64,
        world_size: int = 1,
        num_workers: int = 96,
        seed: int = 42,
        pad_unit: int = 1024,
        no_time_information: bool = True,
        **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Data]:
    """
    Creates a view of the PathFinder dataset for Tonic framework

    :param cache_dir:		    (str):		where to store the dataset
    :param data_dir:		    (str):		where the dataset is located
    :param resolution:          (int):       Resolution of the images.
    :param per_device_batch_size:(int):		Batch size for training.
    :param per_device_eval_batch_size: (int): Batch size for evaluation.
    :param world_size:           (int):       Number of devices for training.
    :param num_workers:          (int):       Number of worker threads for data loading.
    :param seed:                 (int):       Seed for shuffling data.

    :return: train_loader, val_loader, test_loader, data
    """
    
    data_dir = Path(cache_dir).parent / "long-range-arena/pathfinder/"
    cache_dir = data_dir 
    
    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    target_transforms = OneHotLabels(num_classes=2)
    TrainData = partial(PathFinderTonic, data_dir=data_dir, cache_dir=cache_dir, resolution=resolution, split='train', target_transform=target_transforms, tokenize=True)
    ValData = partial(PathFinderTonic, data_dir=data_dir, cache_dir=cache_dir, resolution=resolution, split='val', target_transform=target_transforms, tokenize=True)
    TestData = partial(PathFinderTonic, data_dir=data_dir, cache_dir=cache_dir, resolution=resolution, split='test', target_transform=target_transforms, tokenize=True)

    train_data = TrainData()
    val_data = ValData()
    test_data = TestData()
    
    assert len(train_data) > 0
    assert len(val_data) > 0
    assert len(test_data) > 0

    collate_fn = partial(lra_pathfinder_collate_fn, resolution=(256,), pad_unit=pad_unit, no_time_information=no_time_information)
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=partial(collate_fn),
        eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True
    )
   
    data = Data(
        n_classes=2, num_embeddings=256, train_size=len(train_data)
    )

    return train_loader, val_loader, test_loader, data

def create_pathx_tonic_classification_dataset(
        cache_dir: Union[str, Path] = None,
        data_dir: str = None,
        resolution: int = 128,
        per_device_batch_size: int = 32,
        per_device_eval_batch_size: int = 64,
        world_size: int = 1,
        num_workers: int = 96,
        seed: int = 42,
        pad_unit: int = 1024,
        no_time_information: bool = True,
        **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Data]:
    """
    Creates a view of the PathFinder dataset for Tonic framework

    :param cache_dir:		    (str):		where to store the dataset
    :param data_dir:		    (str):		where the dataset is located
    :param resolution:          (int):       Resolution of the images.
    :param per_device_batch_size:(int):		Batch size for training.
    :param per_device_eval_batch_size: (int): Batch size for evaluation.
    :param world_size:           (int):       Number of devices for training.
    :param num_workers:          (int):       Number of worker threads for data loading.
    :param seed:                 (int):       Seed for shuffling data.

    :return: train_loader, val_loader, test_loader, data
    """

    data_dir = Path(cache_dir).parent / "long-range-arena/pathfinder/"
    cache_dir = data_dir 

    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    target_transforms = OneHotLabels(num_classes=2)
    TrainData = partial(PathFinderTonic, data_dir=data_dir, cache_dir=cache_dir, resolution=resolution, split='train', target_transform=target_transforms, tokenize=True)
    ValData = partial(PathFinderTonic, data_dir=data_dir, cache_dir=cache_dir, resolution=resolution, split='val', target_transform=target_transforms, tokenize=True)
    TestData = partial(PathFinderTonic, data_dir=data_dir, cache_dir=cache_dir, resolution=resolution, split='test', target_transform=target_transforms, tokenize=True)

    train_data = TrainData()
    val_data = ValData()
    test_data = TestData()

    assert len(train_data) > 0
    assert len(val_data) > 0
    assert len(test_data) > 0

    collate_fn = partial(lra_pathfinder_collate_fn, resolution=(256,), pad_unit=pad_unit, no_time_information=no_time_information)
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=partial(collate_fn),
        eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True
    )

    data = Data(
        n_classes=2, num_embeddings=256, train_size=len(train_data)
    )

    return train_loader, val_loader, test_loader, data


###### Physics #######

## EigenWorms

class EigenWormsTonic(Dataset):
    """ListOps dataset for Tonic framework using the ListOps class for data processing."""
    def __init__(
        self,
        data_dir: str,
        split: str = "None",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        assert split in ['train', 'val', 'test'], "split must be 'train', 'val', or 'test'"
        self.dtype = np.dtype([("t", int), ("x", (float, 6)), ("p", int)])
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        if split == 'train':
            self.data = torch.load(self.data_dir / 'training.pt')
        elif split == 'val':
            self.data = torch.load(self.data_dir / 'validation.pt')
        elif split == "test":
            self.data = torch.load(self.data_dir / 'test.pt')
        else:
            raise ValueError

    def __getitem__(self, index):
        input_ids,target = self.data[index]
        
        # Generating artificial time info as an array from 1 to n
        times = np.arange(0, len(input_ids))
        
        # Simulating the polarity as 1 and converting to structured array
        events = make_structured_array(times, input_ids, 1, dtype = self.dtype)
        
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        if self.data is None:
            raise ValueError("Data not loaded properly; self.data is None.")
        return len(self.data)
    
def create_eigenworms_tonic_classification_dataset(
        cache_dir: str = None,
        per_device_batch_size: int = 32,
        per_device_eval_batch_size: int = 64,
        world_size: int = 1,
        num_workers: int = 0,
        seed: int = 42,
        pad_unit: int = 2048,
        cut_mix: float = 0.0,
        no_time_information: bool = False,
        **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Data]:
    """
    Creates a view of the Eigenworms dataset for Tonic framework

    :param cache_dir:		    (str):		where to store the dataset
    :param data_dir:		    (str):		where the dataset is located
    :param per_device_batch_size:(int):		Batch size for training.
    :param per_device_eval_batch_size: (int): Batch size for evaluation.
    :param world_size:           (int):       Number of devices for training.
    :param num_workers:          (int):       Number of worker threads for data loading.
    :param seed:                 (int):       Seed for shuffling data.
    :param validate_on_test:     (bool):      If True, use the test set for validation.
                                             Else use a random validation split from the test set.

    :return: train_loader, val_loader, test_loader, data
    """

    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    data_dir = Path(cache_dir).parent / "eigenworms/processed"

    target_transforms = OneHotLabels(num_classes=5)
    
    TrainData = partial(EigenWormsTonic, data_dir=data_dir, split='train', target_transform=target_transforms)
    ValData = partial(EigenWormsTonic, data_dir=data_dir, split='val', target_transform=target_transforms)
    TestData = partial(EigenWormsTonic, data_dir=data_dir, split='test', target_transform=target_transforms)
    
    train_data = TrainData()
    val_data = ValData()
    test_data = TestData()

    assert len(train_data) > 0
    assert len(val_data) > 0
    assert len(test_data) > 0

    collate_fn = partial(eigenworms_collate_fn, resolution=(6,), pad_unit=pad_unit, no_time_information=no_time_information)
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=partial(collate_fn, cut_mix=cut_mix),
        eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True
    )

    data = Data(
        n_classes=5, num_embeddings=6, train_size=len(train_data)
    )

    return train_loader, val_loader, test_loader, data


#############################################################################################################################
from odelstms.irregular_sampled_datasets import *

class PersonActivityTonic(Dataset):
    """Person activity dataset for Tonic framework using the Person activity class for data processing."""
    def __init__(
        self,
        data_dir: str = "None",
        split: str = "None",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        assert split in ['train', 'val', 'test'], "split must be 'train', 'val', or 'test'"
        self.dtype = np.dtype([("t", float), ("x", (float, 7)), ("p", int)])
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_obj = PersonData(data_dir = data_dir)

        if split == 'train':
            self.data = self.dataset_obj.train_x
            self.times = self.dataset_obj.train_t.squeeze()
            self.target = self.dataset_obj.train_y
        elif split == 'val':
            self.data = self.dataset_obj.test_x
            self.times = self.dataset_obj.test_t.squeeze()
            self.target = self.dataset_obj.test_y
        elif split == "test":
            self.data = self.dataset_obj.test_x
            self.times = self.dataset_obj.test_t.squeeze()
            self.target = self.dataset_obj.test_y
        else:
            raise ValueError

    def __getitem__(self, index):
        input_ids,times,target = self.data[index], self.times[index], self.target[index]

        # Simulating the polarity as 1 and converting to structured array
        events = make_structured_array(times, input_ids, 1, dtype = self.dtype)
        
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        if self.data is None:
            raise ValueError("Data not loaded properly; self.data is None.")
        return len(self.data)
    
def create_person_activity_tonic_classification_dataset(
        cache_dir: str = None,
        per_device_batch_size: int = 32,
        per_device_eval_batch_size: int = 64,
        world_size: int = 1,
        num_workers: int = 0,
        seed: int = 42,
        pad_unit: int = 2048,
        cut_mix: float = 0.0,
        no_time_information: bool = False,
        **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Data]:
    """
    Creates a view of the Eigenworms dataset for Tonic framework

    :param cache_dir:		    (str):		where to store the dataset
    :param data_dir:		    (str):		where the dataset is located
    :param per_device_batch_size:(int):		Batch size for training.
    :param per_device_eval_batch_size: (int): Batch size for evaluation.
    :param world_size:           (int):       Number of devices for training.
    :param num_workers:          (int):       Number of worker threads for data loading.
    :param seed:                 (int):       Seed for shuffling data.
    :param validate_on_test:     (bool):      If True, use the test set for validation.
                                             Else use a random validation split from the test set.

    :return: train_loader, val_loader, test_loader, data
    """

    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    data_dir = Path(cache_dir).parent / "person/ConfLongDemo_JSI.txt"

    target_transforms = OneHotLabels(num_classes=7)
    
    TrainData = partial(PersonActivityTonic, data_dir = data_dir, split='train', target_transform=target_transforms)
    ValData = partial(PersonActivityTonic, data_dir = data_dir, split='val', target_transform=target_transforms)
    TestData = partial(PersonActivityTonic, data_dir = data_dir, split='test', target_transform=target_transforms)
    
    train_data = TrainData()
    val_data = ValData()
    test_data = TestData()

    assert len(train_data) > 0
    assert len(val_data) > 0
    assert len(test_data) > 0

    collate_fn = partial(person_activity_collate_fn, resolution=(7,), pad_unit=pad_unit, no_time_information=no_time_information)
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=partial(collate_fn, cut_mix=cut_mix),
        eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True
    )

    data = Data(
        n_classes=7, num_embeddings=7, train_size=len(train_data)
    )

    return train_loader, val_loader, test_loader, data

## Walker2d 

class WalkerTonic(Dataset):
    """Person activity dataset for Tonic framework using the Person activity class for data processing."""
    def __init__(
        self,
        data_dir: str = None,
        split: str = "None",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        assert split in ['train', 'val', 'test'], "split must be 'train', 'val', or 'test'"
        self.dtype = np.dtype([("t", float), ("x", (float, 17)), ("p", int)])
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_obj = Walker2dImitationData(data_dir=data_dir,seq_len=64)

        if split == 'train':
            self.data = self.dataset_obj.train_x
            self.times = self.dataset_obj.train_times.squeeze()
            self.target = self.dataset_obj.train_y
        elif split == 'val':
            self.data = self.dataset_obj.test_x
            self.times = self.dataset_obj.test_times.squeeze()
            self.target = self.dataset_obj.test_y
        elif split == "test":
            self.data = self.dataset_obj.test_x
            self.times = self.dataset_obj.test_times.squeeze()
            self.target = self.dataset_obj.test_y
        else:
            raise ValueError

    def __getitem__(self, index):
        input_ids,times,target = self.data[index], self.times[index], self.target[index]

        # Simulating the polarity as 1 and converting to structured array
        events = make_structured_array(times, input_ids, 1, dtype = self.dtype)
        
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        if self.data is None:
            raise ValueError("Data not loaded properly; self.data is None.")
        return len(self.data)
    
def create_walker_tonic_classification_dataset(
        cache_dir: str = None,
        per_device_batch_size: int = 32,
        per_device_eval_batch_size: int = 64,
        world_size: int = 1,
        num_workers: int = 0,
        seed: int = 42,
        pad_unit: int = 2048,
        cut_mix: float = 0.0,
        no_time_information: bool = False,
        **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Data]:
    """
    Creates a view of the Eigenworms dataset for Tonic framework

    :param cache_dir:		    (str):		where to store the dataset
    :param data_dir:		    (str):		where the dataset is located
    :param per_device_batch_size:(int):		Batch size for training.
    :param per_device_eval_batch_size: (int): Batch size for evaluation.
    :param world_size:           (int):       Number of devices for training.
    :param num_workers:          (int):       Number of worker threads for data loading.
    :param seed:                 (int):       Seed for shuffling data.
    :param validate_on_test:     (bool):      If True, use the test set for validation.
                                             Else use a random validation split from the test set.

    :return: train_loader, val_loader, test_loader, data
    """

    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    data_dir = Path(cache_dir).parent / "walker/"

    target_transforms = None 
    
    TrainData = partial(WalkerTonic, data_dir=data_dir, split='train', target_transform=target_transforms)
    ValData = partial(WalkerTonic, data_dir=data_dir, split='val', target_transform=target_transforms)
    TestData = partial(WalkerTonic, data_dir=data_dir, split='test', target_transform=target_transforms)
    
    train_data = TrainData()
    val_data = ValData()
    test_data = TestData()

    assert len(train_data) > 0
    assert len(val_data) > 0
    assert len(test_data) > 0

    collate_fn = partial(person_activity_collate_fn, resolution=(17,), pad_unit=pad_unit, no_time_information=no_time_information)
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=partial(collate_fn, cut_mix=cut_mix),
        eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True
    )

    data = Data(
        n_classes=17, num_embeddings=17, train_size=len(train_data)
    )

    return train_loader, val_loader, test_loader, data

## PTB

import os
import torch

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
        seq_len: int = 70,
    ):
        assert split in ['train', 'val', 'test'], "split must be 'train', 'val', or 'test'"
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.seq_len = seq_len

        # Load corpus and split-specific data
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

    
def create_ptb_tonic_classification_dataset(
    data_dir: str = None,
    cache_dir: Optional[Union[str, Path]] = None,
    per_device_batch_size: int = 32,
    per_device_eval_batch_size: int = 64,
    world_size: int = 1,
    num_workers: int = 0,
    seed: int = 42,
    pad_unit: int = 32,
    cut_mix: float = 0.0,
    no_time_information: bool = False,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Data]:
    """
    Creates a view of the PTB dataset for Tonic framework.

    :param data_dir:		    (str): where the dataset is located
    :param cache_dir:		    (Optional[str]): where to store the dataset cache
    :param per_device_batch_size:(int): Batch size for training.
    :param per_device_eval_batch_size: (int): Batch size for evaluation.
    :param world_size:           (int): Number of devices for training.
    :param num_workers:          (int): Number of worker threads for data loading.
    :param seed:                 (int): Seed for shuffling data.
    :param pad_unit:             (int): Padding unit for the tokens, ensuring sequence lengths are padded appropriately.

    :return: train_loader, val_loader, test_loader, data
    """

    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    cache_dir = data_dir = Path(cache_dir).parent / "ptb"

    transforms = None 
    target_transforms = None 

    TrainData = partial(PTBTonic, data_dir=data_dir, cache_dir=cache_dir, split='train', target_transform=target_transforms, transform=transforms)
    ValData = partial(PTBTonic, data_dir=data_dir, cache_dir=cache_dir, split='val', target_transform=target_transforms, transform=transforms)
    TestData = partial(PTBTonic, data_dir=data_dir, cache_dir=cache_dir, split='test', target_transform=target_transforms, transform=transforms)
    
    train_data = TrainData()
    val_data = ValData()
    test_data = TestData()

    assert len(train_data) > 0
    assert len(val_data) > 0
    assert len(test_data) > 0

    collate_fn = partial(ptb_collate_fn, resolution=(10000,), pad_unit=pad_unit, no_time_information=no_time_information)
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=partial(collate_fn, cut_mix=cut_mix),
        eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True
    )

    data = Data(
        n_classes=10000,  # Vocabulary size
        num_embeddings=10000,  # Adjust based on actual token embedding
        train_size=len(train_data)
    )

    return train_loader, val_loader, test_loader, data


### Wikitext 103

class Wikitext2Tonic(Dataset):
    """PTB dataset for Tonic framework using the Dictionary and Corpus classes for data processing."""

    sensor_size = (1,) 

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        cache_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        seq_len: int = 70,
    ):
        assert split in ['train', 'val', 'test'], "split must be 'train', 'val', or 'test'"
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.seq_len = seq_len

        # Load corpus and split-specific data
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
        # Probabilistically determine sequence length
        # Sequence length varies around `self.seq_len` with some randomness
        bptt = self.seq_len if np.random.random() < 0.95 else self.seq_len / 2
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        seq_len = min(seq_len, self.seq_len + 20) #following https://github.com/ChengyueGongR/advsoft/tree/master

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

    
def create_wikitext2_tonic_classification_dataset(
    data_dir: str = "/data/storage/tsoydan/data/wikitext2",
    cache_dir: Optional[Union[str, Path]] = None,
    per_device_batch_size: int = 32,
    per_device_eval_batch_size: int = 64,
    world_size: int = 1,
    num_workers: int = 0,
    seed: int = 42,
    pad_unit: int = 32,
    cut_mix: float = 0.0,
    no_time_information: bool = False,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Data]:
    """
    Creates a view of the Wikitext2 dataset for Tonic framework.

    :param data_dir:		    (str): where the dataset is located
    :param cache_dir:		    (Optional[str]): where to store the dataset cache
    :param per_device_batch_size:(int): Batch size for training.
    :param per_device_eval_batch_size: (int): Batch size for evaluation.
    :param world_size:           (int): Number of devices for training.
    :param num_workers:          (int): Number of worker threads for data loading.
    :param seed:                 (int): Seed for shuffling data.
    :param pad_unit:             (int): Padding unit for the tokens, ensuring sequence lengths are padded appropriately.

    :return: train_loader, val_loader, test_loader, data
    """

    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None


    cache_dir = data_dir = Path(cache_dir).parent / "wikitext2"

    transforms = None 
    target_transforms = None 

    TrainData = partial(Wikitext2Tonic, data_dir=data_dir, cache_dir=cache_dir, split='train', target_transform=target_transforms, transform=transforms)
    ValData = partial(Wikitext2Tonic, data_dir=data_dir, cache_dir=cache_dir, split='val', target_transform=target_transforms, transform=transforms)
    TestData = partial(Wikitext2Tonic, data_dir=data_dir, cache_dir=cache_dir, split='test', target_transform=target_transforms, transform=transforms)
    
    train_data = TrainData()
    val_data = ValData()
    test_data = TestData()

    assert len(train_data) > 0
    assert len(val_data) > 0
    assert len(test_data) > 0

    collate_fn = partial(ptb_collate_fn, resolution=(84608,), pad_unit=pad_unit, no_time_information=no_time_information)
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=partial(collate_fn, cut_mix=cut_mix),
        eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True
    )

    data = Data(
        n_classes=84608,  # Vocabulary size
        num_embeddings=84608,  # Adjust based on actual token embedding
        train_size=len(train_data)
    )

    return train_loader, val_loader, test_loader, data



### Wikitext 103

class Wikitext103Tonic(Dataset):
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

    
def create_wikitext103_tonic_classification_dataset(
    data_dir: str = None,
    cache_dir: Optional[Union[str, Path]] = None,
    per_device_batch_size: int = 32,
    per_device_eval_batch_size: int = 64,
    world_size: int = 1,
    num_workers: int = 0,
    seed: int = 42,
    pad_unit: int = 32,
    cut_mix: float = 0.0,
    no_time_information: bool = False,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Data]:
    """
    Creates a view of the Wikitext103 dataset for Tonic framework.

    :param data_dir:		    (str): where the dataset is located
    :param cache_dir:		    (Optional[str]): where to store the dataset cache
    :param per_device_batch_size:(int): Batch size for training.
    :param per_device_eval_batch_size: (int): Batch size for evaluation.
    :param world_size:           (int): Number of devices for training.
    :param num_workers:          (int): Number of worker threads for data loading.
    :param seed:                 (int): Seed for shuffling data.
    :param pad_unit:             (int): Padding unit for the tokens, ensuring sequence lengths are padded appropriately.

    :return: train_loader, val_loader, test_loader, data
    """

    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    cache_dir = data_dir = Path(cache_dir).parent / "wikitext103"

    transforms = None 
    target_transforms = None 

    TrainData = partial(Wikitext103Tonic, data_dir=data_dir, cache_dir=cache_dir, split='train', target_transform=target_transforms, transform=transforms)
    ValData = partial(Wikitext103Tonic, data_dir=data_dir, cache_dir=cache_dir, split='val', target_transform=target_transforms, transform=transforms)
    TestData = partial(Wikitext103Tonic, data_dir=data_dir, cache_dir=cache_dir, split='test', target_transform=target_transforms, transform=transforms)
    
    train_data = TrainData()
    val_data = ValData()
    test_data = TestData()

    assert len(train_data) > 0
    assert len(val_data) > 0
    assert len(test_data) > 0

    collate_fn = partial(ptb_collate_fn, resolution=(267735,), pad_unit=pad_unit, no_time_information=no_time_information)
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=partial(collate_fn, cut_mix=cut_mix),
        eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True
    )

    data = Data(
        n_classes=267735,  # Vocabulary size
        num_embeddings=267735,  # Adjust based on actual token embedding
        train_size=len(train_data)
    )

    return train_loader, val_loader, test_loader, data




Datasets = {
    "shd-classification": create_events_shd_classification_dataset,
    "ssc-classification": create_events_ssc_classification_dataset,
    "dvs-gesture-classification": create_events_dvs_gesture_classification_dataset,
    "listops-classification": create_listops_tonic_classification_dataset,
    "text-classification": create_imdb_tonic_classification_dataset,
    "retrieval-classification": create_retrieval_tonic_classification_dataset,
    "image-classification": create_image_tonic_classification_dataset,
    "pathfinder-classification": create_pathfinder_tonic_classification_dataset,
    "pathx-classification":create_pathx_tonic_classification_dataset,
    "eigenworms-classification":create_eigenworms_tonic_classification_dataset,
    "personactivity-classification":create_person_activity_tonic_classification_dataset,
    "walker-classification":create_walker_tonic_classification_dataset,
    "ptb-classification": create_ptb_tonic_classification_dataset,
    "wikitext2-classification": create_wikitext2_tonic_classification_dataset,
    "wikitext103-classification": create_wikitext103_tonic_classification_dataset,
}

