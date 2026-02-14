"""Event stream datasets: SHD, SSC, and DVS Gesture."""

from functools import partial
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
import tonic
import tonic.datasets
import tonic.slicers
import tonic.sliced_dataset
import tonic.transforms

from src.transform import (
    Identity, Roll, Rotate, Scale,
    DropEventChunk, Jitter1D, OneHotLabels,
)
from .collate import (
    Data, DataLoader,
    event_stream_collate_fn, event_stream_dataloader,
)

DEFAULT_CACHE_DIR_ROOT = Path('./cache_dir/')


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
    **kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader, Data]:
    """Create dataloaders for the Spiking Heidelberg Digits (SHD) dataset."""
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
        tonic.transforms.UniformNoise(sensor_size=sensor_size, n=(0, noise)),
    ])
    target_transforms = OneHotLabels(num_classes=20)

    train_data = tonic.datasets.SHD(save_to=cache_dir, train=True, transform=transforms, target_transform=target_transforms)
    val_data = tonic.datasets.SHD(save_to=cache_dir, train=True, target_transform=target_transforms)
    test_data = tonic.datasets.SHD(save_to=cache_dir, train=False, target_transform=target_transforms)

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
        batch_size=per_device_batch_size * world_size,
        eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True,
    )
    data = Data(n_classes=20, num_embeddings=700, train_size=len(train_data))
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
    **kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader, Data]:
    """Create dataloaders for the Spiking Speech Commands (SSC) dataset."""
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
        tonic.transforms.UniformNoise(sensor_size=sensor_size, n=(0, noise)),
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
        batch_size=per_device_batch_size * world_size,
        eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True,
    )
    data = Data(n_classes=35, num_embeddings=700, train_size=len(train_data))
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
    **kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader, Data]:
    """Create dataloaders for the DVS Gesture dataset."""
    assert time_skew > 1, "time_skew must be greater than 1"

    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    orig_sensor_size = (128, 128, 2)
    new_sensor_size = (128 // downsampling, 128 // downsampling, 2)
    train_transforms = [
        DropEventChunk(p=0.3, max_drop_size=max_drop_chunk),
        tonic.transforms.DropEvent(p=drop_event),
        tonic.transforms.UniformNoise(sensor_size=new_sensor_size, n=(0, noise)),
        tonic.transforms.TimeSkew(coefficient=(1 / time_skew, time_skew), offset=0),
        tonic.transforms.TimeJitter(std=time_jitter, clip_negative=False, sort_timestamps=True),
        tonic.transforms.SpatialJitter(
            sensor_size=orig_sensor_size, var_x=spatial_jitter,
            var_y=spatial_jitter, clip_outliers=True,
        ),
        tonic.transforms.Downsample(
            sensor_size=orig_sensor_size, target_size=new_sensor_size[:2]
        ) if downsampling > 1 else Identity(),
        Roll(sensor_size=new_sensor_size, p=0.3, max_roll=max_roll),
        Rotate(sensor_size=new_sensor_size, p=0.3, max_angle=max_angle),
        Scale(sensor_size=new_sensor_size, p=0.3, max_scale=max_scale),
    ]

    train_transforms = tonic.transforms.Compose(train_transforms)
    test_transforms = tonic.transforms.Compose([
        tonic.transforms.Downsample(
            sensor_size=orig_sensor_size, target_size=new_sensor_size[:2]
        ) if downsampling > 1 else Identity(),
    ])
    target_transforms = OneHotLabels(num_classes=11)

    TrainData = partial(tonic.datasets.DVSGesture, save_to=cache_dir, train=True)
    TestData = partial(tonic.datasets.DVSGesture, save_to=cache_dir, train=False)

    if validate_on_test:
        val_data = TestData(transform=test_transforms, target_transform=target_transforms)
    else:
        val_data = TrainData(transform=test_transforms, target_transform=target_transforms)
        val_length = int(0.2 * len(val_data))
        indices = torch.randperm(len(val_data), generator=rng)
        val_data = torch.utils.data.Subset(val_data, indices[-val_length:])

    if slice_events > 0:
        slicer = tonic.slicers.SliceByEventCount(
            event_count=slice_events, overlap=slice_events // 2, include_incomplete=True,
        )
        train_subset = (
            torch.utils.data.Subset(TrainData(), indices[:-val_length])
            if not validate_on_test else TrainData()
        )
        train_data = tonic.sliced_dataset.SlicedDataset(
            dataset=train_subset, slicer=slicer,
            transform=train_transforms, target_transform=target_transforms,
            metadata_path=None,
        )
        if slice_val_set:
            val_subset = TestData()
            val_data = tonic.sliced_dataset.SlicedDataset(
                dataset=val_subset, slicer=slicer,
                transform=test_transforms, target_transform=target_transforms,
                metadata_path=None,
            )
    else:
        train_data = torch.utils.data.Subset(
            TrainData(transform=train_transforms, target_transform=target_transforms),
            indices[:-val_length],
        ) if not validate_on_test else TrainData(transform=train_transforms)

    test_data = TestData(transform=test_transforms, target_transform=target_transforms)

    train_collate_fn = partial(
        event_stream_collate_fn,
        resolution=new_sensor_size[:2],
        pad_unit=slice_events if (slice_events != 0 and slice_events < pad_unit) else pad_unit,
        cut_mix=cut_mix,
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
        batch_size=per_device_batch_size * world_size,
        eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True,
    )
    data = Data(n_classes=11, num_embeddings=np.prod(new_sensor_size), train_size=len(train_data))
    return train_loader, val_loader, test_loader, data
