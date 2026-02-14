"""Time-series datasets: EigenWorms, PersonActivity, and Walker2d."""

import os
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tonic.io import make_structured_array

from src.transform import OneHotLabels
from .collate import (
    Data, DataLoader,
    eigenworms_collate_fn, person_activity_collate_fn,
    event_stream_dataloader,
)


# ============================================================
# Inlined dataset classes (originally from odelstms)
# ============================================================

class PersonData:
    """Person activity dataset loader (originally from ODE-LSTM codebase)."""

    class_map = {
        "lying down": 0, "lying": 0,
        "sitting down": 1, "sitting": 1,
        "standing up from lying": 2, "standing up from sitting": 2,
        "standing up from sitting on the ground": 2,
        "walking": 3, "falling": 4,
        "on all fours": 5, "sitting on the ground": 6,
    }
    sensor_ids = {
        "010-000-024-033": 0, "010-000-030-096": 1,
        "020-000-033-111": 2, "020-000-032-221": 3,
    }

    def __init__(self, data_dir=None, seq_len=32):
        self.dataset_path = data_dir
        self.seq_len = seq_len
        self.num_classes = 7
        all_x, all_t, all_y = self.load_csv()
        all_x, all_t, all_y = self.cut_in_sequences(all_x, all_t, all_y, seq_len=seq_len, inc=seq_len // 2)

        total_seqs = all_x.shape[0]
        permutation = np.random.RandomState(98841).permutation(total_seqs)
        test_size = int(0.2 * total_seqs)

        self.test_x = all_x[permutation[:test_size]]
        self.test_y = all_y[permutation[:test_size]]
        self.test_t = all_t[permutation[:test_size]]
        self.train_x = all_x[permutation[test_size:]]
        self.train_t = all_t[permutation[test_size:]]
        self.train_y = all_y[permutation[test_size:]]
        self.feature_size = int(self.train_x.shape[-1])

    def load_csv(self):
        all_x, all_y, all_t = [], [], []
        series_x, series_t, series_y = [], [], []
        last_millis = None

        if not os.path.isfile(self.dataset_path):
            raise FileNotFoundError(
                f"Dataset file not found: {self.dataset_path}. "
                "Please download the ConfLongDemo_JSI.txt dataset."
            )

        with open(self.dataset_path, "r") as f:
            current_person = "A01"
            for line in f:
                arr = line.split(",")
                if len(arr) < 6:
                    break
                if arr[0] != current_person:
                    series_x = np.stack(series_x, axis=0)
                    series_t = np.stack(series_t, axis=0)
                    series_y = np.array(series_y, dtype=np.int32)
                    all_x.append(series_x)
                    all_t.append(series_t)
                    all_y.append(series_y)
                    last_millis = None
                    series_x, series_y, series_t = [], [], []

                millis = np.int64(arr[2]) / (100 * 1000)
                millis_mapped_to_1 = 10.0
                if last_millis is None:
                    elapsed_sec = 0.05
                else:
                    elapsed_sec = float(millis - last_millis) / 1000.0
                elapsed = elapsed_sec * 1000 / millis_mapped_to_1

                last_millis = millis
                current_person = arr[0]
                sensor_id = self.sensor_ids[arr[1]]
                label_col = self.class_map[arr[7].replace("\n", "")]
                feature_col_2 = np.array(arr[4:7], dtype=np.float32)
                feature_col_1 = np.zeros(4, dtype=np.float32)
                feature_col_1[sensor_id] = 1
                feature_col = np.concatenate([feature_col_1, feature_col_2])

                series_x.append(feature_col)
                series_t.append(elapsed)
                series_y.append(label_col)

        return all_x, all_t, all_y

    def cut_in_sequences(self, all_x, all_t, all_y, seq_len, inc=1):
        sequences_x, sequences_t, sequences_y = [], [], []
        for i in range(len(all_x)):
            x, t, y = all_x[i], all_t[i], all_y[i]
            for s in range(0, x.shape[0] - seq_len, inc):
                start = s
                end = start + seq_len
                sequences_x.append(x[start:end])
                sequences_t.append(t[start:end])
                sequences_y.append(y[start:end])
        return (
            np.stack(sequences_x, axis=0),
            np.stack(sequences_t, axis=0).reshape([-1, seq_len, 1]),
            np.stack(sequences_y, axis=0),
        )


class Walker2dImitationData:
    """Walker2d imitation learning dataset loader (originally from ODE-LSTM codebase)."""

    def __init__(self, data_dir, seq_len):
        self.seq_len = seq_len
        all_files = sorted([
            os.path.join(data_dir, d) for d in os.listdir(data_dir) if d.endswith(".npy")
        ])

        self.rng = np.random.RandomState(891374)
        np.random.RandomState(125487).shuffle(all_files)
        test_n = int(0.15 * len(all_files))
        valid_n = int((0.15 + 0.1) * len(all_files))
        test_files = all_files[:test_n]
        valid_files = all_files[test_n:valid_n]
        train_files = all_files[valid_n:]

        train_x, train_t, train_y = self._load_files(train_files)
        valid_x, valid_t, valid_y = self._load_files(valid_files)
        test_x, test_t, test_y = self._load_files(test_files)

        train_x, train_t, train_y = self.perturb_sequences(train_x, train_t, train_y)
        valid_x, valid_t, valid_y = self.perturb_sequences(valid_x, valid_t, valid_y)
        test_x, test_t, test_y = self.perturb_sequences(test_x, test_t, test_y)

        self.train_x, self.train_times, self.train_y = self.align_sequences(train_x, train_t, train_y)
        self.valid_x, self.valid_times, self.valid_y = self.align_sequences(valid_x, valid_t, valid_y)
        self.test_x, self.test_times, self.test_y = self.align_sequences(test_x, test_t, test_y)
        self.input_size = self.train_x.shape[-1]

    def align_sequences(self, set_x, set_t, set_y):
        times, x, y = [], [], []
        for i in range(len(set_y)):
            seq_x, seq_t, seq_y = set_x[i], set_t[i], set_y[i]
            for t in range(0, seq_y.shape[0] - self.seq_len, self.seq_len // 4):
                x.append(seq_x[t: t + self.seq_len])
                times.append(seq_t[t: t + self.seq_len])
                y.append(seq_y[t: t + self.seq_len])
        return (
            np.stack(x, axis=0),
            np.expand_dims(np.stack(times, axis=0), axis=-1),
            np.stack(y, axis=0),
        )

    def perturb_sequences(self, set_x, set_t, set_y):
        x, times, y = [], [], []
        for i in range(len(set_y)):
            seq_x, seq_y = set_x[i], set_y[i]
            new_x, new_times, new_y = [], [], []
            skip = 0
            for t in range(seq_y.shape[0]):
                skip += 1
                if self.rng.rand() < 0.9:
                    new_x.append(seq_x[t])
                    new_times.append(skip)
                    new_y.append(seq_y[t])
                    skip = 0
            x.append(np.stack(new_x, axis=0))
            times.append(np.stack(new_times, axis=0))
            y.append(np.stack(new_y, axis=0))
        return x, times, y

    def _load_files(self, files):
        all_x, all_t, all_y = [], [], []
        for f in files:
            arr = np.load(f)
            x_state = arr[:-1, :].astype(np.float32)
            y = arr[1:, :].astype(np.float32)
            x_times = np.ones(x_state.shape[0])
            all_x.append(x_state)
            all_t.append(x_times)
            all_y.append(y)
        return all_x, all_t, all_y


# ============================================================
# Tonic wrapper classes
# ============================================================

class EigenWormsTonic(Dataset):
    """EigenWorms dataset wrapped for the Tonic event-stream framework."""

    def __init__(self, data_dir, split="None", transform=None, target_transform=None):
        assert split in ['train', 'val', 'test']
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
        input_ids, target = self.data[index]
        times = np.arange(0, len(input_ids))
        events = make_structured_array(times, input_ids, 1, dtype=self.dtype)
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        if self.data is None:
            raise ValueError("Data not loaded properly; self.data is None.")
        return len(self.data)


class PersonActivityTonic(Dataset):
    """Person activity dataset wrapped for the Tonic event-stream framework."""

    def __init__(self, data_dir="None", split="None", transform=None, target_transform=None):
        assert split in ['train', 'val', 'test']
        self.dtype = np.dtype([("t", float), ("x", (float, 7)), ("p", int)])
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_obj = PersonData(data_dir=data_dir)

        if split == 'train':
            self.data = self.dataset_obj.train_x
            self.times = self.dataset_obj.train_t.squeeze()
            self.target = self.dataset_obj.train_y
        else:
            # Both val and test use test split
            self.data = self.dataset_obj.test_x
            self.times = self.dataset_obj.test_t.squeeze()
            self.target = self.dataset_obj.test_y

    def __getitem__(self, index):
        input_ids, times, target = self.data[index], self.times[index], self.target[index]
        events = make_structured_array(times, input_ids, 1, dtype=self.dtype)
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        if self.data is None:
            raise ValueError("Data not loaded properly; self.data is None.")
        return len(self.data)


class WalkerTonic(Dataset):
    """Walker2d imitation dataset wrapped for the Tonic event-stream framework."""

    def __init__(self, data_dir=None, split="None", transform=None, target_transform=None):
        assert split in ['train', 'val', 'test']
        self.dtype = np.dtype([("t", float), ("x", (float, 17)), ("p", int)])
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_obj = Walker2dImitationData(data_dir=data_dir, seq_len=64)

        if split == 'train':
            self.data = self.dataset_obj.train_x
            self.times = self.dataset_obj.train_times.squeeze()
            self.target = self.dataset_obj.train_y
        else:
            self.data = self.dataset_obj.test_x
            self.times = self.dataset_obj.test_times.squeeze()
            self.target = self.dataset_obj.test_y

    def __getitem__(self, index):
        input_ids, times, target = self.data[index], self.times[index], self.target[index]
        events = make_structured_array(times, input_ids, 1, dtype=self.dtype)
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        if self.data is None:
            raise ValueError("Data not loaded properly; self.data is None.")
        return len(self.data)


# ============================================================
# Factory functions
# ============================================================

def create_eigenworms_tonic_classification_dataset(
    cache_dir=None, per_device_batch_size=32, per_device_eval_batch_size=64,
    world_size=1, num_workers=0, seed=42,
    pad_unit=2048, cut_mix=0.0, no_time_information=False, **kwargs,
):
    """Create dataloaders for the EigenWorms time-series classification task."""
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

    train_data, val_data, test_data = TrainData(), ValData(), TestData()
    assert len(train_data) > 0 and len(val_data) > 0 and len(test_data) > 0

    collate_fn = partial(eigenworms_collate_fn, resolution=(6,), pad_unit=pad_unit, no_time_information=no_time_information)
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=partial(collate_fn, cut_mix=cut_mix), eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True,
    )
    data = Data(n_classes=5, num_embeddings=6, train_size=len(train_data))
    return train_loader, val_loader, test_loader, data


def create_person_activity_tonic_classification_dataset(
    cache_dir=None, per_device_batch_size=32, per_device_eval_batch_size=64,
    world_size=1, num_workers=0, seed=42,
    pad_unit=2048, cut_mix=0.0, no_time_information=False, **kwargs,
):
    """Create dataloaders for the PersonActivity time-series classification task."""
    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    data_dir = Path(cache_dir).parent / "person/ConfLongDemo_JSI.txt"

    target_transforms = OneHotLabels(num_classes=7)
    TrainData = partial(PersonActivityTonic, data_dir=data_dir, split='train', target_transform=target_transforms)
    ValData = partial(PersonActivityTonic, data_dir=data_dir, split='val', target_transform=target_transforms)
    TestData = partial(PersonActivityTonic, data_dir=data_dir, split='test', target_transform=target_transforms)

    train_data, val_data, test_data = TrainData(), ValData(), TestData()
    assert len(train_data) > 0 and len(val_data) > 0 and len(test_data) > 0

    collate_fn = partial(person_activity_collate_fn, resolution=(7,), pad_unit=pad_unit, no_time_information=no_time_information)
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=partial(collate_fn, cut_mix=cut_mix), eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True,
    )
    data = Data(n_classes=7, num_embeddings=7, train_size=len(train_data))
    return train_loader, val_loader, test_loader, data


def create_walker_tonic_classification_dataset(
    cache_dir=None, per_device_batch_size=32, per_device_eval_batch_size=64,
    world_size=1, num_workers=0, seed=42,
    pad_unit=2048, cut_mix=0.0, no_time_information=False, **kwargs,
):
    """Create dataloaders for the Walker2d imitation time-series task."""
    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    data_dir = Path(cache_dir).parent / "walker/"

    TrainData = partial(WalkerTonic, data_dir=data_dir, split='train')
    ValData = partial(WalkerTonic, data_dir=data_dir, split='val')
    TestData = partial(WalkerTonic, data_dir=data_dir, split='test')

    train_data, val_data, test_data = TrainData(), ValData(), TestData()
    assert len(train_data) > 0 and len(val_data) > 0 and len(test_data) > 0

    collate_fn = partial(person_activity_collate_fn, resolution=(17,), pad_unit=pad_unit, no_time_information=no_time_information)
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=partial(collate_fn, cut_mix=cut_mix), eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True,
    )
    data = Data(n_classes=17, num_embeddings=17, train_size=len(train_data))
    return train_loader, val_loader, test_loader, data
