## Retrieval

from torch.utils.data import Dataset
from tonic.io import make_structured_array
from S5.s5.dataloaders.lra import *
from typing import Callable, Optional, TypeVar, Dict, Tuple, List, Union
import torch
from pathlib import Path
from typing import Callable, Optional, TypeVar, Dict, Tuple, List, Union
import tonic
from functools import partial
import numpy as np
from torch.utils.data import Dataset
from tonic.io import make_structured_array
from S5.s5.dataloaders.lra import *
from transform import Identity, Roll, Rotate, Scale, DropEventChunk, Jitter1D, OneHotLabels, cut_mix_augmentation

DataLoader = TypeVar('DataLoader')
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



#aan = AAN(_name_="aan",data_dir="/data/storage/tsoydan/data/long-range-arena/retrieval/", cache_dir="/data/storage/tsoydan/data/long-range-arena/retrieval/")
#aan.prepare_data()
#aan.setup(stage="train")
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


def retrieval_collate_fn(batch, resolution, pad_unit, no_time_information=True, tokenize="unique"):
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
    # x1, x2 are inputs, y are targets, z are aux data
    x1, x2, y, *z = zip(*batch)
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

    # concatenate tokens and timesteps
    tokens = np.concatenate((tokens1, tokens2), axis=1)
    timesteps = np.concatenate((timesteps1, timesteps2), axis=1)
    lengths = np.concatenate((lengths1, lengths2))

    if batch_size_one:
        lengths = lengths[None, ...]

    return tokens, y, timesteps, lengths


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
        self.dtype = np.dtype([("t", int), ("x", int), ("p", int)])
        self.transform = transform
        self.target_transform = target_transform
        self.aan = AAN(_name_="aan",data_dir=data_dir, cache_dir=cache_dir, train=(split=='train'))
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
        
        return events1, events2, target

    def __len__(self):
        return len(self.data)
    
def create_retrieval_tonic_classification_dataset(
        cache_dir: Union[str, Path] = "/data/storage/tsoydan/data/long-range-arena/retrieval/",
        data_dir: str = "/data/storage/tsoydan/data/long-range-arena/retrieval/",
        per_device_batch_size: int = 512,
        per_device_eval_batch_size: int = 512,
        world_size: int = 1,
        num_workers: int = 32,
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
    print("[*] Generating AAN Classification Dataset")

    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    target_transforms = OneHotLabels(num_classes=2)
    TrainData = partial(AANTonic,data_dir=data_dir, cache_dir=cache_dir, split='train', target_transform=target_transforms)
    ValData = partial(AANTonic,data_dir=data_dir, cache_dir=cache_dir, split='val', target_transform=target_transforms)
    TestData = partial(AANTonic,data_dir=data_dir, cache_dir=cache_dir, split='test', target_transform=target_transforms)
    
    train_data = TrainData()
    val_data = ValData()
    test_data = TestData()

    collate_fn = partial(retrieval_collate_fn, resolution=(18,), pad_unit=pad_unit, no_time_information=no_time_information)
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=collate_fn,
        eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True
    )
    
    data = Data(
        n_classes=2, num_embeddings=18, train_size=len(train_data)
    )

    return train_loader, val_loader, test_loader, data

train_loader, val_loader, test_loader, data = create_retrieval_tonic_classification_dataset()

# Print the first batch from the train_loader
for batch in train_loader:
    a,b,c,d = batch
    print(np.max(a))
    break



assert 1 == 2







# Text
from torch.utils.data import Dataset
from tonic.io import make_structured_array
from S5.s5.dataloaders.lra import *
from typing import Callable, Optional, TypeVar, Dict, Tuple, List, Union


class IMDBTonic(Dataset):
    """IMDB dataset for Tonic framework using the IMDB class for data processing."""

    def __init__(
        self,
        data_dir: str,
        cache_dir: Optional[str] = None,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        assert split in ['train', 'test'], "split must be 'train' or 'test'"
        self.sensor_size = (129, 1, 1)
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
        cache_dir: Union[str, Path] = "/data/storage/tsoydan/data/long-range-arena/text/",
        data_dir: str = "/data/storage/tsoydan/data/long-range-arena/text/",
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
    print("[*] Generating IMDB Classification Dataset")

    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    target_transforms = None  # Define if needed
    TrainData = partial(IMDBTonic, data_dir=data_dir, cache_dir=cache_dir, split='train', target_transform=target_transforms)
    TestData = partial(IMDBTonic, data_dir=data_dir, cache_dir=cache_dir, split='test', target_transform=target_transforms)
    
    train_data = TrainData()
    test_data = TestData()
    val_data = test_data  # Use test set as validation set 

    collate_fn = partial(event_stream_collate_fn, resolution=(129,), pad_unit=pad_unit, no_time_information=no_time_information)
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=partial(collate_fn, cut_mix=cut_mix),
        eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True
    )
    
    return train_loader, val_loader, test_loader


print("#"*100)

import torch
from torch.utils.data import DataLoader

# Assuming the IMDBTonic class and create_imdb_tonic_classification_dataset function are defined as above

# Parameters for creating the dataset and dataloader
cache_dir = "/data/storage/tsoydan/data/long-range-arena/text/"
data_dir = "/data/storage/tsoydan/data/long-range-arena/text/"
per_device_batch_size = 32
num_workers = 16

# Create the dataset and dataloaders
train_loader, val_loader, test_loader = create_imdb_tonic_classification_dataset(
    cache_dir=cache_dir,
    data_dir=data_dir,
    per_device_batch_size=per_device_batch_size,
    per_device_eval_batch_size=per_device_batch_size,
    world_size=1,
    num_workers=num_workers
)

# Get a batch of data from the train loader
batch = next(iter(train_loader))

# Print the batch data
events, labels = batch
print("Events: ", events)
print("Labels: ", labels)