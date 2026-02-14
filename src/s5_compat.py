"""
Compatibility layer for S5 base classes and utilities.

Provides the SequenceDataset base class and related infrastructure originally
from the S5 project (https://github.com/lindermanlab/S5). These classes are
used by the LRA and image dataset loaders.
"""

import math
from functools import partial
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torchaudio.functional as TF
import torchvision
from einops import rearrange


# --- Utilities ---

def is_list(x):
    return isinstance(x, Sequence) and not isinstance(x, str)


# --- Permutation utilities (from S4 codebase) ---

def bitreversal_po2(n):
    m = int(math.log(n) / math.log(2))
    perm = np.arange(n).reshape(n, 1)
    for i in range(m):
        n1 = perm.shape[0] // 2
        perm = np.hstack((perm[:n1], perm[n1:]))
    return perm.squeeze(0)


def bitreversal_permutation(n):
    m = int(math.ceil(math.log(n) / math.log(2)))
    N = 1 << m
    perm = bitreversal_po2(N)
    return np.extract(perm < n, perm)


def transpose_permutation(h, w):
    indices = np.arange(h * w).reshape((h, w)).T.reshape(h * w)
    return indices


def snake_permutation(h, w):
    indices = np.arange(h * w).reshape((h, w))
    indices[1::2, :] = indices[1::2, ::-1]
    return indices.reshape(h * w)


def hilbert_permutation(n):
    m = int(math.log2(n))
    assert n == 2 ** m
    inds = _hilbert_decode(list(range(n * n)), 2, m)
    ind_x, ind_y = inds.T
    indices = np.arange(n * n).reshape((n, n))
    return indices[ind_x, ind_y]


def _hilbert_decode(hilberts, num_dims, num_bits):
    """Decode Hilbert integers into hypercube locations (from numpy-hilbert-curve)."""
    if num_dims * num_bits > 64:
        raise ValueError(
            f"num_dims={num_dims} and num_bits={num_bits} for {num_dims * num_bits} "
            f"bits total, which can't be encoded into a uint64."
        )
    hilberts = np.atleast_1d(hilberts)
    orig_shape = hilberts.shape
    hh_uint8 = np.reshape(hilberts.ravel().astype('>u8').view(np.uint8), (-1, 8))
    hh_bits = np.unpackbits(hh_uint8, axis=1)[:, -num_dims * num_bits:]
    gray = _binary2gray(hh_bits)
    gray = np.swapaxes(np.reshape(gray, (-1, num_bits, num_dims)), axis1=1, axis2=2)

    for bit in range(num_bits - 1, -1, -1):
        for dim in range(num_dims - 1, -1, -1):
            mask = gray[:, dim, bit]
            gray[:, 0, bit + 1:] = np.logical_xor(gray[:, 0, bit + 1:], mask[:, np.newaxis])
            to_flip = np.logical_and(
                np.logical_not(mask[:, np.newaxis]),
                np.logical_xor(gray[:, 0, bit + 1:], gray[:, dim, bit + 1:])
            )
            gray[:, dim, bit + 1:] = np.logical_xor(gray[:, dim, bit + 1:], to_flip)
            gray[:, 0, bit + 1:] = np.logical_xor(gray[:, 0, bit + 1:], to_flip)

    extra_dims = 64 - num_bits
    padded = np.pad(gray, ((0, 0), (0, 0), (extra_dims, 0)), mode='constant', constant_values=0)
    locs_chopped = np.reshape(padded[:, :, ::-1], (-1, num_dims, 8, 8))
    locs_uint8 = np.squeeze(np.packbits(locs_chopped, bitorder='little', axis=3))
    flat_locs = locs_uint8.view(np.uint64)
    return np.reshape(flat_locs, (*orig_shape, num_dims))


def _right_shift(binary, k=1, axis=-1):
    if binary.shape[axis] <= k:
        return np.zeros_like(binary)
    padding = [(0, 0)] * len(binary.shape)
    padding[axis] = (k, 0)
    slicing = [slice(None)] * len(binary.shape)
    slicing[axis] = slice(None, -k)
    return np.pad(binary[tuple(slicing)], padding, mode='constant', constant_values=0)


def _binary2gray(binary, axis=-1):
    shifted = _right_shift(binary, axis=axis)
    return np.logical_xor(binary, shifted)


# --- Default data path ---

default_data_path = Path(__file__).parent.parent.absolute() / "raw_data"


# --- Collate Mixins ---

class DefaultCollateMixin:
    """Controls collating in the DataLoader."""

    @classmethod
    def _collate_callback(cls, x, *args, **kwargs):
        return x

    _collate_arg_names = []

    @classmethod
    def _return_callback(cls, return_value, *args, **kwargs):
        x, y, *z = return_value
        assert len(z) == len(cls._collate_arg_names)
        return x, y, {k: v for k, v in zip(cls._collate_arg_names, z)}

    @classmethod
    def _collate(cls, batch, *args, **kwargs):
        elem = batch[0]
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                numel = sum(x.numel() for x in batch)
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            x = torch.stack(batch, dim=0, out=out)
            x = cls._collate_callback(x, *args, **kwargs)
            return x
        else:
            return torch.tensor(batch)

    @classmethod
    def _collate_fn(cls, batch, *args, **kwargs):
        x, y, *z = zip(*batch)
        x = cls._collate(x, *args, **kwargs)
        y = cls._collate(y)
        z = [cls._collate(z_) for z_ in z]
        return_value = (x, y, *z)
        return cls._return_callback(return_value, *args, **kwargs)

    collate_args = []

    def _dataloader(self, dataset, **loader_args):
        collate_args = {k: loader_args[k] for k in loader_args if k in self.collate_args}
        loader_args = {k: loader_args[k] for k in loader_args if k not in self.collate_args}
        loader_cls = loader_registry[loader_args.pop("_name_", None)]
        return loader_cls(
            dataset=dataset,
            collate_fn=partial(self._collate_fn, **collate_args),
            **loader_args,
        )


class SequenceResolutionCollateMixin(DefaultCollateMixin):

    @classmethod
    def _collate_callback(cls, x, resolution=None):
        if resolution is None:
            pass
        elif is_list(resolution):
            x = x.squeeze(-1)
            L = x.size(1)
            x = x[:, ::resolution[0]]
            _L = L // resolution[0]
            for r in resolution[1:]:
                x = TF.resample(x, _L, L // r)
                _L = L // r
            x = x.unsqueeze(-1)
        else:
            assert x.ndim >= 2
            n_resaxes = max(1, x.ndim - 2)
            lhs = "b " + " ".join([f"(l{i} res{i})" for i in range(n_resaxes)]) + " ..."
            rhs = " ".join([f"res{i}" for i in range(n_resaxes)]) + " b " + " ".join([f"l{i}" for i in range(n_resaxes)]) + " ..."
            x = rearrange(x, lhs + " -> " + rhs, **{f'res{i}': resolution for i in range(n_resaxes)})
            x = x[tuple([0] * n_resaxes)]
        return x

    @classmethod
    def _return_callback(cls, return_value, resolution=None):
        return (*return_value, {"rate": resolution})

    collate_args = ['resolution']


class ImageResolutionCollateMixin(SequenceResolutionCollateMixin):
    _interpolation = torchvision.transforms.InterpolationMode.BILINEAR
    _antialias = True

    @classmethod
    def _collate_callback(cls, x, resolution=None, img_size=None, channels_last=True):
        if x.ndim < 4:
            return super()._collate_callback(x, resolution=resolution)
        if img_size is None:
            x = super()._collate_callback(x, resolution=resolution)
        else:
            x = rearrange(x, 'b ... c -> b c ...') if channels_last else x
            _size = round(img_size / resolution)
            x = torchvision.transforms.functional.resize(
                x, size=[_size, _size],
                interpolation=cls._interpolation, antialias=cls._antialias,
            )
            x = rearrange(x, 'b c ... -> b ... c') if channels_last else x
        return x

    @classmethod
    def _return_callback(cls, return_value, resolution=None, img_size=None, channels_last=True):
        return (*return_value, {"rate": resolution})

    collate_args = ['resolution', 'img_size', 'channels_last']


# --- DataLoader variants ---

class TBPTTDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, chunk_len, overlap_len, *args, **kwargs):
        super().__init__(dataset, batch_size, *args, **kwargs)
        assert chunk_len is not None and overlap_len is not None
        self.zero = dataset.zero if hasattr(dataset, "zero") else 0
        self.chunk_len = chunk_len
        self.overlap_len = overlap_len

    def __iter__(self):
        for batch in super().__iter__():
            x, y, z = batch
            pad = lambda x, val: torch.cat([x.new_zeros((x.shape[0], self.overlap_len - 1, *x.shape[2:])) + val, x], dim=1)
            x = pad(x, self.zero)
            y = pad(y, 0)
            z = {k: pad(v, 0) for k, v in z.items() if v.ndim > 1}
            _, seq_len, *_ = x.shape
            reset = True
            for seq_begin in list(range(self.overlap_len - 1, seq_len, self.chunk_len))[:-1]:
                from_index = seq_begin - self.overlap_len + 1
                to_index = seq_begin + self.chunk_len
                if self.overlap_len > 0:
                    to_index = min(to_index, seq_len - ((seq_len - self.overlap_len + 1) % self.overlap_len))
                x_chunk = x[:, from_index:to_index]
                if len(y.shape) == 3:
                    y_chunk = y[:, seq_begin:to_index]
                else:
                    y_chunk = y
                z_chunk = {k: v[:, from_index:to_index] for k, v in z.items() if len(v.shape) > 1}
                yield (x_chunk, y_chunk, {**z_chunk, "reset": reset})
                reset = False

    def __len__(self):
        raise NotImplementedError()


loader_registry = {
    "tbptt": TBPTTDataLoader,
    None: torch.utils.data.DataLoader,
}


# --- Base Dataset Classes ---

class SequenceDataset(DefaultCollateMixin):
    """Base class for sequence datasets."""
    registry = {}
    _name_ = NotImplementedError("Dataset must have shorthand name")

    @property
    def init_defaults(self):
        return {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registry[cls._name_] = cls

    def __init__(self, _name_, data_dir=None, **dataset_cfg):
        assert _name_ == self._name_
        self.data_dir = Path(data_dir).absolute() if data_dir is not None else None

        init_args = self.init_defaults.copy()
        init_args.update(dataset_cfg)
        for k, v in init_args.items():
            setattr(self, k, v)

        self.dataset_train = self.dataset_val = self.dataset_test = None
        self.init()

    def init(self):
        pass

    def setup(self):
        raise NotImplementedError

    def split_train_val(self, val_split):
        train_len = int(len(self.dataset_train) * (1.0 - val_split))
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(
            self.dataset_train,
            (train_len, len(self.dataset_train) - train_len),
            generator=torch.Generator().manual_seed(getattr(self, "seed", 42)),
        )

    def train_dataloader(self, **kwargs):
        return self._train_dataloader(self.dataset_train, **kwargs)

    def _train_dataloader(self, dataset, **kwargs):
        if dataset is None:
            return
        kwargs['shuffle'] = 'sampler' not in kwargs
        return self._dataloader(dataset, **kwargs)

    def val_dataloader(self, **kwargs):
        return self._eval_dataloader(self.dataset_val, **kwargs)

    def test_dataloader(self, **kwargs):
        return self._eval_dataloader(self.dataset_test, **kwargs)

    def _eval_dataloader(self, dataset, **kwargs):
        if dataset is None:
            return
        return self._dataloader(dataset, **kwargs)

    def __str__(self):
        return self._name_


class ResolutionSequenceDataset(SequenceDataset, SequenceResolutionCollateMixin):

    def _train_dataloader(self, dataset, train_resolution=None, eval_resolutions=None, **kwargs):
        if train_resolution is None:
            train_resolution = [1]
        if not is_list(train_resolution):
            train_resolution = [train_resolution]
        assert len(train_resolution) == 1
        return super()._train_dataloader(dataset, resolution=train_resolution[0], **kwargs)

    def _eval_dataloader(self, dataset, train_resolution=None, eval_resolutions=None, **kwargs):
        if dataset is None:
            return
        if eval_resolutions is None:
            eval_resolutions = [1]
        if not is_list(eval_resolutions):
            eval_resolutions = [eval_resolutions]
        dataloaders = []
        for resolution in eval_resolutions:
            dataloaders.append(super()._eval_dataloader(dataset, resolution=resolution, **kwargs))
        return (
            {None if res == 1 else str(res): dl for res, dl in zip(eval_resolutions, dataloaders)}
            if dataloaders is not None else None
        )


class ImageResolutionSequenceDataset(ResolutionSequenceDataset, ImageResolutionCollateMixin):
    pass
