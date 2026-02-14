"""Long Range Arena (LRA) datasets: ListOps, IMDB/Text, AAN/Retrieval, Image, PathFinder, PathX."""

import io
import logging
import os
import pickle
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
import torchtext
import torchvision
from PIL import Image
from datasets import DatasetDict, Value, load_dataset
from einops.layers.torch import Rearrange, Reduce
from tonic.io import make_structured_array
from torchtext.vocab import build_vocab_from_iterator

from src.s5_compat import (
    SequenceDataset, ImageResolutionSequenceDataset,
    bitreversal_permutation, transpose_permutation,
    snake_permutation, hilbert_permutation,
    default_data_path,
)
from src.transform import OneHotLabels
from .collate import (
    Data, DataLoader,
    event_stream_collate_fn, lra_text_collate_fn,
    retrieval_collate_fn, lra_image_collate_fn,
    lra_pathfinder_collate_fn,
    event_stream_dataloader, event_stream_dataloader_parallel,
)


# ============================================================
# S5-derived dataset classes (ListOps, IMDB, AAN, CIFAR10, PathFinder)
# ============================================================

def listops_tokenizer(s):
    return s.translate({ord("]"): ord("X"), ord("("): None, ord(")"): None}).split()


class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, path, format, col_idx=None, skip_header=False, csv_reader_params=None):
        if csv_reader_params is None:
            csv_reader_params = {}
        format = format.lower()
        assert format in ["tsv", "csv"]
        with io.open(os.path.expanduser(path), encoding="utf8") as f:
            if format == "csv":
                reader = torchtext.utils.unicode_csv_reader(f, **csv_reader_params)
            elif format == "tsv":
                reader = torchtext.utils.unicode_csv_reader(f, delimiter="\t", **csv_reader_params)
            else:
                reader = f
            if skip_header:
                next(reader)
            self._data = [
                line if col_idx is None else [line[c] for c in col_idx]
                for line in reader
            ]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


class ListOps(SequenceDataset):
    _name_ = "listops"
    d_output = 10
    l_output = 0

    @property
    def init_defaults(self):
        return {"l_max": 2048, "append_bos": False, "append_eos": True, "n_workers": 4}

    @property
    def n_tokens(self):
        return len(self.vocab)

    @property
    def _cache_dir_name(self):
        return f"l_max-{self.l_max}-append_bos-{self.append_bos}-append_eos-{self.append_eos}"

    def init(self):
        if self.data_dir is None:
            self.data_dir = default_data_path / self._name_
        self.cache_dir = self.data_dir / self._cache_dir_name

    def prepare_data(self):
        if self.cache_dir is None:
            for split in ["train", "val", "test"]:
                split_path = self.data_dir / f"basic_{split}.tsv"
                if not split_path.is_file():
                    raise FileNotFoundError(
                        f"File {split_path} not found. Download lra_release.gz from "
                        "https://github.com/google-research/long-range-arena and "
                        "point data_dir to the listops-1000 directory."
                    )
        else:
            self.process_dataset()

    def setup(self, stage=None):
        dataset, self.tokenizer, self.vocab = self.process_dataset()
        self.vocab_size = len(self.vocab)
        dataset.set_format(type="torch", columns=["input_ids", "Target"])
        self.dataset_train, self.dataset_val, self.dataset_test = (
            dataset["train"], dataset["val"], dataset["test"],
        )

        def collate_batch(batch):
            xs, ys = zip(*[(data["input_ids"], data["Target"]) for data in batch])
            lengths = torch.tensor([len(x) for x in xs])
            xs = nn.utils.rnn.pad_sequence(xs, padding_value=self.vocab["<pad>"], batch_first=True)
            ys = torch.tensor(ys)
            return xs, ys, {"lengths": lengths}

        self._collate_fn = collate_batch

    def process_dataset(self):
        cache_dir = None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        if cache_dir is not None and cache_dir.is_dir():
            return self._load_from_cache(cache_dir)

        dataset = load_dataset(
            "csv",
            data_files={
                "train": str(self.data_dir / "basic_train.tsv"),
                "val": str(self.data_dir / "basic_val.tsv"),
                "test": str(self.data_dir / "basic_test.tsv"),
            },
            delimiter="\t", keep_in_memory=True,
        )
        tokenizer = listops_tokenizer
        l_max = self.l_max - int(self.append_bos) - int(self.append_eos)
        tokenize = lambda example: {"tokens": tokenizer(example["Source"])[:l_max]}
        dataset = dataset.map(tokenize, remove_columns=["Source"], keep_in_memory=True, load_from_cache_file=False, num_proc=max(self.n_workers, 1))
        vocab = build_vocab_from_iterator(
            dataset["train"]["tokens"],
            specials=["<pad>", "<unk>"] + (["<bos>"] if self.append_bos else []) + (["<eos>"] if self.append_eos else []),
        )
        vocab.set_default_index(vocab["<unk>"])
        numericalize = lambda example: {
            "input_ids": vocab((["<bos>"] if self.append_bos else []) + example["tokens"] + (["<eos>"] if self.append_eos else []))
        }
        dataset = dataset.map(numericalize, remove_columns=["tokens"], keep_in_memory=True, load_from_cache_file=False, num_proc=max(self.n_workers, 1))
        if cache_dir is not None:
            self._save_to_cache(dataset, tokenizer, vocab, cache_dir)
        return dataset, tokenizer, vocab

    def _save_to_cache(self, dataset, tokenizer, vocab, cache_dir):
        cache_dir = self.cache_dir / self._cache_dir_name
        logger = logging.getLogger(__name__)
        logger.info(f"Saving to cache at {cache_dir}")
        dataset.save_to_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        with open(cache_dir / "vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)

    def _load_from_cache(self, cache_dir):
        assert cache_dir.is_dir()
        logger = logging.getLogger(__name__)
        logger.info(f"Load from cache at {cache_dir}")
        dataset = DatasetDict.load_from_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open(cache_dir / "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        return dataset, tokenizer, vocab


class IMDB(SequenceDataset):
    _name_ = "imdb"
    d_output = 2
    l_output = 0

    @property
    def init_defaults(self):
        return {
            "l_max": 4096, "level": "char", "min_freq": 15, "seed": 42,
            "val_split": 0.0, "append_bos": False, "append_eos": True, "n_workers": 4,
        }

    @property
    def n_tokens(self):
        return len(self.vocab)

    def prepare_data(self):
        if self.cache_dir is None:
            load_dataset(self._name_, cache_dir=self.data_dir)
        else:
            self.process_dataset()

    def setup(self, stage=None):
        assert self.level in ["word", "char"]
        dataset, self.tokenizer, self.vocab = self.process_dataset()
        dataset.set_format(type="torch", columns=["input_ids", "label"])
        dataset_train, self.dataset_test = dataset["train"], dataset["test"]
        if self.val_split == 0.0:
            self.dataset_train, self.dataset_val = dataset_train, None
        else:
            train_val = dataset_train.train_test_split(test_size=self.val_split, seed=self.seed)
            self.dataset_train, self.dataset_val = train_val["train"], train_val["test"]

    def _collate_fn(self, batch):
        xs, ys = zip(*[(data["input_ids"], data["label"]) for data in batch])
        lengths = torch.tensor([len(x) for x in xs])
        xs = nn.utils.rnn.pad_sequence(xs, padding_value=self.vocab["<pad>"], batch_first=True)
        ys = torch.tensor(ys)
        return xs, ys, {"lengths": lengths}

    def process_dataset(self):
        cache_dir = None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        if cache_dir is not None and cache_dir.is_dir():
            return self._load_from_cache(cache_dir)

        dataset = load_dataset(self._name_, cache_dir=self.data_dir)
        dataset = DatasetDict(train=dataset["train"], test=dataset["test"])
        if self.level == "word":
            tokenizer = torchtext.data.utils.get_tokenizer("spacy", language="en_core_web_sm")
        else:
            tokenizer = list
        l_max = self.l_max - int(self.append_bos) - int(self.append_eos)
        tokenize = lambda example: {"tokens": tokenizer(example["text"])[:l_max]}
        dataset = dataset.map(tokenize, remove_columns=["text"], keep_in_memory=True, load_from_cache_file=False, num_proc=max(self.n_workers, 1))
        vocab = build_vocab_from_iterator(
            dataset["train"]["tokens"],
            min_freq=self.min_freq,
            specials=["<pad>", "<unk>"] + (["<bos>"] if self.append_bos else []) + (["<eos>"] if self.append_eos else []),
        )
        vocab.set_default_index(vocab["<unk>"])
        numericalize = lambda example: {
            "input_ids": vocab((["<bos>"] if self.append_bos else []) + example["tokens"] + (["<eos>"] if self.append_eos else []))
        }
        dataset = dataset.map(numericalize, remove_columns=["tokens"], keep_in_memory=True, load_from_cache_file=False, num_proc=max(self.n_workers, 1))
        if cache_dir is not None:
            self._save_to_cache(dataset, tokenizer, vocab, cache_dir)
        return dataset, tokenizer, vocab

    def _save_to_cache(self, dataset, tokenizer, vocab, cache_dir):
        cache_dir = self.cache_dir / self._cache_dir_name
        logger = logging.getLogger(__name__)
        logger.info(f"Saving to cache at {cache_dir}")
        dataset.save_to_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        with open(cache_dir / "vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)

    def _load_from_cache(self, cache_dir):
        assert cache_dir.is_dir()
        logger = logging.getLogger(__name__)
        logger.info(f"Load from cache at {cache_dir}")
        dataset = DatasetDict.load_from_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open(cache_dir / "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        return dataset, tokenizer, vocab

    @property
    def _cache_dir_name(self):
        return f"l_max-{self.l_max}-level-{self.level}-min_freq-{self.min_freq}-append_bos-{self.append_bos}-append_eos-{self.append_eos}"


class AAN(SequenceDataset):
    _name_ = "aan"
    d_output = 2
    l_output = 0

    @property
    def n_tokens(self):
        return len(self.vocab)

    @property
    def init_defaults(self):
        return {"l_max": 4000, "append_bos": False, "append_eos": True, "n_workers": 40}

    @property
    def _cache_dir_name(self):
        return f"l_max-{self.l_max}-append_bos-{self.append_bos}-append_eos-{self.append_eos}"

    def init(self):
        if self.data_dir is None:
            self.data_dir = default_data_path / self._name_
        self.cache_dir = self.data_dir / self._cache_dir_name

    def prepare_data(self):
        if self.cache_dir is None:
            for split in ["train", "eval", "test"]:
                split_path = self.data_dir / f"new_aan_pairs.{split}.tsv"
                if not split_path.is_file():
                    raise FileNotFoundError(
                        f"File {split_path} not found. Download lra_release.gz from "
                        "https://github.com/google-research/long-range-arena."
                    )
        else:
            self.process_dataset()

    def setup(self, stage=None):
        torch.multiprocessing.set_sharing_strategy("file_system")
        dataset, self.tokenizer, self.vocab = self.process_dataset()
        dataset.set_format(type="torch", columns=["input_ids1", "input_ids2", "label"])
        self.dataset_train, self.dataset_val, self.dataset_test = (
            dataset["train"], dataset["val"], dataset["test"],
        )

        def collate_batch(batch):
            xs1, xs2, ys = zip(*[(data["input_ids1"], data["input_ids2"], data["label"]) for data in batch])
            lengths1 = torch.tensor([len(x) for x in xs1])
            lengths2 = torch.tensor([len(x) for x in xs2])
            xs1 = nn.utils.rnn.pad_sequence(xs1, padding_value=self.vocab["<pad>"], batch_first=True)
            xs2 = nn.utils.rnn.pad_sequence(xs2, padding_value=self.vocab["<pad>"], batch_first=True)
            L = max(xs1.size(1), xs2.size(1))
            xs1 = F.pad(xs1, (0, L - xs1.size(1)), value=self.vocab["<pad>"])
            xs2 = F.pad(xs2, (0, L - xs2.size(1)), value=self.vocab["<pad>"])
            ys = torch.tensor(ys)
            xs = torch.cat([xs1, xs2], dim=0)
            lengths = torch.cat([lengths1, lengths2], dim=0)
            return xs, ys, {"lengths": lengths}

        self._collate_fn = collate_batch

    def process_dataset(self):
        cache_dir = None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        if cache_dir is not None and cache_dir.is_dir():
            return self._load_from_cache(cache_dir)

        dataset = load_dataset(
            "csv",
            data_files={
                "train": str(self.data_dir / "new_aan_pairs.train.tsv"),
                "val": str(self.data_dir / "new_aan_pairs.eval.tsv"),
                "test": str(self.data_dir / "new_aan_pairs.test.tsv"),
            },
            delimiter="\t",
            column_names=["label", "input1_id", "input2_id", "text1", "text2"],
            keep_in_memory=True,
        )
        dataset = dataset.remove_columns(["input1_id", "input2_id"])
        new_features = dataset["train"].features.copy()
        new_features["label"] = Value("int32")
        dataset = dataset.cast(new_features)
        tokenizer = list
        l_max = self.l_max - int(self.append_bos) - int(self.append_eos)
        tokenize = lambda example: {
            "tokens1": tokenizer(example["text1"])[:l_max],
            "tokens2": tokenizer(example["text2"])[:l_max],
        }
        dataset = dataset.map(tokenize, remove_columns=["text1", "text2"], keep_in_memory=True, load_from_cache_file=False, num_proc=max(self.n_workers, 1))
        vocab = build_vocab_from_iterator(
            dataset["train"]["tokens1"] + dataset["train"]["tokens2"],
            specials=["<pad>", "<unk>"] + (["<bos>"] if self.append_bos else []) + (["<eos>"] if self.append_eos else []),
        )
        vocab.set_default_index(vocab["<unk>"])
        encode = lambda text: vocab((["<bos>"] if self.append_bos else []) + text + (["<eos>"] if self.append_eos else []))
        numericalize = lambda example: {
            "input_ids1": encode(example["tokens1"]),
            "input_ids2": encode(example["tokens2"]),
        }
        dataset = dataset.map(numericalize, remove_columns=["tokens1", "tokens2"], keep_in_memory=True, load_from_cache_file=False, num_proc=max(self.n_workers, 1))
        if cache_dir is not None:
            self._save_to_cache(dataset, tokenizer, vocab, cache_dir)
        return dataset, tokenizer, vocab

    def _save_to_cache(self, dataset, tokenizer, vocab, cache_dir):
        cache_dir = self.cache_dir / self._cache_dir_name
        logger = logging.getLogger(__name__)
        logger.info(f"Saving to cache at {cache_dir}")
        dataset.save_to_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        with open(cache_dir / "vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)

    def _load_from_cache(self, cache_dir):
        assert cache_dir.is_dir()
        logger = logging.getLogger(__name__)
        logger.info(f"Load from cache at {cache_dir}")
        dataset = DatasetDict.load_from_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open(cache_dir / "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        return dataset, tokenizer, vocab


class CIFAR10(ImageResolutionSequenceDataset):
    _name_ = "cifar"
    d_output = 10
    l_output = 0

    @property
    def init_defaults(self):
        return {
            "permute": None, "grayscale": True, "tokenize": False,
            "augment": False, "cutout": False, "rescale": None,
            "random_erasing": False, "val_split": 0.1, "seed": 42,
        }

    @property
    def d_input(self):
        if self.grayscale:
            return 256 if self.tokenize else 1
        else:
            assert not self.tokenize
            return 3

    def setup(self):
        img_size = 32
        if self.rescale:
            img_size //= self.rescale

        if self.grayscale:
            preprocessors = [torchvision.transforms.Grayscale(), torchvision.transforms.ToTensor()]
            permutations_list = [
                torchvision.transforms.Lambda(lambda x: x.view(1, img_size * img_size).t())
            ]
            if self.tokenize:
                preprocessors.append(torchvision.transforms.Lambda(lambda x: (x * 255).long()))
                permutations_list.append(Rearrange("l 1 -> l"))
            else:
                preprocessors.append(torchvision.transforms.Normalize(mean=122.6 / 255.0, std=61.0 / 255.0))
        else:
            preprocessors = [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
            permutations_list = [
                torchvision.transforms.Lambda(
                    Rearrange("z h w -> (h w) z", z=3, h=img_size, w=img_size)
                )
            ]

        if self.permute == "br":
            permutation = bitreversal_permutation(img_size * img_size)
            permutations_list.append(torchvision.transforms.Lambda(lambda x: x[permutation]))
        elif self.permute == "snake":
            permutation = snake_permutation(img_size, img_size)
            permutations_list.append(torchvision.transforms.Lambda(lambda x: x[permutation]))
        elif self.permute == "hilbert":
            permutation = hilbert_permutation(img_size)
            permutations_list.append(torchvision.transforms.Lambda(lambda x: x[permutation]))
        elif self.permute == "transpose":
            permutation = transpose_permutation(img_size, img_size)
            transform = torchvision.transforms.Lambda(lambda x: torch.cat([x, x[permutation]], dim=-1))
            permutations_list.append(transform)
        elif self.permute == "2d":
            permutation = torchvision.transforms.Lambda(Rearrange("(h w) c -> h w c", h=img_size, w=img_size))
            permutations_list.append(permutation)
        elif self.permute == "2d_transpose":
            permutation = torchvision.transforms.Lambda(Rearrange("(h w) c -> c h w", h=img_size, w=img_size))
            permutations_list.append(permutation)

        if self.augment:
            augmentations = [
                torchvision.transforms.RandomCrop(img_size, padding=4, padding_mode="symmetric"),
                torchvision.transforms.RandomHorizontalFlip(),
            ]
            post_augmentations = []
        else:
            augmentations, post_augmentations = [], []

        transforms_train = augmentations + preprocessors + post_augmentations + permutations_list
        transforms_eval = preprocessors + permutations_list

        transform_train = torchvision.transforms.Compose(transforms_train)
        transform_eval = torchvision.transforms.Compose(transforms_eval)
        self.dataset_train = torchvision.datasets.CIFAR10(
            f"{default_data_path}/{self._name_}", train=True, download=True, transform=transform_train,
        )
        self.dataset_test = torchvision.datasets.CIFAR10(
            f"{default_data_path}/{self._name_}", train=False, transform=transform_eval,
        )

        if self.rescale:
            self.dataset_train.data = self.dataset_train.data.reshape(
                (self.dataset_train.data.shape[0], 32 // self.rescale, self.rescale, 32 // self.rescale, self.rescale, 3)
            ).max(4).max(2).astype(np.uint8)
            self.dataset_test.data = self.dataset_test.data.reshape(
                (self.dataset_test.data.shape[0], 32 // self.rescale, self.rescale, 32 // self.rescale, self.rescale, 3)
            ).max(4).max(2).astype(np.uint8)

        self.split_train_val(self.val_split)

    def __str__(self):
        return f"{'p' if self.permute else 's'}{self._name_}"


class PathFinderDataset(torch.utils.data.Dataset):
    """Path Finder dataset."""
    blacklist = {"pathfinder32/curv_baseline/imgs/0/sample_172.png"}

    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir).expanduser()
        assert self.data_dir.is_dir(), f"data_dir {self.data_dir} does not exist"
        self.transform = transform
        samples = []
        for diff_level in ["curv_contour_length_14"]:
            path_list = sorted(
                list((self.data_dir / diff_level / "metadata").glob("*.npy")),
                key=lambda path: int(path.stem),
            )
            assert path_list, "No metadata found"
            for metadata_file in path_list:
                with open(metadata_file, "r") as f:
                    for metadata in f.read().splitlines():
                        metadata = metadata.split()
                        image_path = Path(diff_level) / metadata[0] / metadata[1]
                        if str(Path(self.data_dir.stem) / image_path) not in self.blacklist:
                            label = int(metadata[3])
                            samples.append((image_path, label))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        with open(self.data_dir / path, "rb") as f:
            sample = Image.open(f).convert("L")
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


class PathFinder(ImageResolutionSequenceDataset):
    _name_ = "pathfinder"
    d_input = 1
    d_output = 2
    l_output = 0

    @property
    def n_tokens(self):
        if self.tokenize:
            return 256

    @property
    def init_defaults(self):
        return {
            "resolution": 32, "sequential": True, "tokenize": False,
            "pool": 1, "val_split": 0.1, "test_split": 0.1, "seed": 42,
        }

    def default_transforms(self):
        transform_list = [torchvision.transforms.ToTensor()]
        if self.pool > 1:
            transform_list.append(Reduce("1 (h h2) (w w2) -> 1 h w", "mean", h2=self.pool, w2=self.pool))
        if self.tokenize:
            transform_list.append(torchvision.transforms.Lambda(lambda x: (x * 255).long()))
        else:
            transform_list.append(torchvision.transforms.Normalize(mean=0.5, std=0.5))
        if self.sequential:
            transform_list.append(
                Rearrange("1 h w -> (h w)") if self.tokenize else Rearrange("1 h w -> (h w) 1")
            )
        else:
            transform_list.append(Rearrange("1 h w -> h w 1"))
        return torchvision.transforms.Compose(transform_list)

    def prepare_data(self):
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"Directory {self.data_dir} not found. Download lra_release.gz from "
                "https://github.com/google-research/long-range-arena."
            )

    def setup(self, stage=None):
        if self.data_dir is None:
            self.data_dir = default_data_path / self._name_ / f"pathfinder{self.resolution}"

        if self.cache_dir is not None:
            cache_file = Path(self.cache_dir / (self._cache_dir_name + '.pt'))
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    dset = torch.load(f)
                self.dataset_train = dset['train']
                self.dataset_val = dset['val']
                self.dataset_test = dset['test']
                return

        torch.multiprocessing.set_sharing_strategy("file_system")
        dataset = PathFinderDataset(self.data_dir, transform=self.default_transforms())
        len_dataset = len(dataset)
        val_len = int(self.val_split * len_dataset)
        test_len = int(self.test_split * len_dataset)
        train_len = len_dataset - val_len - test_len
        self.dataset_train, self.dataset_val, self.dataset_test = torch.utils.data.random_split(
            dataset, [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self.seed),
        )

        def _compile_convert(dset, tag):
            loader = torch.utils.data.DataLoader(dataset=dset, batch_size=len(dset), shuffle=False, drop_last=False)
            inp, out = next(iter(loader))
            return torch.utils.data.TensorDataset(inp, out)

        self.dataset_train = _compile_convert(self.dataset_train, tag='train')
        self.dataset_val = _compile_convert(self.dataset_val, tag='val')
        self.dataset_test = _compile_convert(self.dataset_test, tag='test')

        if self.cache_dir is not None:
            cache_path = Path(self.cache_dir) / (self._cache_dir_name + '.pt')
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            logger = logging.getLogger(__name__)
            logger.info(f"Saving to cache at {cache_path}")
            with open(cache_path, 'wb') as f:
                torch.save({
                    'train': self.dataset_train,
                    'val': self.dataset_val,
                    'test': self.dataset_test,
                }, f)

    @property
    def _cache_dir_name(self):
        return f"pathfinder-resolution-{self.resolution}"


# ============================================================
# Tonic wrapper classes
# ============================================================

class ListOpsTonic(Dataset):
    """ListOps dataset wrapped for the Tonic event-stream framework."""
    sensor_size = (18, 1, 1)

    def __init__(self, data_dir, cache_dir=None, split="None", transform=None, target_transform=None):
        assert split in ['train', 'val', 'test']
        self.sensor_size = (18, 1, 1)
        self.dtype = np.dtype([("t", int), ("x", int), ("p", int)])
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.listops = ListOps(_name_="listops", data_dir=data_dir, cache_dir=cache_dir, train=(split == "train"))
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


class IMDBTonic(Dataset):
    """IMDB dataset wrapped for the Tonic event-stream framework."""

    def __init__(self, data_dir, cache_dir, split='train', transform=None, target_transform=None):
        assert split in ['train', 'test']
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
        times = np.arange(0, len(input_ids))
        events = make_structured_array(times, input_ids, 1, dtype=self.dtype)
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return events, label

    def __len__(self):
        return len(self.data)


class AANTonic(Dataset):
    """AAN retrieval dataset wrapped for the Tonic event-stream framework."""

    def __init__(self, data_dir, cache_dir=None, split='train', transform=None, target_transform=None):
        assert split in ['train', 'val', 'test']
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.aan = AAN(_name_="aan", data_dir=data_dir, cache_dir=cache_dir, train=(split == 'train'))
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
        times1 = np.arange(0, len(input_ids1))
        times2 = np.arange(0, len(input_ids2))
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


class ImageTonic(Dataset):
    """CIFAR10 image dataset wrapped for the Tonic event-stream framework."""
    sensor_size = (256, 1, 1)

    def __init__(self, data_dir, cache_dir=None, split='train', transform=None, target_transform=None):
        assert split in ['train', 'val', 'test']
        self.sensor_size = (256, 1, 1)
        self.dtype = np.dtype([("t", int), ("x", int), ("p", int)])
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        kwargs = {"grayscale": True, "tokenize": True}
        self.image = CIFAR10(_name_="cifar", data_dir=data_dir, cache_dir=cache_dir, **kwargs)
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
        input_ids, target = data
        times = np.arange(0, len(input_ids))
        input_ids = np.squeeze(input_ids)
        events = make_structured_array(times, input_ids, 1, dtype=self.dtype)
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        return len(self.data)


class PathFinderTonic(Dataset):
    """PathFinder dataset wrapped for the Tonic event-stream framework."""
    base_url = "https://github.com/google-research/long-range-arena"
    sensor_size = (32, 32, 1)

    def __init__(self, data_dir, tokenize, cache_dir=None, resolution=32, split="None", transform=None, target_transform=None):
        assert split in ['train', 'val', 'test']
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
        self.pathfinder = PathFinder(
            _name_="pathfinder", data_dir=data_dir, cache_dir=self.cache_dir,
            resolution=resolution, tokenize=tokenize,
        )
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
# Factory functions (create_* datasets)
# ============================================================

def create_listops_tonic_classification_dataset(
    cache_dir=None, data_dir=None,
    per_device_batch_size=32, per_device_eval_batch_size=64,
    world_size=1, num_workers=0, seed=42,
    pad_unit=2048, cut_mix=0.0, no_time_information=False, **kwargs,
):
    """Create dataloaders for the ListOps LRA task."""
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

    train_data, val_data, test_data = TrainData(), ValData(), TestData()
    assert len(train_data) > 0 and len(val_data) > 0 and len(test_data) > 0

    collate_fn = partial(event_stream_collate_fn, resolution=(18,), pad_unit=pad_unit, no_time_information=no_time_information)
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=partial(collate_fn, cut_mix=cut_mix), eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True,
    )
    data = Data(n_classes=10, num_embeddings=18, train_size=len(train_data))
    return train_loader, val_loader, test_loader, data


def create_imdb_tonic_classification_dataset(
    cache_dir=None, data_dir=None,
    per_device_batch_size=32, per_device_eval_batch_size=32,
    world_size=1, num_workers=16, seed=42,
    pad_unit=4096, cut_mix=0.0, no_time_information=False, **kwargs,
):
    """Create dataloaders for the IMDB text classification LRA task."""
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

    train_data, test_data = TrainData(), TestData()
    val_data = test_data

    collate_fn = partial(lra_text_collate_fn, resolution=(135,), pad_unit=pad_unit, no_time_information=no_time_information)
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=partial(collate_fn, cut_mix=cut_mix), eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True,
    )
    data = Data(n_classes=2, num_embeddings=135, train_size=len(train_data))
    return train_loader, val_loader, test_loader, data


def create_retrieval_tonic_classification_dataset(
    cache_dir=None, data_dir=None,
    per_device_batch_size=256, per_device_eval_batch_size=256,
    world_size=1, num_workers=40, seed=42,
    pad_unit=4000, cut_mix=0.0, no_time_information=True, **kwargs,
):
    """Create dataloaders for the AAN retrieval LRA task."""
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

    train_data, val_data, test_data = TrainData(), ValData(), TestData()

    collate_fn = partial(retrieval_collate_fn, resolution=(98,), pad_unit=pad_unit, no_time_information=no_time_information)
    if world_size > 1:
        train_loader, val_loader, test_loader = event_stream_dataloader_parallel(
            train_data, val_data, test_data,
            train_collate_fn=collate_fn, eval_collate_fn=collate_fn,
            batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
            rng=rng, num_workers=num_workers, shuffle_training=True,
        )
    else:
        train_loader, val_loader, test_loader = event_stream_dataloader(
            train_data, val_data, test_data,
            train_collate_fn=collate_fn, eval_collate_fn=collate_fn,
            batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
            rng=rng, num_workers=num_workers, shuffle_training=True,
        )
    data = Data(n_classes=2, num_embeddings=98, train_size=len(train_data))
    return train_loader, val_loader, test_loader, data


def create_image_tonic_classification_dataset(
    cache_dir=None, data_dir=None,
    per_device_batch_size=32, per_device_eval_batch_size=64,
    world_size=1, num_workers=40, seed=42,
    pad_unit=1024, cut_mix=0.0, no_time_information=False, **kwargs,
):
    """Create dataloaders for the CIFAR10 sequential image classification LRA task."""
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

    train_data, val_data, test_data = TrainData(), ValData(), TestData()

    collate_fn = partial(lra_image_collate_fn, resolution=(256,), pad_unit=pad_unit, no_time_information=no_time_information)
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=partial(collate_fn, cut_mix=cut_mix), eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True,
    )
    data = Data(n_classes=10, num_embeddings=256, train_size=len(train_data))
    return train_loader, val_loader, test_loader, data


def create_pathfinder_tonic_classification_dataset(
    cache_dir=None, data_dir=None, resolution=32,
    per_device_batch_size=32, per_device_eval_batch_size=64,
    world_size=1, num_workers=96, seed=42,
    pad_unit=1024, no_time_information=True, **kwargs,
):
    """Create dataloaders for the PathFinder LRA task."""
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

    train_data, val_data, test_data = TrainData(), ValData(), TestData()
    assert len(train_data) > 0 and len(val_data) > 0 and len(test_data) > 0

    collate_fn = partial(lra_pathfinder_collate_fn, resolution=(256,), pad_unit=pad_unit, no_time_information=no_time_information)
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=partial(collate_fn), eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True,
    )
    data = Data(n_classes=2, num_embeddings=256, train_size=len(train_data))
    return train_loader, val_loader, test_loader, data


def create_pathx_tonic_classification_dataset(
    cache_dir=None, data_dir=None, resolution=128,
    per_device_batch_size=32, per_device_eval_batch_size=64,
    world_size=1, num_workers=96, seed=42,
    pad_unit=1024, no_time_information=True, **kwargs,
):
    """Create dataloaders for the PathX LRA task (128x128 resolution)."""
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

    train_data, val_data, test_data = TrainData(), ValData(), TestData()
    assert len(train_data) > 0 and len(val_data) > 0 and len(test_data) > 0

    collate_fn = partial(lra_pathfinder_collate_fn, resolution=(256,), pad_unit=pad_unit, no_time_information=no_time_information)
    train_loader, val_loader, test_loader = event_stream_dataloader(
        train_data, val_data, test_data,
        train_collate_fn=partial(collate_fn), eval_collate_fn=collate_fn,
        batch_size=per_device_batch_size * world_size, eval_batch_size=per_device_eval_batch_size * world_size,
        rng=rng, num_workers=num_workers, shuffle_training=True,
    )
    data = Data(n_classes=2, num_embeddings=256, train_size=len(train_data))
    return train_loader, val_loader, test_loader, data
