"""Fixed stratified split and paired, independent epoch permutations."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from http.client import HTTPException
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms import ToTensor
from config import P, DATA_DIR, SPLIT_PATH


class FashionMNISTTest(FashionMNIST):
    """Keep torchvision parsing/ToTensor, but require only the test resources."""
    resources = tuple(item for item in FashionMNIST.resources if item[0].startswith("t10k-"))

    @property
    def raw_folder(self):
        return str(Path(self.root) / "FashionMNIST" / "raw")

    @property
    def processed_folder(self):
        return str(Path(self.root) / "FashionMNIST" / "processed")

    def __init__(self, root, transform):
        super().__init__(root, train=False, transform=transform, download=False)


@dataclass
class Datasets:
    train: Subset
    validation: Subset
    test: MNIST
    ood: FashionMNIST
    train_indices: np.ndarray
    validation_indices: np.ndarray


def make_split(targets):
    labels = np.asarray(targets)
    rng = np.random.Generator(np.random.PCG64(P.split_seed))
    train, validation = [], []
    for k in range(P.classes):
        indices = rng.permutation(np.flatnonzero(labels == k))
        if len(indices) < P.train_per_class + P.validation_per_class:
            raise ValueError(f"Not enough class-{k} examples")
        train.extend(indices[:P.train_per_class])
        validation.extend(indices[P.train_per_class:P.train_per_class + P.validation_per_class])
    return np.asarray(train, dtype=np.int64), np.asarray(validation, dtype=np.int64)


def download_fashion_test():
    """torchvision.download() fetches BOTH partitions; fetch ONLY test resources.

Use the dataset authors' repository (MD5 values supplied by torchvision).
FashionMNIST(download=False, train=False) then reads only these two files.
"""
    raw = DATA_DIR / "FashionMNIST" / "raw"
    base = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
    for filename, md5 in FashionMNIST.resources:
        if filename.startswith("t10k-"):
            download_and_extract_archive(base + filename, str(raw), md5=md5)


def download_mnist():
    """Same official resource checksums, with CVDF and torchvision mirrors."""
    raw = DATA_DIR / "MNIST" / "raw"
    mirrors = ("https://storage.googleapis.com/cvdf-datasets/mnist/", *MNIST.mirrors)
    for filename, md5 in MNIST.resources:
        failures = []
        for mirror in mirrors:
            try:
                download_and_extract_archive(mirror + filename, str(raw), md5=md5)
                break
            except (OSError, HTTPException, RuntimeError) as error:
                failures.append(f"{mirror}: {error}")
        else:
            raise RuntimeError(f"MNIST download failed for {filename}: {failures}")


def load_datasets(download=False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    transform = ToTensor()
    if download:
        download_mnist()
    pool = MNIST(str(DATA_DIR), train=True, transform=transform, download=False)
    test = MNIST(str(DATA_DIR), train=False, transform=transform, download=False)
    try:
        ood = FashionMNISTTest(str(DATA_DIR), transform=transform)
    except RuntimeError:
        if not download:
            raise
        download_fashion_test()
        ood = FashionMNISTTest(str(DATA_DIR), transform=transform)
    if (len(pool), len(test), len(ood)) != (60000, P.mnist_test_size, P.fashion_test_size):
        raise ValueError("Official dataset sizes do not match Protocol 1.0")
    train, validation = make_split(pool.targets)
    target_hash = sha256(pool.targets.numpy().tobytes()).hexdigest()
    if SPLIT_PATH.exists():
        with np.load(SPLIT_PATH, allow_pickle=False) as saved:
            if not (np.array_equal(saved["train"], train)
                    and np.array_equal(saved["validation"], validation)
                    and saved["target_sha256"].item() == target_hash):
                raise ValueError("Saved split differs from the frozen deterministic split")
    else:
        np.savez(SPLIT_PATH, train=train, validation=validation,
                 split_seed=P.split_seed, target_sha256=target_hash)
    return Datasets(Subset(pool, train.tolist()), Subset(pool, validation.tolist()),
                    test, ood, train, validation)


def epoch_order(seed, epoch, size=P.train_size):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + P.order_seed_offset + epoch)
    return torch.randperm(size, generator=generator)


def loader(dataset, seed=None, epoch=None):
    training = seed is not None
    sampler = epoch_order(seed, epoch, len(dataset)).tolist() if training else None
    # DataLoader's own bookkeeping must not consume the global model RNG.
    generator = torch.Generator(device="cpu").manual_seed(
        seed + P.order_seed_offset + epoch if training else P.split_seed)
    return DataLoader(dataset, batch_size=P.train_batch if training else P.eval_batch,
                      sampler=sampler, shuffle=False, num_workers=P.workers,
                      pin_memory=P.pin_memory, drop_last=P.drop_last, generator=generator)
