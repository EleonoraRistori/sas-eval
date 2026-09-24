from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class ImageSplit:
    """In-memory train/test partition of a collection of image paths."""

    train_paths: tuple[Path, ...]
    test_paths: tuple[Path, ...]


def list_image_paths(root: str | Path) -> list[Path]:
    """
    Recursively discover every supported image below ``root``.

    This deliberately includes images in all subdirectories, including folders
    named ``train`` or ``test``. Directory names have no special meaning.
    """
    root = Path(root)

    if not root.is_dir():
        raise FileNotFoundError(
            f"Image directory does not exist: {root}"
        )

    image_paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        supported_extensions = ", ".join(sorted(IMAGE_EXTENSIONS))

        raise ValueError(
            f"No supported images found under {root}. "
            f"Supported extensions are: {supported_extensions}"
        )

    return image_paths


def create_train_test_split(
    image_paths: Sequence[Path],
    *,
    train_fraction: float,
    seed: int | None,
) -> ImageSplit:
    """
    Randomly split image paths into train and test partitions.

    The split is deterministic when ``seed`` is supplied. At least one image is
    guaranteed in each split, so each input collection must contain at least
    two images.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(
            "train_fraction must be strictly between 0 and 1."
        )

    if len(image_paths) < 2:
        raise ValueError(
            "At least two images are required for an automatic train/test "
            "split."
        )

    shuffled_paths = list(image_paths)

    random_generator = random.Random(seed)
    random_generator.shuffle(shuffled_paths)

    n_train = int(len(shuffled_paths) * train_fraction)

    # Ensure that both train and test receive at least one image.
    n_train = max(1, min(n_train, len(shuffled_paths) - 1))

    return ImageSplit(
        train_paths=tuple(shuffled_paths[:n_train]),
        test_paths=tuple(shuffled_paths[n_train:]),
    )


def discover_and_split_images(
    root: str | Path,
    *,
    train_fraction: float,
    seed: int | None,
) -> ImageSplit:
    """
    Recursively read all images under ``root`` and split them in memory.

    Existing directory names such as ``train/`` and ``test/`` are ignored.
    """
    image_paths = list_image_paths(root)

    return create_train_test_split(
        image_paths,
        train_fraction=train_fraction,
        seed=seed,
    )


def sample_paths(
    image_paths: Sequence[Path],
    *,
    n_samples: int,
    seed: int | None,
) -> list[Path]:
    """
    Deterministically sample up to ``n_samples`` paths without replacement.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")

    sampled_paths = list(image_paths)

    random_generator = random.Random(seed)
    random_generator.shuffle(sampled_paths)

    return sampled_paths[:min(n_samples, len(sampled_paths))]


class LabeledImageDataset(Dataset):
    """
    Dataset for image paths with one fixed binary class label.

    Labels:
    - 0: fake/generated
    - 1: real
    """

    def __init__(
        self,
        image_paths: Sequence[Path],
        *,
        label: int,
        transform,
    ) -> None:
        if not image_paths:
            raise ValueError("Cannot create a dataset with zero images.")

        self.image_paths = list(image_paths)
        self.label = int(label)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return image, self.label


class ImagePathDataset(Dataset):
    """
    Dataset for IG extraction.

    Returns an image tensor, its expected label, and its absolute path.
    """

    def __init__(
        self,
        image_paths: Sequence[Path],
        *,
        label: int,
        transform,
    ) -> None:
        if not image_paths:
            raise ValueError("Cannot create a dataset with zero images.")

        self.image_paths = list(image_paths)
        self.label = int(label)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return image, self.label, str(image_path)