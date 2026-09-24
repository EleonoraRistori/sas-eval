from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ._data import LabeledImageDataset
from ._saliency import get_evaluation_transform


def _create_loader(
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    return DataLoader(**loader_kwargs)


def compute_class_accuracy(
    *,
    model: torch.nn.Module,
    image_paths: Sequence[Path],
    expected_label: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    description: str,
) -> float:
    """Calculate accuracy for a single expected class."""
    model.eval()

    dataset = LabeledImageDataset(
        image_paths,
        label=expected_label,
        transform=get_evaluation_transform(),
    )

    loader = _create_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    correct = 0
    total = 0

    predicted_fake = 0
    predicted_real = 0

    with torch.inference_mode():
        for inputs, labels in tqdm(
            loader,
            desc=f"Accuracy: {description}",
            unit="batch",
        ):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            predictions = model(inputs).argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.numel()

            predicted_fake += (predictions == 0).sum().item()
            predicted_real += (predictions == 1).sum().item()

    accuracy = correct / total

    print(
        f"{description.capitalize()} accuracy: {accuracy:.4f} | "
        f"predicted fake={predicted_fake}, "
        f"predicted real={predicted_real}"
    )

    return accuracy


def compute_mean_median_saliency(
    real_maps: dict[str, np.ndarray],
    fake_maps: dict[str, np.ndarray],
) -> float:
    """
    Calculate the mean of per-image median attribution values.

    This is equivalent to the original `median()` plus `aggregate()` logic.
    """
    attribution_medians = [
        float(np.median(attribution_map))
        for attribution_map in real_maps.values()
    ]

    attribution_medians.extend(
        float(np.median(attribution_map))
        for attribution_map in fake_maps.values()
    )

    # Preserve the original behavior for an empty collection of maps.
    if not attribution_medians:
        return 0.0

    return float(np.mean(attribution_medians))


def compute_sas_metric(
    model: torch.nn.Module,
    *,
    real_test_paths: Sequence[Path],
    fake_test_paths: Sequence[Path],
    real_maps: dict[str, np.ndarray],
    fake_maps: dict[str, np.ndarray],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> float:
    """
    Compute the SAS metric.

    SAS = A + A**100 * (1 - M / 0.1)

    where:
    - A is the balanced real/fake classification accuracy;
    - M is the mean median attribution value.
    """
    model.eval()

    real_accuracy = compute_class_accuracy(
        model=model,
        image_paths=real_test_paths,
        expected_label=1,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        description="real",
    )

    fake_accuracy = compute_class_accuracy(
        model=model,
        image_paths=fake_test_paths,
        expected_label=0,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        description="fake",
    )

    balanced_accuracy = (real_accuracy + fake_accuracy) / 2.0

    mean_median_saliency = compute_mean_median_saliency(
        real_maps,
        fake_maps,
    )

    sas = (
        balanced_accuracy
        + balanced_accuracy ** 100
        * (1.0 - mean_median_saliency / 0.1)
    )

    print(f"Balanced accuracy: {balanced_accuracy:.4f}")
    print(f"Mean median saliency: {mean_median_saliency:.6f}")
    print(f"SAS: {sas:.6f}")

    return float(sas)