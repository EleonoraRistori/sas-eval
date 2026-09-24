from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from captum.attr import IntegratedGradients, NoiseTunnel
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from ._data import ImagePathDataset


def get_evaluation_transform(image_size: int = 224):
    """Evaluation transform matching the original SAS scripts."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])


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


def _extract_maps_for_class(
    *,
    model: torch.nn.Module,
    image_paths: Sequence[Path],
    expected_label: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    description: str,
) -> dict[str, np.ndarray]:
    """
    Extract SmoothGrad Integrated Gradients maps for correctly classified images.
    """
    model.eval()

    dataset = ImagePathDataset(
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

    integrated_gradients = IntegratedGradients(model)
    noise_tunnel = NoiseTunnel(integrated_gradients)

    attribution_maps_by_path: dict[str, np.ndarray] = {}

    progress = tqdm(
        loader,
        desc=f"Integrated Gradients: {description}",
        unit="batch",
    )

    for inputs, labels, paths in progress:
        inputs_device = inputs.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)

        with torch.inference_mode():
            predictions = model(inputs_device).argmax(dim=1)

        correct_mask = predictions.eq(labels_device)
        n_correct = int(correct_mask.sum().item())

        progress.set_postfix(correct=n_correct)

        if n_correct == 0:
            continue

        correct_inputs = inputs[
            correct_mask.detach().cpu()
        ].detach().clone().to(device)

        correct_inputs.requires_grad_(True)

        correct_targets = predictions[correct_mask]

        attributions = noise_tunnel.attribute(
            correct_inputs,
            baselines=torch.zeros_like(correct_inputs),
            target=correct_targets,
            nt_type="smoothgrad",
            stdevs=0.1,
            nt_samples=3,
            nt_samples_batch_size=4,
            n_steps=50,
        )

        # Shape: [batch, 3, height, width] -> [batch, height, width].
        maps = attributions.abs().sum(dim=1).detach().cpu().numpy()

        correctly_classified_paths = [
            path
            for path, is_correct in zip(
                paths,
                correct_mask.detach().cpu().tolist(),
            )
            if is_correct
        ]

        for path, attribution_map in zip(
            correctly_classified_paths,
            maps,
        ):
            attribution_maps_by_path[path] = attribution_map

    return attribution_maps_by_path


def extract_saliency_maps(
    model: torch.nn.Module,
    *,
    real_test_paths: Sequence[Path],
    fake_test_paths: Sequence[Path],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Extract IG maps for correctly classified real and fake test images.

    Label convention:
    - real: 1
    - fake: 0
    """
    if batch_size < 1:
        raise ValueError("Integrated Gradients batch size must be at least 1.")

    model.eval()

    real_maps = _extract_maps_for_class(
        model=model,
        image_paths=real_test_paths,
        expected_label=1,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        description="real",
    )

    fake_maps = _extract_maps_for_class(
        model=model,
        image_paths=fake_test_paths,
        expected_label=0,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        description="fake",
    )

    model.eval()

    return real_maps, fake_maps