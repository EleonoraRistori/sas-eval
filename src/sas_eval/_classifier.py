from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from ._data import LabeledImageDataset


def get_training_transform(image_size: int = 224):
    """Training transform matching the original training script."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
        ),
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
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    return DataLoader(**loader_kwargs)


def train_gsc_classifier(
    *,
    real_train_paths: Sequence[Path],
    fake_train_paths: Sequence[Path],
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    num_workers: int,
    device: torch.device,
) -> torch.nn.Module:
    """
    Train the binary ResNet-18 GSC classifier.

    The classifier preserves the original label convention:

    - fake/generated: 0
    - real: 1
    """
    train_transform = get_training_transform()

    real_dataset = LabeledImageDataset(
        real_train_paths,
        label=1,
        transform=train_transform,
    )

    fake_dataset = LabeledImageDataset(
        fake_train_paths,
        label=0,
        transform=train_transform,
    )

    train_dataset = ConcatDataset([
        real_dataset,
        fake_dataset,
    ])

    train_loader = _create_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        device=device,
    )

    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT,
    )

    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(
            train_loader,
            desc=f"Training epoch {epoch + 1}/{num_epochs}",
            unit="batch",
        )

        for inputs, labels in progress:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_dataset)
        train_accuracy = correct / total

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}, "
            f"Acc: {train_accuracy:.4f}"
        )

    # All downstream stages use the trained model in evaluation mode.
    model.eval()

    return model