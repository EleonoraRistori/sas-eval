from __future__ import annotations

import json
import pickle
from pathlib import Path

import torch

from ._classifier import train_gsc_classifier
from ._data import (
    ImageSplit,
    discover_and_split_images,
    sample_paths,
)
from ._metric import compute_sas_metric
from ._saliency import extract_saliency_maps
from ._utils import (
    default_num_workers,
    derive_ig_batch_size,
    resolve_device,
    set_seed,
)


def compute_sas(
    real_dir: str | Path,
    fake_dir: str | Path,
    *,
    batch_size: int = 32,
    device: str = "auto",
    seed: int | None = 42,
    train_fraction: float = 0.8,
    output_dir: str | Path | None = None,
    num_epochs: int = 30,
    learning_rate: float = 1e-4,
    n_real: int = 5000,
    n_fake: int = 5000,
    num_workers: int | None = None,
) -> float:
    """
    Train a GSC classifier and compute the SAS score.

    All images under ``real_dir`` and ``fake_dir`` are discovered recursively.
    The package always creates an in-memory train/test split, independently for
    real and fake images. Existing folder names, including ``train`` and
    ``test``, are ignored.

    Parameters
    ----------
    real_dir:
        Root directory containing all real images and optional subdirectories.

    fake_dir:
        Root directory containing all generated images for one model and
        optional subdirectories.

    batch_size:
        Base batch size used for training and accuracy evaluation. A smaller
        batch size is derived internally for Integrated Gradients.

    seed:
        Controls image splitting, sampling, model initialization, shuffling,
        and random data augmentation.

    train_fraction:
        Fraction of recursively discovered images allocated to training.

    output_dir:
        Optional directory in which checkpoints, saliency maps, split paths,
        and configuration metadata are saved.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if num_epochs < 1:
        raise ValueError("num_epochs must be at least 1.")

    if n_real < 1 or n_fake < 1:
        raise ValueError("n_real and n_fake must both be at least 1.")

    if not 0.0 < train_fraction < 1.0:
        raise ValueError(
            "train_fraction must be strictly between 0 and 1."
        )

    set_seed(seed)

    resolved_device = resolve_device(device)

    workers = (
        default_num_workers()
        if num_workers is None
        else num_workers
    )

    if workers < 0:
        raise ValueError("num_workers cannot be negative.")

    if resolved_device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    print(f"Using device: {resolved_device}")

    # Use different deterministic seeds for the two independent class splits.
    real_split_seed = seed
    fake_split_seed = None if seed is None else seed + 1

    real_split = discover_and_split_images(
        real_dir,
        train_fraction=train_fraction,
        seed=real_split_seed,
    )

    fake_split = discover_and_split_images(
        fake_dir,
        train_fraction=train_fraction,
        seed=fake_split_seed,
    )

    print(
        f"Real images: "
        f"{len(real_split.train_paths)} train / "
        f"{len(real_split.test_paths)} test"
    )

    print(
        f"Fake images: "
        f"{len(fake_split.train_paths)} train / "
        f"{len(fake_split.test_paths)} test"
    )

    # Select the test population once. The exact same sampled images are used
    # for accuracy and saliency calculations.
    sampled_real_test_paths = sample_paths(
        real_split.train_paths,
        n_samples=n_real,
        seed=None if seed is None else seed + 100,
    )

    sampled_fake_test_paths = sample_paths(
        fake_split.train_paths,
        n_samples=n_fake,
        seed=None if seed is None else seed + 101,
    )

    print(
        f"Test images used for SAS: "
        f"{len(sampled_real_test_paths)} real / "
        f"{len(sampled_fake_test_paths)} fake"
    )

    ig_batch_size = derive_ig_batch_size(batch_size)

    print(
        f"Batch sizes: "
        f"training={batch_size}, "
        f"evaluation={batch_size}, "
        f"IG={ig_batch_size}"
    )

    model = train_gsc_classifier(
        real_train_paths=real_split.train_paths,
        fake_train_paths=fake_split.train_paths,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        num_workers=workers,
        device=resolved_device,
    )

    # Explicitly retain evaluation mode for all post-training operations.
    model.eval()

    real_maps, fake_maps = extract_saliency_maps(
        model,
        real_test_paths=sampled_real_test_paths,
        fake_test_paths=sampled_fake_test_paths,
        batch_size=ig_batch_size,
        num_workers=workers,
        device=resolved_device,
    )

    model.eval()

    score = compute_sas_metric(
        model,
        real_test_paths=sampled_real_test_paths,
        fake_test_paths=sampled_fake_test_paths,
        real_maps=real_maps,
        fake_maps=fake_maps,
        batch_size=batch_size,
        num_workers=workers,
        device=resolved_device,
    )

    if output_dir is not None:
        _save_artifacts(
            output_dir=Path(output_dir),
            model=model,
            real_split=real_split,
            fake_split=fake_split,
            sampled_real_test_paths=sampled_real_test_paths,
            sampled_fake_test_paths=sampled_fake_test_paths,
            real_maps=real_maps,
            fake_maps=fake_maps,
            score=score,
            batch_size=batch_size,
            ig_batch_size=ig_batch_size,
            device=str(resolved_device),
            seed=seed,
            train_fraction=train_fraction,
        )

    return float(score)


def _save_artifacts(
    *,
    output_dir: Path,
    model: torch.nn.Module,
    real_split: ImageSplit,
    fake_split: ImageSplit,
    sampled_real_test_paths,
    sampled_fake_test_paths,
    real_maps,
    fake_maps,
    score: float,
    batch_size: int,
    ig_batch_size: int,
    device: str,
    seed: int | None,
    train_fraction: float,
) -> None:
    """Save optional model, maps, split manifests, and run metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        model.state_dict(),
        output_dir / "gsc_classifier.pth",
    )

    with (output_dir / "integrated_gradients.pkl").open("wb") as file:
        pickle.dump(
            {
                "real": real_maps,
                "fake": fake_maps,
            },
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    split_manifest = {
        "real_train": [str(path) for path in real_split.train_paths],
        "real_test": [str(path) for path in real_split.test_paths],
        "fake_train": [str(path) for path in fake_split.train_paths],
        "fake_test": [str(path) for path in fake_split.test_paths],
        "sampled_real_test": [
            str(path) for path in sampled_real_test_paths
        ],
        "sampled_fake_test": [
            str(path) for path in sampled_fake_test_paths
        ],
    }

    with (output_dir / "splits.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(split_manifest, file, indent=2)

    run_manifest = {
        "sas": score,
        "seed": seed,
        "device": device,
        "train_fraction": train_fraction,
        "batch_size": batch_size,
        "ig_batch_size": ig_batch_size,
        "num_real_train": len(real_split.train_paths),
        "num_real_test": len(real_split.test_paths),
        "num_fake_train": len(fake_split.train_paths),
        "num_fake_test": len(fake_split.test_paths),
        "num_real_test_sampled": len(sampled_real_test_paths),
        "num_fake_test_sampled": len(sampled_fake_test_paths),
        "ig_steps": 50,
        "noise_tunnel_samples": 3,
        "noise_tunnel_standard_deviation": 0.1,
    }

    with (output_dir / "run_manifest.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(run_manifest, file, indent=2)
