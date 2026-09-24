# src/sas_eval/cli.py

from __future__ import annotations

import argparse

from .pipeline import compute_sas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete SAS evaluation pipeline."
    )

    parser.add_argument("--real-train-dir", required=True)
    parser.add_argument("--fake-train-dir", required=True)
    parser.add_argument("--real-test-dir", required=True)
    parser.add_argument("--fake-test-dir", required=True)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-real", type=int, default=5000)
    parser.add_argument("--n-fake", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=None)

    args = parser.parse_args()

    score = compute_sas(
        real_train_dir=args.real_train_dir,
        fake_train_dir=args.fake_train_dir,
        real_test_dir=args.real_test_dir,
        fake_test_dir=args.fake_test_dir,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        n_real=args.n_real,
        n_fake=args.n_fake,
        num_workers=args.num_workers,
    )

    print(f"SAS: {score:.4f}")


if __name__ == "__main__":
    main()