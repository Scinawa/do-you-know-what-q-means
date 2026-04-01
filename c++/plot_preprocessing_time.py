#!/usr/bin/env python3
"""
Plot preprocessing time against dataset size from preprocessing_data.txt.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_input_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        REPO_ROOT / path,
        Path(__file__).resolve().parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return REPO_ROOT / path


def resolve_output_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_preprocessing_points(path: Path) -> tuple[list[int], list[float]]:
    dataset_sizes: list[int] = []
    preprocessing_times_ms: list[float] = []

    with path.open("r", encoding="utf-8") as input_file:
        for raw_line in input_file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Expected 2 columns in '{line}', found {len(parts)}")

            dataset_sizes.append(int(parts[0]))
            preprocessing_times_ms.append(float(parts[1]))

    if not dataset_sizes:
        raise ValueError(f"No preprocessing points found in {path}")

    return dataset_sizes, preprocessing_times_ms


def plot_preprocessing_time(
    dataset_sizes: list[int], preprocessing_times_ms: list[float], output_path: Path
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(dataset_sizes, preprocessing_times_ms, "o-", linewidth=2)
    plt.xlabel("Dataset size (n)")
    plt.ylabel("Preprocessing time (ms)")
    plt.title("Preprocessing Time vs Dataset Size")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot preprocessing time against dataset size."
    )
    parser.add_argument(
        "-i",
        "--input",
        default="preprocessing_data.txt",
        help="Input whitespace-separated file written by q_means_one_sample.cpp",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="plots/preprocessing_time_vs_dataset_size.pdf",
        help="Output plot path",
    )
    args = parser.parse_args()

    input_path = resolve_input_path(args.input)
    output_path = resolve_output_path(args.output)

    dataset_sizes, preprocessing_times_ms = load_preprocessing_points(input_path)
    plot_preprocessing_time(dataset_sizes, preprocessing_times_ms, output_path)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
