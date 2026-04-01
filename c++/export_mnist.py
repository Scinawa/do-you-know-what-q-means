#!/usr/bin/env python3
"""
Export MNIST dataset to binary format for C++ import.

Binary format:
- Header: int N (number of samples), int d (dimensions)
- Data: N * d doubles in row-major order (sample 0 all dims, sample 1 all dims, etc.)
- No normalization (raw pixel values 0-255)
"""

import numpy as np
import struct
import argparse
from sklearn.datasets import fetch_openml


def export_mnist_binary(output_file="mnist_data.bin"):
    """
    Export MNIST dataset to binary format.

    Parameters
    ----------
    output_file : str
        Path to output binary file
    """
    print("Loading MNIST dataset...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist["data"]

    N, d = X.shape
    print(f"Dataset shape: {N} samples, {d} dimensions")
    print(f"Data range: [{X.min():.1f}, {X.max():.1f}]")

    # Convert to double precision (float64)
    X_double = X.astype(np.float64)

    print(f"Writing to {output_file}...")
    with open(output_file, "wb") as f:
        # Write header: N (int), d (int)
        f.write(struct.pack("i", N))  # 4 bytes for int
        f.write(struct.pack("i", d))  # 4 bytes for int

        # Write data: N * d doubles in row-major order
        # Flatten the array to row-major order (C-style)
        X_flat = X_double.flatten("C")
        X_flat.tofile(f)

    print(f"Successfully exported {N} samples with {d} dimensions to {output_file}")
    print(f"File size: {8 + N * d * 8} bytes (8 bytes header + {N * d * 8} bytes data)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export MNIST dataset to binary format"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="mnist_data.bin",
        help="Output binary file path (default: mnist_data.bin)",
    )

    args = parser.parse_args()
    export_mnist_binary(args.output)
