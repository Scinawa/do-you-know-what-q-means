import matplotlib.pyplot as plt
from matplotlib import rc

import numpy as np


def plot_comparison(
    ns,
    avg_kmeans_times,
    avg_eekmeans_times,
    numberkmeansiteration=None,
    numbereekmeansiterations=None,
    avg_P_init_times=None,
    avg_Q_init_times=None,
    filename="plot_comparison.pdf",
):
    plt.figure(figsize=(10, 6))
    plt.plot(ns, avg_kmeans_times, "b-o", label="KMeans iterations")
    plt.plot(ns, avg_eekmeans_times, "r-o", label="EEKMeans iterations")
    if avg_P_init_times is not None:
        plt.plot(ns, avg_P_init_times, "g--s", label="EEKMeans P init")
    if avg_Q_init_times is not None:
        plt.plot(ns, avg_Q_init_times, "m--^", label="EEKMeans Q init")

    # Improved annotation: use bbox for better visibility, offset points more, and use bold font
    if numberkmeansiteration is not None:
        for x, y, num_iter in zip(ns, avg_kmeans_times, numberkmeansiteration):
            plt.annotate(
                f"{num_iter}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 15),
                ha="center",
                color="blue",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.7),
            )
    if numbereekmeansiterations is not None:
        for x, y, num_iter in zip(ns, avg_eekmeans_times, numbereekmeansiterations):
            plt.annotate(
                f"{num_iter}",
                (x, y),
                textcoords="offset points",
                xytext=(0, -25),
                ha="center",
                color="red",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.7),
            )
    plt.xlabel("Dataset size (n)")
    plt.ylabel("Time (seconds)")
    plt.title("Algorithm runtime vs dataset size")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.savefig(filename)


def plot_iteration_time(
    sizes_datasets,
    kmeans_times,
    eekmeans_times,
    title,
    iteration_number_kmeans=None,
    iteration_number_eekmeans=None,
    filename="average_iteration_time.pdf",
):
    plt.figure(figsize=(8, 5))
    plt.plot(sizes_datasets, kmeans_times, "b-o", label="KMeans")
    plt.plot(sizes_datasets, eekmeans_times, "r-o", label="EEKMeans")

    # Add margin to y-limits for annotation visibility
    all_y = kmeans_times + eekmeans_times
    ymin, ymax = min(all_y), max(all_y)
    y_range = ymax - ymin
    plt.ylim(ymin - 0.2 * y_range, ymax + 0.15 * y_range)

    # Annotate each point with its iteration number if provided
    if iteration_number_kmeans is not None:
        for i, (x, y, num_iter) in enumerate(
            zip(sizes_datasets, kmeans_times, iteration_number_kmeans)
        ):
            # Adjust annotation position for first/last points if needed
            xytext = (0, 15)
            if i == len(sizes_datasets) - 1:
                xytext = (-20, 10)
            plt.annotate(
                f"{num_iter}",
                (x, y),
                textcoords="offset points",
                xytext=xytext,
                ha="center",
                color="blue",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.7),
            )
    if iteration_number_eekmeans is not None:
        for i, (x, y, num_iter) in enumerate(
            zip(sizes_datasets, eekmeans_times, iteration_number_eekmeans)
        ):
            xytext = (0, -25)
            if i == 0:
                xytext = (20, -25)
            plt.annotate(
                f"{num_iter}",
                (x, y),
                textcoords="offset points",
                xytext=xytext,
                ha="center",
                color="red",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.7),
            )

    plt.xlabel("Dataset size (n)")
    plt.ylabel("Time (seconds)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
