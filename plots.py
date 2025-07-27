import matplotlib.pyplot as plt
from matplotlib import rc

import numpy as np
import itertools


# # OLD
# def plot_comparison(
#     ns,
#     avg_kmeans_times,
#     avg_eekmeans_times,
#     numberkmeansiteration=None,
#     numbereekmeansiterations=None,
#     avg_P_init_times=None,
#     avg_Q_init_times=None,
#     filename="plot_comparison.pdf",
# ):
#     plt.figure(figsize=(10, 6))
#     plt.plot(ns, avg_kmeans_times, "b-o", label="KMeans iterations")
#     plt.plot(ns, avg_eekmeans_times, "r-o", label="EEKMeans iterations")
#     if avg_P_init_times is not None:
#         plt.plot(ns, avg_P_init_times, "g--s", label="EEKMeans P init")
#     if avg_Q_init_times is not None:
#         plt.plot(ns, avg_Q_init_times, "m--^", label="EEKMeans Q init")

#     # Improved annotation: use bbox for better visibility, offset points more, and use bold font
#     if numberkmeansiteration is not None:
#         for x, y, num_iter in zip(ns, avg_kmeans_times, numberkmeansiteration):
#             plt.annotate(
#                 f"{num_iter}",
#                 (x, y),
#                 textcoords="offset points",
#                 xytext=(0, 15),
#                 ha="center",
#                 color="blue",
#                 fontsize=10,
#                 fontweight="bold",
#                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.7),
#             )
#     if numbereekmeansiterations is not None:
#         for x, y, num_iter in zip(ns, avg_eekmeans_times, numbereekmeansiterations):
#             plt.annotate(
#                 f"{num_iter}",
#                 (x, y),
#                 textcoords="offset points",
#                 xytext=(0, -25),
#                 ha="center",
#                 color="red",
#                 fontsize=10,
#                 fontweight="bold",
#                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.7),
#             )
#     plt.xlabel("Dataset size (n)")
#     plt.ylabel("Time (seconds)")
#     plt.title("Algorithm runtime vs dataset size")
#     plt.grid(True, linestyle="--", alpha=0.7)
#     plt.legend()
#     plt.savefig(filename)


def plot_experiment_one(averaged_results, filename="experiment_one_results.pdf"):
    plt.figure(figsize=(10, 6))
    for label, data in averaged_results.items():
        plt.plot(data["avg_movements"], label=label)
    plt.xlabel("Iteration")
    plt.ylabel("Average Movement")
    plt.yscale("log")
    plt.title("Experiment One: Average Movement per Iteration")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.savefig(filename)
    return


def plot_exp_two_iteration_time(
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

    # Plot EEKMeans times for each epsilon
    colors = [
        color for color in plt.cm.tab10.colors
    ]  # Use tab10 colormap for up to 10 distinct colors

    for i, (epsilon, times) in enumerate(eekmeans_times.items()):
        color = colors[i % len(colors)]
        plt.plot(
            sizes_datasets,
            times,
            "-o",
            label=f"EEKMeans (ε={epsilon})",
            color=color,
        )

    # Add margin to y-limits for annotation visibility
    all_y = kmeans_times + [time for times in eekmeans_times.values() for time in times]
    ymin, ymax = min(all_y), max(all_y)
    y_range = ymax - ymin
    plt.ylim(ymin - 0.2 * y_range, ymax + 0.15 * y_range)

    # Annotate KMeans points with iteration numbers if provided
    if iteration_number_kmeans is not None:
        for i, (x, y, num_iter) in enumerate(
            zip(sizes_datasets, kmeans_times, iteration_number_kmeans)
        ):
            xytext = (0, 10)
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

    # Annotate EEKMeans points with iteration numbers for each epsilon
    if iteration_number_eekmeans is not None:
        for epsilon, iter_numbers in iteration_number_eekmeans.items():
            for i, (x, y, num_iter) in enumerate(
                zip(sizes_datasets, eekmeans_times[epsilon], iter_numbers)
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
                    bbox=dict(
                        boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.7
                    ),
                )

    plt.xlabel("Dataset size (n)")
    plt.ylabel("Time (seconds)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_exp_three_rss(
    sizes_datasets,
    rss_differences,
    filename="rss_differences.pdf",
    title="Average RSS Difference vs Dataset Size",
):
    plt.figure(figsize=(8, 5))

    # Plot RSS differences for each epsilon
    colors = [
        color for color in plt.cm.tab10.colors
    ]  # Use tab10 colormap for up to 10 distinct colors

    for i, (epsilon, rss_diff) in enumerate(rss_differences.items()):
        color = colors[i % len(colors)]
        plt.plot(
            sizes_datasets,
            rss_diff,
            "-o",
            label=f"RSS Difference (ε={epsilon})",
            color=color,
        )

    # Add margin to y-limits for better visibility
    all_y = [value for values in rss_differences.values() for value in values]
    ymin, ymax = min(all_y), max(all_y)
    y_range = ymax - ymin
    plt.ylim(ymin - 0.2 * y_range, ymax + 0.15 * y_range)

    plt.xlabel("Dataset size (n)")
    plt.ylabel("RSS Difference")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_exp_four_rss(
    thetas,
    rss_differences,
    filename="rss_differences.pdf",
    title="Average RSS Difference vs Theta",
):

    print("Debug: thetas =", thetas)
    print("Debug: rss_differences =", rss_differences)
    print("Debug: rss_differences_values =", list(rss_differences.values()))

    plt.figure(figsize=(8, 5))

    # Plot RSS differences for each epsilon
    colors = [
        color for color in plt.cm.tab10.colors
    ]  # Use tab10 colormap for up to 10 distinct colors

    for i, (epsilon, rss_diff) in enumerate(rss_differences.items()):
        color = colors[i % len(colors)]
        plt.plot(
            thetas,
            rss_diff,
            "-o",
            label=f"RSS Difference (ε={epsilon})",
            color=color,
        )

    # Add margin to y-limits for better visibility
    all_y = [value for values in rss_differences.values() for value in values]
    ymin, ymax = min(all_y), max(all_y)
    y_range = ymax - ymin
    plt.ylim(ymin - 0.2 * y_range, ymax + 0.15 * y_range)
    # Set x-ticks to thetas for better readability
    plt.xticks(thetas, rotation=45)
    plt.xlabel("Theta")
    plt.ylabel("RSS Difference")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
