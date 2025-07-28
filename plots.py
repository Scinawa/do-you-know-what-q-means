import matplotlib.pyplot as plt
from matplotlib import rc

import numpy as np
import itertools


def plot_movements(averaged_results, filename="experiment_one_results.pdf"):
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


def plot_time(
    x_axis_name,
    x_values,
    kmeans_times,
    eekmeans_times,
    title,
    iteration_number_kmeans=None,
    iteration_number_eekmeans=None,
    filename="average_iteration_time.pdf",
):
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, kmeans_times, "b-o", label="KMeans")

    # Plot EEKMeans times for each epsilon
    colors = [
        color for color in plt.cm.tab10.colors
    ]  # Use tab10 colormap for up to 10 distinct colors

    for i, (epsilon, times) in enumerate(eekmeans_times.items()):
        color = colors[i % len(colors)]
        plt.plot(
            x_values,
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
            zip(x_values, kmeans_times, iteration_number_kmeans)
        ):
            xytext = (0, 10)
            if i == len(x_values) - 1:
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
                zip(x_values, eekmeans_times[epsilon], iter_numbers)
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

    plt.xlabel(x_axis_name)
    plt.ylabel("Time (seconds)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# keep this comment
# previously called plot_exp_three_rss
def plot_rss(
    x_axis_name,
    x_values,
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
            x_values,
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

    plt.xlabel(x_axis_name)
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


def plot_iterations(
    x_axis_name,
    x_values,
    iteration_number_kmeans,
    iteration_number_eekmeans,
    title,
    filename="number_of_iterations.pdf",
):
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, iteration_number_kmeans, "b-o", label="KMeans")

    # Plot EEKMeans iterations for each epsilon
    colors = [
        color for color in plt.cm.tab10.colors
    ]  # Use tab10 colormap for up to 10 distinct colors

    for i, (epsilon, iter_numbers) in enumerate(iteration_number_eekmeans.items()):
        color = colors[i % len(colors)]
        plt.plot(
            x_values,
            iter_numbers,
            "-o",
            label=f"EEKMeans (ε={epsilon})",
            color=color,
        )

    # Add margin to y-limits for better visibility
    all_y = iteration_number_kmeans + [
        num
        for iter_numbers in iteration_number_eekmeans.values()
        for num in iter_numbers
    ]
    ymin, ymax = min(all_y), max(all_y)
    y_range = ymax - ymin
    plt.ylim(ymin - 0.2 * y_range, ymax + 0.15 * y_range)

    plt.xlabel(x_axis_name)
    plt.ylabel("Number of Iterations")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
