import argparse
import pickle
import re
from pathlib import Path

import numpy as np


MARKERS = ["v", "^", "s", "x", "*", "D", "P", "h"]
COLORS = ["blue", "green", "orange", "red", "purple", "brown", "pink", "gray"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot EXP3 RSS differences between EEKMeans and KMeans."
    )
    parser.add_argument("pkl_path", type=Path, help="Path to an EXP3 results pickle.")
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of EEKMeans epsilon curves to plot after sorting by epsilon.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PDF path. Defaults to plots/exp3_rss_vs_theta_<input-stem>.pdf.",
    )
    return parser.parse_args()


def epsilon_from_algorithm_name(name):
    match = re.search(r"=([0-9]+(?:\.[0-9]+)?)", name)
    if match is None:
        raise ValueError(f"Could not parse epsilon from algorithm name: {name}")
    return float(match.group(1))


def format_epsilon(epsilon):
    return str(int(epsilon)) if epsilon.is_integer() else str(epsilon)


def load_results(pkl_path):
    with pkl_path.open("rb") as results_file:
        return pickle.load(results_file)


def select_eekmeans_algorithms(results, k):
    if k < 1:
        raise ValueError("--k must be at least 1")

    algorithms = [
        (epsilon_from_algorithm_name(name), name)
        for name in results.keys()
        if name != "KMeans"
    ]
    if not algorithms:
        raise ValueError("No EEKMeans results were found in the pickle.")

    return sorted(algorithms)[:k]


def final_rss_values(results, algorithm, theta):
    values = []
    for repetition in results[algorithm][theta]:
        if not repetition:
            raise ValueError(
                f"Found an empty repetition for algorithm {algorithm}, theta {theta}."
            )
        values.append(repetition[-1]["RSS"])
    return values


def compute_rss_differences(results, selected_algorithms):
    if "KMeans" not in results:
        raise ValueError("The pickle does not contain KMeans baseline results.")

    thetas = list(results["KMeans"].keys())
    theta_values = [float(theta) for theta in thetas]
    rss_differences = {}

    for epsilon, algorithm in selected_algorithms:
        differences = []
        for theta in thetas:
            kmeans_rss = final_rss_values(results, "KMeans", theta)
            eekmeans_rss = final_rss_values(results, algorithm, theta)
            if len(kmeans_rss) != len(eekmeans_rss):
                raise ValueError(
                    f"Mismatched repetition counts for {algorithm}, theta {theta}: "
                    f"{len(kmeans_rss)} KMeans vs {len(eekmeans_rss)} EEKMeans."
                )

            paired_percent_differences = [
                ((eekmeans_value - kmeans_value) / kmeans_value) * 100
                for kmeans_value, eekmeans_value in zip(kmeans_rss, eekmeans_rss)
            ]
            differences.append(np.mean(paired_percent_differences))
        rss_differences[epsilon] = differences

    return theta_values, rss_differences


def default_output_path(pkl_path):
    return Path("plots") / f"exp3_rss_vs_theta_{pkl_path.stem}.pdf"


def plot_rss_differences(thetas, rss_differences, output_path):
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401 - registers the "science" Matplotlib style.

    plt.style.use("science")
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams.update({"font.size": 13})

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 2, 2])
    plt.grid(color="gray", linestyle="dotted", linewidth=0.1)

    for i, (epsilon, rss_diff) in enumerate(rss_differences.items()):
        marker = MARKERS[i % len(MARKERS)]
        color = COLORS[i % len(COLORS)]
        marker_size = 11 if marker in {"x", "*"} else 9
        ax.plot(
            thetas,
            rss_diff,
            label=r"$\varepsilon$ = " + format_epsilon(epsilon),
            color=color,
            marker=marker,
            linewidth=2.0,
            markersize=marker_size,
        )

    ax.tick_params(
        axis="both", which="major", direction="inout", length=8, width=2, labelsize=17
    )
    ax.tick_params(axis="both", which="minor", direction="inout", length=4, width=1)
    ax.set_xlabel(r"Class imbalance $\theta$", fontsize=20)
    ax.set_ylabel(
        r"Relative Residual Sum of Squares ($\Delta\mathrm{RSS}$) (\%)",
        fontsize=20,
    )
    plt.xlim([min(thetas), max(thetas)])
    legend = ax.legend(loc="upper left", prop={"size": 13}, shadow=True, frameon=True)
    legend.get_frame().set_alpha(0.7)
    legend.get_frame().set_facecolor("white")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()


def print_summary(output_path, thetas, rss_differences):
    print(f"Saved plot to {output_path}")
    print("Theta values:", [round(theta, 4) for theta in thetas])
    print("Selected epsilons:", [format_epsilon(eps) for eps in rss_differences.keys()])
    print("RSS differences (%):")
    for epsilon, differences in rss_differences.items():
        print(
            f"  epsilon {format_epsilon(epsilon)}:",
            [round(float(value), 3) for value in differences],
        )


def main():
    args = parse_args()
    output_path = (
        args.output if args.output is not None else default_output_path(args.pkl_path)
    )

    results = load_results(args.pkl_path)
    selected_algorithms = select_eekmeans_algorithms(results, args.k)
    thetas, rss_differences = compute_rss_differences(results, selected_algorithms)
    plot_rss_differences(thetas, rss_differences, output_path)
    print_summary(output_path, thetas, rss_differences)


if __name__ == "__main__":
    main()
