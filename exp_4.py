import logging
import os
import datetime

# import time
import pickle
import numpy as np


from plots import plot_exp_four_rss
from utils import (
    create_extended_dataset,
    make_dataset_skewed,
    sample_gaussian_mixture,
    stepwise_kmeans,
)

os.environ["NUMEXPR_MAX_THREADS"] = "16"


def postprocess_results(results, thetas):
    # Compute RSS differences between KMeans and other algorithms
    rss_differences = {}

    for key in results.keys():
        if key == "KMeans":
            continue

        # Compute RSS differences for each theta
        rss_differences[key] = []
        for theta in thetas:
            kmeans_rss = [
                clustering_event[-1]["RSS"]
                for clustering_event in results["KMeans"][theta]
            ]
            algorithm_rss = [
                clustering_event[-1]["RSS"] for clustering_event in results[key][theta]
            ]

            # Compute the difference in averages for the current theta
            rss_differences[key].append(np.mean(kmeans_rss) - np.mean(algorithm_rss))

    return rss_differences


def experiment_four(
    logger,
    size_dataset,
    thetas,
    repetitions,
    n_clusters,
    max_iter,
    dataset,
    tol,
    epsilons,
    delta,
    constant_enabled,
    sample_beginning,
    filename,
    read=None,
):
    # # we have a single one in exp2
    # epsilon = epsilons[0]

    if read is not None:
        with open(read, "rb") as f:
            results = pickle.load(f)
    else:
        # Initialize results dictionary
        results = {"KMeans": {}}
        for epsilon in epsilons:
            results[f"EEKMeans-ε={epsilon}"] = {}

        for theta in thetas:
            logger.info(f"Running experiment for theta={theta}")

            for key in results.keys():
                results[key][theta] = []

            if dataset == "gaussian_mixture":
                X, Y = sample_gaussian_mixture(n=size_dataset, d=1000, k=n_clusters)
            elif dataset == "mnist":
                X, Y = create_extended_dataset(size_dataset)
            else:
                raise ValueError(
                    "Invalid dataset. Choose 'gaussian_mixture' or 'mnist'."
                )

            # TODO skew the dataset if needed
            X, Y = make_dataset_skewed(X, Y, theta, logger)
            n = X.shape[0]

            for i in range(repetitions):
                logger.info(f"Repetition {i + 1} of {repetitions}")
                # # KMeans
                results["KMeans"][theta].append(
                    stepwise_kmeans(
                        X,
                        n_clusters=n_clusters,
                        max_iter=max_iter,
                        tol=tol,
                        algorithm="KMeans",
                        logger=logger,
                        seed_for_centroids=i,
                    )
                )

                # EEKMeans
                for epsilon in epsilons:
                    logger.info(
                        f"Rep. {i + 1} of {repetitions} with EEKMeans-ε={epsilon}"
                    )

                    # EEKMeans
                    results[f"EEKMeans-ε={epsilon}"][theta].append(
                        stepwise_kmeans(
                            X,
                            n_clusters=n_clusters,
                            max_iter=max_iter,
                            tol=tol,
                            algorithm="EEKMeans",
                            logger=logger,
                            # Parameters for EEKMeans
                            epsilon=epsilon,
                            delta=delta,
                            constant_enabled=constant_enabled,
                            sample_beginning=sample_beginning,
                            seed_for_centroids=i,
                        )
                    )

                logger.info("*" * 50)

        # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename
        with open(filename + ".pkl", "wb") as f:
            pickle.dump(results, f)
        logger.info(f"Results saved to {filename}")

    # Postprocess results to get RSS differences for each theta
    rss_differences = postprocess_results(results, thetas)
    # rss_differences is a DICTIONARY with keys as algorithm names and values as lists of RSS differences

    # Plot average RSS differences
    plot_exp_four_rss(
        thetas,
        rss_differences=rss_differences,  # pass the dictionary of rss differences
        filename="rss_differences_" + filename + ".pdf",
        title="Average RSS Difference vs Theta",
    )
    logger.info("Average RSS differences plotted.")


# Example usage
if __name__ == "__main__":
    # np.random.seed(36)  # Fix randomness globally
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger("EXP4")

    size_dataset = 40000  # Size of the dataset
    thetas = np.linspace(0.05, 1, 5, dtype=float)
    repetitions = 6
    n_clusters = 10
    max_iter = 40
    tol = 15
    epsilons = (200, 300, 400)  # , 400, 500)  # (250, 450) --- IGNORE ---
    delta = 0.5
    constant_enabled = False
    sample_beginning = True
    dataset = "mnist"  # "gaussian_mixture" or "mnist"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    param_str = (
        f"dataset_{dataset}_k_{n_clusters}_maxiter_{max_iter}_tol_{tol}_eps_{epsilons}_delta_{delta}_"
        f"constenabled_{constant_enabled}_samplebeginning_{sample_beginning}_reps_{repetitions}_"
        f"t_{timestamp}"
    )

    experiment_four(
        logger,
        size_dataset,
        thetas,
        repetitions,
        n_clusters,
        max_iter,
        dataset,
        tol,
        epsilons,
        delta,
        constant_enabled,
        sample_beginning,
        filename=f"experiment_four_results_{param_str}",
        # read="experiment_four_results_dataset_mnist_k_10_maxiter_40_tol_15_eps_(200, 300)_delta_0.5_constenabled_False_samplebeginning_True_reps_2_t_20250727_132956.pkl",
    )
