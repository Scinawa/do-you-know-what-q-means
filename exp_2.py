import logging
import os
import pickle
import datetime

import numpy as np

from plots import plot_exp_two_iteration_time
from utils import create_extended_dataset, sample_gaussian_mixture, stepwise_kmeans

os.environ["NUMEXPR_MAX_THREADS"] = "16"


def postprocess_results(results):
    # Calculate average iteration durations for KMeans and EEKMeans
    # Initialize dictionaries to store averages for each algorithm
    avg_single_iteration_duration = {key: [] for key in results.keys()}
    avg_total_iteration_duration = {key: [] for key in results.keys()}
    avg_number_of_iterations = {key: [] for key in results.keys()}
    avg_total_clustering_duration = {key: [] for key in results.keys()}

    for key in results.keys():
        for n in sizes_datasets:

            avg_single_iteration_duration[key].append(
                np.mean(
                    [
                        np.mean(
                            [
                                iteration["iteration_duration"]
                                for iteration in results[key][n][rep]
                            ]
                        )
                        for rep in range(len(results[key][n]))
                    ]
                )
            )

            avg_total_iteration_duration[key].append(
                np.mean(
                    [
                        np.sum(
                            [
                                iteration["iteration_duration"]
                                for iteration in results[key][n][rep]
                            ]
                        )
                        for rep in range(len(results[key][n]))
                    ]
                )
            )

            avg_number_of_iterations[key].append(
                np.mean([len(clustering_event) for clustering_event in results[key][n]])
            )
            avg_total_clustering_duration[key].append(
                np.mean(
                    [
                        np.sum(
                            [
                                iteration["iteration_duration"]
                                for iteration in results[key][n][rep]
                            ]
                        )
                        + np.sum(
                            [
                                iteration["P_times"]
                                for iteration in results[key][n][rep]
                                if iteration["P_times"] is not None
                            ]
                        )
                        + np.sum(
                            [
                                iteration["Q_times"]
                                for iteration in results[key][n][rep]
                                if iteration["Q_times"] is not None
                            ]
                        )
                        for rep in range(len(results[key][n]))
                    ]
                )
            )

    return (
        avg_single_iteration_duration,
        avg_total_iteration_duration,
        avg_number_of_iterations,
        avg_total_clustering_duration,
    )


def experiment_two(
    logger,
    sizes_datasets,
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

        for n in sizes_datasets:
            logger.info(f"Running experiment for n={n} (dataset size: {n})")

            for key in results.keys():
                results[key][n] = []

            if dataset == "gaussian_mixture":
                X, _ = sample_gaussian_mixture(n=n, d=1000, k=n_clusters)
            elif dataset == "mnist":
                X, _ = create_extended_dataset(n)
            else:
                raise ValueError(
                    "Invalid dataset. Choose 'gaussian_mixture' or 'mnist'."
                )

            # TODO skew the dataset if needed
            # X, Y = skew_dataset(X, Y, n, n_clusters)
            # n = X.shape[0]

            for i in range(repetitions):
                logger.info(f"Repetition {i + 1} of {repetitions}")
                # # KMeans
                results["KMeans"][n].append(
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
                    results[f"EEKMeans-ε={epsilon}"][n].append(
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

    # Postprocess results to get average iteration durations, total iteration durations, and average number of iterations
    (
        average_single_iteration_durations,
        average_total_iteration_durations,
        average_number_of_iterations,
        average_total_clustering_duration,
    ) = postprocess_results(results)

    # Plot average iteration times
    plot_exp_two_iteration_time(
        sizes_datasets,
        kmeans_times=average_single_iteration_durations["KMeans"],
        eekmeans_times={
            epsilon: average_single_iteration_durations[f"EEKMeans-ε={epsilon}"]
            for epsilon in epsilons
        },
        filename="average_iteration_times_" + filename + ".pdf",
        title="Average Single Iteration Time vs Dataset Size",
    )
    logger.info("Average iteration times plotted.")
    # Plot average total iteration times
    plot_exp_two_iteration_time(
        sizes_datasets,
        kmeans_times=average_total_iteration_durations["KMeans"],
        eekmeans_times={
            epsilon: average_total_iteration_durations[f"EEKMeans-ε={epsilon}"]
            for epsilon in epsilons
        },
        iteration_number_kmeans=average_number_of_iterations["KMeans"],
        iteration_number_eekmeans={
            epsilon: average_number_of_iterations[f"EEKMeans-ε={epsilon}"]
            for epsilon in epsilons
        },
        filename="average_total_iteration_time_" + filename + ".pdf",
        title="Average Total Iteration Time vs Dataset Size",
    )
    logger.info("Average total iteration times plotted.")

    # Plot average total clustering durations
    plot_exp_two_iteration_time(
        sizes_datasets,
        kmeans_times=average_total_clustering_duration["KMeans"],
        eekmeans_times={
            epsilon: average_total_clustering_duration[f"EEKMeans-ε={epsilon}"]
            for epsilon in epsilons
        },
        iteration_number_kmeans=average_number_of_iterations["KMeans"],
        iteration_number_eekmeans={
            epsilon: average_number_of_iterations[f"EEKMeans-ε={epsilon}"]
            for epsilon in epsilons
        },
        filename="average_total_clustering_duration_" + filename + ".pdf",
        title="Average Total Clustering Duration vs Dataset Size",
    )
    logger.info("Average total clustering durations plotted.")


# Example usage
if __name__ == "__main__":
    # np.random.seed(36)  # Fix randomness globally
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger("EXP2")

    sizes_datasets = np.linspace(60000, 200000, 5, dtype=int)
    repetitions = 4
    n_clusters = 10
    max_iter = 65
    tol = 15
    epsilons = (200, 300, 400, 500)  # (250, 450) --- IGNORE ---
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

    experiment_two(
        logger,
        sizes_datasets,
        repetitions,
        n_clusters,
        max_iter,
        dataset,
        tol,
        epsilons,
        delta,
        constant_enabled,
        sample_beginning,
        filename=f"experiment_two_results_{param_str}",
        # read="experiment_two_results_dataset_mnist_k_10_maxiter_65_tol_15_eps_(250, 450)_delta_0.5_constenabled_False_samplebeginning_True_reps_2_t_20250726_152659.pkl",  # "results_20250722_121746.pkl",
    )
