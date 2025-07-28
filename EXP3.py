import logging
import os
import datetime

# import time
import pickle
import numpy as np


from plots import (
    plot_exp_four_rss,
    plot_time,
    plot_iterations,
)
from utils import (
    create_extended_dataset,
    make_dataset_skewed,
    sample_gaussian_mixture,
    stepwise_kmeans,
)

os.environ["NUMEXPR_MAX_THREADS"] = "16"


def postprocess_results_rss(results, thetas):
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


def postprocess_results_time(results, thetas):
    # Calculate average iteration durations for KMeans and EEKMeans
    # Initialize dictionaries to store averages for each algorithm
    avg_single_iteration_duration = {key: [] for key in results.keys()}
    avg_total_iteration_duration = {key: [] for key in results.keys()}
    avg_number_of_iterations = {key: [] for key in results.keys()}
    avg_total_clustering_duration = {key: [] for key in results.keys()}

    for key in results.keys():
        for theta in thetas:
            avg_single_iteration_duration[key].append(
                np.mean(
                    [
                        np.mean(
                            [
                                iteration["iteration_duration"]
                                for iteration in results[key][theta][rep]
                            ]
                        )
                        for rep in range(len(results[key][theta]))
                    ]
                )
            )

            avg_total_iteration_duration[key].append(
                np.mean(
                    [
                        np.sum(
                            [
                                iteration["iteration_duration"]
                                for iteration in results[key][theta][rep]
                            ]
                        )
                        for rep in range(len(results[key][theta]))
                    ]
                )
            )

            avg_number_of_iterations[key].append(
                np.mean(
                    [len(clustering_event) for clustering_event in results[key][theta]]
                )
            )

            avg_total_clustering_duration[key].append(
                np.mean(
                    [
                        np.sum(
                            [
                                iteration["iteration_duration"]
                                for iteration in results[key][theta][rep]
                            ]
                        )
                        + np.sum(
                            [
                                iteration["P_times"]
                                for iteration in results[key][theta][rep]
                                if iteration["P_times"] is not None
                            ]
                        )
                        + np.sum(
                            [
                                iteration["Q_times"]
                                for iteration in results[key][theta][rep]
                                if iteration["Q_times"] is not None
                            ]
                        )
                        for rep in range(len(results[key][theta]))
                    ]
                )
            )

    return (
        avg_single_iteration_duration,
        avg_total_iteration_duration,
        avg_number_of_iterations,
        avg_total_clustering_duration,
    )


def experiment_three(
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
    experiment_name,
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
        with open(f"results/{experiment_name}_{filename}.pkl", "wb") as f:
            pickle.dump(results, f)
        logger.info(f"Results saved to {filename}")

    # Postprocess results to get RSS differences for each theta
    rss_differences = postprocess_results_rss(results, thetas)
    # rss_differences is a DICTIONARY with keys as algorithm names and values as lists of RSS differences

    # Plot average RSS differences
    plot_exp_four_rss(
        thetas,
        rss_differences=rss_differences,  # pass the dictionary of rss differences
        filename=f"plots/{experiment_name}_rss_differences_{filename}.pdf",
        title="Exp3: RSS Difference between KMeans and EEKMeans vs Class Imbalance (Theta)",
    )
    logger.info("Average RSS differences plotted.")

    # Postprocess results to get average iteration durations, total iteration durations, and average number of iterations
    (
        average_single_iteration_durations,
        average_total_iteration_durations,
        average_number_of_iterations,
        average_total_clustering_duration,
    ) = postprocess_results_time(results, thetas)

    # Plot average iteration times
    plot_time(
        x_axis_name="Class Imbalance (Theta)",
        x_values=thetas,
        kmeans_times=average_single_iteration_durations["KMeans"],
        eekmeans_times={
            epsilon: average_single_iteration_durations[f"EEKMeans-ε={epsilon}"]
            for epsilon in epsilons
        },
        filename=f"plots/{experiment_name}_average_iteration_times_{filename}.pdf",
        title="Exp3: Average Single Iteration Time vs Class Imbalance (Theta)",
    )
    logger.info("Average iteration times plotted.")

    # Plot average total iteration times
    plot_time(
        x_axis_name="Class Imbalance (Theta)",
        x_values=thetas,
        kmeans_times=average_total_iteration_durations["KMeans"],
        eekmeans_times={
            epsilon: average_total_iteration_durations[f"EEKMeans-ε={epsilon}"]
            for epsilon in epsilons
        },
        filename=f"plots/{experiment_name}_average_total_iteration_time_{filename}.pdf",
        title="Exp3: Average Total Iteration Time vs Class Imbalance (Theta)",
    )
    logger.info("Average total iteration times plotted.")

    # Plot average total clustering durations
    plot_time(
        x_axis_name="Class Imbalance (Theta)",
        x_values=thetas,
        kmeans_times=average_total_clustering_duration["KMeans"],
        eekmeans_times={
            epsilon: average_total_clustering_duration[f"EEKMeans-ε={epsilon}"]
            for epsilon in epsilons
        },
        filename=f"plots/{experiment_name}_average_total_clustering_duration_{filename}.pdf",
        title="Exp3: Average Total Clustering Duration vs Class Imbalance (Theta)",
    )
    logger.info("Average total clustering durations plotted.")

    # Plot average number of iterations
    plot_iterations(
        x_axis_name="Class Imbalance (Theta)",
        x_values=thetas,
        iteration_number_kmeans=average_number_of_iterations["KMeans"],
        iteration_number_eekmeans={
            epsilon: average_number_of_iterations[f"EEKMeans-ε={epsilon}"]
            for epsilon in epsilons
        },
        filename=f"plots/{experiment_name}_average_number_of_iterations_{filename}.pdf",
        title="Exp3: Average Number of Iterations vs Class Imbalance (Theta)",
    )
    logger.info("Average number of iterations plotted.")


# Example usage
if __name__ == "__main__":
    """
    Experiment 3: Performance Analysis with Class Imbalance

    This experiment examines how the performance of KMeans and EEKMeans algorithms
    is affected by class imbalance in the dataset. The parameter theta controls
    the level of imbalance between classes (lower theta = higher imbalance).

    The experiment measures RSS differences between the algorithms across different
    theta values to understand which algorithm better handles skewed data distributions.
    """

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # Configure logging to both file and console
    log_filename = f"logs/EXP4_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Create a formatter
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Set up file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logger = logging.getLogger("EXP4")
    logger.info(f"Logging to file: {log_filename}")

    size_dataset = 60000  # Size of the dataset
    thetas = np.linspace(0.05, 1, 5, dtype=float)
    repetitions = 4
    n_clusters = 10
    max_iter = 40
    tol = 15
    epsilons = (200, 300, 400, 500)  # , 400, 500)  # (250, 450) --- IGNORE ---
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
    experiment_name = filename = os.path.splitext(os.path.basename(__file__))[0]

    experiment_three(
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
        filename=f"{param_str}",
        experiment_name=experiment_name,
        read="results/EXP3_dataset_mnist_k_10_maxiter_40_tol_15_eps_(200, 300, 400, 500)_delta_0.5_constenabled_False_samplebeginning_True_reps_4_t_20250728_032126.pkl",
    )
