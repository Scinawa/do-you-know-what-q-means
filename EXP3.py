import logging
import os
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from pprint import pformat

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


_CLUSTERING_WORKER_CONFIG = {}


def _init_clustering_worker(
    X,
    n_clusters,
    max_iter,
    tol,
    delta,
    constant_enabled,
    sample_beginning,
    logger_name,
):
    global _CLUSTERING_WORKER_CONFIG
    _CLUSTERING_WORKER_CONFIG = {
        "X": X,
        "n_clusters": n_clusters,
        "max_iter": max_iter,
        "tol": tol,
        "delta": delta,
        "constant_enabled": constant_enabled,
        "sample_beginning": sample_beginning,
        "logger_name": logger_name,
    }


def _run_kmeans(rep_index):
    if not _CLUSTERING_WORKER_CONFIG:
        raise RuntimeError("Clustering worker has not been initialized.")

    logger = logging.getLogger(_CLUSTERING_WORKER_CONFIG["logger_name"])
    trace = stepwise_kmeans(
        _CLUSTERING_WORKER_CONFIG["X"],
        n_clusters=_CLUSTERING_WORKER_CONFIG["n_clusters"],
        max_iter=_CLUSTERING_WORKER_CONFIG["max_iter"],
        tol=_CLUSTERING_WORKER_CONFIG["tol"],
        algorithm="KMeans",
        logger=logger,
        seed_for_centroids=rep_index,
    )
    return rep_index, "KMeans", trace


def _run_eekmeans_for_epsilon(rep_index, epsilon):
    if not _CLUSTERING_WORKER_CONFIG:
        raise RuntimeError("Clustering worker has not been initialized.")

    result_key = f"EEKMeans-ε={epsilon}"
    logger = logging.getLogger(_CLUSTERING_WORKER_CONFIG["logger_name"])
    trace = stepwise_kmeans(
        _CLUSTERING_WORKER_CONFIG["X"],
        n_clusters=_CLUSTERING_WORKER_CONFIG["n_clusters"],
        max_iter=_CLUSTERING_WORKER_CONFIG["max_iter"],
        tol=_CLUSTERING_WORKER_CONFIG["tol"],
        algorithm="EEKMeans",
        logger=logger,
        epsilon=epsilon,
        delta=_CLUSTERING_WORKER_CONFIG["delta"],
        constant_enabled=_CLUSTERING_WORKER_CONFIG["constant_enabled"],
        sample_beginning=_CLUSTERING_WORKER_CONFIG["sample_beginning"],
        seed_for_centroids=rep_index,
    )
    return rep_index, result_key, trace


def _validate_results_for_theta(results, theta, repetitions):
    for key, theta_results in results.items():
        if theta not in theta_results:
            raise ValueError(f"Missing theta={theta} results for {key}.")

        traces = theta_results[theta]
        if len(traces) != repetitions:
            raise ValueError(
                f"{key} theta={theta} has {len(traces)} repetitions; "
                f"expected {repetitions}."
            )

        missing_repetitions = [
            rep_index for rep_index, trace in enumerate(traces) if trace is None
        ]
        if missing_repetitions:
            raise ValueError(
                f"{key} theta={theta} is missing repetitions {missing_repetitions}."
            )


def postprocess_results_rss(results, thetas, size_dataset):
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
            rss_differences[key].append(
                ((np.mean(algorithm_rss) - np.mean(kmeans_rss)) / np.mean(kmeans_rss))
                * 100
            )

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
        if len(epsilons) == 0:
            raise ValueError("epsilons must contain at least one value.")

        for theta in thetas:
            logger.info(f"Running experiment for theta={theta}")

            for key in results.keys():
                results[key][theta] = [None] * repetitions

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

            with ProcessPoolExecutor(
                max_workers=len(epsilons) + 1,
                initializer=_init_clustering_worker,
                initargs=(
                    X,
                    n_clusters,
                    max_iter,
                    tol,
                    delta,
                    constant_enabled,
                    sample_beginning,
                    f"{logger.name}.worker",
                ),
            ) as executor:
                for i in range(repetitions):
                    logger.info(f"Repetition {i + 1} of {repetitions}")

                    futures = [executor.submit(_run_kmeans, i)]
                    logger.info(f"Rep. {i + 1} of {repetitions} with KMeans")

                    # EEKMeans
                    for epsilon in epsilons:
                        logger.info(
                            f"Rep. {i + 1} of {repetitions} with EEKMeans-ε={epsilon}"
                        )

                        futures.append(
                            executor.submit(_run_eekmeans_for_epsilon, i, epsilon)
                        )

                    for future in as_completed(futures):
                        rep_index, result_key, trace = future.result()
                        if result_key not in results:
                            raise KeyError(
                                f"Unexpected clustering result key: {result_key}"
                            )
                        if results[result_key][theta][rep_index] is not None:
                            raise ValueError(
                                f"Duplicate result for {result_key}, theta={theta}, "
                                f"repetition={rep_index}."
                            )
                        results[result_key][theta][rep_index] = trace

                    logger.info("*" * 50)

            _validate_results_for_theta(results, theta, repetitions)

        # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename
        with open(f"results/{experiment_name}_{filename}.pkl", "wb") as f:
            pickle.dump(results, f)
        logger.info(f"Results saved to {filename}")

    # Postprocess results to get RSS differences for each theta
    rss_differences = postprocess_results_rss(results, thetas, size_dataset)
    # rss_differences is a DICTIONARY with keys as algorithm names and values as lists of RSS differences

    # Plot average RSS differences
    plot_exp_four_rss(
        thetas,
        rss_differences=rss_differences,  # pass the dictionary of rss differences
        filename=f"plots/{experiment_name}_rss_differences_{filename}.pdf",
        title="Exp3: RSS Difference between EEKMeans and KMeans vs Class Imbalance (Theta)",
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

    formatted_rss = {
        key: [round(float(val), 2) for val in values]
        for key, values in rss_differences.items()
    }

    print("Value of theta: " + pformat(thetas))
    print(f"RSS Differences:\n {pformat(formatted_rss)}")

    print(
        "Average number of iterations:\n "
        + pformat(
            {
                key: [round(float(val), 2) for val in values]
                for key, values in average_number_of_iterations.items()
            }
        )
    )
    print(
        "Average single iteration durations:\n "
        + pformat(
            {
                key: [round(float(val), 2) for val in values]
                for key, values in average_single_iteration_durations.items()
            }
        )
    )
    print(
        "Average total iteration durations:\n "
        + pformat(
            {
                key: [round(float(val), 2) for val in values]
                for key, values in average_total_iteration_durations.items()
            }
        )
    )
    print(
        "Average total clustering duration:\n "
        + pformat(
            {
                key: [round(float(val), 2) for val in values]
                for key, values in average_total_clustering_duration.items()
            }
        )
    )


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

    size_dataset = 170000  # Size of the dataset
    # thetas = np.linspace(0.01, 1, 5, dtype=float)
    thetas = np.linspace(0.01, 1, 7, dtype=float)  # Adjusted to avoid zero samples
    repetitions = 10
    n_clusters = 10
    max_iter = 50
    tol = 38.25
    epsilons = (127.5, 255.0, 382.5, 510.0, 637.5)
    delta = 0.01
    constant_enabled = False
    sample_beginning = True
    dataset = "mnist"  # "gaussian_mixture" or "mnist"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    param_str = (
        f"dataset_{dataset}_k_{n_clusters}_maxiter_{max_iter}_tol_{tol}_eps_{epsilons}_delta_{delta}_"
        f"constenabled_{constant_enabled}_samplebeginning_{sample_beginning}_reps_{repetitions}_"
        f"thetas_{thetas}_t_{timestamp}"
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
        read=None,
        # read="results/EXP3_dataset_mnist_k_10_maxiter_70_tol_12_eps_(200, 300)_delta_0.5_constenabled_False_samplebeginning_True_reps_5_thetas_[0.01 0.34 0.67 1.  ]_t_20250729_183101.pkl",
    )
