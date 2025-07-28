import logging
import pickle
import datetime
import os
import sys

import numpy as np

from utils import sample_gaussian_mixture, stepwise_kmeans, create_mnist_dataset
from plots import plot_movements


def postprocess_results(results):
    """
    Post-process the clustering results to extract relevant metrics.
    """
    averaged_results = {}
    for key, repetitions_data in results.items():
        max_iterations = max(len(rep) for rep in repetitions_data.values())
        avg_movements = np.zeros(max_iterations)
        avg_iterations = 0

        for rep_data in repetitions_data.values():
            avg_iterations += len(rep_data)
            for i, iteration_data in enumerate(rep_data):
                avg_movements[i] += iteration_data["movement"]

        avg_movements /= len(repetitions_data)
        avg_iterations /= len(repetitions_data)

        averaged_results[key] = {
            "avg_movements": avg_movements.tolist(),
            "avg_iterations": avg_iterations,
        }

    # Log the averaged results
    for key, data in averaged_results.items():
        logging.info(f"{key}: Avg Movements = {data['avg_movements']}")
        logging.info(f"{key}: Avg Iterations = {data['avg_iterations']}")

    return averaged_results


def experiment_one(
    logger,
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
    if read is not None:
        with open(read, "rb") as f:
            results = pickle.load(f)
    else:
        # Initialize results dictionary
        results = {"KMeans": {}}
        for epsilon in epsilons:
            results[f"EEKMeans-ε={epsilon}"] = {}

        # Ensure X is initialized based on the dataset
        if dataset == "gaussian_mixture":
            X, _ = sample_gaussian_mixture(n=80000, d=900, k=n_clusters)
        elif dataset == "mnist":
            X, _ = create_mnist_dataset()
        else:
            raise ValueError("Invalid dataset. Choose 'gaussian_mixture' or 'mnist'.")

        # TODO skew the dataset if needed
        # X, Y = skew_dataset(X, Y, n, n_clusters)
        # n = X.shape[0]

        for rep in range(repetitions):
            logger.info(f"Rep. {rep + 1} of {repetitions} with KMeans")
            # # KMeans
            results["KMeans"][rep] = stepwise_kmeans(
                X,
                n_clusters=n_clusters,
                max_iter=max_iter,
                tol=tol,
                algorithm="KMeans",
                logger=logger,
                seed_for_centroids=rep,
            )

            # EEKMeans
            for epsilon in epsilons:
                logger.info(
                    f"Rep. {rep + 1} of {repetitions} with EEKMeans-ε={epsilon}"
                )
                results[f"EEKMeans-ε={epsilon}"][rep] = stepwise_kmeans(
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
                    seed_for_centroids=rep,
                )

        logger.info("*" * 50)

        # Save the experiment results to a file
        with open(f"results/{experiment_name}_{filename}.pkl", "wb") as f:
            pickle.dump(results, f)

    # Post-process results
    averaged_results = postprocess_results(results)

    plot_movements(averaged_results, f"plots/{experiment_name}_{filename}.pdf")


# Example usage
if __name__ == "__main__":
    """
    Experiment 1: Movement of Centroids Analysis

    This experiment analyzes how the movement of centroids evolves over iterations
    for both KMeans and EEKMeans algorithms. It runs multiple repetitions on fixed-size
    datasets and tracks the average movement per iteration to understand convergence behavior.
    """

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # Configure logging to both file and console
    log_filename = f"logs/EXP1_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Create a formatter
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Set up file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logger = logging.getLogger("EXP1")
    logger.info(f"Logging to file: {log_filename}")

    repetitions = 10
    n_clusters = 10
    max_iter = 30
    tol = 0
    epsilons = (200, 300, 400, 500)
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

    experiment_one(
        logger,
        repetitions=repetitions,
        n_clusters=n_clusters,
        max_iter=max_iter,
        dataset=dataset,
        tol=tol,
        epsilons=epsilons,
        delta=delta,
        constant_enabled=constant_enabled,
        sample_beginning=sample_beginning,
        filename=f"{param_str}",
        experiment_name=experiment_name,
        read="results/EXP1_dataset_mnist_k_10_maxiter_30_tol_0_eps_(200, 300, 400, 500)_delta_0.5_constenabled_False_samplebeginning_True_reps_10_t_20250727_201120.pkl",
        # read="experiment_one_results_dataset_gaussian_mixture_k_10_maxiter_28_tol_0_eps_(200, 500)_delta_0.5_constenabled_False_samplebeginning_True_reps_3_t_20250726_142753.pkl",
    )
