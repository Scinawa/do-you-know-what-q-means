import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


import matplotlib.pyplot as plt
from matplotlib import rc
import datetime

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import fetch_openml
from epsiloneta_kmeans import KMeans, EEKMeans
import logging
import time
from utils import create_dataset, residual_sum_of_squares, calculate_centroid_error
from utils import create_extended_dataset, sample_gaussian_mixture
from plots import plot_experiment_one


import pickle


def stepwise_kmeans(
    X,
    n_clusters,
    max_iter,
    tol,
    algorithm,
    logger,
    seed_for_centroids=None,
    epsilon=None,
    delta=None,
    sample_beginning=None,
    seed_for_samples=None,
    constant_enabled=None,
):
    logging.info(f"Starting {algorithm} clustering with {n_clusters} clusters")

    if algorithm == "KMeans":
        clustering = KMeans(
            X=X,
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            random_seed_centroids=seed_for_centroids,
            logger=logger,
        )
    elif algorithm == "EEKMeans":
        clustering = EEKMeans(
            X=X,
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            random_seed_centroids=seed_for_centroids,
            logger=logger,
            # Parameters for EEKMeans
            sample_beginning=sample_beginning,
            seed_for_samples=seed_for_samples,
            constant_enabled=constant_enabled,
            epsilon=epsilon,
            delta=delta,
        )
    else:
        raise ValueError("Invalid algorithm. Choose 'KMeans' or 'EEKMeans'.")

    clustering.fit(X)

    clustering_traceback = []
    for i in range(clustering.n_iter_):
        _tmp = {
            "iteration": i,
            "centroids": clustering.centroids_per_iteration[i],
            "movement": clustering.movements_per_iteration[i],
            "iteration_duration": clustering.duration_of_iteration[i],
            "P_times": (
                clustering.P_times[i]
                if hasattr(clustering, "P_times") and i < len(clustering.P_times)
                else None
            ),
            "Q_times": (
                clustering.Q_times[i]
                if hasattr(clustering, "Q_times") and i < len(clustering.Q_times)
                else None
            ),
        }
        clustering_traceback.append(_tmp)

        ##### DEBUGGING GIMMIK
        tmp_no_centroids = {k: v for k, v in _tmp.items() if k != "centroids"}
        items = list(tmp_no_centroids.items())
        for j in range(0, len(items), 5):
            msg = " | ".join(
                [
                    (
                        f"{k}: {v:.4f}"
                        if isinstance(v, (float)) and v is not None
                        else f"{k}: {v}"
                    )
                    for k, v in items[j : j + 5]
                ]
            )
            logger.debug(msg)
        ##### DEBUGGING GIMMIK

    return clustering_traceback


def plot_errors(
    kmeans_iterations,
    kmeans_movements,
    eekmeans_iterations,
    eekmeans_movements,
    filename="MSE_changes_comparison.pdf",
):
    plt.figure(figsize=(10, 6))
    plt.plot(kmeans_iterations, kmeans_movements, "b-o", label="KMeans")
    plt.plot(eekmeans_iterations, eekmeans_movements, "r-o", label="EEKMeans")
    plt.xlabel("Iteration")
    plt.ylabel("Pure units")
    plt.title("Clustering MSE and changes in centroids by iteration")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.savefig(filename)


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
    read=None,
):
    if read is not None:
        with open(read, "rb") as f:
            results = pickle.load(f)
    else:
        results = {"KMeans": {}}
        for epsilon in epsilons:
            results[f"EEKMeans-ε={epsilon}"] = {}

        if dataset == "gaussian_mixture":
            X, _ = sample_gaussian_mixture(n=60000, k=n_clusters)
        elif dataset == "mnist":
            X, _ = create_dataset()

        # TODO skew the dataset if needed
        # X, Y = skew_dataset(X, Y, n, n_clusters)
        # n = X.shape[0]

        for rep in range(repetitions):
            logging.info(f"Repetition {rep + 1} of {repetitions}")
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
                logging.info(f"Running EEKMeans with epsilon={epsilon}")
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

        logging.info("*" * 50)

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

    # Save the averaged results to a file
    with open(filename, "wb") as f:
        pickle.dump(averaged_results, f)

    filename_plot = filename.replace(".pkl", f"_{timestamp}.pdf")

    plot_experiment_one(averaged_results, filename_plot)


# Example usage
if __name__ == "__main__":
    np.random.seed(36)  # Fix randomness globally
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger("test")

    # sizes_datasets = np.linspace(50000, 60000, 2, dtype=int)
    repetitions = 6
    n_clusters = 10
    max_iter = 28
    tol = 0
    epsilons = (200, 300, 400, 500)
    delta = 0.5
    constant_enabled = False
    sample_beginning = True
    dataset = "mnist"  # or "mnist"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    param_str = (
        f"dataset_{dataset}_k_{n_clusters}_maxiter_{max_iter}_tol_{tol}_eps_{epsilons}_delta_{delta}_"
        f"constenabled_{constant_enabled}_samplebeginning_{sample_beginning}_reps_{repetitions}_"
        f"t_{timestamp}"
    )

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
        filename=f"experiment_one_results_{param_str}.pkl",
        # read="experiment_one_results_n_10_maxiter_65_tol_35_eps_(200, 250, 300)_delta_0.5_const_False_samplebeg_True_reps_2_t_20250725_221953.pkl",
    )
