import logging
import os

# import time
import pickle

import numpy as np

# from sklearn.datasets import fetch_openml
from epsiloneta_kmeans import KMeans, EEKMeans
from plots import plot_iteration_time
import datetime

# from sklearn.datasets import make_blobs

from utils import create_extended_dataset, sample_gaussian_mixture

os.environ["NUMEXPR_MAX_THREADS"] = "16"


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
        raise ValueError("Invalid algorithm. Choose 'kmeans' or 'eekmeans'.")

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


def postprocess_results(results):
    # Calculate average iteration durations for KMeans and EEKMeans
    avg_iteration_duration = {
        "KMeans": [],
        "EEKMeans": [],
    }
    avg_total_clustering_duration = {
        "KMeans": [],
        "EEKMeans": [],
    }
    avg_number_of_iterations = {
        "KMeans": [],
        "EEKMeans": [],
    }
    for key in results.keys():
        for n in sizes_datasets:

            avg_iteration_duration[key].append(
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

            avg_total_clustering_duration[key].append(
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

    return (
        avg_iteration_duration,
        avg_total_clustering_duration,
        avg_number_of_iterations,
    )


def experiment_two(
    logger,
    sizes_datasets,
    repetitions,
    n_clusters,
    max_iter,
    dataset,
    tol,
    epsilon,
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
        results = {"KMeans": {}, "EEKMeans": {}}

        for n in sizes_datasets:
            logging.info(f"Running experiment for n={n} (dataset size: {n})")

            results["KMeans"][n] = []
            results["EEKMeans"][n] = []

            if dataset == "gaussian_mixture":
                X, _ = sample_gaussian_mixture(n=n, theta=0.1, d=1000, k=n_clusters)
            elif dataset == "mnist":
                X, _ = create_extended_dataset(n)

            for i in range(repetitions):
                logging.info(f"Repetition {i + 1} of {repetitions}")
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
                results["EEKMeans"][n].append(
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

            logging.info("*" * 50)

        # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename
        with open(filename, "wb") as f:
            pickle.dump(results, f)
        logging.info(f"Results saved to {filename}")

    # Postprocess results to get average iteration durations, total iteration durations, and average number of iterations
    (
        average_iteration_durations,
        average_clustering_durations,
        average_number_of_iterations,
    ) = postprocess_results(results)

    # Plot average iteration times
    plot_iteration_time(
        sizes_datasets,
        kmeans_times=average_iteration_durations["KMeans"],
        eekmeans_times=average_iteration_durations["EEKMeans"],
        filename="average_iteration_times.pdf",
        title="Average Iteration Time vs Dataset Size",
    )

    # Plot average clustering times
    plot_iteration_time(
        sizes_datasets,
        kmeans_times=average_clustering_durations["KMeans"],
        eekmeans_times=average_clustering_durations["EEKMeans"],
        iteration_number_kmeans=average_number_of_iterations["KMeans"],
        iteration_number_eekmeans=average_number_of_iterations["EEKMeans"],
        filename="average_clustering_time.pdf",
        title="Average Clustering Time vs Dataset Size",
    )


# Example usage
if __name__ == "__main__":
    np.random.seed(36)  # Fix randomness globally
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger("test")

    sizes_datasets = np.linspace(50000, 60000, 2, dtype=int)
    repetitions = 2
    n_clusters = 10
    max_iter = 65
    tol = 12
    epsilon = 250
    delta = 0.5
    constant_enabled = False
    sample_beginning = True
    dataset = "gaussian_mixture"  # or "mnist"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    param_str = (
        f"n_{n_clusters}_maxiter_{max_iter}_tol_{tol}_eps_{epsilon}_delta_{delta}_"
        f"const_{constant_enabled}_samplebeg_{sample_beginning}_reps_{repetitions}_"
        f"sizes_{sizes_datasets[0]}-{sizes_datasets[-1]}_{timestamp}"
    )

    experiment_two(
        logger,
        sizes_datasets,
        repetitions,
        n_clusters,
        max_iter,
        dataset,
        tol,
        epsilon,
        delta,
        constant_enabled,
        sample_beginning,
        filename=f"results_{param_str}.pkl",
        read=None,  # "results_20250722_121746.pkl",
    )
