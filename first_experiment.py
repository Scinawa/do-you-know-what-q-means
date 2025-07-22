import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


import matplotlib.pyplot as plt
from matplotlib import rc


import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import fetch_openml
from epsiloneta_kmeans import KMeans, EEKMeans
import logging
import time
from utils import residual_sum_of_squares, calculate_centroid_error
import pickle


def create_dataset(percentile=100):
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X_, Y_ = mnist["data"], mnist["target"]
    norms_ = np.linalg.norm(X_, axis=1)
    percentile = 95  # percentile to filter norms
    threshold = np.percentile(norms_, percentile)
    mask = norms_ <= threshold
    X = X_[mask]
    Y = Y_[mask]
    return X, Y


def create_extended_dataset(n):
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X, Y_ = mnist["data"], mnist["target"]

    # Extend X and Y to length n by randomly sampling from X and Y (with replacement)
    if len(X) < n:
        num_to_add = n - len(X)
        indices = np.random.choice(len(X), size=num_to_add, replace=True)
        X_tended = np.vstack([X, X[indices]])
        Y_tended = np.concatenate([Y_, Y_[indices]])
    else:
        X_tended = X[:n]
        Y_tended = Y_[:n]

    return X_tended, Y_tended


def stepwise_kmeans(
    X,
    n_clusters,
    max_iter,
    tol,
    random_state=36,
    algorithm="kmeans",
    logger=None,
    epsilon=1,
    delta=0.5,
    constant_enabled=False,
    sample_beginning=True,
    seed_for_samples=None,
):
    """
    Perform stepwise KMeans or EEKMeans clustering, storing centroids and errors at each iteration.

    Parameters
    ----------
    X : np.ndarray
        Input data for clustering.
    n_clusters : int
        Number of clusters.
    max_iter : int, default=10
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    random_state : int, default=42
        Random seed for reproducibility.
    algorithm : str, default="kmeans"
        Algorithm to use ("kmeans" or "eekmeans").
    logger : logging.Logger, default=None
        Logger for output messages.
    epsilon : float, default=1
        Approximation parameter for EEKMeans.
    delta : float, default=0.5
        Failure probability parameter for EEKMeans.
    constant_enabled : bool, default=False
        Whether to use theoretical constants for sampling sizes.
    sample_beginning : bool, default=True
        Whether to sample P and Q at initialization (EEKMeans only).

    Returns
    -------
    list of dict
        A list containing centroids and error at each iteration.
    """
    np.random.seed(random_state)

    if algorithm == "kmeans":
        clustering = KMeans(
            X=X,
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            random_seed_centroids=random_state,
            logger=logger,
        )
    elif algorithm == "eekmeans":
        clustering = EEKMeans(
            X=X,
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            random_seed_centroids=random_state,
            seed_for_samples=seed_for_samples,
            logger=logger,
            constant_enabled=constant_enabled,
            epsilon=epsilon,
            delta=delta,
            sample_beginning=sample_beginning,
        )
    else:
        raise ValueError("Invalid algorithm. Choose 'kmeans' or 'eekmeans'.")

    logging.info(f"Starting {algorithm} clustering with {n_clusters} clusters")
    clustering.fit(X)

    results = []
    for i, (c, e) in enumerate(zip(clustering.list_centroids, clustering.movements)):
        result = {
            "iteration": i,
            "centroids": c,
            "movement": e,
            "iteration_duration": clustering.iteration_duration[i],
        }
        if algorithm == "eekmeans" and i == 0:
            result["elapsed_P_init"] = getattr(clustering, "elapsed_P_init", None)
            result["elapsed_Q_init"] = getattr(clustering, "elapsed_Q_init", None)
        results.append(result)
        logging.info(
            f"Iteration {i + 1}: movement = {e:.4f}, duration = {clustering.iteration_duration[i]:.4f} seconds"
        )

    return results


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


def experiment_one(logger):
    X, _ = create_dataset(percentile=95)
    n_clusters = 10  # Number of clusters for MNIST digits
    max_iter = 30
    tol = 14
    epsilon = 250  # approximation parameter
    delta = 0.5  # failure probability
    constant_enabled = False  # Whether to use theoretical constants
    sample_beginning = True  # Whether to sample P and Q at initialization

    logging.info(
        f"Parameters: n_clusters={n_clusters}, max_iter={max_iter}, tol={tol}, "
        f"epsilon={epsilon}, delta={delta}, constant_enabled={constant_enabled}, "
        f"sample_beginning={sample_beginning}"
    )

    # Run both algorithms
    results_kmeans = stepwise_kmeans(
        X,
        n_clusters=n_clusters,
        max_iter=max_iter,
        tol=tol,
        algorithm="kmeans",
        logger=logger,
        constant_enabled=constant_enabled,
    )
    results_eekmeans = stepwise_kmeans(
        X,
        n_clusters=n_clusters,
        max_iter=max_iter,
        tol=tol,
        algorithm="eekmeans",
        logger=logger,
        epsilon=epsilon,
        delta=delta,
        constant_enabled=constant_enabled,
        sample_beginning=sample_beginning,
    )

    # Extract iteration numbers and errors
    kmeans_iterations = [r["iteration"] for r in results_kmeans]
    kmeans_movements = [r["movement"] for r in results_kmeans]
    eekmeans_iterations = [r["iteration"] for r in results_eekmeans]
    eekmeans_movements = [r["movement"] for r in results_eekmeans]

    # Set logging to INFO level to reduce debug output
    logging.getLogger().setLevel(logging.INFO)

    # Plot the movements
    plot_errors(
        kmeans_iterations, kmeans_movements, eekmeans_iterations, eekmeans_movements
    )

    # Calculate RSS for both models using their final centroids
    final_centroids_kmeans = results_kmeans[-1]["centroids"]
    final_centroids_eekmeans = results_eekmeans[-1]["centroids"]

    rss_kmeans = residual_sum_of_squares(X, final_centroids_kmeans) / X.shape[0]
    rss_eekmeans = residual_sum_of_squares(X, final_centroids_eekmeans) / X.shape[0]

    logging.info(f"KMeans RSS: {rss_kmeans:.4f}")
    logging.info(f"EEKMeans RSS: {rss_eekmeans:.4f}")
    logging.info(
        f"Total KMeans iterations duration: {np.average([r['iteration_duration'] for r in results_kmeans]):.4f} seconds"
    )
    logging.info(
        f"Total EEKMeans iterations duration: {np.average([r['iteration_duration'] for r in results_eekmeans]):.4f} seconds"
    )


def plot_comparison(
    ns,
    avg_kmeans_times,
    avg_eekmeans_times,
    numberkmeansiteration=None,
    numbereekmeansiterations=None,
    avg_P_init_times=None,
    avg_Q_init_times=None,
    filename="plot_comparison.pdf",
):
    plt.figure(figsize=(10, 6))
    plt.plot(ns, avg_kmeans_times, "b-o", label="KMeans iterations")
    plt.plot(ns, avg_eekmeans_times, "r-o", label="EEKMeans iterations")
    if avg_P_init_times is not None:
        plt.plot(ns, avg_P_init_times, "g--s", label="EEKMeans P init")
    if avg_Q_init_times is not None:
        plt.plot(ns, avg_Q_init_times, "m--^", label="EEKMeans Q init")
    if numberkmeansiteration is not None:
        for x, y, num_iter in zip(ns, avg_kmeans_times, numberkmeansiteration):
            plt.annotate(
                f"{num_iter}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                color="blue",
            )
    if numbereekmeansiterations is not None:
        for x, y, num_iter in zip(ns, avg_eekmeans_times, numbereekmeansiterations):
            plt.annotate(
                f"{num_iter}",
                (x, y),
                textcoords="offset points",
                xytext=(0, -15),
                ha="center",
                color="red",
            )
    plt.xlabel("Dataset size (n)")
    plt.ylabel("Time (seconds)")
    plt.title("Algorithm runtime vs dataset size")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.savefig(filename)


def experiment_two(logger, read=None):
    if read is not None:
        logging.info(f"Reading results from pickle file: {read}")
        with open(read, "rb") as f:
            results_dict = pickle.load(f)
        import pdb

        pdb.set_trace()
        ns = results_dict["ns"]
        sum_kmeans_times = results_dict["sum_kmeans_times"]
        sum_eekmeans_times = results_dict["sum_eekmeans_times"]
        kmeans_iterations = results_dict["kmeans_iterations"]
        eekmeans_iterations = results_dict["eekmeans_iterations"]
        avg_P_init_times = results_dict["avg_P_init_times"]
        avg_Q_init_times = results_dict["avg_Q_init_times"]

        plot_comparison(
            ns,
            sum_kmeans_times,
            sum_eekmeans_times,
            numberkmeansiteration=kmeans_iterations,
            numbereekmeansiterations=eekmeans_iterations,
            avg_P_init_times=avg_P_init_times,
            avg_Q_init_times=avg_Q_init_times,
        )
        return

    ns = np.linspace(30000, 350000, 12, dtype=int)
    repetitions = 10  # Number of repetitions for EEKMeans
    sum_kmeans_times = []
    sum_eekmeans_times = []
    avg_P_init_times = []
    avg_Q_init_times = []
    kmeans_iterations = []
    eekmeans_iterations = []
    results_eekmeans = []

    n_clusters = 10
    max_iter = 65
    tol = 15
    epsilon = 250
    delta = 0.5
    constant_enabled = False
    sample_beginning = True

    for n in ns:
        logging.info(f"Running experiment for n={n} (dataset size: {n})")
        X, _ = create_extended_dataset(n)
        for i in range(repetitions):
            # KMeans
            results_kmeans = stepwise_kmeans(
                X,
                n_clusters=n_clusters,
                max_iter=max_iter,
                tol=tol,
                algorithm="kmeans",
                logger=logger,
                constant_enabled=constant_enabled,
            )
            # EEKMeans

            logging.info(f"Running EEKMeans repetition {i + 1} for n={n}")
            results_eekmeans.append(
                stepwise_kmeans(
                    X,
                    n_clusters=n_clusters,
                    max_iter=max_iter,
                    tol=tol,
                    algorithm="eekmeans",
                    logger=logger,
                    epsilon=epsilon,
                    delta=delta,
                    constant_enabled=constant_enabled,
                    sample_beginning=sample_beginning,
                    seed_for_samples=i,
                )
            )

        # Extract iteration durations from results
        iteration_duration_kmeans = [r["iteration_duration"] for r in results_kmeans]
        iteration_duration_eekmeans = [
            r["iteration_duration"] for r in results_eekmeans
        ]

        sum_kmeans_times.append(np.sum(iteration_duration_kmeans))
        sum_eekmeans_times.append(np.sum(iteration_duration_eekmeans))

        kmeans_iterations.append(len(iteration_duration_kmeans))
        eekmeans_iterations.append(len(iteration_duration_eekmeans))

        # Extract P and Q init times from EEKMeans results (first iteration only)
        elapsed_P_init = results_eekmeans[0].get("elapsed_P_init", None)
        elapsed_Q_init = results_eekmeans[0].get("elapsed_Q_init", None)
        avg_P_init_times.append(elapsed_P_init)
        avg_Q_init_times.append(elapsed_Q_init)
        logging.info(
            f"n={n}: KMeans iterations={len(iteration_duration_kmeans)}, EEKMeans iterations={len(iteration_duration_eekmeans)}"
        )
        logging.info(
            f"n={n}: KMeans avg iter {sum_kmeans_times[-1]:.4f}s, EEKMeans avg iter {sum_eekmeans_times[-1]:.4f}s"
        )
        logging.info(
            f"n={n}: EEKMeans P init {elapsed_P_init:.4f}s, Q init {elapsed_Q_init:.4f}s"
        )
        print("*" * 50)

    # Prepare data to pickle
    results_dict = {
        "ns": ns,
        "sum_kmeans_times": sum_kmeans_times,
        "sum_eekmeans_times": sum_eekmeans_times,
        "kmeans_iterations": kmeans_iterations,
        "eekmeans_iterations": eekmeans_iterations,
        "avg_P_init_times": avg_P_init_times,
        "avg_Q_init_times": avg_Q_init_times,
    }
    with open("experiment_two_results.pkl", "wb") as f:
        pickle.dump(results_dict, f)

    plot_comparison(
        ns,
        sum_kmeans_times,
        sum_eekmeans_times,
        numberkmeansiteration=kmeans_iterations,
        numbereekmeansiterations=eekmeans_iterations,
        avg_P_init_times=avg_P_init_times,
        avg_Q_init_times=avg_Q_init_times,
    )


# Example usage
if __name__ == "__main__":
    np.random.seed(36)  # Fix randomness globally
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger("test")

    # experiment_one(logger)

    experiment_two(logger)  # , read="experiment_two_results.pkl")
