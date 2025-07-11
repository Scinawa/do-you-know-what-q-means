import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


import matplotlib.pyplot as plt
from matplotlib import rc

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import fetch_openml
from epsiloneta_kmeans import KMeans, EEKMeans
import logging
import time


def residual_sum_of_squares(X, centroids):
    """
    Calculate the residual sum of squares (RSS) for the given data and centroids.

    Parameters
    ----------
    X : np.ndarray
        Input data points.

    centroids : np.ndarray
        Cluster centroids.

    Returns
    -------
    float
        Residual sum of squares.
    """
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    closest_centroid_distances = np.min(distances, axis=1)
    rss = np.sum(closest_centroid_distances**2)
    return rss


def calculate_centroid_error(centroids1, centroids2):
    """
    Calculate the total error between two sets of centroids and return the sorted centroids2.

    Parameters
    ----------
    centroids1 : list of np.ndarray
        Centroids from the first model.

    centroids2 : list of np.ndarray
        Centroids from the second model.

    Returns
    -------
    float
        Total error (sum of distances between matched centroids).
    np.ndarray
        Sorted centroids2 to correspond to centroids1.
    """

    centroids1 = np.array(centroids1)
    centroids2 = np.array(centroids2)

    # Compute the distance matrix between centroids
    distance_matrix = np.linalg.norm(centroids1[:, np.newaxis] - centroids2, axis=2)

    # Find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    total_error = distance_matrix[row_ind, col_ind].sum()
    sorted_centroids2 = centroids2[col_ind]

    return total_error, sorted_centroids2


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
    random_state=42,
    algorithm="kmeans",
    logger=None,
    epsilon=1,
    delta=0.5,
    constant_enabled=False,
    sample_beginning=True,
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
            random_state=random_state,
            logger=logger,
        )
    elif algorithm == "eekmeans":
        clustering = EEKMeans(
            X=X,
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            logger=logger,
            constant_enabled=constant_enabled,
            epsilon=epsilon,
            delta=delta,
            sample_beginning=sample_beginning,
        )
    else:
        raise ValueError("Invalid algorithm. Choose 'kmeans' or 'eekmeans'.")

    clustering.fit(X)

    # Use the lists from the clustering object - both classes use 'movements' now instead of 'list_errors'
    results = []
    for i, (c, e) in enumerate(zip(clustering.list_centroids, clustering.movements)):
        results.append({"iteration": i + 1, "centroids": c, "movement": e})
        logging.debug(f"Iteration {i + 1}: movement = {e:.4f}")

    return results, np.average(clustering.iteration_duration)


def experiment_one():
    X, _ = create_dataset(percentile=95)
    n_clusters = 10  # Number of clusters for MNIST digits
    max_iter = 15
    tol = 12
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
    results_kmeans, avg_kmeans = stepwise_kmeans(
        X,
        n_clusters=n_clusters,
        max_iter=max_iter,
        tol=tol,
        algorithm="kmeans",
        logger=logger,
        constant_enabled=constant_enabled,
    )
    results_eekmeans, avg_eekmeans = stepwise_kmeans(
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
    plt.figure(figsize=(10, 6))
    plt.plot(kmeans_iterations, kmeans_movements, "b-o", label="KMeans")
    plt.plot(eekmeans_iterations, eekmeans_movements, "r-o", label="EEKMeans")

    # Add labels and title
    plt.xlabel("Iteration")
    plt.ylabel("Pure units")
    plt.title("Clustering MSE and changes in centroids by iteration")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    # Save the plot
    plt.savefig("MSE_changes_comparison.pdf")

    # Calculate RSS for both models using their final centroids
    final_centroids_kmeans = results_kmeans[-1]["centroids"]
    final_centroids_eekmeans = results_eekmeans[-1]["centroids"]

    rss_kmeans = residual_sum_of_squares(X, final_centroids_kmeans) / X.shape[0]
    rss_eekmeans = residual_sum_of_squares(X, final_centroids_eekmeans) / X.shape[0]

    logging.info(f"KMeans RSS: {rss_kmeans:.4f}")
    logging.info(f"EEKMeans RSS: {rss_eekmeans:.4f}")
    logging.info(f"Average KMeans iteration duration: {avg_kmeans:.4f} seconds")
    logging.info(f"Average EEKMeans iteration duration: {avg_eekmeans:.4f} seconds")


def experiment_two():

    ns = np.linspace(30000, 300000, 10, dtype=int)
    avg_kmeans_times = []
    avg_eekmeans_times = []

    n_clusters = 10
    max_iter = 10
    tol = 12
    epsilon = 250
    delta = 0.5
    constant_enabled = False
    sample_beginning = True

    for n in ns:
        logging.info(f"Running experiment for n={n} (dataset size: {n})")
        X, _ = create_extended_dataset(n)
        # KMeans
        _, avg_kmeans = stepwise_kmeans(
            X,
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            algorithm="kmeans",
            logger=logger,
            constant_enabled=constant_enabled,
        )
        # EEKMeans
        _, avg_eekmeans = stepwise_kmeans(
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
        avg_kmeans_times.append(avg_kmeans)
        avg_eekmeans_times.append(avg_eekmeans)
        logging.info(
            f"n={n}: KMeans avg iter {avg_kmeans:.4f}s, EEKMeans avg iter {avg_eekmeans:.4f}s"
        )

    plt.figure(figsize=(10, 6))
    plt.plot(ns, avg_kmeans_times, "b-o", label="KMeans")
    plt.plot(ns, avg_eekmeans_times, "r-o", label="EEKMeans")
    plt.xlabel("Dataset size (n)")
    plt.ylabel("Average iteration duration (seconds)")
    plt.title("Average iteration duration vs dataset size")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.savefig("avg_iteration_duration_vs_n.pdf")


# Example usage
if __name__ == "__main__":
    np.random.seed(42)  # Fix randomness globally
    # DEBUG
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger("test")

    # experiment_one()

    experiment_two()
