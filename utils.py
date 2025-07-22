import numpy as np


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
