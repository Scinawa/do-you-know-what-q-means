import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import make_blobs
from sklearn.datasets import fetch_openml


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


def make_dataset_skewed(X, Y, theta=0.0002):
    """
    Skew the dataset so that only a theta fraction of the last class remains,
    and the rest of the dataset is filled with samples from the other classes (uniformly).
    Returns:
        X_combined: np.ndarray of shape (n, d)
        Y_combined: np.ndarray of shape (n,)
    """
    n = X.shape[0]
    k = len(np.unique(Y))

    # Get samples from the last class
    X_last_class_ = X[Y == (k - 1)]
    skewed_last_class = int(X_last_class_.shape[0] * theta)

    X_last_class = X_last_class_[:skewed_last_class]
    Y_last_class = np.full(X_last_class.shape[0], k - 1, dtype=Y.dtype)

    # Get samples from other classes
    X_other_classes = X[Y != (k - 1)]
    Y_other_classes = Y[Y != (k - 1)]
    new_sample_per_class = int(np.ceil((n - skewed_last_class) / (k - 1)))

    new_samples = []
    new_labels = []
    for label in range(k - 1):

        class_samples = X_other_classes[Y_other_classes == label]
        indices = np.random.choice(
            len(class_samples), size=new_sample_per_class, replace=True
        )
        new_samples.append(class_samples[indices])
        new_labels.append(np.full(new_sample_per_class, label, dtype=Y.dtype))

    X_ = np.vstack(new_samples)
    Y_ = np.concatenate(new_labels)

    # Combine with last class
    X_combined = np.vstack([X_, X_last_class])
    Y_combined = np.concatenate([Y_, Y_last_class])

    # Shuffle the combined dataset
    indices = np.random.permutation(len(Y_combined))
    X_combined = X_combined[indices]
    Y_combined = Y_combined[indices]

    # Ensure the final dataset has the same number of samples as the original
    X_combined = X_combined[:n]
    Y_combined = Y_combined[:n]
    print("size of the matrices after skewing:", X_combined.shape, Y_combined.shape)

    return X_combined, Y_combined


def sample_gaussian_mixture(n, d=50, k=10):
    """
    Generate a dataset X of n points in d dimensions from k Gaussians using sklearn's make_blobs.
    The theta parameter is ignored in this implementation.
    Returns:
        X: np.ndarray of shape (n, d)
        labels: np.ndarray of shape (n,) with cluster labels
    """
    X, labels = make_blobs(
        n_samples=n, n_features=d, centers=k, cluster_std=2.5, random_state=None
    )
    # Recreate the dataset X by taking only \theta fraction of the points of the last class.
    # Add uniformly points to the other classes to reach n points in total.

    return X, labels


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
