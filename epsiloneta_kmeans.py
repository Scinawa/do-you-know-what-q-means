import numpy as np
import scipy as sp
from sklearn.datasets import fetch_openml
from scipy.stats import rv_discrete
import sys
import timeit
import logging


class EEKMeans:
    """
    Q-means clustering algorithm.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form.

    max_iter : int, default=50
        Maximum number of iterations for the algorithm.

    tol : float, default=1e-4
        Tolerance for convergence.

    random_state : int, default=None
        Random seed for reproducibility.

    logger : logging.Logger, default=None
        Logger for output messages. If None, a default logger is created.
    """

    def __init__(
        self,
        X,
        n_clusters=None,
        max_iter=50,
        epsilon=1,
        delta=0.5,
        tol=1e-4,
        random_state=None,
        constant_enabled=False,
        logger=None,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.X = X
        self.epsilon = epsilon
        self.delta = delta
        self.n = len(X)
        self.constant_enabled = constant_enabled

        # Set up logger
        if logger is None:
            self.logger = logging.getLogger("EEKMeans")
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.WARNING)
        else:
            self.logger = logger

        # Compute norms and distribution parameters
        self.V_norm = np.linalg.norm(self.X, ord=2)
        self.norms = np.linalg.norm(self.X, axis=1)
        self.V_norm_2_1 = np.sum(self.norms)

        self.p, self.q = self._compute_p_q()

        probabilities_ = self.norms / self.V_norm_2_1

        self.distrib_over_norms = rv_discrete(
            values=(np.arange(len(self.X)), probabilities_)
        )

        # self.V_norm_fro = np.linalg.norm(X, ord="fro")  # Frobenius norm of the matrix
        # tmp_ = (self.V_norm_2_1**2 / self.n**2) < (self.V_norm_fro**2 / self.n)
        # self.logger.debug(
        #     "Security check: %s, %f, %f",
        #     tmp_,
        #     (self.V_norm_2_1**2 / self.n**2),
        #     (np.linalg.norm(X, ord="fro") ** 2 / self.n),
        # )

    def _initialize_centroids(self, X):
        """Initialize centroids using the provided methods."""
        if self.n_clusters is not None:
            # Fall back to random initialization
            indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            return X[indices]

    def _compute_p_q(self):
        if self.constant_enabled:
            constant_factor_p = 48
            constant_factor_q = 64
        else:
            constant_factor_p = 1
            constant_factor_q = 1

        p = int(
            np.ceil(
                constant_factor_p
                * ((self.V_norm**2 / self.n) * (self.n_clusters**2 / self.epsilon**2))
                * np.log(self.n_clusters / self.delta)
            )
        )
        q = int(
            np.ceil(
                constant_factor_q
                * (
                    (self.V_norm_2_1**2 / self.n**2)
                    * (self.n_clusters**2 / self.epsilon**2)
                )
                * np.log(self.n_clusters / self.delta)
            )
        )
        return p, q

    def fit(self, X):
        """
        Fit the Q-means clustering model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,), optional
            Target labels for initialization.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n, d = X.shape
        k = self.n_clusters
        convergence = self.tol

        # Initialize centroids
        centroids = self._initialize_centroids(X)
        self.logger.debug(f"Initial centroids shape: {centroids.shape}")

        iteration = 0
        error = np.inf  # Initialize error to a large value
        self.logger.info("Beginning main loop of EEKMeans.fit()")

        while (error > convergence) and iteration < self.max_iter:
            # Measure time to sample P
            start_time_P = timeit.default_timer()
            P = np.random.choice(len(X), size=self.p, replace=True)
            elapsed_P = timeit.default_timer() - start_time_P
            self.logger.info(f"Time to sample P: {elapsed_P:.6f} seconds")

            # Measure time to sample Q
            start_time_Q = timeit.default_timer()
            Q = self.distrib_over_norms.rvs(size=self.q)
            elapsed_Q = timeit.default_timer() - start_time_Q
            self.logger.info(f"Time to sample Q: {elapsed_Q:.6f} seconds")

            # Step: Create Pⱼ and Qⱼ sets
            P_sets = [[] for _ in range(k)]
            Q_sets = [[] for _ in range(k)]

            # Combine P and Q for labeling step
            PQ = np.concatenate([P, Q])

            # Step: For i ∈ P ∪ Q, label ℓᵢᵗ = arg minⱼ∈[k] ||vᵢ - cⱼᵗ||
            labels = np.zeros(len(PQ), dtype=int)
            for idx, i in enumerate(PQ):
                distances = np.linalg.norm(X[i] - centroids, axis=1)
                labels[idx] = np.argmin(distances)

            for i, sample in enumerate(P):
                label = labels[i]
                P_sets[label].append(sample)

            for i, sample in enumerate(Q):
                label = labels[len(P) + i]
                Q_sets[label].append(sample)

            # check if P_sets and Q_sets are not empty
            self.logger.debug("P_sets lengths: %s", [len(P_sets[j]) for j in range(k)])
            self.logger.debug("Q_sets lengths: %s", [len(Q_sets[j]) for j in range(k)])

            # Step 7: Update centroids cⱼᵗ⁺¹ = (||V||₂,₁/n|Pⱼ|) * Σᵢ∈Qⱼ (vᵢ/q) * (||vᵢ||)
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                if len(P_sets[j]) > 0 and len(Q_sets[j]) > 0:
                    weighted_sum = np.zeros(X.shape[1])

                    Q_indices = np.array(Q_sets[j])
                    weighted_sum = (
                        X[Q_indices] / self.norms[Q_indices, np.newaxis]
                    ).sum(axis=0) * (self.V_norm_2_1 / self.q)

                    # Update the centroid for cluster j
                    coefficient = self.p / (n * len(P_sets[j]))
                    new_centroids[j] = coefficient * weighted_sum
                else:
                    self.logger.warning(
                        f"Empty P or Q set for cluster {j} (|P_j|={len(P_sets[j])}, |Q_j|={len(Q_sets[j])})"
                    )

            # Compute change in centroids for convergence check
            error = np.mean(np.linalg.norm(new_centroids - centroids, axis=1))
            centroids = new_centroids

            iteration += 1
            self.logger.info(f"Iteration {iteration}, Error: {error:.6f}")

        self.logger.info(f"Algorithm converged after {iteration} iterations")

        # Store results as attributes
        self.cluster_centers_ = centroids
        self.n_iter_ = iteration

        return self

    def predict(self, X):
        """
        Predict the closest cluster for each sample in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("Model not fitted yet. Call 'fit' before using 'predict'.")

        labels = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            distances = np.linalg.norm(X[i] - self.cluster_centers_, axis=1)
            labels[i] = np.argmin(distances)

        return labels
