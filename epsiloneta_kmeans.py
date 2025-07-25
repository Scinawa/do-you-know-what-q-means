import timeit
import logging

import numpy as np
from scipy.stats import rv_discrete


class BaseKMeans:
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
        random_seed_centroids=None,
        constant_enabled=False,
        logger=None,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_seed_centroids = random_seed_centroids
        self.X = X
        self.epsilon = epsilon
        self.delta = delta
        self.n = len(X)
        self.constant_enabled = constant_enabled

        # Set up logger
        if logger is None:
            self.logger = logging.getLogger("KMeans")
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

        self.movements_per_iteration = []
        self.centroids_per_iteration = []
        self.duration_of_iteration = []

    def _initialize_centroids(self, X):
        """Initialize centroids using the provided methods."""
        if self.random_seed_centroids is not None:
            np.random.seed(self.random_seed_centroids)  # Fix randomness
        if self.n_clusters is not None:
            indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            return X[indices]

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


class EEKMeans(BaseKMeans):
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

    sample_beginning : bool, default=True
        If True, sample P and Q only once at initialization. If False, sample at each iteration.
    """

    def __init__(
        self,
        X,
        n_clusters=None,
        max_iter=50,
        epsilon=1,
        delta=0.5,
        tol=1e-4,
        random_seed_centroids=None,
        seed_for_samples=None,
        constant_enabled=False,
        logger=None,
        sample_beginning=True,
    ):
        super().__init__(
            X=X,
            n_clusters=n_clusters,
            max_iter=max_iter,
            epsilon=epsilon,
            delta=delta,
            tol=tol,
            random_seed_centroids=random_seed_centroids,
            constant_enabled=constant_enabled,
            logger=logger,
        )

        # Compute norms and distribution parameters
        self.V_norm = np.linalg.norm(self.X, ord=2)
        self.norms = np.linalg.norm(self.X, axis=1)
        self.V_norm_2_1 = np.sum(self.norms)
        self.sample_beginning = sample_beginning
        self.seed_for_samples = seed_for_samples

        self.P_times = []
        self.Q_times = []

        self.p, self.q = self._compute_p_q()
        self.logger.debug(f"Sample sizes: p={self.p}, q={self.q}")

        # If sample_beginning is True, sample P and Q once during initialization
        if self.sample_beginning:
            self.P, self.Q = self._sample_P_Q()

    def _sample_P_Q(self):

        probabilities_ = self.norms / self.V_norm_2_1

        self.distrib_over_norms = rv_discrete(
            values=(np.arange(len(self.X)), probabilities_)
        )

        np.random.seed(self.seed_for_samples)

        start_time_P_init = timeit.default_timer()
        P = np.random.choice(len(self.X), size=self.p, replace=True)
        P_smp_time = timeit.default_timer() - start_time_P_init
        self.P_times.append(P_smp_time)

        start_time_Q_init = timeit.default_timer()
        Q = self.distrib_over_norms.rvs(size=self.q)
        Q_smp_time = timeit.default_timer() - start_time_Q_init
        self.Q_times.append(Q_smp_time)

        self.logger.debug(
            f"Sampling Q: {Q_smp_time:.6f}s, Sampling P: {P_smp_time:.6f}s"
        )

        return P, Q

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

        k = self.n_clusters

        # Initialize centroids
        centroids = self._initialize_centroids(X)

        iteration = 0
        error = np.inf  # Initialize error to a large value
        self.logger.debug("Beginning main loop of EEKMeans.fit()")

        while (error > self.tol) and iteration < self.max_iter:
            start_time_iteration = timeit.default_timer()  # Start timing the iteration

            # Sample P and Q if not sampled at initialization
            if not self.sample_beginning:
                P, Q = self._sample_P_Q()
                self.logger.debug(f"Time to sample P: {self.P_times[-1]:.6f} seconds")
                self.logger.debug(f"Time to sample Q: {self.Q_times[-1]:.6f} seconds")
            else:
                # Use pre-sampled P and Q
                P = self.P
                Q = self.Q

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

            # Step 7: Update centroids
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                if len(P_sets[j]) > 0 and len(Q_sets[j]) > 0:
                    weighted_sum = np.zeros(X.shape[1])

                    Q_indices = np.array(Q_sets[j])
                    weighted_sum = (
                        X[Q_indices] / self.norms[Q_indices, np.newaxis]
                    ).sum(axis=0) * (self.V_norm_2_1 / self.q)

                    # Update the centroid for cluster j
                    coefficient = self.p / (self.n * len(P_sets[j]))
                    new_centroids[j] = coefficient * weighted_sum
                else:
                    self.logger.warning(
                        f"Empty P or Q set for cluster {j} (|P_j|={len(P_sets[j])}, |Q_j|={len(Q_sets[j])})"
                    )

            # Compute change in centroids for convergence check
            error = np.mean(np.linalg.norm(new_centroids - centroids, axis=1))
            centroids = new_centroids

            # Store error and centroids for this iteration
            self.movements_per_iteration.append(error)
            self.centroids_per_iteration.append(centroids.copy())

            # Calculate and store the duration of this iteration
            iteration_time = timeit.default_timer() - start_time_iteration
            self.duration_of_iteration.append(iteration_time)

            self.logger.debug(
                f"EEKMeans Iteration {iteration}: error={error:.4f}, duration={iteration_time:.4f}s"
            )

            iteration += 1

        # Store results as attributes
        self.cluster_centers_ = centroids
        self.n_iter_ = iteration

        return self


class KMeans(BaseKMeans):
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
        random_seed_centroids=None,
        logger=None,
    ):
        super().__init__(
            X=X,
            n_clusters=n_clusters,
            max_iter=max_iter,
            epsilon=epsilon,
            delta=delta,
            tol=tol,
            random_seed_centroids=random_seed_centroids,
            logger=logger,
        )

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
        if self.random_seed_centroids is not None:
            np.random.seed(self.random_seed_centroids)  # Fix randomness

        # Initialize centroids
        centroids = self._initialize_centroids(X)

        iteration = 0
        error = np.inf  # Initialize error to a large value
        self.logger.debug("Beginning main loop of KMeans.fit()")

        while (error > self.tol) and iteration < self.max_iter:
            # Start timing the iteration
            start_time_iteration = timeit.default_timer()

            # Step 1: Assign each point to the nearest centroid
            labels = np.zeros(self.n, dtype=int)
            for i in range(self.n):
                distances = np.linalg.norm(X[i] - centroids, axis=1)
                labels[i] = np.argmin(distances)

            # Step 2: Update centroids as the mean of assigned points
            new_centroids = np.zeros_like(centroids)
            for j in range(self.n_clusters):
                points_in_cluster = X[labels == j]
                if len(points_in_cluster) > 0:
                    new_centroids[j] = points_in_cluster.mean(axis=0)
                else:
                    pass

            # Compute and store change in centroids
            error = np.mean(np.linalg.norm(new_centroids - centroids, axis=1))
            self.movements_per_iteration.append(error)

            # Store centroids for this iteration
            centroids = new_centroids
            self.centroids_per_iteration.append(centroids.copy())

            # Compute and store the duration of this iteration
            iteration_time = timeit.default_timer() - start_time_iteration
            self.duration_of_iteration.append(iteration_time)

            self.logger.debug(
                f"KMeans Iteration {iteration}: error={error:.4f}, duration={iteration_time:.4f}s"
            )

            iteration += 1

        # Store results as attributes
        self.cluster_centers_ = centroids
        self.n_iter_ = iteration

        return self
