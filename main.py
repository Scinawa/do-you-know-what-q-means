import numpy as np
import logging

# import matplotlib.pyplot as plt
import scipy as sp
from sklearn.datasets import fetch_openml
from scipy.stats import rv_discrete
import sys
import epsiloneta_kmeans as ekm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


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


def main():
    X, Y = create_dataset(percentile=95)
    epsilon = 200  # approximation parameter
    delta = 0.1  # failure probability
    n = len(X)  # number of samples
    k = len(set(Y))  # number of clusters
    tolerance = 50  # convergence parameter

    logger = logging.getLogger("main")

    eekmeans = ekm.EEKMeans(
        X=X,
        n_clusters=k,
        epsilon=epsilon,
        delta=delta,
        tol=tolerance,
        constant_enabled=False,
        logger=logger,
    )

    logger.info(f"||V|| (spectral norm): {eekmeans.V_norm:.2f}")
    logger.info(f"||V||_2,1 (ell_2,1 norm): {eekmeans.V_norm_2_1:.2f}")
    logger.info(f"Number of samples: {eekmeans.n}")
    logger.info(f"Number of clusters: {eekmeans.n_clusters}")
    logger.info(f"Epsilon: {eekmeans.epsilon}")
    logger.info(f"Delta: {eekmeans.delta}")
    logger.info(f"Tolerance: {eekmeans.tol}")
    logger.info(f"Sample size p: {eekmeans.p}")
    logger.info(f"Sample size q: {eekmeans.q}")

    eekmeans.fit(X)


if __name__ == "__main__":
    main()
