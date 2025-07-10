# ε-k-means: Quantum-Inspired Clustering Algorithm

This repository contains an implementation of the "ε-k-means" algorithm, inspired by the q-means quantum clustering algorithm described in the paper ["Do you know what q-means?"](https://arxiv.org/abs/2308.09701) by Arjan Cornelissen, João F. Doriguello, Alessandro Luongo, and Ewin Tang.

## Background

Clustering is a fundamental technique in data analysis, with k-means being one of the most popular clustering algorithms. This implementation provides a classical version of the quantum-inspired algorithm presented in the paper, which performs ε-k-means clustering - an approximate version of k-means clustering.

The algorithm achieves logarithmic dependence on the number of data points (n), making it potentially efficient for very large datasets while maintaining good approximation guarantees.

## Features

- Implementation of the ε-k-means algorithm (EEKMeans class)
- Sampling-based approach that approximates traditional k-means
- Configurable approximation parameters (ε and δ)
- Demonstration using the MNIST dataset

## Requirements

- NumPy
- SciPy
- scikit-learn
- Python 3.6+

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/do-you-know-what-k-means.git
cd do-you-know-what-k-means
```

Install dependencies:

```bash
pip install numpy scipy scikit-learn
```

## Usage

The main algorithm is implemented in the `EEKMeans` class. Here's a simple example of how to use it:

```python
import numpy as np
from epsiloneta_kmeans import EEKMeans

# Create or load your dataset
X = np.random.rand(1000, 10)  # Example dataset with 1000 samples, 10 features

# Set parameters
k = 5  # Number of clusters
epsilon = 0.1  # Approximation parameter
delta = 0.1  # Failure probability

# Initialize and fit the model
model = EEKMeans(X=X, n_clusters=k, epsilon=epsilon, delta=delta)
model.fit(X)

# Get cluster assignments for new data
labels = model.predict(X)
```

### Example with MNIST

The repository includes an example with the MNIST dataset in `main.py`:

```bash
python main.py
```

This script loads the MNIST dataset, filters it based on norm percentile, and runs the ε-k-means algorithm to cluster the images.

## Algorithm Details

The ε-k-means algorithm works by:

1. Estimating centroids through sampling instead of using the entire dataset
2. Using two different sampling techniques:
   - Uniform sampling (set P)
   - Norm-weighted sampling (set Q)
3. Iteratively improving the centroids using these samples until convergence

The algorithm has the following complexity:
- Classical dequantized version: O(‖V‖²ₙk²/ε²(kd+log n))
- Quantum version: O(‖V‖ₙ√k^(5/2)d/ε(k√+log n))

Where:
- ‖V‖ₙ is the norm of the data matrix
- k is the number of clusters
- d is the dimension of the data
- n is the number of data points
- ε is the approximation parameter

## Parameters

The `EEKMeans` class accepts the following parameters:

- `X`: Input data matrix
- `n_clusters`: Number of clusters (k)
- `max_iter`: Maximum number of iterations
- `epsilon`: Approximation parameter (ε)
- `delta`: Failure probability (δ)
- `tol`: Convergence tolerance
- `random_state`: Random seed for reproducibility
- `constant_enabled`: Whether to use theoretical constants for sampling sizes

## References

```@article{doriguello2023you,
  title={Do you know what q-means?},
  author={Doriguello, Jo{\~a}o F and Luongo, Alessandro and Tang, Ewin},
  journal={arXiv preprint arXiv:2308.09701},
  year={2023}
}
```

## License

[MIT License](LICENSE)
