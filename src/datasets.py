import numpy as np
from sklearn.datasets import fetch_openml
from scipy.spatial.distance import cdist


def load_mnist_subset(n_samples: int = 2000) -> np.ndarray:
    """
    Fetches the MNIST dataset and returns a normalized subset.
    Each image becomes a 784-dimensional point (28x28 flattened).
    """
    print("Downloading/Loading MNIST dataset (this might take a few seconds)...")
    # Fetch MNIST from OpenML
    X, _ = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto"
    )

    # 1. Normalize pixels from [0, 255] to [0.0, 1.0]
    # This keeps Euclidean distances mathematically stable
    X = X / 255.0

    # 2. Subsample to n_samples to keep the algorithm fast for testing
    indices = np.random.choice(X.shape[0], n_samples, replace=False)
    return X[indices]
