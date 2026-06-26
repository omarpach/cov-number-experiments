import numpy as np
from sklearn.datasets import fetch_openml


def load_mnist_subset(n_samples: int = 2000, seed: int = 42) -> np.ndarray:
    print("Downloading/Loading MNIST dataset...")
    X, _ = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto"
    )

    X = X / 255.0

    # Initialize the random number generator with the provided seed
    rng = np.random.default_rng(seed)

    # Subsample reproducibly
    indices = rng.choice(X.shape[0], n_samples, replace=False)
    return X[indices]
