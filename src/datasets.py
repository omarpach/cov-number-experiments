import numpy as np
from sklearn.datasets import fetch_openml, make_swiss_roll, make_moons, make_circles


def load_dataset(name: str, n_samples: int = 2000, seed: int = 42) -> np.ndarray:
    """
    Loads and normalizes various datasets for covering number experiments.
    All datasets are normalized to a [0, 1] range to keep distances comparable.
    """
    rng = np.random.default_rng(seed)

    if name == "MNIST":
        print("Downloading/Loading MNIST dataset...")
        X, _ = fetch_openml(
            "mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto"
        )
        X = X / 255.0
        indices = rng.choice(X.shape[0], n_samples, replace=False)
        return X[indices]

    elif name == "FashionMNIST":
        print("Downloading/Loading Fashion-MNIST dataset...")
        X, _ = fetch_openml(
            "Fashion-MNIST", version=1, return_X_y=True, as_frame=False, parser="auto"
        )
        X = X / 255.0
        indices = rng.choice(X.shape[0], n_samples, replace=False)
        return X[indices]

    elif name == "SwissRoll":
        print(f"Generating 3D Swiss Roll dataset (N={n_samples})...")
        X, _ = make_swiss_roll(n_samples=n_samples, random_state=seed)
        X = (X - np.min(X)) / (np.max(X) - np.min(X))
        return X

    elif name == "Moons":
        print(f"Generating 2D Two Moons dataset (N={n_samples})...")
        # noise adds realistic scatter to the curves
        X, _ = make_moons(n_samples=n_samples, noise=0.1, random_state=seed)
        X = (X - np.min(X)) / (np.max(X) - np.min(X))
        return X

    elif name == "Circles":
        print(f"Generating 2D Concentric Circles dataset (N={n_samples})...")
        # factor is the distance between the inner and outer circles
        X, _ = make_circles(
            n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed
        )
        X = (X - np.min(X)) / (np.max(X) - np.min(X))
        return X

    else:
        raise ValueError(f"Unknown dataset: {name}")
