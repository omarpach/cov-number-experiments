import numpy as np
from sklearn.datasets import fetch_openml, make_swiss_roll, make_moons, make_circles
from sklearn.preprocessing import Normalizer


def load_dataset(name: str, n_samples: int = 2000, seed: int = 42, image_size: int = 64) -> np.ndarray:
    """
    Loads and normalizes various datasets for covering number experiments.
    All datasets are normalized to a [0, 1] range to keep distances comparable.
    
    Args:
        name (str): The name of the dataset to load.
        n_samples (int): The number of samples to return.
        seed (int): Random seed for reproducibility.
        image_size (int): Target resolution (e.g., 64 for 64x64) used by image datasets like CelebA.
    """
    rng = np.random.default_rng(seed)

    if name == "MNIST":
        print("Downloading/Loading MNIST dataset...")
        X, _ = fetch_openml(
            "mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto"
        )
        X = X / 255.0
        X = Normalizer().fit_transform(X)
        indices = rng.choice(X.shape[0], n_samples, replace=False)
        return X[indices]

    elif name == "FashionMNIST":
        print("Downloading/Loading Fashion-MNIST dataset...")
        X, _ = fetch_openml(
            "Fashion-MNIST", version=1, return_X_y=True, as_frame=False, parser="auto"
        )
        X = X / 255.0
        X = Normalizer().fit_transform(X)
        indices = rng.choice(X.shape[0], n_samples, replace=False)
        return X[indices]

    elif name == "SwissRoll":
        print(f"Generating 3D Swiss Roll dataset (N={n_samples})...")
        X, _ = make_swiss_roll(n_samples=n_samples, random_state=seed)
        X = (X - np.min(X)) / (np.max(X) - np.min(X))
        X = Normalizer().fit_transform(X)
        return X

    elif name == "Moons":
        print(f"Generating 2D Two Moons dataset (N={n_samples})...")
        # noise adds realistic scatter to the curves
        X, _ = make_moons(n_samples=n_samples, noise=0.1, random_state=seed)
        X = (X - np.min(X)) / (np.max(X) - np.min(X))
        X = Normalizer().fit_transform(X)
        return X

    elif name == "Circles":
        print(f"Generating 2D Concentric Circles dataset (N={n_samples})...")
        # factor is the distance between the inner and outer circles
        X, _ = make_circles(
            n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed
        )
        X = (X - np.min(X)) / (np.max(X) - np.min(X))
        X = Normalizer().fit_transform(X)
        return X

    elif name == "CelebA":
        try:
            from torchvision import datasets, transforms
        except ImportError:
            raise ImportError("Loading CelebA requires PyTorch and Torchvision. Run: pip install torchvision")
        
        print(f"Loading CelebA dataset (N={n_samples}, Size={image_size}x{image_size})...")
        
        # Original CelebA images are 178x218 RGB. 
        # We crop the face and resize to `image_size` x `image_size`.
        transform = transforms.Compose([
            transforms.CenterCrop(148),
            transforms.Resize(image_size),
            transforms.ToTensor() # Automatically normalizes image to [0.0, 1.0]
        ])
        
        # split="all" merges train, val, and test into one massive pool of 202,599 images
        dataset = datasets.CelebA(root="./data", split="all", transform=transform, download=True)
        
        # Ensure we don't request more samples than exist
        safe_n_samples = min(n_samples, len(dataset))
        
        # Randomly pick the indices using your existing seeded random number generator
        indices = rng.choice(len(dataset), safe_n_samples, replace=False)
        
        # Only load the specifically requested subset into RAM
        X = []
        for idx in indices:
            img, _ = dataset[idx]
            # img is (3, image_size, image_size). Flatten to a 1D vector.
            X.append(img.numpy().flatten())
            
        return np.array(X)

    else:
        raise ValueError(f"Unknown dataset: {name}")