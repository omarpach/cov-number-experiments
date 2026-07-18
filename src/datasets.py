import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.datasets import make_swiss_roll, make_moons, make_circles

def load_dataset(name: str, n_samples: int = 2000, seed: int = 42, image_size: int = 64, num_workers: int = 4) -> torch.Tensor:
    """
    Loads and normalizes various datasets for covering number experiments.
    Returns a flattened PyTorch Tensor of shape (N, Features).
    
    Args:
        name (str): The name of the dataset to load.
        n_samples (int): The number of samples to return.
        seed (int): Random seed for reproducibility.
        image_size (int): Target resolution used by image datasets like CelebA.
        num_workers (int): Number of subprocesses for data loading.
    """
    rng = np.random.default_rng(seed)

    if name in ["MNIST", "FashionMNIST"]:
        print(f"Downloading/Loading {name} dataset...")
        
        # ToTensor() automatically scales pixels to [0.0, 1.0]
        transform = transforms.Compose([transforms.ToTensor()])
        DatasetClass = datasets.MNIST if name == "MNIST" else datasets.FashionMNIST
        
        train_dataset = DatasetClass(root="./data", train=True, transform=transform, download=True)
        test_dataset = DatasetClass(root="./data", train=False, transform=transform, download=True)
        dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        
        safe_n_samples = min(n_samples, len(dataset))
        indices = rng.choice(len(dataset), safe_n_samples, replace=False).tolist()
        
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=safe_n_samples,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        images, _ = next(iter(loader))
        X = images.view(safe_n_samples, -1) # Flatten to (N, Features)
        
        # Apply L2 Normalization (Replacing sklearn's Normalizer)
        X = F.normalize(X, p=2, dim=1)
        return X

    elif name in ["SwissRoll", "Moons", "Circles"]:
        print(f"Generating {name} dataset (N={n_samples})...")
        
        if name == "SwissRoll":
            X_np, _ = make_swiss_roll(n_samples=n_samples, random_state=seed)
        elif name == "Moons":
            X_np, _ = make_moons(n_samples=n_samples, noise=0.1, random_state=seed)
        elif name == "Circles":
            X_np, _ = make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed)
            
        # Min-Max scaling
        X_np = (X_np - np.min(X_np)) / (np.max(X_np) - np.min(X_np))
        
        # Convert to Tensor and apply L2 Normalization
        X = torch.tensor(X_np, dtype=torch.float32)
        X = F.normalize(X, p=2, dim=1)
        return X

    elif name == "CelebA":
        print(f"Loading CelebA dataset (N={n_samples}, Size={image_size}x{image_size})...")
        
        transform = transforms.Compose([
            transforms.CenterCrop(148),
            transforms.Resize(image_size),
            transforms.ToTensor() # Automatically normalizes image to [0.0, 1.0]
        ])
        
        dataset = datasets.CelebA(root="./data", split="all", transform=transform, download=True)
        safe_n_samples = min(n_samples, len(dataset))
        indices = rng.choice(len(dataset), safe_n_samples, replace=False).tolist()
        
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=safe_n_samples,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        images, _ = next(iter(loader))
        # Flatten to (N, Features)
        return images.view(safe_n_samples, -1)

    else:
        raise ValueError(f"Unknown dataset: {name}")