# main.py
import numpy as np
import scipy.spatial.distance as dist
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from metric_space import MetricSpace
from algorithms import (
    uncovered_first,
)  # Assuming you saved the algorithm in algorithm.py


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


def plot_cover_gallery(cover_points: np.ndarray, title: str):
    """Plots a grid of the images that made it into the eta-cover."""
    n_images = min(len(cover_points), 25)  # Plot up to 25 images
    grid_size = int(np.ceil(np.sqrt(n_images)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    fig.suptitle(title, fontsize=14)

    for i, ax in enumerate(axes.flat):
        if i < n_images:
            # Reshape the 784-D point back into a 28x28 image for human viewing
            img = cover_points[i].reshape(28, 28)
            ax.imshow(img, cmap="gray")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def get_exact_min_max_distances(
    points: np.ndarray, batch_size: int = 2000
) -> tuple[float, float]:
    """
    Computes the exact global minimum and maximum pairwise distances
    without storing the full distance matrix in memory.
    """
    N = points.shape[0]
    global_min = np.inf
    global_max = -np.inf

    # Iterate through the data in manageable chunks
    for i in range(0, N, batch_size):
        chunk = points[i : i + batch_size]

        # We only compare the chunk to points from 'i' onwards to save compute.
        distances = dist.cdist(chunk, points[i:], metric="euclidean")

        if distances.size > 0:
            # 1. Get the MAX safely BEFORE we alter the matrix
            # (A self-distance of 0.0 will never be the maximum)
            local_max = np.max(distances)
            if local_max > global_max:
                global_max = local_max

            # 2. Hide the self-distances with infinity
            # (Because chunk[0] == points[i], the self-distances sit perfectly on the diagonal)
            np.fill_diagonal(distances, np.inf)

            # 3. Now get the MIN safely
            local_min = np.min(distances)
            if local_min < global_min:
                global_min = local_min

        print(f"Processed up to row {min(i + batch_size, N)} / {N}...")

    return float(global_min), float(global_max)


if __name__ == "__main__":
    # --- 1. Load Data ---
    N_SAMPLES = 2000
    mnist_data = load_mnist_subset(n_samples=N_SAMPLES)

    # --- 2. Initialize Metric Space ---
    # Passing exactly what your new MetricSpace class expects
    space = MetricSpace(
        points=mnist_data, distance=dist.euclidean, distance_name="euclidean"
    )

    print("\nCreated Metric Space:")
    print(f"Total Points: {space.num_points}")
    print(f"Dimensions: {space.dimensions}D")

    # --- Execution ---
    min_dist, max_dist = get_exact_min_max_distances(mnist_data, batch_size=2500)
    print(f"\nFinal Min Distance: {min_dist:.4f}")
    print(f"Final Max Distance: {max_dist:.4f}")

    # --- 3. Calibrate Eta ---
    # In 784D space, points are very far apart. Let's find the median distance
    # of a small sample to pick a smart eta automatically.
    sample_distances = dist.pdist(space.points[:500], metric="euclidean")
    median_dist = np.median(sample_distances)
    print(f"Median distance between random MNIST digits: {median_dist:.2f}")

    # Set eta to be slightly less than the median distance to ensure we capture
    # distinct variations of handwriting styles.
    ETA = float(median_dist * 0.70)
    print(f"Running Uncovered-First Algorithm with eta = {ETA:.2f}...\n")

    # --- 4. Run the Algorithm ---
    # Note: We pass space.distance_name to ensure the cdist matches your space definition
    cover = uncovered_first(space, eta=ETA, metric_name=space.distance_name)

    # --- 5. Analyze and Visualize ---
    compression_ratio = (1 - (len(cover) / space.num_points)) * 100
    print("Eta-Cover complete!")
    print(f"Original size: {space.num_points}")
    print(f"Cover size: {len(cover)} (Compressed by {compression_ratio:.1f}%)")

    # Show the "skeleton" of the dataset
    plot_cover_gallery(
        cover,
        f"MNIST $\\eta$-Cover ($\\eta={ETA:.2f}$)\nShowing first 25 representative digits",
    )
