import numpy as np
import torch
from scipy.spatial.distance import cdist, squareform, pdist


def get_exact_min_max_distances(
    points: np.ndarray, batch_size: int = 2000, use_gpu: bool = True
) -> tuple[float, float]:
    """
    Computes the exact global minimum and maximum pairwise distances
    without storing the full distance matrix in memory.
    """
    N = points.shape[0]
    global_min = np.inf
    global_max = -np.inf

    if use_gpu:
        try:
            if torch.cuda.is_available():
                points_tensor = torch.tensor(points, dtype=torch.float32, device="cuda")
                
                for i in range(0, N, batch_size):
                    chunk = points_tensor[i : i + batch_size]
                    
                    # We only compare the chunk to points from 'i' onwards to save compute.
                    distances = torch.cdist(chunk, points_tensor[i:])
                    
                    if distances.numel() > 0:
                        local_max = torch.max(distances).item()
                        if local_max > global_max:
                            global_max = local_max
                            
                        # PyTorch's fill_diagonal_ handles the non-square sub-matrix perfectly
                        distances.fill_diagonal_(float('inf'))
                        
                        local_min = torch.min(distances).item()
                        if local_min < global_min:
                            global_min = local_min
                            
                    print(f"Processed up to row {min(i + batch_size, N)} / {N}...")
                    
                return float(global_min), float(global_max)
        except ImportError:
            pass

    # CPU Fallback
    # Iterate through the data in manageable chunks
    for i in range(0, N, batch_size):
        chunk = points[i : i + batch_size]

        # We only compare the chunk to points from 'i' onwards to save compute.
        distances = cdist(chunk, points[i:], metric="euclidean")

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


def generate_eta_range(
    min_dist: float, max_dist: float, num_steps: int = 20
) -> np.ndarray:
    """
    Generates a logarithmic sequence of eta values for covering number experiments.

    The range starts at min_dist / 2 and goes up to max_dist. The returned
    array is sorted in descending order (largest eta first).

    Args:
        min_dist: The minimum global pairwise distance.
        max_dist: The maximum global pairwise distance.
        num_steps: The number of eta values to generate.

    Returns:
        np.ndarray: A 1D array of eta values.
    """
    # Calculate the lower bound: half of the minimum pairwise distance
    eta_min = min_dist / 2.0
    eta_max = max_dist

    # Mathematical guard: geomspace cannot handle starting at 0.
    # If identical points exist and min_dist is 0.0, bump the minimum
    # to a tiny epsilon to prevent mathematical errors.
    if eta_min <= 0:
        eta_min = 1e-6

    # Edge case guard
    if eta_min >= eta_max:
        return np.array([eta_max])

    # Generate logarithmically spaced values
    etas = np.linspace(eta_min, eta_max, num=num_steps)

    # Reverse the array so the largest etas are first.
    # A larger eta produces a smaller cover, meaning the algorithm runs much faster.
    # Testing largest-first gives you immediate output in your logs before
    # hitting the computationally heavy small-eta runs.
    return etas[::-1]

def get_distance_matrix(points: np.ndarray, use_gpu: bool = True) -> np.ndarray:
    """
    Computes the full NxN pairwise Euclidean distance matrix.
    Leverages PyTorch/CUDA if available for a massive speedup on high-dimensional data.
    """
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                # Convert to float32 to halve memory usage (20GB -> 10GB for 50k points)
                # and drastically improve GPU throughput.
                points_tensor = torch.tensor(points, dtype=torch.float32, device="cuda")
                
                # Perform the highly parallelized computation
                dist_matrix_tensor = torch.cdist(points_tensor, points_tensor)
                
                # Pull the result back to system RAM as a standard NumPy array
                return dist_matrix_tensor.cpu().numpy()
            else:
                print("Warning: CUDA is not available. Falling back to CPU calculation.")
        except ImportError:
            print("Warning: PyTorch is not installed. Falling back to CPU calculation.")

    # Fallback to the standard SciPy CPU calculation (Float64)
    return squareform(pdist(points, metric="euclidean"))