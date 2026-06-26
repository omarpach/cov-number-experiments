import time
import numpy as np
from scipy.spatial.distance import pdist, squareform
from src.logger import ExperimentLogger
from src.datasets import load_mnist_subset
from src.distances import generate_eta_range
from src.algorithms import uncovered_first


def main():
    logger = ExperimentLogger("results/experiment_log_v2.csv")
    dataset_name = "MNIST_Subset"

    # Experiment parameters
    sizes_to_test = [5000, 10000, 20000]
    EXPERIMENT_SEED = 42  # Change this to run different variations
    NUM_STEPS = 20

    for N in sizes_to_test:
        print(f"\n{'=' * 50}")
        print(f"EXPERIMENT: N = {N:,} | Seed = {EXPERIMENT_SEED}")
        print(f"{'=' * 50}")

        # 1. Load exact subset
        points = load_mnist_subset(n_samples=N, seed=EXPERIMENT_SEED)
        D = points.shape[1]

        # 2. Precompute the Distance Matrix for this N
        print(f"Precomputing distance matrix for {N:,} points...")
        matrix_start = time.perf_counter()
        dist_matrix = squareform(pdist(points, metric="euclidean"))
        matrix_end = time.perf_counter()
        print(f"Matrix computed in {matrix_end - matrix_start:.2f}s")

        # 3. Calculate Exact Bounds
        np.fill_diagonal(dist_matrix, np.inf)
        min_dist = np.min(dist_matrix)

        np.fill_diagonal(dist_matrix, 0.0)
        max_dist = np.max(dist_matrix)

        etas = generate_eta_range(min_dist, max_dist, num_steps=NUM_STEPS)
        print(f"Exact bounds -> Min: {min_dist:.4f}, Max: {max_dist:.4f}")
        print(f"Starting sweeps across {len(etas)} \u03b7 values...\n")

        # 4. Sweep Etas
        for current_eta in etas:
            start_time = time.perf_counter()

            cover_indices = uncovered_first(dist_matrix, eta=current_eta)

            end_time = time.perf_counter()
            run_time = end_time - start_time
            cov_num = len(cover_indices)

            logger.log_run(
                dataset=dataset_name,
                seed=EXPERIMENT_SEED,
                n=N,
                d=D,
                eta=current_eta,
                cov_num=cov_num,
                runtime=run_time,
            )

            print(
                f"  \u2713 \u03b7 = {current_eta:<6.4f} | Cover size: {cov_num:<6} | Time: {run_time:.4f}s"
            )


if __name__ == "__main__":
    main()
