import time
import numpy as np
from scipy.spatial.distance import pdist, squareform
from src.logger import ExperimentLogger
from src.datasets import load_mnist_subset
from src.distances import generate_eta_range
from src.algorithms import uncovered_first, farthest_first, greedy


def main():
    logger = ExperimentLogger("results/experiment_log_v3.csv")
    dataset_name = "MNIST_Subset"

    sizes_to_test = [5000]
    # sizes_to_test = list(range(10000, 70001, 10000))
    num_steps = 30
    experiment_seed = 69

    for N in sizes_to_test:
        print(f"\n{'=' * 60}")
        print(f"EXPERIMENT: N = {N:,} | Seed = {experiment_seed}")
        print(f"{'=' * 60}")

        points = load_mnist_subset(n_samples=N, seed=experiment_seed)
        D = points.shape[1]

        print(f"Precomputing distance matrix for {N:,} points...")
        matrix_start = time.perf_counter()
        dist_matrix = squareform(pdist(points, metric="euclidean"))
        matrix_end = time.perf_counter()
        print(f"Matrix computed in {matrix_end - matrix_start:.2f}s")

        np.fill_diagonal(dist_matrix, np.inf)
        min_dist = np.min(dist_matrix)

        np.fill_diagonal(dist_matrix, 0.0)
        max_dist = np.max(dist_matrix)

        etas = generate_eta_range(min_dist, max_dist, num_steps=num_steps)
        print(f"Exact bounds -> Min: {min_dist:.4f}, Max: {max_dist:.4f}")
        print(f"Starting sweeps across {len(etas)} \u03b7 values...\n")

        for current_eta in etas:
            # --- 1. Uncovered-First ---
            start_uf = time.perf_counter()
            uf_cover = uncovered_first(dist_matrix, eta=current_eta)
            time_uf = time.perf_counter() - start_uf
            logger.log_run(
                dataset_name,
                experiment_seed,
                N,
                D,
                "Uncovered-First",
                current_eta,
                len(uf_cover),
                time_uf,
            )

            # --- 2. Farthest-First ---
            start_ff = time.perf_counter()
            ff_cover = farthest_first(dist_matrix, eta=current_eta)
            time_ff = time.perf_counter() - start_ff
            logger.log_run(
                dataset_name,
                experiment_seed,
                N,
                D,
                "Farthest-First",
                current_eta,
                len(ff_cover),
                time_ff,
            )

            # --- 3. Greedy ---
            start_gr = time.perf_counter()
            gr_cover = greedy(dist_matrix, eta=current_eta)
            time_gr = time.perf_counter() - start_gr
            logger.log_run(
                dataset_name,
                experiment_seed,
                N,
                D,
                "Greedy",
                current_eta,
                len(gr_cover),
                time_gr,
            )

            # Compact console output to compare runtimes and sizes
            print(
                f"  \u2713 \u03b7 = {current_eta:<6.4f} | Covers -> UF: {len(uf_cover):<5} FF: {len(ff_cover):<5} GR: {len(gr_cover):<5}"
            )
            print(
                f"                 | Times  -> UF: {time_uf:.3f}s FF: {time_ff:.3f}s GR: {time_gr:.3f}s"
            )


if __name__ == "__main__":
    main()
