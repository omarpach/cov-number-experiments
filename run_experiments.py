import time
from src.logger import ExperimentLogger
from src.datasets import load_mnist_subset
from src.distances import get_exact_min_max_distances, generate_eta_range
from src.algorithms import uncovered_first


def main():
    # 1. Initialize logger
    logger = ExperimentLogger("results/experiment_log.csv")

    # 2. Setup Data
    dataset_name = "MNIST_Subset"
    N = 20000
    points = load_mnist_subset(n_samples=N)
    D = points.shape[1]

    # 3. Calculate Bounds
    print(f"Calculating min/max distances for {dataset_name}...")
    # min_dist, max_dist = get_exact_min_max_distances(points)
    etas = generate_eta_range(0.8, 14, num_steps=10)
    print("Min distance: 0.8, Max distance: 14")

    print(f"Generated {len(etas)} eta values. Starting experiments...")

    # 4. Run loop
    for current_eta in etas:
        print(f"Running eta = {current_eta:.4f}...")

        start_time = time.perf_counter()

        # Run your covering algorithm directly on the numpy array
        cover = uncovered_first(points, eta=current_eta, metric_name="euclidean")

        end_time = time.perf_counter()
        run_time = end_time - start_time
        cov_num = len(cover)

        # 5. Log immediately
        logger.log_run(
            dataset=dataset_name,
            n=N,
            d=D,
            eta=current_eta,
            cov_num=cov_num,
            runtime=run_time,
        )

        print(f"  -> Cover size: {cov_num} | Time: {run_time:.2f}s")


if __name__ == "__main__":
    main()
