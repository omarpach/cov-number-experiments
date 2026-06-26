import csv
from pathlib import Path
from typing import Union


class ExperimentLogger:
    def __init__(self, filepath: Union[str, Path] = "results/experiment_log.csv"):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        self.fieldnames = [
            "dataset",
            "seed",  # NEW: Tracking the random seed
            "num_points",
            "dimensions",
            "eta",
            "covering_number",
            "runtime_seconds",
        ]

        if not self.filepath.exists():
            with open(self.filepath, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log_run(
        self,
        dataset: str,
        seed: int,
        n: int,
        d: int,
        eta: float,
        cov_num: int,
        runtime: float,
    ):
        with open(self.filepath, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "num_points": n,
                    "dimensions": d,
                    "eta": eta,
                    "covering_number": cov_num,
                    "runtime_seconds": runtime,
                }
            )
