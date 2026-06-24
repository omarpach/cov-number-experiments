import csv
from pathlib import Path
from typing import Union


class ExperimentLogger:
    def __init__(self, filepath: Union[str, Path] = "results/experiment_log.csv"):
        """
        Initializes the logger. Creates the output directory if it doesn't exist
        and writes the CSV headers if the file is new.
        """
        self.filepath = Path(filepath)

        # Ensure the 'results/' directory exists
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        self.fieldnames = [
            "dataset",
            "num_points",
            "dimensions",
            "eta",
            "covering_number",
            "runtime_seconds",
        ]

        # If the file doesn't exist, create it and write the header
        if not self.filepath.exists():
            with open(self.filepath, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log_run(
        self, dataset: str, n: int, d: int, eta: float, cov_num: int, runtime: float
    ):
        """
        Appends a single experimental run to the CSV file.
        """
        with open(self.filepath, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(
                {
                    "dataset": dataset,
                    "num_points": n,
                    "dimensions": d,
                    "eta": eta,
                    "covering_number": cov_num,
                    "runtime_seconds": runtime,
                }
            )
