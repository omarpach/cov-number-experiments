# metric_space.py
import numpy as np
import scipy.spatial.distance as dist
from dataclasses import dataclass
from typing import Callable

Metric = Callable[[np.ndarray, np.ndarray], float]


@dataclass
class MetricSpace:
    """
    A modular finite metric space where data is stored in a continuous NumPy block.
    """

    points: np.ndarray
    distance: Metric
    distance_name: str

    @property
    def num_points(self) -> int:
        return self.points.shape[0]

    @property
    def dimensions(self) -> int:
        return self.points.shape[1]

    def distance_by_index(self, i: int, j: int) -> float:
        """
        Calculates the distance between the i-th and j-th point
        using your chosen point-to-point distance function.
        """
        return self.distance(self.points[i], self.points[j])

    def get_pairwise_matrix(self) -> np.ndarray:
        """
        Computes the complete NxN square distance matrix.
        Uses SciPy's cdist to instantly map your distance function across the matrix.
        """
        # dist.cdist natively accepts functions like dist.euclidean
        # and computes everything in optimized C loops.
        return dist.cdist(self.points, self.points, metric=self.distance)
