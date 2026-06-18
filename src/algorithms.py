import numpy as np
import scipy.spatial.distance as dist
from metric_space import MetricSpace


def uncovered_first(
    space: MetricSpace, eta: float, metric_name: str = "euclidean"
) -> np.ndarray:
    """
    Implements the Uncovered-First eta-Cover Construction algorithm.

    Returns:
        np.ndarray: A 2D array containing the points that form the eta-cover Q.
    """
    m = space.num_points
    if m == 0:
        return np.array([])

    points = space.points

    # Q_indices stores the row indices of points added to the cover.
    # We initialize Q with x^1 (index 0 in Python)
    Q_indices = [0]

    # for i = 2 to m do (in Python, indices 1 to m-1)
    for i in range(1, m):
        candidate_point = points[i : i + 1]  # Sliced as a 2D array (1, d) for cdist
        current_cover = points[Q_indices]  # Sliced as (k, d) where k is size of Q

        # Calculate distances from the candidate point to ALL points currently in Q
        # cdist returns a matrix of shape (1, k), we flatten it to 1D
        distances_to_Q = dist.cdist(
            candidate_point, current_cover, metric=metric_name
        ).flatten()

        # \rho(x^i, Q) is the minimum distance to the set Q
        rho = np.min(distances_to_Q)

        # if \rho(x^i, Q) > \eta then Q <- Q U {x^i}
        if rho > eta:
            Q_indices.append(i)

    # Return the actual points making up the cover
    return points[Q_indices]
