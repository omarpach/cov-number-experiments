import numpy as np
from tqdm import tqdm


def uncovered_first(
    dist_matrix: np.ndarray,
    eta: float,
) -> list[int]:
    """
    Implements the Uncovered-First eta-Cover algorithm using a precomputed distance matrix.

    Returns:
        list[int]: The indices of the points that form the eta-cover Q.
    """
    num_points = dist_matrix.shape[0]
    if num_points == 0:
        return []

    Q_indices = [0]

    # Track the minimum distance from every point to the current cover Q.
    # Initially, the cover only contains the first point (index 0).
    min_dist_to_Q = dist_matrix[0].copy()

    for i in tqdm(
        range(1, num_points), desc=f"Covering (\u03b7={eta:.2f})", leave=False, ncols=80
    ):
        # If the minimum distance to the cover is greater than eta, it's uncovered
        if min_dist_to_Q[i] > eta:
            Q_indices.append(i)

            # Vectorized update: The new minimum distance to the cover for all points
            # is the element-wise minimum of the old distances and the distances to the new point 'i'.
            min_dist_to_Q = np.minimum(min_dist_to_Q, dist_matrix[i])

    return Q_indices
