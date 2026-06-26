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


def farthest_first(
    dist_matrix: np.ndarray,
    eta: float,
) -> list[int]:
    """
    Implements the Farthest-First eta-Cover construction.
    """
    num_points = dist_matrix.shape[0]
    if num_points == 0:
        return []

    Q_indices = [0]
    min_dist_to_Q = dist_matrix[0].copy()

    # We iterate up to num_points - 1 times, but early stopping will trigger long before
    for _ in tqdm(
        range(1, num_points), desc=f"FF Cover (\u03b7={eta:.2f})", leave=False, ncols=80
    ):
        # 1. Find the point that is furthest from the current cover
        # np.argmax instantly returns the index of the largest value
        candidate_idx = int(np.argmax(min_dist_to_Q))
        max_dist = min_dist_to_Q[candidate_idx]

        # 2. Check the condition
        if max_dist > eta:
            Q_indices.append(candidate_idx)
            # Vectorized update
            min_dist_to_Q = np.minimum(min_dist_to_Q, dist_matrix[candidate_idx])
        else:
            # EARLY STOPPING: The absolute furthest point is within eta.
            # Therefore, all points are covered.
            break

    return Q_indices


def greedy(
    dist_matrix: np.ndarray,
    eta: float,
) -> list[int]:
    """
    Implements the Greedy eta-Cover construction based on static eta-ball sizes.
    """
    num_points = dist_matrix.shape[0]
    if num_points == 0:
        return []

    # 1. Precompute the size of the eta-ball for every single point.
    # (dist_matrix <= eta) creates a boolean matrix (True if inside ball).
    # Summing across axis=1 counts the True values (the size of the ball) for each point.
    ball_sizes = np.sum(dist_matrix <= eta, axis=1)

    # 2. Sort the point indices by their ball size in descending order
    # argsort sorts ascending, so we use [::-1] to reverse it
    sorted_indices = np.argsort(ball_sizes)[::-1]

    # Initialize Q with the point that has the largest ball
    first_idx = sorted_indices[0]
    Q_indices = [first_idx]
    min_dist_to_Q = dist_matrix[first_idx].copy()

    # 3. Iterate through the remaining points in greedy order
    for i in tqdm(
        range(1, num_points),
        desc=f"Greedy Cover (\u03b7={eta:.2f})",
        leave=False,
        ncols=80,
    ):
        candidate_idx = sorted_indices[i]

        # If this point is not yet covered by our current set Q
        if min_dist_to_Q[candidate_idx] > eta:
            Q_indices.append(candidate_idx)
            min_dist_to_Q = np.minimum(min_dist_to_Q, dist_matrix[candidate_idx])

    return Q_indices
