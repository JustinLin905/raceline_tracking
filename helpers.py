import numpy as np
from numpy.typing import ArrayLike


def distance_to_centerline(position: np.ndarray, centerline: np.ndarray) -> float:
    """Compute the minimum distance from a position to the racetrack centerline.

    Args:
        position (np.ndarray): The (x, y) position of the vehicle.
        centerline (np.ndarray): The array of (x, y) points defining the racetrack centerline.

    Returns:
        float: The minimum distance to the centerline.
    """
    deltas = centerline - position
    distances = np.linalg.norm(deltas, axis=1)
    return np.min(distances)


def index_of_closest_point(position: np.ndarray, centerline: np.ndarray) -> int:
    """Find the index of the closest point on the centerline to the given position.

    Args:
        position (np.ndarray): The (x, y) position of the vehicle.
        centerline (np.ndarray): The array of (x, y) points defining the racetrack centerline.

    Returns:
        int: The index of the closest point on the centerline.
    """
    deltas = centerline - position
    distances = np.linalg.norm(deltas, axis=1)
    return np.argmin(distances)


def estimate_upcoming_curvature(
    state: ArrayLike, waypoints: ArrayLike, preview_points: int = 10
):
    """
    Estimate the maximum curvature in the upcoming section of the path.
    Used for dynamic lookahead adjustment.

    Args:
        state: Current state
        waypoints: Raceline waypoints
        preview_points: Number of points to check ahead

    Returns:
        Maximum curvature in upcoming section
    """
    current_pos = np.array([state[0], state[1]])
    closest_idx = index_of_closest_point(current_pos, waypoints)

    max_curvature = 0.0

    for i in range(min(preview_points, len(waypoints) - 2)):
        idx = (closest_idx + i) % len(waypoints)
        idx_next = (closest_idx + i + 1) % len(waypoints)
        idx_next_next = (closest_idx + i + 2) % len(waypoints)

        # Look ahead using three consecutive forward points
        p1 = waypoints[idx]
        p2 = waypoints[idx_next]
        p3 = waypoints[idx_next_next]

        # Menger curvature
        area = 0.5 * abs(
            (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
        )
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)

        if a * b * c > 1e-6:
            curvature = 4 * area / (a * b * c)
            max_curvature = max(max_curvature, curvature)

    return max_curvature
