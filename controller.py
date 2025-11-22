import numpy as np
from numpy.typing import ArrayLike

from simulator import RaceTrack

# Global variable to store raceline data
raceline_data = None

def load_raceline(filepath: str):
    """Load raceline waypoints from CSV file"""
    global raceline_data
    data = np.loadtxt(filepath, comments="#", delimiter=",")
    raceline_data = data  # data contains [x, y] coordinates
    return raceline_data

def find_closest_waypoint(position: ArrayLike, waypoints: ArrayLike):
    """Find the index of the closest waypoint to current position"""
    distances = np.linalg.norm(waypoints - position, axis=1)
    return np.argmin(distances)

def pure_pursuit(state: ArrayLike, waypoints: ArrayLike, lookahead_distance: float, wheelbase: float):
    """
    Pure Pursuit algorithm for path tracking

    Args:
        state: Current state [x, y, steering_angle, velocity, heading]
        waypoints: Array of [x, y] waypoint coordinates
        lookahead_distance: Distance to look ahead on the path
        wheelbase: Vehicle wheelbase length

    Returns:
        Desired steering angle
    """
    # Current position and heading
    x, y = state[0], state[1]
    theta = state[4]
    current_pos = np.array([x, y])

    # Find closest waypoint
    closest_idx = find_closest_waypoint(current_pos, waypoints)

    # Search for lookahead point starting from closest waypoint
    lookahead_idx = closest_idx
    for i in range(closest_idx, closest_idx + len(waypoints)):
        idx = i % len(waypoints)
        dist = np.linalg.norm(waypoints[idx] - current_pos)
        if dist >= lookahead_distance:
            lookahead_idx = idx
            break

    # Lookahead point
    lookahead_point = waypoints[lookahead_idx]

    # Transform lookahead point to vehicle frame
    dx = lookahead_point[0] - x
    dy = lookahead_point[1] - y

    # Rotate to vehicle frame
    dx_veh = dx * np.cos(-theta) - dy * np.sin(-theta)
    dy_veh = dx * np.sin(-theta) + dy * np.cos(-theta)

    # Calculate curvature and desired steering angle
    ld_actual = np.sqrt(dx_veh**2 + dy_veh**2)
    if ld_actual < 0.1:
        ld_actual = 0.1

    curvature = 2.0 * dy_veh / (ld_actual ** 2)
    desired_steering = np.arctan(wheelbase * curvature)

    return desired_steering

def compute_desired_velocity(state: ArrayLike, waypoints: ArrayLike, lookahead_distance: float,
                             base_velocity: float = 80.0, max_velocity: float = 100.0):
    """
    Compute desired velocity based on upcoming path curvature

    Args:
        state: Current state
        waypoints: Raceline waypoints
        lookahead_distance: Distance to look ahead
        base_velocity: Base velocity for straight sections
        max_velocity: Maximum allowed velocity

    Returns:
        Desired velocity
    """
    current_pos = np.array([state[0], state[1]])
    closest_idx = find_closest_waypoint(current_pos, waypoints)

    # Look ahead to estimate curvature
    num_points = min(20, len(waypoints))
    curvature_sum = 0.0

    for i in range(num_points):
        idx = (closest_idx + i) % len(waypoints)
        idx_next = (closest_idx + i + 1) % len(waypoints)
        idx_prev = (closest_idx + i - 1) % len(waypoints)

        # Estimate curvature using three consecutive points
        p1 = waypoints[idx_prev]
        p2 = waypoints[idx]
        p3 = waypoints[idx_next]

        # Menger curvature formula
        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)

        if a * b * c > 1e-6:
            curvature = 4 * area / (a * b * c)
        else:
            curvature = 0.0

        curvature_sum += curvature

    avg_curvature = curvature_sum / num_points

    # Adjust velocity based on curvature (higher curvature = lower velocity)
    if avg_curvature > 0.01:
        desired_vel = base_velocity * (1.0 - min(avg_curvature * 50, 0.5))
    else:
        desired_vel = base_velocity

    return np.clip(desired_vel, 20.0, max_velocity)

def lower_controller(
    state : ArrayLike, desired : ArrayLike, parameters : ArrayLike
) -> ArrayLike:
    """
    Lower-level controller that converts desired steering angle and velocity
    to steering rate and acceleration commands.

    Args:
        state: Current state [x, y, steering_angle, velocity, heading]
        desired: Desired [steering_angle, velocity]
        parameters: Vehicle parameters

    Returns:
        Control inputs [steering_rate, acceleration]
    """
    assert(desired.shape == (2,))

    # Extract current state
    current_steering = state[2]
    current_velocity = state[3]

    # Desired values
    desired_steering = desired[0]
    desired_velocity = desired[1]

    # Extract parameters
    max_steering_vel = parameters[9]  # max steering rate
    max_acceleration = parameters[10]  # max acceleration

    # PD controller for steering
    Kp_steer = 3.0
    Kd_steer = 0.5

    steering_error = desired_steering - current_steering
    # Simple derivative approximation (would need state history for true derivative)
    steering_rate = Kp_steer * steering_error
    steering_rate = np.clip(steering_rate, -max_steering_vel, max_steering_vel)

    # PD controller for velocity
    Kp_vel = 2.0
    Kd_vel = 0.1

    velocity_error = desired_velocity - current_velocity
    acceleration = Kp_vel * velocity_error
    acceleration = np.clip(acceleration, -max_acceleration, max_acceleration)

    return np.array([steering_rate, acceleration]).T

def controller(
    state : ArrayLike, parameters : ArrayLike, racetrack : RaceTrack
) -> ArrayLike:
    """
    High-level controller that computes desired steering angle and velocity

    Args:
        state: Current state [x, y, steering_angle, velocity, heading]
        parameters: Vehicle parameters
        racetrack: RaceTrack object

    Returns:
        Desired [steering_angle, velocity]
    """
    global raceline_data

    # Load raceline if not already loaded
    if raceline_data is None:
        # This assumes raceline file follows naming convention
        # You may need to modify this path based on your setup
        import os
        import sys
        if len(sys.argv) >= 3:
            raceline_path = sys.argv[2]
        else:
            # Default fallback
            raceline_path = "./racetracks/Montreal_raceline.csv"

        if os.path.exists(raceline_path):
            load_raceline(raceline_path)
        else:
            # Fallback: use centerline if raceline not available
            raceline_data = racetrack.centerline

    # Get vehicle parameters
    wheelbase = parameters[0]
    current_velocity = state[3]

    # Adaptive lookahead distance based on velocity
    base_lookahead = 15.0
    lookahead_distance = base_lookahead + current_velocity * 0.3
    lookahead_distance = np.clip(lookahead_distance, 10.0, 50.0)

    # Compute desired steering using Pure Pursuit
    desired_steering = pure_pursuit(state, raceline_data, lookahead_distance, wheelbase)

    # Compute desired velocity based on upcoming curvature
    desired_velocity = compute_desired_velocity(state, raceline_data, lookahead_distance)

    return np.array([desired_steering, desired_velocity]).T