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
    Pure Pursuit algorithm for path tracking with cross-track error correction

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

    # Find closest waypoint on path
    closest_idx = find_closest_waypoint(current_pos, waypoints)
    
    # Find lookahead point: search forward along path from closest point
    # Accumulate arc length until we reach lookahead_distance
    lookahead_idx = closest_idx
    accumulated_dist = 0.0
    
    for i in range(1, len(waypoints)):
        idx = (closest_idx + i) % len(waypoints)
        idx_prev = (closest_idx + i - 1) % len(waypoints)
        
        # Distance along path (not straight line to vehicle)
        segment_length = np.linalg.norm(waypoints[idx] - waypoints[idx_prev])
        accumulated_dist += segment_length
        
        if accumulated_dist >= lookahead_distance:
            lookahead_idx = idx
            break
    
    # Get the lookahead point
    lookahead_point = waypoints[lookahead_idx]
    
    # Vector from vehicle to lookahead point (global frame)
    dx = lookahead_point[0] - x
    dy = lookahead_point[1] - y
    
    # Transform to vehicle frame (forward is x, left is y)
    dx_veh = dx * np.cos(-theta) - dy * np.sin(-theta)
    dy_veh = dx * np.sin(-theta) + dy * np.cos(-theta)
    
    # Actual distance to lookahead point
    ld_actual = np.sqrt(dx_veh**2 + dy_veh**2)
    
    # Prevent division by zero
    if ld_actual < 0.1:
        ld_actual = 0.1
    
    # Pure Pursuit steering law: δ = atan(2 * L * sin(α) / Ld)
    # Where α is the angle to the lookahead point in vehicle frame
    # sin(α) ≈ dy_veh / ld_actual for small angles
    # Simplified: curvature = 2 * dy_veh / ld_actual^2
    curvature = 2.0 * dy_veh / (ld_actual ** 2)
    
    # Convert curvature to steering angle using bicycle model
    desired_steering = np.arctan(wheelbase * curvature)
    
    # Add proportional cross-track error correction for better line following on straights
    # Calculate cross-track error (perpendicular distance to closest point)
    closest_point = waypoints[closest_idx]
    
    # Find path direction at closest point
    next_idx = (closest_idx + 1) % len(waypoints)
    path_vector = waypoints[next_idx] - closest_point
    path_length = np.linalg.norm(path_vector)
    
    if path_length > 0.01:
        path_vector = path_vector / path_length  # Normalize
        
        # Vector from closest point to vehicle
        to_vehicle = current_pos - closest_point
        
        # Cross-track error (positive = left of path, negative = right of path)
        cross_track_error = np.cross(path_vector, to_vehicle)
        
        # Add correction term (small gain to avoid fighting main controller)
        k_cte = 0.15  # Cross-track error gain
        steering_correction = -k_cte * cross_track_error
        
        desired_steering += steering_correction
    
    return desired_steering

def compute_desired_velocity(state: ArrayLike, waypoints: ArrayLike, lookahead_distance: float,
                             base_velocity: float = 80.0, max_velocity: float = 100.0):
    """
    Compute desired velocity based on upcoming path curvature using physics-based limits.
    Uses forward-looking approach to anticipate tight corners.

    Args:
        state: Current state [x, y, steering_angle, velocity, heading]
        waypoints: Raceline waypoints
        lookahead_distance: Distance to look ahead (not used in this implementation)
        base_velocity: Base velocity for straight sections
        max_velocity: Maximum allowed velocity

    Returns:
        Desired velocity based on tightest upcoming corner
    """
    current_pos = np.array([state[0], state[1]])
    closest_idx = find_closest_waypoint(current_pos, waypoints)

    # Physics-based parameter: maximum lateral acceleration (in m/s^2)
    # For racing: ~1.5g = 15 m/s^2, for safe driving: ~0.8g = 8 m/s^2
    max_lateral_accel = 12.0  # Tune this based on desired aggressiveness
    
    # Look ahead to find tightest corner
    num_lookahead_points = min(30, len(waypoints))  # Look ahead ~30 waypoints
    min_safe_velocity = base_velocity

    for i in range(num_lookahead_points):
        idx = (closest_idx + i) % len(waypoints)
        idx_prev = (idx - 1) % len(waypoints)
        idx_next = (idx + 1) % len(waypoints)

        # Estimate curvature using three consecutive points
        p1 = waypoints[idx_prev]
        p2 = waypoints[idx]
        p3 = waypoints[idx_next]

        # Menger curvature formula
        # Area of triangle formed by three points
        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                        (p3[0] - p1[0]) * (p2[1] - p1[1]))
        
        # Side lengths of triangle
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)

        if a * b * c > 1e-6:
            curvature = 4 * area / (a * b * c)
        else:
            curvature = 0.0

        # Physics-based velocity limit: v_max = sqrt(a_lat_max / κ)
        # This ensures the car doesn't exceed lateral acceleration limits
        if curvature > 1e-4:  # Only apply limit if curvature is significant
            safe_velocity = np.sqrt(max_lateral_accel / curvature)
        else:
            safe_velocity = base_velocity
        
        # Take the minimum velocity needed for upcoming section
        min_safe_velocity = min(min_safe_velocity, safe_velocity)

    # Apply velocity limits with safety margin
    desired_vel = min(min_safe_velocity, base_velocity)
    
    return np.clip(desired_vel, 15.0, max_velocity)

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

def estimate_upcoming_curvature(state: ArrayLike, waypoints: ArrayLike, preview_points: int = 10):
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
    closest_idx = find_closest_waypoint(current_pos, waypoints)
    
    max_curvature = 0.0
    
    for i in range(min(preview_points, len(waypoints))):
        idx = (closest_idx + i) % len(waypoints)
        idx_prev = (idx - 1) % len(waypoints)
        idx_next = (idx + 1) % len(waypoints)
        
        p1 = waypoints[idx_prev]
        p2 = waypoints[idx]
        p3 = waypoints[idx_next]
        
        # Menger curvature
        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                        (p3[0] - p1[0]) * (p2[1] - p1[1]))
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)
        
        if a * b * c > 1e-6:
            curvature = 4 * area / (a * b * c)
            max_curvature = max(max_curvature, curvature)
    
    return max_curvature

def controller(
    state : ArrayLike, parameters : ArrayLike, racetrack : RaceTrack
) -> ArrayLike:
    """
    High-level controller that computes desired steering angle and velocity.
    Uses dynamic lookahead based on both velocity and upcoming path curvature.

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

    # Dynamic lookahead distance based on velocity AND upcoming curvature
    # Estimate upcoming path curvature
    upcoming_curvature = estimate_upcoming_curvature(state, raceline_data, preview_points=15)
    
    # Base lookahead from velocity (higher speed = look further)
    base_lookahead = 12.0 + current_velocity * 0.4
    
    # Curvature adjustment: high curvature ahead = increase lookahead to prepare earlier
    # Low curvature (straight) = can use shorter lookahead
    if upcoming_curvature > 0.01:  # Significant curve ahead
        # Increase lookahead proportionally to curvature (more curve = look further)
        curvature_factor = 1.0 + min(upcoming_curvature * 20.0, 1.5)  # Up to 2.5x increase
    else:  # Straight section
        curvature_factor = 0.9  # Slightly reduce for efficiency
    
    lookahead_distance = base_lookahead * curvature_factor
    lookahead_distance = np.clip(lookahead_distance, 8.0, 60.0)  # Increased max for tight corners

    # Compute desired steering using Pure Pursuit with dynamic lookahead
    desired_steering = pure_pursuit(state, raceline_data, lookahead_distance, wheelbase)

    # Compute desired velocity based on upcoming curvature (physics-based)
    # Uses forward-looking approach to anticipate corners
    desired_velocity = compute_desired_velocity(
        state, 
        raceline_data, 
        lookahead_distance,
        base_velocity=75.0,  # Target velocity on straights
        max_velocity=100.0   # Absolute maximum
    )

    return np.array([desired_steering, desired_velocity]).T