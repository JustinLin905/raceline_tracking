import numpy as np
from numpy.typing import ArrayLike
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulator import RaceTrack

from blocks import S1, S2
from helpers import estimate_upcoming_curvature


class C1:
    def __init__(self):
        # Parameters for velocity PID controller
        self.K_p = 3.0
        self.K_i = 0.1
        self.K_d = 0.5
        self.prev_error = 0.0
        self.integral_error = 0.0
        self.max_integral = 5.0

    def step(
        self, state: np.ndarray, parameters: np.ndarray, desired_velocity: float
    ) -> float:
        """
        A PID controller, outputs an acceleration command to reach desired velocity
        Uses adaptive gains based on speed and whether braking or accelerating
        """
        current_velocity = state[3]
        error = desired_velocity - current_velocity

        # Speed-dependent proportional gain
        if error < 0:  # Braking
            if current_velocity < 50:
                K_p = 4.0
            elif current_velocity > 80:
                K_p = 4.0
            else:
                K_p = 3.2 + ((current_velocity - 50) / 30) * 0.8
        else:  # Accelerating
            K_p = self.K_p

        # Integral term with anti-windup
        self.integral_error += error
        self.integral_error = np.clip(self.integral_error, -self.max_integral, self.max_integral)

        # Derivative term
        derivative = error - self.prev_error
        self.prev_error = error

        # PID formula with adaptive Kp
        a = K_p * error + self.K_i * self.integral_error + self.K_d * derivative

        max_acceleration = parameters[10]
        return np.clip(a, -max_acceleration, max_acceleration)


class C2:
    def __init__(self):
        # Parameters for steering PID controller
        self.K_p = 8.2
        self.K_i = 0.2
        self.K_d = 2.2  # Reduced to prevent oscillations in tight corners
        self.prev_error = 0.0
        self.integral_error = 0.0
        self.max_integral = 1.5

        # Derivative filtering to smooth out noise
        self.derivative_filter_alpha = 0.35  # Low-pass filter coefficient
        self.filtered_derivative = 0.0

    def step(
        self, state: np.ndarray, parameters: np.ndarray, desired_steering: float
    ) -> float:
        """
        A PID controller, outputs a steering rate command to reach desired steering angle
        Uses filtered derivative to prevent oscillations in tight corners
        """
        current_steering = state[2]
        steering_error = desired_steering - current_steering

        # Integral term with anti-windup
        self.integral_error += steering_error
        self.integral_error = np.clip(self.integral_error, -self.max_integral, self.max_integral)

        # Derivative term with low-pass filtering to reduce noise
        raw_derivative = steering_error - self.prev_error
        self.filtered_derivative = (
            self.derivative_filter_alpha * raw_derivative +
            (1 - self.derivative_filter_alpha) * self.filtered_derivative
        )
        self.prev_error = steering_error

        # PID formula with filtered derivative
        steering_rate = (
            self.K_p * steering_error +
            self.K_i * self.integral_error +
            self.K_d * self.filtered_derivative
        )

        max_steering_vel = parameters[9]
        return np.clip(steering_rate, -max_steering_vel, max_steering_vel)


# Create instances of blocks
c1 = C1()
c2 = C2()

s1 = S1()
s2 = S2()


def lower_controller(
    state: ArrayLike, desired: ArrayLike, parameters: ArrayLike
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
    assert desired.shape == (2,)

    steering_rate_command = c2.step(state, parameters, desired[0])
    acceleration_command = c1.step(state, parameters, desired[1])

    return np.array([steering_rate_command, acceleration_command]).T


def controller(
    state: ArrayLike, parameters: ArrayLike, racetrack
) -> ArrayLike:
    # Vehicle parameters
    wheelbase = parameters[0]
    current_velocity = state[3]

    # Velocity-based lookahead (simpler is better)
    # Higher speed = look further ahead
    base_lookahead = 15.0 + current_velocity * 0.2
    lookahead_distance = np.clip(base_lookahead, 15.0, 45.0)

    desired_steering = s2.step(
        state, racetrack.centerline, lookahead_distance, wheelbase
    )
    desired_velocity = s1.step(state, racetrack.centerline)

    return np.array([desired_steering, desired_velocity]).T
