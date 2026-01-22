import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pendulum import double_pendulum_derivs, L1, L2


def animate_pendulum(theta1_0, theta2_0, omega1_0=0.0, omega2_0=0.0, t_max=40, interval=2):
    """
    Animate a double pendulum with given initial conditions.

    Parameters:
        theta1_0: initial angle of first pendulum (rad)
        theta2_0: initial angle of second pendulum (rad)
        omega1_0: initial angular velocity of first pendulum (rad/s)
        omega2_0: initial angular velocity of second pendulum (rad/s)
        t_max: simulation time (s)
        interval: animation interval in ms (lower = faster)
    """
    y0 = [theta1_0, omega1_0, theta2_0, omega2_0]

    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, int(t_max * 100))

    sol = solve_ivp(double_pendulum_derivs, t_span, y0, t_eval=t_eval, max_step=0.01)

    theta1 = sol.y[0]
    theta2 = sol.y[2]

    # Convert to cartesian coordinates
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)

    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)

    # plotting
    figure, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Double Pendulum Trajectory")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_xlim(-L1 - L2 - 0.5, L1 + L2 + 0.5)
    ax.set_ylim(-L1 - L2 - 0.5, L1 + L2 + 0.5)

    rod_lines = ax.plot([], [], lw=2)[0]
    point1 = ax.plot([], [], 'o', color="k", markersize=2)[0]
    point2 = ax.plot([], [], 'o', color="k", markersize=2)[0]
    trail_line = ax.plot([], [], lw=1, color='c', alpha=0.3)[0]

    def init():
        rod_lines.set_data([], [])
        point1.set_data([], [])
        point2.set_data([], [])
        trail_line.set_data([], [])
        return rod_lines, point1, point2, trail_line

    def update(frame):
        x0 = y0_coord = 0
        x_1, y_1 = x1[frame], y1[frame]
        x_2, y_2 = x2[frame], y2[frame]
        rod_lines.set_data([x0, x_1, x_2], [y0_coord, y_1, y_2])
        point1.set_data([x_1], [y_1])
        point2.set_data([x_2], [y_2])
        trail_length = 1000
        if frame > trail_length:
            trail_line.set_data(x2[frame-trail_length:frame+1], y2[frame-trail_length:frame+1])
        else:
            trail_line.set_data(x2[:frame+1], y2[:frame+1])
        return rod_lines, point1, point2, trail_line

    ani = animation.FuncAnimation(figure, update, frames=len(x2), interval=interval,
                                  init_func=init, blit=True)

    plt.show()


def animate_comparison(initial_angle, t_max=30, interval=20):
    """Animate double pendulum with simple harmonic motion overlay."""
    t = np.linspace(0, t_max, int(t_max * 50))

    # Double pendulum simulation
    sol = solve_ivp(double_pendulum_derivs, (0, t_max),
                    [initial_angle, 0, initial_angle, 0], t_eval=t, max_step=0.01)
    theta1_dp = sol.y[0]
    theta2_dp = sol.y[2]

    # Fit harmonic motion
    w = minimize_scalar(lambda w: np.sum((theta1_dp - initial_angle*np.cos(w*t))**2),
                        bounds=(0.1, 15), method="bounded").x
    theta_harmonic = initial_angle * np.cos(w * t)

    # Double pendulum cartesian coordinates
    x1_dp = L1 * np.sin(theta1_dp)
    y1_dp = -L1 * np.cos(theta1_dp)
    x2_dp = x1_dp + L2 * np.sin(theta2_dp)
    y2_dp = y1_dp - L2 * np.cos(theta2_dp)

    # Simple harmonic pendulum cartesian (single pendulum, length L1 + L2)
    L_total = L1 + L2
    x_harm = L_total * np.sin(theta_harmonic)
    y_harm = -L_total * np.cos(theta_harmonic)

    # Calculate tight bounds including origin and all motion
    all_x = np.concatenate([[0], x1_dp, x2_dp, x_harm])
    all_y = np.concatenate([[0], y1_dp, y2_dp, y_harm])
    padding = 0.2
    x_min, x_max = all_x.min() - padding, all_x.max() + padding
    y_min, y_max = all_y.min() - padding, all_y.max() + padding

    # Setup plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title(f'Double Pendulum vs Harmonic Motion (θ₀ = {initial_angle:.2f} rad)')

    # Double pendulum (blue)
    dp_rod, = ax.plot([], [], 'b-', lw=2, label='Double Pendulum')
    dp_mass, = ax.plot([], [], 'bo', markersize=10)

    # Harmonic pendulum (red)
    harm_rod, = ax.plot([], [], 'r--', lw=2, label='Harmonic Motion')
    harm_mass, = ax.plot([], [], 'ro', markersize=10)

    ax.legend(loc='upper right')

    def init():
        dp_rod.set_data([], [])
        dp_mass.set_data([], [])
        harm_rod.set_data([], [])
        harm_mass.set_data([], [])
        return dp_rod, dp_mass, harm_rod, harm_mass

    def update(frame):
        # Double pendulum
        dp_rod.set_data([0, x1_dp[frame], x2_dp[frame]], [0, y1_dp[frame], y2_dp[frame]])
        dp_mass.set_data([x2_dp[frame]], [y2_dp[frame]])

        # Harmonic pendulum
        harm_rod.set_data([0, x_harm[frame]], [0, y_harm[frame]])
        harm_mass.set_data([x_harm[frame]], [y_harm[frame]])

        return dp_rod, dp_mass, harm_rod, harm_mass

    ani = animation.FuncAnimation(fig, update, frames=len(t), interval=interval,
                                  init_func=init, blit=True)
    plt.show()
