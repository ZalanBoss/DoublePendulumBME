import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pendulum import double_pendulum_derivs, L1, L2


DEFAULT_COLORS = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown']


def animate_pendulum(pendulums, t_max=40, interval=2, trail_length=1000, show_legend=True,
                     title="Double Pendulum Trajectory", show_animation=True, derivs=None):
    if derivs is None:
        derivs = double_pendulum_derivs
    if isinstance(pendulums, dict):
        pendulums = [pendulums]

    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, int(t_max * 100))

    # Simulate all pendulums
    simulations = []
    for i, p in enumerate(pendulums):
        theta1_0 = p['theta1_0']
        theta2_0 = p['theta2_0']
        omega1_0 = p.get('omega1_0', 0.0)
        omega2_0 = p.get('omega2_0', 0.0)
        color = p.get('color', DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
        label = p.get('label', f'Pendulum {i+1}')
        solver = p.get('solver', None)

        y0 = [theta1_0, omega1_0, theta2_0, omega2_0]

        if solver is not None:
            t, y = solver(derivs, t_span, y0, t_eval)
            theta1 = y[0]
            theta2 = y[2]
            omega1 = y[1]
            omega2 = y[3]
        else:
            sol = solve_ivp(derivs, t_span, y0, t_eval=t_eval, max_step=0.01)
            theta1 = sol.y[0]
            omega1 = sol.y[1]
            theta2 = sol.y[2]
            omega2 = sol.y[3]

        # Convert to cartesian coordinates
        x1 = L1 * np.sin(theta1)
        y1 = -L1 * np.cos(theta1)
        x2 = x1 + L2 * np.sin(theta2)
        y2 = y1 - L2 * np.cos(theta2)

        simulations.append({
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'color': color, 'label': label,
            'theta1': theta1, 'theta2': theta2,
            'omega1': omega1, 'omega2': omega2
        })

    if show_animation:
        # Setup plot
        figure, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)
        ax.set_aspect('equal')

        # Fixed axis limits based on max pendulum reach
        max_reach = L1 + L2
        padding = 0.5
        ax.set_xlim(-max_reach - padding, max_reach + padding)
        ax.set_ylim(-max_reach - padding, max_reach + padding)

        # Create plot elements for each pendulum
        artists = []
        for sim in simulations:
            color = sim['color']
            label = sim['label']
            rod_line, = ax.plot([], [], lw=2, color=color, label=label)
            point1, = ax.plot([], [], 'o', color=color, markersize=6)
            point2, = ax.plot([], [], 'o', color=color, markersize=8)
            trail_line, = ax.plot([], [], lw=1, color=color, alpha=0.3)
            artists.append({
                'rod': rod_line, 'point1': point1, 'point2': point2, 'trail': trail_line
            })

        if show_legend:
            ax.legend(loc='upper right')

        all_artists = []
        for a in artists:
            all_artists.extend([a['rod'], a['point1'], a['point2'], a['trail']])

        def init():
            for a in artists:
                a['rod'].set_data([], [])
                a['point1'].set_data([], [])
                a['point2'].set_data([], [])
                a['trail'].set_data([], [])
            return all_artists

        def update(frame):
            for sim, a in zip(simulations, artists):
                x1, y1 = sim['x1'][frame], sim['y1'][frame]
                x2, y2 = sim['x2'][frame], sim['y2'][frame]

                a['rod'].set_data([0, x1, x2], [0, y1, y2])
                a['point1'].set_data([x1], [y1])
                a['point2'].set_data([x2], [y2])

                if frame > trail_length:
                    a['trail'].set_data(sim['x2'][frame-trail_length:frame+1],
                                        sim['y2'][frame-trail_length:frame+1])
                else:
                    a['trail'].set_data(sim['x2'][:frame+1], sim['y2'][:frame+1])

            return all_artists
        
        ani = animation.FuncAnimation(figure, update, frames=len(simulations[0]['x1']),
                                    interval=interval, init_func=init, blit=True)

        plt.show()

    return simulations


def animate_comparison(initial_angle, t_max=30, interval=20, compare_endpoints=False):
    """Animate double pendulum with simple harmonic motion overlay."""
    t = np.linspace(0, t_max, int(t_max * 50))

    # Double pendulum simulation
    sol = solve_ivp(double_pendulum_derivs, (0, t_max),
                    [initial_angle, 0, initial_angle, 0], t_eval=t, max_step=0.01)
    theta1_dp = sol.y[0]
    theta2_dp = sol.y[2]

    # Fit harmonic motion
    if compare_endpoints:
        w = minimize_scalar(lambda w: np.sum((np.array([
            np.sin(theta1_dp) + np.sin(theta2_dp) - 2*np.sin(initial_angle*np.cos(w*t)),
            -np.cos(theta1_dp) - np.cos(theta2_dp) + 2*np.cos(initial_angle*np.cos(w*t))])**2))).x  # type: ignore
    else:
        w = minimize_scalar(lambda w: np.sum((theta1_dp - initial_angle*np.cos(w*t))**2),
                            bounds=(0.1, 15), method="bounded").x  # type: ignore
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
    method = "Endpoint Matching" if compare_endpoints else "Angle Matching"
    ax.set_title(f'Double Pendulum vs Harmonic Motion (θ₀ = {initial_angle:.2f} rad, {method})')

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


