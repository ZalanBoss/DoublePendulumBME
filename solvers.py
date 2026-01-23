import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from animations import animate_pendulum
from pendulum import double_pendulum_derivs, L1, L2, m1, m2, g


DEFAULT_COLORS = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown']


def explicit_euler(derivs, t_span, y0, t_eval):
    y = np.zeros((4, len(t_eval)))
    y[:, 0] = y0
    for i in range(len(t_eval) - 1):
        dt = t_eval[i+1] - t_eval[i]
        dy = np.array(derivs(t_eval[i], y[:, i]))
        y[:, i+1] = y[:, i] + dt * dy

    return t_eval, y


def implicit_euler(derivs, t_span, y0, t_eval):
    y = np.zeros((4, len(t_eval)))
    y[:, 0] = y0

    # we want to find y_n+1 by F(y_n+1) = yn+1 - yn - dy*dt
    for i in range(len(t_eval)-1):
        dt = t_eval[i+1] - t_eval[i]
        F = lambda new_y: new_y - y[:, i] - dt * np.array(derivs(t_eval[i+1], new_y)) # find the zeros of this function

        y_initial = y[:, i] + dt*np.array(derivs(t_eval[i+1], y[:, i]))
        y[:, i+1] = fsolve(F, y_initial) # type: ignore

    return t_eval, y


def compute_energy(theta1, omega1, theta2, omega2):
    v1x = L1 * omega1 * np.cos(theta1)
    v1y = L1 * omega1 * np.sin(theta1)
    v2x = v1x + L2 * omega2 * np.cos(theta2)
    v2y = v1y + L2 * omega2 * np.sin(theta2)

    T = 0.5 * m1 * (v1x**2 + v1y**2) + 0.5 * m2 * (v2x**2 + v2y**2)

    y1 = -L1 * np.cos(theta1)
    y2 = y1 - L2 * np.cos(theta2)
    U = m1 * g * y1 + m2 * g * y2

    return T + U


def plot_energy(pendulums, t_max=40, title="Energy vs Time"):
    """Plot total energy vs time for multiple pendulum solvers."""
    if isinstance(pendulums, dict):
        pendulums = [pendulums]

    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, int(t_max * 100))

    fig, ax = plt.subplots(figsize=(10, 6))

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
            t, y = solver(double_pendulum_derivs, t_span, y0, t_eval)
            theta1, omega1, theta2, omega2 = y[0], y[1], y[2], y[3]
        else:
            sol = solve_ivp(double_pendulum_derivs, t_span, y0, t_eval=t_eval, max_step=0.01)
            theta1, omega1, theta2, omega2 = sol.y[0], sol.y[1], sol.y[2], sol.y[3]
            t = t_eval

        energy = compute_energy(theta1, omega1, theta2, omega2)
        ax.plot(t, energy, color=color, label=label)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Total Energy (J)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.show()


def main(plot_energy_flag=False):
    # Initial conditions
    theta1_0 = np.pi / 2
    theta2_0 = np.pi / 2

    pendulums = [
        {'theta1_0': theta1_0, 'theta2_0': theta2_0,
         'color': 'blue', 'label': 'Explicit Euler', 'solver': explicit_euler},
        {'theta1_0': theta1_0, 'theta2_0': theta2_0,
         'color': 'red', 'label': 'Implicit Euler', 'solver': implicit_euler},
        {'theta1_0': theta1_0, 'theta2_0': theta2_0,
         'color': 'green', 'label': 'Built In (RK45)'},
    ]

    if plot_energy_flag:
        plot_energy(pendulums, t_max=100, title="Energy Conservation - Solver Comparison")
    else:
        animate_pendulum(pendulums, t_max=20, title="Double Pendulum - Solver Comparison")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--energy', action='store_true', help='Plot energy instead of animation')
    args = parser.parse_args()
    main(plot_energy_flag=args.energy)
