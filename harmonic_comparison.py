import argparse
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from pendulum import double_pendulum_derivs
from animations import animate_comparison


def harmonic_comparison(initial_angle1, t_max=30, compare_endpoints=False):
    t = np.linspace(0, t_max, 500)
    y = solve_ivp(double_pendulum_derivs, (0, t_max), [initial_angle1, 0, initial_angle1, 0], t_eval=t).y
    theta1, omega1, theta2, omega2 = y

    if compare_endpoints:
        w = minimize_scalar(lambda w: np.sum((np.array([np.sin(theta1) + np.sin(theta2) - 2*np.sin(initial_angle1*np.cos(w*t)), 
                                                        -np.cos(theta2)-np.cos(theta1)+2*np.cos(initial_angle1*np.cos(w*t))])**2))).x # type: ignore
    else:
        w = minimize_scalar(lambda w: np.sum((theta1 - initial_angle1*np.cos(w*t))**2), bounds=(0.1, 15), method="bounded").x # type: ignore

    f = initial_angle1*np.cos(w*t)

    rms = np.sqrt(np.mean((theta1 - f)**2))

    return f, w, rms, t, theta1


angles = np.array([i/100 for i in range(0, 300, 2)])
threshold = 0.1

def plot_rms():
    rms_values = []
    for angle in angles:
        f, w, rms, t, theta1 = harmonic_comparison(angle)
        rms_values.append(rms)

    plt.figure(figsize=(8, 5))
    plt.plot(angles, rms_values, 'b-', linewidth=1.5)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold = {threshold}')
    plt.xlabel('Initial Angle (rad)')
    plt.ylabel('RMS Error (rad)')
    plt.title('RMS Error vs Initial Angle')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Compare double pendulum to harmonic motion')
    parser.add_argument('--animate', action='store_true', help='Show animation instead of RMS plot')
    parser.add_argument('--angle', type=float, default=None, help='Initial angle for animation (default: auto-select good angle)')
    parser.add_argument('--endpoint', action='store_true', help='Use endpoint matching instead of angle matching')
    args = parser.parse_args()

    if args.animate:
        angle = args.angle if args.angle is not None else 0.01
        print(f"Animating with initial angle: {angle:.2f} rad")
        print(f"Method: {'Endpoint matching' if args.endpoint else 'Angle matching'}")
        print(f"RMS {harmonic_comparison(angle, compare_endpoints=args.endpoint)[2]}")
        animate_comparison(angle, compare_endpoints=args.endpoint)
    else:
        plot_rms()


if __name__ == "__main__":
    main()









    



