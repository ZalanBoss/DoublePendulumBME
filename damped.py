import numpy as np
import matplotlib.pylab as plt
from numpy.linalg import solve
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

import pendulum
from solvers import plot_energy, compute_energy

g = 9.81

def damped_double_pendulum_derivs(t, y, gamma1 = 0.2, gamma2 = 0.2):
    theta1, omega1, theta2, omega2 = y

    A = np.cos(theta2 - theta1)
    B = 2*g*np.sin(theta1) - (omega2**2)*np.sin(theta2-theta1) + gamma1*omega1
    C = g*np.sin(theta2) + (omega1**2)*np.sin(theta2-theta1)+gamma2*omega2

    denominator = 2 - A**2

    alpha1 = (-B +A*C)/denominator
    alpha2 = (A*B - 2*C)/denominator


    return [omega1, alpha1, omega2, alpha2]

def fit_exponential(gammas, relaxation_times):
    gammas = np.array(gammas)
    relaxation_times = np.array(relaxation_times)

    def model(x, a, b, c):
        return a * np.exp(-b * x) + c

    popt, err = curve_fit(model, gammas, relaxation_times, p0=[10, 1, 0], maxfev=5000)
    return popt, err  # (a, b, c)

def energy_vs_gamma(threshold=1/np.e, gamma_max=5, t_max=100, show=True):
    gamma_space = np.linspace(0.01, gamma_max, 150)  

    fig, ax = plt.subplots(figsize=(10, 6))

    gammas = []
    relaxation_times = []

    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, 1000)

    for gamma in gamma_space:
        derivs_gamma = lambda t, y, g=gamma: damped_double_pendulum_derivs(t, y, gamma1=g, gamma2=g)
        sol = solve_ivp(derivs_gamma, t_span, [np.pi/2, 0, np.pi/2, 0], t_eval=t_eval)

        theta1, omega1, theta2, omega2 = sol.y

        E = compute_energy(theta1, omega1, theta2, omega2)
        E_max = np.abs(E[-1])

        # Find time when E/E[0] <= threshold
        for i in range(len(E)):
            if np.abs((E_max+E[i])/(E[0]+E_max)) <= threshold:
                gammas.append(gamma)
                relaxation_times.append(t_eval[i])
                break

    if show: 
        ax.plot(gammas, relaxation_times, label=f"Relaxation time (E/E₀ = {threshold:.2f})")

        
        fit = fit_exponential(gammas, relaxation_times)[0]
        gammas = np.array(gammas)
        ax.plot(gammas, fit[0]*np.exp(-fit[1]*gammas)+fit[2], label=f"Best Fit: τ(γ) = {fit[0]:.1f}·e^(-{fit[1]:.1f}γ) + {fit[2]:.1f}\nError = {fit[1]}")

        ax.set_xlabel("Damping coefficient γ (kg·m²/s)")
        ax.set_ylabel("Relaxation time τ (s)")
        ax.set_title("Energy Relaxation Time vs Damping Coefficient")
        ax.legend()
        ax.grid(True)
        plt.show()
    return gammas, relaxation_times

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Damped double pendulum simulation')
    parser.add_argument('--animate', action='store_true', help='Run animation')
    parser.add_argument('--energy', action='store_true', help='Run energy plots')
    args = parser.parse_args()

    ps = [{'theta1_0': np.pi/4, 'theta2_0': np.pi/4}]

    # If no flags specified, run everything
    run_all = not args.animate and not args.energy

    if args.animate or run_all:
        from animations import animate_pendulum
        animate_pendulum(ps, derivs=damped_double_pendulum_derivs)

    if args.energy or run_all:
        plot_energy(ps, derivs=damped_double_pendulum_derivs)
        gammas, relax_times = energy_vs_gamma()
        fit = fit_exponential(gammas, relax_times)[0]

        print(f"τ(γ) = {fit[0]:.3f}·e^(-{fit[1]:.3f}γ) + {fit[2]:.3f}")


if __name__ == "__main__":
    main()
