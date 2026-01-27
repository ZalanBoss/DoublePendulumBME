import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pendulum import L1, L2, m1, m2, g
from animations import animate_pendulum
from solvers import compute_energy

# Characteristic values
A_drive = 2  # strength of the drive
omega_drive = 0 # frequency of the drive
den = 4
theta1_0 =  np.pi / den
theta2_0 =  np.pi / den

def double_p_d_drive(t,y):
    theta1, omega1, theta2, omega2 = y

    drive = A_drive * np.cos(omega_drive * t)

    # Helper variables
    delta = theta2 - theta1
    den1 = (m1 + m2)*L1 - m2*L1*np.cos(delta)**2
    den2 = (L2/L1) * den1

    # Equations of motion
    ddtheta1 = (m2*L1*omega1**2*np.sin(delta)*np.cos(delta) +
                m2*g*np.sin(theta2)*np.cos(delta) +
                m2*L2*omega2**2*np.sin(delta) -
                (m1 + m2)*g*np.sin(theta1) + 
                drive) / den1 

    ddtheta2 = (-m2*L2*omega2**2*np.sin(delta)*np.cos(delta) +
                (m1 + m2)*(g*np.sin(theta1)*np.cos(delta) -
                L1*omega1**2*np.sin(delta) -
                g*np.sin(theta2))) / den2

    return [omega1, ddtheta1, omega2, ddtheta2]

pendulums = [
    {'theta1_0': theta1_0, 'theta2_0': theta2_0,
        'color': 'green', 'label': 'RK045',
        'deriv_func': double_p_d_drive},
]

# At the resonance freq
omega_drive = 1.77
sim = animate_pendulum(pendulums,t_max=70)

# Frequency sweep to evalute the resonance frequency
frequencies = np.linspace(0.5, 5.0, 50) 
max_energies = []


# Initial conditions
y0 = [theta1_0, 0.0, theta2_0, 0.0] 
t_span = (0, 70)  
t_eval = np.linspace(0, 50, 5000)

for omega in frequencies:
    omega_drive = omega 
    sol = solve_ivp(double_p_d_drive, t_span, y0, t_eval=t_eval, max_step=0.01)
    
    # Calculate energy for this frequency
    energies = compute_energy(sol.y[0], sol.y[1], sol.y[2], sol.y[3])
    max_energies.append(np.max(energies))
    print(f"Finished {round(omega,3)}")

# Plotting the Resonance Curve
plt.figure(figsize=(10, 6))
plt.plot(frequencies, max_energies, marker='o', linestyle='-', color='blue')
plt.title("Resonance Curve: Max Energy vs. Driving Frequency")
plt.xlabel(r"Driving Frequency $\Omega_{drive}$ [rad/s]")
plt.ylabel("Maximum Total Energy reached (J)")
plt.grid(True)
plt.show()

