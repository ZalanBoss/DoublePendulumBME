"""Demonstrate sensitivity to initial conditions (chaos)."""
import numpy as np
import matplotlib.pyplot as plt

from animations import animate_pendulum

# Initial angles
difference = 0.01
theta1_0 = np.pi / 2 + difference
theta2_0 = np.pi / 2 + difference
theta3_0 = np.pi / 2 + difference / 10
theta4_0 = np.pi / 2 + difference / 10
theta_ref_1 = np.pi / 2 
theta_ref_2 = np.pi / 2 

pendulums = [
    {'theta1_0': theta1_0, 'theta2_0': theta2_0,
        'color': 'red', 'label': f'pi / 2 + {difference}'},
    {'theta1_0': theta_ref_1, 'theta2_0': theta_ref_2,
        'color': 'blue', 'label': f'pi / 2'},
    {'theta1_0': theta3_0, 'theta2_0': theta4_0,
        'color': 'green', 'label': f'pi / 2 + {difference / 10}'},
]

sim = animate_pendulum(pendulums=pendulums, t_max= 100, interval = 3, show_animation=True)



# For various initial differences
position_difference = []
starting_separation = 0.001
max_initial_separation = 0.1
num = 3
for diff in np.linspace(starting_separation, max_initial_separation, num=num, endpoint=True):
    theta1_0 = np.pi / 2 + diff
    theta2_0 = np.pi / 2 + diff
    theta_ref_1 = np.pi / 2 
    theta_ref_2 = np.pi / 2 
    #print(diff)
    pendulums = [
    {'theta1_0': theta1_0, 'theta2_0': theta2_0,
        'color': 'red', 'label': 'Built In (RK45)'},
    {'theta1_0': theta_ref_1, 'theta2_0': theta_ref_2,
        'color': 'blue', 'label': 'Built In (RK45)'},
]
    sim = animate_pendulum(pendulums=pendulums, t_max= 100, interval = 2, show_animation=False)
    position_difference.append(np.sqrt((sim[0]['x2'][::5] - sim[1]['x2'][::5])**2 +  \
                               (sim[0]['y2'][::5] - sim[1]['y2'][::5])**2))

# Plotting
plt.figure(figsize=(8,8))
time = np.arange(len(sim[0]['x2'][::5])) * 0.05
step = (max_initial_separation - starting_separation) / (num - 1)
for i in range(num):
    plt.plot(time, position_difference[i], label=f"Difference = {round(starting_separation +  i * step,4)}")

plt.yscale('log')
plt.xlabel("Time")
plt.ylabel("Difference in endpoint on a log scale")
plt.legend()
plt.show()
