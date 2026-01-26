import numpy as np
from animations import animate_pendulum
import matplotlib.pyplot as plt

#Initial angles
difference = 0.05
theta1_0 = np.pi / 2 + difference
theta2_0 = np.pi / 2 + difference
gamma1_0 = np.pi / 2 
gamma2_0 = np.pi / 2 

pendulums = [
    {'theta1_0': theta1_0, 'theta2_0': theta2_0,
        'color': 'red', 'label': 'Built In (RK45)'},
    {'theta1_0': gamma1_0, 'theta2_0': gamma2_0,
        'color': 'blue', 'label': 'Built In (RK45)'},
]

sim = animate_pendulum(pendulums=pendulums, t_max= 100, interval = 3, show_animation=True)


# For various initial differences
position_difference = []

for diff in np.linspace(0.01, 0.1, num=3, endpoint=True):
    theta1_0 = np.pi / 2 + diff
    theta2_0 = np.pi / 2 + diff
    gamma1_0 = np.pi / 2 
    gamma2_0 = np.pi / 2 
    #print(diff)
    pendulums = [
    {'theta1_0': theta1_0, 'theta2_0': theta2_0,
        'color': 'red', 'label': 'Built In (RK45)'},
    {'theta1_0': gamma1_0, 'theta2_0': gamma2_0,
        'color': 'blue', 'label': 'Built In (RK45)'},
]
    sim = animate_pendulum(pendulums=pendulums, t_max= 100, interval = 3, show_animation=False)
    position_difference.append(np.sqrt((sim[0]['x2'][::5] - sim[1]['x2'][::5])**2 +  \
                               (sim[0]['y2'][::5] - sim[1]['y2'][::5])**2))

# Plotting
plt.figure(figsize=(8,8))
time = np.arange(len(sim[0]['x2'][::5])) * 0.05

for i in range(3):
    plt.plot(time, position_difference[i], label=f"Difference = {round(0.01 + i * 0.045,3)}")

plt.yscale('log')
plt.xlabel("Time")
plt.ylabel("Difference in endpoints on a log scale")
plt.legend()
plt.show()
