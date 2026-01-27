"""Phase space visualization for double pendulum."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from animations import animate_pendulum
from pendulum import L1, L2


def remove_lines(theta):
    """Replace discontinuities (angle wrapping) with NaN to prevent plot artifacts."""
    # indices where the jump is larger than pi 
    diff = np.diff(theta)
    indices = np.where(np.abs(diff) > np.pi)[0]
    
    # change to  NaNs to prevent "jump lines"
    theta_clean = theta.astype(float)
    theta_clean[indices] = np.nan
    return theta_clean


# Initial values
diff = 2.27435  # for chaotic behavior after some time
theta1_0 =  np.pi / diff
theta2_0 =  np.pi / diff
t_max = 200

pendulums = [
    {'theta1_0': theta1_0, 'theta2_0': theta2_0,
        'color': 'green', 'label': 'RK045'},
]
sim = animate_pendulum(pendulums=pendulums,show_animation= True,t_max=t_max)[0]

# Take the angles and angular velocity
theta1 = (sim['theta1'] + np.pi) % (2 * np.pi) - np.pi
theta2 = (sim['theta2'] + np.pi) % (2 * np.pi) - np.pi
omega1 = sim['omega1']
omega2 = sim['omega2']
theta1 = remove_lines(theta1)
theta2 = remove_lines(theta2)

plt.title(r'$\theta_1$ vs $\theta_2$')
plt.xlabel(r'$\theta_1$')
plt.ylabel(r'$\theta_2$')
plt.plot(theta1, theta2)
plt.show()

plt.title(r'$\theta_2$ vs $\omega_2$')
plt.xlabel(r'$\theta_2$')
plt.ylabel(r'$\omega_2$')
plt.plot(theta2, omega2)
plt.show()


def phase_animation():
    """Animate the trajectory in θ₁-θ₂ phase space."""
    figure, ax = plt.subplots(figsize=(6,6))
    ax.set_title("Angular phase")
    ax.set_xlabel("theta_first")
    ax.set_ylabel("theta_second")
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_xlim(np.nanmin(theta1) - 0.1, np.nanmax(theta1) + 0.1)
    ax.set_ylim(np.nanmin(theta2) - 0.1, np.nanmax(theta2) + 0.1)

    #Artists
    line = ax.plot([], [], color='b', lw=1)[0]
    point = ax.plot([],[], color = 'b')[0]

    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point

    def update(frame):
        # FIX 3: Plot everything from index 0 to 'frame' to show the trail
        line.set_data(theta1[:frame], theta2[:frame])
        # Plot the current position as a dot
        point.set_data([theta1[frame]], [theta2[frame]])
        return line, point

    ani = animation.FuncAnimation(figure, update, frames=len(theta1),interval = 2,
                                init_func=init ,blit=True)
    plt.show()

#phase_animation()

# 3d time plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel(r'$\theta_1$')
ax.set_ylabel(r'$\theta_2$')
ax.set_zlabel('Time (s)')
ax.set_title(r'$\theta_1$ vs. $\theta_2$ vs. Time')
ax.plot(theta1, theta2, np.linspace(0, t_max, len(theta1)), lw=0.5)
plt.show()

#3d angular velocity phase dependence
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel(r'$\theta_1$')
ax.set_ylabel(r'$\theta_2$')
ax.set_zlabel(r'$\omega_1$')
ax.set_title(r'$\theta_1$ vs. $\theta_2$ vs. $\omega_1$')
ax.plot(theta1, theta2, omega2)
plt.show()
