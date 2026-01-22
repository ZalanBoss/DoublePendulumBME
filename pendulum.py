import numpy as np

# physical values
m1 = 1.0   # mass of first pendulum
m2 = 1.0   # mass of second pendulum
L1 = 1.0   # length of first rod
L2 = 1.0   # length of second rod
g  = 9.81  # gravity


def double_pendulum_derivs(t, y):
    theta1, omega1, theta2, omega2 = y

    # Helper variables
    delta = theta2 - theta1
    den1 = (m1 + m2)*L1 - m2*L1*np.cos(delta)*np.cos(delta)
    den2 = (L2/L1) * den1

    # Equations of motion
    ddtheta1 = (m2*L1*omega1**2*np.sin(delta)*np.cos(delta) +
                m2*g*np.sin(theta2)*np.cos(delta) +
                m2*L2*omega2**2*np.sin(delta) -
                (m1 + m2)*g*np.sin(theta1)) / den1

    ddtheta2 = (-m2*L2*omega2**2*np.sin(delta)*np.cos(delta) +
                (m1 + m2)*(g*np.sin(theta1)*np.cos(delta) -
                L1*omega1**2*np.sin(delta) -
                g*np.sin(theta2))) / den2

    return [omega1, ddtheta1, omega2, ddtheta2]


def main():
    from animations import animate_pendulum
    animate_pendulum(np.pi/2, np.pi/2 + 0.01)


if __name__ == "__main__":
    main()

