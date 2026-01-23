### Plan for presentation of double pendulum
## Motivation (no code)
Introduction to topic - why chaotic systems are important to consider (fluid flow, weather, non-ideal gases)
## Physics Background (no code)
Introduction of the coordinate system and analytical method used to get the equation of motion. Explanation of coupled differential equation and why the Lagrangian is a better choice than Newtonian mechanics. Brief exploration of functionals.
## Importance of Numerical Solvers (no code)
The coupled diffeq cannot be solved analytically, we use numerical methods to get an approximate solution.
## Relation to harmonic oscillations
For which initial conditions will our system behave approximately similar to harmonic motion. Ties to the next point.
## Chaos: Sensitivity to initial conditions
How initial conditions affect the outcome of the system (visualization helpful here)
## Different Ways of solving the diffeq (advantage, disadvantages)
We explore the following solvers: explicit euler, implicit euler, rk4, velocity verlet
See how they perform, how the systems behaves after a certain time
## Energy Conservation: How the system behaves with different methods of solvers
Explore how the solvers above conserve energy in the long run.
## Phase Diagram
Explore different visualizations of the 4d phase space.
## Parameter Space exploration
Explore how different values for the parameter change the behaviour of the system (visualization).
## Forced double pendulum (optional)
We do this if we have time : |
