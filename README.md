# Thermosnooker

A 2D thermodynamic billiards simulation with multi-ball collisions, designed for exploring OOP, Kinetic theory, and Brownian-like behaviour


## Features
- Elastic collisions between many balls with accurate kinematics
- Wall collisions and energy/momentum tracking
- Analysis of divergence from Ideal Gas Law
- Optional Brownian-style perturbations for comparison to theory
- Small, readable Python codebase


## Roadmap
- Implementation of Quadtrees or similar approach in next_collision method to remove o(n^2) pair checks
- Use a JIT compilier such as Numba to accelerate runtime when calculating next collisions
- Start analysis of Brownian motion, for example, calculate the mean square displacement of the big ball

## Acknowledgements
This project originated as part of the Advanced Classic Physics module at Imperial College London 2024-25 taught by Alexander Richards.