# Continuous Optimization Framework for Agricultural Coverage Path Planning (CPP)

This repository contains the source code, simulation environment, and experimental data supporting the research paper: **"Continuous-optimization genetic algorithm approach for the Coverage Problem"**.

This project implements a **Real-Coded Genetic Algorithm (GA)** designed to solve the Coverage Path Planning (CPP) problem in unstructured agricultural environments. Unlike traditional discrete approaches, this framework optimizes the spatial placement of waypoints in a continuous domain to maximize field coverage while minimizing operational costs (path length and turns).

## 🚀 Key Features

* **Continuous Space Optimization:** Trajectory generation based on real-valued coordinates rather than discrete grid decomposition.
* **Hybrid Evolutionary Strategy:** Implements a $(\mu+\lambda)$ strategy with BLX-$\alpha$ recombination.
* **Novel Mutation Operator:** Combines Gaussian local refinement with a **Teleportation Mutation** mechanism to escape local optima in complex geometric configurations.
* **Dynamic Penalty Handling:** A robust fitness function that strictly enforcing coverage constraints ($\ge 90\%$) while optimizing efficiency.
* **Visualization Tools:** Automated generation of trajectory maps, convergence profiles, and Pareto front approximations.
