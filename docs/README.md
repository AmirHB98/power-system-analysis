# Power System Analysis Algorithms Documentation

This documentation provides detailed explanations of the power system analysis algorithms implemented in the library. Each algorithm is documented with its mathematical foundation, implementation details, and a flowchart diagram illustrating the algorithm's workflow.

## Table of Contents

### Power Flow Analysis Methods

1. [Newton-Raphson Method](newton_raphson.md)
   - Quadratic convergence rate
   - Forms and solves full Jacobian matrix
   - Suitable for most power systems

2. [Gauss-Seidel Method](gauss_seidel.md)
   - Simple iterative approach
   - Uses acceleration factor for improved convergence
   - Lower computational requirements per iteration

3. [Fast Decoupled Method](fast_decoupled.md)
   - Simplified version of Newton-Raphson
   - Decouples active and reactive power equations
   - Constant Jacobian matrices that need to be inverted only once

4. [Power Perturbation Method](power_perturbation.md)
   - Alternative approach using power injections
   - Directly perturbs power injections to achieve convergence
   - Can handle certain ill-conditioned systems better

### Economic Optimization

1. [Economic Dispatch with Losses](economic_dispatch.md)
   - Optimizes generator outputs to minimize cost
   - Accounts for transmission losses using B-coefficients
   - Handles generator limits and constraints

## Implementation Overview

All algorithms are implemented in the `PowerSystem` class in `src/power_system.py`. The implementation follows the mathematical formulations described in each algorithm's documentation and is based on Prof. Hadi Saadat's approach to power system analysis.

## Usage Examples

Each algorithm documentation includes usage examples showing how to set up and run the algorithm with the `PowerSystem` class. For more comprehensive examples, refer to the example scripts in the `examples/` directory.
