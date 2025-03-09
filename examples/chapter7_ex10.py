# %% imports
"""
Five-bus system economic dispatch test
Replicating the MATLAB example with B-coefficients for losses
"""
import numpy as np

import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import PowerSystem directly
from src.power_system import PowerSystem

# %% Create power system instance
ps = PowerSystem()

# System parameters
ps.basemva = 100.0
ps.accuracy = 0.0001
ps.maxiter = 10

# %% Data
# Data for the 5-bus system
# Bus Bus  Voltage Angle   ---Load---- -------Generator----- Injected
# No  code Mag.    Degree  MW    Mvar  MW  Mvar Qmin Qmax     Mvar
busdata = [
    [1,   1,    1.06,    0.0,     0,     0,     0,   0,    10,   50,    0],
    [2,   2,    1.045,   0.0,    20,    10,    40,  30,    10,   50,    0],
    [3,   2,    1.03,    0.0,    20,    15,    30,  10,    10,   40,    0],
    [4,   0,    1.00,    0.0,    50,    30,     0,   0,     0,    0,    0],
    [5,   0,    1.00,    0.0,    60,    40,     0,   0,     0,    0,    0]
]

# Line data
# Bus bus   R      X     1/2 B   Line code
# nl  nr  p.u.   p.u.   p.u.     = 1 for lines, > 1 or < 1 tr. tap at bus nl
linedata = [
    [1,   2,   0.02,   0.06,   0.030,   1],
    [1,   3,   0.08,   0.24,   0.025,   1],
    [2,   3,   0.06,   0.18,   0.020,   1],
    [2,   4,   0.06,   0.18,   0.020,   1],
    [2,   5,   0.04,   0.12,   0.015,   1],
    [3,   4,   0.01,   0.03,   0.010,   1],
    [4,   5,   0.08,   0.24,   0.025,   1]
]

# Cost function matrix (alpha, beta, gamma coefficients)
cost = np.array([
    [200,  7.0,   0.008],
    [180,  6.3,   0.009],
    [140,  6.8,   0.007]
])

# Generator limits in MW
mwlimits = np.array([
    [10, 85],
    [10, 80],
    [10, 70]
])

print("=" * 80)
print("FIVE-BUS SYSTEM ECONOMIC DISPATCH TEST WITH B-COEFFICIENTS FOR LOSSES")
print("=" * 80)

# Load the data
ps.load_data(busdata, linedata)

# %% Form the bus admittance matrix
print("\nForming bus admittance matrix (lfybus)...")
ps.lfybus()

# %% Run the power flow solution by Newton-Raphson method
print("\nSolving power flow by Newton-Raphson method (lfnewton)...")
ps.lfnewton()

# %% Print the power flow solution
print("\nInitial power flow solution:")
ps.busout()

# %% Calculate B-coefficients (loss formula)
print("\nCalculating B-coefficients (bloss)...")
ps.bloss()

# %% Compute generation cost with current schedule
print("\nComputing initial generation cost (gencost)...")
initial_cost = ps.gencost(cost=cost)

# %% Obtain the optimum dispatch with loss formula
print("\nPerforming economic dispatch with losses (dispatch)...")
ps.dispatch(Pdt=ps.Pdt, cost=cost, mwlimits=mwlimits)

# %% Iteration until dpslack is within tolerance
print("\nIterating until slack bus power mismatch is within tolerance...")
dpslack = 1.0  # Initialize to a large value
iteration = 0

while dpslack > 0.001:
    iteration += 1
    print(f"\nIteration {iteration}:")
    
    # Run a new power flow solution
    print("Running power flow...")
    ps.lfnewton()
    
    # Update loss coefficients
    print("Updating B-coefficients...")
    ps.bloss()
    
    # Obtain optimum dispatch with new B-coefficients
    print("Performing economic dispatch...")
    ps.dispatch(Pdt=ps.Pdt, cost=cost, mwlimits=mwlimits)
    
    # Get the slack bus and its scheduled power
    for i in range(len(ps.kb)):
        if ps.kb[i] == 1:  # Slack bus
            slack_idx = i
            dpslack = abs(ps.Pg[i] - ps.Pgg[0]) / ps.basemva
            print(f"Slack bus real power mismatch = {dpslack:.4f} pu")
            break

# %% Print the final results
print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

# Print the final power flow solution
print("\nFinal power flow solution:")
ps.busout()

# %% Compute generation cost with optimum scheduling
print("\nFinal generation cost with optimum scheduling:")
final_cost = ps.gencost(cost=cost)

print("\n" + "=" * 80)
print("SUMMARY OF OPTIMIZATION PROCESS")
print("=" * 80)
print(f"Total load demand: {ps.Pdt:.2f} MW")
print(f"Final system loss: {sum(ps.Pgg) - ps.Pdt:.2f} MW")
print(f"Number of optimization iterations: {iteration}")
print(f"Final lambda (incremental cost): {ps.lambda_:.6f} $/MWh")
print(f"Initial generation cost: {initial_cost:.2f} $/h")
print(f"Final generation cost: {final_cost:.2f} $/h")
print(f"Cost reduction: {initial_cost - final_cost:.2f} $/h ({(initial_cost - final_cost)/initial_cost*100:.2f}%)")
print(f"Final generation schedule (MW):")
for i, pg in enumerate(ps.Pgg):
    print(f"  Generator {i+1}: {pg:.2f} MW")

# %%
