# %% imports
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
# Data for the 26-bus system
# Bus Bus  Voltage Angle   ---Load---- -------Generator----- Injected
# No  code Mag.    Degree  MW    Mvar  MW  Mvar Qmin Qmax     Mvar
busdata = [
    [1,   1,    1.025,   0.0,    51,    41,     0,   0,     0,   0,    4],
    [2,   2,    1.020,   0.0,    22,    15,    79,   0,    40, 250,    0],
    [3,   2,    1.025,   0.0,    64,    50,    20,   0,    40, 150,    0],
    [4,   2,    1.050,   0.0,    25,    10,   100,   0,    25,  80,    2],
    [5,   2,    1.045,   0.0,    50,    30,   300,   0,    40, 160,    5],
    [6,   0,    1.00,    0.0,    76,    29,     0,   0,     0,   0,    2],
    [7,   0,    1.00,    0.0,     0,     0,     0,   0,     0,   0,    0],
    [8,   0,    1.00,    0.0,     0,     0,     0,   0,     0,   0,    0],
    [9,   0,    1.00,    0.0,    89,    50,     0,   0,     0,   0,    3],
    [10,  0,    1.00,    0.0,     0,     0,     0,   0,     0,   0,    0],
    [11,  0,    1.00,    0.0,    25,    15,     0,   0,     0,   0,  1.5],
    [12,  0,    1.00,    0.0,    89,    48,     0,   0,     0,   0,    2],
    [13,  0,    1.00,    0.0,    31,    15,     0,   0,     0,   0,    0],
    [14,  0,    1.00,    0.0,    24,    12,     0,   0,     0,   0,    0],
    [15,  0,    1.00,    0.0,    70,    31,     0,   0,     0,   0,  0.5],
    [16,  0,    1.00,    0.0,    55,    27,     0,   0,     0,   0,    0],
    [17,  0,    1.00,    0.0,    78,    38,     0,   0,     0,   0,    0],
    [18,  0,    1.00,    0.0,   153,    67,     0,   0,     0,   0,    0],
    [19,  0,    1.00,    0.0,    75,    15,     0,   0,     0,   0,    5],
    [20,  0,    1.00,    0.0,    48,    27,     0,   0,     0,   0,    0],
    [21,  0,    1.00,    0.0,    46,    23,     0,   0,     0,   0,    0],
    [22,  0,    1.00,    0.0,    45,    22,     0,   0,     0,   0,    0],
    [23,  0,    1.00,    0.0,    25,    12,     0,   0,     0,   0,    0],
    [24,  0,    1.00,    0.0,    54,    27,     0,   0,     0,   0,    0],
    [25,  0,    1.00,    0.0,    28,    13,     0,   0,     0,   0,    0],
    [26,  2,    1.015,   0.0,    40,    20,    60,   0,    15,  50,    0]
]

# Line data
# Bus bus   R      X     1/2 B   Line code
# nl  nr  p.u.   p.u.   p.u.     = 1 for lines, > 1 or < 1 tr. tap at bus nl
linedata = [
    [1,   2,   0.00055,  0.00480,  0.03000,   1],
    [1,  18,   0.00130,  0.01150,  0.06000,   1],
    [2,   3,   0.00146,  0.05130,  0.05000,   0.96],
    [2,   7,   0.01030,  0.05860,  0.01800,   1],
    [2,   8,   0.00740,  0.03210,  0.03900,   1],
    [2,  13,   0.00357,  0.09670,  0.02500,   0.96],
    [2,  26,   0.03230,  0.19670,  0.00000,   1],
    [3,  13,   0.00070,  0.00548,  0.00050,   1.017],
    [4,   8,   0.00080,  0.02400,  0.00010,   1.050],
    [4,  12,   0.00160,  0.02070,  0.01500,   1.050],
    [5,   6,   0.00690,  0.03000,  0.09900,   1],
    [6,   7,   0.00535,  0.03060,  0.00105,   1],
    [6,  11,   0.00970,  0.05700,  0.00010,   1],
    [6,  18,   0.00374,  0.02220,  0.00120,   1],
    [6,  19,   0.00350,  0.06600,  0.04500,   0.95],
    [6,  21,   0.00500,  0.09000,  0.02260,   1],
    [7,   8,   0.00120,  0.00693,  0.00010,   1],
    [7,   9,   0.00095,  0.04290,  0.02500,   0.95],
    [8,  12,   0.00200,  0.01800,  0.02000,   1],
    [9,  10,   0.00104,  0.04930,  0.00100,   1],
    [10, 12,   0.00247,  0.01320,  0.01000,   1],
    [10, 19,   0.05470,  0.23600,  0.00000,   1],
    [10, 20,   0.00660,  0.01600,  0.00100,   1],
    [10, 22,   0.00690,  0.02980,  0.00500,   1],
    [11, 25,   0.09600,  0.27000,  0.01000,   1],
    [11, 26,   0.01650,  0.09700,  0.00400,   1],
    [12, 14,   0.03270,  0.08020,  0.00000,   1],
    [12, 15,   0.01800,  0.05980,  0.00000,   1],
    [13, 14,   0.00460,  0.02710,  0.00100,   1],
    [13, 15,   0.01160,  0.06100,  0.00000,   1],
    [13, 16,   0.01793,  0.08880,  0.00100,   1],
    [14, 15,   0.00690,  0.03820,  0.00000,   1],
    [15, 16,   0.02090,  0.05120,  0.00000,   1],
    [16, 17,   0.09900,  0.06000,  0.00000,   1],
    [16, 20,   0.02390,  0.05850,  0.00000,   1],
    [17, 18,   0.00320,  0.06000,  0.03800,   1],
    [17, 21,   0.22900,  0.44500,  0.00000,   1],
    [19, 23,   0.03000,  0.13100,  0.00000,   1],
    [19, 24,   0.03000,  0.12500,  0.00200,   1],
    [19, 25,   0.11900,  0.22490,  0.00400,   1],
    [20, 21,   0.06570,  0.15700,  0.00000,   1],
    [20, 22,   0.01500,  0.03660,  0.00000,   1],
    [21, 24,   0.04760,  0.15100,  0.00000,   1],
    [22, 23,   0.02900,  0.09900,  0.00000,   1],
    [22, 24,   0.03100,  0.08800,  0.00000,   1],
    [23, 25,   0.09870,  0.11680,  0.00000,   1]
]

# Cost function matrix (alpha, beta, gamma coefficients)
cost = np.array([
    [240,  7.0,   0.007],
    [200,  10,    0.0095],
    [220,  8.5,   0.009],
    [200,  11,    0.009],
    [220,  10.5,  0.0080],
    [190,  12,    0.0075]
])

# Generator limits in MW
mwlimits = np.array([
    [100, 500],
    [50,  200],
    [80,  300],
    [50,  150],
    [50,  200],
    [50,  120]
])

print("=" * 80)
print("ECONOMIC DISPATCH EXAMPLE WITH B-COEFFICIENTS FOR LOSSES")
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