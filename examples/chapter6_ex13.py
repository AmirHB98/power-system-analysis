# chapter6_decouple_ex11.py - Test Fast Decoupled method with IEEE 30-bus system

# %% Imports
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import PowerSystem and visualization functions
from src.power_system import PowerSystem
from src.power_viz import plot_power_system, create_bus_table, create_line_table
from src.power_position import get_ieee30_positions, get_system_positions

# Create power system instance
ps = PowerSystem()

# System parameters - identical to MATLAB
ps.basemva = 100.0
ps.accuracy = 0.001
ps.maxiter = 50

# IEEE 30-BUS TEST SYSTEM (American Electric Power)
# Bus Bus  Voltage Angle   ---Load---- -------Generator----- Injected
# No  code Mag.    Degree  MW    Mvar  MW  Mvar Qmin Qmax     Mvar
busdata = [
    [1,   1,    1.06,    0.0,     0.0,   0.0,    0.0,  0.0,   0,   0,       0],
    [2,   2,    1.043,   0.0,   21.70,  12.7,   40.0,  0.0, -40,  50,       0],
    [3,   0,    1.0,     0.0,     2.4,   1.2,    0.0,  0.0,   0,   0,       0],
    [4,   0,    1.06,    0.0,     7.6,   1.6,    0.0,  0.0,   0,   0,       0],
    [5,   2,    1.01,    0.0,    94.2,  19.0,    0.0,  0.0, -40,  40,       0],
    [6,   0,    1.0,     0.0,     0.0,   0.0,    0.0,  0.0,   0,   0,       0],
    [7,   0,    1.0,     0.0,    22.8,  10.9,    0.0,  0.0,   0,   0,       0],
    [8,   2,    1.01,    0.0,    30.0,  30.0,    0.0,  0.0, -10,  40,       0],
    [9,   0,    1.0,     0.0,     0.0,   0.0,    0.0,  0.0,   0,   0,       0],
    [10,  0,    1.0,     0.0,     5.8,   2.0,    0.0,  0.0,  -6,  24,      19],
    [11,  2,    1.082,   0.0,     0.0,   0.0,    0.0,  0.0,   0,   0,       0],
    [12,  0,    1.0,     0.0,    11.2,   7.5,    0.0,  0.0,   0,   0,       0],
    [13,  2,    1.071,   0.0,     0.0,   0.0,    0.0,  0.0,  -6,  24,       0],
    [14,  0,    1.0,     0.0,     6.2,   1.6,    0.0,  0.0,   0,   0,       0],
    [15,  0,    1.0,     0.0,     8.2,   2.5,    0.0,  0.0,   0,   0,       0],
    [16,  0,    1.0,     0.0,     3.5,   1.8,    0.0,  0.0,   0,   0,       0],
    [17,  0,    1.0,     0.0,     9.0,   5.8,    0.0,  0.0,   0,   0,       0],
    [18,  0,    1.0,     0.0,     3.2,   0.9,    0.0,  0.0,   0,   0,       0],
    [19,  0,    1.0,     0.0,     9.5,   3.4,    0.0,  0.0,   0,   0,       0],
    [20,  0,    1.0,     0.0,     2.2,   0.7,    0.0,  0.0,   0,   0,       0],
    [21,  0,    1.0,     0.0,    17.5,  11.2,    0.0,  0.0,   0,   0,       0],
    [22,  0,    1.0,     0.0,     0.0,   0.0,    0.0,  0.0,   0,   0,       0],
    [23,  0,    1.0,     0.0,     3.2,   1.6,    0.0,  0.0,   0,   0,       0],
    [24,  0,    1.0,     0.0,     8.7,   6.7,    0.0,  0.0,   0,   0,      4.3],
    [25,  0,    1.0,     0.0,     0.0,   0.0,    0.0,  0.0,   0,   0,       0],
    [26,  0,    1.0,     0.0,     3.5,   2.3,    0.0,  0.0,   0,   0,       0],
    [27,  0,    1.0,     0.0,     0.0,   0.0,    0.0,  0.0,   0,   0,       0],
    [28,  0,    1.0,     0.0,     0.0,   0.0,    0.0,  0.0,   0,   0,       0],
    [29,  0,    1.0,     0.0,     2.4,   0.9,    0.0,  0.0,   0,   0,       0],
    [30,  0,    1.0,     0.0,    10.6,   1.9,    0.0,  0.0,   0,   0,       0]
]

# Line data - exactly matching MATLAB
# Bus bus   R      X     1/2 B   Line code
# nl  nr  p.u.   p.u.   p.u.     = 1 for lines, > 1 or < 1 tr. tap at bus nl
linedata = [
    [1,   2,   0.0192,   0.0575,   0.02640,    1],
    [1,   3,   0.0452,   0.1852,   0.02040,    1],
    [2,   4,   0.0570,   0.1737,   0.01840,    1],
    [3,   4,   0.0132,   0.0379,   0.00420,    1],
    [2,   5,   0.0472,   0.1983,   0.02090,    1],
    [2,   6,   0.0581,   0.1763,   0.01870,    1],
    [4,   6,   0.0119,   0.0414,   0.00450,    1],
    [5,   7,   0.0460,   0.1160,   0.01020,    1],
    [6,   7,   0.0267,   0.0820,   0.00850,    1],
    [6,   8,   0.0120,   0.0420,   0.00450,    1],
    [6,   9,   0.0,      0.2080,   0.0,     0.978],
    [6,  10,   0,         .5560,   0,       0.969],
    [9,  11,   0,         .2080,   0,           1],
    [9,  10,   0,         .1100,   0,           1],
    [4,  12,   0,         .2560,   0,       0.932],
    [12, 13,   0,         .1400,   0,           1],
    [12, 14,   .1231,     .2559,   0,           1],
    [12, 15,   .0662,     .1304,   0,           1],
    [12, 16,   .0945,     .1987,   0,           1],
    [14, 15,   .2210,     .1997,   0,           1],
    [16, 17,   .0824,     .1923,   0,           1],
    [15, 18,   .1073,     .2185,   0,           1],
    [18, 19,   .0639,     .1292,   0,           1],
    [19, 20,   .0340,     .0680,   0,           1],
    [10, 20,   .0936,     .2090,   0,           1],
    [10, 17,   .0324,     .0845,   0,           1],
    [10, 21,   .0348,     .0749,   0,           1],
    [10, 22,   .0727,     .1499,   0,           1],
    [21, 22,   .0116,     .0236,   0,           1],
    [15, 23,   .1000,     .2020,   0,           1],
    [22, 24,   .1150,     .1790,   0,           1],
    [23, 24,   .1320,     .2700,   0,           1],
    [24, 25,   .1885,     .3292,   0,           1],
    [25, 26,   .2544,     .3800,   0,           1],
    [25, 27,   .1093,     .2087,   0,           1],
    [28, 27,   0,         .3960,   0,       0.968],
    [27, 29,   .2198,     .4153,   0,           1],
    [27, 30,   .3202,     .6027,   0,           1],
    [29, 30,   .2399,     .4533,   0,           1],
    [8,  28,   .0636,     .2000,   0.0214,      1],
    [6,  28,   .0169,     .0599,   0.065,      1]
]

# Follow same sequence as MATLAB
# 1. Load the data
ps.load_data(busdata, linedata)

# 2. Form the bus admittance matrix
ps.lfybus()

# 3. Run the Fast Decoupled power flow
ps.decouple()

# 4. Print the power flow solution
print("\n=========== Fast Decoupled Method Results ===========")
ps.busout()

# 5. Calculate line flows and losses
ps.lineflow()

# Add visualization
# print("\nCreating visualization...")
# fig, ax = plt.subplots(figsize=(12, 10))

# # Get positions for visualization
# pos = get_system_positions(ps, system_name='ieee30')
# if pos is None:
#     pos = get_ieee30_positions()

# # Plot the system
# plot_power_system(ps, node_positions=pos, ax=ax, 
#                  title="IEEE 30-Bus System (Fast Decoupled Method)")

# plt.tight_layout()
# plt.savefig("ieee30_fast_decoupled.png", dpi=300)
# plt.show()

# Print iteration count and convergence information
print(f"\nNumber of iterations: {ps.iter}")
print(f"Maximum power mismatch: {ps.maxerror}")

# Create tables for detailed analysis
bus_df = create_bus_table(ps)
line_df = create_line_table(ps)

print("\nProgram completed successfully.")

# %%
