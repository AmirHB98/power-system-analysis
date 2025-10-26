# %% Main imports
# Imports
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import PowerSystem and visualization functions
from src.power_position import get_system_positions
from src.power_system import PowerSystem
from src.power_viz import create_bus_table, plot_power_system

# Create power system instance
ps = PowerSystem()

# System parameters
ps.basemva = 100.0
ps.accuracy = 0.001
ps.maxiter = 100
ps.accel = 1.8  # Acceleration factor for Gauss-Seidel

# %% Data
# IEEE 30-BUS TEST SYSTEM (American Electric Power)
# Bus Bus  Voltage Angle   ---Load---- -------Generator----- Injected
# No  code Mag.    Degree  MW    Mvar  MW  Mvar Qmin Qmax     Mvar
busdata = [
    [1, 1, 1.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0],
    [2, 2, 1.043, 0.0, 21.70, 12.7, 40.0, 0.0, -40, 50, 0],
    [3, 0, 1.0, 0.0, 2.4, 1.2, 0.0, 0.0, 0, 0, 0],
    [4, 0, 1.06, 0.0, 7.6, 1.6, 0.0, 0.0, 0, 0, 0],
    [5, 2, 1.01, 0.0, 94.2, 19.0, 0.0, 0.0, -40, 40, 0],
    [6, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0],
    [7, 0, 1.0, 0.0, 22.8, 10.9, 0.0, 0.0, 0, 0, 0],
    [8, 2, 1.01, 0.0, 30.0, 30.0, 0.0, 0.0, -10, 60, 0],
    [9, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0],
    [10, 0, 1.0, 0.0, 5.8, 2.0, 0.0, 0.0, 0, 0, 19],
    [11, 2, 1.082, 0.0, 0.0, 0.0, 0.0, 0.0, -6, 24, 0],
    [12, 0, 1.0, 0.0, 11.2, 7.5, 0.0, 0.0, 0, 0, 0],
    [13, 2, 1.071, 0.0, 0.0, 0.0, 0.0, 0.0, -6, 24, 0],
    [14, 0, 1.0, 0.0, 6.2, 1.6, 0.0, 0.0, 0, 0, 0],
    [15, 0, 1.0, 0.0, 8.2, 2.5, 0.0, 0.0, 0, 0, 0],
    [16, 0, 1.0, 0.0, 3.5, 1.8, 0.0, 0.0, 0, 0, 0],
    [17, 0, 1.0, 0.0, 9.0, 5.8, 0.0, 0.0, 0, 0, 0],
    [18, 0, 1.0, 0.0, 3.2, 0.9, 0.0, 0.0, 0, 0, 0],
    [19, 0, 1.0, 0.0, 9.5, 3.4, 0.0, 0.0, 0, 0, 0],
    [20, 0, 1.0, 0.0, 2.2, 0.7, 0.0, 0.0, 0, 0, 0],
    [21, 0, 1.0, 0.0, 17.5, 11.2, 0.0, 0.0, 0, 0, 0],
    [22, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0],
    [23, 0, 1.0, 0.0, 3.2, 1.6, 0.0, 0.0, 0, 0, 0],
    [24, 0, 1.0, 0.0, 8.7, 6.7, 0.0, 0.0, 0, 0, 4.3],
    [25, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0],
    [26, 0, 1.0, 0.0, 3.5, 2.3, 0.0, 0.0, 0, 0, 0],
    [27, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0],
    [28, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0],
    [29, 0, 1.0, 0.0, 2.4, 0.9, 0.0, 0.0, 0, 0, 0],
    [30, 0, 1.0, 0.0, 10.6, 1.9, 0.0, 0.0, 0, 0, 0],
]

# Line data
# Bus bus   R      X     1/2 B   Line code
# nl  nr  p.u.   p.u.   p.u.     = 1 for lines, > 1 or < 1 tr. tap at bus nl
linedata = [
    [1, 2, 0.0192, 0.0575, 0.02640, 1],
    [1, 3, 0.0452, 0.1852, 0.02040, 1],
    [2, 4, 0.0570, 0.1737, 0.01840, 1],
    [3, 4, 0.0132, 0.0379, 0.00420, 1],
    [2, 5, 0.0472, 0.1983, 0.02090, 1],
    [2, 6, 0.0581, 0.1763, 0.01870, 1],
    [4, 6, 0.0119, 0.0414, 0.00450, 1],
    [5, 7, 0.0460, 0.1160, 0.01020, 1],
    [6, 7, 0.0267, 0.0820, 0.00850, 1],
    [6, 8, 0.0120, 0.0420, 0.00450, 1],
    [6, 9, 0.0, 0.2080, 0.0, 0.978],
    [6, 10, 0, 0.5560, 0, 0.969],
    [9, 11, 0, 0.2080, 0, 1],
    [9, 10, 0, 0.1100, 0, 1],
    [4, 12, 0, 0.2560, 0, 0.932],
    [12, 13, 0, 0.1400, 0, 1],
    [12, 14, 0.1231, 0.2559, 0, 1],
    [12, 15, 0.0662, 0.1304, 0, 1],
    [12, 16, 0.0945, 0.1987, 0, 1],
    [14, 15, 0.2210, 0.1997, 0, 1],
    [16, 17, 0.0824, 0.1923, 0, 1],
    [15, 18, 0.1073, 0.2185, 0, 1],
    [18, 19, 0.0639, 0.1292, 0, 1],
    [19, 20, 0.0340, 0.0680, 0, 1],
    [10, 20, 0.0936, 0.2090, 0, 1],
    [10, 17, 0.0324, 0.0845, 0, 1],
    [10, 21, 0.0348, 0.0749, 0, 1],
    [10, 22, 0.0727, 0.1499, 0, 1],
    [21, 22, 0.0116, 0.0236, 0, 1],
    [15, 23, 0.1000, 0.2020, 0, 1],
    [22, 24, 0.1150, 0.1790, 0, 1],
    [23, 24, 0.1320, 0.2700, 0, 1],
    [24, 25, 0.1885, 0.3292, 0, 1],
    [25, 26, 0.2544, 0.3800, 0, 1],
    [25, 27, 0.1093, 0.2087, 0, 1],
    [28, 27, 0, 0.3960, 0, 0.968],
    [27, 29, 0.2198, 0.4153, 0, 1],
    [27, 30, 0.3202, 0.6027, 0, 1],
    [29, 30, 0.2399, 0.4533, 0, 1],
    [8, 28, 0.0636, 0.2000, 0.0214, 1],
    [6, 28, 0.0169, 0.0599, 0.065, 1],
]

# Load the data
ps.load_data(busdata, linedata)

# Form the bus admittance matrix
ps.lfybus()

# Run the Gauss-Seidel power flow
t1 = time.time()
ps.lfgauss()
t2 = time.time()
# Print the power flow solution
print("\n=========== Gauss-Seidel Method Results ===========")
print("Runtime: ", t2 - t1)
ps.busout()

# Calculate line flows and losses
# ps.lineflow()

# Compare with Newton-Raphson (optional)
print("\n=========== Running Newton-Raphson for Comparison ===========")
ps_newton = PowerSystem()
ps_newton.load_data(busdata, linedata)
ps_newton.lfybus()
t1 = time.time()
ps_newton.newton_raphson()
t2 = time.time()

print("\n=========== Newton-Raphson Method Results ===========")
print("Runtime: ", t2 - t1)
ps_newton.busout()


print("\n=========== Running Decoupled Newton-Raphson for Comparison ===========")
ps_dnewton = PowerSystem()
ps_dnewton.load_data(busdata, linedata)
ps_dnewton.lfybus()
t1 = time.time()
ps_dnewton.nr_decoupled()
t2 = time.time()
print("\n=========== Decoupled Newton-Raphson Method Results ===========")
print("Runtime: ", t2 - t1)
ps_dnewton.busout()


print("\n=========== Running Fast Decoupled Newton-Raphson for Comparison ===========")
ps_fd = PowerSystem()
ps_fd.load_data(busdata, linedata)
ps_fd.lfybus()
t1 = time.time()
ps_fd.fast_decoupled()
t2 = time.time()
print("\n=========== Fast Decoupled Newton-Raphson Method Results ===========")
print("Runtime: ", t2 - t1)
ps_fd.busout()


print("\n=========== Running DC for Comparison ===========")
ps_dc = PowerSystem()
ps_dc.load_data(busdata, linedata)
ps_dc.lfybus()
t1 = time.time()
ps_dc.lfdc()
t2 = time.time()
print("\n=========== DC Method Results ===========")
print("Runtime: ", t2 - t1)
ps_dc.busout()


# Visualization
print("\nCreating visualization...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Get positions
pos = get_system_positions(ps, system_name="ieee30")

# Plot both results
plot_power_system(
    ps, node_positions=pos, ax=ax1, title="IEEE 30-Bus System (Gauss-Seidel)"
)
plot_power_system(
    ps_newton, node_positions=pos, ax=ax2, title="IEEE 30-Bus System (Newton-Raphson)"
)

plt.tight_layout()
plt.show()

print("\nCreating visualization...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Get positions
pos = get_system_positions(ps, system_name="ieee30")

# Plot both results
plot_power_system(
    ps_dnewton,
    node_positions=pos,
    ax=ax1,
    title="IEEE 30-Bus System (Decoupled Newton-Raphson)",
)
plot_power_system(
    ps_fd,
    node_positions=pos,
    ax=ax2,
    title="IEEE 30-Bus System (Fast Decoupled Newton-Raphson)",
)

plt.tight_layout()
plt.show()
# Create tables
bus_df_gs = create_bus_table(ps)
bus_df_nr = create_bus_table(ps_newton)

# Print max absolute difference between the two methods
vm_diff = np.abs(ps.Vm - ps_newton.Vm)
angle_diff = np.abs(ps.deltad - ps_newton.deltad)

print(f"\nMaximum absolute voltage magnitude difference: {np.max(vm_diff):.6f} pu")
print(f"Maximum absolute voltage angle difference: {np.max(angle_diff):.6f} degrees")
print(f"Number of iterations for Gauss-Seidel: {ps.iter}")
print(f"Number of iterations for Newton-Raphson: {ps_newton.iter}")

print("\nProgram completed successfully.")

# %%
