# %% Main imports
# Imports
import os
import sys

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
    [1, 1, 1.05, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 1.0, 0, 400, 250, 0, 0, 0, 0, 0],
    [3, 2, 1.04, 0, 0, 0, 200, 0, 0, 200, 0],
]

# Line data
# Bus bus   R      X     1/2 B   Line code
# nl  nr  p.u.   p.u.   p.u.     = 1 for lines, > 1 or < 1 tr. tap at bus nl
linedata = [
    [1, 2, 0.02, 0.04, 0, 1],
    [1, 3, 0.01, 0.03, 0, 1],
    [2, 3, 0.0125, 0.025, 0, 1],
]

# Load the data
ps.load_data(busdata, linedata)

# Form the bus admittance matrix
ps.lfybus()

# Run the Gauss-Seidel power flow
ps.lfgauss()

# Print the power flow solution
print("\n=========== Gauss-Seidel Method Results ===========")
ps.busout()

# Calculate line flows and losses
# ps.lineflow()

# Compare with Newton-Raphson (optional)
print("\n=========== Running Newton-Raphson for Comparison ===========")
ps_newton = PowerSystem()
ps_newton.load_data(busdata, linedata)
ps_newton.lfybus()
ps_newton.newton_raphson()  # change it back later
print("\n=========== Newton-Raphson Method Results ===========")
ps_newton.busout()

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
# plt.show()

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
