# %% imports
import numpy as np
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import PowerSystem directly
from src.power_system import PowerSystem

# %% Set system parameters
ps = PowerSystem()

ps.basemva = 100.0
ps.accuracy = 0.0001
ps.maxiter = 10

# %% Bus data
#        Bus Bus  Voltage Angle   ---Load---- -------Generator----- Injected
#        No  code Mag.    Degree  MW    Mvar  MW  Mvar Qmin Qmax    Mvar
busdata = [
    [1, 1, 1.06, 0.0, 0, 0, 0, 0, 10, 50, 0],
    [2, 2, 1.045, 0.0, 20, 10, 40, 30, 10, 50, 0],
    [3, 2, 1.03, 0.0, 20, 15, 30, 10, 10, 40, 0],
    [4, 0, 1.00, 0.0, 50, 30, 0, 0, 0, 0, 0],
    [5, 0, 1.00, 0.0, 60, 40, 0, 0, 0, 0, 0]
]

# Line data
#         Bus bus   R      X     1/2 B   Line code
#         nl  nr  p.u.   p.u.   p.u.     = 1 for lines, > 1 or < 1 tr. tap at bus nl
linedata = [
    [1, 2, 0.02, 0.06, 0.030, 1],
    [1, 3, 0.08, 0.24, 0.025, 1],
    [2, 3, 0.06, 0.18, 0.020, 1],
    [2, 4, 0.06, 0.18, 0.020, 1],
    [2, 5, 0.04, 0.12, 0.015, 1],
    [3, 4, 0.01, 0.03, 0.010, 1],
    [4, 5, 0.08, 0.24, 0.025, 1]
]

# Load the data
ps.load_data(busdata, linedata)

# Form the bus admittance matrix
ps.lfybus()

# %% Run the Newton-Raphson power flow
ps.lfnewton()

# Print the power flow solution
ps.busout()

# %% Obtain the loss formula coefficients
B, B0, B00 = ps.bloss()

# %% Print the total loss from the B coefficients
Pgg_array = np.array(ps.Pgg)
PL = Pgg_array @ (B / ps.basemva) @ Pgg_array + B0 @ Pgg_array + B00 * ps.basemva
print(f"Total system loss calculated from B coefficients = {PL:.5f} MW")

# %%
