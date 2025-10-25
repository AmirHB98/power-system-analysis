import os
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.power_system import PowerSystem

# 5-bus test system
busdata = [
    [1, 1, 1.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0],
    [2, 2, 1.045, 0.0, 20.0, 10.0, 40.0, 0.0, -40, 50, 0],
    [3, 0, 1.0, 0.0, 45.0, 15.0, 0.0, 0.0, 0, 0, 0],
    [4, 0, 1.0, 0.0, 40.0, 5.0, 0.0, 0.0, 0, 0, 0],
    [5, 0, 1.0, 0.0, 60.0, 10.0, 0.0, 0.0, 0, 0, 0],
]

linedata = [
    [1, 2, 0.02, 0.06, 0.06, 1],
    [1, 3, 0.08, 0.24, 0.05, 1],
    [2, 3, 0.06, 0.18, 0.04, 1],
    [2, 4, 0.06, 0.18, 0.04, 1],
    [2, 5, 0.04, 0.12, 0.03, 1],
    [3, 4, 0.01, 0.03, 0.02, 1],
    [4, 5, 0.08, 0.24, 0.05, 1],
]

ps = PowerSystem()
ps.basemva = 100.0
ps.accuracy = 0.0001
ps.maxiter = 20
ps.load_data(busdata, linedata)
ps.lfybus()
ps.V = np.ones(ps.nbus) + 1j * np.zeros(ps.nbus)
# print(ps.nl)
# print(ps.nr)

# print(np.round(ps.Ybus, 2))

YV = 0 + 1j * 0
n = 1
t1 = time.perf_counter()
for L in range(ps.nbr):
    if ps.nl[L] - 1 == n:
        k = ps.nr[L] - 1
        YV += ps.Ybus[n, k] * ps.V[k]
    elif ps.nr[L] - 1 == n:
        k = ps.nl[L] - 1
        YV += ps.Ybus[n, k] * ps.V[k]

t2 = time.perf_counter()
# print(YV)
print(t2 - t1)
elm_Y = np.delete(ps.Ybus[n, :], n)
YV = np.dot(elm_Y, np.transpose(np.delete(ps.V, n)))
t3 = time.perf_counter()
# print(YV)
print(t3 - t2)
