# %% imports
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import PowerSystem directly
from src.power_system import PowerSystem

# %% functions
def test_economic_dispatch():
    # Book Chapter 7 example 5
    # Create a PowerSystem instance
    ps = PowerSystem()
    
    # Set the input data
    cost = np.array([
        [500, 5.3, 0.004],
        [400, 5.5, 0.006],
        [200, 5.8, 0.009]
    ])
    
    limits = np.array([
        [200, 450],
        [150, 350],
        [100, 225]
    ])
    
    # Set total demand
    Pdt = 800.0
    
    print("Running economic dispatch...")
    
    # Call the dispatch function
    Pgg, lambda_, PL = ps.dispatch(Pdt=Pdt, cost=cost, mwlimits=limits)
    
    # Calculate the total generation cost
    totalcost = ps.gencost(Pgg=Pgg, cost=cost)
    
    # Check results against MATLAB output
    matlab_lambda = 8.5  # From MATLAB output
    matlab_Pgg = np.array([400, 250, 150])  # From MATLAB output
    matlab_cost = 6682.5  # From MATLAB output
    
    print("\nValidation against MATLAB results:")
    print(f"Lambda - Python: {lambda_:.6f}, MATLAB: {matlab_lambda:.6f}, Diff: {abs(lambda_ - matlab_lambda):.6f}")
    print(f"Total cost - Python: {totalcost:.2f}, MATLAB: {matlab_cost:.2f}, Diff: {abs(totalcost - matlab_cost):.2f}")
    print("Generation values:")
    for i, (py_pg, mat_pg) in enumerate(zip(Pgg, matlab_Pgg)):
        print(f"  Gen {i+1} - Python: {py_pg:.1f}, MATLAB: {mat_pg:.1f}, Diff: {abs(py_pg - mat_pg):.1f}")

def test_economic_dispatch_case2():
    # Book Chapter 7 example 6
    # Create a PowerSystem instance
    ps = PowerSystem()
    
    # Set the input data for the second test case
    cost = np.array([
        [500, 5.3, 0.004],
        [400, 5.5, 0.006],
        [200, 5.8, 0.009]
    ])
    
    limits = np.array([
        [200, 450],
        [150, 350],
        [100, 225]
    ])
    
    # Set total demand for second case
    Pdt = 975.0
    
    print("Running economic dispatch for second case (Pdt = 975)...")
    
    # Call the dispatch function
    Pgg, lambda_, PL = ps.dispatch(Pdt=Pdt, cost=cost, mwlimits=limits)
    
    # Calculate the total generation cost
    totalcost = ps.gencost(Pgg=Pgg, cost=cost)
    
    # Check results against MATLAB output for second case
    matlab_lambda = 9.4  # From MATLAB output
    matlab_Pgg = np.array([450, 325, 200])  # From MATLAB output
    matlab_cost = 8236.25  # From MATLAB output
    
    print("\nValidation against MATLAB results (Second Case):")
    print(f"Lambda - Python: {lambda_:.6f}, MATLAB: {matlab_lambda:.6f}, Diff: {abs(lambda_ - matlab_lambda):.6f}")
    print(f"Total cost - Python: {totalcost:.2f}, MATLAB: {matlab_cost:.2f}, Diff: {abs(totalcost - matlab_cost):.2f}")
    print("Generation values:")
    for i, (py_pg, mat_pg) in enumerate(zip(Pgg, matlab_Pgg)):
        print(f"  Gen {i+1} - Python: {py_pg:.1f}, MATLAB: {mat_pg:.1f}, Diff: {abs(py_pg - mat_pg):.1f}")

# %% main
if __name__ == "__main__":
    test_economic_dispatch()
    test_economic_dispatch_case2()

# %%
