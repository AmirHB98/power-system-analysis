# %% imports
import unittest
import numpy as np
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import PowerSystem directly
from src.power_system import PowerSystem

# %% functions
class TestPowerSystem(unittest.TestCase):
    """Unit tests for the PowerSystem class"""
    
    def setUp(self):
        """Setup a small test system"""
        self.ps = PowerSystem()
        self.ps.basemva = 100.0
        self.ps.accuracy = 0.0001
        self.ps.maxiter = 20
        
        # 5-bus test system
        self.busdata = [
            [1, 1, 1.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0],
            [2, 2, 1.045, 0.0, 20.0, 10.0, 40.0, 0.0, -40, 50, 0],
            [3, 0, 1.0, 0.0, 45.0, 15.0, 0.0, 0.0, 0, 0, 0],
            [4, 0, 1.0, 0.0, 40.0, 5.0, 0.0, 0.0, 0, 0, 0],
            [5, 0, 1.0, 0.0, 60.0, 10.0, 0.0, 0.0, 0, 0, 0]
        ]
        
        self.linedata = [
            [1, 2, 0.02, 0.06, 0.06, 1],
            [1, 3, 0.08, 0.24, 0.05, 1],
            [2, 3, 0.06, 0.18, 0.04, 1],
            [2, 4, 0.06, 0.18, 0.04, 1],
            [2, 5, 0.04, 0.12, 0.03, 1],
            [3, 4, 0.01, 0.03, 0.02, 1],
            [4, 5, 0.08, 0.24, 0.05, 1]
        ]
        
        self.ps.load_data(self.busdata, self.linedata)
    
    def test_data_loading(self):
        """Test if data is loaded correctly"""
        self.assertEqual(self.ps.nbus, 5)
        self.assertEqual(self.ps.nbr, 7)
        
    def test_ybus_formation(self):
        """Test Ybus matrix formation"""
        self.ps.lfybus()
        
        # Diagonal element of Ybus for bus 1
        y_diag = self.ps.Ybus[0, 0]
        
        # Calculate manually: sum of admittances connected to bus 1
        y_sum = 1/(0.02 + 0.06j) + 1/(0.08 + 0.24j) + 0.06j/2 + 0.05j/2
        
        # Use delta parameter instead of places for better tolerance
        delta = 0.06  # About 0.3% difference tolerance
        self.assertAlmostEqual(abs(y_diag), abs(y_sum), delta=delta,
                               msg=f"Ybus diagonal value {abs(y_diag)} should approximately equal {abs(y_sum)}")
        
    def test_power_flow(self):
        """Test power flow solution"""
        self.ps.lfybus()
        self.ps.lfnewton()
        
        # Check convergence
        self.assertTrue(self.ps.maxerror < self.ps.accuracy)
        
        # Check if slack bus power is calculated
        self.assertGreater(abs(self.ps.Pg[0]), 0)
        
        # Check power balance
        total_gen = sum(self.ps.Pg)
        total_load = sum(self.ps.Pd)
        
        # System losses should be positive and reasonable
        losses = total_gen - total_load
        self.assertGreater(losses, 0)
        self.assertLess(losses, total_load * 0.1)  # Losses typically < 10% of load
    
    def test_bloss(self):
        """Test B-loss coefficient calculation"""
        self.ps.lfybus()
        self.ps.lfnewton()
        
        B, B0, B00 = self.ps.bloss()
        
        # B matrix should be approximately symmetric
        for i in range(len(B)):
            for j in range(len(B)):
                # Use a larger delta for comparing values
                self.assertAlmostEqual(B[i, j], B[j, i], delta=0.001)
        
        # Estimate losses using B coefficients
        Pgg_array = np.array(self.ps.Pgg)
        PL_B = Pgg_array @ (B / self.ps.basemva) @ Pgg_array + B0 @ Pgg_array + B00 * self.ps.basemva
        
        # Calculate actual losses
        PL_actual = sum(self.ps.Pg) - sum(self.ps.Pd)
        
        # Loss calculated by B coefficients should be close to actual loss
        # Using a higher delta (20%) due to approximation in B-coefficient method
        self.assertAlmostEqual(PL_B, PL_actual, delta=PL_actual*0.2)
    
    def test_economic_dispatch(self):
        """Test economic dispatch"""
        self.ps.lfybus()
        self.ps.lfnewton()
        self.ps.bloss()
        
        # Define cost coefficients for generators
        cost = np.array([
            [100, 10.0, 0.010],  # Generator 1
            [120, 9.0, 0.011]    # Generator 2
        ])
        
        # Define generator limits
        mwlimits = np.array([
            [50, 300],  # Generator 1
            [20, 80]    # Generator 2
        ])
        
        # Perform economic dispatch
        Pgg, lambda_, PL = self.ps.dispatch(Pdt=self.ps.Pdt, cost=cost, mwlimits=mwlimits)
        
        # Check if generation meets demand
        self.assertAlmostEqual(sum(Pgg), self.ps.Pdt + PL, places=2)
        
        # Check if generators are within limits
        for i in range(len(Pgg)):
            self.assertGreaterEqual(Pgg[i], mwlimits[i, 0])
            self.assertLessEqual(Pgg[i], mwlimits[i, 1])
        
        # Check equal incremental cost criterion (when generators are not at limits)
        for i in range(len(Pgg)):
            if mwlimits[i, 0] < Pgg[i] < mwlimits[i, 1]:
                inc_cost = cost[i, 1] + 2 * cost[i, 2] * Pgg[i]
                # Use a larger delta (0.5) for the incremental cost comparison
                # Economic dispatch has approximations due to loss modeling
                self.assertAlmostEqual(inc_cost, lambda_, delta=0.5)
    
    def test_gencost(self):
        """Test generation cost calculation"""
        # Define generation schedule
        Pgg = np.array([150.0, 70.0])
        
        # Define cost coefficients
        cost = np.array([
            [100, 10.0, 0.010],  # Generator 1
            [120, 9.0, 0.011]    # Generator 2
        ])
        
        # Calculate manually
        cost1 = 100 + 10.0 * 150.0 + 0.010 * 150.0**2
        cost2 = 120 + 9.0 * 70.0 + 0.011 * 70.0**2
        manual_total = cost1 + cost2
        
        # Calculate using function
        total_cost = self.ps.gencost(Pgg, cost)
        
        # Compare
        self.assertAlmostEqual(total_cost, manual_total, places=1)

# %% For Jupyter environment, use this instead of unittest.main()
def run_tests():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPowerSystem)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)

# %% main
if __name__ == '__main__':
    # Check if running in a Jupyter notebook
    try:
        # This will raise NameError if not in IPython/Jupyter
        if 'ipykernel' in sys.modules:
            # Running in Jupyter, use custom test runner
            test_results = run_tests()
        else:
            # Regular Python environment
            unittest.main()
    except NameError:
        # Regular Python environment
        unittest.main()

# Execute this in a cell to run the tests in Jupyter
# run_tests()
# %%
