import numpy as np
from numpy import zeros, ones, conj, pi, sin, cos, sqrt, abs, angle, max, argmax, sum, real, imag
import pandas as pd
from typing import Tuple, List, Dict, Optional

class PowerSystem:
    def __init__(self):
        # System parameters
        self.basemva = 100.0
        self.accuracy = 0.001
        self.maxiter = 10
        
        # Bus data
        self.nbus = 0
        self.busdata = None
        self.Vm = None
        self.delta = None
        self.deltad = None
        self.V = None
        self.P = None
        self.Q = None
        self.S = None
        self.Pd = None
        self.Qd = None
        self.Pg = None
        self.Qg = None
        self.Qmin = None
        self.Qmax = None
        self.Qsh = None
        self.kb = None
        self.Pgg = None
        self.Qgg = None
        self.yload = None
        
        # Line data
        self.linedata = None
        self.nl = None
        self.nr = None
        self.R = None
        self.X = None
        self.Bc = None
        self.a = None
        self.nbr = 0
        self.y = None
        self.Ybus = None
        
        # System totals
        self.Pdt = 0
        self.Qdt = 0
        self.Pgt = 0
        self.Qgt = 0
        self.Qsht = 0
        
        # Loss coefficients
        self.B = None
        self.B0 = None
        self.B00 = None
        
        # Solution info
        self.iter = 0
        self.maxerror = 0
        self.tech = ""
        
    def load_data(self, busdata=None, linedata=None):
        """Load bus and line data"""
        if busdata is not None:
            self.busdata = np.array(busdata, dtype=float)
            self.nbus = len(self.busdata)
        
        if linedata is not None:
            self.linedata = np.array(linedata, dtype=float)
            self.nbr = len(self.linedata)
            
    def lfybus(self):
        """This function forms the bus admittance matrix for power flow solution"""
        # Complex number j
        j = 1j
        
        # Extract data from linedata
        self.nl = self.linedata[:, 0].astype(int)
        self.nr = self.linedata[:, 1].astype(int)
        self.R = self.linedata[:, 2]
        self.X = self.linedata[:, 3]
        self.Bc = j * self.linedata[:, 4]
        self.a = self.linedata[:, 5]
        
        # Number of buses and branches
        self.nbr = len(self.linedata)
        self.nbus = max(np.maximum(self.nl, self.nr))
        
        # Set tap to 1 where tap is 0
        self.a = np.where(self.a <= 0, 1.0, self.a)
        
        # Calculate branch admittance
        Z = self.R + j * self.X
        self.y = 1.0 / Z
        
        # Initialize Ybus to zero
        self.Ybus = zeros((int(self.nbus), int(self.nbus)), dtype=complex)
        
        # Formation of the off-diagonal elements
        for k in range(self.nbr):
            nl_idx = int(self.nl[k] - 1)  # -1 for 0-indexed arrays
            nr_idx = int(self.nr[k] - 1)
            self.Ybus[nl_idx, nr_idx] -= self.y[k] / self.a[k]
            self.Ybus[nr_idx, nl_idx] = self.Ybus[nl_idx, nr_idx]
        
        # Formation of the diagonal elements
        for n in range(1, int(self.nbus) + 1):
            n_idx = n - 1  # 0-indexed
            for k in range(self.nbr):
                if self.nl[k] == n:
                    self.Ybus[n_idx, n_idx] += self.y[k] / (self.a[k] ** 2) + self.Bc[k]
                elif self.nr[k] == n:
                    self.Ybus[n_idx, n_idx] += self.y[k] + self.Bc[k]
        
        # Clear Pgg as was done in MATLAB code
        self.Pgg = None
        
    def lfnewton(self):
        """Power flow solution by Newton-Raphson method"""
        ns = 0  # Number of slack buses
        ng = 0  # Number of generator (PV) buses
        self.Vm = zeros(int(self.nbus))
        self.delta = zeros(int(self.nbus))
        self.yload = zeros(int(self.nbus), dtype=complex)
        self.deltad = zeros(int(self.nbus))
        self.kb = zeros(int(self.nbus), dtype=int)
        
        nbus_int = int(self.nbus)
        
        # Initialize arrays
        self.P = zeros(nbus_int)
        self.Q = zeros(nbus_int)
        self.V = zeros(nbus_int, dtype=complex)
        self.S = zeros(nbus_int, dtype=complex)
        self.Pd = zeros(nbus_int)
        self.Qd = zeros(nbus_int)
        self.Pg = zeros(nbus_int)
        self.Qg = zeros(nbus_int)
        self.Qmin = zeros(nbus_int)
        self.Qmax = zeros(nbus_int)
        self.Qsh = zeros(nbus_int)
        
        # Process bus data
        for k in range(len(self.busdata)):
            n = int(self.busdata[k, 0])
            n_idx = n - 1  # 0-indexed array
            
            self.kb[n_idx] = int(self.busdata[k, 1])
            self.Vm[n_idx] = self.busdata[k, 2]
            self.delta[n_idx] = self.busdata[k, 3]
            self.Pd[n_idx] = self.busdata[k, 4]
            self.Qd[n_idx] = self.busdata[k, 5]
            self.Pg[n_idx] = self.busdata[k, 6]
            self.Qg[n_idx] = self.busdata[k, 7]
            self.Qmin[n_idx] = self.busdata[k, 8]
            self.Qmax[n_idx] = self.busdata[k, 9]
            self.Qsh[n_idx] = self.busdata[k, 10]
            
            if self.Vm[n_idx] <= 0:
                self.Vm[n_idx] = 1.0
                self.V[n_idx] = 1.0 + 0j
            else:
                self.delta[n_idx] = pi / 180 * self.delta[n_idx]
                self.V[n_idx] = self.Vm[n_idx] * (cos(self.delta[n_idx]) + 1j * sin(self.delta[n_idx]))
                self.P[n_idx] = (self.Pg[n_idx] - self.Pd[n_idx]) / self.basemva
                self.Q[n_idx] = (self.Qg[n_idx] - self.Qd[n_idx] + self.Qsh[n_idx]) / self.basemva
                self.S[n_idx] = self.P[n_idx] + 1j * self.Q[n_idx]
                
        # Count bus types
        nss = zeros(nbus_int, dtype=int)  # Cumulative number of slack buses
        ngs = zeros(nbus_int, dtype=int)  # Cumulative number of generator buses
        
        for k in range(nbus_int):
            if self.kb[k] == 1:
                ns += 1
            if self.kb[k] == 2:
                ng += 1
            ngs[k] = ng
            nss[k] = ns
            
        # Prepare for Newton-Raphson
        Ym = abs(self.Ybus)
        t = angle(self.Ybus)
        
        # Number of equations: 2*nbus - ng - 2*ns
        m = 2 * nbus_int - ng - 2 * ns
        
        # Initialize iteration
        self.maxerror = 1.0
        converge = 1
        self.iter = 0
        
        # Start of iterations
        while self.maxerror >= self.accuracy and self.iter <= self.maxiter:
            # Initialize Jacobian matrix for each iteration
            A = zeros((m, m))
            
            self.iter += 1
            
            # Build Jacobian matrix and compute mismatches
            DC = zeros(m)  # Mismatch vector
            nn_index = 0  # Index for equations related to P mismatches
            lm_index = 0  # Index for equations related to Q mismatches
            
            for n in range(nbus_int):
                nn = n - nss[n]  # Adjust for slack buses for P mismatch
                lm = nbus_int + n - ngs[n] - nss[n] - ns  # Adjust for Q mismatch
                
                # Initialize terms for Jacobian elements
                J11 = 0
                J22 = 0
                J33 = 0
                J44 = 0
                
                # Calculate diagonal and off-diagonal elements
                for i in range(self.nbr):
                    if self.nl[i] - 1 == n or self.nr[i] - 1 == n:
                        if self.nl[i] - 1 == n:
                            l = self.nr[i] - 1
                        if self.nr[i] - 1 == n:
                            l = self.nl[i] - 1
                            
                        # Calculating parts of the diagonal elements
                        J11 += self.Vm[n] * self.Vm[l] * Ym[n, l] * sin(t[n, l] - self.delta[n] + self.delta[l])
                        J33 += self.Vm[n] * self.Vm[l] * Ym[n, l] * cos(t[n, l] - self.delta[n] + self.delta[l])
                        
                        if self.kb[n] != 1:  # Not a slack bus
                            J22 += self.Vm[l] * Ym[n, l] * cos(t[n, l] - self.delta[n] + self.delta[l])
                            J44 += self.Vm[l] * Ym[n, l] * sin(t[n, l] - self.delta[n] + self.delta[l])
                            
                        # Calculate off-diagonal elements
                        if self.kb[n] != 1 and self.kb[l] != 1:
                            lk = nbus_int + l - ngs[l] - nss[l] - ns
                            ll = l - nss[l]
                            
                            # Off-diagonal elements of J1
                            if nn >= 0 and ll >= 0 and nn < m and ll < m:
                                A[nn, ll] = -self.Vm[n] * self.Vm[l] * Ym[n, l] * sin(t[n, l] - self.delta[n] + self.delta[l])
                            
                            # Off-diagonal elements of J2
                            if self.kb[l] == 0 and nn >= 0 and lk >= 0 and nn < m and lk < m:
                                A[nn, lk] = self.Vm[n] * Ym[n, l] * cos(t[n, l] - self.delta[n] + self.delta[l])
                            
                            # Off-diagonal elements of J3
                            if self.kb[n] == 0 and lm >= 0 and ll >= 0 and lm < m and ll < m:
                                A[lm, ll] = -self.Vm[n] * self.Vm[l] * Ym[n, l] * cos(t[n, l] - self.delta[n] + self.delta[l])
                            
                            # Off-diagonal elements of J4
                            if self.kb[n] == 0 and self.kb[l] == 0 and lm >= 0 and lk >= 0 and lm < m and lk < m:
                                A[lm, lk] = -self.Vm[n] * Ym[n, l] * sin(t[n, l] - self.delta[n] + self.delta[l])
                
                # Calculate power mismatches
                Pk = self.Vm[n]**2 * Ym[n, n] * cos(t[n, n]) + J33
                Qk = -self.Vm[n]**2 * Ym[n, n] * sin(t[n, n]) - J11
                
                # For slack bus
                if self.kb[n] == 1:
                    self.P[n] = Pk
                    self.Q[n] = Qk
                
                # For generator (PV) bus
                if self.kb[n] == 2:
                    self.Q[n] = Qk
                    
                    # Handle generator reactive power limits
                    if self.Qmax[n] != 0:
                        Qgc = self.Q[n] * self.basemva + self.Qd[n] - self.Qsh[n]
                        if self.iter <= 7 and self.iter > 2:
                            if Qgc < self.Qmin[n]:
                                self.Vm[n] += 0.01
                            elif Qgc > self.Qmax[n]:
                                self.Vm[n] -= 0.01
                
                # Set the jacobian matrix elements
                if self.kb[n] != 1 and nn >= 0 and nn < m:
                    # Diagonal elements of J1
                    A[nn, nn] = J11
                    # Power mismatch for P
                    DC[nn] = self.P[n] - Pk
                
                if self.kb[n] == 0 and lm >= 0 and nn >= 0 and lm < m and nn < m:
                    # Diagonal elements of J2
                    A[nn, lm] = 2 * self.Vm[n] * Ym[n, n] * cos(t[n, n]) + J22
                    # Diagonal elements of J3
                    A[lm, nn] = J33
                    # Diagonal elements of J4
                    A[lm, lm] = -2 * self.Vm[n] * Ym[n, n] * sin(t[n, n]) - J44
                    # Power mismatch for Q
                    DC[lm] = self.Q[n] - Qk
            
            # Solve for the correction vector
            try:
                DX = np.linalg.solve(A, DC)
            except np.linalg.LinAlgError:
                print("Matrix singular, using pseudo-inverse for solution")
                DX = np.linalg.pinv(A) @ DC
            
            # Update voltage angle and magnitude
            for n in range(nbus_int):
                nn = n - nss[n]
                lm = nbus_int + n - ngs[n] - nss[n] - ns
                
                if self.kb[n] != 1 and nn >= 0 and nn < len(DX):
                    self.delta[n] += DX[nn]
                
                if self.kb[n] == 0 and lm >= 0 and lm < len(DX):
                    self.Vm[n] += DX[lm]
            
            # Calculate maximum error
            self.maxerror = max(abs(DC))
            
            # Check for convergence
            if self.iter == self.maxiter and self.maxerror > self.accuracy:
                print(f"WARNING: Iterative solution did not converge after {self.iter} iterations.")
                converge = 0
        
        # Set solution status
        if converge != 1:
            self.tech = "ITERATIVE SOLUTION DID NOT CONVERGE"
        else:
            self.tech = "Power Flow Solution by Newton-Raphson Method"
        
        # Update voltage in rectangular form
        self.V = self.Vm * (cos(self.delta) + 1j * sin(self.delta))
        self.deltad = 180 / pi * self.delta
        
        # Update bus powers
        k = 0
        self.Pgg = []
        self.Qgg = []
        
        for n in range(nbus_int):
            if self.kb[n] == 1:
                k += 1
                self.S[n] = self.P[n] + 1j * self.Q[n]
                self.Pg[n] = self.P[n] * self.basemva + self.Pd[n]
                self.Qg[n] = self.Q[n] * self.basemva + self.Qd[n] - self.Qsh[n]
                self.Pgg.append(self.Pg[n])
                self.Qgg.append(self.Qg[n])
            elif self.kb[n] == 2:
                k += 1
                self.S[n] = self.P[n] + 1j * self.Q[n]
                self.Qg[n] = self.Q[n] * self.basemva + self.Qd[n] - self.Qsh[n]
                self.Pgg.append(self.Pg[n])
                self.Qgg.append(self.Qg[n])
            
            self.yload[n] = (self.Pd[n] - 1j * self.Qd[n] + 1j * self.Qsh[n]) / (self.basemva * self.Vm[n]**2)
            
        # Calculate system totals
        self.Pgt = sum(self.Pg)
        self.Qgt = sum(self.Qg)
        self.Pdt = sum(self.Pd)
        self.Qdt = sum(self.Qd)
        self.Qsht = sum(self.Qsh)
        
        # Update busdata with new voltage values
        self.busdata[:, 2] = self.Vm
        self.busdata[:, 3] = self.deltad
        
    def lfgauss(self):
        """Power flow solution by Gauss-Seidel method"""
        # Initialization
        nbus_int = int(self.nbus)
        self.Vm = np.zeros(nbus_int)
        self.delta = np.zeros(nbus_int)
        self.yload = np.zeros(nbus_int, dtype=complex)
        self.deltad = np.zeros(nbus_int)
        self.kb = np.zeros(nbus_int, dtype=int)
        self.V = np.zeros(nbus_int, dtype=complex)
        self.P = np.zeros(nbus_int)
        self.Q = np.zeros(nbus_int)
        self.S = np.zeros(nbus_int, dtype=complex)
        self.Pd = np.zeros(nbus_int)
        self.Qd = np.zeros(nbus_int)
        self.Pg = np.zeros(nbus_int)
        self.Qg = np.zeros(nbus_int)
        self.Qmin = np.zeros(nbus_int)
        self.Qmax = np.zeros(nbus_int)
        self.Qsh = np.zeros(nbus_int)
        DV = np.zeros(nbus_int)
        
        # Process bus data
        for k in range(len(self.busdata)):
            n = int(self.busdata[k, 0])
            n_idx = n - 1  # 0-indexed array
            
            self.kb[n_idx] = int(self.busdata[k, 1])
            self.Vm[n_idx] = self.busdata[k, 2]
            self.delta[n_idx] = self.busdata[k, 3]
            self.Pd[n_idx] = self.busdata[k, 4]
            self.Qd[n_idx] = self.busdata[k, 5]
            self.Pg[n_idx] = self.busdata[k, 6]
            self.Qg[n_idx] = self.busdata[k, 7]
            self.Qmin[n_idx] = self.busdata[k, 8]
            self.Qmax[n_idx] = self.busdata[k, 9]
            self.Qsh[n_idx] = self.busdata[k, 10]
            
            if self.Vm[n_idx] <= 0:
                self.Vm[n_idx] = 1.0
                self.V[n_idx] = 1.0 + 0j
            else:
                self.delta[n_idx] = np.pi / 180 * self.delta[n_idx]
                self.V[n_idx] = self.Vm[n_idx] * (np.cos(self.delta[n_idx]) + 1j * np.sin(self.delta[n_idx]))
                self.P[n_idx] = (self.Pg[n_idx] - self.Pd[n_idx]) / self.basemva
                self.Q[n_idx] = (self.Qg[n_idx] - self.Qd[n_idx] + self.Qsh[n_idx]) / self.basemva
                self.S[n_idx] = self.P[n_idx] + 1j * self.Q[n_idx]
            
            DV[n_idx] = 0
        
        # Initialize variables for convergence
        converge = 1
        Vc = np.zeros(nbus_int, dtype=complex)
        Sc = np.zeros(nbus_int, dtype=complex)
        DP = np.zeros(nbus_int, dtype=float)
        DQ = np.zeros(nbus_int, dtype=float)
        
        # Set default parameters if not already set
        accel = getattr(self, 'accel', 1.3)
        accuracy = getattr(self, 'accuracy', 0.001)
        maxiter = getattr(self, 'maxiter', 100)
        
        # Start iteration
        self.iter = 0
        self.maxerror = 10
        
        while self.maxerror >= accuracy and self.iter <= maxiter:
            self.iter += 1
            
            for n in range(nbus_int):
                YV = 0 + 0j
                
                # Calculate voltage contribution from connected buses
                for L in range(self.nbr):
                    if self.nl[L] - 1 == n:
                        k = self.nr[L] - 1
                        YV += self.Ybus[n, k] * self.V[k]
                    elif self.nr[L] - 1 == n:
                        k = self.nl[L] - 1
                        YV += self.Ybus[n, k] * self.V[k]
                
                # Calculate complex power
                Sc[n] = np.conj(self.V[n]) * (self.Ybus[n, n] * self.V[n] + YV)
                Sc[n] = np.conj(Sc[n])
                
                # Calculate power mismatches
                DP[n] = self.P[n] - np.real(Sc[n])
                DQ[n] = self.Q[n] - np.imag(Sc[n])
                
                # Slack bus (type 1)
                if self.kb[n] == 1:
                    self.S[n] = Sc[n]
                    self.P[n] = np.real(Sc[n])
                    self.Q[n] = np.imag(Sc[n])
                    DP[n] = 0
                    DQ[n] = 0
                    Vc[n] = self.V[n]
                # PV bus (type 2)
                elif self.kb[n] == 2:
                    self.Q[n] = np.imag(Sc[n])
                    self.S[n] = self.P[n] + 1j * self.Q[n]
                    
                    # Check reactive power limits for generator buses
                    if self.Qmax[n] != 0:
                        Qgc = self.Q[n] * self.basemva + self.Qd[n] - self.Qsh[n]
                        if abs(DQ[n]) <= 0.005 and self.iter >= 10:  # After 10 iterations
                            if DV[n] <= 0.045:  # Total change not exceeding .05 pu
                                if Qgc < self.Qmin[n]:
                                    self.Vm[n] += 0.005
                                    DV[n] += 0.005
                                elif Qgc > self.Qmax[n]:
                                    self.Vm[n] -= 0.005
                                    DV[n] += 0.005
                
                # Calculate new voltage for non-slack buses
                if self.kb[n] != 1:
                    Vc[n] = (np.conj(self.S[n]) / np.conj(self.V[n]) - YV) / self.Ybus[n, n]
                
                # Update voltage
                if self.kb[n] == 0:  # PQ bus
                    self.V[n] += accel * (Vc[n] - self.V[n])
                elif self.kb[n] == 2:  # PV bus
                    VcI = np.imag(Vc[n])
                    VcR = np.sqrt(self.Vm[n]**2 - VcI**2)
                    Vc[n] = VcR + 1j * VcI
                    self.V[n] += accel * (Vc[n] - self.V[n])
            
            # Check convergence - Fix for the TypeError
            self.maxerror = float(np.max([np.max(np.abs(DP)), np.max(np.abs(DQ))]))
            
            if self.iter == maxiter and self.maxerror > accuracy:
                print(f"\nWARNING: Iterative solution did not converge after {self.iter} iterations.\n")
                converge = 0
        
        # Set solution status
        if converge != 1:
            self.tech = "ITERATIVE SOLUTION DID NOT CONVERGE"
        else:
            self.tech = "Power Flow Solution by Gauss-Seidel Method"
        
        # Update results
        k = 0
        self.Pgg = []  # Initialize as list

        for n in range(nbus_int):
            self.Vm[n] = abs(self.V[n])
            self.deltad[n] = np.angle(self.V[n]) * 180 / np.pi
            
            if self.kb[n] == 1:  # Slack bus
                self.S[n] = self.P[n] + 1j * self.Q[n]
                self.Pg[n] = self.P[n] * self.basemva + self.Pd[n]
                self.Qg[n] = self.Q[n] * self.basemva + self.Qd[n] - self.Qsh[n]
                k += 1
                self.Pgg.append(self.Pg[n])
            elif self.kb[n] == 2:  # Generator bus
                k += 1
                self.Pgg.append(self.Pg[n])
                self.S[n] = self.P[n] + 1j * self.Q[n]
                self.Qg[n] = self.Q[n] * self.basemva + self.Qd[n] - self.Qsh[n]
            
            self.yload[n] = (self.Pd[n] - 1j * self.Qd[n] + 1j * self.Qsh[n]) / (self.basemva * self.Vm[n]**2)
        
        # Calculate system totals
        self.Pgt = sum(self.Pg)
        self.Qgt = sum(self.Qg)
        self.Pdt = sum(self.Pd)
        self.Qdt = sum(self.Qd)
        self.Qsht = sum(self.Qsh)
        
        # Update busdata with new voltage values
        self.busdata[:, 2] = self.Vm
        self.busdata[:, 3] = self.deltad
    
    def decouple(self):
        """Power flow solution by Fast Decoupled method"""
        # Initialization
        ns = 0  # counter for slack buses
        nbus_int = int(self.nbus)
        self.Vm = np.zeros(nbus_int)
        self.delta = np.zeros(nbus_int)
        self.yload = np.zeros(nbus_int, dtype=complex)
        self.deltad = np.zeros(nbus_int)
        self.kb = np.zeros(nbus_int, dtype=int)
        self.V = np.zeros(nbus_int, dtype=complex)
        self.P = np.zeros(nbus_int)
        self.Q = np.zeros(nbus_int)
        self.S = np.zeros(nbus_int, dtype=complex)
        self.Pd = np.zeros(nbus_int)
        self.Qd = np.zeros(nbus_int)
        self.Pg = np.zeros(nbus_int)
        self.Qg = np.zeros(nbus_int)
        self.Qmin = np.zeros(nbus_int)
        self.Qmax = np.zeros(nbus_int)
        self.Qsh = np.zeros(nbus_int)
        nss = np.zeros(nbus_int, dtype=int)  # Cumulative number of slack buses
        
        # Process bus data
        for k in range(len(self.busdata)):
            n = int(self.busdata[k, 0])
            n_idx = n - 1  # 0-indexed array
            
            self.kb[n_idx] = int(self.busdata[k, 1])
            self.Vm[n_idx] = self.busdata[k, 2]
            self.delta[n_idx] = self.busdata[k, 3]
            self.Pd[n_idx] = self.busdata[k, 4]
            self.Qd[n_idx] = self.busdata[k, 5]
            self.Pg[n_idx] = self.busdata[k, 6]
            self.Qg[n_idx] = self.busdata[k, 7]
            self.Qmin[n_idx] = self.busdata[k, 8]
            self.Qmax[n_idx] = self.busdata[k, 9]
            self.Qsh[n_idx] = self.busdata[k, 10]
            
            if self.Vm[n_idx] <= 0:
                self.Vm[n_idx] = 1.0
                self.V[n_idx] = 1.0 + 0j
            else:
                self.delta[n_idx] = np.pi / 180 * self.delta[n_idx]
                self.V[n_idx] = self.Vm[n_idx] * (np.cos(self.delta[n_idx]) + 1j * np.sin(self.delta[n_idx]))
                self.P[n_idx] = (self.Pg[n_idx] - self.Pd[n_idx]) / self.basemva
                self.Q[n_idx] = (self.Qg[n_idx] - self.Qd[n_idx] + self.Qsh[n_idx]) / self.basemva
                self.S[n_idx] = self.P[n_idx] + 1j * self.Q[n_idx]
            
            if self.kb[n_idx] == 1:  # Count slack buses
                ns += 1
            nss[n_idx] = ns
        
        # Extract Ybus components
        Ym = np.abs(self.Ybus)
        t = np.angle(self.Ybus)
        
        # First pass to determine matrix sizes
        n_nonsw = 0  # non-slack buses (type 0 and 2)
        n_pq = 0     # PQ buses (type 0)
        
        for n in range(nbus_int):
            if self.kb[n] == 0 or self.kb[n] == 2:
                n_nonsw += 1
            if self.kb[n] == 0:
                n_pq += 1
        
        # Create matrices of appropriate size
        B1 = np.zeros((n_nonsw, n_nonsw))
        B2 = np.zeros((n_pq, n_pq))
        
        # Fill B1 matrix - exactly as in MATLAB
        ii = 0
        for ib in range(nbus_int):
            if self.kb[ib] == 0 or self.kb[ib] == 2:  # non-slack buses
                ii += 1
                jj = 0
                for jb in range(nbus_int):
                    if self.kb[jb] == 0 or self.kb[jb] == 2:  # non-slack buses
                        jj += 1
                        B1[ii-1, jj-1] = np.imag(self.Ybus[ib, jb])
        
        # Fill B2 matrix - exactly as in MATLAB
        ii = 0
        for ib in range(nbus_int):
            if self.kb[ib] == 0:  # PQ buses only
                ii += 1
                jj = 0
                for jb in range(nbus_int):
                    if self.kb[jb] == 0:  # PQ buses only
                        jj += 1
                        B2[ii-1, jj-1] = np.imag(self.Ybus[ib, jb])
        
        # Invert matrices
        try:
            B1inv = np.linalg.inv(B1)
            B2inv = np.linalg.inv(B2)
        except np.linalg.LinAlgError:
            print("Matrix inversion failed. Using pseudoinverse.")
            B1inv = np.linalg.pinv(B1)
            B2inv = np.linalg.pinv(B2)
        
        # Start iterations
        self.maxerror = 1.0
        converge = 1
        self.iter = 0
        
        # Default parameters
        accuracy = getattr(self, 'accuracy', 0.001)
        maxiter = getattr(self, 'maxiter', 100)
        
        while self.maxerror >= accuracy and self.iter <= maxiter:
            self.iter += 1
            
            # Initialize mismatch arrays
            DP = np.zeros(n_nonsw)  # for non-slack buses
            DQ = np.zeros(n_pq)     # for PQ buses only
            DPV = np.zeros(n_nonsw) # normalized by voltage
            DQV = np.zeros(n_pq)    # normalized by voltage
            
            id_count = 0  # counter for non-slack buses
            iv_count = 0  # counter for PQ buses
            
            # Calculate mismatches for all buses
            for n in range(nbus_int):
                # Calculate power at each bus
                J11 = 0
                J33 = 0
                
                for i in range(self.nbr):
                    if self.nl[i] - 1 == n or self.nr[i] - 1 == n:
                        if self.nl[i] - 1 == n:
                            l = self.nr[i] - 1
                        else:
                            l = self.nl[i] - 1
                        
                        J11 += self.Vm[n] * self.Vm[l] * Ym[n, l] * np.sin(t[n, l] - self.delta[n] + self.delta[l])
                        J33 += self.Vm[n] * self.Vm[l] * Ym[n, l] * np.cos(t[n, l] - self.delta[n] + self.delta[l])
                
                # Calculate P and Q at each bus
                Pk = self.Vm[n]**2 * Ym[n, n] * np.cos(t[n, n]) + J33
                Qk = -self.Vm[n]**2 * Ym[n, n] * np.sin(t[n, n]) - J11
                
                # For slack bus, update P and Q
                if self.kb[n] == 1:
                    self.P[n] = Pk
                    self.Q[n] = Qk
                
                # For PV buses, update Q and check reactive limits
                elif self.kb[n] == 2:
                    self.Q[n] = Qk
                    
                    # FIXED: Using Qd instead of Pd for reactive power calculation
                    Qgc = self.Q[n] * self.basemva + self.Qd[n] - self.Qsh[n]
                    if self.Qmax[n] != 0:
                        # FIXED: Match MATLAB condition (between iterations 10-20)
                        if self.iter <= 20 and self.iter >= 10:
                            if Qgc < self.Qmin[n]:
                                self.Vm[n] += 0.005
                            elif Qgc > self.Qmax[n]:
                                self.Vm[n] -= 0.005
                
                # Calculate mismatches for non-slack buses
                if self.kb[n] != 1:
                    DP[id_count] = self.P[n] - Pk
                    DPV[id_count] = (self.P[n] - Pk) / self.Vm[n]
                    id_count += 1
                
                # Calculate mismatches for PQ buses
                if self.kb[n] == 0:
                    DQ[iv_count] = self.Q[n] - Qk
                    DQV[iv_count] = (self.Q[n] - Qk) / self.Vm[n]
                    iv_count += 1
            
            # Solve for angle and voltage corrections
            Dd = -np.matmul(B1inv, DPV)
            DV = -np.matmul(B2inv, DQV)
            
            # Apply corrections
            id_count = 0
            iv_count = 0
            
            for n in range(nbus_int):
                # Update angles for non-slack buses
                if self.kb[n] != 1:
                    self.delta[n] += Dd[id_count]
                    id_count += 1
                
                # Update voltages for PQ buses
                if self.kb[n] == 0:
                    self.Vm[n] += DV[iv_count]
                    iv_count += 1
            
            # Calculate maximum error
            self.maxerror = np.max([np.max(np.abs(DP)), np.max(np.abs(DQ))])
            
            if self.iter == maxiter and self.maxerror > accuracy:
                print(f"\nWARNING: Iterative solution did not converge after {self.iter} iterations.\n")
                converge = 0
        
        # Set solution status
        if converge != 1:
            self.tech = "ITERATIVE SOLUTION DID NOT CONVERGE"
        else:
            self.tech = "Power Flow Solution by Fast Decoupled Method"
        
        # Update voltage in rectangular form
        self.V = self.Vm * (np.cos(self.delta) + 1j * np.sin(self.delta))
        self.deltad = 180 / np.pi * self.delta
        
        # Update bus powers and calculate totals
        k = 0
        self.Pgg = []
        
        for n in range(nbus_int):
            if self.kb[n] == 1:  # Slack bus
                self.S[n] = self.P[n] + 1j * self.Q[n]
                self.Pg[n] = self.P[n] * self.basemva + self.Pd[n]
                self.Qg[n] = self.Q[n] * self.basemva + self.Qd[n] - self.Qsh[n]
                k += 1
                self.Pgg.append(self.Pg[n])
            elif self.kb[n] == 2:  # Generator bus
                self.S[n] = self.P[n] + 1j * self.Q[n]
                self.Qg[n] = self.Q[n] * self.basemva + self.Qd[n] - self.Qsh[n]
                k += 1
                self.Pgg.append(self.Pg[n])
            
            self.yload[n] = (self.Pd[n] - 1j * self.Qd[n] + 1j * self.Qsh[n]) / (self.basemva * self.Vm[n]**2)
        
        # Calculate system totals
        self.Pgt = sum(self.Pg)
        self.Qgt = sum(self.Qg)
        self.Pdt = sum(self.Pd)
        self.Qdt = sum(self.Qd)
        self.Qsht = sum(self.Qsh)
        
        # Update busdata with new voltage values
        self.busdata[:, 2] = self.Vm
        self.busdata[:, 3] = self.deltad

    def perturbation(self):
        """Power Flow Solution by the Power Perturbation Technique."""
        # Import necessary libraries at the beginning
        ns = 0
        nbus = len(self.busdata)
        
        # Initialize arrays
        kb = np.zeros(nbus+1, dtype=int)
        Vm = np.zeros(nbus+1)
        delta = np.zeros(nbus+1)
        Pd = np.zeros(nbus+1)
        Qd = np.zeros(nbus+1)
        Pg = np.zeros(nbus+1)
        Qg = np.zeros(nbus+1)
        Qmin = np.zeros(nbus+1)
        Qmax = np.zeros(nbus+1)
        Qsh = np.zeros(nbus+1)
        P = np.zeros(nbus+1)
        Q = np.zeros(nbus+1)
        S = np.zeros(nbus+1, dtype=complex)
        V = np.zeros(nbus+1, dtype=complex)
        nss = np.zeros(nbus+1, dtype=int)
        
        self.yload = np.zeros(nbus+1, dtype=complex)
        
        # Process bus data
        for k in range(nbus):
            n = int(self.busdata[k, 0])
            kb[n] = int(self.busdata[k, 1])
            Vm[n] = self.busdata[k, 2]
            delta[n] = self.busdata[k, 3]
            Pd[n] = self.busdata[k, 4]
            Qd[n] = self.busdata[k, 5]
            Pg[n] = self.busdata[k, 6]
            Qg[n] = self.busdata[k, 7]
            Qmin[n] = self.busdata[k, 8]
            Qmax[n] = self.busdata[k, 9]
            Qsh[n] = self.busdata[k, 10]
            
            if Vm[n] <= 0:
                Vm[n] = 1.0
                V[n] = 1.0 + 0j
            else:
                delta[n] = np.pi/180 * delta[n]
                V[n] = Vm[n] * (np.cos(delta[n]) + 1j * np.sin(delta[n]))
                P[n] = (Pg[n] - Pd[n]) / self.basemva
                Q[n] = (Qg[n] - Qd[n] + Qsh[n]) / self.basemva
                S[n] = P[n] + 1j * Q[n]
            
            if kb[n] == 1:
                ns += 1
            nss[n] = ns
        
        # Count non-swing buses
        ii = 0
        for ib in range(1, nbus+1):
            if kb[ib] == 0 or kb[ib] == 2:
                ii += 1
        
        # Create Y1 matrix
        Y1 = np.zeros((ii+1, ii+1), dtype=complex)
        
        # Fill Y1 matrix
        ii = 0
        for ib in range(1, nbus+1):
            if kb[ib] == 0 or kb[ib] == 2:
                ii += 1
                jj = 0
                for jb in range(1, nbus+1):
                    if kb[jb] == 0 or kb[jb] == 2:
                        jj += 1
                        Y1[ii, jj] = self.Ybus[ib-1, jb-1]
        
        # Copy to YS
        YS = Y1.copy()
        
        # Initialize I array for iteration
        I = np.zeros(ii+1, dtype=complex)
        
        # Initialize iteration
        maxerror = 1.0
        self.iter = 0
        
        # Iteration loop
        while maxerror >= self.accuracy and self.iter <= self.maxiter:
            self.iter += 1
            
            # Calculate injections
            for n in range(1, nbus+1):
                nn = n - nss[n]
                if kb[n] != 1:  # Not a swing bus
                    I[nn] = 0
                    
                    # Find connections to swing buses
                    for i in range(len(self.nl)):
                        nl_i = self.nl[i]
                        nr_i = self.nr[i]
                        if nl_i == n or nr_i == n:
                            if nl_i == n:
                                l = nr_i
                            else:
                                l = nl_i
                            
                            if kb[l] == 1:  # Connected to swing bus
                                I[nn] = I[nn] - self.Ybus[n-1, l-1] * V[l]
                    
                    # Update YS diagonal
                    YS[nn, nn] = Y1[nn, nn] - np.conj(S[n]) / (Vm[n]**2)
            
            # Solve system using non-conjugated I (best match to MATLAB)
            A1 = YS[1:ii+1, 1:ii+1]
            Vk = np.linalg.solve(A1, I[1:ii+1])
                
            # Update voltages and powers
            Pk = P.copy()
            Qk = Q.copy()
            
            for n in range(1, nbus+1):
                if kb[n] != 1:  # Not swing bus
                    nn = n - nss[n]
                    V[n] = Vk[nn-1]  # Adjust index for 0-based Python
                    Pk[n] = (abs(V[n])**2) / (Vm[n]**2) * P[n]
                    Qk[n] = (abs(V[n])**2) / (Vm[n]**2) * Q[n]
                    Vm[n] = abs(V[n])
                
                if kb[n] == 1:  # Swing bus
                    Pk[n] = P[n]
                    Qk[n] = Q[n]
            
            # Calculate mismatches
            DP = Pk - P
            DQ = Qk - Q
            
            # Calculate maximum error
            maxerror = np.max([np.max(np.abs(DP)), np.max(np.abs(DQ))])
        
        # Calculate swing bus power
        for n in range(1, nbus+1):
            if kb[n] == 1:  # Swing bus
                S1 = 0
                for j in range(1, nbus+1):
                    S1 += np.conj(self.Ybus[n-1, j-1]) * np.conj(V[j])
                
                S[n] = V[n] * S1
                Pk[n] = np.real(S[n])
                Qk[n] = np.imag(S[n])
                
                # Update the generation values for the swing bus
                P[n] = Pk[n]
                Q[n] = Qk[n]
                Pg[n] = P[n] * self.basemva + Pd[n]
                Qg[n] = Q[n] * self.basemva + Qd[n] - Qsh[n]
        
        # Update values
        P = Pk
        Q = Qk
        S = P + 1j * Q
        
        # Calculate angles
        delta = np.angle(V)
        deltad = 180/np.pi * delta
        
        # Create standard-format 0-indexed arrays for compatibility with other methods
        self.nbus = nbus
        self.kb = np.zeros(nbus, dtype=int)
        self.Vm = np.zeros(nbus)
        self.delta = np.zeros(nbus)
        self.deltad = np.zeros(nbus)
        self.P = np.zeros(nbus)
        self.Q = np.zeros(nbus)
        self.S = np.zeros(nbus, dtype=complex)
        self.Pg = np.zeros(nbus)
        self.Qg = np.zeros(nbus)
        self.Pd = np.zeros(nbus)
        self.Qd = np.zeros(nbus)
        self.Qsh = np.zeros(nbus)
        self.V = np.zeros(nbus, dtype=complex)
        
        # Create correctly-sized yload array
        self.yload = np.zeros(nbus, dtype=complex)
        
        # Map 1-indexed arrays to 0-indexed for compatibility
        for i in range(nbus):
            n = int(self.busdata[i, 0])
            n_idx = i  # 0-indexed position
            
            self.kb[n_idx] = kb[n]
            self.Vm[n_idx] = Vm[n]
            self.delta[n_idx] = delta[n]
            self.deltad[n_idx] = deltad[n]
            self.V[n_idx] = V[n]
            self.P[n_idx] = P[n]
            self.Q[n_idx] = Q[n]
            self.S[n_idx] = S[n]
            self.Pg[n_idx] = Pg[n]
            self.Qg[n_idx] = Qg[n]
            self.Pd[n_idx] = Pd[n]
            self.Qd[n_idx] = Qd[n]
            self.Qsh[n_idx] = Qsh[n]
            
            # Update yload for line flow calculations
            self.yload[n_idx] = (Pd[n] - 1j * Qd[n] + 1j * Qsh[n]) / (self.basemva * Vm[n]**2)
        
        # Update busdata with new voltage values
        for i in range(nbus):
            self.busdata[i, 2] = self.Vm[i]
            self.busdata[i, 3] = self.deltad[i]
        
        # Initialize Pgg and Qgg as lists
        k = 0
        self.Pgg = []
        self.Qgg = []
        
        # Populate Pgg and Qgg
        for n in range(nbus):
            if self.kb[n] == 1:  # Slack bus
                k += 1
                self.Pgg.append(self.Pg[n])
                self.Qgg.append(self.Qg[n])
            elif self.kb[n] == 2:  # Generator buses
                k += 1
                self.Pgg.append(self.Pg[n])
                self.Qgg.append(self.Qg[n])
        
        # Calculate system totals
        self.Pgt = np.sum(self.Pg)
        self.Qgt = np.sum(self.Qg)
        self.Pdt = np.sum(self.Pd)
        self.Qdt = np.sum(self.Qd)
        self.Qsht = np.sum(self.Qsh)
        self.maxerror = maxerror
        
        # Set solution status
        self.tech = "Power Flow Solution by Power Perturbation Technique"

    def busout(self):
        """Print power flow solution in tabulated form"""
        print(self.tech)
        print(f"Maximum Power Mismatch = {self.maxerror}")
        print(f"No. of Iterations = {self.iter}\n")
        
        headers = [
            "Bus  Voltage  Angle    ------Load------    ---Generation---   Injected",
            "No.  Mag.     Degree     MW       Mvar       MW       Mvar       Mvar "
        ]
        
        print("\n".join(headers))
        print("="*75)
        
        for n in range(int(self.nbus)):
            n_idx = n  # 0-indexed
            print(f"{n+1:5d} {self.Vm[n_idx]:7.3f} {self.deltad[n_idx]:8.3f} "
                  f"{self.Pd[n_idx]:9.3f} {self.Qd[n_idx]:9.3f} {self.Pg[n_idx]:9.3f} "
                  f"{self.Qg[n_idx]:9.3f} {self.Qsh[n_idx]:8.3f}")
        
        print(f"\nTotal              {self.Pdt:9.3f} {self.Qdt:9.3f} {self.Pgt:9.3f} "
              f"{self.Qgt:9.3f} {self.Qsht:9.3f}\n")
        
    def lineflow(self):
        """Compute line flow and losses"""
        SLT = 0  # Initialize total line losses
        
        print("\n")
        print("Line Flow and Losses")
        print("="*75)
        print("--Line--  Power at bus & line flow    --Line loss--  Transformer")
        print("from  to    MW      Mvar     MVA       MW      Mvar      tap")
        print("="*75)
        
        for n in range(1, int(self.nbus)+1):
            n_idx = n-1  # 0-indexed
            busprt = 0
            
            for L in range(int(self.nbr)):
                if busprt == 0:
                    print(f"\n{n:6d}      {self.P[n_idx]*self.basemva:9.3f} "
                          f"{self.Q[n_idx]*self.basemva:9.3f} {abs(self.S[n_idx]*self.basemva):9.3f}")
                    busprt = 1
                
                # Line connected to bus n
                if self.nl[L] == n:  # From side
                    k = int(self.nr[L])
                    k_idx = k-1
                    
                    # Calculate line flows
                    In = (self.V[n_idx] - self.a[L]*self.V[k_idx])*self.y[L]/(self.a[L]**2) + self.Bc[L]/self.a[L]**2*self.V[n_idx]
                    Ik = (self.V[k_idx] - self.V[n_idx]/self.a[L])*self.y[L] + self.Bc[L]*self.V[k_idx]
                    
                    # Line flow from n to k
                    Snk = self.V[n_idx] * conj(In) * self.basemva
                    
                    # Line flow from k to n
                    Skn = self.V[k_idx] * conj(Ik) * self.basemva
                    
                    # Line loss
                    SL = Snk + Skn
                    SLT += SL
                    
                    print(f"{k:12d} {real(Snk):9.3f} {imag(Snk):9.3f} {abs(Snk):9.3f} "
                          f"{real(SL):9.3f} {imag(SL):9.3f}", end="")
                    
                    if self.a[L] != 1:
                        print(f" {self.a[L]:9.3f}")
                    else:
                        print("")
                    
                elif self.nr[L] == n:  # To side
                    k = int(self.nl[L])
                    k_idx = k-1
                    
                    # Calculate line flows
                    In = (self.V[n_idx] - self.V[k_idx]/self.a[L])*self.y[L] + self.Bc[L]*self.V[n_idx]
                    Ik = (self.V[k_idx] - self.a[L]*self.V[n_idx])*self.y[L]/self.a[L]**2 + self.Bc[L]/self.a[L]**2*self.V[k_idx]
                    
                    # Line flow from n to k
                    Snk = self.V[n_idx] * conj(In) * self.basemva
                    
                    # Line flow from k to n
                    Skn = self.V[k_idx] * conj(Ik) * self.basemva
                    
                    # Line loss
                    SL = Snk + Skn
                    SLT += SL
                    
                    print(f"{k:12d} {real(Snk):9.3f} {imag(Snk):9.3f} {abs(Snk):9.3f} "
                          f"{real(SL):9.3f} {imag(SL):9.3f}")
        
        # Total losses (divide by 2 as each loss is counted twice)
        SLT = SLT / 2
        print(f"\nTotal loss                         {real(SLT):9.3f} {imag(SLT):9.3f}")
        
    def bloss(self):
        """Obtain B-coefficients of the loss formula as a function of real power generation"""
        # Calculate Zbus from Ybus
        Zbus = np.linalg.inv(self.Ybus)
        
        # Initialize counter for generator buses
        ngg = 0
        
        # Calculate current injections (adjusted for Python indexing)
        I = -1/self.basemva * (self.Pd - 1j*self.Qd) / np.conj(self.V)
        ID = np.sum(I)
        
        # Find generator buses and slack bus
        ks = 0  # Slack bus index (0-indexed)
        for k in range(int(self.nbus)):
            if self.kb[k] == 1:  # Slack bus
                ks = k
            if self.kb[k] != 0:  # Generator bus (including slack)
                ngg += 1
        
        # Calculate distribution factors
        d1 = I / ID
        DD = np.sum(d1 * Zbus[ks, :])
        
        # Setup generator and load indices arrays (using complex datatype)
        t1 = np.zeros(ngg, dtype=complex)
        d = []
        
        kg = 0  # Generator counter
        kd = 0  # Load counter
        
        for k in range(int(self.nbus)):
            if self.kb[k] != 0:  # Generator bus
                t1[kg] = Zbus[ks, k] / DD
                kg += 1
            else:  # Load bus
                d.append(I[k] / ID)
                kd += 1
        
        nd = int(self.nbus) - ngg  # Number of load buses
        
        # Form C matrices
        C1g = np.zeros((int(self.nbus), ngg))
        kg = 0
        
        for k in range(int(self.nbus)):
            if self.kb[k] != 0:
                kg += 1
                C1g[k, kg-1] = 1
        
        C1gg = np.eye(ngg)
        C1D = np.zeros((ngg, 1))
        C1 = np.hstack((C1g, d1.reshape(-1, 1)))
        
        # Form C2 matrix
        C2gD = np.vstack((C1gg, -t1.reshape(1, -1)))
        CnD = np.vstack((C1D, -t1[0]))
        C2 = np.hstack((C2gD, CnD))
        
        # Form C matrix
        C = np.matmul(C1, C2)
        
        # Calculate alpha coefficients
        al = np.zeros(ngg+1, dtype=complex)
        kg = 0
        
        for k in range(int(self.nbus)):
            if self.kb[k] != 0:
                # Prevent division by zero
                if self.Pg[k] > 1e-6:
                    al[kg] = (1 - 1j*((self.Qg[k] + self.Qsh[k]) / self.Pg[k])) / np.conj(self.V[k])
                else:
                    al[kg] = 1 / np.conj(self.V[k])  # Simplified for zero generation
                kg += 1
        
        # Add slack bus alpha
        al[ngg] = -self.V[ks] / Zbus[ks, ks]
        
        # Form alpha matrix
        alph = np.zeros((ngg+1, ngg+1), dtype=complex)
        for k in range(ngg+1):
            alph[k, k] = al[k]
        
        # Calculate T matrix using conjugate transpose more consistently
        T = np.matmul(np.matmul(np.matmul(alph, np.conj(C.T)), np.real(Zbus)), np.matmul(np.conj(C), np.conj(alph)))
        
        # Form B matrices
        BB = 0.5 * (T + np.conj(T))
        
        # Extract B, B0, B00
        self.B = np.zeros((ngg, ngg))
        self.B0 = np.zeros(ngg)
        
        for k in range(ngg):
            for m in range(ngg):
                self.B[k, m] = np.real(BB[k, m])
            self.B0[k] = 2 * np.real(BB[ngg, k])
        
        self.B00 = np.real(BB[ngg, ngg])
        
        # Calculate total system loss
        Pgg_array = np.array(self.Pgg)
        PL = np.dot(Pgg_array, np.dot(self.B/self.basemva, Pgg_array)) + np.dot(self.B0, Pgg_array) + self.B00 * self.basemva
        
        # Better formatted output
        np.set_printoptions(precision=6)
        print("B matrix:")
        print(self.B)
        print("\nB0 vector:")
        print(self.B0)
        print("\nB00 value:")
        print(self.B00)
        print(f"Total system loss = {PL:.6f} MW")
        
        return self.B, self.B0, self.B00
    
    def dispatch(self, Pdt=None, cost=None, mwlimits=None):
        """Economic dispatch using B coefficients for losses"""
        if Pdt is None:
            Pdt = self.Pdt
        
        if cost is None:
            print("Cost function matrix required")
            return
        
        ngg = len(cost)
        
        if mwlimits is None:
            mwlimits = np.column_stack((np.zeros(ngg), np.inf * np.ones(ngg)))
        
        if self.B is None:
            self.B = np.zeros((ngg, ngg))
        
        if self.B0 is None:
            self.B0 = np.zeros(ngg)
        
        if self.B00 is None:
            self.B00 = 0
                
        # Scaled B coefficients
        Bu = self.B / self.basemva
        B00u = self.basemva * self.B00
        
        # Cost function coefficients
        alpha = cost[:, 0]
        beta = cost[:, 1]
        gama = cost[:, 2]
        
        # Generator limits
        Pmin = mwlimits[:, 0]
        Pmax = mwlimits[:, 1]
        
        # Check feasibility
        if Pdt > np.sum(Pmax):
            print("Total demand is greater than the total sum of maximum generation.")
            print("No feasible solution. Reduce demand or correct generator limits.")
            return
        elif Pdt < np.sum(Pmin):
            print("Total demand is less than the total sum of minimum generation.")
            print("No feasible solution. Increase demand or correct generator limits.")
            return
        
        # Weights for active generators
        wgt = np.ones(ngg)
        
        # Initialize lambda
        if not hasattr(self, 'lambda_') or self.lambda_ is None:
            self.lambda_ = np.max(beta)
        
        lambda_ = self.lambda_
        
        # Newton-Raphson iteration
        iterp = 0
        DelP = 10
        
        # Initialize Pgg
        Pgg = np.zeros(ngg)
        
        while abs(DelP) >= 0.0001 and iterp < 200:
            iterp += 1
            
            # Form E matrix and Dx vector
            E = np.copy(Bu)  # Reset E for each iteration
            Dx = np.zeros(ngg)
            
            for k in range(ngg):
                if wgt[k] == 1:
                    E[k, k] = gama[k] / lambda_ + Bu[k, k]
                    Dx[k] = 0.5 * (1 - self.B0[k] - beta[k] / lambda_)
                else:
                    E[k, k] = 1
                    Dx[k] = 0
                    
                    for m in range(ngg):
                        if m != k:
                            E[k, m] = 0
            
            # Solve for power generation
            try:
                PP = np.linalg.solve(E, Dx)
            except np.linalg.LinAlgError:
                print(f"Matrix singular at iteration {iterp}, using pseudo-inverse")
                PP = np.linalg.pinv(E) @ Dx
            
            # Update Pgg for active generators
            for k in range(ngg):
                if wgt[k] == 1:
                    Pgg[k] = PP[k]
            
            Pgtt = np.sum(Pgg)
            
            # Calculate losses
            PL = np.dot(Pgg, np.dot(Bu, Pgg)) + np.dot(self.B0, Pgg) + B00u
            
            # Calculate residual
            DelP = Pdt + PL - Pgtt
            
            # Check generator limits
            for k in range(ngg):
                if Pgg[k] > Pmax[k] and abs(DelP) <= 0.001:
                    Pgg[k] = Pmax[k]
                    wgt[k] = 0
                elif Pgg[k] < Pmin[k] and abs(DelP) <= 0.001:
                    Pgg[k] = Pmin[k]
                    wgt[k] = 0
            
            # Recalculate losses and residual
            PL = np.dot(Pgg, np.dot(Bu, Pgg)) + np.dot(self.B0, Pgg) + B00u
            DelP = Pdt + PL - np.sum(Pgg)
            
            # Calculate gradient for lambda update
            grad = np.zeros(ngg)
            
            for k in range(ngg):
                if wgt[k] == 1:  # Only calculate gradient for active generators
                    BP = 0
                    for m in range(ngg):
                        if m != k:
                            BP += Bu[k, m] * Pgg[m]
                    
                    denominator = 2 * (gama[k] + lambda_ * Bu[k, k])**2
                    if denominator > 1e-10:  # Avoid division by very small numbers
                        grad[k] = (gama[k] * (1 - self.B0[k]) + Bu[k, k] * beta[k] - 2 * gama[k] * BP) / denominator
                    else:
                        grad[k] = 0
            
            sumgrad = np.dot(wgt, grad)
            
            # Update lambda (with safety check)
            if abs(sumgrad) > 1e-6:  # Avoid division by very small numbers
                Delambda = DelP / sumgrad
                # Limit the lambda change to avoid oscillations
                if abs(Delambda) > 0.5 * lambda_:
                    Delambda = 0.5 * lambda_ * np.sign(Delambda)
                lambda_ += Delambda
            else:
                # If gradient is too small, use a direct adjustment
                if DelP > 0:
                    lambda_ *= 1.05  # Increase lambda by 5%
                else:
                    lambda_ *= 0.95  # Decrease lambda by 5%
        
        # Store result
        self.lambda_ = lambda_
        self.Pgg = np.array(Pgg)  # Convert to NumPy array
        
        print(f"Incremental cost of delivered power (system lambda) = {lambda_:.6f} $/MWh")
        print("Optimal Dispatch of Generation:")
        print(self.Pgg)
        
        # Update busdata with new generation schedule if it exists
        if self.busdata is not None:
            ng = 0
            for k in range(len(self.busdata)):
                bus_num = int(self.busdata[k, 0])
                bus_idx = bus_num - 1
                if hasattr(self, 'kb') and self.kb is not None and bus_idx < len(self.kb) and self.kb[bus_idx] != 0:
                    if ng < len(self.Pgg):
                        self.busdata[k, 6] = self.Pgg[ng]  # Update Pg in busdata
                        ng += 1
            
            # Check slack bus mismatch if kb exists
            if hasattr(self, 'kb') and self.kb is not None:
                for k in range(int(self.nbus)):
                    if self.kb[k] == 1:  # Slack bus
                        dpslack = abs(self.Pg[k] - self.Pgg[0]) / self.basemva
                        print(f"Absolute value of the slack bus real power mismatch, dpslack = {dpslack:.4f} pu")
                        break
        
        return self.Pgg, lambda_, PL

    def gencost(self, Pgg=None, cost=None):
        """Compute the total generation cost"""
        if Pgg is None:
            if hasattr(self, 'Pgg') and self.Pgg is not None:
                Pgg = self.Pgg
            else:
                print("Error: No generation values provided")
                return None
        
        if cost is None:
            print("Error: Cost function matrix required")
            return None
        
        # Convert Pgg to NumPy array if it's a list
        Pgg = np.array(Pgg)
        
        ngg = len(cost)
        Pmt = np.vstack((np.ones(ngg), Pgg, Pgg**2))
        
        costv = np.zeros(ngg)
        for i in range(ngg):
            costv[i] = np.dot(cost[i, :], Pmt[:, i])
        
        totalcost = np.sum(costv)
        print(f"\nTotal generation cost = {totalcost:.2f} $/h")
        
        return totalcost

