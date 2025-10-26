# Fast Decoupled Power Flow Method

## Overview

The Fast Decoupled power flow method is an efficient technique for solving power flow problems in electrical power systems. It is a simplified version of the Newton-Raphson method that takes advantage of the weak coupling between active power and voltage magnitude, and between reactive power and voltage angle in transmission systems. This decoupling significantly reduces computational requirements while maintaining good convergence properties for most practical power systems.

## Mathematical Foundation

The Fast Decoupled method is based on two key simplifications of the Newton-Raphson method:

1. The coupling between active power and voltage magnitude (∂P/∂|V|) and between reactive power and voltage angle (∂Q/∂δ) is neglected.
2. The Jacobian matrix elements are simplified and kept constant during iterations.

### Decoupled Power Flow Equations

The standard Newton-Raphson power flow equations are decoupled into two separate sets:

\[
   \begin{matrix}
   [\frac{\Delta P}{|V|}] = [B^{\prime}] [\Delta\delta] & [\frac{\Delta Q}{|V|}] = [B^{\prime\prime}] [\Delta |V|]
   \end{matrix}
\]

Where:
- $\Delta P$ and $\Delta Q$ are the active and reactive power mismatches
- $\Delta \delta$ and $\Delta |V|$ are the voltage angle and magnitude corrections
- $B^{\prime}$ and $B^{\prime\prime}$ are constant matrices derived from the bus admittance matrix

## Algorithm Implementation

```mermaid
flowchart TD
    A[Start] --> B[Initialize voltage magnitudes and angles]
    B --> C[Form bus admittance matrix Ybus]
    C --> D["Discover bus types (slack,PV and PQ)"]
    D --> E["Calculte inv(B') by removing slack bus index from imag(Ybus)"]
    E --> F["Calculate inv(B'') by removing slack and PV bus indices from imag(Ybus)"]
    F --> G["Calculate power mismatches ΔP and ΔQ divide them by |V|"]
    G --> H{"Convergence check: max(|ΔP|, |ΔQ|) < tolerance?"}
    H --> |Yes| I[Calculate line flows and losses]
    H --> |No| O[Are Q generation limits vilated in PV buses?]
    O --> |Yes| P["Set the bus type to PQ, Track bus type indices and recalculate inv(B'')"]
    O --> |No| J["Solve for angle corrections: Δδ = -inv(B')·ΔP/|V|"]
    P --> J
    J --> K[Update voltage angles]
    K --> L["Solve for magnitude corrections: Δ|V| = -inv(B'')·ΔQ/|V|"]
    L --> M[Update voltage magnitudes]
    M --> G
    I --> N[End]
```


## Implementation Details

The Fast Decoupled power flow method is implemented in the `fast_decoupled()` method of the `PowerSystem` class. Here's a breakdown of the key steps:

1. **Initialization**:
   - Set up arrays for bus voltages, angles, and power values
   - Process bus data to determine bus types (slack, PV, PQ)

2. **Matrix Formation**:
   - Form the $B^{\prime}$ matrix for non-slack buses (PV and PQ)
   - Form the $B^{\prime\prime}$ matrix for PQ buses only
   - Invert these matrices (done only once)

3. **Iteration Process**:
   - Track bus type indices
   - Calculate power mismatches at each bus
   - Normalize mismatches by voltage magnitude
   - Solve for angle corrections using $B^{\prime}$ matrix
   - Update voltage angles
   - Solve for magnitude corrections using $B^{\prime\prime}$ matrix
   - Update voltage magnitudes for PQ buses
   - Check for convergence

4. **Handling Generator Reactive Power Limits**:
   - For PV buses, check if reactive power limits are violated
   - If limits are violated change bus type to PQ

## Code Excerpt

```python
def decouple(self):
    """Power flow solution by Fast Decoupled method"""
    # Initialization and bus data processing
    # ...
    
    # Form the suspetance matrix
    B = np.imag(Ybus)

    # Create a dictionary to track bus types
    bus_type_inds = {
            "PQ": np.where(self.kb == 0)[0].tolist(),
            "PV": np.where(self.kb == 2)[0].tolist(),
            "Slack": np.where(self.kb == 1)[0].tolist(),
        }
    
    kb = self.kb.copy()
    # Start iterations
    while self.maxerror >= accuracy and self.iter <= maxiter:
      # Update P and Q according to current states
       Sc = np.multiply(self.V, np.conj(np.dot(self.V, self.Ybus)))

      # if Q limits are violated in PV buses, set Qg to Qmax or Qmin and change bus type to PQ
      for n in bus_type_inds["PV"]:
         # ...
         kb[n] = 0

      
      # Update bus type dictionary

      # Update B2 and inv(B2)

      
      # Calculate changes in voltage phase and magnitude
      Ddelta = np.linalg.solve(B1, elm_DP)
      DVm = np.linalg.solve(B2, elm_DQ)

      # Add zeros for PV and slack buses in DVm
      # Add a zero for slack bus in Ddelata

      # Update voltage phases and magnitudes
      self.delta += Ddelta
      self.Vm += DVm
      
      # Calculate maximum error
      self.maxerror = np.max([np.max(np.abs(DP)), np.max(np.abs(DQ))])
```

## Advantages and Limitations

### Advantages
- Faster computation per iteration compared to Newton-Raphson
- Constant Jacobian matrices that need to be inverted only once
- Good convergence for most practical power systems
- Memory efficient for large systems

### Limitations
- Slower convergence rate compared to Newton-Raphson
- May have convergence issues for heavily loaded systems
- Not suitable for systems with high R/X ratios
- Performance degrades for ill-conditioned systems

## Usage Example

```python
# Create a power system instance
ps = PowerSystem()

# Set parameters
ps.basemva = 100.0
ps.accuracy = 0.001
ps.maxiter = 20

# Load bus and line data
ps.load_data(busdata, linedata)

# Form the bus admittance matrix
ps.lfybus()

# Run Fast Decoupled power flow
ps.decouple()

# Print results
ps.busout()
```

## References

1. B. Stott and O. Alsac, "Fast Decoupled Load Flow," IEEE Transactions on Power Apparatus and Systems, vol. PAS-93, no. 3, pp. 859-869, 1974.
2. Hadi Saadat, "Power System Analysis," McGraw-Hill, 1999.
3. J. Grainger and W. Stevenson, "Power System Analysis," McGraw-Hill, 1994.