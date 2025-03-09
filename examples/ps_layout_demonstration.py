# %% imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.power_system import PowerSystem
from src.power_viz import plot_power_system
from src.power_position import get_system_positions

# %% functions
def demonstrate_layouts(ps, save_path=None):
    """
    Demonstrate different layout options for a power system
    
    Parameters:
    -----------
    ps : PowerSystem
        The power system object to visualize
    save_path : str, optional
        Path to save the figure
    """
    # Create a figure with subplots for different layouts
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # List of layout types to demonstrate
    layouts = ['spring', 'spectral', 'kamada_kawai', 'hierarchical', 'grid']
    
    # Plot each layout
    for i, layout in enumerate(layouts):
        ax = fig.add_subplot(gs[i//3, i%3])
        
        # Generate positions using the specified layout algorithm
        pos = get_system_positions(ps, layout_type=layout, seed=42)
        
        # Create the visualization
        plot_power_system(ps, node_positions=pos, 
                         show_values=False, show_line_flows=False, ax=ax)
        
        ax.set_title(f"{layout.capitalize()} Layout")
    
    # Add a title for the whole figure
    if ps.nbus == 30:
        fig.suptitle(f"IEEE 30-Bus System: Layout Algorithm Comparison", fontsize=16)
    else:
        fig.suptitle(f"{ps.nbus}-Bus System: Layout Algorithm Comparison", fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the title
    
    # Save the figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig

def main():
    """Main function to run the demonstration"""
    # Create power system instance
    ps = PowerSystem()
    ps.basemva = 100.0
    ps.accuracy = 0.001
    ps.maxiter = 10

    # IEEE 30-BUS TEST SYSTEM (American Electric Power)
    busdata = [
        [1,   1,    1.06,    0.0,     0.0,   0.0,    0.0,  0.0,   0,   0,       0],
        [2,   2,    1.043,   0.0,   21.70,  12.7,   40.0,  0.0, -40,  50,       0],
        [3,   0,    1.0,     0.0,     2.4,   1.2,    0.0,  0.0,   0,   0,       0],
        [4,   0,    1.06,    0.0,     7.6,   1.6,    0.0,  0.0,   0,   0,       0],
        [5,   2,    1.01,    0.0,    94.2,  19.0,    0.0,  0.0, -40,  40,       0],
        [6,   0,    1.0,     0.0,     0.0,   0.0,    0.0,  0.0,   0,   0,       0],
        [7,   0,    1.0,     0.0,    22.8,  10.9,    0.0,  0.0,   0,   0,       0],
        [8,   2,    1.01,    0.0,    30.0,  30.0,    0.0,  0.0, -10,  60,       0],
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

    # Line data
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
        [6,  28,   .0169,     .0599,   0.065,       1]
    ]

    # Load the data
    ps.load_data(busdata, linedata)

    # Form the bus admittance matrix
    ps.lfybus()

    # Run the Newton-Raphson power flow
    ps.lfnewton()

    # Create and show the layout demonstration
    fig = demonstrate_layouts(ps, save_path="layout_comparison.png")
    plt.show()

    # Now demonstrate on the IEEE 14-bus system
    # Create a simplified 14-bus system
    ps14 = PowerSystem()
    ps14.basemva = 100.0
    ps14.accuracy = 0.001
    ps14.maxiter = 10

    # IEEE 14-BUS TEST SYSTEM
    busdata14 = [
        [1,  1, 1.060, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0],
        [2,  2, 1.045, 0.0, 21.7, 12.7, 40.0, 0.0, -40, 50, 0],
        [3,  2, 1.010, 0.0, 94.2, 19.0, 0.0, 0.0, -40, 40, 0],
        [4,  0, 1.000, 0.0, 47.8, -3.9, 0.0, 0.0, 0, 0, 0],
        [5,  0, 1.000, 0.0, 7.6, 1.6, 0.0, 0.0, 0, 0, 0],
        [6,  2, 1.070, 0.0, 11.2, 7.5, 0.0, 0.0, -6, 24, 0],
        [7,  0, 1.000, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0],
        [8,  2, 1.090, 0.0, 0.0, 0.0, 0.0, 0.0, -6, 24, 0],
        [9,  0, 1.000, 0.0, 29.5, 16.6, 0.0, 0.0, 0, 0, 0],
        [10, 0, 1.000, 0.0, 9.0, 5.8, 0.0, 0.0, 0, 0, 0],
        [11, 0, 1.000, 0.0, 3.5, 1.8, 0.0, 0.0, 0, 0, 0],
        [12, 0, 1.000, 0.0, 6.1, 1.6, 0.0, 0.0, 0, 0, 0],
        [13, 0, 1.000, 0.0, 13.5, 5.8, 0.0, 0.0, 0, 0, 0],
        [14, 0, 1.000, 0.0, 14.9, 5.0, 0.0, 0.0, 0, 0, 0]
    ]

    # Simplified IEEE 14-bus system line data
    linedata14 = [
        [1, 2, 0.01938, 0.05917, 0.02640, 1],
        [1, 5, 0.05403, 0.22304, 0.02190, 1],
        [2, 3, 0.04699, 0.19797, 0.01870, 1],
        [2, 4, 0.05811, 0.17632, 0.01700, 1],
        [2, 5, 0.05695, 0.17388, 0.01730, 1],
        [3, 4, 0.06701, 0.17103, 0.01640, 1],
        [4, 5, 0.01335, 0.04211, 0.00640, 1],
        [4, 7, 0, 0.20912, 0, 0.978],
        [4, 9, 0, 0.55618, 0, 0.969],
        [5, 6, 0, 0.25202, 0, 0.932],
        [6, 11, 0.09498, 0.19890, 0, 1],
        [6, 12, 0.12291, 0.25581, 0, 1],
        [6, 13, 0.06615, 0.13027, 0, 1],
        [7, 8, 0, 0.17615, 0, 1],
        [7, 9, 0, 0.11001, 0, 1],
        [9, 10, 0.03181, 0.08450, 0, 1],
        [9, 14, 0.12711, 0.27038, 0, 1],
        [10, 11, 0.08205, 0.19207, 0, 1],
        [12, 13, 0.22092, 0.19988, 0, 1],
        [13, 14, 0.17093, 0.34802, 0, 1]
    ]

    # Load the data
    ps14.load_data(busdata14, linedata14)

    # Form the bus admittance matrix
    ps14.lfybus()

    # Run the Newton-Raphson power flow
    ps14.lfnewton()

    # Create and show the layout demonstration for IEEE 14-bus system
    fig = demonstrate_layouts(ps14, save_path="layout_comparison_ieee14.png")
    plt.show()

# %% Main function
if __name__ == "__main__":
    main()

# %%
