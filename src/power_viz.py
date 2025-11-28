import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import Normalize


def plot_power_system(
    ps,
    node_positions=None,
    figsize=(12, 10),
    show_values=True,
    show_line_flows=True,
    show_bus_labels=True,
    node_size=300,
    ax=None,
    colormap="viridis",
    title="Power System Network",
):
    """
    Visualize the power system network

    Parameters:
    -----------
    ps : PowerSystem
        The power system object with solution
    node_positions : dict, optional
        Dictionary of node positions {bus_num: (x, y)}
    figsize : tuple, optional
        Figure size
    show_values : bool, optional
        Show voltage values
    show_line_flows : bool, optional
        Show line flow values
    show_bus_labels : bool, optional
        Show bus labels
    node_size : int, optional
        Size of nodes
    ax : matplotlib.axes.Axes, optional
        Axes to plot on; if None, a new figure will be created
    colormap : str, optional
        Matplotlib colormap name for coloring lines
    title : str, optional
        Title for the plot

    Returns:
    --------
    matplotlib.axes.Axes
        The axes containing the plot
    """
    # Create graph
    G = nx.Graph()

    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Add nodes
    for i in range(int(ps.nbus)):
        bus_num = i + 1  # 1-indexed bus number
        bus_type = ps.kb[i]

        # Determine node color by bus type
        if bus_type == 1:  # Slack bus
            node_color = "red"
            node_shape = "s"  # square
        elif bus_type == 2:  # PV (generator) bus
            node_color = "green"
            node_shape = "o"  # circle
        else:  # PQ (load) bus
            node_color = "blue"
            node_shape = "o"  # circle

        # Add node with attributes
        G.add_node(
            bus_num,
            color=node_color,
            shape=node_shape,
            voltage=ps.Vm[i],
            angle=ps.deltad[i],
            Pd=ps.Pd[i],
            Qd=ps.Qd[i],
            Pg=ps.Pg[i],
            Qg=ps.Qg[i],
            bus_type=bus_type,
        )

    # Add edges
    edge_colors = []
    edge_widths = []
    edge_data = []

    for i in range(int(ps.nbr)):
        from_bus = int(ps.nl[i])
        to_bus = int(ps.nr[i])

        # Edge attributes
        r = ps.R[i]
        x = ps.X[i]
        b = ps.Bc[i].imag if hasattr(ps.Bc[i], "imag") else ps.Bc[i]
        tap = ps.a[i]

        # Handle zero or near-zero resistance
        if abs(r) < 1e-6:
            # For transformer or zero-resistance lines, use a fixed value
            xr_ratio = 10.0 if abs(x) > 1e-6 else 0.0
        else:
            # Normal case - calculate X/R ratio
            xr_ratio = abs(x / r)

        # Limit extreme values for better color mapping
        xr_ratio = min(xr_ratio, 10.0)  # Cap at 10 for better color distribution

        # Add edge with attributes
        G.add_edge(from_bus, to_bus, r=r, x=x, b=b, tap=tap, xr_ratio=xr_ratio)

        # Store data for coloring
        edge_colors.append(xr_ratio)

        # Line thickness based on impedance (lower impedance = thicker line)
        impedance = np.sqrt(r**2 + x**2)
        impedance = max(impedance, 0.01)  # Avoid zero impedance
        line_width = 3 * (1 / (1 + impedance * 10))  # Scale for visibility
        edge_widths.append(line_width)

        # Store edge data for custom drawing
        edge_data.append((from_bus, to_bus, xr_ratio, line_width, tap))

    # Set positions
    if node_positions is None:
        # Use spring layout if positions not provided
        pos = nx.spring_layout(G, seed=42)
    else:
        pos = node_positions

    # Draw nodes based on type
    for bus_type, color in [(1, "red"), (2, "green"), (0, "blue")]:
        node_list = [
            n for n, data in G.nodes(data=True) if data["bus_type"] == bus_type
        ]
        if node_list:  # Only draw if there are nodes of this type
            nx.draw_networkx_nodes(
                G, pos, nodelist=node_list, node_color=color, node_size=node_size, ax=ax
            )

    # Create colormap for edges with safe min/max values
    min_xr = 0.0  # Start from zero for better color range
    max_xr = 10.0  # Cap at 10 for transformers and high X/R ratios
    norm = Normalize(vmin=min_xr, vmax=max_xr)

    # Custom drawing of edges with transformers
    for from_bus, to_bus, xr_ratio, width, tap in edge_data:
        try:
            # Get positions
            x1, y1 = pos[from_bus]
            x2, y2 = pos[to_bus]

            # Draw the line with color based on X/R ratio
            color = plt.cm.get_cmap(colormap)(norm(xr_ratio))

            # If it's a transformer (tap != 1), add a small circle in the middle
            if abs(tap - 1.0) > 1e-6:
                # Draw main line
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, zorder=1)

                # Draw transformer symbol
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                # Much smaller size (0.005 instead of 0.02), black outline, orange fill
                circle = plt.Circle(
                    (mid_x, mid_y),
                    0.008,
                    edgecolor="black",
                    facecolor="orange",
                    linewidth=1,
                    zorder=3,
                    alpha=0.5,
                )
                ax.add_patch(circle)
            else:
                # Regular line
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, zorder=1)
        except Exception as e:
            print(f"Error drawing edge {from_bus}-{to_bus}: {e}")

    # Add voltage labels
    if show_values:
        voltage_labels = {}
        for bus_num, data in G.nodes(data=True):
            i = bus_num - 1  # 0-indexed
            voltage_labels[bus_num] = f"{data['voltage']:.3f}"

        # Position the voltage labels slightly above the nodes
        voltage_pos = {node: (pos[node][0], pos[node][1] + 0.05) for node in pos}
        nx.draw_networkx_labels(
            G,
            voltage_pos,
            labels=voltage_labels,
            font_size=8,
            font_color="black",
            ax=ax,
        )

    # Add bus labels
    if show_bus_labels:
        bus_labels = {node: str(node) for node in G.nodes()}
        nx.draw_networkx_labels(
            G,
            pos,
            labels=bus_labels,
            font_size=10,
            font_color="white",
            font_weight="bold",
            ax=ax,
        )

    # Add line flow labels
    if show_line_flows and hasattr(ps, "V"):
        line_labels = {}

        for i in range(int(ps.nbr)):
            try:
                from_bus = int(ps.nl[i])
                to_bus = int(ps.nr[i])

                # Calculate line flow
                n_idx = from_bus - 1  # 0-indexed
                k_idx = to_bus - 1

                # Calculate line flows (simplified)
                if ps.a[i] == 1:  # Regular line
                    y_ik = ps.y[i]
                    flow = abs(ps.V[n_idx] - ps.V[k_idx]) * abs(y_ik) * ps.basemva
                else:  # Transformer
                    flow = (
                        abs(ps.V[n_idx] - ps.V[k_idx] / ps.a[i])
                        * abs(ps.y[i])
                        * ps.basemva
                    )

                line_labels[(from_bus, to_bus)] = f"{flow:.1f} MVA"
            except Exception as e:
                # Skip if flow calculation fails
                print(f"Error calculating flow for line {i+1}: {e}")

        # Position the flow labels at the middle of the edges
        edge_pos = {
            (u, v): (0.5 * (pos[u][0] + pos[v][0]), 0.5 * (pos[u][1] + pos[v][1]))
            for u, v in G.edges()
        }

        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=line_labels, font_size=7, ax=ax
        )

    # Add a colorbar
    fig = ax.get_figure()
    sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(colormap), norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label="X/R Ratio")

    # Add legend
    ax.plot([], [], "rs", markersize=10, label="Slack Bus")
    ax.plot([], [], "go", markersize=10, label="Generator Bus")
    ax.plot([], [], "bo", markersize=10, label="Load Bus")
    ax.plot(
        [],
        [],
        marker="o",
        markersize=10,
        markerfacecolor="orange",
        markeredgecolor="black",
        label="Transformer",
    )
    ax.legend(loc="best")

    ax.set_title(title)
    ax.axis("off")

    return ax


def plot_comparative_visualizations(
    ps, layout_types=["spectral", "hierarchical"], figsize=(15, 8), save_path=None
):
    """
    Create a comparison of different visualizations of the same power system.

    Parameters:
    -----------
    ps : PowerSystem
        The power system object
    layout_types : list, optional
        List of layout algorithms to compare
    figsize : tuple, optional
        Figure size
    save_path : str, optional
        Path to save the figure

    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plots
    """
    from power_positions import get_system_positions

    # Create figure with subplots
    n_layouts = len(layout_types)
    fig, axes = plt.subplots(1, n_layouts, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # Plot each layout
    for i, layout in enumerate(layout_types):
        # Generate positions
        pos = get_system_positions(ps, layout_type=layout, seed=42)

        # Plot on the specific subplot
        plot_power_system(
            ps,
            node_positions=pos,
            show_values=False,
            show_line_flows=True,
            ax=axes[i],
            title=f"{layout.capitalize()} Layout",
        )

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def animate_power_system(ps, frames=30, interval=100, save_path=None):
    """
    Create an animated visualization of a power system where the layout evolves.

    Parameters:
    -----------
    ps : PowerSystem
        The power system object
    frames : int, optional
        Number of animation frames
    interval : int, optional
        Time between frames in milliseconds
    save_path : str, optional
        Path to save the animation as a GIF

    Returns:
    --------
    matplotlib.animation.FuncAnimation
        The animation object
    """
    from matplotlib.animation import FuncAnimation
    from power_positions import get_system_positions

    # Create initial figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get initial positions from spectral layout
    initial_pos = get_system_positions(ps, layout_type="spectral")

    # Get target positions from hierarchical layout
    target_pos = get_system_positions(ps, layout_type="hierarchical")

    # Create a function to interpolate between layouts
    def update(frame):
        # Clear previous frame
        ax.clear()

        # Calculate interpolated positions
        t = frame / (frames - 1)  # Interpolation parameter [0, 1]
        current_pos = {}

        for node in initial_pos:
            x1, y1 = initial_pos[node]
            x2, y2 = target_pos[node]
            current_pos[node] = (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

        # Plot the current state
        plot_power_system(
            ps,
            node_positions=current_pos,
            show_values=False,
            show_line_flows=False,
            ax=ax,
            title=f"Layout Animation (Frame {frame+1}/{frames})",
        )

        return (ax,)

    # Create animation
    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)

    # Save if requested
    if save_path:
        anim.save(save_path, writer="pillow", fps=10)

    return anim


def create_bus_table(ps):
    """
    Create a Pandas DataFrame with bus information
    """
    import pandas as pd

    data = []
    for i in range(int(ps.nbus)):
        bus_type = "Slack" if ps.kb[i] == 1 else "PV" if ps.kb[i] == 2 else "PQ"

        data.append(
            {
                "Bus": i + 1,
                "Type": bus_type,
                "Voltage (pu)": ps.Vm[i],
                "Angle (deg)": ps.deltad[i],
                "Generation (MW)": ps.Pg[i],
                "Generation (Mvar)": ps.Qg[i],
                "Load (MW)": ps.Pd[i],
                "Load (Mvar)": ps.Qd[i],
                "Shunt (Mvar)": ps.Qsh[i],
            }
        )

    df = pd.DataFrame(data)
    return df


def create_line_table(ps):
    """
    Create a Pandas DataFrame with line information
    """
    import pandas as pd

    data = []
    for i in range(int(ps.nbr)):
        from_bus = int(ps.nl[i])
        to_bus = int(ps.nr[i])

        # Calculate line impedance
        z_mag = np.sqrt(ps.R[i] ** 2 + ps.X[i] ** 2)
        z_ang = np.degrees(np.arctan2(ps.X[i], ps.R[i]))

        data.append(
            {
                "From Bus": from_bus,
                "To Bus": to_bus,
                "R (pu)": ps.R[i],
                "X (pu)": ps.X[i],
                "B (pu)": ps.Bc[i].imag if hasattr(ps.Bc[i], "imag") else ps.Bc[i],
                "Tap Ratio": ps.a[i],
                "Z (pu)": f"{z_mag:.4f}∠{z_ang:.2f}°",
                "Line Type": "Transformer" if ps.a[i] != 1 else "Line",
            }
        )

    df = pd.DataFrame(data)
    return df


# Example of custom bus positions for IEEE 30-bus system
def get_ieee30_positions():
    """
    Returns predefined positions for IEEE 30-bus system
    These are approximate positions based on the typical IEEE 30-bus diagram
    """
    positions = {
        1: (0, 0),
        2: (1, 0),
        3: (0, -1),
        4: (1, -1),
        5: (2, 0),
        6: (2, -1),
        7: (3, 0),
        8: (3, -2),
        9: (3, -1),
        10: (4, -1),
        11: (3, -0.5),
        12: (1, -2),
        13: (1, -3),
        14: (0, -2),
        15: (0, -3),
        16: (0, -4),
        17: (3, -3),
        18: (-1, -3),
        19: (-2, -3),
        20: (4, -2),
        21: (5, -1),
        22: (6, -1),
        23: (-1, -4),
        24: (0, -5),
        25: (1, -5),
        26: (2, -5),
        27: (3, -5),
        28: (3, -3),
        29: (4, -5),
        30: (5, -5),
    }
    return positions


def get_26bus_positions():
    """
    Returns predefined positions for 26-bus system
    These are approximate positions based on the typical 20-bus diagram
    """
    positions = {
        1: (2, 0),
        2: (3, 0),
        3: (5, 0),
        4: (3, -2),
        5: (1, -1),
        6: (1, -2),
        7: (2, -2),
        8: (3, -1),
        9: (2, -3),
        10: (3, -4),
        11: (0, -4),
        12: (3, -3),
        13: (5, -1),
        14: (4, -3),
        15: (4, -5),
        16: (5, -3),
        17: (2, -7),
        18: (0, -2),
        19: (2, -4),
        20: (4, -6),
        21: (2, -6),
        22: (3, -6),
        23: (1, -5),
        24: (3, -5),
        25: (1, -4),
        26: (0, -1),
    }
    return positions


# Example usage
# from power_system import PowerSystem
# ps = PowerSystem()
# ps.load_data(busdata, linedata)
# ps.lfybus()
# ps.lfnewton()
#
# # Create visualization
# pos = get_ieee30_positions()
# plot_power_system(ps, node_positions=pos)
# plt.show()
#
# # Create tables
# bus_df = create_bus_table(ps)
# line_df = create_line_table(ps)
#
# print(bus_df)
# print(line_df)
