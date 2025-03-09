import networkx as nx
import numpy as np
from typing import Dict, Tuple, List, Optional, Union


def generate_power_system_positions(ps, layout_type='spectral', geographic_data=None, seed=42):
    """
    Generate positions for power system buses based on different layout algorithms.
    
    Parameters:
    -----------
    ps : PowerSystem
        The power system object with buses and lines
    layout_type : str, optional
        The type of layout algorithm to use:
        - 'spring': Force-directed layout
        - 'spectral': Spectral layout (good for grid-like structures)
        - 'kamada_kawai': Force-directed layout with minimal edge crossings
        - 'hierarchical': Places buses in levels based on shortest path from slack bus
        - 'geographical': Places buses based on provided geographic data
        - 'grid': Simple grid layout with slack, generator, and load buses separated
    geographic_data : dict, optional
        Dictionary mapping bus numbers to (latitude, longitude) tuples,
        required if layout_type='geographical'
    seed : int, optional
        Random seed for reproducible layouts
    
    Returns:
    --------
    dict
        Dictionary mapping bus numbers to (x, y) positions
    """
    # Create a NetworkX graph from the power system
    G = nx.Graph()
    
    # Add nodes (buses)
    for i in range(int(ps.nbus)):
        bus_num = i + 1  # 1-indexed bus number
        bus_type = ps.kb[i]
        G.add_node(bus_num, type=bus_type)
    
    # Add edges (lines)
    for i in range(int(ps.nbr)):
        from_bus = int(ps.nl[i])
        to_bus = int(ps.nr[i])
        impedance = np.sqrt(ps.R[i]**2 + ps.X[i]**2)
        # Use impedance as edge weight (higher impedance = longer edge)
        if impedance < 1e-6:
            # For transformers or zero-impedance lines, use a small value
            impedance = 0.01
        G.add_edge(from_bus, to_bus, weight=impedance, tap=ps.a[i])
    
    # Generate positions based on selected layout algorithm
    if layout_type == 'spring':
        # Force-directed layout with impedance as edge weight
        pos = nx.spring_layout(G, weight='weight', seed=seed)
        
    elif layout_type == 'spectral':
        # Spectral layout - often good for grid-like structures
        pos = nx.spectral_layout(G, weight='weight')
        
    elif layout_type == 'kamada_kawai':
        # Force-directed layout with minimal edge crossings
        pos = nx.kamada_kawai_layout(G, weight='weight')
        
    elif layout_type == 'hierarchical':
        # Custom hierarchical layout based on distance from slack bus
        
        # Find the slack bus
        slack_buses = [n for n, data in G.nodes(data=True) if data['type'] == 1]
        if not slack_buses:
            # If no slack bus, use the first bus
            slack_bus = 1
        else:
            slack_bus = slack_buses[0]
        
        # Calculate shortest path lengths from slack bus
        path_lengths = nx.single_source_shortest_path_length(G, slack_bus)
        
        # Add any buses that aren't connected to the slack bus
        disconnected = set(G.nodes()) - set(path_lengths.keys())
        max_length = max(path_lengths.values()) if path_lengths else 0
        for node in disconnected:
            path_lengths[node] = max_length + 1
        
        # Group nodes by path length (hierarchical levels)
        levels = {}
        for node, length in path_lengths.items():
            if length not in levels:
                levels[length] = []
            levels[length].append(node)
        
        # Position nodes in each level
        pos = {}
        max_level = max(levels.keys())
        
        for level, nodes in levels.items():
            # Sort nodes in each level
            nodes.sort()
            # Calculate y-coordinate based on level
            y = 1.0 - level / (max_level + 1)
            # Position nodes horizontally with equal spacing
            n_nodes = len(nodes)
            for i, node in enumerate(nodes):
                x = (i + 1) / (n_nodes + 1)
                pos[node] = (x, y)
        
    elif layout_type == 'geographical' and geographic_data is not None:
        # Use provided geographic coordinates
        pos = {}
        for bus_num, (lat, lon) in geographic_data.items():
            # Scale latitude and longitude to reasonable range
            pos[bus_num] = (lon, lat)
        
    elif layout_type == 'grid':
        # Grid layout based on bus types
        pos = {}
        
        # Group buses by type
        slack_buses = []
        gen_buses = []
        load_buses = []
        
        for i in range(int(ps.nbus)):
            bus_num = i + 1
            if ps.kb[i] == 1:  # Slack bus
                slack_buses.append(bus_num)
            elif ps.kb[i] == 2:  # Generator bus
                gen_buses.append(bus_num)
            else:  # Load bus
                load_buses.append(bus_num)
        
        # Place slack buses at the top
        n_slack = len(slack_buses)
        for i, bus in enumerate(slack_buses):
            pos[bus] = ((i + 1) / (n_slack + 1), 0.9)
        
        # Place generator buses in the middle
        n_gen = len(gen_buses)
        for i, bus in enumerate(gen_buses):
            pos[bus] = ((i + 1) / (n_gen + 1), 0.6)
        
        # Place load buses at the bottom in a grid pattern
        n_load = len(load_buses)
        grid_width = int(np.ceil(np.sqrt(n_load)))
        
        for i, bus in enumerate(load_buses):
            row = i // grid_width
            col = i % grid_width
            pos[bus] = ((col + 1) / (grid_width + 1), 0.4 - row * 0.2)
        
    else:
        # Default to spring layout if the specified layout type is not valid
        pos = nx.spring_layout(G, seed=seed)
    
    # Scale positions to a nice range
    min_x = min(x for x, y in pos.values())
    max_x = max(x for x, y in pos.values())
    min_y = min(y for x, y in pos.values())
    max_y = max(y for x, y in pos.values())
    
    # Rescale to range [0,1] in both dimensions
    for node in pos:
        x, y = pos[node]
        x_scaled = (x - min_x) / (max_x - min_x) if max_x > min_x else 0.5
        y_scaled = (y - min_y) / (max_y - min_y) if max_y > min_y else 0.5
        pos[node] = (x_scaled, y_scaled)
    
    # Post-processing to improve layout
    # 1. Detect transformers and adjust positions
    for i in range(int(ps.nbr)):
        if abs(ps.a[i] - 1.0) > 1e-6:  # This is a transformer
            from_bus = int(ps.nl[i])
            to_bus = int(ps.nr[i])
            if from_bus in pos and to_bus in pos:
                # Make transformer connections more visible
                # by slightly moving connected buses
                x1, y1 = pos[from_bus]
                x2, y2 = pos[to_bus]
                
                # Calculate the midpoint
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                
                # Move buses slightly away from each other to emphasize the transformer
                move_factor = 0.05
                dir_x = x1 - x2
                dir_y = y1 - y2
                dist = np.sqrt(dir_x**2 + dir_y**2)
                
                if dist > 1e-6:
                    dir_x /= dist
                    dir_y /= dist
                    
                    # Move buses slightly to emphasize transformer
                    pos[from_bus] = (x1 + move_factor * dir_x, y1 + move_factor * dir_y)
                    pos[to_bus] = (x2 - move_factor * dir_x, y2 - move_factor * dir_y)
    
    # 2. Resolve overlaps
    # Simple iterative approach to push overlapping nodes apart
    overlap_margin = 0.05
    max_iters = 50
    for _ in range(max_iters):
        overlaps_resolved = True
        for i in range(int(ps.nbus)):
            bus1 = i + 1
            x1, y1 = pos[bus1]
            
            for j in range(i+1, int(ps.nbus)):
                bus2 = j + 1
                x2, y2 = pos[bus2]
                
                # Check distance
                dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if dist < overlap_margin:
                    # Nodes overlap, push them apart
                    overlaps_resolved = False
                    
                    # Direction vector
                    dir_x = x2 - x1
                    dir_y = y2 - y1
                    
                    if dist < 1e-6:
                        # Nodes at exactly the same position
                        # Move in random direction
                        angle = np.random.uniform(0, 2*np.pi)
                        dir_x = np.cos(angle)
                        dir_y = np.sin(angle)
                        dist = 1e-6
                    else:
                        # Normalize direction
                        dir_x /= dist
                        dir_y /= dist
                    
                    # Calculate push factor (move just enough to resolve overlap)
                    push = (overlap_margin - dist) / 2
                    
                    # Apply push to both nodes
                    pos[bus1] = (x1 - push * dir_x, y1 - push * dir_y)
                    pos[bus2] = (x2 + push * dir_x, y2 + push * dir_y)
        
        if overlaps_resolved:
            break
    
    # 3. Scale to desired range (keep in unit square)
    # After resolving overlaps, make sure we're still in [0,1] range
    min_x = min(x for x, y in pos.values())
    max_x = max(x for x, y in pos.values())
    min_y = min(y for x, y in pos.values())
    max_y = max(y for x, y in pos.values())
    
    # Add a small margin
    margin = 0.05
    scale_x = (1.0 - 2*margin) / (max_x - min_x) if max_x > min_x else 1.0
    scale_y = (1.0 - 2*margin) / (max_y - min_y) if max_y > min_y else 1.0
    
    for node in pos:
        x, y = pos[node]
        x_new = margin + (x - min_x) * scale_x
        y_new = margin + (y - min_y) * scale_y
        pos[node] = (x_new, y_new)
    
    return pos


def get_system_positions(ps, system_name=None, layout_type='spectral', geographic_data=None, seed=42):
    """
    Get positions for a power system visualization.
    
    Parameters:
    -----------
    ps : PowerSystem
        The power system object
    system_name : str, optional
        Name of standard test system ('ieee14', 'ieee30', etc.) for predefined positions
    layout_type : str, optional
        Layout algorithm if predefined positions aren't available or desired:
        - 'spring': Force-directed layout
        - 'spectral': Spectral layout (good for grid-like structures)
        - 'kamada_kawai': Force-directed layout with minimal edge crossings
        - 'hierarchical': Places buses in levels based on shortest path from slack bus
        - 'geographical': Places buses based on provided geographic data
        - 'grid': Simple grid layout with slack, generator, and load buses separated
    geographic_data : dict, optional
        Geographic coordinates if using geographical layout
    seed : int, optional
        Random seed for layout algorithms
    
    Returns:
    --------
    dict
        Dictionary of bus positions {bus_number: (x, y)}
    """
    # Predefined positions for standard test systems
    if system_name == 'ieee30' and ps.nbus == 30:
        # Original predefined positions for IEEE 30-bus system
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
            30: (5, -5)
        }
        
        # Normalize to [0,1] range
        min_x = min(x for x, y in positions.values())
        max_x = max(x for x, y in positions.values())
        min_y = min(y for x, y in positions.values())
        max_y = max(y for x, y in positions.values())
        
        for node in positions:
            x, y = positions[node]
            x_norm = (x - min_x) / (max_x - min_x)
            y_norm = (y - min_y) / (max_y - min_y)
            positions[node] = (x_norm, y_norm)
            
        return positions
        
    elif system_name == 'ieee14' and ps.nbus == 14:
        # Predefined positions for IEEE 14-bus system
        positions = {
            1: (0, 0),
            2: (2, 0),
            3: (3, 0),
            4: (4, 0),
            5: (5, 0),
            6: (3, -2),
            7: (3, -3),
            8: (3, -4),
            9: (6, -2),
            10: (7, -2),
            11: (8, -2),
            12: (8, -3),
            13: (7, -3),
            14: (6, -3)
        }
        
        # Normalize to [0,1] range
        min_x = min(x for x, y in positions.values())
        max_x = max(x for x, y in positions.values())
        min_y = min(y for x, y in positions.values())
        max_y = max(y for x, y in positions.values())
        
        for node in positions:
            x, y = positions[node]
            x_norm = (x - min_x) / (max_x - min_x)
            y_norm = (y - min_y) / (max_y - min_y)
            positions[node] = (x_norm, y_norm)
            
        return positions
    
    # For all other cases, generate positions based on system topology
    return generate_power_system_positions(ps, layout_type=layout_type, 
                                         geographic_data=geographic_data, seed=seed)


# For backward compatibility
def get_ieee30_positions():
    """
    Returns predefined positions for IEEE 30-bus system
    These are approximate positions based on the typical IEEE 30-bus diagram
    
    Note: This function is kept for backward compatibility.
    New code should use get_system_positions() instead.
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


# Example usage function
def demonstrate_layouts(ps):
    """
    Demonstrate different layout options for a power system
    
    Parameters:
    -----------
    ps : PowerSystem
        The power system object to visualize
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from power_viz import plot_power_system
    
    # Create a figure with subplots for different layouts
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # List of layout types to demonstrate
    layouts = ['spring', 'spectral', 'kamada_kawai', 'hierarchical', 'grid']
    
    # Plot each layout
    for i, layout in enumerate(layouts):
        ax = fig.add_subplot(gs[i//3, i%3])
        pos = get_system_positions(ps, layout_type=layout, seed=42)
        plot_power_system(ps, node_positions=pos, figsize=(5, 4), 
                         show_values=False, show_line_flows=False, ax=ax)
        ax.set_title(f"{layout.capitalize()} Layout")
    
    plt.tight_layout()
    return fig