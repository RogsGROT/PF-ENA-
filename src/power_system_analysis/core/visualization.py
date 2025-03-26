"""
Power System Visualization Tool

This module provides visualization capabilities for power system networks,
specifically the IEEE 14-bus system with dynamic components.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import pandas as pd
import andes
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from andes.utils.paths import get_case

# Add the src directory to the Python path if running directly
if __name__ == "__main__":
    import pathlib
    src_dir = str(pathlib.Path(__file__).parent.parent.parent)
    if src_dir not in sys.path:
        sys.path.append(src_dir)

# Import the system setup function
from power_system_analysis.core.step_1_system_setup import setup_ieee14_dynamic, add_battery

# Define file paths at module level
RAW_FILE = get_case('ieee14/ieee14.raw')
DYR_FILE = get_case('ieee14/ieee14.dyr')

class PowerSystemVisualizer:
    """
    A class to visualize power system networks, flows, and components.
    """
    
    def __init__(self, system, title="IEEE 14-Bus Power System", custom_layout=None):
        """
        Initialize the visualizer with a power system.
        
        Args:
            system: An ANDES power system object
            title: Title for the visualization
            custom_layout: Optional dictionary mapping bus numbers to (x,y) coordinates
                         e.g., {1: (0,0), 2: (1,0), ...}
        """
        self.system = system
        self.title = title
        
        # Use custom layout if provided, otherwise use default IEEE 14-bus layout
        self.node_positions = custom_layout if custom_layout is not None else {
            1: (0, 0),       # Slack bus
            2: (1, 0),       # Generator at bus 2
            3: (2, 0),       # Generator at bus 3
            4: (2, -1),      
            5: (1, -1),      
            6: (3, 0),       # Generator at bus 6
            7: (3, -1),      
            8: (4, -1),      # Generator at bus 8
            9: (4, -2),      
            10: (5, -2),     
            11: (4, -3),     
            12: (5, -3),     
            13: (4.5, -3.5), 
            14: (5.5, -3),   
        }
        
        # Initialize the graph
        self.graph = nx.DiGraph()
        
        # Process system components
        self._process_system()
        
    def _process_system(self):
        """Process system components and build the graph."""
        # Create an empty graph
        self.graph = nx.DiGraph()

        # Add nodes (buses)
        for i in range(len(self.system.Bus)):
            bus_idx = self.system.Bus.idx.v[i]
            v = self.system.Bus.v.v[i]
            a = self.system.Bus.a.v[i]
            vn = self.system.Bus.Vn.v[i]
            
            # Initialize node attributes
            self.graph.add_node(
                bus_idx,
                voltage=v,
                angle=a,
                vn=vn,
                has_gen=False,
                has_load=False,
                type='bus',
                gen_p=0,
                gen_q=0,
                load_p=0,
                load_q=0
            )

        # Add generator information - use p0 instead of p to match setup_ieee14_dynamic
        for i in range(len(self.system.PV)):
            bus_idx = self.system.PV.bus.v[i]
            p = self.system.PV.p0.v[i] * self.system.config.mva  # Use p0 instead of p
            q = self.system.PV.q.v[i] * self.system.config.mva
            name = str(self.system.PV.name.v[i]) if hasattr(self.system.PV, 'name') and i < len(self.system.PV.name.v) else ''
            
            self.graph.nodes[bus_idx]['has_gen'] = True
            self.graph.nodes[bus_idx]['gen_p'] = p
            self.graph.nodes[bus_idx]['gen_q'] = q
            self.graph.nodes[bus_idx]['type'] = 'generator'

        # Add slack bus information
        for i in range(len(self.system.Slack)):
            bus_idx = self.system.Slack.bus.v[i]
            p = self.system.Slack.p.v[i] * self.system.config.mva
            q = self.system.Slack.q.v[i] * self.system.config.mva
            
            self.graph.nodes[bus_idx]['has_gen'] = True
            self.graph.nodes[bus_idx]['type'] = 'slack'
            self.graph.nodes[bus_idx]['gen_p'] = p
            self.graph.nodes[bus_idx]['gen_q'] = q

        # Process PQ elements (loads and batteries)
        for i in range(len(self.system.PQ)):
            bus_idx = self.system.PQ.bus.v[i]
            p = self.system.PQ.p0.v[i] * self.system.config.mva
            q = self.system.PQ.q0.v[i] * self.system.config.mva
            name = str(self.system.PQ.name.v[i]) if hasattr(self.system.PQ, 'name') and i < len(self.system.PQ.name.v) else ''
            
            if 'BATT_' in name:
                # This is a battery - add it as a generator
                # Note: p and q are already negative in the PQ model for generation
                # We need to negate them to show as positive generation
                self.graph.nodes[bus_idx]['has_gen'] = True
                self.graph.nodes[bus_idx]['gen_p'] = -p  # Negate to show as positive generation
                self.graph.nodes[bus_idx]['gen_q'] = -q  # Negate to show as positive generation
                self.graph.nodes[bus_idx]['type'] = 'battery'
            else:
                # This is a regular load
                self.graph.nodes[bus_idx]['has_load'] = True
                self.graph.nodes[bus_idx]['load_p'] = p
                self.graph.nodes[bus_idx]['load_q'] = q

        # Add lines (edges) - match step_1_system_setup.py calculations
        for i in range(len(self.system.Line)):
            line_idx = self.system.Line.idx.v[i]
            from_bus = self.system.Line.bus1.v[i]
            to_bus = self.system.Line.bus2.v[i]
            
            # Check if this is a transformer
            is_transformer = False
            if hasattr(self.system.Line, 'tap') and self.system.Line.tap.v[i] != 1.0:
                is_transformer = True
            
            # Get tap ratio
            tap = self.system.Line.tap.v[i] if hasattr(self.system.Line, 'tap') else 1.0
            
            # Get bus voltages and angles
            v1 = self.system.Bus.v.v[self.system.Bus.idx.v.index(from_bus)]
            v2 = self.system.Bus.v.v[self.system.Bus.idx.v.index(to_bus)]
            a1 = self.system.Bus.a.v[self.system.Bus.idx.v.index(from_bus)]
            a2 = self.system.Bus.a.v[self.system.Bus.idx.v.index(to_bus)]
            
            # Get line parameters
            r = self.system.Line.r.v[i]
            x = self.system.Line.x.v[i]
            
            # Calculate power flows using complex numbers - identical to step_1_system_setup.py
            y = 1/(r + 1j*x)  # Line admittance
            v1_complex = v1 * np.exp(1j * a1)  # Complex voltage at from bus
            v2_complex = v2 * np.exp(1j * a2)  # Complex voltage at to bus
            
            # Calculate current and complex power flow from bus1 to bus2
            i_complex = (v1_complex/tap - v2_complex) * y
            s_from = v1_complex/tap * np.conj(i_complex)
            
            # Calculate current and power at "to" end
            i_complex_to = (v2_complex - v1_complex/tap) * y  # Current in opposite direction
            s_to = v2_complex * np.conj(i_complex_to)
            
            # Convert to MW and MVar
            p_from = s_from.real * self.system.config.mva
            q_from = s_from.imag * self.system.config.mva
            p_to = s_to.real * self.system.config.mva
            q_to = s_to.imag * self.system.config.mva
            
            # Add edge with attributes
            self.graph.add_edge(
                from_bus, 
                to_bus, 
                id=line_idx,
                p_from=p_from,
                q_from=q_from,
                p_to=p_to,
                q_to=q_to,
                is_transformer=is_transformer
            )
    
    def plot(self):
        """Create a visualization of the power system."""
        # Create figure and axis
        fig1, ax1 = plt.subplots(figsize=(15, 10))
        
        # Use the predefined node positions
        pos = self.node_positions
        
        # Draw nodes
        node_colors = []
        node_sizes = []
        node_shapes = []
        
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node]['type']
            if node_type == 'slack':
                node_colors.append('red')
                node_sizes.append(1000)
                node_shapes.append('s')  # square
            elif node_type == 'generator':
                node_colors.append('lightgreen')
                node_sizes.append(1000)
                node_shapes.append('o')  # circle
            elif node_type == 'battery':
                node_colors.append('yellow')
                node_sizes.append(1000)
                node_shapes.append('h')  # hexagon
            elif node_type == 'load':
                node_colors.append('lightblue')
                node_sizes.append(800)
                node_shapes.append('o')  # circle
            else:  # bus
                node_colors.append('white')
                node_sizes.append(800)
                node_shapes.append('o')  # circle
        
        # Draw nodes by shape
        for shape in set(node_shapes):
            # Get indices of nodes with this shape
            node_idx = [i for i, s in enumerate(node_shapes) if s == shape]
            nodes = [list(self.graph.nodes())[i] for i in node_idx]
            colors = [node_colors[i] for i in node_idx]
            sizes = [node_sizes[i] for i in node_idx]
            
            nx.draw_networkx_nodes(
                self.graph,
                pos,
                nodelist=nodes,
                node_color=colors,
                node_size=sizes,
                node_shape=shape,
                edgecolors='black'
            )
        
        # Draw edges (lines and transformers)
        edge_colors = []
        edge_widths = []
        
        # Create a list to store edge labels
        edge_labels = {}
        
        # Process each edge
        for u, v, data in self.graph.edges(data=True):
            # Determine color and width based on whether it's a transformer
            if data['is_transformer']:
                edge_colors.append('orange')
                edge_widths.append(2)
            else:
                edge_colors.append('black')
                edge_widths.append(1)
            
            # Get the positions of the nodes
            pos_u = pos[u]
            pos_v = pos[v]
            
            # Get power flows
            p_from = data['p_from']
            q_from = data['q_from']
            
            # Create label with complex power flow notation
            label = f"{p_from:.1f}+j{q_from:.1f}"
            
            # Add label to the dictionary
            edge_labels[(u, v)] = label
        
        # Draw the edges with arrows
        nx.draw_networkx_edges(
            self.graph, pos,
            edge_color=edge_colors,
            width=edge_widths,
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            min_source_margin=25,
            min_target_margin=25
        )
        
        # Draw edge labels
        nx.draw_networkx_edge_labels(
            self.graph, pos,
            edge_labels=edge_labels,
            font_size=8
        )
        
        # Add node labels with better positioning
        labels = {}
        for node in self.graph.nodes():
            node_info = []
            node_info.append(f"Bus {node}")
            
            if self.graph.nodes[node]['has_gen']:
                p = self.graph.nodes[node]['gen_p']
                q = self.graph.nodes[node]['gen_q']
                node_info.append(f"G: {p:.1f}+j{q:.1f}")
            
            if self.graph.nodes[node]['has_load']:
                p = self.graph.nodes[node]['load_p']
                q = self.graph.nodes[node]['load_q']
                node_info.append(f"L: {p:.1f}+j{q:.1f}")
            
            v = self.graph.nodes[node]['voltage']
            a = self.graph.nodes[node]['angle'] * 180/np.pi  # Convert to degrees
            node_info.append(f"V: {v:.3f}∠{a:.1f}°")
            
            labels[node] = '\n'.join(node_info)
        
        # Add node labels with better positioning (higher z-order)
        for node, label in labels.items():
            x, y = pos[node]
            plt.annotate(
                label,
                xy=(x, y),
                xytext=(0, 20),  # Offset text 20 points above node
                textcoords='offset points',
                ha='center',
                va='bottom',
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    fc='white',
                    ec='gray',
                    alpha=0.9
                ),
                zorder=5  # Higher z-order than line labels
            )
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
                      markersize=15, label='Slack Bus', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen',
                      markersize=15, label='Generator', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='h', color='w', markerfacecolor='yellow',
                      markersize=15, label='Battery', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
                      markersize=15, label='Load Bus', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                      markersize=15, label='Bus', markeredgecolor='black'),
            plt.Line2D([0], [0], color='orange', label='Transformer'),
            plt.Line2D([0], [0], color='black', label='Line'),
            plt.Line2D([0], [0], marker='>', color='black', label='Power Flow Direction')
        ]
        
        ax1.legend(
            handles=legend_elements,
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            title='Components'
        )
        
        # Set title and adjust layout
        ax1.set_title(self.title, pad=20, fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        
        # Create power flow summary figure
        fig2 = self._create_power_flow_summary()
        
        return fig1, fig2
    
    def plot_voltage_profile(self, figsize=(10, 6)):
        """
        Plot the voltage profile for all buses.
        
        Args:
            figsize: Figure size tuple (width, height)
        """
        # Prepare data
        bus_ids = list(self.graph.nodes())
        voltages = [self.graph.nodes[n]['voltage'] for n in bus_ids]
        voltage_kv = [self.graph.nodes[n]['voltage'] * self.graph.nodes[n]['vn'] for n in bus_ids]
        
        # Sort by bus ID
        sorted_indices = np.argsort(bus_ids)
        bus_ids = [bus_ids[i] for i in sorted_indices]
        voltages = [voltages[i] for i in sorted_indices]
        voltage_kv = [voltage_kv[i] for i in sorted_indices]
        
        # Create figure with twin axes for pu and kV
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # Plot pu voltages
        bars = ax1.bar(bus_ids, voltages, color='skyblue', alpha=0.7)
        ax1.set_xlabel('Bus ID')
        ax1.set_ylabel('Voltage (pu)')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Add upper and lower voltage limits (typically 0.95-1.05 pu)
        ax1.axhline(y=1.05, color='r', linestyle='--', alpha=0.5, label='Upper limit (1.05 pu)')
        ax1.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='Lower limit (0.95 pu)')
        
        # Add values on top of bars
        for bar, voltage in zip(bars, voltages):
            height = bar.get_height()
            ax1.annotate(f'{voltage:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)
        
        # Create twin axis for kV
        ax2 = ax1.twinx()
        ax2.plot(bus_ids, voltage_kv, 'ro-', label='Voltage (kV)')
        ax2.set_ylabel('Voltage (kV)')
        
        # Add values next to points
        for i, v in enumerate(voltage_kv):
            ax2.annotate(f'{v:.1f} kV',
                        xy=(bus_ids[i], v),
                        xytext=(5, 0),  # 5 points horizontal offset
                        textcoords="offset points",
                        ha='left', va='center',
                        fontsize=8,
                        color='red')
        
        # Add legend
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.title('Voltage Profile of the System', fontsize=14)
        plt.tight_layout()
        
        return fig, (ax1, ax2)
    
    def plot_power_injections(self, figsize=(12, 6)):
        """
        Plot power injections (generation and load) at each bus.
        
        Args:
            figsize: Figure size tuple (width, height)
        """
        # Prepare data
        bus_ids = sorted(list(self.graph.nodes()))
        gen_p = [self.graph.nodes[n]['gen_p'] for n in bus_ids]
        load_p = [-self.graph.nodes[n]['load_p'] for n in bus_ids]  # Negative for loads
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Width of bars
        width = 0.35
        
        # Create positions for bars
        x = np.arange(len(bus_ids))
        
        # Plot generation and load bars
        bars1 = ax.bar(x - width/2, gen_p, width, label='Generation (MW)', color='green', alpha=0.7)
        bars2 = ax.bar(x + width/2, load_p, width, label='Load (MW)', color='red', alpha=0.7)
        
        # Add net power as a line
        net_power = [g + l for g, l in zip(gen_p, load_p)]
        ax.plot(x, net_power, 'bo-', label='Net Power (MW)')
        
        # Add labels, title and legend
        ax.set_xlabel('Bus ID')
        ax.set_ylabel('Power (MW)')
        ax.set_title('Power Injections at Each Bus')
        ax.set_xticks(x)
        ax.set_xticklabels(bus_ids)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add values on top of bars
        for bars, values in zip([bars1, bars2], [gen_p, load_p]):
            for bar, value in zip(bars, values):
                height = bar.get_height()
                annotation_height = max(height, 0.1) if height > 0 else min(height, -0.1)
                ax.annotate(f'{value:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, annotation_height),
                            xytext=(0, 3 if height >= 0 else -15),  # offset
                            textcoords="offset points",
                            ha='center', va='bottom' if height >= 0 else 'top',
                            fontsize=8)
        
        # Add net power values
        for i, v in enumerate(net_power):
            ax.annotate(f'{v:.1f}',
                        xy=(x[i], v),
                        xytext=(0, 5),  # 5 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8,
                        color='blue')
        
        plt.tight_layout()
        
        return fig, ax
    
    def _create_power_flow_summary(self):
        """Create a summary table of power flows."""
        # Create figure and axis
        fig = plt.figure(figsize=(12, 16))  # Increased height for three tables
        
        # Create three subplots with appropriate height ratios
        gs = plt.GridSpec(3, 1, height_ratios=[1, 1, 3], hspace=0.8)  # Three tables with more space
        ax1 = plt.subplot(gs[0])  # Generator table
        ax2 = plt.subplot(gs[1])  # Load table
        ax3 = plt.subplot(gs[2])  # Line flows table
        
        ax1.axis('tight')
        ax1.axis('off')
        ax2.axis('tight')
        ax2.axis('off')
        ax3.axis('tight')
        ax3.axis('off')
        
        # First table: Generator outputs - match format from Excel
        gen_data = []
        gen_columns = ['Bus', 'Type', 'P (MW)', 'Q (MVar)']
        
        # Add generator data in sorted order
        for node in sorted(self.graph.nodes()):
            if self.graph.nodes[node]['has_gen']:
                node_type = self.graph.nodes[node]['type']
                p = self.graph.nodes[node]['gen_p']
                q = self.graph.nodes[node]['gen_q']
                
                gen_data.append([
                    f"{node}",
                    node_type.capitalize(),
                    f"{p:.2f}",
                    f"{q:.2f}"
                ])
        
        # Create generator table
        gen_table = ax1.table(
            cellText=gen_data,
            colLabels=gen_columns,
            cellLoc='center',
            loc='center',
            colColours=['lightgray']*len(gen_columns)
        )
        
        # Style the generator table
        gen_table.auto_set_font_size(False)
        gen_table.set_fontsize(9)
        gen_table.scale(1.2, 1.5)
        
        # Second table: Load demands - match format from Excel
        load_data = []
        load_columns = ['Bus', 'P (MW)', 'Q (MVar)']
        
        # Add load data in sorted order
        for node in sorted(self.graph.nodes()):
            if self.graph.nodes[node]['has_load']:
                p = self.graph.nodes[node]['load_p']
                q = self.graph.nodes[node]['load_q']
                
                load_data.append([
                    f"{node}",
                    f"{p:.2f}",
                    f"{q:.2f}"
                ])
        
        # Create load table
        load_table = ax2.table(
            cellText=load_data,
            colLabels=load_columns,
            cellLoc='center',
            loc='center',
            colColours=['lightgray']*len(load_columns)
        )
        
        # Style the load table
        load_table.auto_set_font_size(False)
        load_table.set_fontsize(9)
        load_table.scale(1.2, 1.5)
        
        # Third table: Line flows - match format from Excel
        line_data = []
        line_columns = ['From Bus', 'To Bus', 'P From→To (MW)', 'Q From→To (MVar)', 
                        'P To←From (MW)', 'Q To←From (MVar)', 'Type']
        
        # Add line data in sorted order
        edges = sorted(self.graph.edges(data=True), key=lambda x: (x[0], x[1]))
        for u, v, data_dict in edges:
            p_from = data_dict['p_from']
            q_from = data_dict['q_from']
            p_to = data_dict['p_to']
            q_to = data_dict['q_to']
            
            line_data.append([
                f"{u}",
                f"{v}",
                f"{p_from:.2f}",
                f"{q_from:.2f}",
                f"{p_to:.2f}",
                f"{q_to:.2f}",
                "Transformer" if data_dict['is_transformer'] else "Line"
            ])
        
        # Create line flow table
        line_table = ax3.table(
            cellText=line_data,
            colLabels=line_columns,
            cellLoc='center',
            loc='center',
            colColours=['lightgray']*len(line_columns)
        )
        
        # Style the line flow table
        line_table.auto_set_font_size(False)
        line_table.set_fontsize(9)
        line_table.scale(1.2, 1.5)
        
        # Set titles with padding
        plt.suptitle("Power Flow Summary", fontsize=14, y=0.98)
        ax1.set_title("Generator Outputs", pad=50)
        ax2.set_title("Load Demands", pad=50)
        ax3.set_title("Line and Transformer Flows", pad=50)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        return fig

    def save_visualizations(self, output_dir='visualizations'):
        """
        Save all visualizations to an output directory.
        
        Args:
            output_dir: Directory to save the visualizations
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create and save system diagram
        fig1, fig2 = self.plot()
        fig1.savefig(os.path.join(output_dir, 'system_diagram.png'), dpi=300, bbox_inches='tight')
        fig2.savefig(os.path.join(output_dir, 'power_flow_summary.png'), dpi=300, bbox_inches='tight')
        
        plt.close(fig1)
        plt.close(fig2)
        
        print(f"Visualizations saved to {output_dir}/ directory")


def create_system_visualization(system=None, battery_buses=None, output_dir='visualizations', custom_layout=None):
    """
    Create and save visualizations for a power system.
    
    Args:
        system: ANDES power system object (if None, creates a new one)
        battery_buses: List of bus indices and power values to add batteries to
        output_dir: Directory to save visualizations
        custom_layout: Optional dictionary mapping bus numbers to (x,y) coordinates
    
    Returns:
        The visualizer object and the system object
    """
    if system is None:
        if battery_buses:
            # Use setup_ieee14_dynamic so that battery cases are handled exactly as in step 1
            battery_list = []
            battery_p_list = []
            num_batteries = len(battery_buses) // 2
            for i in range(num_batteries):
                bus = int(battery_buses[i*2])
                power = float(battery_buses[i*2 + 1])
                print(f"Adding a {power} MW battery at bus {bus}...")
                battery_list.append(bus)
                battery_p_list.append(power)
            system = setup_ieee14_dynamic(battery_bus=battery_list, battery_p=battery_p_list, export_excel=False)
        else:
            system = setup_ieee14_dynamic()
    
    # Define grid-based layout if none is provided
    if custom_layout is None:
        grid_layout = {
            12: (1, 0),
            13: (4, 0),
            14: (6, 0),
            11: (3, 3),
            10: (5, 3),
            9: (8, 3),
            6: (0, 4),
            5: (2, 6),
            8: (4, 5),
            7: (6, 5),
            1: (0, 8),
            4: (8, 8),
            2: (2, 10),
            3: (5, 10)
        }
        custom_layout = grid_layout
    
    title = "IEEE 14-Bus Power System"
    if battery_buses:
        battery_info = [f"Bus {battery_buses[i]} ({battery_buses[i+1]} MW)" 
                        for i in range(0, len(battery_buses), 2)]
        title += f" with Batteries at {', '.join(battery_info)}"
    
    visualizer = PowerSystemVisualizer(system, title=title, custom_layout=custom_layout)
    visualizer.save_visualizations(output_dir)
    
    return visualizer, system


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Power System Visualization Tool')
    parser.add_argument('--battery', nargs='+', 
                      help='Bus indices and power values to add batteries to. Format: BUS1 POWER1 BUS2 POWER2 ...')
    parser.add_argument('--output', type=str, default='visualizations', help='Output directory for visualizations')
    parser.add_argument('--system', type=str, help='Path to saved system pickle file')
    
    args = parser.parse_args()
    
    # Process battery arguments
    battery_args = None
    if args.battery:
        if len(args.battery) % 2 != 0:
            print("Error: Battery arguments must be in pairs (BUS POWER)")
            sys.exit(1)
        
        # Convert arguments to proper types
        battery_args = []
        for i in range(0, len(args.battery), 2):
            try:
                bus = int(args.battery[i])
                power = float(args.battery[i + 1])
                battery_args.extend([bus, power])
            except ValueError:
                print(f"Error: Invalid battery argument. Bus must be an integer and power must be a number.")
                sys.exit(1)
    
    # Check if a system file was provided
    system = None
    if args.system and os.path.exists(args.system):
        try:
            print(f"Loading system from {args.system}...")
            system = andes.system.System()
            system = system.load(args.system)
        except Exception as e:
            print(f"Error loading system: {str(e)}")
            system = None
    
    # Create visualizations
    print("Creating power system visualizations...")
    visualizer, system = create_system_visualization(
        system=system,
        battery_buses=battery_args,
        output_dir=args.output
    )
    
    # Show interactive plots if running in interactive mode
    print("\nVisualizations saved. Displaying interactive plots.")
    print("Close plot windows to exit.")
    
    plt.ion()  # Turn on interactive mode
    
    # Show system diagram
    fig1, _ = visualizer.plot()
    
    # Show voltage profile
    fig2, _ = visualizer.plot_voltage_profile()
    
    # Show power injections
    fig3, _ = visualizer.plot_power_injections()
    
    plt.show(block=True)  # Block until all figures are closed
