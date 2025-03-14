import andes
import numpy as np
import pandas as pd
import os
import sys

def generate_ena_output(ss, config_name):
    """
    Generate power flow output in format compatible with ENA analysis.
    We'll format this in the Excel structure needed for the MATLAB code.
    """
    # Run power flow if not already run
    if not hasattr(ss, 'dae'):
        ss.PFlow.run()
    
    # Get all buses
    buses = list(range(1, len(ss.Bus)+1))
    
    # Create an empty matrix for the flow data
    n_buses = len(buses)
    flow_matrix = np.zeros((n_buses+3, n_buses+3))
    
    # Fill the matrix with power flow data
    # Format follows the required ENA matrix structure
    
    # Row 0, Column 0 (empty)
    # flow_matrix[0,0] = 0
    
    # Row 0 is imports
    for i, bus in enumerate(buses):
        # Sum of all injections into the bus from outside the system
        # For simplicity, we'll use generator outputs as imports
        for gen in ss.StaticGen.idx.v:
            if ss.StaticGen.bus.v[ss.StaticGen.idx.v.index(gen)] == bus:
                flow_matrix[0, i+1] = ss.StaticGen.p.v[ss.StaticGen.idx.v.index(gen)]
                
        for gen in ss.PV.idx.v:
            if ss.PV.bus.v[ss.PV.idx.v.index(gen)] == bus:
                flow_matrix[0, i+1] = ss.PV.p.v[ss.PV.idx.v.index(gen)]
                
        for gen in ss.Slack.idx.v:
            if ss.Slack.bus.v[ss.Slack.idx.v.index(gen)] == bus:
                flow_matrix[0, i+1] = ss.Slack.p.v[ss.Slack.idx.v.index(gen)]
    
    # Internal flows (from lines)
    for line_idx in ss.Line.idx.v:
        i = ss.Line.idx.v.index(line_idx)
        from_bus = ss.Line.bus1.v[i]
        to_bus = ss.Line.bus2.v[i]
        
        # Get power flow values (absolute values as ENA requires positive flows)
        p_from = abs(ss.Line.get(src='p', idx=line_idx, attr='v'))
        
        # Add to the matrix (from bus to bus)
        flow_matrix[from_bus, to_bus] += p_from
        
    # Add transformer flows
    for trafo_idx in ss.Transformer.idx.v:
        i = ss.Transformer.idx.v.index(trafo_idx)
        from_bus = ss.Transformer.bus1.v[i]
        to_bus = ss.Transformer.bus2.v[i]
        
        # Get power flow values
        p_from = abs(ss.Transformer.get(src='p1', idx=trafo_idx, attr='v'))
        
        # Add to the matrix
        flow_matrix[from_bus, to_bus] += p_from
    
    # Columns n+1 and n+2 are exports and dissipation
    for i, bus in enumerate(buses):
        # Sum of all outgoing power from the bus to outside the system
        # For simplicity, loads are treated as exports
        for load_idx in ss.PQ.idx.v:
            if ss.PQ.bus.v[ss.PQ.idx.v.index(load_idx)] == bus:
                flow_matrix[i+1, n_buses+1] = abs(ss.PQ.p.v[ss.PQ.idx.v.index(load_idx)])
        
        # Losses are assigned to dissipation column
        # For simplicity, we'll use a small percentage of the total flow as losses
        total_flow = sum(flow_matrix[i+1, :])
        flow_matrix[i+1, n_buses+2] = total_flow * 0.03  # Assume 3% losses
    
    # Create a DataFrame for better visualization
    cols = ['0'] + [str(i) for i in buses] + ['Exports', 'Dissipation']
    rows = ['0'] + [str(i) for i in buses] + ['', '']
    
    flow_df = pd.DataFrame(flow_matrix, columns=cols, index=rows)
    
    # Save to Excel in the format expected by MATLAB ENA code
    excel_file = f'ena_input_{config_name}.xlsx'
    with pd.ExcelWriter(excel_file) as writer:
        flow_df.to_excel(writer, sheet_name='main')
    
    print(f"Created ENA input file: {excel_file}")
    return flow_df

if __name__ == "__main__":
    # Check if we have a system file as input
    if len(sys.argv) > 1:
        system_file = sys.argv[1]
        config_name = sys.argv[2] if len(sys.argv) > 2 else 'default'
    else:
        # Default to loading a saved system or creating a new one
        if os.path.exists('ieee14_dynamic.pkl'):
            system_file = 'ieee14_dynamic.pkl'
            config_name = 'base'
        else:
            print("No system file provided. Please run 1_system_setup.py first or provide a system file.")
            sys.exit(1)
    
    print(f"Loading system from {system_file}...")
    try:
        ss = andes.system.System()
        ss = ss.load(system_file)
        
        # Generate ENA output for the system
        generate_ena_output(ss, config_name)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)