import andes
import numpy as np
import pandas as pd
from andes.utils.paths import get_case
import os
import shutil
import argparse
import itertools
from tqdm import tqdm

# Define file paths at module level
RAW_FILE = get_case('ieee14/ieee14.raw')
DYR_FILE = get_case('ieee14/ieee14.dyr')

def export_system_data_to_excel(ss, writer, sheet_name):
    """Export power system data to Excel file with all tables in a single sheet"""
    
    # Create lists to store all rows of data
    all_rows = []
    
    # 1. Branch Power Flow Data with Losses
    all_rows.append(['BRANCH POWER FLOW DATA'])
    all_rows.append(['From Bus', 'To Bus', 'P From(MW)', 'Q From(MVar)', 'P To(MW)', 'Q To(MVar)', 'P Loss(MW)', 'Q Loss(MVar)', 'Loading(%)', 'Tap Ratio'])
    for i in range(len(ss.Line)):
        from_bus = ss.Line.bus1.v[i]
        to_bus = ss.Line.bus2.v[i]
        
        # Get bus voltages and angles
        v1 = ss.Bus.v.v[ss.Bus.idx.v.index(from_bus)]
        v2 = ss.Bus.v.v[ss.Bus.idx.v.index(to_bus)]
        a1 = ss.Bus.a.v[ss.Bus.idx.v.index(from_bus)]
        a2 = ss.Bus.a.v[ss.Bus.idx.v.index(to_bus)]
        
        # Get line parameters
        r = ss.Line.r.v[i]
        x = ss.Line.x.v[i]
        tap = ss.Line.tap.v[i] if hasattr(ss.Line, 'tap') else 1.0
        
        # Calculate power flows
        y = 1/(r + 1j*x)
        v1_complex = v1 * np.exp(1j * a1)
        v2_complex = v2 * np.exp(1j * a2)
        
        # Calculate current and power at "from" end
        i_complex = (v1_complex/tap - v2_complex) * y
        s_complex_from = v1_complex/tap * np.conj(i_complex)
        
        # Calculate current and power at "to" end
        i_complex_to = -i_complex  # Current in opposite direction
        s_complex_to = v2_complex * np.conj(i_complex_to)
        
        # Convert to MW/MVar
        p_from = s_complex_from.real * ss.config.mva
        q_from = s_complex_from.imag * ss.config.mva
        p_to = s_complex_to.real * ss.config.mva
        q_to = s_complex_to.imag * ss.config.mva
        
        # Calculate losses (absolute value of sum of powers)
        p_loss = abs(p_from + p_to)
        q_loss = abs(q_from + q_to)
        
        # Calculate line loading
        s_from = abs(s_complex_from) * ss.config.mva
        s_rating = ss.Line.rate_a.v[i] if ss.Line.rate_a.v[i] > 0 else 100
        loading = s_from/s_rating * 100 if s_rating else 0
        
        all_rows.append([
            from_bus, 
            to_bus, 
            p_from, 
            q_from,
            p_to,
            q_to,
            p_loss,
            q_loss,
            loading,
            tap
        ])
    
    # Add spacing
    all_rows.append([])
    all_rows.append([])
    
    # 2. Load Data (excluding batteries)
    all_rows.append(['LOAD DATA'])
    all_rows.append(['Bus', 'P(MW)', 'Q(MVar)', 'Name'])
    for i in range(len(ss.PQ)):
        bus_idx = ss.PQ.bus.v[i]
        name = ss.PQ.name.v[i] if hasattr(ss.PQ, 'name') and i < len(ss.PQ.name.v) else ''
        is_battery = "BATT" in name if name else False
        
        # Only include non-battery loads in the load table
        if not is_battery:
            all_rows.append([
                bus_idx,
                ss.PQ.p0.v[i] * ss.config.mva,
                ss.PQ.q0.v[i] * ss.config.mva,
                name
            ])
    
    # Add spacing
    all_rows.append([])
    all_rows.append([])
    
    # 3. Generator Data (including batteries)
    all_rows.append(['GENERATOR DATA'])
    all_rows.append(['Bus', 'Type', 'P(MW)', 'Q(MVar)', 'Vset(pu)', 'Name'])
    
    # PV generators
    if hasattr(ss, 'PV'):
        for i in range(len(ss.PV)):
            bus_idx = ss.PV.bus.v[i]
            name = ss.PV.name.v[i] if hasattr(ss.PV, 'name') and i < len(ss.PV.name.v) else ''
            all_rows.append([
                bus_idx,
                'PV',
                ss.PV.p0.v[i] * ss.config.mva,
                ss.PV.q.v[i] * ss.config.mva,
                ss.PV.v0.v[i],
                name
            ])
    
    # Slack generator
    for i in range(len(ss.Slack)):
        bus_idx = ss.Slack.bus.v[i]
        name = ss.Slack.name.v[i] if hasattr(ss.Slack, 'name') and i < len(ss.Slack.name.v) else 'Slack'
        all_rows.append([
            bus_idx,
            'Slack',
            ss.Slack.p.v[i] * ss.config.mva,
            ss.Slack.q.v[i] * ss.config.mva,
            ss.Slack.v0.v[i],
            name
        ])
    
    # Battery generators (from PQ model)
    for i in range(len(ss.PQ)):
        bus_idx = ss.PQ.bus.v[i]
        name = ss.PQ.name.v[i] if hasattr(ss.PQ, 'name') and i < len(ss.PQ.name.v) else ''
        is_battery = "BATT" in name if name else False
        
        # Include batteries in the generator table with negated power values
        if is_battery:
            all_rows.append([
                bus_idx,
                'Battery',
                -ss.PQ.p0.v[i] * ss.config.mva,  # Negate to show as generation
                -ss.PQ.q0.v[i] * ss.config.mva,  # Negate to show as generation
                ss.Bus.v.v[ss.Bus.idx.v.index(bus_idx)],  # Use the actual bus voltage
                name
            ])
    
    # Convert to DataFrame and save to Excel
    df = pd.DataFrame(all_rows)
    df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
    
    # Auto-adjust column widths
    worksheet = writer.sheets[sheet_name]
    for idx, col in enumerate(worksheet.columns, 1):
        max_length = 0
        column = [cell.value for cell in col]
        for cell in column:
            try:
                if len(str(cell)) > max_length:
                    max_length = len(str(cell))
            except:
                pass
        adjusted_width = (max_length + 2)
        worksheet.column_dimensions[chr(64 + idx)].width = adjusted_width

def setup_ieee14_dynamic(setup=True, battery_bus=None, battery_p=None, distribute_to_gens=True, distribution_method='equal'):
    """Create and configure an IEEE 14-bus system with dynamic models
    
    Args:
        setup (bool): Whether to setup the system and run power flow
        battery_bus (int or list): Bus number(s) to connect batteries
        battery_p (float or list): Power output(s) of batteries in MW
        distribute_to_gens (bool): Whether to distribute battery output among generators
        distribution_method (str): Method to distribute power changes ('equal' or 'proportional')
    """
    # Load the IEEE 14-bus system with both power flow and dynamic data
    ss = andes.load(RAW_FILE, addfile=DYR_FILE, setup=False)
    
    # Set correct voltage bases
    for i in range(len(ss.Bus)):
        bus_idx = ss.Bus.idx.v[i]
        if bus_idx in [1, 2, 3, 4, 5, 8]:
            ss.Bus.Vn.v[i] = 138.0  # 138 kV base for buses 1-5 and 8
        else:
            ss.Bus.Vn.v[i] = 69.0   # 69 kV base for the rest
    
    # Configure simulation parameters
    ss.config.freq = 60  # System frequency in Hz
    ss.config.mva = 100  # System base MVA
    ss.config.t0 = 0.0   # Start time
    ss.config.tf = 20.0  # End time
    ss.config.fixt = True  # Fix time step
    ss.config.tstep = 0.01  # Time step size
    
    # Get original generation data before adding batteries
    if distribute_to_gens:
        # Get PV generator data
        pv_p_original = [ss.PV.p0.v[i] * ss.config.mva for i in range(len(ss.PV))]
        pv_buses = [ss.PV.bus.v[i] for i in range(len(ss.PV))]
        
        # Get slack generator data
        slack_p_original = [ss.Slack.p0.v[i] * ss.config.mva for i in range(len(ss.Slack))]
        slack_buses = [ss.Slack.bus.v[i] for i in range(len(ss.Slack))]
        
        # Combine all generators
        all_gen_p = pv_p_original + slack_p_original
        total_gen_capacity = sum(all_gen_p)
        num_generators = len(pv_p_original) + len(slack_p_original)
    
    # Add battery if specified, as a PQ element
    total_battery_p = 0
    if battery_bus is not None and battery_p is not None:
        # Convert single values to lists for consistent handling
        if not isinstance(battery_bus, list):
            battery_bus = [battery_bus]
        if not isinstance(battery_p, list):
            battery_p = [battery_p]
        
        # Use an index that guarantees uniqueness even if batteries share the same bus
        for i, (bus, p) in enumerate(zip(battery_bus, battery_p)):
            unique_idx = bus * 10 + i  # Unique index per battery at the same bus
            print(f"Adding {p} MW battery at bus {bus} as PQ element with idx {unique_idx}...")
            add_battery(ss, bus, unique_idx, p)
            total_battery_p += p
    
    # Adjust generator outputs to account for battery injection
    if distribute_to_gens and total_battery_p > 0 and num_generators > 0:
        if distribution_method == 'equal':
            # Distribute battery power equally among all generators
            equal_reduction = total_battery_p / num_generators
            
            # Apply equal reduction to PV generators
            for i in range(len(ss.PV)):
                new_p = max(0.1, pv_p_original[i] - equal_reduction)  # Ensure generators stay on by keeping minimum output
                ss.PV.p0.v[i] = new_p / ss.config.mva
                print(f"Adjusted PV generator at bus {pv_buses[i]} from {pv_p_original[i]:.2f} MW to {new_p:.2f} MW (-{equal_reduction:.2f} MW)")
            
            # Apply equal reduction to slack generator
            for i in range(len(ss.Slack)):
                new_p = max(0.1, slack_p_original[i] - equal_reduction)  # Ensure generators stay on
                ss.Slack.p0.v[i] = new_p / ss.config.mva
                print(f"Adjusted Slack generator at bus {slack_buses[i]} from {slack_p_original[i]:.2f} MW to {new_p:.2f} MW (-{equal_reduction:.2f} MW)")
        
        elif distribution_method == 'proportional':
            # Distribute battery power proportionally among all generators
            scale_factor = 1 - (total_battery_p / total_gen_capacity)
            
            # Apply proportional reduction to PV generators
            for i in range(len(ss.PV)):
                reduction = pv_p_original[i] * (1 - scale_factor)
                new_p = pv_p_original[i] * scale_factor
                ss.PV.p0.v[i] = new_p / ss.config.mva
                print(f"Adjusted PV generator at bus {pv_buses[i]} from {pv_p_original[i]:.2f} MW to {new_p:.2f} MW (-{reduction:.2f} MW)")
            
            # Apply proportional reduction to slack generator
            for i in range(len(ss.Slack)):
                reduction = slack_p_original[i] * (1 - scale_factor)
                new_p = slack_p_original[i] * scale_factor
                ss.Slack.p0.v[i] = new_p / ss.config.mva
                print(f"Adjusted Slack generator at bus {slack_buses[i]} from {slack_p_original[i]:.2f} MW to {new_p:.2f} MW (-{reduction:.2f} MW)")
    
    if setup:
        # Setup the system
        ss.setup()
        
        # Run normal power flow
        print("Running power flow...")
        ss.PFlow.run()
        print("Power flow converged successfully.")
    
    return ss

def add_battery(ss, bus_idx, idx, p_mw=40, q_mvar=0):
    """Add a battery as a PQ element representing its power injection.
    
    The battery is modeled as a PQ injection. For generation (discharge), we use a negative
    p_mw value in the PQ model since PQ is conventionally a load model.
    
    Args:
        ss: ANDES system
        bus_idx: Bus number to connect the battery
        idx: Unique identifier for the battery
        p_mw: Power output in MW (positive means generation/discharge)
        q_mvar: Reactive power in MVar
    """
    # For PQ model, generation is negative (injecting power into the system)
    # So we need to negate the p_mw value if it's positive (generation)
    pq_value = -p_mw / ss.config.mva  # Negate for PQ model (negative = generation)
    
    ss.add('PQ', {
        'bus': bus_idx,
        'name': f'BATT_{bus_idx}_{idx}',
        'idx': 100 + idx,
        'p0': pq_value,   # Negative value for generation in PQ model
        'q0': -q_mvar / ss.config.mva,  # Negate reactive power as well for consistency
    })

def generate_all_configurations(mw1, mw2, distribute_to_gens=True, distribution_method='equal'):
    """Generate all possible battery configurations and export to Excel"""
    filename = 'system_data_all_configs.xlsx'
    print(f"Generating all configurations and exporting to {filename}")
    print(f"Using {distribution_method} distribution method for generator adjustments")
    
    # Create Excel writer
    with pd.ExcelWriter(filename, engine='openpyxl', mode='w') as writer:
        # Base case (no batteries)
        print("\nProcessing base case (no batteries)...")
        ss = setup_ieee14_dynamic(distribute_to_gens=distribute_to_gens, distribution_method=distribution_method)
        export_system_data_to_excel(ss, writer, 'Base')
        
        # Single battery configurations
        print("\nProcessing single battery configurations...")
        for bus in tqdm(range(1, 15)):
            sheet_name = f"{bus:02d}"
            ss = setup_ieee14_dynamic(
                battery_bus=[bus], 
                battery_p=[mw1], 
                distribute_to_gens=distribute_to_gens,
                distribution_method=distribution_method
            )
            export_system_data_to_excel(ss, writer, sheet_name)
        
        # Two battery configurations - eliminate duplicates
        print("\nProcessing two battery configurations (no duplicates)...")
        total_cases = sum(1 for _ in itertools.combinations_with_replacement(range(1, 15), 2))
        with tqdm(total=total_cases) as pbar:
            # Use combinations_with_replacement to get unique pairs (including same bus twice)
            for bus1, bus2 in itertools.combinations_with_replacement(range(1, 15), 2):
                sheet_name = f"{bus1:02d}{bus2:02d}"
                ss = setup_ieee14_dynamic(
                    battery_bus=[bus1, bus2], 
                    battery_p=[mw1, mw2], 
                    distribute_to_gens=distribute_to_gens,
                    distribution_method=distribution_method
                )
                export_system_data_to_excel(ss, writer, sheet_name)
                pbar.update(1)
    
    print(f"\nAll configurations have been exported to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate all possible battery configurations for IEEE 14-bus system')
    parser.add_argument('--mw', nargs=2, type=float, required=True, metavar=('MW1', 'MW2'),
                      help='Power output in MW for first and second battery')
    parser.add_argument('--distribute', action='store_true', default=True,
                      help='Distribute battery output among all generators')
    parser.add_argument('--distribution-method', choices=['equal', 'proportional'], default='equal',
                      help='Method to distribute power changes among generators (equal or proportional)')
    
    args = parser.parse_args()
    generate_all_configurations(
        args.mw[0], 
        args.mw[1], 
        distribute_to_gens=args.distribute,
        distribution_method=args.distribution_method
    )