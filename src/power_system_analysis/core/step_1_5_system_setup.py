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
        p_loss = abs(p_from + p_to)  # Use absolute value for losses
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
    
    # 2. Load Data
    all_rows.append(['LOAD DATA'])
    all_rows.append(['Bus', 'P(MW)', 'Q(MVar)'])
    for i in range(len(ss.PQ)):
        bus_idx = ss.PQ.bus.v[i]
        all_rows.append([
            bus_idx,
            ss.PQ.p0.v[i] * ss.config.mva,
            ss.PQ.q0.v[i] * ss.config.mva
        ])
    
    # Add spacing
    all_rows.append([])
    all_rows.append([])
    
    # 3. Generator Data
    all_rows.append(['GENERATOR DATA'])
    all_rows.append(['Bus', 'Type', 'P(MW)', 'Q(MVar)', 'Vset(pu)', 'Is Battery'])
    # PV generators
    for i in range(len(ss.PV)):
        bus_idx = ss.PV.bus.v[i]
        name = str(ss.PV.name.v[i])
        is_battery = "Yes" if "BATT" in name else "No"
        gen_type = "Battery" if "BATT" in name else "PV"
        all_rows.append([
            bus_idx,
            gen_type,
            ss.PV.p0.v[i] * ss.config.mva,
            ss.PV.q.v[i] * ss.config.mva,
            ss.PV.v0.v[i],
            is_battery
        ])
    # Slack generator
    for i in range(len(ss.Slack)):
        bus_idx = ss.Slack.bus.v[i]
        all_rows.append([
            bus_idx,
            'Slack',
            ss.Slack.p.v[i] * ss.config.mva,
            ss.Slack.q.v[i] * ss.config.mva,
            ss.Slack.v0.v[i],
            "No"
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

def setup_ieee14_dynamic(setup=True, battery_bus=None, battery_p=None):
    """Create and configure an IEEE 14-bus system with dynamic models"""
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
    
    # Add battery if specified
    if battery_bus is not None and battery_p is not None:
        # Convert single values to lists for consistent handling
        if not isinstance(battery_bus, list):
            battery_bus = [battery_bus]
        if not isinstance(battery_p, list):
            battery_p = [battery_p]
        
        # Add each battery
        for bus, p in zip(battery_bus, battery_p):
            print(f"Adding {p} MW battery at bus {bus}...")
            add_battery(ss, bus, bus, p)  # Using bus number as idx
    
    if setup:
        # Setup the system
        ss.setup()
        
        # Run initial power flow
        print("Running power flow...")
        ss.PFlow.run()
        print("Power flow converged successfully.")
    
    return ss

def add_battery(ss, bus_idx, idx, p_mw=-10, q_mvar=0):
    """Add a battery with REGCA1 generator and REECA1 controller"""
    # Get the voltage base of the connection bus
    bus_vn = ss.Bus.Vn.v[ss.Bus.idx.v.index(bus_idx)]
    
    # First add PV generator to represent the battery's power injection
    ss.add('PV', {
        'bus': bus_idx,
        'name': f'BATT_{bus_idx}',
        'idx': 100 + idx,
        'Sn': 100.0,  # MVA rating
        'Vn': bus_vn,  # kV rating (matches connection bus voltage)
        'v0': 1.0,    # Initial voltage setpoint
        'p0': p_mw / ss.config.mva,   # Convert MW to per-unit
        'qmax': 50,   # Maximum reactive power
        'qmin': -50,  # Minimum reactive power
        'pmax': 50,   # Maximum active power
        'pmin': -50,  # Minimum active power
        'ra': 0.0,    # Armature resistance
        'xs': 0.1,    # Synchronous reactance
    })
    
    # Add renewable generator (REGCA1)
    ss.add('REGCA1', {
        'idx': 100 + idx,
        'bus': bus_idx,
        'gen': 100 + idx,
        'Sn': 100.0,  # MVA rating
        'Tg': 0.02,   # Converter time constant
        'Rrpwr': 10,  # Power ramp rate limit
        'Brkpt': 0.9, # Breakpoint for low voltage active current management
        'Zerox': 0.4, # Zero crossing voltage
        'Lvpl1': 1.0, # Voltage for LVPL gain of 1.0
    })
    
    # Add renewable exciter control (REECA1)
    ss.add('REECA1', {
        'reg': 100 + idx,  # ID of the generator to control
        'gen': 100 + idx,  # Generator index
        'busr': bus_idx,   # Bus to regulate
        'PFLAG': 0,        # P speed-dependency flag
        'PFFLAG': 0,       # Power factor flag
        'VFLAG': 0,        # Voltage control flag
        'QFLAG': 0,        # Q control flag
        'PQFLAG': 0,       # P/Q priority flag
        'Vdip': 0.9,       # Low voltage threshold
        'Vup': 1.1,        # High voltage threshold
        'Trv': 0.02,       # Voltage measurement filter time constant
        'dbd1': -0.01,     # Deadband for voltage error (lower)
        'dbd2': 0.01,      # Deadband for voltage error (upper)
        'Kqv': 2,          # Reactive power control gain
        'Tp': 0.02,        # Active power filter time constant
        'Tiq': 0.02,       # Reactive power filter time constant
        'Vref1': 1.0,      # Voltage reference 1
        'Vref0': 1.0,      # Voltage reference 0
    })

def generate_all_configurations(mw1, mw2):
    """Generate all possible battery configurations and export to Excel"""
    filename = 'system_data_all_configs.xlsx'
    print(f"Generating all configurations and exporting to {filename}")
    
    # Create Excel writer
    with pd.ExcelWriter(filename, engine='openpyxl', mode='w') as writer:
        # Base case (no batteries)
        print("\nProcessing base case (no batteries)...")
        ss = setup_ieee14_dynamic()
        export_system_data_to_excel(ss, writer, 'Base')
        
        # Single battery configurations
        print("\nProcessing single battery configurations...")
        for bus in tqdm(range(1, 15)):
            sheet_name = f"{bus:02d}"
            ss = setup_ieee14_dynamic(battery_bus=[bus], battery_p=[mw1])
            export_system_data_to_excel(ss, writer, sheet_name)
        
        # Two battery configurations
        print("\nProcessing two battery configurations...")
        for bus1, bus2 in tqdm(itertools.product(range(1, 15), range(1, 15))):
            sheet_name = f"{bus1:02d}{bus2:02d}"
            ss = setup_ieee14_dynamic(battery_bus=[bus1, bus2], battery_p=[mw1, mw2])
            export_system_data_to_excel(ss, writer, sheet_name)
    
    print(f"\nAll configurations have been exported to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate all possible battery configurations for IEEE 14-bus system')
    parser.add_argument('--mw', nargs=2, type=float, required=True, metavar=('MW1', 'MW2'),
                      help='Power output in MW for first and second battery')
    
    args = parser.parse_args()
    generate_all_configurations(args.mw[0], args.mw[1]) 