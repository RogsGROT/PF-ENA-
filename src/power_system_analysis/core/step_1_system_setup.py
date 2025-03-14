import andes
import numpy as np
import pandas as pd
from andes.utils.paths import get_case
import os
import shutil

def setup_ieee14_dynamic(setup=True):
    """Create and configure an IEEE 14-bus system with dynamic models
    
    Args:
        setup (bool, optional): Whether to run setup after loading. Defaults to True.
    """
    # First, modify the base case Excel file to set correct voltage bases
    # Get the path to the original file
    original_file = get_case('ieee14/ieee14_full.xlsx')
    
    # Create a copy in the current directory
    local_file = 'ieee14_modified.xlsx'
    shutil.copy2(original_file, local_file)
    
    # Read all sheets from the Excel file
    xls = pd.ExcelFile(local_file)
    sheets = {}
    for sheet_name in xls.sheet_names:
        sheets[sheet_name] = pd.read_excel(xls, sheet_name=sheet_name)
    
    # Update voltage bases in the Bus sheet
    sheets['Bus']['Vn'] = sheets['Bus'].apply(lambda row: 138.0 if row['idx'] <= 5 else 69.0, axis=1)
    
    # Save all sheets back to Excel
    with pd.ExcelWriter(local_file) as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    # Now load the modified file
    ss = andes.load(local_file, setup=setup)
    
    # Configure simulation parameters
    ss.config.freq = 60  # System frequency in Hz
    ss.config.mva = 100  # System base MVA
    ss.config.t0 = 0.0   # Start time
    ss.config.tf = 20.0  # End time
    ss.config.fixt = True  # Fix time step
    ss.config.tstep = 0.01  # Time step size
    
    if setup:
        # Print transformer information for debugging
        for i in range(len(ss.Line)):
            if hasattr(ss.Line, 'tap') and ss.Line.tap.v[i] != 1.0:
                from_bus = ss.Line.bus1.v[i]
                to_bus = ss.Line.bus2.v[i]
                
                # Get voltage levels
                from_vn = ss.Bus.Vn.v[ss.Bus.idx.v.index(from_bus)]
                to_vn = ss.Bus.Vn.v[ss.Bus.idx.v.index(to_bus)]
                
                print(f"Transformer {ss.Line.idx.v[i]} (Bus {from_bus}-{to_bus}): {from_vn:.1f}/{to_vn:.1f} kV, ratio={ss.Line.tap.v[i]:.4f}")
        
        # Run initial power flow
        print("\nRunning initial power flow...")
        ss.PFlow.run()
        print("Initial power flow converged successfully.")
    
    return ss

def add_battery(ss, bus_idx, idx, p_mw=-10, q_mvar=0):
    """Add a battery with REGCA1 generator and REECA1 controller"""
    # First add PV generator to represent the battery's power injection
    ss.add('PV', {
        'bus': bus_idx,
        'name': f'BATT_{bus_idx}',
        'idx': 100 + idx,
        'Sn': 100.0,  # MVA rating
        'Vn': 69.0,   # kV rating (matches bus 4 voltage)
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

if __name__ == "__main__":
    # Test the system setup
    ss = setup_ieee14_dynamic()
    
    # Test adding a battery
    print("\nTesting battery addition...")
    ss_battery = andes.load(get_case('ieee14/ieee14_full.xlsx'), setup=False)
    add_battery(ss_battery, 4, 0)  # Add a battery at bus 4
    ss_battery.setup()
    ss_battery.PFlow.run()
    print("Power flow with battery converged successfully.")
    
    print("\nSystem setup completed successfully.")