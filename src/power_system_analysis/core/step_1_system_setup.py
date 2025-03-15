import andes
import numpy as np
from andes.utils.paths import get_case
import os
import shutil
import argparse

# Define file paths at module level
RAW_FILE = get_case('ieee14/ieee14.raw')
DYR_FILE = get_case('ieee14/ieee14.dyr')

def setup_ieee14_dynamic(setup=True):
    """Create and configure an IEEE 14-bus system with dynamic models
    
    Args:
        setup (bool, optional): Whether to run setup after loading. Defaults to True.
    """
    # Load the IEEE 14-bus system with both power flow and dynamic data
    ss = andes.run(RAW_FILE, addfile=DYR_FILE, no_output=True)
    
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

if __name__ == "__main__":
    # Add argument parser
    parser = argparse.ArgumentParser(description='IEEE 14-bus system test with optional battery configuration')
    parser.add_argument('--battery', nargs='+', metavar=('BUS', 'POWER'),
                       help='Add batteries at specified buses with given powers in MW. Format: BUS1 POWER1 BUS2 POWER2 ... Example: --battery 4 20 5 30')
    args = parser.parse_args()
    
    print("Starting IEEE 14-bus system test with dynamic models...")
    
    # Initialize the system
    if args.battery:
        # Validate battery arguments
        if len(args.battery) % 2 != 0:
            print("Error: Battery arguments must be in pairs (BUS POWER)")
            exit(1)
            
        print("\nLoading system from ieee14.raw and ieee14.dyr files...")
        ss = andes.load(RAW_FILE, addfile=DYR_FILE, setup=False)
        
        # Add all batteries
        num_batteries = len(args.battery) // 2
        for i in range(num_batteries):
            bus = int(args.battery[i*2])
            power = float(args.battery[i*2 + 1])
            print(f"Adding a {power} MW battery at bus {bus}...")
            add_battery(ss, bus, i, p_mw=power)
        
        # Setup and run power flow
        ss.setup()
        ss.PFlow.run()
        
        # Print power flow results with batteries
        print("\n=== Power Flow Results with Batteries ===")
        
        # Print bus voltages
        print("\nBus Voltages:")
        print("Bus   Voltage(pu)  Angle(deg)  Vn(kV)")
        print("-" * 45)
        for i in range(len(ss.Bus)):
            bus_idx = ss.Bus.idx.v[i]
            v = ss.Bus.v.v[i]
            a = ss.Bus.a.v[i] * 180/np.pi
            vn = ss.Bus.Vn.v[i]
            print(f"{bus_idx:3d}  {v:10.4f}  {a:10.4f}  {vn:7.2f}")
        
        # Print all generator outputs including batteries
        print("\nGenerator and Battery Outputs:")
        print("Bus   Type        P(MW)     Q(MVar)   Vset(pu)")
        print("-" * 55)
        
        # Print regular generators (PV)
        for i in range(len(ss.PV)):
            bus_idx = ss.PV.bus.v[i]
            p = ss.PV.p0.v[i] * ss.config.mva
            q = ss.PV.q.v[i] * ss.config.mva
            v = ss.PV.v0.v[i]
            name = str(ss.PV.name.v[i])  # Convert to string
            gen_type = "Battery" if "BATT" in name else "Generator"
            print(f"{bus_idx:3d}  {gen_type:<10}  {p:8.2f}  {q:8.2f}  {v:8.4f}")
        
        # Print Slack bus
        for i in range(len(ss.Slack)):
            bus_idx = ss.Slack.bus.v[i]
            p = ss.Slack.p.v[i] * ss.config.mva
            q = ss.Slack.q.v[i] * ss.config.mva
            v = ss.Slack.v0.v[i]
            print(f"{bus_idx:3d}  Slack      {p:8.2f}  {q:8.2f}  {v:8.4f}")
        
        # Print loads
        print("\nLoad Demands:")
        print("Bus   P(MW)     Q(MVar)")
        print("-" * 30)
        for i in range(len(ss.PQ)):
            bus_idx = ss.PQ.bus.v[i]
            p = ss.PQ.p0.v[i] * ss.config.mva
            q = ss.PQ.q0.v[i] * ss.config.mva
            print(f"{bus_idx:3d}  {p:8.2f}  {q:8.2f}")
        
        # Print branch flows
        print("\nBranch Power Flows:")
        print("From  To    P(MW)     Q(MVar)   Loading(%)")
        print("-" * 45)
        
        for i in range(len(ss.Line)):
            from_bus = ss.Line.bus1.v[i]
            to_bus = ss.Line.bus2.v[i]
            
            v1 = ss.Bus.v.v[ss.Bus.idx.v.index(from_bus)]
            v2 = ss.Bus.v.v[ss.Bus.idx.v.index(to_bus)]
            a1 = ss.Bus.a.v[ss.Bus.idx.v.index(from_bus)]
            a2 = ss.Bus.a.v[ss.Bus.idx.v.index(to_bus)]
            
            r = ss.Line.r.v[i]
            x = ss.Line.x.v[i]
            tap = ss.Line.tap.v[i] if hasattr(ss.Line, 'tap') else 1.0
            
            y = 1/(r + 1j*x)
            v1_complex = v1 * np.exp(1j * a1)
            v2_complex = v2 * np.exp(1j * a2)
            i_complex = (v1_complex/tap - v2_complex) * y
            s_complex = v1_complex/tap * np.conj(i_complex)
            
            p = s_complex.real * ss.config.mva
            q = s_complex.imag * ss.config.mva
            s = abs(s_complex) * ss.config.mva
            
            s_rating = ss.Line.rate_a.v[i] if ss.Line.rate_a.v[i] > 0 else 100
            loading = s/s_rating * 100 if s_rating else 0
            
            print(f"{from_bus:3d}  {to_bus:3d}  {p:8.2f}  {q:8.2f}  {loading:8.2f}")
        
        # Print system summary
        print("\nSystem Summary:")
        total_gen_p = sum(ss.PV.p0.v) * ss.config.mva + sum(ss.Slack.p.v) * ss.config.mva
        total_gen_q = sum(ss.PV.q.v) * ss.config.mva + sum(ss.Slack.q.v) * ss.config.mva
        total_load_p = sum(ss.PQ.p0.v) * ss.config.mva
        total_load_q = sum(ss.PQ.q0.v) * ss.config.mva
        losses_p = total_gen_p - total_load_p
        losses_q = total_gen_q - total_load_q
        
        print(f"Total Generation: {total_gen_p:.2f} MW + j{total_gen_q:.2f} MVar")
        print(f"Total Load: {total_load_p:.2f} MW + j{total_load_q:.2f} MVar")
        print(f"Total Losses: {losses_p:.2f} MW + j{losses_q:.2f} MVar")
        
        print("\nBattery System Summary:")
        print(f"Number of PV devices (including batteries): {len(ss.PV)}")
        print(f"Number of REGCA1 devices: {len(ss.REGCA1)}")
        print(f"Number of REECA1 devices: {len(ss.REECA1)}")
        
    else:
        # Run base case without batteries
        print("\nRunning base case (no batteries)...")
        ss = setup_ieee14_dynamic()
    
    print("\nSystem setup completed successfully.")