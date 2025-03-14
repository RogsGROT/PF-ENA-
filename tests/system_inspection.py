"""System inspection and testing module for IEEE 14-bus system."""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.power_system_analysis.core.step_1_system_setup import setup_ieee14_dynamic, add_battery
import pandas as pd
import numpy as np
from tabulate import tabulate  # For nice table output

def inspect_system():
    """
    Load the IEEE 14-bus system and print detailed information about its components
    """
    print("Loading IEEE 14-bus system...")
    # Use the setup function to get correct voltage levels
    ss = setup_ieee14_dynamic()
    
    # Run initial power flow
    ss.PFlow.run()
    
    print("\n" + "="*80)
    print("IEEE 14-BUS SYSTEM OVERVIEW")
    print("="*80)
    
    # Print basic system information
    print(f"\nSystem Base: {ss.config.mva} MVA, {ss.config.freq} Hz")
    print(f"Number of Buses: {len(ss.Bus)}")
    print(f"Number of Lines: {len(ss.Line)}")
    # Count transformers (lines with tap ratio != 1.0)
    n_transformers = sum(1 for i in range(len(ss.Line)) if hasattr(ss.Line, 'tap') and ss.Line.tap.v[i] != 1.0)
    print(f"Number of Transformers: {n_transformers}")
    print(f"Number of Generators: {len(ss.PV) + len(ss.Slack)}")
    print(f"Number of Loads: {len(ss.PQ)}")
    
    # 1. BUS INFORMATION
    print("\n" + "="*80)
    print("BUS INFORMATION")
    print("="*80)
    
    bus_data = []
    for i in range(len(ss.Bus)):
        idx = ss.Bus.idx.v[i]
        name = ss.Bus.name.v[i] if hasattr(ss.Bus, 'name') else f"Bus_{idx}"
        v = ss.Bus.v.v[i]
        vn = ss.Bus.Vn.v[i]
        angle = ss.Bus.a.v[i]
        area = ss.Bus.area.v[i] if hasattr(ss.Bus, 'area') else "N/A"
        zone = ss.Bus.zone.v[i] if hasattr(ss.Bus, 'zone') else "N/A"
        
        # Convert voltage to real units (kV)
        v_kv = v * vn
        
        bus_data.append([idx, name, f"{v:.4f}", f"{v_kv:.1f}", f"{angle:.2f}", area, zone])
    
    headers = ["Bus ID", "Name", "Voltage (pu)", "Voltage (kV)", "Angle (deg)", "Area", "Zone"]
    print(tabulate(bus_data, headers=headers, tablefmt="grid"))
    
    # 2. GENERATOR INFORMATION
    print("\n" + "="*80)
    print("GENERATOR INFORMATION")
    print("="*80)
    
    # First, Slack generator
    print("\nSlack Generator:")
    slack_data = []
    for i in range(len(ss.Slack)):
        idx = ss.Slack.idx.v[i]
        bus = ss.Slack.bus.v[i]
        p = ss.Slack.p.v[i] * ss.config.mva  # Convert to MW
        q = ss.Slack.q.v[i] * ss.config.mva  # Convert to MVar
        v = ss.Slack.v.v[i]
        qmax = ss.Slack.qmax.v[i] * ss.config.mva  # Convert to MVar
        qmin = ss.Slack.qmin.v[i] * ss.config.mva  # Convert to MVar
        pmax = ss.Slack.pmax.v[i] * ss.config.mva if hasattr(ss.Slack, 'pmax') else "N/A"  # Convert to MW
        pmin = ss.Slack.pmin.v[i] * ss.config.mva if hasattr(ss.Slack, 'pmin') else "N/A"  # Convert to MW
        
        slack_data.append([idx, bus, f"{p:.2f}", f"{q:.2f}", f"{v:.4f}", 
                           f"{qmax:.2f}", f"{qmin:.2f}", pmax, pmin])
    
    headers = ["Gen ID", "Bus", "P (MW)", "Q (MVar)", "V (pu)", "Qmax (MVar)", "Qmin (MVar)", "Pmax (MW)", "Pmin (MW)"]
    print(tabulate(slack_data, headers=headers, tablefmt="grid"))
    
    # PV generators
    print("\nPV Generators:")
    pv_data = []
    for i in range(len(ss.PV)):
        idx = ss.PV.idx.v[i]
        bus = ss.PV.bus.v[i]
        p = ss.PV.p.v[i] * ss.config.mva  # Convert to MW
        q = ss.PV.q.v[i] * ss.config.mva  # Convert to MVar
        v = ss.PV.v.v[i]
        qmax = ss.PV.qmax.v[i] * ss.config.mva  # Convert to MVar
        qmin = ss.PV.qmin.v[i] * ss.config.mva  # Convert to MVar
        pmax = ss.PV.pmax.v[i] * ss.config.mva if hasattr(ss.PV, 'pmax') else "N/A"  # Convert to MW
        pmin = ss.PV.pmin.v[i] * ss.config.mva if hasattr(ss.PV, 'pmin') else "N/A"  # Convert to MW
        
        pv_data.append([idx, bus, f"{p:.2f}", f"{q:.2f}", f"{v:.4f}", 
                       f"{qmax:.2f}", f"{qmin:.2f}", pmax, pmin])
    
    print(tabulate(pv_data, headers=headers, tablefmt="grid"))
    
    # 3. DYNAMIC GENERATOR MODELS
    print("\n" + "="*80)
    print("DYNAMIC GENERATOR MODELS")
    print("="*80)
    
    # Check if GENROU models exist
    if hasattr(ss, 'GENROU'):
        print("\nGENROU Models (Round Rotor Generator):")
        genrou_data = []
        for i in range(len(ss.GENROU)):
            idx = ss.GENROU.idx.v[i]
            gen = ss.GENROU.gen.v[i]
            bus_idx = None
            # Find the bus for this generator
            for j in range(len(ss.PV)):
                if ss.PV.idx.v[j] == gen:
                    bus_idx = ss.PV.bus.v[j]
                    break
            for j in range(len(ss.Slack)):
                if ss.Slack.idx.v[j] == gen:
                    bus_idx = ss.Slack.bus.v[j]
                    break
                
            xd = ss.GENROU.xd.v[i]
            xq = ss.GENROU.xq.v[i]
            xdp = ss.GENROU.xd1.v[i]  # xd1 is xd'
            xqp = ss.GENROU.xq1.v[i]  # xq1 is xq'
            H = ss.GENROU.M.v[i] / 2  # Convert M to H
            
            genrou_data.append([idx, gen, bus_idx, f"{xd:.4f}", f"{xq:.4f}", 
                              f"{xdp:.4f}", f"{xqp:.4f}", f"{H:.2f}"])
        
        headers = ["Model ID", "Gen ID", "Bus", "Xd (pu)", "Xq (pu)", "Xd' (pu)", "Xq' (pu)", "H (s)"]
        print(tabulate(genrou_data, headers=headers, tablefmt="grid"))
    
    # Check if governor models exist (TGOV1)
    if hasattr(ss, 'TGOV1'):
        print("\nTGOV1 Models (Turbine Governor):")
        tgov_data = []
        for i in range(len(ss.TGOV1)):
            idx = ss.TGOV1.idx.v[i]
            syn = ss.TGOV1.syn.v[i]  # syn is the generator index in TGOV1
            R = ss.TGOV1.R.v[i]
            T1 = ss.TGOV1.T1.v[i]
            T2 = ss.TGOV1.T2.v[i]
            
            tgov_data.append([idx, syn, f"{R:.4f}", f"{T1:.4f}", f"{T2:.4f}"])
        
        headers = ["Model ID", "Gen ID", "R (pu)", "T1 (s)", "T2 (s)"]
        print(tabulate(tgov_data, headers=headers, tablefmt="grid"))
    
    # 4. LOAD INFORMATION
    print("\n" + "="*80)
    print("LOAD INFORMATION")
    print("="*80)
    
    load_data = []
    for i in range(len(ss.PQ)):
        idx = ss.PQ.idx.v[i]
        bus = ss.PQ.bus.v[i]
        p = ss.PQ.p0.v[i] * ss.config.mva  # Convert to MW
        q = ss.PQ.q0.v[i] * ss.config.mva  # Convert to MVar
        
        load_data.append([idx, bus, f"{p:.2f}", f"{q:.2f}"])
    
    headers = ["Load ID", "Bus", "P (MW)", "Q (MVar)"]
    print(tabulate(load_data, headers=headers, tablefmt="grid"))
    
    # 5. LINE INFORMATION
    print("\n" + "="*80)
    print("LINE INFORMATION")
    print("="*80)
    
    line_data = []
    for i in range(len(ss.Line)):
        idx = ss.Line.idx.v[i]
        from_bus = ss.Line.bus1.v[i]
        to_bus = ss.Line.bus2.v[i]
        r = ss.Line.r.v[i]
        x = ss.Line.x.v[i]
        b = ss.Line.b.v[i]
        rate_a = ss.Line.rate_a.v[i] * ss.config.mva if hasattr(ss.Line, 'rate_a') else "N/A"  # Convert to MVA
        
        line_data.append([idx, from_bus, to_bus, f"{r:.4f}", f"{x:.4f}", f"{b:.4f}", rate_a])
    
    headers = ["Line ID", "From Bus", "To Bus", "R (pu)", "X (pu)", "B (pu)", "Rating (MVA)"]
    print(tabulate(line_data, headers=headers, tablefmt="grid"))
    
    # 6. TRANSFORMER INFORMATION
    print("\n" + "="*80)
    print("TRANSFORMER INFORMATION")
    print("="*80)
    
    transformer_data = []
    for i in range(len(ss.Line)):
        # Only include lines that have tap ratios != 1.0 (these are transformers)
        if hasattr(ss.Line, 'tap') and ss.Line.tap.v[i] != 1.0:
            idx = ss.Line.idx.v[i]
            from_bus = ss.Line.bus1.v[i]
            to_bus = ss.Line.bus2.v[i]
            r = ss.Line.r.v[i]
            x = ss.Line.x.v[i]
            ratio = ss.Line.tap.v[i]
            angle = ss.Line.phi.v[i] if hasattr(ss.Line, 'phi') else 0.0
            rate_a = ss.Line.rate_a.v[i] * ss.config.mva if hasattr(ss.Line, 'rate_a') else "N/A"  # Convert to MVA
            
            # Get voltage levels
            from_vn = ss.Bus.Vn.v[ss.Bus.idx.v.index(from_bus)]
            to_vn = ss.Bus.Vn.v[ss.Bus.idx.v.index(to_bus)]
            
            transformer_data.append([idx, from_bus, to_bus, f"{from_vn:.1f}", f"{to_vn:.1f}", 
                                  f"{r:.4f}", f"{x:.4f}", f"{ratio:.4f}", f"{angle:.2f}", rate_a])
    
    headers = ["Transformer ID", "From Bus", "To Bus", "From kV", "To kV", 
              "R (pu)", "X (pu)", "Ratio", "Angle (deg)", "Rating (MVA)"]
    print(tabulate(transformer_data, headers=headers, tablefmt="grid"))
    
    # 7. POWER FLOW SUMMARY
    print("\n" + "="*80)
    print("POWER FLOW SUMMARY")
    print("="*80)
    
    # Total generation
    total_p_gen = (sum(ss.PV.p.v) + sum(ss.Slack.p.v)) * ss.config.mva  # Convert to MW
    total_q_gen = (sum(ss.PV.q.v) + sum(ss.Slack.q.v)) * ss.config.mva  # Convert to MVar
    
    # Total load
    total_p_load = sum(ss.PQ.p0.v) * ss.config.mva  # Convert to MW
    total_q_load = sum(ss.PQ.q0.v) * ss.config.mva  # Convert to MVar
    
    # Calculate losses (generation - load)
    p_losses = total_p_gen - total_p_load
    q_losses = total_q_gen - total_q_load
    
    print(f"Total Generation: {total_p_gen:.2f} MW, {total_q_gen:.2f} MVar")
    print(f"Total Load: {total_p_load:.2f} MW, {total_q_load:.2f} MVar")
    print(f"System Losses: {p_losses:.2f} MW, {q_losses:.2f} MVar")
    print(f"Loss Percentage: {100 * p_losses / total_p_gen:.2f}%")
    
    return ss

if __name__ == "__main__":
    # Inspect the system and get detailed information
    ss = inspect_system()
    
    # Now add a battery and see how it changes the system
    print("\n\n" + "="*80)
    print("SYSTEM WITH BATTERY ADDED")
    print("="*80)
    
    # Create a fresh copy of the system
    ss_battery = setup_ieee14_dynamic(setup=False)
    
    # Add battery and run power flow
    print("\nAdding a battery at Bus 4...")
    add_battery(ss_battery, 4, 0, p_mw=10)  # Add a 10 MW battery at bus 4
    ss_battery.setup()  # Setup the system after adding the battery
    
    # Print initial conditions before power flow
    print("\nInitial conditions before power flow:")
    print(f"Total PV Generation: {sum(ss_battery.PV.p.v) * ss_battery.config.mva:.2f} MW")
    print(f"Total Slack Generation: {sum(ss_battery.Slack.p.v) * ss_battery.config.mva:.2f} MW")
    
    ss_battery.PFlow.run()
    print("Power flow with battery converged successfully.")
    
    # Print final conditions after power flow
    print("\nFinal conditions after power flow:")
    print(f"Total PV Generation: {sum(ss_battery.PV.p.v) * ss_battery.config.mva:.2f} MW")
    print(f"Total Slack Generation: {sum(ss_battery.Slack.p.v) * ss_battery.config.mva:.2f} MW")
    
    # Show generation before and after
    print("\nGeneration Comparison (Before vs After Battery Addition):")
    
    # Calculate total generation before
    total_p_gen_before = (sum(ss.PV.p.v) + sum(ss.Slack.p.v)) * ss.config.mva  # Convert to MW
    slack_p_before = ss.Slack.p.v[0] * ss.config.mva  # Convert to MW
    
    # Calculate total generation after
    total_p_gen_after = (sum(ss_battery.PV.p.v) + sum(ss_battery.Slack.p.v)) * ss_battery.config.mva  # Convert to MW
    slack_p_after = ss_battery.Slack.p.v[0] * ss_battery.config.mva  # Convert to MW
    
    # Find the battery's generation (it should be the last PV generator)
    battery_p = ss_battery.PV.p.v[-1] * ss_battery.config.mva  # Convert to MW
    
    print(f"Total Generation Before: {total_p_gen_before:.2f} MW")
    print(f"Total Generation After: {total_p_gen_after:.2f} MW")
    print(f"Battery Generation: {battery_p:.2f} MW")
    print(f"Slack Bus Generation Before: {slack_p_before:.2f} MW")
    print(f"Slack Bus Generation After: {slack_p_after:.2f} MW")
    print(f"Change in Slack Generation: {slack_p_after - slack_p_before:.2f} MW")
    
    # Print detailed PV generation breakdown
    print("\nPV Generation Breakdown:")
    print("Before:")
    for i in range(len(ss.PV)):
        p = ss.PV.p.v[i] * ss.config.mva
        print(f"PV {i+1}: {p:.2f} MW")
    
    print("\nAfter:")
    for i in range(len(ss_battery.PV)):
        p = ss_battery.PV.p.v[i] * ss_battery.config.mva
        print(f"PV {i+1}: {p:.2f} MW")
    
    # Compare voltages
    print("\nVoltage Profile Comparison at Selected Buses:")
    voltage_data = []
    for bus_id in [1, 4, 5, 9, 14]:  # Selected buses of interest
        idx = ss.Bus.idx.v.index(bus_id)
        v_before = ss.Bus.v.v[idx]
        
        idx_after = ss_battery.Bus.idx.v.index(bus_id)
        v_after = ss_battery.Bus.v.v[idx_after]
        
        voltage_data.append([bus_id, f"{v_before:.4f}", f"{v_after:.4f}", 
                            f"{100*(v_after-v_before)/v_before:.2f}%"])
    
    headers = ["Bus ID", "Voltage Before (pu)", "Voltage After (pu)", "Change (%)"]
    print(tabulate(voltage_data, headers=headers, tablefmt="grid"))