import argparse
import numpy as np
from step_1_system_setup import setup_ieee14_dynamic
import andes
import os
import sys
import pandas as pd
import time

def compute_line_loading(ss, i):
    """
    Compute loading percentage for a specific line in the system.
    
    Args:
        ss: ANDES system object
        i: Line index
        
    Returns:
        loading: Line loading percentage
    """
    from_bus = ss.Line.bus1.v[i]
    to_bus = ss.Line.bus2.v[i]
    try:
        bus1_index = list(ss.Bus.idx.v).index(from_bus)
        bus2_index = list(ss.Bus.idx.v).index(to_bus)
    except Exception as e:
        print(f"Error finding bus index: {e}")
        return 0.0

    v1 = ss.Bus.v.v[bus1_index]
    v2 = ss.Bus.v.v[bus2_index]
    a1 = ss.Bus.a.v[bus1_index] if hasattr(ss.Bus, 'a') else 0.0
    a2 = ss.Bus.a.v[bus2_index] if hasattr(ss.Bus, 'a') else 0.0
    r = ss.Line.r.v[i]
    x = ss.Line.x.v[i]
    tap = ss.Line.tap.v[i] if hasattr(ss.Line, 'tap') else 1.0
    y = 1 / (r + 1j * x)
    v1_complex = v1 * np.exp(1j * a1)
    v2_complex = v2 * np.exp(1j * a2)
    i_complex = (v1_complex / tap - v2_complex) * y
    s_complex_from = (v1_complex / tap) * np.conj(i_complex)
    s_from = abs(s_complex_from) * ss.config.mva
    s_rating = ss.Line.rate_a.v[i] if (hasattr(ss.Line, 'rate_a') and ss.Line.rate_a.v[i] > 0) else 100
    loading = s_from / s_rating * 100 if s_rating != 0 else 0
    return loading

def get_generator_buses(ss):
    """
    Get a list of all generator buses in the system.
    
    Args:
        ss: ANDES system object
        
    Returns:
        list of generator buses and their types (PV or Slack)
    """
    gen_buses = []
    
    # Get Slack generator
    for i in range(len(ss.Slack)):
        bus = ss.Slack.bus.v[i]
        gen_buses.append((bus, 'Slack'))
    
    # Get PV generators
    for i in range(len(ss.PV)):
        bus = ss.PV.bus.v[i]
        gen_buses.append((bus, 'PV'))
    
    # Get GENROU dynamic models
    genrou_buses = set()
    if hasattr(ss, 'GENROU'):
        for i in range(len(ss.GENROU)):
            idx = ss.GENROU.idx.v[i]
            bus = None
            # Find corresponding static generator
            for j in range(len(ss.PV)):
                if ss.PV.idx.v[j] == idx:
                    bus = ss.PV.bus.v[j]
                    break
            if bus:
                genrou_buses.add(bus)
    
    # Print info about GENROU models found
    if genrou_buses:
        print(f"GENROU dynamic models found at buses: {sorted(list(genrou_buses))}")
    
    return gen_buses

def cascade_failure_simulation(initial_trip_gen_bus, overload_threshold=100.0, battery_bus=None, battery_power=None):
    """
    Simulate cascade failure starting with a generator trip.
    
    Args:
        initial_trip_gen_bus: Bus number of the generator to trip initially
        overload_threshold: Line loading percentage threshold for tripping
        battery_bus: List of bus indices where batteries are installed
        battery_power: List of battery power outputs in MW
        
    Returns:
        events: List of events that occurred during the cascade
    """
    events = []
    iteration = 0
    current_time = 5.0  # Start at t=5s

    # Load a fresh system instance for the initial event
    ss_initial = setup_ieee14_dynamic(setup=True, export_excel=False,
                                    battery_bus=battery_bus, battery_p=battery_power)
    
    # Get list of all generators
    gen_buses = get_generator_buses(ss_initial)
    print(f"Generators in the system: {gen_buses}")
    
    # Find the generator index to trip
    initial_gen_idx = None
    initial_gen_type = None
    for i, (bus, gen_type) in enumerate(gen_buses):
        if bus == initial_trip_gen_bus:
            initial_gen_idx = i
            initial_gen_type = gen_type
            break
    
    if initial_gen_idx is None:
        print(f"Error: No generator found at bus {initial_trip_gen_bus}")
        return events, False
    
    # Trip the generator
    if initial_gen_type == 'Slack':
        print(f"Initial event: Tripping Slack generator at bus {initial_trip_gen_bus} at t={current_time:.1f}s")
        # For ANDES, we just set the status of the generator to 0 - this is simpler
        for i in range(len(ss_initial.Slack)):
            if ss_initial.Slack.bus.v[i] == initial_trip_gen_bus:
                print(f"Setting u=0 for Slack generator at bus {initial_trip_gen_bus}")
                # Just disable it without trying to change slack bus assignment
                if hasattr(ss_initial.Slack, 'u'):
                    ss_initial.Slack.u.v[i] = 0
                break
    else:
        # Regular PV generator trip
        print(f"Initial event: Tripping PV generator at bus {initial_trip_gen_bus} at t={current_time:.1f}s")
        
        # Find the generator and trip it
        for i in range(len(ss_initial.PV)):
            if ss_initial.PV.bus.v[i] == initial_trip_gen_bus:
                print(f"Setting u=0 for PV generator at bus {initial_trip_gen_bus}")
                if hasattr(ss_initial.PV, 'u'):
                    ss_initial.PV.u.v[i] = 0
                break
    
    events.append({
        "iteration": iteration, 
        "event": "Initial generator trip", 
        "component": f"Generator at bus {initial_trip_gen_bus} ({initial_gen_type})",
        "time": current_time
    })
    
    # Save the list of tripped components so far
    tripped_generators = {initial_trip_gen_bus}
    tripped_lines = set()
    
    # Run initial power flow to see the immediate effects
    try:
        ss_initial.PFlow.run()
        print("Power flow converged after generator trip.")
    except Exception as e:
        print(f"Power flow failed after generator trip: {e}")
        return events, False
    
    # Advance to t=20s for checking line loadings
    current_time = 20.0
    
    # Check for line overloads after the initial generator trip
    overloaded_lines = []
    for i in range(len(ss_initial.Line)):
        if ss_initial.Line.u.v[i] == 1:
            loading = compute_line_loading(ss_initial, i)
            print(f"Line {ss_initial.Line.idx.v[i]}: loading = {loading:.2f}%")
            if np.isnan(loading):
                continue
            if loading > overload_threshold:
                overloaded_lines.append((i, loading))
    
    if overloaded_lines:
        # Identify the most overloaded line
        comp_index, max_loading = max(overloaded_lines, key=lambda x: x[1])
        comp_name = ss_initial.Line.idx.v[comp_index]
        print(f"Overload detected on line {comp_name}: {max_loading:.2f}%. Tripping this line at t={current_time:.1f}s")
        tripped_lines.add(comp_name)
        events.append({
            "iteration": iteration,
            "event": "Overload trip",
            "component": comp_name,
            "loading": max_loading,
            "time": current_time
        })
    
    # Continue with cascade simulation in 15-second intervals
    while True:
        iteration += 1
        print(f"\nCascade iteration {iteration}:")
        
        # Advance time by 15 seconds
        current_time += 15.0
        
        # Reload a fresh system instance
        ss = setup_ieee14_dynamic(setup=True, export_excel=False,
                                battery_bus=battery_bus, battery_p=battery_power)
        
        # Apply all previously tripped generators
        for gen_bus in tripped_generators:
            # Handle slack generator
            for i in range(len(ss.Slack)):
                if ss.Slack.bus.v[i] == gen_bus:
                    if hasattr(ss.Slack, 'u'):
                        ss.Slack.u.v[i] = 0
                        print(f"Disabled Slack generator at bus {gen_bus}")
                    break
            
            # Handle PV generators
            for i in range(len(ss.PV)):
                if ss.PV.bus.v[i] == gen_bus:
                    if hasattr(ss.PV, 'u'):
                        ss.PV.u.v[i] = 0
                        print(f"Disabled PV generator at bus {gen_bus}")
                    break
        
        # Apply all previously tripped lines
        for line in tripped_lines:
            ss.Line.u.v[ss.Line.idx.v.index(line)] = 0
            print(f"Disabled line {line}")
        
        # Run power flow to update system state after applying all trips
        try:
            print("Running power flow after applying all trips...")
            ss.PFlow.run()
            print("Power flow converged successfully.")
        except Exception as e:
            print(f"Power flow failed after applying trips: {e}")
            break
        
        # Check for line overloads
        overloaded_lines = []
        for i in range(len(ss.Line)):
            if ss.Line.u.v[i] == 1 and ss.Line.idx.v[i] not in tripped_lines:
                loading = compute_line_loading(ss, i)
                print(f"Line {ss.Line.idx.v[i]}: loading = {loading:.2f}%")
                if np.isnan(loading):
                    continue
                if loading > overload_threshold:
                    overloaded_lines.append((i, loading))
        
        if not overloaded_lines:
            print("No overloads detected. System stabilized.")
            break
        else:
            # Identify the most overloaded line
            comp_index, max_loading = max(overloaded_lines, key=lambda x: x[1])
            comp_name = ss.Line.idx.v[comp_index]
            print(f"Overload detected on line {comp_name}: {max_loading:.2f}%. Tripping this line at t={current_time:.1f}s")
            tripped_lines.add(comp_name)
            events.append({
                "iteration": iteration,
                "event": "Overload trip",
                "component": comp_name,
                "loading": max_loading,
                "time": current_time
            })
    
    return events, len(tripped_lines) > 0  # Did a cascade occur?

def export_results_to_excel(events, filename='generator_cascade_results.xlsx'):
    """
    Export cascade simulation results to an Excel file.
    
    Args:
        events: List of events from the cascade simulation
        filename: Output Excel filename
    """
    # Convert events to DataFrame
    df = pd.DataFrame(events)
    
    # Create Excel writer
    try:
        with pd.ExcelWriter(filename, mode='w') as writer:
            df.to_excel(writer, sheet_name='Cascade Events', index=False)
        print(f"Results exported to {filename}")
    except Exception as e:
        print(f"Error exporting results: {e}")

def main():
    parser = argparse.ArgumentParser(description='Cascade failure simulation with generator trips and optional battery addition.')
    parser.add_argument('--battery', nargs='+', metavar=('BUS', 'POWER'), type=float,
                        help='Add batteries at specified buses with specified power in MW. Example: --battery 4 40 9 30')
    parser.add_argument('--trip-gen', type=int, default=None,
                        help='Bus number of the generator to trip initially (default: None, meaning run all generators)')
    parser.add_argument('--threshold', type=float, default=100.0,
                        help='Overload threshold percentage (default: 100.0)')
    parser.add_argument('--export-results', action='store_true', default=True,
                        help='Export results to Excel (default: True)')
    
    # Print timing explanation
    print("\nTiming plan for cascade simulation:")
    print("t=0s:  System is at steady state with selected configuration")
    print("t=5s:  Initial generator trip occurs")
    print("t=20s: System checked for overloads, first line trip occurs if needed")
    print("       Each subsequent event occurs at 15-second intervals")
    
    args = parser.parse_args()
    
    battery_bus = None
    battery_power = None
    if args.battery:
        battery_bus = [int(bus) for bus in args.battery[::2]]
        battery_power = [power for power in args.battery[1::2]]
        print(f"Batteries will be added at buses {', '.join(map(str, battery_bus))} with {', '.join(map(str, battery_power))} MW.")
    
    # Create a base system to identify all generators
    base_system = setup_ieee14_dynamic(setup=True, export_excel=False)
    generator_buses = [bus for bus, gen_type in get_generator_buses(base_system)]
    
    results = []
    
    # If a specific generator is specified, run only that one
    if args.trip_gen is not None:
        if args.trip_gen not in generator_buses:
            print(f"Error: No generator found at bus {args.trip_gen}")
            return
        
        generator_to_trip = [args.trip_gen]
    else:
        # Otherwise, run all generators
        generator_to_trip = generator_buses
    
    # Run simulations for each generator
    for gen_bus in generator_to_trip:
        print(f"\n{'='*80}")
        print(f"SIMULATING TRIP OF GENERATOR AT BUS {gen_bus}")
        print(f"{'='*80}\n")
        
        # Run the cascade simulation with generator trip
        events, had_cascade = cascade_failure_simulation(
            initial_trip_gen_bus=gen_bus,
            overload_threshold=args.threshold,
            battery_bus=battery_bus,
            battery_power=battery_power
        )
        
        # Store the results
        for event in events:
            event['initial_trip_gen'] = gen_bus
            event['had_cascade'] = had_cascade
        
        results.extend(events)
        
        # Print summary
        print("\nCascade events log:")
        for event in events:
            print(f"t={event['time']:.1f}s: {event['event']} of {event['component']}" + 
                  (f" (loading: {event['loading']:.2f}%)" if 'loading' in event else ""))
        
        # Wait a moment before the next simulation
        time.sleep(1)
    
    # Export results if requested
    if args.export_results and results:
        # Determine filename
        if battery_bus:
            battery_str = "_".join([f"B{b}P{p}" for b, p in zip(battery_bus, battery_power)])
            filename = f"gen_cascade_battery_{battery_str}.xlsx"
        else:
            filename = "gen_cascade_base.xlsx"
            
        export_results_to_excel(results, filename)

if __name__ == "__main__":
    main()