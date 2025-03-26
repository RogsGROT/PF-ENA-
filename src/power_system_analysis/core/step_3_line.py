import argparse
import numpy as np
from step_1_system_setup import setup_ieee14_dynamic
import andes

def compute_line_loading(ss, i):
    # (Same as before)
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

def cascade_failure_simulation(initial_trip_line_index=0, overload_threshold=100.0):
    events = []
    iteration = 0

    # Load a fresh system instance for the initial event
    ss_initial = setup_ieee14_dynamic(setup=True, export_excel=False)
    initial_line = ss_initial.Line.idx.v[initial_trip_line_index]
    print(f"Initial event: Tripping line {initial_line}")
    ss_initial.Line.alter('u', initial_line, 0)
    events.append({"iteration": iteration, "event": "Initial trip", "component": initial_line})
    
    # Save the list of tripped components so far
    tripped_components = {initial_line}
    
    while True:
        iteration += 1
        print(f"\nCascade iteration {iteration}:")
        
        # Reload a fresh system instance
        ss = setup_ieee14_dynamic(setup=True, export_excel=False)
        
        # Apply all previous trips to the fresh system copy
        for comp in tripped_components:
            ss.Line.alter('u', comp, 0)
        
        # Run power flow
        try:
            ss.PFlow.run()
        except Exception as e:
            print(f"Power flow failed: {e}")
            break
        
        # Run a short time-domain simulation
        ss.config.tf = 5.0
        try:
            ss.TDS.run()
        except Exception as e:
            print(f"Time-domain simulation failed: {e}")
            break
        
        # Check line loadings on lines that are still in service
        overloaded_lines = []
        for i in range(len(ss.Line)):
            if ss.Line.u.v[i] == 1:
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
            print(f"Overload detected on line {comp_name}: {max_loading:.2f}%. Tripping this line.")
            tripped_components.add(comp_name)
            events.append({
                "iteration": iteration,
                "event": "Overload trip",
                "component": comp_name,
                "loading": max_loading
            })
    
    return events

def main():
    parser = argparse.ArgumentParser(description='Cascade failure simulation with optional battery addition.')
    parser.add_argument('--battery', nargs=2, metavar=('BUS', 'POWER'), type=float,
                        help='Add a battery at specified bus with specified power in MW. Example: --battery 4 40')
    parser.add_argument('--trip-line', type=int, default=12,
                        help='Index of the line to trip initially (default: 12)')
    parser.add_argument('--threshold', type=float, default=100.0,
                        help='Overload threshold percentage (default: 100.0)')
    
    args = parser.parse_args()
    
    battery_bus = None
    battery_power = None
    if args.battery:
        battery_bus = [int(args.battery[0])]
        battery_power = [args.battery[1]]
        print(f"Battery will be added at bus {battery_bus[0]} with {battery_power[0]} MW.")
    
    # Pass battery parameters to the setup function
    def setup_system():
        return setup_ieee14_dynamic(setup=True, export_excel=False,
                                    battery_bus=battery_bus, battery_p=battery_power)
    
    # Load a fresh system instance for the initial event
    ss_initial = setup_system()
    initial_line = ss_initial.Line.idx.v[args.trip_line]
    print(f"Initial event: Tripping line {initial_line}")
    ss_initial.Line.alter('u', initial_line, 0)
    
    # Save the list of tripped components so far
    tripped_components = {initial_line}
    events = [{"iteration": 0, "event": "Initial trip", "component": initial_line}]
    iteration = 0
    
    while True:
        iteration += 1
        print(f"\nCascade iteration {iteration}:")
        # Reload a fresh system instance, now including the battery if specified
        ss = setup_system()
        # Apply all previous trips to the fresh system copy
        for comp in tripped_components:
            ss.Line.alter('u', comp, 0)
        try:
            ss.PFlow.run()
        except Exception as e:
            print(f"Power flow failed: {e}")
            break
        ss.config.tf = 5.0
        try:
            ss.TDS.run()
        except Exception as e:
            print(f"Time-domain simulation failed: {e}")
            break
        overloaded_lines = []
        for i in range(len(ss.Line)):
            if ss.Line.u.v[i] == 1:
                loading = compute_line_loading(ss, i)
                print(f"Line {ss.Line.idx.v[i]}: loading = {loading:.2f}%")
                if np.isnan(loading):
                    continue
                if loading > args.threshold:
                    overloaded_lines.append((i, loading))
        if not overloaded_lines:
            print("No overloads detected. System stabilized.")
            break
        else:
            comp_index, max_loading = max(overloaded_lines, key=lambda x: x[1])
            comp_name = ss.Line.idx.v[comp_index]
            print(f"Overload detected on line {comp_name}: {max_loading:.2f}%. Tripping this line.")
            tripped_components.add(comp_name)
            events.append({
                "iteration": iteration,
                "event": "Overload trip",
                "component": comp_name,
                "loading": max_loading
            })
    
    print("\nCascade events log:")
    for event in events:
        print(event)

if __name__ == "__main__":
    main()