import numpy as np
import pandas as pd
import argparse
from step_1_system_setup import setup_ieee14_dynamic

def trip_line(ss, line_index):
    """Trip (remove from service) the specified line.
    
    This example assumes the ANDES model has a 'status' attribute for lines.
    Setting it to 0 is used to indicate an outage.
    """
    # Mark the line as out of service
    ss.Line.status.v[line_index] = 0
    print(f"Line {line_index} tripped at simulation time.")

def cascade_check(ss):
    """Check for overload conditions and trip elements if necessary.
    
    This routine examines each line and generator. In this simple example,
    if a line's loading exceeds 100% or if a generator's output exceeds a limit,
    the element is tripped. (You can adjust these criteria as needed.)
    """
    # Check lines (assuming ss.Line.loading is updated during simulation)
    for i in range(len(ss.Line)):
        # Skip already tripped lines (status==0)
        if hasattr(ss.Line, 'status') and ss.Line.status.v[i] == 0:
            continue
        loading = ss.Line.loading.v[i] if hasattr(ss.Line, 'loading') else 0
        if loading > 100:
            print(f"Line {i} overloaded (loading: {loading:.1f}%), tripping.")
            trip_line(ss, i)
    
    # Check generators (using a placeholder limit; adjust according to your model)
    for i in range(len(ss.Generator)):
        if hasattr(ss.Generator, 'status') and ss.Generator.status.v[i] == 0:
            continue
        p_output = ss.Generator.p.v[i] * ss.config.mva
        p_limit = ss.Generator.pmax.v[i] if hasattr(ss.Generator, 'pmax') else 100
        if abs(p_output) > p_limit:
            print(f"Generator at bus {ss.Generator.bus.v[i]} overloaded (P: {p_output:.2f} MW), tripping.")
            ss.Generator.status.v[i] = 0

def run_simulation_with_cascade(ss, trip_line_index):
    """Run a dynamic simulation for one scenario in which a specified line is tripped at t = 1.0 s.
    
    The simulation loop (or the use of a built-in event scheduler) is used to:
      – Reset the system state (if needed)
      – Schedule the initial line trip at t = 1.0 s
      – Periodically check for overloads that cause further (cascade) trips
      – Record a few metrics from the simulation
       
    Returns:
      A dictionary with metrics from this simulation run.
    """
    # Reset simulation to initial conditions (if a reset method exists; otherwise, re-setup)
    if hasattr(ss, 'reset'):
        ss.reset()
    else:
        ss.setup()

    # Define the event to trip the specified line at t = 1.0 s
    def initial_trip_event():
        print("Initial event at t = 1.0 s: Tripping specified line.")
        trip_line(ss, trip_line_index)
        
    # If the simulation scheduler is available, add the event (otherwise we call it manually)
    if hasattr(ss, 'add_event'):
        ss.add_event(1.0, initial_trip_event)
    else:
        # Manual scheduling: the event will be triggered when t reaches 1.0 s in the loop below.
        pass

    # Optionally, you could also schedule periodic cascade-check events.
    # For this example, we implement the cascade check within a manual simulation loop.
    cascade_check_times = np.arange(1.0, ss.config.tf, 0.1)
    cascade_check_idx = 0

    # Prepare to record simulation metrics (time series for frequency and voltage)
    time_series = []
    frequency_series = []
    voltage_series = []

    # Simulation loop (using a fixed time step)
    t = ss.config.t0
    dt = ss.config.tstep
    while t <= ss.config.tf:
        # If no built-in event scheduler, trigger the initial trip manually at t = 1.0 s.
        if abs(t - 1.0) < dt/2:
            initial_trip_event()
        
        # Periodically run cascade check
        if cascade_check_idx < len(cascade_check_times) and abs(t - cascade_check_times[cascade_check_idx]) < dt/2:
            cascade_check(ss)
            cascade_check_idx += 1
        
        # Advance simulation by one step (assuming a step integration method exists)
        if hasattr(ss, 'integrate_step'):
            ss.integrate_step()
        else:
            # If no step method, assume the simulation advances internally.
            pass
        
        # Record current state metrics (using placeholder methods/attributes)
        current_freq = ss.get_frequency() if hasattr(ss, 'get_frequency') else ss.config.freq
        current_volt = np.mean(ss.Bus.v.v) if hasattr(ss.Bus, 'v') else 1.0
        time_series.append(t)
        frequency_series.append(current_freq)
        voltage_series.append(current_volt)
        t += dt

    # Compute example metrics after simulation:
    max_freq_deviation = np.max(np.abs(np.array(frequency_series) - ss.config.freq))
    max_voltage_deviation = np.max(np.abs(np.array(voltage_series) - 1.0))
    
    # Count how many lines and generators were tripped (based on status)
    lines_tripped = sum(1 for i in range(len(ss.Line))
                        if hasattr(ss.Line, 'status') and ss.Line.status.v[i] == 0)
    generators_tripped = 0
    if hasattr(ss, 'Generator'):
        generators_tripped = sum(1 for i in range(len(ss.Generator))
                                if hasattr(ss.Generator, 'status') and ss.Generator.status.v[i] == 0)

    metrics = {
        'max_frequency_deviation': max_freq_deviation,
        'max_voltage_deviation': max_voltage_deviation,
        'lines_tripped': lines_tripped,
        'generators_tripped': generators_tripped,
        'time_series': time_series,
        'frequency_series': frequency_series,
        'voltage_series': voltage_series,
    }
    return metrics

def run_all_line_trip_scenarios(ss):
    """For each eligible (non-transformer) line in the system, run a simulation scenario 
    in which that line is tripped at t = 1.0 s, and collect cascade metrics.
    
    Returns a dictionary mapping scenario names to their metrics.
    """
    results = {}
    num_lines = len(ss.Line)
    for i in range(num_lines):
        # Exclude lines that are transformers. (Here, we assume that if a line has a tap value not equal to 1.0, it is a transformer.)
        if hasattr(ss.Line, 'tap'):
            if ss.Line.tap.v[i] != 1.0:
                continue
        
        print(f"\nRunning scenario for tripping line {i} (from bus {ss.Line.bus1.v[i]} to bus {ss.Line.bus2.v[i]})")
        metrics = run_simulation_with_cascade(ss, i)
        results[f"line_{i}"] = metrics
    
    return results

def export_metrics_to_excel(metrics_dict, filename="cascade_metrics.xlsx"):
    """Export the collected simulation metrics for each scenario to an Excel file."""
    rows = []
    for scenario, metrics in metrics_dict.items():
        row = {
            'Scenario': scenario,
            'Max Frequency Deviation (Hz)': metrics['max_frequency_deviation'],
            'Max Voltage Deviation (p.u.)': metrics['max_voltage_deviation'],
            'Lines Tripped': metrics['lines_tripped'],
            'Generators Tripped': metrics['generators_tripped'],
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_excel(filename, index=False)
    print(f"Metrics exported to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cascade failure scenarios by tripping lines.")
    parser.add_argument('--battery', nargs=2, metavar=('BUS', 'POWER'), type=float,
                        help="Add a battery at specified bus with specified power in MW. Example: --battery 4 20")
    parser.add_argument('--export-excel', action='store_true', default=False,
                        help="Export simulation metrics to an Excel file")
    args = parser.parse_args()

    # Setup the IEEE 14-bus system using the step 1 function.
    if args.battery:
        bus = int(args.battery[0])
        power = args.battery[1]
        ss = setup_ieee14_dynamic(battery_bus=[bus], battery_p=[power], export_excel=False)
    else:
        ss = setup_ieee14_dynamic(export_excel=False)

    # Run the cascade simulation scenarios for each eligible line.
    simulation_results = run_all_line_trip_scenarios(ss)

    # Print the results for each scenario.
    print("\nCascade simulation results:")
    for scenario, metrics in simulation_results.items():
        print(f"Scenario {scenario}:")
        print(f"  Max Frequency Deviation: {metrics['max_frequency_deviation']:.2f} Hz")
        print(f"  Max Voltage Deviation: {metrics['max_voltage_deviation']:.2f} p.u.")
        print(f"  Lines Tripped: {metrics['lines_tripped']}")
        print(f"  Generators Tripped: {metrics['generators_tripped']}")

    # Optionally export metrics to Excel.
    if args.export_excel:
        export_metrics_to_excel(simulation_results)