import andes
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime

# Import from our modular scripts
from .step_1_system_setup import add_battery
from .step_2_ena_output import generate_ena_output

def run_cascading_failure_analysis(ss, battery_buses=None, config_name='base'):
    """
    Perform cascading failure analysis on the system.
    Tests all specified fault scenarios and evaluates system performance.
    """
    # Create a copy of the system to work with
    sim_ss = ss.deepcopy()
    
    # Add batteries if specified
    if battery_buses:
        for i, bus in enumerate(battery_buses):
            add_battery(sim_ss, bus, i)
    
    # Run initial power flow to establish baseline
    try:
        sim_ss.PFlow.run()
        
        # Generate ENA input for nominal case
        generate_ena_output(sim_ss, config_name)
        
        # Get baseline metrics
        baseline = {
            'bus_voltage': {bus: sim_ss.Bus.v.v[sim_ss.Bus.idx.v.index(bus)] for bus in range(1, len(sim_ss.Bus)+1)},
            'line_loading': {line: sim_ss.Line.get(src='i', idx=line, attr='v') / sim_ss.Line.get(src='rate_a', idx=line, attr='v') * 100 
                             for line in sim_ss.Line.idx.v if sim_ss.Line.get(src='rate_a', idx=line, attr='v') > 0},
            'gen_output': {gen: sim_ss.StaticGen.get(src='p', idx=gen, attr='v') for gen in sim_ss.StaticGen.idx.v}
        }
        
        # Initialize storage for cascade analysis results
        cascade_results = {
            'config': config_name,
            'battery_buses': battery_buses if battery_buses else [],
            'scenarios': []
        }
        
        # SCENARIO 1: Load increase at each loading node
        for load_idx in sim_ss.PQ.idx.v:
            load_bus = sim_ss.PQ.bus.v[sim_ss.PQ.idx.v.index(load_idx)]
            p_orig = sim_ss.PQ.p.v[sim_ss.PQ.idx.v.index(load_idx)]
            q_orig = sim_ss.PQ.q.v[sim_ss.PQ.idx.v.index(load_idx)]
            
            print(f"Testing load increase at bus {load_bus} (PQ idx: {load_idx})...")
            
            # Create a copy for this specific scenario
            scenario_ss = sim_ss.deepcopy()
            
            # Prepare scenario metrics
            scenario_metrics = {
                'scenario_type': 'load_increase',
                'fault_location': f'bus_{load_bus}',
                'load_idx': int(load_idx),
                'severity': [],
                'convergence': [],
                'min_voltage': [],
                'max_loading': [],
                'max_freq_dev': [],
                'failure_propagation': []
            }
            
            # Test increasing severity levels
            for increase_pct in [20, 50, 100, 150, 200]:
                try:
                    # Reset to original state
                    test_ss = scenario_ss.deepcopy()
                    
                    # Apply load increase
                    p_new = p_orig * (1 + increase_pct/100)
                    q_new = q_orig * (1 + increase_pct/100)
                    
                    # Set up the event
                    test_ss.add('TimerAction', {
                        'uid': load_idx,
                        'model': 'PQ',
                        'enable': 1,
                        'idx': load_idx,
                        'time': 1.0,  # Apply at t=1s
                        'action': {
                            'p': p_new,
                            'q': q_new,
                        }
                    })
                    
                    # Run the simulation
                    test_ss.TDS.config.tf = 10.0
                    test_ss.TDS.run()
                    
                    # Get key metrics
                    voltage_data = test_ss.TDS.get_data('bus', bus=list(range(1, len(test_ss.Bus)+1)), var='v')
                    min_voltage = voltage_data.min().min()
                    
                    freq_data = test_ss.TDS.get_data('bus', bus=list(range(1, len(test_ss.Bus)+1)), var='freq')
                    max_freq_dev = abs(freq_data - 60.0).max().max()
                    
                    # Check for cascading failures
                    failures = []
                    for bus_idx in range(1, len(test_ss.Bus)+1):
                        bus_voltage = voltage_data.loc[:, bus_idx].min()
                        if bus_voltage < 0.90:  # Voltage collapse threshold
                            failures.append(f"Bus_{bus_idx}_voltage_collapse")
                    
                    # Check generator stability
                    if 'GENROU' in test_ss.models:
                        gen_speed = test_ss.TDS.get_data('GENROU', var='omega')
                        for gen_idx in test_ss.GENROU.idx.v:
                            if abs(gen_speed.loc[:, gen_idx].max() - 1.0) > 0.02:  # More than 2% speed deviation
                                gen_bus = test_ss.GENROU.bus.v[test_ss.GENROU.idx.v.index(gen_idx)]
                                failures.append(f"Gen_{gen_idx}_at_Bus_{gen_bus}_unstable")
                    
                    # Record metrics
                    scenario_metrics['severity'].append(increase_pct)
                    scenario_metrics['convergence'].append(True)
                    scenario_metrics['min_voltage'].append(float(min_voltage))
                    scenario_metrics['max_freq_dev'].append(float(max_freq_dev))
                    scenario_metrics['max_loading'].append(None)  # Not calculated in dynamic sim
                    scenario_metrics['failure_propagation'].append(failures)
                    
                except Exception as e:
                    print(f"  Simulation failed at {increase_pct}% increase: {str(e)}")
                    scenario_metrics['severity'].append(increase_pct)
                    scenario_metrics['convergence'].append(False)
                    scenario_metrics['min_voltage'].append(None)
                    scenario_metrics['max_freq_dev'].append(None)
                    scenario_metrics['max_loading'].append(None)
                    scenario_metrics['failure_propagation'].append(["simulation_diverged"])
            
            # Add this scenario to results
            cascade_results['scenarios'].append(scenario_metrics)
        
        # SCENARIO 2: Generator outages
        gen_buses = list(sim_ss.PV.bus.v) + list(sim_ss.Slack.bus.v)
        for i, gen_bus in enumerate(gen_buses):
            print(f"Testing generator outage at bus {gen_bus}...")
            
            # Create a copy for this specific scenario
            scenario_ss = sim_ss.deepcopy()
            
            # Get the generator index
            gen_idx = None
            is_slack = False
            for idx in scenario_ss.PV.idx.v:
                bus = scenario_ss.PV.bus.v[scenario_ss.PV.idx.v.index(idx)]
                if bus == gen_bus:
                    gen_idx = idx
                    break
            
            if gen_idx is None:
                for idx in scenario_ss.Slack.idx.v:
                    bus = scenario_ss.Slack.bus.v[scenario_ss.Slack.idx.v.index(idx)]
                    if bus == gen_bus:
                        gen_idx = idx
                        is_slack = True
                        break
            
            if gen_idx is None:
                print(f"  No generator found at bus {gen_bus}")
                continue
            
            # Prepare scenario metrics
            scenario_metrics = {
                'scenario_type': 'generator_outage',
                'fault_location': f'bus_{gen_bus}',
                'gen_idx': int(gen_idx),
                'convergence': None,
                'min_voltage': None,
                'max_loading': None,
                'max_freq_dev': None,
                'failure_propagation': []
            }
            
            try:
                # Set up the generator trip event
                test_ss = scenario_ss.deepcopy()
                
                # For a slack bus, we should convert it to a PV bus before tripping
                if is_slack:
                    # We can't directly trip the slack, so convert to PV and trip
                    slack_p = test_ss.Slack.p.v[0]
                    slack_v = test_ss.Slack.v.v[0]
                    
                    # Add a timer to convert slack to PV
                    test_ss.add('TimerAction', {
                        'uid': 1000,
                        'model': 'Slack',
                        'enable': 1,
                        'idx': gen_idx,
                        'time': 0.5,  # Apply at t=0.5s
                        'action': {
                            'u': 0.0,  # Disable slack
                        }
                    })
                    
                    # After slack is disabled, enable PV with same values
                    test_ss.add('PV', {
                        'idx': 9999,
                        'bus': gen_bus,
                        'p': slack_p,
                        'v': slack_v,
                        'u': 0,  # Initially disabled
                    })
                    
                    test_ss.add('TimerAction', {
                        'uid': 1001,
                        'model': 'PV',
                        'enable': 1,
                        'idx': 9999,
                        'time': 0.6,  # Apply at t=0.6s (just after slack disabled)
                        'action': {
                            'u': 1.0,  # Enable PV
                        }
                    })
                    
                    # Trip the PV gen
                    test_ss.add('TimerAction', {
                        'uid': 1002,
                        'model': 'PV',
                        'enable': 1,
                        'idx': 9999,
                        'time': 1.0,  # Apply at t=1.0s
                        'action': {
                            'u': 0.0,  # Trip generator
                        }
                    })
                else:
                    # Normal PV generator trip
                    test_ss.add('TimerAction', {
                        'uid': 1,
                        'model': 'PV',
                        'enable': 1,
                        'idx': gen_idx,
                        'time': 1.0,  # Apply at t=1s
                        'action': {
                            'u': 0.0,  # Trip generator
                        }
                    })
                
                # Run the simulation
                test_ss.TDS.config.tf = 10.0
                test_ss.TDS.run()
                
                # Get key metrics
                voltage_data = test_ss.TDS.get_data('bus', bus=list(range(1, len(test_ss.Bus)+1)), var='v')
                min_voltage = voltage_data.min().min()
                
                freq_data = test_ss.TDS.get_data('bus', bus=list(range(1, len(test_ss.Bus)+1)), var='freq')
                max_freq_dev = abs(freq_data - 60.0).max().max()
                
                # Check for cascading failures
                failures = []
                for bus_idx in range(1, len(test_ss.Bus)+1):
                    bus_voltage = voltage_data.loc[:, bus_idx].min()
                    if bus_voltage < 0.90:  # Voltage collapse threshold
                        failures.append(f"Bus_{bus_idx}_voltage_collapse")
                
                # Check generator stability
                if 'GENROU' in test_ss.models:
                    gen_speed = test_ss.TDS.get_data('GENROU', var='omega')
                    for other_gen_idx in test_ss.GENROU.idx.v:
                        # For non-tripped generators, check stability
                        if (not is_slack and other_gen_idx != gen_idx) or (is_slack and other_gen_idx != 0):
                            if abs(gen_speed.loc[:, other_gen_idx].max() - 1.0) > 0.02:
                                other_gen_bus = test_ss.GENROU.bus.v[test_ss.GENROU.idx.v.index(other_gen_idx)]
                                failures.append(f"Gen_{other_gen_idx}_at_Bus_{other_gen_bus}_unstable")
                
                # Record metrics
                scenario_metrics['convergence'] = True
                scenario_metrics['min_voltage'] = float(min_voltage)
                scenario_metrics['max_freq_dev'] = float(max_freq_dev)
                scenario_metrics['max_loading'] = None  # Not calculated in dynamic sim
                scenario_metrics['failure_propagation'] = failures
                
            except Exception as e:
                print(f"  Generator outage simulation failed: {str(e)}")
                scenario_metrics['convergence'] = False
                scenario_metrics['failure_propagation'] = ["simulation_diverged"]
            
            # Add this scenario to results
            cascade_results['scenarios'].append(scenario_metrics)
        
        # SCENARIO 3: Line outages
        for line_idx in sim_ss.Line.idx.v:
            i = sim_ss.Line.idx.v.index(line_idx)
            from_bus = sim_ss.Line.bus1.v[i]
            to_bus = sim_ss.Line.bus2.v[i]
            
            print(f"Testing line outage for line {line_idx} ({from_bus}-{to_bus})...")
            
            # Create a copy for this specific scenario
            scenario_ss = sim_ss.deepcopy()
            
            # Prepare scenario metrics
            scenario_metrics = {
                'scenario_type': 'line_outage',
                'fault_location': f'line_{line_idx}_{from_bus}_{to_bus}',
                'line_idx': int(line_idx),
                'convergence': None,
                'min_voltage': None,
                'max_loading': None,
                'max_freq_dev': None,
                'failure_propagation': []
            }
            
            try:
                # Set up the line trip event
                test_ss = scenario_ss.deepcopy()
                
                test_ss.add('TimerAction', {
                    'uid': 1,
                    'model': 'Line',
                    'enable': 1,
                    'idx': line_idx,
                    'time': 1.0,  # Apply at t=1s
                    'action': {
                        'u': 0.0,  # Trip line
                    }
                })
                
                # Run the simulation
                test_ss.TDS.config.tf = 10.0
                test_ss.TDS.run()
                
                # Get key metrics
                voltage_data = test_ss.TDS.get_data('bus', bus=list(range(1, len(test_ss.Bus)+1)), var='v')
                min_voltage = voltage_data.min().min()
                
                freq_data = test_ss.TDS.get_data('bus', bus=list(range(1, len(test_ss.Bus)+1)), var='freq')
                max_freq_dev = abs(freq_data - 60.0).max().max()
                
                # Check for cascading failures
                failures = []
                for bus_idx in range(1, len(test_ss.Bus)+1):
                    bus_voltage = voltage_data.loc[:, bus_idx].min()
                    if bus_voltage < 0.90:  # Voltage collapse threshold
                        failures.append(f"Bus_{bus_idx}_voltage_collapse")
                
                # Check line loadings (if any are over threshold, add to failures)
                # We'll check this by running a quick power flow at the end of the simulation
                test_ss.PFlow.run()
                for other_line_idx in test_ss.Line.idx.v:
                    if other_line_idx != line_idx and test_ss.Line.u.v[test_ss.Line.idx.v.index(other_line_idx)] > 0:
                        j = test_ss.Line.idx.v.index(other_line_idx)
                        rate_a = test_ss.Line.rate_a.v[j]
                        if rate_a > 0:
                            i_line = test_ss.Line.get(src='i', idx=other_line_idx, attr='v')
                            loading = i_line / rate_a * 100
                            if loading > 100:
                                other_from = test_ss.Line.bus1.v[j]
                                other_to = test_ss.Line.bus2.v[j]
                                failures.append(f"Line_{other_line_idx}_{other_from}_{other_to}_overloaded")
                
                # Check generator stability
                if 'GENROU' in test_ss.models:
                    gen_speed = test_ss.TDS.get_data('GENROU', var='omega')
                    for gen_idx in test_ss.GENROU.idx.v:
                        if abs(gen_speed.loc[:, gen_idx].max() - 1.0) > 0.02:
                            gen_bus = test_ss.GENROU.bus.v[test_ss.GENROU.idx.v.index(gen_idx)]
                            failures.append(f"Gen_{gen_idx}_at_Bus_{gen_bus}_unstable")
                
                # Record metrics
                scenario_metrics['convergence'] = True
                scenario_metrics['min_voltage'] = float(min_voltage)
                scenario_metrics['max_freq_dev'] = float(max_freq_dev)
                scenario_metrics['max_loading'] = None  # Not easily accessible in this context
                scenario_metrics['failure_propagation'] = failures
                
            except Exception as e:
                print(f"  Line outage simulation failed: {str(e)}")
                scenario_metrics['convergence'] = False
                scenario_metrics['failure_propagation'] = ["simulation_diverged"]
            
            # Add this scenario to results
            cascade_results['scenarios'].append(scenario_metrics)
        
        # Save all results to Excel
        excel_file = f'cascade_results_{config_name}.xlsx'
        
        # Convert to DataFrame and save
        scenarios_df = pd.DataFrame()
        for scenario in cascade_results['scenarios']:
            if scenario['scenario_type'] == 'load_increase':
                # Load increase scenarios have multiple severity levels
                for i, severity in enumerate(scenario['severity']):
                    row = {
                        'config': config_name,
                        'scenario_type': scenario['scenario_type'],
                        'fault_location': scenario['fault_location'],
                        'severity': severity,
                        'convergence': scenario['convergence'][i] if i < len(scenario['convergence']) else None,
                        'min_voltage': scenario['min_voltage'][i] if i < len(scenario['min_voltage']) else None,
                        'max_freq_dev': scenario['max_freq_dev'][i] if i < len(scenario['max_freq_dev']) else None,
                        'max_loading': scenario['max_loading'][i] if i < len(scenario['max_loading']) else None,
                        'failure_propagation': str(scenario['failure_propagation'][i]) if i < len(scenario['failure_propagation']) else None
                    }
                    scenarios_df = pd.concat([scenarios_df, pd.DataFrame([row])], ignore_index=True)
            else:
                row = {
                    'config': config_name,
                    'scenario_type': scenario['scenario_type'],
                    'fault_location': scenario['fault_location'],
                    'severity': 100,  # Default severity for non-load increase scenarios
                    'convergence': scenario['convergence'],
                    'min_voltage': scenario['min_voltage'],
                    'max_freq_dev': scenario['max_freq_dev'],
                    'max_loading': scenario['max_loading'],
                    'failure_propagation': str(scenario['failure_propagation'])
                }
                scenarios_df = pd.concat([scenarios_df, pd.DataFrame([row])], ignore_index=True)
            
        # Save the results
        with pd.ExcelWriter(excel_file) as writer:
            scenarios_df.to_excel(writer, sheet_name='Scenarios', index=False)
            
            # Include summary metrics
            summary_data = {
                'config': [config_name],
                'battery_buses': [str(battery_buses)],
                'total_scenarios': [len(cascade_results['scenarios'])],
                'convergence_rate': [scenarios_df['convergence'].mean()],
                'min_voltage_overall': [scenarios_df['min_voltage'].min()],
                'max_freq_dev_overall': [scenarios_df['max_freq_dev'].max()],
                'cascading_failures': [sum(1 for x in scenarios_df['failure_propagation'] if x != '[]' and x != 'None')]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"Cascade analysis results saved to {excel_file}")
        return cascade_results
        
    except Exception as e:
        print(f"Error in cascade analysis: {str(e)}")
        return None

if __name__ == "__main__":
    # Check if we have a system file and configuration as input
    if len(sys.argv) > 1:
        system_file = sys.argv[1]
        config_name = sys.argv[2] if len(sys.argv) > 2 else 'base'
        battery_buses_str = sys.argv[3] if len(sys.argv) > 3 else None
        
        # Parse battery buses if provided
        battery_buses = None
        if battery_buses_str:
            try:
                battery_buses = [int(bus) for bus in battery_buses_str.split(',')]
            except:
                print("Error parsing battery buses. Format should be comma-separated integers.")
                sys.exit(1)
    else:
        # Default to loading a saved system
        if os.path.exists('ieee14_dynamic.pkl'):
            system_file = 'ieee14_dynamic.pkl'
            config_name = 'base'
            battery_buses = None
        else:
            print("No system file provided. Please run 1_system_setup.py first or provide a system file.")
            sys.exit(1)
    
    print(f"Loading system from {system_file}...")
    try:
        ss = andes.system.System()
        ss = ss.load(system_file)
        
        # Run the cascade analysis
        print(f"Running cascade analysis with config '{config_name}' and battery buses {battery_buses}...")
        results = run_cascading_failure_analysis(ss, battery_buses, config_name)
        
        if results:
            print("Analysis completed successfully!")
        else:
            print("Analysis failed.")
            sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)