import andes
import pandas as pd
import os
import sys
import itertools
from datetime import datetime

# Import from our modular scripts
from .step_1_system_setup import setup_ieee14_dynamic
from .step_3_cascade_analysis import run_cascading_failure_analysis

def run_all_configurations():
    """Run cascade analysis for multiple battery configurations"""
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"cascade_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    os.chdir(results_dir)
    
    # Initialize the base system
    base_ss = setup_ieee14_dynamic()
    
    # Track all configurations and their results
    all_results = []
    
    # 1. Base case (no batteries)
    print("\nAnalyzing base case...")
    base_results = run_cascading_failure_analysis(base_ss, battery_buses=None, config_name='base')
    if base_results:
        all_results.append(base_results)
    
    # 2. Single battery configurations
    for bus in range(1, 15):  # IEEE 14-bus system has buses 1-14
        config_name = f'batt_{bus}'
        print(f"\nAnalyzing configuration with battery at bus {bus}...")
        config_results = run_cascading_failure_analysis(base_ss, battery_buses=[bus], config_name=config_name)
        if config_results:
            all_results.append(config_results)
    
    # 3. Two battery configurations
    for bus1, bus2 in itertools.combinations(range(1, 15), 2):
        config_name = f'batt_{bus1}_{bus2}'
        print(f"\nAnalyzing configuration with batteries at buses {bus1} and {bus2}...")
        config_results = run_cascading_failure_analysis(base_ss, battery_buses=[bus1, bus2], config_name=config_name)
        if config_results:
            all_results.append(config_results)
    
    # Create consolidated results file
    consolidated_df = pd.DataFrame()
    for result in all_results:
        for scenario in result['scenarios']:
            if scenario['scenario_type'] == 'load_increase':
                for i, severity in enumerate(scenario['severity']):
                    row = {
                        'config': result['config'],
                        'battery_buses': str(result['battery_buses']),
                        'scenario_type': scenario['scenario_type'],
                        'fault_location': scenario['fault_location'],
                        'severity': severity,
                        'convergence': scenario['convergence'][i] if i < len(scenario['convergence']) else None,
                        'min_voltage': scenario['min_voltage'][i] if i < len(scenario['min_voltage']) and scenario['min_voltage'][i] is not None else None,
                        'max_freq_dev': scenario['max_freq_dev'][i] if i < len(scenario['max_freq_dev']) and scenario['max_freq_dev'][i] is not None else None,
                        'max_loading': scenario['max_loading'][i] if i < len(scenario['max_loading']) and scenario['max_loading'][i] is not None else None,
                        'failure_propagation': str(scenario['failure_propagation'][i]) if i < len(scenario['failure_propagation']) else None
                    }
                    consolidated_df = pd.concat([consolidated_df, pd.DataFrame([row])], ignore_index=True)
            else:
                row = {
                    'config': result['config'],
                    'battery_buses': str(result['battery_buses']),
                    'scenario_type': scenario['scenario_type'],
                    'fault_location': scenario['fault_location'],
                    'severity': 100,
                    'convergence': scenario['convergence'],
                    'min_voltage': scenario['min_voltage'],
                    'max_freq_dev': scenario['max_freq_dev'],
                    'max_loading': scenario['max_loading'],
                    'failure_propagation': str(scenario['failure_propagation'])
                }
                consolidated_df = pd.concat([consolidated_df, pd.DataFrame([row])], ignore_index=True)
    
    # Save consolidated results
    consolidated_df.to_excel('all_cascade_results.xlsx', index=False)
    
    # Calculate and save summary metrics
    summary_df = pd.DataFrame()
    for config in consolidated_df['config'].unique():
        config_df = consolidated_df[consolidated_df['config'] == config]
        battery_buses = config_df['battery_buses'].iloc[0]
        
        # Get metrics by scenario type
        for scenario_type in ['load_increase', 'generator_outage', 'line_outage']:
            scenario_df = config_df[config_df['scenario_type'] == scenario_type]
            if scenario_df.empty:
                continue
                
            summary_row = {
                'config': config,
                'battery_buses': battery_buses,
                'scenario_type': scenario_type,
                'num_scenarios': len(scenario_df),
                'convergence_rate': scenario_df['convergence'].mean(),
                'min_voltage_overall': scenario_df['min_voltage'].min(),
                'max_freq_dev_overall': scenario_df['max_freq_dev'].max(),
                'cascading_failures': sum(1 for x in scenario_df['failure_propagation'] if x != '[]' and x != 'None'),
                'failure_rate': sum(1 for x in scenario_df['failure_propagation'] if x != '[]' and x != 'None') / len(scenario_df)
            }
            summary_df = pd.concat([summary_df, pd.DataFrame([summary_row])], ignore_index=True)
        
        # Overall metrics for this configuration
        summary_row = {
            'config': config,
            'battery_buses': battery_buses,
            'scenario_type': 'all',
            'num_scenarios': len(config_df),
            'convergence_rate': config_df['convergence'].mean(),
            'min_voltage_overall': config_df['min_voltage'].min(),
            'max_freq_dev_overall': config_df['max_freq_dev'].max(),
            'cascading_failures': sum(1 for x in config_df['failure_propagation'] if x != '[]' and x != 'None'),
            'failure_rate': sum(1 for x in config_df['failure_propagation'] if x != '[]' and x != 'None') / len(config_df)
        }
        summary_df = pd.concat([summary_df, pd.DataFrame([summary_row])], ignore_index=True)
    
    # Save summary
    summary_df.to_excel('cascade_summary.xlsx', index=False)
    
    print(f"\nAll results saved in directory: {results_dir}")
    print(f"Consolidated results in: all_cascade_results.xlsx")
    print(f"Summary metrics in: cascade_summary.xlsx")
    
    # Copy all the ENA input files to the results directory
    ena_files = [f for f in os.listdir('..') if f.startswith('ena_input_')]
    for file in ena_files:
        os.system(f'cp ../{file} .')
    
    print("\nENA input files copied to results directory.")
    print("Next steps:")
    print("1. Run the MATLAB ENA analysis on the 'ena_input_*.xlsx' files")
    print("2. Combine the ENA metrics with the cascade analysis results")
    print("3. Analyze correlations between ENA metrics and system resilience")
    
    return all_results, consolidated_df, summary_df

if __name__ == "__main__":
    print("Running all battery configurations for IEEE 14-bus system")
    print("This will run multiple simulations and may take a long time to complete.")
    print("Results will be saved in a timestamped directory.")
    
    run_choice = input("Do you want to proceed? (y/n): ")
    if run_choice.lower() == 'y':
        try:
            run_all_configurations()
        except Exception as e:
            print(f"Error running configurations: {str(e)}")
            sys.exit(1)
    else:
        print("Operation cancelled.")