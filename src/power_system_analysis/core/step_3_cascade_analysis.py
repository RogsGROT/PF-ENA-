import andes
import os
import numpy as np
import matplotlib.pyplot as plt
from andes.utils.paths import get_case
from IPython import get_ipython
import pandas as pd

# Enable interactive plotting if in IPython
ipython = get_ipython()
if ipython is not None:
    ipython.magic('matplotlib inline')

# Define the IEEE 14-bus case files
RAW_FILE = get_case('ieee14/ieee14.raw')
DYR_FILE = get_case('ieee14/ieee14.dyr')

# Create a folder for output if it doesn't exist
output_dir = 'cascade_analysis_output'
os.makedirs(output_dir, exist_ok=True)

# Define the base case output name
out_name = os.path.join(output_dir, 'ieee14_cascade')

def run_cascade_analysis():
    """Run cascade failure analysis by adding a line trip and extracting metrics"""
    # Load the system (without setup)
    ss = andes.load(RAW_FILE, addfile=DYR_FILE, setup=False)
    
    # Add a line trip event (Line 1) at t=1.0 second
    line_idx = 1
    trip_time = 1.0
    line_name = f'Line_{line_idx}'
    
    # Add the trip event
    ss.add('Toggle', {
        'model': 'Line',
        'dev': line_name, 
        't': trip_time
    })
    
    # Configure simulation parameters
    ss.config.tf = 15.0  # Simulation end time
    
    # Setup the system
    ss.setup()
    
    # Run power flow first
    print("Running power flow...")
    ss.PFlow.run()
    
    if not ss.PFlow.converged:
        print("Power flow did not converge. Check system parameters.")
        return None
    
    print("Power flow converged successfully. Running time domain simulation...")
    
    # Run time domain simulation
    ss.TDS.config.tf = 15.0  # Set simulation time frame
    ss.TDS.run()
    
    print("Simulation completed.")
    
    # Extract and analyze data from time series
    extract_and_analyze_data(ss)
    
    # Generate plots using ANDES plotting
    generate_plots(ss)
    
    return ss

def extract_and_analyze_data(ss):
    """Extract and analyze data from the simulation time series"""
    print("\n=== Data Extraction and Analysis ===")
    
    # 1. Access the time series data
    t = ss.dae.ts.t  # Time array
    print(f"Simulation time range: {t[0]} to {t[-1]} seconds")
    print(f"Number of time steps: {len(t)}")
    
    # 2. Extract bus voltage data (algebraic variables)
    bus_voltages = ss.dae.ts.y[:, ss.Bus.v.a]
    print(f"Shape of bus voltage data: {bus_voltages.shape}")
    
    # 3. Extract generator rotor speeds (differential variables)
    if hasattr(ss, 'GENROU'):
        gen_speeds = ss.dae.ts.x[:, ss.GENROU.omega.a]
        print(f"Shape of generator speed data: {gen_speeds.shape}")
        
        # 4. Extract generator rotor angles (differential variables)
        gen_angles = ss.dae.ts.x[:, ss.GENROU.delta.a]
        print(f"Shape of generator angle data: {gen_angles.shape}")
    
    # Calculate key metrics
    metrics = {}
    
    # Voltage stability metrics
    min_voltage = np.min(bus_voltages)
    min_voltage_time = t[np.argmin(np.min(bus_voltages, axis=1))]
    max_deviation = np.max(np.abs(bus_voltages - 1.0))
    
    metrics['min_voltage'] = min_voltage
    metrics['min_voltage_time'] = min_voltage_time
    metrics['max_voltage_deviation'] = max_deviation
    
    # Frequency stability metrics (if GENROU model exists)
    if hasattr(ss, 'GENROU'):
        nominal_freq = 60  # Hz
        freq_deviation = np.abs(gen_speeds - 1.0) * nominal_freq
        max_freq_deviation = np.max(freq_deviation)
        min_freq = np.min(gen_speeds) * nominal_freq
        
        metrics['max_frequency_deviation_hz'] = max_freq_deviation
        metrics['frequency_nadir_hz'] = min_freq
        metrics['frequency_nadir_time'] = t[np.argmin(np.min(gen_speeds, axis=1))]
        
        # Calculate maximum angle difference between any two generators
        max_angle_diff = []
        for i in range(len(t)):
            angles_at_t = gen_angles[i, :]
            max_diff = np.max(angles_at_t) - np.min(angles_at_t)
            max_angle_diff.append(max_diff)
        
        max_angle_diff = np.array(max_angle_diff)
        metrics['max_angle_separation_rad'] = np.max(max_angle_diff)
        metrics['max_angle_separation_deg'] = np.max(max_angle_diff) * 180 / np.pi
        metrics['max_angle_separation_time'] = t[np.argmax(max_angle_diff)]
    
    # Print metrics
    print("\n=== Cascade Failure Analysis Metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Create a DataFrame for easier analysis (optional)
    df = ss.dae.ts.unpack(df=True)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, 'cascade_metrics.csv'), index=False)
    print(f"Metrics saved to {os.path.join(output_dir, 'cascade_metrics.csv')}")
    
    return metrics

def generate_plots(ss):
    """Generate plots using ANDES plotting functions"""
    print("\n=== Generating Plots ===")
    
    # Method 1: Using ANDES CLI commands
    if ipython is not None:
        # Plot bus voltages
        ipython.magic(f'!andes plot {ss.files.lst} 0 --xargs "v Bus" --ylabel "Voltage (pu)" --save')
        
        # Plot generator speeds
        if hasattr(ss, 'GENROU'):
            ipython.magic(f'!andes plot {ss.files.lst} 0 --xargs "omega GENROU" --ylabel "Speed (pu)" --save')
            
            # Plot generator angles
            ipython.magic(f'!andes plot {ss.files.lst} 0 --xargs "delta GENROU" --ylabel "Angle (rad)" --save')
    else:
        print("IPython not available for CLI commands")
    
    # Method 2: Direct plotting with matplotlib using extracted data
    plt.figure(figsize=(12, 9))
    
    # Plot bus voltages
    plt.subplot(2, 2, 1)
    bus_voltages = ss.dae.ts.y[:, ss.Bus.v.a]
    for i in range(bus_voltages.shape[1]):
        plt.plot(ss.dae.ts.t, bus_voltages[:, i], label=f'Bus {ss.Bus.idx.v[i]}')
    plt.title('Bus Voltages')
    plt.ylabel('Voltage (pu)')
    plt.xlabel('Time (s)')
    plt.grid(True)
    
    # Plot generator data if GENROU exists
    if hasattr(ss, 'GENROU'):
        # Plot generator speeds
        plt.subplot(2, 2, 2)
        gen_speeds = ss.dae.ts.x[:, ss.GENROU.omega.a]
        for i in range(gen_speeds.shape[1]):
            plt.plot(ss.dae.ts.t, gen_speeds[:, i], label=f'Gen {ss.GENROU.idx.v[i]}')
        plt.title('Generator Speeds')
        plt.ylabel('Speed (pu)')
        plt.xlabel('Time (s)')
        plt.grid(True)
        
        # Plot generator angles
        plt.subplot(2, 2, 3)
        gen_angles = ss.dae.ts.x[:, ss.GENROU.delta.a]
        for i in range(gen_angles.shape[1]):
            plt.plot(ss.dae.ts.t, gen_angles[:, i], label=f'Gen {ss.GENROU.idx.v[i]}')
        plt.title('Generator Angles')
        plt.ylabel('Angle (rad)')
        plt.xlabel('Time (s)')
        plt.grid(True)
        
        # Plot generator electrical powers
        plt.subplot(2, 2, 4)
        if hasattr(ss.GENROU, 'Pe'):
            gen_powers = ss.dae.ts.y[:, ss.GENROU.Pe.a]
            for i in range(gen_powers.shape[1]):
                plt.plot(ss.dae.ts.t, gen_powers[:, i], label=f'Gen {ss.GENROU.idx.v[i]}')
            plt.title('Generator Electrical Powers')
            plt.ylabel('Power (pu)')
            plt.xlabel('Time (s)')
            plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cascade_analysis_plots.png'), dpi=300)
    print(f"Plots saved to {os.path.join(output_dir, 'cascade_analysis_plots.png')}")
    plt.show()

if __name__ == "__main__":
    # Run the cascade analysis
    ss = run_cascade_analysis()
    
    # If you already have a System object, you can extract and analyze data
    if ss is not None:
        print("\nCascade analysis completed successfully. Review the metrics and plots.")
        print(f"Output files are available in the '{output_dir}' directory.")
    else:
        print("Cascade analysis failed. Check error messages above.")