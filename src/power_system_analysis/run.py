#!/usr/bin/env python3
"""
Master script to run the power system analysis workflow.
This script guides users through the different steps of the analysis.
"""

import os
import sys
import argparse
from datetime import datetime

def display_menu():
    """Display the main menu for the power system analysis workflow."""
    print("\n========= Power System Analysis Workflow =========")
    print("1. Setup the IEEE 14-bus system with dynamic models")
    print("2. Generate ENA input file for a specific configuration")
    print("3. Run cascading failure analysis for a specific configuration")
    print("4. Run all configurations (comprehensive analysis)")
    print("5. Exit")
    print("==================================================")
    return input("Enter your choice (1-5): ")

def main():
    """Main function to run the power system analysis workflow."""
    
    while True:
        choice = display_menu()
        
        if choice == '1':
            print("\nSetting up the IEEE 14-bus system...")
            os.system('python 1_system_setup.py')
            
        elif choice == '2':
            # Generate ENA input file
            system_file = input("\nEnter the system file path (default: 'ieee14_dynamic.pkl'): ") or 'ieee14_dynamic.pkl'
            config_name = input("Enter configuration name (default: 'base'): ") or 'base'
            
            if not os.path.exists(system_file):
                print(f"Error: System file '{system_file}' not found. Run option 1 first.")
                continue
                
            os.system(f'python 2_ena_output.py {system_file} {config_name}')
            
        elif choice == '3':
            # Run cascading failure analysis
            system_file = input("\nEnter the system file path (default: 'ieee14_dynamic.pkl'): ") or 'ieee14_dynamic.pkl'
            config_name = input("Enter configuration name (default: 'base'): ") or 'base'
            battery_input = input("Enter battery buses (comma-separated, e.g. '4,7' or leave empty for none): ")
            
            if not os.path.exists(system_file):
                print(f"Error: System file '{system_file}' not found. Run option 1 first.")
                continue
                
            if battery_input:
                os.system(f'python 3_cascade_analysis.py {system_file} {config_name} {battery_input}')
            else:
                os.system(f'python 3_cascade_analysis.py {system_file} {config_name}')
                
        elif choice == '4':
            # Run all configurations
            print("\nThis will run comprehensive analysis with all battery configurations.")
            print("WARNING: This may take a long time to complete.")
            os.system('python 4_run_configurations.py')
            
        elif choice == '5':
            print("\nExiting the power system analysis workflow. Goodbye!")
            break
            
        else:
            print("\nInvalid choice. Please enter a number between 1 and 5.")
    
if __name__ == "__main__":
    main()