"""
Power System Analysis Package

This package provides tools for analyzing power systems, including:
- System setup and configuration
- Power flow analysis
- Dynamic simulation
- Cascade failure analysis
- Visualization
"""

from .core.step_1_system_setup import setup_ieee14_dynamic, add_battery
from .core.step_2_ena_output import process_all_configurations
from .core.step_3_cascade_analysis import run_cascade_analysis
from .core.visualization import PowerSystemVisualizer

__all__ = [
    'setup_ieee14_dynamic',
    'add_battery',
    'process_all_configurations',
    'run_cascade_analysis',
    'PowerSystemVisualizer',
]

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com" 