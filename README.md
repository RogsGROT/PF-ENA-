# Power System Analysis

A comprehensive Python framework for analyzing power system resilience, focusing on cascading failures and the impact of battery storage on system stability. The framework is built around the IEEE 14-bus system and uses ANDES for power system simulation.

## Project Structure

```
power-system-analysis/
├── data/
│   └── ieee14/           # IEEE 14-bus system data files
│       ├── ieee14.dyr
│       └── ieee14_modified.xlsx
├── docs/                 # Documentation
├── src/
│   └── power_system_analysis/
│       ├── core/         # Core functionality
│       │   ├── step_1_system_setup.py     # System setup and initialization
│       │   ├── 2_ena_output.py            # Energy Network Analysis output generation
│       │   ├── 3_cascade_analysis.py      # Cascade failure analysis
│       │   └── 4_run_configurations.py     # Configuration runners
│       ├── utils/        # Utility functions
│       └── models/       # Custom models
├── tests/               # Test files
│   └── system_inspection.py  # System setup tests
├── setup.py            # Package installation
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## Installation

### Development Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/power-system-analysis.git
cd power-system-analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install in development mode:
```bash
pip install -e ".[dev]"
```

### User Installation

```bash
pip install git+https://github.com/yourusername/power-system-analysis.git
```

## Usage

### Basic Usage

```python
from power_system_analysis import setup_ieee14_dynamic, inspect_system

# Set up the IEEE 14-bus system
ss = setup_ieee14_dynamic()

# Inspect the system
inspect_system()
```

### Adding a Battery

```python
from power_system_analysis import add_battery

# Add a 10 MW battery at bus 4
add_battery(ss, bus_idx=4, idx=0, p_mw=10)
```

### Running Cascade Analysis

```python
from power_system_analysis import run_cascading_failure_analysis

# Run cascade analysis with batteries at buses 4 and 7
results = run_cascading_failure_analysis(ss, battery_buses=[4, 7])
```

## Features

1. **System Setup**: Configure the IEEE 14-bus system with correct voltage levels and transformer ratios
2. **System Inspection**: Detailed analysis of system components and state
3. **ENA Output**: Generate Energy Network Analysis compatible files
4. **Cascade Analysis**: Test system resilience under various fault scenarios:
   - Load increases (20% to 200%)
   - Generator outages
   - Line outages
5. **Battery Integration**: Test impact of battery storage at different locations

## Testing

Run the test suite:

```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{power_system_analysis,
  author = {Your Name},
  title = {Power System Analysis},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/power-system-analysis}
}
```