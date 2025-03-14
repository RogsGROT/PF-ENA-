from setuptools import setup, find_packages

setup(
    name="power_system_analysis",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "andes>=1.8.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "tabulate>=0.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="Power system analysis tools for IEEE 14-bus system with battery storage",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="power-systems, battery-storage, ieee14-bus, cascading-failures",
    url="https://github.com/yourusername/power-system-analysis",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
) 