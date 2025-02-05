# CowStudyApp

A Python package for processing and analyzing sensor data from cattle-mounted GPS and accelerometer devices, developed for Montana State University's Ranch Sciences and Industrial Engineering Departments.

## Overview

CowStudyApp processes three types of data:
- GPS location data (5-minute intervals)
- Accelerometer data (1-minute intervals)
- Labeled activity observations (1-minute intervals)

The package handles data loading, validation, feature extraction, and merging of these data sources.

## Todo

* Clean up the merging of data sources
* Unify configurations through inheritance 
* Build in the analysis section of the code
* Save logs to a file
* Documentation for changing sources
* Pass specific features to the analyzer

## Installation

### Development Installation
```bash
# Clone the repository
git clone https://github.com/Aspho1/CowStudyApp.git
cd CowStudyApp

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development version
pip install -e ".[dev]"
```

### Requirements
- Python ≥ 3.8
- See `pyproject.toml` for full dependency list

## Usage

### Basic Usage
```python
from pathlib import Path
from cowstudyapp.config import AppConfig
from cowstudyapp.io import DataLoader

# Load configuration
config = AppConfig.load("config/default.yaml")

# Initialize loader
loader = DataLoader(config)

# Load and process data
data = loader.load_data()  # Load raw data
merged_data = loader.load_and_merge()  # Load and merge with features
```

### Configuration

Create a `config/default.yaml` file:
```yaml
format: "multiple_files"
gps_directory: "data/raw/RB_22/gps"
accelerometer_directory: "data/raw/RB_22/accelerometer"
labeled_data_path: "data/raw/RB_22/labeled/gps_observations_2022.csv"

validation:
  timezone: "America/Denver"
  start_datetime: "2022-01-15 00:00:00"
  end_datetime: "2022-03-22 23:59:59"
  lat_min: 45.0
  lat_max: 47.0
  lon_min: -111.5
  lon_max: -110.5
  excluded_devices: [841, 1005]

features:
  enable_axis_features: true
  enable_magnitude_features: true
  feature_types:
    - "BASIC_STATS"
    - "PEAK_FEATURES"
    - "CORRELATION"
  window_size: 300
```

### Data Processing Pipeline

1. **Data Loading**
   - Reads raw CSV files with metadata headers
   - Converts timestamps to POSIX time
   - Standardizes column names

2. **Validation**
   - Checks data quality (GPS DOP, satellite count)
   - Validates coordinate bounds
   - Ensures proper time intervals
   - Handles missing data

3. **Feature Extraction**
   - Computes acceleration magnitude
   - Generates time-domain features
   - Optional: spectral features, rolling statistics

4. **Data Merging**
   - Aligns GPS and accelerometer data
   - Aggregates features to 5-minute windows
   - Integrates labeled observations

## Available Features

### Accelerometer Features
- Basic statistics (mean, variance)
- Peak features (peak-to-peak, crest factor)
- Signal magnitude area
- Correlation between axes
- Spectral features (optional)

### GPS Features
- UTM coordinates
- Quality indicators (DOP, satellite count)
- Time-aligned with accelerometer data

## Development

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
# Format code
black .
isort .

# Type checking
mypy src/cowstudyapp

# Linting
flake8 .
```

## Project Structure

### dev

├── .github/
│   ├── workflows/
│   │   ├── ci.yml
├── config/
│   ├── default_old.yaml
│   ├── default.yaml
├── data/
│   ├── processed/
│   ├── raw/
├── src/
│   ├── cowstudyapp/
│   │   ├── __init__.py
│   │   ├── config_old.py
│   │   ├── config.py
│   │   ├── core.py
│   │   ├── features_old.py
│   │   ├── features.py
│   │   ├── io.py
│   │   ├── main.py
│   │   ├── merge.py
│   │   ├── utils.py
│   │   ├── validation.py
│   ├── build_processed_data.py
├── tests/
│   ├── __init__.py
│   ├── test_core.py
│   ├── test_features.py
│   ├── test_io.py
│   ├── test_merge.py
│   ├── test_utils.py
├── .gitignore
├── pyproject.toml
├── README.md
├── show-tree.ps1

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- Thomas Lipinski (thomaslipinski@montana.edu)

## Acknowledgments

Developed for Montana State University's Ranch Sciences and Industrial Engineering Departments.

