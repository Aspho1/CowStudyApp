# CowStudyApp

A Python package for processing and analyzing sensor data from cattle-mounted GPS and accelerometer devices, developed for Montana State University's Ranch Sciences and Industrial Engineering Departments.

## Overview

CowStudyApp processes three types of data:
- GPS location data (5-minute intervals)
- Accelerometer data (1-minute intervals)
- Labeled activity observations (1-minute intervals)

The package handles data loading, validation, feature extraction, and merging of these data sources.

## Todo

* Handle when analysis.PRODUCT training_information dataset doesn't have an activity column in hmm (no labels)
* Generate ALL_ACTIVITIES instead of hardcoding it in the `run_hmm.r` script
* Figure out what is happening with reading csv's in R (what becomes NA)
* Save logs to a file
* Documentation for changing sources
* Test tests
* Fix or **delete** src/main.py
* Make distribution plots have colored bars by activity
* Fix hardcoded cutoff in `run_hmm.r` prepare_hmm_data
* for reading different data sets, check if the header row is expected. 

* Can we plot the distributions fits of the untrained model to see how much the model changes the parameters?


## Concerns
* The RB_19 data is confusing. The accelerometer data has loads of duplicates and datetimes which are nonsensical (such as data from 2045). Also there is no indication as to which timezone the data is recordd in.
<!-- * The accelerometer magnitudes seem waaay higher. -->
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

* Example `src/build_processed_data.py`

```python
from pathlib import Path
from cowstudyapp.io import DataLoader
from cowstudyapp.utils import from_posix
from cowstudyapp.config import ConfigManager
from cowstudyapp.merge import DataMerger


config_path = Path("config/default.yaml")   
config = ConfigManager.load(config_path)

# Load raw Acceelerometer, GPS, and label data
loader = DataLoader(config)
data = loader.load_data()

for key, df in data.items():
    print(f"\n{key.upper()} Data:")
    print(f"Total records: {len(df)}")
    print(f"Time range: {from_posix(df['posix_time'].min())} to {from_posix(df['posix_time'].max())}")
    print(f"Unique devices: {df['device_id'].unique()}")

# Combine the three datasources. 
merger = DataMerger()
merged_df = merger.merge_sensor_data(data)

# Save results
output_path = Path("data/processed/RB_22/all_cows_labeled.csv")
output_path.parent.mkdir(exist_ok=True)
merged_df.to_csv(output_path, index=False)

```

### Configuration

Create a `config/default.yaml` file:
```yaml
common:
  timezone: "America/Denver"
  gps_sample_interval: 300          # Expected seconds between GPS
  gps_sample_interval: 60           # Expected seconds between Accelerometer 
  excluded_devices: [841, 1005] 

io:
  format: "multiple_files"
  gps_directory: "data/raw/RB_22/gps"
  accelerometer_directory: "data/raw/RB_22/accelerometer"
  labeled_data_path: "data/raw/RB_22/labeled/gps_observations_2022.csv"
  file_pattern: "*.csv"

validation:
  start_datetime: "2022-01-15 00:00:00"
  end_datetime: "2022-03-22 23:59:59"
  lat_min: -90
  lat_max: 90
  lon_min: -180
  lon_max: 180
  accel_min: -41
  accel_max: 41
  temp_min: -99
  temp_max: 99
  min_satellites: 0
  max_dop: 10.0
  COVERAGE_THRESHOLD: 70            # Minimum ratio of (actual / expected) GPS data needed to include the device

features:
  enable_axis_features: true        # Compute features for each axes
  enable_magnitude_features: true   # Compute features for the magnitudes
  feature_types:
    - "BASIC_STATS"                 # Mean, variance of the window
    - "ZERO_CROSSINGS"              # Count of transitions from - to + or + to -
    - "PEAK_FEATURES"               # Statistics covering the peaks of the window
    - "ENTROPY"                     # Compute signal entropy features
    - "CORRELATION"                 # Compute correlation between axes
    - "SPECTRAL"                    # Compute frequency domain features
 
labels:
  # labeled_agg_method: "MODE"          # The most frequent activity in the window
  labeled_agg_method: "RAW"           # The activity at the end of the window
  # labeled_agg_method: "PERCENTILE"    # A weighted choice of activity

  valid_activities:
    - "Resting"
    - "Grazing"
    - "Traveling"
    - "Fighting"
    - "Scratching"
    - "Drinking"
    - "Mineral"

testing:
  device_id: 824  # Test with single cow 

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

### Label Aggregation Methods
- Raw: Use the last activity in the window as the window label 
- Mode: Use the most common activity in the window as the window label (NOT IMPLEMENTED)
- Percentile: Use a custom hierarchical strategy to determine the window label (NOT IMPLEMENTED)

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
```
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
```
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

