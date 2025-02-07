# tests/test_io.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from cowstudyapp.dataset_building.io import DataLoader
from cowstudyapp.config import IoConfig, DataValidationConfig, DataFormat

# tests/test_io.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from cowstudyapp.dataset_building.io import DataLoader
from cowstudyapp.config import IoConfig, DataValidationConfig, DataFormat

# Constants
DATEFORMAT = '%m/%d/%Y %I:%M:%S %p'

@pytest.fixture
def sample_gps_data(tmp_path):
    """Create a realistic GPS sample file"""
    header = (
        "Product Type: Litetrack 800\n"
        "Product ID: 824\n"
        "Firmware Version: V8.107.0\n"
        "\n"
    )
    
    # Create sample data with realistic dates and times
    dates = pd.date_range('2022-01-15', periods=100, freq='5min')
    formatted_dates = [d.strftime(DATEFORMAT) for d in dates]  # Convert to correct format
    
    data = pd.DataFrame({
        'GMT Time': formatted_dates,
        'Latitude': np.random.uniform(45.5, 45.7, 100),
        'Longitude': np.random.uniform(-111.1, -111.0, 100),
        'Altitude': np.random.uniform(1400, 1600, 100),
        'Duration': np.random.uniform(1, 60, 100),
        'Temperature': np.random.uniform(10, 20, 100),
        'DOP': np.random.uniform(0.8, 5.0, 100),
        'Satellites': np.random.randint(4, 12, 100),
        'Cause of Fix': 'GPS Schedule'
    })
    
    # Add some duplicate timestamps
    duplicates = data.iloc[:10].copy()
    duplicates['Latitude'] += 0.0001  # Slight variation in duplicates
    data = pd.concat([data, duplicates])
    
    # Save to file
    file_path = tmp_path / "gps" / "GPS_824.csv"
    file_path.parent.mkdir(exist_ok=True)
    
    with open(file_path, 'w') as f:
        f.write(header)
        data.to_csv(f, index=False)
    
    return file_path

@pytest.fixture
def sample_accelerometer_data(tmp_path):
    """Create a realistic accelerometer sample file"""
    header = (
        "Product Type: Litetrack 800\n"
        "Product ID: 824\n"
        "Firmware Version: V8.107.0\n"
        "\n"
    )
    
    # Create sample data with correct date format
    dates = pd.date_range('2022-01-15', periods=500, freq='1min')
    formatted_dates = [d.strftime(DATEFORMAT) for d in dates]
    
    data = pd.DataFrame({
        'GMT Time': formatted_dates,
        'X': np.random.uniform(-16, 16, 500),
        'Y': np.random.uniform(-16, 16, 500),
        'Z': np.random.uniform(-16, 16, 500),
        'Temperature [C]': np.random.uniform(10, 20, 500)
    })
    
    # Save to file
    file_path = tmp_path / "accelerometer" / "ACC_824.csv"
    file_path.parent.mkdir(exist_ok=True)
    
    with open(file_path, 'w') as f:
        f.write(header)
        data.to_csv(f, index=False)
    
    return file_path


@pytest.fixture
def test_config(tmp_path):
    """Create test configuration"""
    return IoConfig(
        format=DataFormat.MULTIPLE_FILES,
        gps_directory=tmp_path / "gps",
        accelerometer_directory=tmp_path / "accelerometer",
        file_pattern="*.csv",
        validation=DataValidationConfig(
            start_datetime=datetime(2022, 1, 15),
            end_datetime=datetime(2022, 3, 22),
            lat_min=45.0,
            lat_max=46.0,
            lon_min=-112.0,
            lon_max=-111.0,
            accel_min=-16.0,
            accel_max=16.0,
            temp_min=-40.0,
            temp_max=60.0,
            min_satellites=4,
            max_dop=10.0
        )
    )

def test_gps_processing(sample_gps_data, test_config):
    """Test GPS data processing"""
    loader = DataLoader(test_config)
    data = loader.load_data()
    
    assert 'gps' in data
    gps_df = data['gps']
    
    # Check basic structure
    assert 'posix_time' in gps_df.columns
    assert 'latitude' in gps_df.columns
    assert 'device_id' in gps_df.columns
    
    # Check data quality
    assert gps_df['device_id'].nunique() == 1
    assert gps_df['satellites'].min() >= 4
    assert gps_df['dop'].max() <= 10.0
    
    # Check for duplicate handling
    assert len(gps_df['posix_time'].unique()) == len(gps_df)

def test_accelerometer_processing(sample_accelerometer_data, test_config):
    """Test accelerometer data processing"""
    loader = DataLoader(test_config)
    data = loader.load_data()
    
    assert 'accelerometer' in data
    acc_df = data['accelerometer']
    
    # Check basic structure
    assert 'posix_time' in acc_df.columns
    assert 'x' in acc_df.columns
    assert 'accel_magnitude' in acc_df.columns
    
    # Check data quality
    assert acc_df['device_id'].nunique() == 1
    assert acc_df['x'].between(-16, 16).all()
    
    # Check time frequency
    time_diffs = np.diff(sorted(acc_df['posix_time'].unique()))
    assert np.median(time_diffs) == 60  # 1-minute intervals

def test_full_pipeline(sample_gps_data, sample_accelerometer_data, test_config):
    """Test entire data processing pipeline"""
    loader = DataLoader(test_config)
    
    # Test unmerged data
    data = loader.load_data()
    assert 'gps' in data
    assert 'accelerometer' in data
    
    # Test merged data
    merged_df = loader.load_and_merge()
    assert len(merged_df) > 0
    assert 'latitude' in merged_df.columns
    assert 'accel_magnitude' in merged_df.columns
    
    # Check time alignment
    assert merged_df['posix_time'].is_monotonic_increasing


# # Example of what the dates will look like:
# print("Sample date format:", pd.Timestamp('2022-01-15 10:30:00').strftime(DATEFORMAT))
# # Output: "01/15/2022 10:30:00 AM"
