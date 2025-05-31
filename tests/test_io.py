# tests/test_io.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from cowstudyapp.dataset_building.io import DataLoader
from cowstudyapp.dataset_building.merge import DataMerger

from .test_config_factory import create_test_config_manager

# Constants
DATEFORMAT = '%m/%d/%Y %I:%M:%S %p'
CSV_DATEFORMAT = '%Y-%m-%d'
CSV_TIMEFORMAT = '%H:%M:%S'

@pytest.fixture
def test_config():
    """Configuration for testing"""
    cfg = create_test_config_manager()
    # Set a specific date range that matches our test data
    start_date = datetime(2022, 1, 15)
    end_date = datetime(2022, 1, 16)
    cfg.validation.start_datetime = start_date
    cfg.validation.end_datetime = end_date
    return cfg

@pytest.fixture
def sample_dates():
    """Create a consistent set of dates for all test data"""
    # Base date range for all data types
    start_date = datetime(2022, 1, 15, 8, 0, 0)  # Start at 8 AM
    end_date = datetime(2022, 1, 15, 18, 0, 0)   # End at 6 PM

    # Create different frequency series for different data types
    gps_dates = pd.date_range(start_date, end_date, freq='5min')
    acc_dates = pd.date_range(start_date, end_date, freq='1min')

    # Calculate overlap times for labeled data (subset of GPS times)
    labeled_dates = acc_dates[10:20]  # Take 10 GPS observations for labeling

    return {
        'gps': gps_dates,
        'accelerometer': acc_dates,
        'labeled': labeled_dates
    }

@pytest.fixture
def sample_labeled_data(test_config, sample_dates):
    """Create realistic labeled data that corresponds to GPS data"""
    # Get the dates that have labels
    label_dates = sample_dates['labeled']

    # Create the labeled data
    data = pd.DataFrame({
        'date': [d.strftime(CSV_DATEFORMAT) for d in label_dates],
        'time': [d.strftime(CSV_TIMEFORMAT) for d in label_dates],
        'cow_id': ['8132'] * len(label_dates),
        'observer': ['TestObserver'] * len(label_dates),
        'activity': ['Grazing', 'Grazing', 'Resting', 'Resting', 'Grazing',
                    'Grazing', 'Resting', 'Resting', 'Resting', 'Resting'],
        'collar': ['824'] * len(label_dates)  # Match to GPS/ACC device ID
    })

    # Save to file
    file_path = test_config.io.labeled_data_path
    data.to_csv(file_path, index=False)
    return file_path

@pytest.fixture
def sample_gps_data(test_config, sample_dates):
    """Create a realistic GPS sample file"""
    header = (
        "Product Type: Litetrack 800\n"
        "Product ID: 824\n"
        "Firmware Version: V8.107.0\n"
        "\n"
    )

    # Format dates for Vectronic format
    formatted_dates = [d.strftime(DATEFORMAT) for d in sample_dates['gps']]

    # Create sample data
    data = pd.DataFrame({
        'GMT Time': formatted_dates,
        'Latitude': np.random.uniform(45.5, 45.7, len(formatted_dates)),
        'Longitude': np.random.uniform(-111.1, -111.0, len(formatted_dates)),
        'Altitude': np.random.uniform(1400, 1600, len(formatted_dates)),
        'Duration': np.random.uniform(1, 60, len(formatted_dates)),
        'Temperature': np.random.uniform(10, 20, len(formatted_dates)),
        'DOP': np.random.uniform(0.8, 5.0, len(formatted_dates)),
        'Satellites': np.random.randint(4, 12, len(formatted_dates)),
        'Cause of Fix': 'GPS Schedule'
    })

    # Save to file
    file_path = test_config.io.gps_directory / "GPS_824.csv"

    with open(file_path, 'w') as f:
        f.write(header)
        data.to_csv(f, index=False)

    return file_path

@pytest.fixture
def sample_accelerometer_data(test_config, sample_dates):
    """Create a realistic accelerometer sample file"""
    header = (
        "Product Type: Litetrack 800\n"
        "Product ID: 824\n"
        "Firmware Version: V8.107.0\n"
        "\n"
    )

    # Format dates for Vectronic format
    formatted_dates = [d.strftime(DATEFORMAT) for d in sample_dates['accelerometer']]

    # Create sample data
    data = pd.DataFrame({
        'GMT Time': formatted_dates,
        'X': np.random.uniform(-16, 16, len(formatted_dates)),
        'Y': np.random.uniform(-16, 16, len(formatted_dates)),
        'Z': np.random.uniform(-16, 16, len(formatted_dates)),
        'Temperature [C]': np.random.uniform(10, 20, len(formatted_dates))
    })

    # Save to file
    file_path = test_config.io.accelerometer_directory / "ACC_824.csv"

    with open(file_path, 'w') as f:
        f.write(header)
        data.to_csv(f, index=False)

    return file_path



@pytest.fixture
def loaded_data(sample_gps_data, sample_accelerometer_data, sample_labeled_data, test_config):
    """Load all data and return the dictionary for other tests to use"""
    loader = DataLoader(test_config)
    data = loader.load_data()
    return data


def test_full_pipeline(loaded_data):
    """Test data loading process"""
    data = loaded_data

    # Verify GPS data
    assert 'gps' in data
    gps_df = data['gps']
    assert 'posix_time' in gps_df.columns
    assert 'latitude' in gps_df.columns
    assert 'device_id' in gps_df.columns
    assert gps_df['device_id'].nunique() == 1

    # print("Problem child")
    # print(gps_df['device_id'].values)
    assert 824 in gps_df['device_id'].values
    assert gps_df['satellites'].min() >= 4
    assert gps_df['dop'].max() <= 10.0
    assert len(gps_df['posix_time'].unique()) == len(gps_df)  # No duplicates

    # Verify accelerometer data
    assert 'accelerometer' in data
    acc_df = data['accelerometer']
    assert 'posix_time' in acc_df.columns


    # print(acc_df.columns)
    for acc_feature in ['x_entropy', 'y_entropy', 'z_entropy', 'magnitude_entropy', 'xy_corr',
       'yz_corr', 'xz_corr', 'x_mean', 'x_var', 'y_mean', 'y_var', 'z_mean',
       'z_var', 'magnitude_mean', 'magnitude_var', 'x_dominant_freq',
       'x_dominant_period_minutes', 'x_spectral_centroid', 'y_dominant_freq',
       'y_dominant_period_minutes', 'y_spectral_centroid', 'z_dominant_freq',
       'z_dominant_period_minutes', 'z_spectral_centroid',
       'magnitude_dominant_freq', 'magnitude_dominant_period_minutes',
       'magnitude_spectral_centroid', 'x_peak_to_peak', 'x_crest_factor',
       'x_impulse_factor', 'y_peak_to_peak', 'y_crest_factor',
       'y_impulse_factor', 'z_peak_to_peak', 'z_crest_factor',
       'z_impulse_factor', 'magnitude_peak_to_peak', 'magnitude_crest_factor',
       'magnitude_impulse_factor', 'x_zcr', 'y_zcr', 'z_zcr', 'magnitude_zcr',
       'device_id', 'posix_time']:

        assert acc_feature in acc_df.columns


    assert acc_df['device_id'].nunique() == 1
    assert 824 in acc_df['device_id'].values

    # Verify time intervals
    gps_time_diffs = np.diff(sorted(gps_df['posix_time'].unique()))
    assert np.median(gps_time_diffs) == 300  # 5-minute intervals

    acc_time_diffs = np.diff(sorted(acc_df['posix_time'].unique()))
    assert np.median(acc_time_diffs) == 300  # 5-minute intervals

    # Verify labeled data if loaded
    if 'labeled' in data:
        labeled_df = data['labeled']
        assert 'activity' in labeled_df.columns
        assert 'collar' in labeled_df.columns
        assert 'posix_time' in labeled_df.columns
        assert 'Grazing' in labeled_df['activity'].values
        assert 'Resting' in labeled_df['activity'].values


    merger = DataMerger()
    merged_df = merger.merge_sensor_data(loaded_data)

    print(merged_df.head())
    assert len(merged_df) > 0
    assert 'latitude' in merged_df.columns
    assert 'magnitude_mean' in merged_df.columns

    # Check time alignment and sorting
    assert merged_df['posix_time'].is_monotonic_increasing

    # Check that accelerometer data was properly merged with GPS
    # (every GPS timestamp should have accelerometer data)
    gps_times = set(gps_df['posix_time'])
    merged_with_acc = merged_df.dropna(subset=['magnitude_mean'])
    assert len(merged_with_acc) > 0

    # Check that labeled data was properly merged if available
    if 'activity' in merged_df.columns:
        labeled_rows = merged_df.dropna(subset=['activity'])
        assert len(labeled_rows) > 0
        assert set(labeled_rows['activity'].unique()) == {'Grazing', 'Resting'}

    print("\nData processing pipeline test passed successfully!")
