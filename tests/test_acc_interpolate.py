from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from cowstudyapp.dataset_building.validation import DataValidator
from cowstudyapp.config import DataValidationConfig, ConfigManager

def test_interpolation():
    """Test that interpolation only fills single value gaps"""
    


    config_path = Path("config/RB_19_config.yaml")   
    config = ConfigManager.load(config_path).validation
    # config = load_config(config_path)

    
    # Create test data with gaps
    base_time = int(config.start_datetime.timestamp())
    test_data = {
        'posix_time': [
            base_time,
            base_time + 60,  # 1 min
            base_time + 120, # 2 min
            base_time + 180, # 3 min
            base_time + 240, # 4 min
            base_time + 300, # 5 min
            base_time + 360, # 6 min
        ],
        'x' : [1,  np.nan, 3,  4,  np.nan, np.nan, 7],
        'y' : [10, np.nan, 30, 40, np.nan, np.nan, 10],
        'z' : [10, np.nan, 30, 40, np.nan, np.nan, 10],
        'device_id': ['a',  None, 'a','a',   None,   None, 'a'],
        'temperature_acc': [20] * 7
    }
    
    df = pd.DataFrame(test_data)
    
    missing_mask = df.isna().any(axis=1)
    print(df[missing_mask])

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

    for missing_i in df[missing_mask].index:
        if not ((missing_i > 0) & (missing_i < df.shape[0])):
            continue
        
        prev_i = missing_i - 1
        next_i = missing_i + 1

        if (prev_i in df[missing_mask].index) | (next_i in df[missing_mask].index):
            continue

        for nc in numeric_cols:
            df.loc[missing_i, nc] = (df.loc[prev_i, nc] + df.loc[next_i, nc])/2
        for nnc in non_numeric_cols:
            df.loc[missing_i, nnc] = df.loc[prev_i, nnc]


    # print(df)


    validator = DataValidator(config)
    
    # Run validation
    result_df, stats = validator.validate_accelerometer(df)
    
    # Expected results:
    # [1, 2, 3, 4, nan, nan, 7] - only single gap between 1-3 should be filled
    
    print("\nOriginal data:")
    # print(df[['posix_time', 'x', 'y','z']].to_string())
    print(df.to_string())
    
    print("\nProcessed data:")
    # print(result_df[['posix_time', 'x', 'y', 'z']].to_string())
    print(result_df.to_string())
    
    # Assertions
    assert result_df.loc[1, 'x'] == 2  # Single gap should be interpolated
    assert np.isnan(result_df.loc[4, 'x'])  # Double gap should remain NaN
    assert np.isnan(result_df.loc[5, 'x'])  # Double gap should remain NaN
    
    print("\nAll tests passed!")

if __name__ == '__main__':
    test_interpolation()