# src/cowstudyapp/validation.py
from datetime import datetime
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import pytz
from .config_old import DataValidationConfig
# from .config import DataValidationConfig

class DataValidator:
    '''
    This both filters and reports low quality data.
    '''
    GPS_INTERVAL = 300  # 5 minutes
    ACC_INTERVAL = 60  # 1 minute
    MAX_ACC_GAP = 2  # maximum number of minutes to interpolate
    
    def __init__(self, config: DataValidationConfig):
        self.config = config


    def validate_gps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate GPS data and return cleaned DataFrame"""
        df = df.copy()
        initial_records = len(df)
        print(f"------------------------------{df['device_id'].iloc[0]}---------------------------\nInitial GPS records: {initial_records}")
        
        # 1. First handle duplicates
        duplicates = df[df.duplicated(['posix_time'], keep=False)]
        if len(duplicates) > 0:
            print("\nDuplicate GPS fixes found:")
            print(f"Number of timestamps with duplicates: {len(duplicates['posix_time'].unique())}")
            print(f"Total duplicate records: {len(duplicates)}")
            
            # Print example of duplicates
            print("\nExample of duplicate fixes:")
            example_time = duplicates['posix_time'].iloc[0]
            print(df[df['posix_time'] == example_time][
                ['posix_time', 'latitude', 'longitude', 'altitude', 'dop']
            ])
            
            # Group and average
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'posix_time' in numeric_cols:
                numeric_cols.remove('posix_time')
                
            agg_dict = {col: 'mean' if col in numeric_cols else 'first' 
                    for col in df.columns if col != 'posix_time'}
            
            df = (df.groupby('posix_time', as_index=False)
                .agg(agg_dict))
            
            print(f"After averaging duplicates: {len(df)} unique records")

        # 2. Apply quality filters to actual data points
        valid_quality = (
            (df['satellites'] >= self.config.min_satellites) &
            (df['dop'] <= self.config.max_dop)
        )
        invalid_quality = df[~valid_quality]
        if len(invalid_quality) > 0:
            print(f"\nRemoving {len(invalid_quality)} records with poor quality")
            print("Example of poor quality records:")
            print(invalid_quality[['posix_time', 'satellites', 'dop']].head())
            df = df[valid_quality]

        # 3. Apply coordinate bounds
        valid_coords = (
            (df['latitude'].between(self.config.lat_min, self.config.lat_max)) &
            (df['longitude'].between(self.config.lon_min, self.config.lon_max))
        )
        invalid_coords = df[~valid_coords]
        if len(invalid_coords) > 0:
            print(f"\nRemoving {len(invalid_coords)} records with invalid coordinates")
            print("Example of invalid coordinates:")
            print(invalid_coords[['posix_time', 'latitude', 'longitude']].head())
            df = df[valid_coords]

        # 4. Apply time range filter
        df = self._validate_timerange(df)
        
        # 5. Report time coverage
        if self.config.start_datetime and self.config.end_datetime:
            start_posix = int(self.config.start_datetime.timestamp())
            end_posix = int(self.config.end_datetime.timestamp())
            total_intervals = (end_posix - start_posix) // self.GPS_INTERVAL + 1
            actual_records = len(df)
            coverage_pct = (actual_records / total_intervals) * 100
            
            print(f"\nTime Coverage Analysis:")
            print(f"Time range: {self.config.start_datetime} to {self.config.end_datetime}")
            print(f"Expected 5-minute intervals: {total_intervals}")
            print(f"Actual valid records: {actual_records}")
            print(f"Coverage: {coverage_pct:.1f}%")

            # Is atleast 30% of the data is missing, skip this data. 
            if coverage_pct < 70:
                return None
            
            # Optional: create full time range with NaN for missing values
            if False:  # Set to True if you want the missing intervals filled with NaN
                df, missing_pct = self._validate_time_frequency(df, self.GPS_INTERVAL)
        
        print(f"\nFinal GPS records: {len(df)}")
        return df



    def validate_accelerometer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate accelerometer data and return cleaned DataFrame"""
        df = df.copy()
        print(f"------------------------------{df['device_id'].iloc[0]}---------------------------\nInitial accelerometer records: {len(df)}")
        
        # Time bounds
        df = self._validate_timerange(df)
        df, missing_pct = self._validate_time_frequency(
            df, 
            self.ACC_INTERVAL, 
            interpolate=True
        )

        if missing_pct > 0:
            print(f"Accelerometer data missing {missing_pct:.5f}% of expected records")
            print(f"Expected: 1 record every {self.ACC_INTERVAL} seconds")
            print(f"Interpolated {int(df.shape[0] * missing_pct * 0.01)} gaps ≤ {self.MAX_ACC_GAP} minutes")

        # Acceleration bounds
        for axis in ['x', 'y', 'z']:
            valid_accel = df[axis].between(self.config.accel_min, self.config.accel_max)
            if not valid_accel.all():
                invalid = df[~valid_accel]
                print(f"Found {len(invalid)} records with invalid {axis}")
                df = df[valid_accel]

        # Temperature bounds
        valid_temp = df['temperature_acc'].between(self.config.temp_min, self.config.temp_max)
        if not valid_temp.all():
            invalid = df[~valid_temp]
            print(f"Found {len(invalid)} records with invalid temperature")
            df = df[valid_temp]

        return df


    def _validate_time_frequency(
        self, 
        df: pd.DataFrame, 
        interval: int,
        interpolate: bool = False
    ) -> Tuple[pd.DataFrame, float]:
        """
        Validate and optionally fix time frequency of data based on configured time range.
        
        Args:
            df: Input DataFrame
            interval: Expected interval in seconds
            interpolate: Whether to interpolate missing values
            
        Returns:
            Tuple of (processed DataFrame, percentage of missing data)
        """
        # Get configured time range
        if not (self.config.start_datetime and self.config.end_datetime):
            raise ValueError("start_datetime and end_datetime must be configured for time frequency validation")
        
        start_posix = int(self.config.start_datetime.timestamp())
        end_posix = int(self.config.end_datetime.timestamp())
        
        # Round start to nearest interval
        start_posix = start_posix - (start_posix % interval)
        
        # Create complete time range
        full_index = pd.DataFrame({
            'posix_time': range(
                start_posix,
                end_posix + interval,
                interval
            )
        })

        # Sort by time
        df = df.sort_values('posix_time')
        
        # Reindex data
        df = df.set_index('posix_time').reindex(full_index.posix_time)
        
        # Calculate missing percentage
        total_expected = len(full_index)
        records_present = df.notna().any(axis=1).sum()
        missing = total_expected - records_present
        missing_pct = (missing / total_expected) * 100
        
        print(f"Time range: {self.config.start_datetime} to {self.config.end_datetime}")
        print(f"Expected records: {total_expected}")
        print(f"Records present: {records_present}")
        print(f"Missing records: {missing}")
        
        if interpolate:
            # Interpolate small gaps
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = df[col].interpolate(
                    method='linear',
                    limit=self.MAX_ACC_GAP
                )
            
            # Forward fill non-numeric columns
            non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
            for col in non_numeric_cols:
                df[col] = df[col].ffill(limit=self.MAX_ACC_GAP)
        
        # Reset index
        df = df.reset_index()
        
        return df, missing_pct



    def _validate_timerange(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply time range filters if configured"""
        if 'posix_time' not in df.columns:
            raise ValueError("DataFrame must have 'posix_time' column")

        # Convert config datetime to POSIX if provided
        start_time = int(self.config.start_datetime.timestamp()) if self.config.start_datetime else None
        end_time = int(self.config.end_datetime.timestamp()) if self.config.end_datetime else None

        # Filter using POSIX times
        if start_time:
            df = df[df['posix_time'] >= start_time]
        if end_time:
            df = df[df['posix_time'] <= end_time]

        return df