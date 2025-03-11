# src/cowstudyapp/validation.py
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pytz
from cowstudyapp.utils import from_posix

from ..config import DataValidationConfig
# from .config import DataValidationConfig

class DataValidator:
    '''
    This both filters and reports low quality data.
    '''
    # MAX_ACC_GAP = 2  # maximum number of minutes to interpolate
    
    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.ACC_INTERVAL = self.config.acc_sample_interval
        self.GPS_INTERVAL = self.config.gps_sample_interval
        # self.validation_stats: Dict[str, Any] = {}

    # def get_validation_stats(self):
    #     """Return the validation statistics for the most recent validation"""
    #     return self.validation_stats


    def validate_gps(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Validate GPS data and return cleaned DataFrame"""
        df = df.copy()

        # Initialize stats dictionary
        stats: Dict[str, Any]= {}
        
        # Apply time range filter
        df, timerange_stats = self._filter_timerange(df)
        stats['points_outside_of_study'] = timerange_stats


        df, zero_stats = self._filter_zero_vals_gps(df)
        stats['zero_val_stats'] = zero_stats
        
        df, frequency_stats = self._validate_time_frequency(df, self.GPS_INTERVAL)
        stats['frequency_stats'] = frequency_stats



        # print(df.columns)

        # Validate GPS-specific columns
        gps_columns = ['latitude', 'longitude', 'dop', 'satellites', 'altitude', 'temperature_gps']
        df, value_stats = self._validate_and_filter_values(df, gps_columns)
        stats['value_validation_stats'] = value_stats
        
        # print("!!!!!!!!!")
        # print(df[df.isna().any(axis=1)])

        # 4. Finally analyze gaps and interpolate with clean data only
        df, gap_statistics = self._gap_analysis_and_interpolation(
            df, 
            interval=self.GPS_INTERVAL, 
            interpolate=False # We could make this be true.... lets see the gaps
        )
        stats['gap_stats'] = gap_statistics

        # stats["final_rows"] = len(df)

        return df, stats

    def _filter_zero_vals_gps(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Analyze and filter zero coordinate values in GPS data.
        
        Args:
            df: DataFrame with GPS coordinates
            
        Returns:
            Tuple[DataFrame, Dict]: Filtered DataFrame and statistics about zero coordinates
        """
        stats: Dict[str, Any] = {}
        
        # Analyze zero coordinates by day
        zero_coords = (df['latitude'] == 0) & (df['longitude'] == 0)
        
        if zero_coords.any():
            # Convert posix time to local date
            tz = pytz.timezone(self.config.timezone)
            df['date'] = pd.to_datetime(df['posix_time'], unit='s', utc=True)\
                .dt.tz_convert(tz)\
                .dt.date
            
            # Calculate zero coordinates per day
            daily_zeros = df[zero_coords].groupby('date').size()
            daily_totals = df.groupby('date').size()
            daily_zero_pcts = (daily_zeros / daily_totals * 100).round(2)
            
            # Get the top 3 days with highest zero coordinate percentages
            worst_days = daily_zero_pcts.nlargest(3)
            
            stats['zero_coordinates'] = {
                'total_zero_coords': int(zero_coords.sum()),
                'total_records': len(df),
                'overall_daily_percentage': float((zero_coords.sum() / len(df) * 100).round(2)),
                'worst_days': {
                    str(date): {
                        'zero_count': int(daily_zeros[date]),
                        'total_records': int(daily_totals[date]),
                        'percentage': float(pct)
                    }
                    for date, pct in worst_days.items()
                }
            }
            
            print("\nZero coordinate analysis:")
            print(f"Total zero coordinates: {stats['zero_coordinates']['total_zero_coords']}")
            print(f"Overall percentage: {stats['zero_coordinates']['overall_daily_percentage']}%")
            print("\nWorst days for zero coordinates:")
            for date, day_stats in stats['zero_coordinates']['worst_days'].items():
                print(f"{date}: {day_stats['percentage']}% "
                    f"({day_stats['zero_count']}/{day_stats['total_records']} records)")
            
            # Remove the temporary date column
            df = df.drop(columns=['date'])
            
            # Filter out zero coordinates
            df = df[~zero_coords]
        
        return df, stats


    def calculate_expected_records(self, interval:int) -> int:
        if ((self.config.end_datetime is not None) and (self.config.start_datetime is not None)):
            return int((self.config.end_datetime - self.config.start_datetime).total_seconds() / interval)
        raise ValueError("End and Start times must be defined in config")

    def validate_accelerometer(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Validate accelerometer data and return cleaned DataFrame"""
        df = df.copy()
        device_id = str(df['device_id'].iloc[0])
        
        # Initialize stats dictionary
        stats: Dict[str, Any]= {}

        df, timerange_stats = self._filter_timerange(df)
        stats['points_outside_of_study'] = timerange_stats
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print(1, df.head())
        
        df, frequency_stats = self._validate_time_frequency(df, self.ACC_INTERVAL)
        stats['frequency_stats'] = frequency_stats
        print(2, df.head())

        accel_columns = ['x', 'y', 'z', 'temperature_acc']
        df, value_stats = self._validate_and_filter_values(df, accel_columns)
        stats['value_validation_stats'] = value_stats
        print(3, df.head())


        df, gap_statistics = self._gap_analysis_and_interpolation(df, interval=self.ACC_INTERVAL, interpolate=True)
        stats['gap_stats'] = gap_statistics
        print(4, df.head())
        
        return df, stats


#########################################################################################################

    def _validate_time_frequency(self, df: pd.DataFrame, interval: int) -> Tuple[pd.DataFrame, Dict]:
        """Create and merge df to the index of the expected records based on index. 
        Report statistics on saturation of the expected interval.

        Args:
            df (pd.DataFrame): DataFrame already filtered to the study period
            interval (int): The expected time between records

        Returns:
            Tuple[pd.DataFrame, Dict]: A DataFrame with the index between start and end time 
            with intervals interval. Additionally, returns a dictionary of statistics on saturation.
        """        

        frequency_stats: Dict[str,Any] = {}

        tz = pytz.timezone(self.config.timezone)
        start_posix = int(self.config.start_datetime.astimezone(tz).timestamp()) if self.config.start_datetime else None
        end_posix = int(self.config.end_datetime.astimezone(tz).timestamp()) if self.config.end_datetime else None

        # start_posix = int(self.config.start_datetime.timestamp()) 
        # end_posix = int(self.config.end_datetime.timestamp()) 
        # start_posix = max(int(self.config.start_datetime.timestamp()), df['posix_time'].min())
        # end_posix = min(int(self.config.end_datetime.timestamp()), df['posix_time'].max())
        
        if (not (isinstance(start_posix, int) and isinstance(start_posix, int))):
            raise ValueError("Error in validation -> _validate_time_frequency: start and/or end times not processed.")

        start_posix = start_posix - (start_posix % interval)
        
        # Create complete time range
        full_index = pd.DataFrame({
            'posix_time': range(
                start_posix,
                end_posix + interval,
                interval
            )
        })

        df = df.sort_values('posix_time')
        
        # Handle duplicates
        duplicates = df[df.duplicated(['posix_time'], keep=False)]

        print(f"\nFound {len(duplicates)} duplicate posix_time records")
        if len(duplicates) > 0:
            frequency_stats['duplicates'] = {}
            frequency_stats['duplicates']['n'] = duplicates.shape[0]

            print(f"Number of unique timestamps with duplicates: {len(duplicates['posix_time'].unique())}")
            frequency_stats['duplicates']['unique_times_with_dupes'] = len(duplicates['posix_time'].unique())
            
            print("\nExample duplicates:")
            example_time = duplicates['posix_time'].iloc[0]
            print(df[df['posix_time'] == example_time])
            frequency_stats['duplicates']['example_duplicate'] = df[df['posix_time'] == example_time].to_dict()
            
            df = df.drop_duplicates('posix_time', keep='first')
            print(f"After removing duplicates: {len(df)} records")
            frequency_stats['duplicates']['n_after_dropping'] = len(df)


        # Create a temporary DataFrame with the full time range
        full_df = pd.DataFrame(index=full_index['posix_time'])
        
        # Merge with original data
        df = df.set_index('posix_time')
        df = full_df.join(df)
        # Calculate missing percentage
        total_expected = len(full_index)
        records_present = df.notna().any(axis=1).sum()
        missing = total_expected - records_present
        missing_pct = (missing / total_expected) * 100
        
        frequency_stats.update({
            # 'n_expected': total_expected,
            'n_available': records_present,
            'n_missing': missing,
            'pct_missing': missing_pct
        })

        print(f"Time range: {self.config.start_datetime} to {self.config.end_datetime}")
        print(f"Expected records: {total_expected}")
        print(f"Records present: {records_present}")
        print(f"Missing records: {missing}")
        
        # print(df[df.isna().any(axis=1)])/
        df.reset_index(inplace=True)
        return df, frequency_stats
    





    def _gap_analysis_and_interpolation(self, df: pd.DataFrame, interval: int, interpolate:bool = False) -> Tuple[pd.DataFrame, Dict]:
        """
        Analyze gaps in the data using already identified missing records.
        Missing records are rows where all relevant columns are NaN.
        """        
        stats: Dict[str, Any] = {}
        
        # Identify missing records (all NaN rows)
        missing_records = df[df.isna().any(axis=1)].copy()
        
        if len(missing_records) > 0:
            # Sort by time to find consecutive gaps
            missing_records = missing_records.sort_values('posix_time')
            
            # Calculate time differences between consecutive missing records
            time_diffs = missing_records['posix_time'].diff()
            
            # Start new gap when time difference is greater than interval
            gap_breaks = time_diffs > interval
            gap_groups = gap_breaks.cumsum()
            
            # Group consecutive missing times into gaps
            gaps = []
            for group_id, group in missing_records.groupby(gap_groups):
                gap = {
                    'start_time': from_posix(group['posix_time'].iloc[0]).strftime("%Y-%m-%d %H:%M:%S"),
                    'end_time': from_posix(group['posix_time'].iloc[-1]).strftime("%Y-%m-%d %H:%M:%S"),
                    'gap_length_seconds': int(group['posix_time'].iloc[-1] - group['posix_time'].iloc[0] + interval),
                    'gap_length_intervals': len(group),
                    'missing_times': group['posix_time'].tolist()
                }
                gaps.append(gap)
            
            # Print gap analysis
            gap_lengths = [gap['gap_length_intervals'] for gap in gaps]
            gap_counts = pd.Series(gap_lengths).value_counts().sort_index()
            # for gap in gaps:
            #     print(f"\nGap from {gap['start_time']} to {gap['end_time']}")
            #     print(f"Length: {gap['gap_length_intervals']} intervals ({gap['gap_length_seconds']} seconds)")
            # Calculate gap distribution
            print(f"\nFound {len(gaps)} gaps:")
            print(gap_counts)
            
            # printable_gaps = gaps
            # printable_gaps['start_time'] = printable_gaps['start_time'].strftime("%Y-%m-%d %H:%M:%S")
            # printable_gaps['end_time'] = printable_gaps['end_time'].strftime("%Y-%m-%d %H:%M:%S")

            stats['gap_analysis'] = {
                'total_gaps': len(gaps),
                'total_missing_records': len(missing_records),
                'gap_distribution': gap_counts.to_dict(),
                'gaps': gaps
            }


            if interpolate:
                # Only interpolate single-interval gaps
                single_interval_gaps = [
                    gap for gap in gaps 
                    if gap['gap_length_intervals'] == 1
                ]
                
                n_interpolated = 0
                for gap in single_interval_gaps:
                    gap_time = gap['missing_times'][0]
                    gap_idx = df[df['posix_time'] == gap_time].index[0]
                    
                    # Get values before and after the gap
                    before_time = gap_time - interval
                    after_time = gap_time + interval
                    
                    before_row = df[df['posix_time'] == before_time]
                    after_row = df[df['posix_time'] == after_time]

                    # Check if we have actual data (not just posix_time) in before and after rows
                    has_before_data = not before_row.drop('posix_time', axis=1).isna().all(axis=1).iloc[0]
                    has_after_data = not after_row.drop('posix_time', axis=1).isna().all(axis=1).iloc[0]
                    
                    if has_before_data and has_after_data:

                        # Forward fill categorical and ID columns
                        categorical_cols = df.select_dtypes(include=['object']).columns
                        id_cols = [col for col in df.columns if 'id' in col.lower()]
                        for col in categorical_cols.union(id_cols):
                            if col != 'posix_time':  # Skip timestamp column
                                df.loc[gap_idx, col] = before_row[col].iloc[0]
                        
                        # Average numeric columns
                        numeric_cols = df.select_dtypes(include=np.number).columns
                        numeric_cols = numeric_cols.drop(['posix_time'])  # Skip timestamp column
                        
                        for col in numeric_cols:
                            before_val = before_row[col].iloc[0]
                            after_val = after_row[col].iloc[0]
                            if not (pd.isna(before_val) or pd.isna(after_val)):
                                df.loc[gap_idx, col] = (before_val + after_val) / 2
                        
                        n_interpolated += 1
                
                stats['interpolated_gaps'] = n_interpolated
        # # print("!!!!!!!!!")
        stats['dropped_na_records'] = df.isna().any(axis=0).sum()
        # print(len(df[df.isna().any(axis=1)]))      

        # print(df.head())
        df = df.dropna(how='any', axis=0) 
        # print(df.head())
        
        return df, stats



    def _validate_and_filter_values(self, df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate and optionally filter values based on configured rules.
        
        Args:
            df: DataFrame to validate
            columns: List of column names to validate
        
        Returns:
            Tuple of (filtered DataFrame, validation statistics)
        """
        stats: Dict[str, Any] = {
            'value_issues': {},
            'filtered_records': 0
        }
        
        # Create mask for records to filter
        filter_mask = pd.Series(False, index=df.index)
        
        for column in columns:
            if column not in self.config.validation_rules:
                continue
                
            rule = self.config.validation_rules[column]
            column_stats = {
                'total_violations': 0,
                'nan_count': 0,
                'out_of_bounds_count': 0,
                'bound_violation_examples': []  # Changed from 'examples' to be more specific
            }
            
            # Check for NaN values
            nan_mask = df[column].isna()
            column_stats['nan_count'] = nan_mask.sum()
            
            # Check for out of bounds values
            bounds_mask = pd.Series(False, index=df.index)
            valid_data = df[~nan_mask]
            
            if not valid_data.empty:
                lower_bound_mask = pd.Series(False, index=df.index)
                upper_bound_mask = pd.Series(False, index=df.index)
            
                # Apply min bound if configured
                if rule.min_value is not None:
                    lower_bound_mask = pd.Series(
                        (valid_data[column] < rule.min_value),
                        index=valid_data.index
                    )
                    bounds_mask.loc[lower_bound_mask.index] |= lower_bound_mask
                    
                # Apply max bound if configured
                if rule.max_value is not None:
                    upper_bound_mask = pd.Series(
                        (valid_data[column] > rule.max_value),
                        index=valid_data.index
                    )
                    bounds_mask.loc[upper_bound_mask.index] |= upper_bound_mask
            
            # Combine violations and update statistics
            violation_mask = nan_mask | bounds_mask
            column_stats['total_violations'] = violation_mask.sum()
            column_stats['out_of_bounds_count'] = bounds_mask.sum()
            
            # Collect examples of bound violations only (not NaN)
            if bounds_mask.any():
                # Get examples of lower bound violations
                if rule.min_value is not None and lower_bound_mask.any():
                    lower_examples = df.loc[lower_bound_mask.index[lower_bound_mask]].head(2)
                    for _, row in lower_examples.iterrows():
                        column_stats['bound_violation_examples'].append({
                            'posix_time': row['posix_time'],
                            'value': row[column],
                            'violation': f"below minimum ({rule.min_value})"
                        })
                
                # Get examples of upper bound violations
                if rule.max_value is not None and upper_bound_mask.any():
                    upper_examples = df.loc[upper_bound_mask.index[upper_bound_mask]].head(2)
                    for _, row in upper_examples.iterrows():
                        column_stats['bound_violation_examples'].append({
                            'posix_time': row['posix_time'],
                            'value': row[column],
                            'violation': f"above maximum ({rule.max_value})"
                        })
                
                print(f"\nValidation issues for {column}:")
                print(f"- NaN values: {column_stats['nan_count']}")
                print(f"- Out of bounds: {column_stats['out_of_bounds_count']}")
                
                # Print bound violation examples
                if column_stats['bound_violation_examples']:
                    print("Examples of bound violations:")
                    for example in column_stats['bound_violation_examples']:
                        print(f"  Time: {from_posix(example['posix_time'])}, "
                            f"Value: {example['value']:.2f}, "
                            f"Issue: {example['violation']}")
                
                # Update filter mask if rule requires filtering
                if rule.filter_invalid:
                    # filter_mask |= violation_mask
                    filter_mask |= bounds_mask
            
            stats['value_issues'][column] = column_stats
        
        # Apply filtering if needed
        if filter_mask.any():
            original_count = len(df)
            df = df[~filter_mask]
            filtered_count = original_count - len(df)
            stats['filtered_records'] = filtered_count
            print(f"\nFiltered {filtered_count} records due to validation rules")
        
        return df, stats


    def _filter_timerange(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply time range filters if configured"""
        
        if 'posix_time' not in df.columns:
            raise ValueError("DataFrame must have 'posix_time' column")

        stats: Dict[str, Any] = {}

        # Convert config datetime to POSIX if provided
        start_time = int(self.config.start_datetime.timestamp()) if self.config.start_datetime else None
        end_time = int(self.config.end_datetime.timestamp()) if self.config.end_datetime else None

        # Check records outside interval
        after_start = df['posix_time'] >= start_time
        if sum(~after_start) > 0:
            stats['before_study'] = sum(~after_start)

        before_end = df['posix_time'] <= end_time
        if sum(~before_end) > 0:
            stats['after_study'] = sum(~before_end)

        # Filter using POSIX times
        if start_time:
            df = df[df['posix_time'] >= start_time]
        if end_time:
            df = df[df['posix_time'] <= end_time]


        print("\nTime range analysis:")
        print(f"Data start time: {from_posix(df['posix_time'].min())}")
        print(f"Data end time: {from_posix(df['posix_time'].max())}")
        print(f"Expected start time: {from_posix(start_time)}")
        print(f"Expected end time: {from_posix(end_time)}")

        # Record first and last timestamps in filtered data
        if not df.empty:
            stats['first_record_in_study'] = from_posix(int(df['posix_time'].min()))
            stats['last_record_in_study'] = from_posix(int(df['posix_time'].max()))

        return df, stats