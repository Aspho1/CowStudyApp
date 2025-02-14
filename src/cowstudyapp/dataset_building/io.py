# src/cowstudyapp/io.py
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple
import logging

from ..utils import add_posix_column, round_timestamps, process_time_column, from_posix  # Add this import
from .features import GPSFeatures, FeatureComputation #apply_feature_extraction

from .labels import LabelAggregation
from ..config import ConfigManager
from .validation import DataValidator


class DataLoader:

    DATEFORMATS = ['%m/%d/%Y %I:%M:%S %p', '%m/%d/%Y %H:%M']

    def __init__(self, config: ConfigManager):

        if config.io is None:
            raise ValueError("IO configuration is required")
        if config.validation is None:
            raise ValueError("Validation configuration is required")
        if config.labels is None:
            raise ValueError("Label configuration is required")
        if config.features is None:
            raise ValueError("Feature configuration is required")
        

        self.config = config.io
        self.validator = DataValidator(config.validation)  # Pass validation config
        self.labeler = LabelAggregation(config.labels)
        self.feature = FeatureComputation(config.features)
        # self.merger = DataMerger() 

        self.quality_report : Dict[str, Any] = {
            'timestamp': datetime.now().isoformat(),

            # For each GPS file, create an item of device_id => stats
            # After all files have been processed, fill in a summary item
            'gps': {},

            # For each accelermometer file, create an item of device_id => stats
            # After all files have been processed, fill in a summary item
            'accelerometer': {},

            # For each label file, create an item of device_id => stats
            # After all files have been processed, fill in a summary item
            'labels': {},

            'config': {
                'validation': config.validation.__dict__,
                'io': config.io.__dict__,
                'labels': config.labels.__dict__,
                'features': config.features.__dict__
            }
        }


#################### File handling

    def save_quality_report(self):
        """Save data quality report to JSON file"""
        # Convert any remaining non-serializable types
        def convert_to_serializable(obj):
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            # elif isinstance(obj, None):
            #     return "NA"
            elif isinstance(obj, (dict, list)):
                return obj
            return str(obj)

        # Recursively convert all values in the quality report
        def convert_dict(d):
            result = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    result[k] = convert_dict(v)
                elif isinstance(v, list):
                    result[k] = [convert_to_serializable(item) for item in v]
                else:
                    result[k] = convert_to_serializable(v)
            return result

        serializable_report = convert_dict(self.quality_report)

        report_path = 'data/processed/RB_19/data_quality_report.json'
        with open(report_path, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        print(f"Saved data quality report to {report_path}")

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load unmerged data.
        
        Returns:
            Dict with separate 'gps', 'accelerometer', and 'label' DataFrames
        """
        data = self._load_multiple_files()
        return data

    def _load_multiple_files(self) -> Dict[str, pd.DataFrame]:
        gps_files = list(self.config.gps_directory.glob(self.config.file_pattern))
        accel_files = list(self.config.accelerometer_directory.glob(self.config.file_pattern))
        # logging.info(f"Processing {len(gps_files)} GPS files and {len(accel_files)} accelerometer files")
        print(f"Processing {len(gps_files)} GPS files and {len(accel_files)} accelerometer files")
    
        gps_data = pd.concat([self._process_gps(f) for f in gps_files])
        accel_data = pd.concat([self._process_accel(f) for f in accel_files])
        label_data = self._process_labeled_data(self.config.labeled_data_path)
        self.save_quality_report()
        
        return {
            'gps': gps_data, 
            'accelerometer': accel_data, 
            'label' : label_data}
        

#################### Accelerometer

    def _process_accel_with_headers(self, file_path: Path) -> pd.DataFrame:
        # logging.debug(f"Processing accelerometer file with headers: {file_path}")
        print(f"Processing accelerometer file with headers: {file_path}")
        df = self._process_csv_file_with_headers(file_path=file_path)
        # logging.info(f"Processed {len(df)} accelerometer records")
        print(f"Processed {len(df)} accelerometer records")

        # Rename columns to standardized names
        column_mapping = {
            'GMT Time': 'gmt_time',
            'X': 'x',
            'Y': 'y',
            'Z': 'z',
            'Temperature [C]': 'temperature_acc'
        }
        df.rename(columns=column_mapping, inplace=True)
        return df

    def _process_accel_no_headers(self,file_path: Path) -> pd.DataFrame:
        print(f"Processing accelerometer file no headers: {file_path}")
        df = pd.read_csv(file_path, parse_dates=['time'])
        print(f"Processed {len(df)} accelerometer records")
        # print(df.head())

        column_mapping = {
            'time': 'mountain_time',
            'X': 'x',
            'Y': 'y',
            'Z': 'z',
            'temp': 'temperature_acc',
            "collar" : "device_id"
        }
        df.rename(columns=column_mapping, inplace=True)
        
        datetime_parsed = False
        for dateformat in self.DATEFORMATS:
            try:
                df["mountain_time"] = pd.to_datetime(df['mountain_time'], format=dateformat)
                df["gmt_time"] = df["mountain_time"].dt.tz_localize(self.config.timezone)
                df["gmt_time"] = df["gmt_time"].dt.tz_convert(None)
                datetime_parsed = True
                break
            except ValueError:
                continue

        if not datetime_parsed:
            raise ValueError(f"Could not parse dates in {file_path} with any of the known formats: {self.DATEFORMATS}")
        
        # print(df.head())

        # df.drop_duplicates(subset="gmt_time", inplace=True, keep='first')
        df = add_posix_column(df, timestamp_column='gmt_time')

        df.drop(columns=["gmt_time", 'mountain_time'], inplace=True)
        
        return df

    def _process_accel(self, file_path: Path) -> pd.DataFrame:
        device_id = None
        format_type = ''
        try:
            # Try to detect file format
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()


            if first_line.lower().startswith('product type'): 
                format_type = 'original'
            elif first_line.lower().startswith('time'): 
                format_type = 'headless'
            else:
                raise ValueError(f"Unknown accelerometer data format in {file_path}")

            # Process based on format
            if format_type == 'original':
                df = self._process_accel_with_headers(file_path) 
            
            if format_type == 'headless': 
                df = self._process_accel_no_headers(file_path)

            print(f"Observed file type is: {format_type}")
            print(df.head())

            device_id = str(df['device_id'].iloc[0])
            if 'accelerometer' not in self.quality_report:
                self.quality_report['accelerometer'] = {}
            if device_id not in self.quality_report['accelerometer']:
                self.quality_report['accelerometer'][device_id] = {}

            self.quality_report['accelerometer'][device_id] = {}
            self.quality_report['accelerometer'][device_id]['format_type'] = format_type

            df[['x', 'y', 'z']] *= 0.3138128

            # print("!!!!!!!!!!!!")
            # print(df["posix_time"].apply(from_posix).head())

            df, accel_valid_stats = self.validator.validate_accelerometer(df)
            # validation_results = self.validator.get_validation_stats()
            # print("!!!!!!!!!!!!!!!!!!!!!!!!")
            # print(df["posix_time"].apply(from_posix).head())

            print(f"Computing features for device {device_id}...")
            df = self.feature.compute_features(df)

            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # print(df["posix_time"].apply(from_posix).head())

            # Ensure JSON serializable values
            self.quality_report['accelerometer'][device_id] = {
                'file': str(file_path),
                'format': format_type,
                'validation_results': accel_valid_stats,
                'time_range': {
                    'start': int(df['posix_time'].min()),  # Convert np.int64 to int
                    'end': int(df['posix_time'].max())
                }
            }
            
            # Update summary statistics
            self._update_accelerometer_summary(device_id, format_type, df)
            
            return df
            
        except Exception as e:
            # self._handle_accelerometer_error(device_id, file_path, format_type, e)
            raise e

    def _update_accelerometer_summary(self, device_id: str, format_type: str, df: pd.DataFrame):
        """Update summary statistics for accelerometer processing"""
        if 'summary' not in self.quality_report['accelerometer']:
            self.quality_report['accelerometer']['summary'] = {}
        summary = self.quality_report['accelerometer']['summary']
        summary['total_devices'] = int(len(self.quality_report['accelerometer']))
        summary['total_records'] = int(summary.get('total_records', 0) + len(df))
        summary['files_processed'] = int(summary.get('files_processed', 0) + 1)
        
        if 'format_counts' not in summary:
            summary['format_counts'] = {'original': 0, 'pivot': 0}
        summary['format_counts'][format_type] = int(summary['format_counts'].get(format_type, 0) + 1)

    def _handle_accelerometer_error(self, device_id: Optional[str], file_path: Path, format_type: str, error: Exception):
        """Handle errors in accelerometer processing"""
        error_info = {
            'file': str(file_path),
            'error': str(error),
            'format_detected': format_type,
            'processing_stage': 'format_detection' if device_id is None else 'data_processing'
        }
        
        if device_id:
            self.quality_report['accelerometer']['devices'][device_id] = error_info
        else:
            if 'errors' not in self.quality_report['accelerometer']:
                self.quality_report['accelerometer']['errors'] = {}
            self.quality_report['accelerometer']['errors'][str(file_path)] = error_info
        
        if 'error_count' not in self.quality_report['accelerometer']['summary']:
            self.quality_report['accelerometer']['summary']['error_count'] = 0
        self.quality_report['accelerometer']['summary']['error_count'] += 1

#################### GPS

    def _collect_gps_stats(self, df: pd.DataFrame, format_type: str, file_path: Path) -> Dict[str, Any]:
        """Collect statistics about GPS data"""
        device_id = str(df['device_id'].iloc[0])
        
        stats = {
            'device_id': device_id,
            'format_type': format_type,
            'file': str(file_path),
            'total_records': len(df),
            'time_range': {
                'start': from_posix(df['posix_time'].min()),
                'end': from_posix(df['posix_time'].max())
            },
            'coordinate_bounds': {
                'latitude': {
                    'min': float(df['latitude'].min()),
                    'max': float(df['latitude'].max())
                },
                'longitude': {
                    'min': float(df['longitude'].min()),
                    'max': float(df['longitude'].max())
                },
                'altitude': {
                    'min': float(df['altitude'].min()),
                    'max': float(df['altitude'].max())
                }
            },
            'quality_metrics': {
                'satellites': {
                    'min': int(df['satellites'].min()),
                    'max': int(df['satellites'].max()),
                    'mean': float(df['satellites'].mean()),
                    'median': float(df['satellites'].median())
                },
                'dop': {
                    'min': float(df['dop'].min()),
                    'max': float(df['dop'].max()),
                    'mean': float(df['dop'].mean()),
                    'median': float(df['dop'].median())
                },
                'temperature': {
                    'min': float(df['temperature_gps'].min()),
                    'max': float(df['temperature_gps'].max()),
                    'mean': float(df['temperature_gps'].mean())
                }
            },
            'sampling_stats': {
                'expected_interval': self.config.gps_sample_interval,
                'actual_intervals': {
                    'min': float(df['posix_time'].diff().min()),
                    'max': float(df['posix_time'].diff().max()),
                    'mean': float(df['posix_time'].diff().mean()),
                    'median': float(df['posix_time'].diff().median())
                }
            }
        }
        
        return stats

    def _update_gps_summary(self, device_stats: Dict[str, Any]):
        """Update summary statistics for GPS processing"""
        if 'summary' not in self.quality_report['gps']:
            self.quality_report['gps']['summary'] = {
                'total_devices': 0,
                'total_records': 0,
                'files_processed': 0,
                'time_range': {
                    'start': None,
                    'end': None
                },
                'coordinate_bounds': {
                    'latitude': {'min': float('inf'), 'max': float('-inf')},
                    'longitude': {'min': float('inf'), 'max': float('-inf')},
                    'altitude': {'min': float('inf'), 'max': float('-inf')}
                },
                'quality_metrics': {
                    'satellites': {'min': float('inf'), 'max': float('-inf')},
                    'dop': {'min': float('inf'), 'max': float('-inf')},
                },
                'devices': {}
            }
        
        summary = self.quality_report['gps']['summary']
        
        # Update device count and records
        summary['total_devices'] = len(self.quality_report['gps'].get('devices', {}))
        summary['total_records'] += device_stats['total_records']
        summary['files_processed'] += 1
        
        # Update time range
        device_start = device_stats['time_range']['start']
        device_end = device_stats['time_range']['end']
        if summary['time_range']['start'] is None or device_start < summary['time_range']['start']:
            summary['time_range']['start'] = device_start
        if summary['time_range']['end'] is None or device_end > summary['time_range']['end']:
            summary['time_range']['end'] = device_end
            
        # Update coordinate bounds
        for coord in ['latitude', 'longitude', 'altitude']:
            summary['coordinate_bounds'][coord]['min'] = min(
                summary['coordinate_bounds'][coord]['min'],
                device_stats['coordinate_bounds'][coord]['min']
            )
            summary['coordinate_bounds'][coord]['max'] = max(
                summary['coordinate_bounds'][coord]['max'],
                device_stats['coordinate_bounds'][coord]['max']
            )
            
        # Store device-specific summary
        summary['devices'][device_stats['device_id']] = {
            'records': device_stats['total_records'],
            'time_range': device_stats['time_range']
        }

    def _process_gps(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Process GPS data file and collect statistics"""
        device_id = None
        try:
            # Initial processing
            df = self._process_csv_file_with_headers(file_path=file_path)
            device_id = str(df['device_id'].iloc[0])
            
            # Record initial stats
            initial_records = len(df)
            
            # Standard column mapping and processing
            column_mapping = {
                'Latitude': 'latitude',
                'Longitude': 'longitude',
                'Altitude': 'altitude',
                'Duration': 'duration',
                'Temperature': 'temperature_gps',
                'DOP': 'dop',
                'Satellites': 'satellites',
                'Cause of Fix': 'cause_of_fix'
            }
            df.rename(columns=column_mapping, inplace=True)

            # Filter zero coordinates
            zero_coords = df[(df['latitude'] == 0) & (df['longitude'] == 0)]
            df = df[(df['latitude'] != 0) & (df['longitude'] != 0)]
            
            # Add UTM coordinates
            df = GPSFeatures.add_utm_coordinates(df)
            
            # Round timestamps
            df = round_timestamps(df, col='posix_time', interval=self.config.gps_sample_interval)
            
            # Select final columns
            desired_columns = [
                'posix_time', 'device_id', 'latitude', 'longitude', 'altitude',
                'temperature_gps', 'dop', 'satellites', 'utm_easting', 'utm_northing'
            ]
            df = df[desired_columns]
            
            # Collect statistics
            stats = self._collect_gps_stats(df, 'original', file_path)
            stats.update({
                'processing_stats': {
                    'initial_records': initial_records,
                    'zero_coordinates_removed': len(zero_coords),
                    'final_records': len(df)
                }
            })
            
            # Store device-specific stats
            if 'devices' not in self.quality_report['gps']:
                self.quality_report['gps']['devices'] = {}
            self.quality_report['gps']['devices'][device_id] = stats
            
            # Update summary statistics
            self._update_gps_summary(stats)
            
            # Validate and return
            validated_df = self.validator.validate_gps(df)
            if validated_df is not None:
                stats['validation'] = {
                    'records_after_validation': len(validated_df),
                    'validation_success_rate': f"{(len(validated_df) / len(df)) * 100:.2f}%"
                }
            
            return validated_df
            
        except Exception as e:
            # Log failed files
            self.quality_report['gps']['errors'] = self.quality_report['gps'].get('errors', {})
            self.quality_report['gps']['errors'][str(file_path)] = {
                'device_id': device_id,
                'file': str(file_path),
                'error': str(e)
            }
            raise e

#################### Label

    def _process_labeled_data_NOValidation(self, file_path: Path) -> pd.DataFrame:
        # Try to detect file format
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
        
        if first_line.lower().startswith('date,time'): # Original format
            return self._process_labeled_data_standard(file_path)
        elif first_line.lower().startswith('time'): # New pivot format
            return self._process_labeled_data_pivot(file_path)
        else:
            raise ValueError(f"Unknown labeled data format in {file_path}")

    def _collect_label_stats(self, df: pd.DataFrame, format_type: str) -> Dict[str, Any]:
        """Collect statistics about labeled data"""
        stats = {
            'format_type': format_type,
            'total_records': len(df),
            'unique_devices': df['device_id'].nunique(),
            'devices': {},
            'activity_counts': df['activity'].value_counts().to_dict(),
            'overall_time_range': {
                'start': from_posix(df['posix_time'].min()),
                'end': from_posix(df['posix_time'].max())
            }
        }
        
        # Per-device statistics
        for device_id in df['device_id'].unique():
            device_df = df[df['device_id'] == device_id]
            stats['devices'][str(device_id)] = {
                'records': len(device_df),
                'activity_counts': device_df['activity'].value_counts().to_dict(),
                'time_range': {
                    'start': from_posix(device_df['posix_time'].min()),
                    'end': from_posix(device_df['posix_time'].max())
                }
            }
        
        return stats

    def _process_labeled_data(self, file_path: Path) -> pd.DataFrame:
        stats:Dict[str,Any] = {}
        try:
            # Try to detect file format
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
            
            # Process data based on format
            if first_line.lower().startswith('date,time'): 
                df, stats = self._process_labeled_data_standard(file_path)
            elif first_line.lower().startswith('time'): 
                df, stats = self._process_labeled_data_pivot(file_path)
            else:
                raise ValueError(f"Unknown labeled data format in {file_path}")
            
            df["device_id"] = df["device_id"].astype(int)
            stats['file'] = str(file_path)
            # Add summary section
            self.quality_report['labels'] = stats
            
            # print("\nLabel Quality Summary:")
            # print(f"Total records: {len(df)}")
            # print(f"Unique devices: {df['device_id'].nunique()}")
            # print("Records per device:")
            # print(df.groupby('device_id').size())
            # print("\nActivities per device:")
            # print(df.groupby(['device_id', 'activity']).size().unstack())
            
            return df
            
        except Exception as e:
            self.quality_report['labels']['error'] = {
                'file': str(file_path),
                'error': str(e)
            }
            raise e
        
    def _process_labeled_data_pivot(self, file_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Process pivot-style labeled data format"""
        # Read the data with low_memory=False to avoid dtype warnings
        df = pd.read_csv(file_path, parse_dates=['time'], low_memory=False)
        
        print("\nInitial data shape:", df.shape)
        print("Columns:", df.columns.tolist())
        
        # Melt the dataframe to long format
        id_vars = ['time']
        value_vars = [col for col in df.columns if col != 'time']
        
        df_melted = df.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name='tag_id',
            value_name='activity'
        )
        
        print("\nAfter melting:")
        print(df_melted.head())
        print(df_melted['activity'].value_counts())
        print(f"Number of records: {len(df_melted)}")
        
        
        df_melted['tag_id'] = df_melted['tag_id'].astype(str).str.zfill(4)
        
        # Drop rows with NaN tag_ids
        df_melted = df_melted.dropna(subset=['tag_id'])
        print(f"\nAfter dropping NaN tag_ids: {len(df_melted)} records")
            
        df_melted['device_id'] = df_melted['tag_id'].map(self.config.tag_to_device)
        
        # Drop rows where mapping failed
        df_melted = df_melted.dropna(subset=['device_id'])
        print(f"\nAfter mapping to collar_ids (device_id): {len(df_melted)} records")
        
        # Drop rows with missing activities
        df_melted = df_melted.dropna(subset=['activity'])
        print(f"After dropping NaN activities: {len(df_melted)} records")
        
        # Map activity labels to values
        if self.config.label_to_value is None:
            raise ValueError("label_to_value mapping is not configured")
            
        df_melted['activity'] = df_melted['activity'].map(self.config.label_to_value)
        
        # Convert to UTC and create posix time
        df_melted['mst_time'] = pd.to_datetime(df_melted['time']).dt.tz_localize('America/Denver')
        df_melted['posix_time'] = df_melted['mst_time'].dt.tz_convert('UTC').astype('int64') // 10**9
        
        # Add 5-minute window column
        df_melted['posix_time_5min'] = (df_melted['posix_time'] // self.config.gps_sample_interval) * self.config.gps_sample_interval
        
        # Select and rename columns
        result = df_melted[['posix_time', 'posix_time_5min', 'device_id', 'activity']]
        
        # Add any additional processing from original method
        result = self.labeler.compute_labels(result)


        stats = self._collect_label_stats(result, 'pivot')
        
        # Add format-specific information
        stats.update({
            'initial_shape': df.shape,
            'records_after_melting': len(df_melted),
            'records_after_dropping_nan_tags': len(df_melted[~df_melted['tag_id'].isna()]),
            'records_after_mapping': len(df_melted[~df_melted['device_id'].isna()]),
            'records_after_dropping_nan_activities': len(df_melted[~df_melted['activity'].isna()]),
            'tag_id_mapping_success_rate': f"{(len(df_melted[~df_melted['device_id'].isna()]) / len(df_melted)) * 100:.2f}%"
        })
        
        # Store in quality report
        # self.quality_report['labels']['pivot_format'] = stats
        
        # Print summary
        print("\nLabel Quality Summary (Pivot Format):")
        print(f"Total records: {stats['total_records']}")
        print(f"Unique devices: {stats['unique_devices']}")
        print(f"\nTime range: {stats['overall_time_range']['start']} to {stats['overall_time_range']['end']}")
        print("\nActivity counts:")
        for activity, count in stats['activity_counts'].items():
            print(f"{activity}: {count}")
        print("\nPer-device statistics:")
        for device_id, device_stats in stats['devices'].items():
            print(f"\nDevice {device_id}:")
            print(f"Records: {device_stats['records']}")
            print(f"Time range: {device_stats['time_range']['start']} to {device_stats['time_range']['end']}")
            print("Activities:", device_stats['activity_counts'])
        
        return result, stats
        



        
        return result

    def _process_labeled_data_standard(self,file_path: Path) -> Tuple[pd.DataFrame, Dict[str,Any]]:
        df = pd.read_csv(
            file_path,
            names=["date", "time", "cow_id", "observer", "activity", "device_id"],
            parse_dates=['date'],
            skiprows=1
        )
        print(f"Initial records: {len(df)}")
        
        # Clean and process the data
        print("Processing and cleaning data...")
        df = (df
            # Remove duplicates
            .drop_duplicates(
                subset=["date", "time", "cow_id", "observer", "activity", "device_id"],
                keep='first'
            )
            # Process time column
            .assign(
                time=lambda x: pd.to_timedelta(x['time'].apply(process_time_column))
            )
            # Combine date and time
            .assign(
                mst_time=lambda x: x['date'] + x['time']
            )
            # Fill missing activities with mode
            .assign(
                activity=lambda x: x['activity'].fillna(x['activity'].mode()[0])
            )
            # Drop unnecessary columns and reorder
            .drop(columns=['date', 'time'])
            [['mst_time', 'cow_id', 'observer', 'activity', 'device_id']]
        )
        
        print(f"After initial cleaning: {len(df)}")
        
        # Convert device_id to integer
        df[["device_id"]] = df[["device_id"]].astype(int)
        
        # Ensure mst_time is a datetime64 object without timezone first
        df['mst_time'] = pd.to_datetime(df['mst_time'], errors='coerce').dt.tz_localize('America/Denver')
        
        # Check for invalid entries that failed conversion
        invalid_dates = df['mst_time'].isnull().sum()
        if invalid_dates > 0:
            print(f"\nWarning: {invalid_dates} invalid date entries found:")
            print(df[df['mst_time'].isnull()])
        
        # Drop invalid dates
        df = df.dropna(subset=['mst_time'])
        print(f"Records after dropping invalid dates: {len(df)}")

        # Create POSIX time
        df['posix_time'] = df['mst_time'].dt.tz_convert('UTC').astype('int64') // 10**9

        df.drop(columns='mst_time',inplace=True)
        
        # Print summary statistics
        print("\nData Summary:")
        print(f"Total observations: {len(df)}")
        print(f"Unique collars: {df['device_id'].nunique()}")
        print(f"Unique activities: {df['activity'].unique()}")
        print("\nObservations per collar:")
        print(df['device_id'].value_counts().sort_index())

        df['posix_time_5min'] = (df['posix_time'] // self.config.gps_sample_interval) * self.config.gps_sample_interval
        
        df = self.labeler.compute_labels(df)

        # Make a validate function here
        # How do we handle NA in labeled data?
        # Currently we just get lucky the RAW aggreagation doesnt land on them. 
        # df = 

        stats = self._collect_label_stats(df, 'standard')
        
        # Add format-specific information
        stats.update({
            'initial_records': len(df),
            'invalid_dates': invalid_dates,
            'records_after_cleaning': len(df),
            'observer_counts': df['observer'].value_counts().to_dict() if 'observer' in df.columns else None
        })
        
        # Store in quality report
        self.quality_report['labels']['standard_format'] = stats
        
        # Print summary
        print("\nLabel Quality Summary (Standard Format):")
        print(f"Total records: {stats['total_records']}")
        print(f"Unique devices: {stats['unique_devices']}")
        print(f"\nTime range: {stats['overall_time_range']['start']} to {stats['overall_time_range']['end']}")
        print("\nActivity counts:")
        for activity, count in stats['activity_counts'].items():
            print(f"{activity}: {count}")
        print("\nPer-device statistics:")
        for device_id, device_stats in stats['devices'].items():
            print(f"\nDevice {device_id}:")
            print(f"Records: {device_stats['records']}")
            print(f"Time range: {device_stats['time_range']['start']} to {device_stats['time_range']['end']}")
            print("Activities:", device_stats['activity_counts'])

        return df, stats

#################### Generic with headers

    def _process_csv_file_with_headers(self, file_path: Path) -> pd.DataFrame:
        """
        Process a CSV file and return a DataFrame.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame with standardized column names and extracted metadata
        """
        try:
                
            # Read metadata from first three lines
            with open(file_path, 'r') as f:
                metadata_lines = [f.readline().strip() for _ in range(3)]
                
            metadata = {}
            for line in metadata_lines:
                if ':' in line:
                    key, value = [x.strip() for x in line.split(':', 1)]
                    metadata[key] = value

            # print(file_path)
            # Read the actual data
            df = pd.read_csv(
                file_path,
                skiprows=4,
                # parse_dates=['GMT Time'],
                # date_format = ,
                # date_parser=lambda x: pd.to_datetime(x, format='%m/%d/%Y %H:%M')
            )

            datetime_parsed = False
            for dateformat in self.DATEFORMATS:
                try:
                    df["GMT Time"] = pd.to_datetime(df['GMT Time'], format=dateformat, utc=True)
                    datetime_parsed = True
                    break
                except ValueError:
                    continue

            if not datetime_parsed:
                raise ValueError(f"Could not parse dates in {file_path} with any of the known formats: {self.DATEFORMATS}")
            
            # Add metadata as columns
            try:
                product_id = metadata.get('Product ID', '0')
                product_id = ''.join(c for c in product_id if c.isdigit())  # Keep only digits
                df['device_id'] = int(product_id) if product_id else 0
            except ValueError as e:
                logging.warning(f"Could not parse Product ID from metadata in {file_path}: {str(e)}")
                df['device_id'] = 0

            try:
                device_type = metadata.get('Product Type', 'unknown')
                device_type = ''.join(c for c in device_type.split(",") if len(c) > 0)
                df['device_type'] = device_type if device_type else 'unknown'
            except ValueError as e:
                logging.warning(f"Could not parse Device Type from metadata in {file_path}: {str(e)}")
                df['device_type'] = 'unknown'

            try:
                firmware_version = metadata.get('Firmware Version', 'unknown')
                firmware_version = ''.join(c for c in firmware_version.split(",") if len(c) > 0)  # Keep only digits
                df['firmware_version'] = firmware_version if firmware_version else 'unknown'
            except ValueError as e:
                logging.warning(f"Could not parse Firmware Version from metadata in {file_path}: {str(e)}")
                df['firmware_version'] = 'unknown'

            # df['firmware_version'] = metadata.get('Firmware Version', 'unknown')
            # print(df.head())

            # Add posix time
            df = add_posix_column(df)

            df.drop(columns="GMT Time", inplace=True)

            # Drop duplicates based on timestamp
            df.drop_duplicates(subset=['posix_time'], keep='first', inplace=True)

            return df
    
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Error processing {file_path}: {str(e)}")