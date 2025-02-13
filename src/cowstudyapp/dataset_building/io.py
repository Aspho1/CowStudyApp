# src/cowstudyapp/io.py
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

from ..utils import add_posix_column, round_timestamps, process_time_column  # Add this import
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

        self.quality_report = {
            'gps': {
                'devices': {},
                'summary': {
                    'total_devices': 0,
                    'total_records': 0,
                    'files_processed': 0
                }
            },
            'accelerometer': {
                'devices': {},
                'summary': {
                    'total_devices': 0,
                    'total_records': 0,
                    'files_processed': 0
                }
            },
            'labels': {
                'devices': {},
                'summary': {
                    'total_devices': 0,
                    'total_records': 0
                }
            },
            'timestamp': datetime.now().isoformat(),
            'config': {
                'validation': config.validation.__dict__,
                'io': config.io.__dict__,
                'labels': config.labels.__dict__,
                'features': config.features.__dict__
            }
        }


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
        logging.info(f"Processing {len(gps_files)} GPS files and {len(accel_files)} accelerometer files")
    
        gps_data = pd.concat([self._process_gps(f) for f in gps_files])
        accel_data = pd.concat([self._process_accel(f) for f in accel_files])
        self.save_quality_report()
        label_data = self._process_labeled_data(self.config.labeled_data_path)
        
        return {
            'gps': gps_data, 
            'accelerometer': accel_data, 
            'label' : label_data}
        


    def _process_accel_with_headers(self, file_path: Path) -> pd.DataFrame:
        logging.debug(f"Processing accelerometer file: {file_path}")
        df = self._process_csv_file(file_path=file_path)
        logging.info(f"Processed {len(df)} accelerometer records")

        # Rename columns to standardized names
        column_mapping = {
            'GMT Time': 'gmt_time',
            'X': 'x',
            'Y': 'y',
            'Z': 'z',
            'Temperature [C]': 'temperature_acc'
        }
        df.rename(columns=column_mapping, inplace=True)

        df['gmt_time'] = pd.to_datetime(df['gmt_time'], utc=True)

        df = add_posix_column(df,timestamp_column="gmt_time")
        df.drop(columns=['mountain_time'], inplace=True)

        # Add posix
        return df


    def _process_accel_no_headers(self,file_path: Path) -> pd.DataFrame:

        df = pd.read_csv(file_path, parse_dates=['time'])
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

        # Convert accelerometer readings to m/s^2
        df[['x', 'y', 'z']] *= 0.3138128
    
        # print(df.head())
        datetime_parsed = False
        for dateformat in self.DATEFORMATS:
            try:
                df["gmt_time"] = pd.to_datetime(df['mountain_time'], format=dateformat)
                df["gmt_time"] = df["gmt_time"].dt.tz_localize(self.config.timezone)
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
                format_type = 'pivot'
            else:
                raise ValueError(f"Unknown accelerometer data format in {file_path}")

            # Process based on format
            df = self._process_accel_with_headers(file_path) if format_type == 'original' else self._process_accel_no_headers(file_path)
            device_id = str(df['device_id'].iloc[0])
            df[['x', 'y', 'z']] *= 0.3138128

            df = self.validator.validate_accelerometer(df)
            validation_results = self.validator.get_validation_stats()

            print(f"Computing features for device {device_id}...")
            df = self.feature.compute_features(df)

            # Ensure JSON serializable values
            self.quality_report['accelerometer']['devices'][device_id] = {
                'file': str(file_path),
                'format': format_type,
                'validation_results': validation_results[device_id],
                'time_range': {
                    'start': int(df['posix_time'].min()),  # Convert np.int64 to int
                    'end': int(df['posix_time'].max())
                }
            }
            
            # Update summary statistics
            self._update_accelerometer_summary(device_id, format_type, df)
            
            return df
            
        except Exception as e:
            self._handle_accelerometer_error(device_id, file_path, format_type, e)
            raise e

    def _update_accelerometer_summary(self, device_id: str, format_type: str, df: pd.DataFrame):
        """Update summary statistics for accelerometer processing"""
        summary = self.quality_report['accelerometer']['summary']
        summary['total_devices'] = int(len(self.quality_report['accelerometer']['devices']))
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

    def _process_gps(self, file_path: Path) -> Optional[pd.DataFrame]:
        device_id = None
        try:
            df = self._process_csv_file(file_path=file_path)
            device_id = df['device_id'].iloc[0]
            column_mapping = {        
                # 'GMT Time' : 'gmt_time',
                'Latitude' : 'latitude', 
                'Longitude' : 'longitude', 
                'Altitude' : 'altitude', 
                'Duration' : 'duration', 
                'Temperature' : 'temperature_gps', 
                'DOP' : 'dop', 
                'Satellites' : 'satellites',
                'Cause of Fix' : 'cause_of_fix' 
            }
            df.rename(columns=column_mapping, inplace=True)

            df = df[(df['latitude'] != 0) & (df['longitude'] != 0)]
            
            df = GPSFeatures.add_utm_coordinates(df)

            df = round_timestamps(df, col='posix_time', interval=self.config.gps_sample_interval)
            # Ensure columns are in a consistent order
            desired_columns = [
                # 'gmt_time', 
                'posix_time'
                , 'device_id'
                # , 'device_type'
                # , 'firmware_version'
                , 'latitude'
                , 'longitude'
                , 'altitude'
                # , 'duration', 
                , 'temperature_gps'
                , 'dop'
                , 'satellites'
                #, 'cause_of_fix'
                , 'utm_easting'
                , 'utm_northing'
            ]
            df = df[desired_columns]
        
            self.quality_report['gps']['devices'][str(device_id)] = {
                'file': str(file_path),
                'initial_records': len(df),
                'final_records': len(df) if df is not None else 0,
                'validation_results': self.validator.get_validation_stats()
            }
            
            # Update summary
            self.quality_report['gps']['summary']['total_devices'] = len(self.quality_report['gps']['devices'])
            self.quality_report['gps']['summary']['total_records'] += len(df) if df is not None else 0
            self.quality_report['gps']['summary']['files_processed'] += 1
            
            return self.validator.validate_gps(df)
            
        except Exception as e:
            # Log failed files
            self.quality_report['gps'][str(device_id) if device_id else str(file_path)] = {
                'file': str(file_path),
                'error': str(e)
            }
            raise e


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


    def _process_labeled_data(self, file_path: Path) -> pd.DataFrame:
        try:
            # Try to detect file format
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
            
            # Process data based on format
            if first_line.lower().startswith('date,time'): 
                df = self._process_labeled_data_standard(file_path)
            elif first_line.lower().startswith('time'): 
                df = self._process_labeled_data_pivot(file_path)
            else:
                raise ValueError(f"Unknown labeled data format in {file_path}")
            
            for device_id in df['device_id'].unique():
                device_df = df[df['device_id'] == device_id]
                self.quality_report['labels'][str(device_id)] = {
                    'file': str(file_path),
                    'initial_records': len(device_df),
                    'unique_activities': device_df['activity'].nunique(),
                    'activity_counts': device_df['activity'].value_counts().to_dict(),
                    'validation_results': self.validator.get_validation_stats()
                }
            
            # Add summary section
            self.quality_report['labels']['summary'] = {
                'file': str(file_path),
                'total_records': len(df),
                'unique_devices': df['device_id'].nunique(),
                'unique_activities': df['activity'].nunique(),
                'activity_counts_all': df['activity'].value_counts().to_dict()
            }
            
            print("\nLabel Quality Summary:")
            print(f"Total records: {len(df)}")
            print(f"Unique devices: {df['device_id'].nunique()}")
            print("Records per device:")
            print(df.groupby('device_id').size())
            print("\nActivities per device:")
            print(df.groupby(['device_id', 'activity']).size().unstack())
            
            return df
            
        except Exception as e:
            self.quality_report['labels']['error'] = {
                'file': str(file_path),
                'error': str(e)
            }
            raise e
        

    def _process_labeled_data_pivot(self, file_path: Path) -> pd.DataFrame:
        """Process pivot-style labeled data format"""
        # Read the data
        df = pd.read_csv(file_path, parse_dates=['time'])
        
        # Melt the dataframe to long format
        df_melted = df.melt(
            id_vars=['time'],
            var_name='tag_id',
            value_name='activity'
        )

        print(df_melted.head())
        
        # Convert tag_ids to device_ids
        df_melted['device_id'] = df_melted['tag_id'].map(self.config.tag_to_device)
        
        # Drop rows with missing activities
        df_melted = df_melted.dropna(subset=['activity'])

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
        
        return result


    def _process_labeled_data_standard(self,file_path: Path) -> pd.DataFrame:
        df = pd.read_csv(
            file_path,
            names=["date", "time", "cow_id", "observer", "activity", "collar_id"],
            parse_dates=['date'],
            skiprows=1
        )
        print(f"Initial records: {len(df)}")
        
        # Clean and process the data
        print("Processing and cleaning data...")
        df = (df
            # Remove duplicates
            .drop_duplicates(
                subset=["date", "time", "cow_id", "observer", "activity", "collar_id"],
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
            [['mst_time', 'cow_id', 'observer', 'activity', 'collar_id']]
        )
        
        print(f"After initial cleaning: {len(df)}")
        
        # Convert collar_id to integer
        df[["collar_id"]] = df[["collar_id"]].astype(int)
        
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
        print(f"Unique collars: {df['collar_id'].nunique()}")
        print(f"Unique activities: {df['activity'].unique()}")
        print("\nObservations per collar:")
        print(df['collar_id'].value_counts().sort_index())

        df['posix_time_5min'] = (df['posix_time'] // self.config.gps_sample_interval) * self.config.gps_sample_interval
        
        df = self.labeler.compute_labels(df)

        # Make a validate function here
        # How do we handle NA in labeled data?
        # Currently we just get lucky the RAW aggreagation doesnt land on them. 
        # df = 

        return df

    def _process_csv_file(self, file_path: Path) -> pd.DataFrame:
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