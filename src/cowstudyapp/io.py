# src/cowstudyapp/io.py
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

from .utils import add_posix_column, round_timestamps, process_time_column  # Add this import
from .features import AccelerometerFeatures, GPSFeatures, apply_feature_extraction

from .config_old import DataSourceConfig, LabelAggTypeType 
# from .config import DataSourceConfig 

from .validation import DataValidator
from cowstudyapp.merge import DataMerger

class DataLoader:

    DATEFORMAT = '%m/%d/%Y %I:%M:%S %p'

    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.validator = DataValidator(config.validation)  # Pass validation config
        self.merger = DataMerger()  # DataMerger doesn't need config anymore

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load unmerged data.
        
        Returns:
            Dict with separate 'gps' and 'accelerometer' DataFrames
        """
        data = self._load_multiple_files()
        return data

    def load_and_merge(self) -> pd.DataFrame:
        """
        Load and merge sensor data.
        
        Returns:
            Merged DataFrame with all features
        """
        data = self.load_data()
        return self.merger.merge_sensor_data(data)



    def _load_multiple_files(self) -> Dict[str, pd.DataFrame]:
        gps_files = list(self.config.gps_directory.glob(self.config.file_pattern))
        accel_files = list(self.config.accelerometer_directory.glob(self.config.file_pattern))
        logging.info(f"Processing {len(gps_files)} GPS files and {len(accel_files)} accelerometer files")
        
        gps_data = pd.concat([self._process_gps(f) for f in gps_files])
        accel_data = pd.concat([self._process_accel(f) for f in accel_files])
        label_data = self._process_labeled_data(self.config.labeled_data_path)
        
        return {
            'gps': gps_data, 
                'accelerometer': accel_data, 
                'label' : label_data}

    def _process_accel(self, file_path: Path) -> pd.DataFrame:
        # for ed in self.config.excluded_devices:
        #     if file_path.as_posix().__contains__(ed):
        #         continue
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

        # Convert accelerometer readings to m/s^2
        df[['x', 'y', 'z']] *= 0.3138128
        

        df = self.validator.validate_accelerometer(df)
        # df = apply_feature_extraction(df,self.config.features)
        df = apply_feature_extraction(df,self.config.features)

        return df



    def _process_gps(self, file_path: Path) -> Optional[pd.DataFrame]:

        df = self._process_csv_file(file_path=file_path)

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

        df = round_timestamps(df, col='posix_time', interval=self.config.features.gps_sample_interval)
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
        
        df = self.validator.validate_gps(df)
        
        return df

    def _process_labeled_data(self,file_path: Path) -> pd.DataFrame:
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
        
        # Group and get last values of specified columns
        grouped_df = df.groupby('posix_time_5min').agg({
            'activity': 'last',
            'collar_id': 'last',
            'observer': 'last'
        }).reset_index()

        grouped_df.rename(columns={'posix_time_5min': 'posix_time', 'collar_id': 'device_id'}, inplace=True)

        print(grouped_df.head())

        return grouped_df

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
            df["GMT Time"] = pd.to_datetime(df['GMT Time'], format=self.DATEFORMAT, utc=True)

            # Add metadata as columns
            df['device_id'] = int(metadata.get('Product ID', 0))
            df['device_type'] = metadata.get('Product Type', 'unknown')
            df['firmware_version'] = metadata.get('Firmware Version', 'unknown')
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