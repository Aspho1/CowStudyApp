# src/cowstudyapp/io.py
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple
import logging
import pytz

from ..utils import (
    add_posix_column,
    round_timestamps,
    process_time_column,
    from_posix,
)  # Add this import
from .features import GPSFeatures, FeatureComputation  # apply_feature_extraction

from .labels import LabelAggregation
from cowstudyapp.config import ConfigManager
from .validation import DataValidator


class DataLoader:

    DATEFORMATS = ["%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M"]

    def __init__(self, config: ConfigManager):
        """

        Returns
        -------
        object
        """
        if config.io is None:
            raise ValueError("IO configuration is required")
        if config.validation is None:
            raise ValueError("Validation configuration is required")
        if config.labels is None:
            raise ValueError("Label configuration is required")
        if config.features is None:
            raise ValueError("Feature configuration is required")

        # print("!!!!!!!!!!!!!", config.validation.dataset_name)

        self.config = config.io
        self.validation_config = config.validation
        self.validator = DataValidator(config.validation)  # Pass validation config
        self.labeler = LabelAggregation(config.labels)
        self.feature = FeatureComputation(config.features)
        # self.merger = DataMerger()
        self.excluded_devices = config.validation.excluded_devices

        self.quality_report: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            # For each GPS file, create an item of device_id => stats
            # After all files have been processed, fill in a summary item
            "gps": {
                "total_initial_records": 0,
                "total_final_records": 0,
                "valid_timerange": 0,
                'non_zero_vals': 0,
                "non_duplicates_post_standardization": 0,
                "valid_values": 0,
                "expected_records_per_device": self.validator.calculate_expected_records(
                    interval=self.config.gps_sample_interval
                ),
                'gaps': {},
                "devices_processed": 0,
                "devices": {},
            },
            # For each accelermometer file, create an item of device_id => stats
            # After all files have been processed, fill in a summary item
            "accelerometer": {
                "total_initial_records": 0,
                "valid_timerange": 0,
                "non_duplicates_post_standardization": 0,
                "valid_values": 0,
                "total_final_records": 0,
                "expected_records_per_device": self.validator.calculate_expected_records(
                    interval=self.config.acc_sample_interval
                ),
                "gaps": {},
                "total_windows_computed": 0,
                "devices_processed": 0,
                "devices": {},
            },
            # For each label file, create an item of device_id => stats
            # After all files have been processed, fill in a summary item
            "labels": {"devices": {}, "initial_records": 0, "final_records": 0},
            "config": {
                "validation": config.validation.__dict__,
                "io": config.io.__dict__,
                "labels": config.labels.__dict__,
                "features": config.features.__dict__,
            },
            "excluded_devices": {  # Add this section
                "devices": list(config.validation.excluded_devices),
                "filtered_records": {"gps": {}, "accelerometer": {}, "labels": {}},
            },
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
                return obj.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(obj, datetime):
                return obj.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(obj, Path):
                return str(obj)
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

        # output_dir = self.config.processed_data_path.parent
        report_path = f"{self.config.processed_data_path}_dqr.json"

        # Ensure the directory exists
        # output_dir.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w") as f:
            json.dump(serializable_report, f, indent=2)
        print(f"Saved data quality report to {report_path}")

    def load_data(self, aggregated: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Load unmerged data.

        Returns:
            Dict with separate 'gps', 'accelerometer', and 'label' DataFrames
        """
        data = self._load_multiple_files(aggregated=aggregated)
        return data

    def _load_multiple_files(self, aggregated: bool = True) -> Dict[str, pd.DataFrame]:
        gps_files = list(self.config.gps_directory.glob("*.csv"))
        accel_files = list(
            self.config.accelerometer_directory.glob("*.csv")
        )
        # logging.info(f"Processing {len(gps_files)} GPS files and {len(accel_files)} accelerometer files")
        print(
            f"Processing {len(gps_files)} GPS files and {len(accel_files)} accelerometer files"
        )
        print(f"Excluded devices: {', '.join([str(x) for x in self.excluded_devices])}")

        if not (
            self.validator.config.start_datetime and self.validator.config.end_datetime
        ):
            raise ValueError(
                "start_datetime and end_datetime must be configured for time frequency validation"
            )


        out: Dict[str, pd.DataFrame] = {}
        tz = pytz.timezone(self.config.timezone)
        startt = self.validator.config.start_datetime.astimezone(tz)
        endt = self.validator.config.end_datetime.astimezone(tz)

        self.quality_report["start_datetime"] = startt.strftime(self.DATEFORMATS[0])
        self.quality_report["end_datetime"] = endt.strftime(self.DATEFORMATS[0])

        if len(gps_files) > 1:
            out['gps'] = pd.concat([self._process_gps(f) for f in gps_files])
        else:
            print(gps_files)
            print(len(gps_files))
            print(gps_files[0])

            out['gps'] = self._process_gps(gps_files[0])


        self.quality_report["gps"]["total_expected_records"] = (
            self.quality_report["gps"]["expected_records_per_device"]
            * self.quality_report["gps"]["devices_processed"]
        )

        if len(accel_files) > 1:
            out['accelerometer'] = pd.concat([self._process_accel(f, aggregated=aggregated) for f in accel_files])
        else:

            out['accelerometer'] = self._process_accel(accel_files[0], aggregated=aggregated)



        self.quality_report["accelerometer"]["total_expected_records"] = (
            self.quality_report["accelerometer"]["expected_records_per_device"]
            * self.quality_report["accelerometer"]["devices_processed"]
        )


        if self.config.labeled_data_path is not None:
            out['label'] = self._process_labeled_data(
                self.config.labeled_data_path, aggregated=aggregated
            )


        self.save_quality_report()

        return out

    #################### Accelerometer

    def _process_accel_with_headers(self, file_path: Path) -> Optional[pd.DataFrame]:
        # logging.debug(f"Processing accelerometer file with headers: {file_path}")
        # print(f"Processing accelerometer file with headers: {file_path}")
        df: Optional[pd.DataFrame] = self._process_csv_file_with_headers(
            file_path=file_path
        )
        if df is None:
            return df
        # logging.info(f"Processed {len(df)} accelerometer records")
        # print(f"Processed {len(df)} accelerometer records")

        # Rename columns to standardized names
        column_mapping = {
            "GMT Time": "gmt_time",
            "X": "x",
            "Y": "y",
            "Z": "z",
            "Temperature [C]": "temperature_acc",
        }
        df.rename(columns=column_mapping, inplace=True)
        return df

    def _process_accel_no_headers(self, file_path: Path) -> Optional[pd.DataFrame]:
        # print(f"Processing accelerometer file no headers: {file_path}")
        df = pd.read_csv(file_path, parse_dates=["time"])
        # print(f"Loaded {len(df)} accelerometer records from `{str(file_path)}`")
        # print(df.head())

        column_mapping = {
            "time": "mountain_time",
            "X": "x",
            "Y": "y",
            "Z": "z",
            "temp": "temperature_acc",
            "collar": "device_id",
        }
        df.rename(columns=column_mapping, inplace=True)

        device_id = df["device_id"].iloc[0]
        if device_id in self.excluded_devices:
            logging.info(f"Skipping excluded device {device_id} from {file_path}")
            self.quality_report["excluded_devices"]["filtered_records"][
                "accelerometer"
            ][str(device_id)] = {
                "file": str(file_path),
                "total_records": len(df),
                "reason": "excluded_device",
            }
            return None

        datetime_parsed = False
        for dateformat in self.DATEFORMATS:
            try:
                df["mountain_time"] = pd.to_datetime(
                    df["mountain_time"], format=dateformat
                )
                df["gmt_time"] = df["mountain_time"].dt.tz_localize(
                    self.config.timezone
                )
                df["gmt_time"] = df["gmt_time"].dt.tz_convert(None)
                datetime_parsed = True
                break
            except ValueError:
                continue

        if not datetime_parsed:
            raise ValueError(
                f"Could not parse dates in {file_path} with any of the known formats: {self.DATEFORMATS}"
            )

        # print(df.head())

        # df.drop_duplicates(subset="gmt_time", inplace=True, keep='first')
        df = add_posix_column(df, timestamp_column="gmt_time")

        df.drop(columns=["gmt_time", "mountain_time"], inplace=True)

        return df

    def _process_accel(
        self, file_path: Path, aggregated: bool = True
    ) -> Optional[pd.DataFrame]:
        device_id = None
        format_type = ""
        try:
            # Try to detect file format
            with open(file_path, "r") as f:
                first_line = f.readline().strip()

            if first_line.lower().startswith("product type"):
                format_type = "original"
            elif first_line.lower().startswith("time"):
                format_type = "headless"
            else:
                raise ValueError(f"Unknown accelerometer data format in {file_path}")

            # Process based on format
            if format_type == "original":
                df = self._process_accel_with_headers(file_path=file_path)

            if format_type == "headless":
                df = self._process_accel_no_headers(file_path=file_path)

            if df is None:
                return df

            stats: Dict[str, Any] = {}

            device_id = str(df["device_id"].iloc[0])
            df[["x", "y", "z"]] *= 0.3138128

            print(
                f"------------------------------{device_id}---------------------------"
            )
            print(f"Source file path: {str(file_path)}")
            stats["file_path"] = str(file_path)
            stats["format_type"] = format_type

            stats["initial_records"] = df.shape[0]
            print(f"Initial accelerometer records: {stats['initial_records']}")
            self.quality_report["accelerometer"]["total_initial_records"] += stats[
                "initial_records"
            ]

            df2, accel_valid_stats = self.validator.validate_accelerometer(df)
            stats.update(accel_valid_stats)

            stats["final_records"] = df2.shape[0]
            print(f"Final accelerometer records: {stats['final_records']}")
            self.quality_report["accelerometer"]["total_final_records"] += stats["final_records"]
            self.quality_report["accelerometer"]["valid_timerange"] += stats["valid_timerange"]
            self.quality_report["accelerometer"]["non_duplicates_post_standardization"] += stats["non_duplicates_post_standardization"]
            self.quality_report["accelerometer"]["valid_values"] += stats["valid_values"]
            # self.quality_report['gps']['gaps'] 

            # Add in gaps

            if 'gap_analysis' in stats['gap_stats']:
                for k in stats['gap_stats']['gap_analysis']['gap_distribution'].keys():
                    if k in self.quality_report['accelerometer']['gaps'].keys():
                        self.quality_report['accelerometer']['gaps'][k] = self.quality_report['accelerometer']['gaps'][k] + stats['gap_stats']['gap_analysis']['gap_distribution'][k]
                    else:
                        self.quality_report['accelerometer']['gaps'][k] = stats['gap_stats']['gap_analysis']['gap_distribution'][k]


            # self.quality_report["accelerometer"]["total_final_records"] += stats[
            #     "final_records"
            # ]

            print(f"Computing features for device {device_id}...")

            if aggregated:
                df2, feature_stats = self.feature.compute_features(df2)
                stats["acc_feature_stats"] = feature_stats

                stats["windows_computed"] = df2.shape[0]
                self.quality_report["accelerometer"]["total_windows_computed"] += stats[
                    "windows_computed"
                ]

            if device_id not in self.quality_report["accelerometer"]["devices"]:
                self.quality_report["accelerometer"]["devices"][device_id] = stats

            self.quality_report["accelerometer"]["devices_processed"] += 1
            return df2

        except Exception as e:
            # self._handle_accelerometer_error(device_id, file_path, format_type, e)
            raise e

    #################### GPS

    def _process_gps(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Process GPS data file and collect statistics"""
        device_id = None

        try:
            # Initial processing
            df = self._process_csv_file_with_headers(file_path=file_path)

            if df is None:
                return df

            stats: Dict[str, Any] = {}

            device_id = str(df["device_id"].iloc[0])
            # self.quality_report['gps'][device_id] = {}

            initial_records = len(df)
            # print(f"!!!!!!Initial GPS records: {initial_records}")

            print(
                f"------------------------------{device_id}---------------------------"
            )
            print(f"Initial GPS records: {initial_records}")
            stats["initial_records"] = initial_records

            print(f"Source file path: {str(file_path)}")
            stats["file_path"] = str(file_path)
            self.quality_report["gps"]["total_initial_records"] += stats[
                "initial_records"
            ]

            # Standard column mapping and processing
            column_mapping = {
                "Latitude": "latitude",
                "Longitude": "longitude",
                "Altitude": "altitude",
                "Duration": "duration",
                "Temperature": "temperature_gps",
                "DOP": "dop",
                "Satellites": "satellites",
                "Cause of Fix": "cause_of_fix",
            }
            df.rename(columns=column_mapping, inplace=True)

            # Add UTM coordinates
            df = GPSFeatures.add_utm_coordinates(df)
            # print(f"ADDED UTM COORDS: {len(df)}")

            # Round timestamps
            df = round_timestamps(
                df,
                col="posix_time",
                interval=self.config.gps_sample_interval,
                direction="nearest",
            )
            # print(f"ROUNDED TIME STEPS: {initial_records}")

            # Select final columns
            desired_columns = [
                "posix_time",
                "device_id",
                "latitude",
                "longitude",
                "altitude",
                "temperature_gps",
                "dop",
                "satellites",
                "utm_easting",
                "utm_northing",
            ]
            df = df[desired_columns]

            # Collect statistics

            # print(df.columns)
            df, valid_gps_stats = self.validator.validate_gps(df)
            
            stats.update(valid_gps_stats)
            # for k in stats.keys():
            #     print(k)
            # print(f"VALIDATED GPS: {len(df)}")
            # print(f"EXPECTED: {(self.validation_config.end_datetime - self.validation_config.start_datetime).days + 1} -- {((self.validation_config.end_datetime - self.validation_config.start_datetime).days +1)* 288}")

            stats["final_records"] = df.shape[0]
            self.quality_report["gps"]["total_final_records"] += stats["final_records"]
            self.quality_report["gps"]["valid_timerange"] += stats["valid_timerange"]
            self.quality_report["gps"]["non_zero_vals"] += stats["non_zero_vals"]
            self.quality_report["gps"]["non_duplicates_post_standardization"] += stats["non_duplicates_post_standardization"]
            self.quality_report["gps"]["valid_values"] += stats["valid_values"]
            # self.quality_report['gps']['gaps'] 

            # Add in gaps
            for k in stats['gap_stats']['gap_analysis']['gap_distribution'].keys():
                if k in self.quality_report['gps']['gaps'].keys():
                    self.quality_report['gps']['gaps'][k] = self.quality_report['gps']['gaps'][k] + stats['gap_stats']['gap_analysis']['gap_distribution'][k]
                else:
                    self.quality_report['gps']['gaps'][k] = stats['gap_stats']['gap_analysis']['gap_distribution'][k]



            if device_id not in self.quality_report["gps"]["devices"]:
                self.quality_report["gps"]["devices"][device_id] = stats

            self.quality_report["gps"]["devices_processed"] += 1
            # return 
            return df

        except Exception as e:
            raise e

    #################### Label

    def _collect_label_stats(
        self, df: pd.DataFrame, format_type: str
    ) -> Dict[str, Any]:
        """Collect statistics about labeled data"""
        stats = {
            "format_type": format_type,
            "total_records": len(df),
            "unique_devices": df["device_id"].nunique(),
            "devices": {},
            "activity_counts": df["activity"].value_counts().to_dict(),
            "overall_time_range": {
                "start": from_posix(df["posix_time"].min()),
                "end": from_posix(df["posix_time"].max()),
            },
        }

        # Per-device statistics
        for device_id in df["device_id"].unique():
            device_df = df[df["device_id"] == device_id]
            stats["devices"][str(device_id)] = {
                "records": len(device_df),
                "activity_counts": device_df["activity"].value_counts().to_dict(),
                "time_range": {
                    "start": from_posix(device_df["posix_time"].min()),
                    "end": from_posix(device_df["posix_time"].max()),
                },
            }

        return stats

    def _process_labeled_data(
        self, file_path: Path, aggregated: bool = True
    ) -> pd.DataFrame:
        stats: Dict[str, Any] = {}
        try:
            # Try to detect file format
            with open(file_path, "r") as f:
                first_line = f.readline().strip()

            # Process data based on format
            if first_line.lower().startswith("date,time"):
                df, stats = self._process_labeled_data_standard(
                    file_path, aggregated=aggregated
                )
            elif first_line.lower().startswith("time"):
                df, stats = self._process_labeled_data_pivot(
                    file_path, aggregated=aggregated
                )
            else:
                raise ValueError(f"Unknown labeled data format in {file_path}")

            df["device_id"] = df["device_id"].astype(int)
            stats["file"] = str(file_path)
            # Add summary section

            initial_devices = set(df["device_id"].unique())
            initial_records = len(df)

            # Filter excluded devices
            df = df[~df["device_id"].isin(self.excluded_devices)]

            # Update stats with exclusion information
            excluded_devices = initial_devices & set(self.excluded_devices)
            if excluded_devices:
                stats["excluded_devices"] = {
                    "devices": list(excluded_devices),
                    "records_removed": initial_records - len(df),
                    "initial_records": initial_records,
                    "remaining_records": len(df),
                }
                print(
                    f"\nFiltered {stats['excluded_devices']['records_removed']} records from excluded devices {excluded_devices}"
                )

            self.quality_report["labels"] = stats
            return df

        except Exception as e:
            self.quality_report["labels"]["error"] = {
                "file": str(file_path),
                "error": str(e),
            }
            raise e

    def _process_labeled_data_pivot(
        self, file_path: Path, aggregated: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Process pivot-style labeled data format"""
        # Read the data with low_memory=False to avoid dtype warnings
        df = pd.read_csv(file_path, parse_dates=["time"], low_memory=False)

        print("\nInitial data shape:", df.shape)
        print("Columns:", df.columns.tolist())

        # Melt the dataframe to long format
        id_vars = ["time"]
        value_vars = [col for col in df.columns if col != "time"]

        df_melted = df.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name="tag_id",
            value_name="activity",
        )

        print("\nAfter melting:")
        print(df_melted.head())
        print(df_melted["activity"].value_counts())
        print(f"Number of records: {len(df_melted)}")

        df_melted["tag_id"] = df_melted["tag_id"].astype(str).str.zfill(4)

        # Drop rows with NaN tag_ids
        df_melted = df_melted.dropna(subset=["tag_id"])
        print(f"\nAfter dropping NaN tag_ids: {len(df_melted)} records")

        df_melted["device_id"] = df_melted["tag_id"].map(self.config.tag_to_device)

        # Drop rows where mapping failed
        df_melted = df_melted.dropna(subset=["device_id"])
        print(f"\nAfter mapping to collar_ids (device_id): {len(df_melted)} records")

        # Drop rows with missing activities
        df_melted = df_melted.dropna(subset=["activity"])
        print(f"After dropping NaN activities: {len(df_melted)} records")

        # Map activity labels to values
        if self.config.label_to_value is None:
            raise ValueError("label_to_value mapping is not configured")

        df_melted["activity"] = df_melted["activity"].map(self.config.label_to_value)

        # Convert to UTC and create posix time
        df_melted["mst_time"] = pd.to_datetime(df_melted["time"]).dt.tz_localize(
            "America/Denver"
        )
        df_melted["posix_time"] = (
            df_melted["mst_time"].dt.tz_convert("UTC").astype("int64") // 10**9
        )

        # Add 5-minute window column
        # df_melted['posix_time_5min'] = (df_melted['posix_time'] // self.config.gps_sample_interval) * self.config.gps_sample_interval

        # Select and rename columns
        result = df_melted[["posix_time", "device_id", "activity"]]

        if aggregated:
            result = self.labeler.compute_labels(result)

        stats = self._collect_label_stats(result, "pivot")

        # Add format-specific information
        stats.update(
            {
                "initial_shape": df.shape,
                "records_after_melting": len(df_melted),
                "records_after_dropping_nan_tags": len(
                    df_melted[~df_melted["tag_id"].isna()]
                ),
                "records_after_mapping": len(df_melted[~df_melted["device_id"].isna()]),
                "records_after_dropping_nan_activities": len(
                    df_melted[~df_melted["activity"].isna()]
                ),
                "tag_id_mapping_success_rate": f"{(len(df_melted[~df_melted['device_id'].isna()]) / len(df_melted)) * 100:.2f}%",
            }
        )

        # Store in quality report
        # self.quality_report['labels']['pivot_format'] = stats

        # Print summary
        print("\nLabel Quality Summary (Pivot Format):")
        print(f"Total records: {stats['total_records']}")
        print(f"Unique devices: {stats['unique_devices']}")
        print(
            f"\nTime range: {stats['overall_time_range']['start']} to {stats['overall_time_range']['end']}"
        )
        print("\nActivity counts:")
        for activity, count in stats["activity_counts"].items():
            print(f"{activity}: {count}")
        print("\nPer-device statistics:")
        for device_id, device_stats in stats["devices"].items():
            print(f"\nDevice {device_id}:")
            print(f"Records: {device_stats['records']}")
            print(
                f"Time range: {device_stats['time_range']['start']} to {device_stats['time_range']['end']}"
            )
            print("Activities:", device_stats["activity_counts"])

        return result, stats

    def _process_labeled_data_standard(
        self, file_path: Path, aggregated: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        df = pd.read_csv(
            file_path,
            names=["date", "time", "cow_id", "observer", "activity", "device_id"],
            parse_dates=["date"],
            skiprows=1,
        )
        print(f"Initial records: {len(df)}")

        # Clean and process the data
        print("Processing and cleaning data...")
        df = (
            df
            # Remove duplicates
            .drop_duplicates(
                subset=["date", "time", "cow_id", "observer", "activity", "device_id"],
                keep="first",
            )
            # Process time column
            .assign(
                time=lambda x: pd.to_timedelta(x["time"].apply(process_time_column))
            )
            # Combine date and time
            .assign(mst_time=lambda x: x["date"] + x["time"])
            # Fill missing activities with mode
            .assign(activity=lambda x: x["activity"].fillna(x["activity"].mode()[0]))
            # Drop unnecessary columns and reorder
            .drop(columns=["date", "time"])[
                ["mst_time", "cow_id", "observer", "activity", "device_id"]
            ]
        )

        print(f"After initial cleaning: {len(df)}")

        # Convert device_id to integer
        df[["device_id"]] = df[["device_id"]].astype(int)

        # Ensure mst_time is a datetime64 object without timezone first
        df["mst_time"] = pd.to_datetime(df["mst_time"], errors="coerce").dt.tz_localize(
            self.config.timezone
        )

        # Check for invalid entries that failed conversion
        invalid_dates = df["mst_time"].isnull().sum()
        if invalid_dates > 0:
            print(f"\nWarning: {invalid_dates} invalid date entries found:")
            print(df[df["mst_time"].isnull()])

        # Drop invalid dates
        df = df.dropna(subset=["mst_time"])
        print(f"Records after dropping invalid dates: {len(df)}")

        # Create POSIX time
        df["posix_time"] = df["mst_time"].dt.tz_convert("UTC").astype("int64") // 10**9

        df.drop(columns="mst_time", inplace=True)

        # Print summary statistics
        print("\nData Summary:")
        print(f"Total observations: {len(df)}")
        print(f"Unique collars: {df['device_id'].nunique()}")
        print(f"Unique activities: {df['activity'].unique()}")
        print("\nObservations per collar:")
        print(df["device_id"].value_counts().sort_index())

        # df['posix_time_5min'] = (df['posix_time'] // self.config.gps_sample_interval) * self.config.gps_sample_interval

        if aggregated:
            df = self.labeler.compute_labels(df)

        # Make a validate function here
        # How do we handle NA in labeled data?
        # Currently we just get lucky the RAW aggreagation doesnt land on them.
        # df =

        stats = self._collect_label_stats(df, "standard")

        # Add format-specific information
        stats.update(
            {
                "initial_records": len(df),
                "invalid_dates": invalid_dates,
                "records_after_cleaning": len(df),
                "observer_counts": (
                    df["observer"].value_counts().to_dict()
                    if "observer" in df.columns
                    else None
                ),
            }
        )

        # Store in quality report
        self.quality_report["labels"]["standard_format"] = stats

        # # Print summary
        # print("\nLabel Quality Summary (Standard Format):")
        # print(f"Total records: {stats['total_records']}")
        # print(f"Unique devices: {stats['unique_devices']}")
        # print(
        #     f"\nTime range: {stats['overall_time_range']['start']} to {stats['overall_time_range']['end']}"
        # )
        # print("\nActivity counts:")
        # for activity, count in stats["activity_counts"].items():
        #     print(f"{activity}: {count}")
        # print("\nPer-device statistics:")
        # for device_id, device_stats in stats["devices"].items():
        #     print(f"\nDevice {device_id}:")
        #     print(f"Records: {device_stats['records']}")
        #     print(
        #         f"Time range: {device_stats['time_range']['start']} to {device_stats['time_range']['end']}"
        #     )
        #     print("Activities:", device_stats["activity_counts"])

        return df, stats

    #################### Generic with headers

    def _process_csv_file_with_headers(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Process a CSV file and return a DataFrame.

        Args:
            file_path: Path to the CSV file

        Returns:
            DataFrame with standardized column names and extracted metadata
        """
        try:

            # Read metadata from first three lines
            with open(file_path, "r") as f:
                metadata_lines = [f.readline().strip() for _ in range(3)]

            metadata = {}
            for line in metadata_lines:
                if ":" in line:
                    key, value = [x.strip() for x in line.split(":", 1)]
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
                    df["GMT Time"] = pd.to_datetime(
                        df["GMT Time"], format=dateformat, utc=True
                    )
                    datetime_parsed = True
                    break
                except ValueError:
                    continue

            if not datetime_parsed:
                raise ValueError(
                    f"Could not parse dates in {file_path} with any of the known formats: {self.DATEFORMATS}"
                )

            #################### NEW
            try:
                product_id = metadata.get("Product ID", "0")
                product_id = "".join(
                    c for c in product_id if c.isdigit()
                )  # Keep only digits
                device_id = int(product_id) if product_id else 0

                # Check if device should be excluded - early return if excluded
                if device_id in self.excluded_devices:
                    logging.info(
                        f"Skipping excluded device {device_id} from {file_path}"
                    )
                    self.quality_report["excluded_devices"]["filtered_records"]["gps"][
                        str(device_id)
                    ] = {"file": str(file_path), "reason": "excluded_device"}
                    return None

                df["device_id"] = device_id

            except ValueError as e:
                logging.warning(
                    f"Could not parse Product ID from metadata in {file_path}: {str(e)}"
                )
                df["device_id"] = 0

            # try:
            #     product_id = metadata.get('Product ID', '0')
            #     product_id = ''.join(c for c in product_id if c.isdigit())  # Keep only digits
            #     df['device_id'] = int(product_id) if product_id else 0
            # except ValueError as e:
            #     logging.warning(f"Could not parse Product ID from metadata in {file_path}: {str(e)}")
            #     df['device_id'] = 0

            try:
                device_type = metadata.get("Product Type", "unknown")
                device_type = "".join(c for c in device_type.split(",") if len(c) > 0)
                df["device_type"] = device_type if device_type else "unknown"
            except ValueError as e:
                logging.warning(
                    f"Could not parse Device Type from metadata in {file_path}: {str(e)}"
                )
                df["device_type"] = "unknown"

            try:
                firmware_version = metadata.get("Firmware Version", "unknown")
                firmware_version = "".join(
                    c for c in firmware_version.split(",") if len(c) > 0
                )  # Keep only digits
                df["firmware_version"] = (
                    firmware_version if firmware_version else "unknown"
                )
            except ValueError as e:
                logging.warning(
                    f"Could not parse Firmware Version from metadata in {file_path}: {str(e)}"
                )
                df["firmware_version"] = "unknown"

            # df['firmware_version'] = metadata.get('Firmware Version', 'unknown')
            # print(df.head())

            # Add posix time
            df = add_posix_column(df)

            df.drop(columns="GMT Time", inplace=True)

            # Drop duplicates based on timestamp
            df.drop_duplicates(subset=["posix_time"], keep="first", inplace=True)

            return df

        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Error processing {file_path}: {str(e)}")
