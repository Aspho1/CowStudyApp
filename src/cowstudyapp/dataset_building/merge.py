# src/cowstudyapp/merge.py
from datetime import timedelta
import pandas as pd
from typing import Dict
import logging
from ..config import ConfigManager
# from .config import AppConfig
from .features import FeatureComputation, FeatureValidationError

class DataMerger:
    """
    Handles merging of GPS and accelerometer data with feature computation.
    """
    def __init__(self):
        """
        Initialize DataMerger.
        """
        # self.config = config
        self.logger = logging.getLogger(__name__)
        # self.feature_computer = FeatureComputation(config.features)

    def merge_sensor_data(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge GPS, accelerometer, and label data with feature computation.
        
        Args:
            data: Dictionary containing 'gps', 'accelerometer' and 'label' DataFrames
            
        Returns:
            Merged DataFrame with computed features and associated labels
            
        Raises:
            ValueError: If required data is missing or invalid
        """
        # Validate input
        if not all(k in data for k in ['gps', 'accelerometer', 'label']):
            raise ValueError("All of 'gps', 'accelerometer', and 'label' data are required")
        try:
            # Copy data to avoid modifying originals
            gps_df = data['gps'].copy()
            acc_df = data['accelerometer'].copy()
            label_df = data['label'].copy()

            # First merge accelerometer and GPS data
            acc_gps_df = pd.merge(
                acc_df,
                gps_df,
                on=['device_id', 'posix_time'],
                how='inner'
            )

            # Then merge with labels
            merged_df = pd.merge(
                acc_gps_df,
                label_df,
                on=['device_id', 'posix_time'],
                how='left'
            )
            # Log merge statistics
            self._log_merge_statistics(merged_df)
            
            return merged_df

        except FeatureValidationError as e:
            self.logger.error(f"Feature computation failed: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Merge failed: {e}")
            raise

    def _log_merge_statistics(self, df: pd.DataFrame) -> None:
        """Log statistics about the merged data."""
        self.logger.info("\nMerge Summary:")
        
        for device_id, device_data in df.groupby('device_id'):
            total_records = len(device_data)
            records_with_gps = device_data['latitude'].notna().sum()
            records_without_gps = device_data['latitude'].isna().sum()
            
            self.logger.info(f"\nDevice ID: {device_id}")
            self.logger.info(f"Total records: {total_records}")
            self.logger.info(f"Records with GPS: {records_with_gps}")
            self.logger.info(f"Records without GPS: {records_without_gps}")
            self.logger.info(f"GPS coverage: {(records_with_gps/total_records)*100:.1f}%")

            # Log feature statistics
            feature_cols = [col for col in df.columns 
                          if col not in ['device_id', 'posix_time', 'latitude', 'longitude']]
            self.logger.info(f"Computed features: {len(feature_cols)}")
            
            # Log any completely missing features
            missing_features = [col for col in feature_cols 
                              if device_data[col].isna().all()]
            if missing_features:
                self.logger.warning(f"Missing features for device {device_id}: {missing_features}")