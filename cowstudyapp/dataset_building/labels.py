# import numpy as np
import pandas as pd
# from typing import List, Dict, Optional, Set
# from dataclasses import dataclass
# from enum import Enum, auto
from cowstudyapp.config import (
    LabelAggTypeType,
    LabelConfig
    )
from pandas.api.typing import DataFrameGroupBy


def get_mode(x):
    # Get the mode, handle case where there might be multiple modes
    modes = x.mode()
    return modes.iloc[0] if not modes.empty else None


class LabelAggregation:
    def __init__(self, config: LabelConfig):
        self.config = config

    @staticmethod
    def compute_raw(gdf: DataFrameGroupBy) -> pd.DataFrame:
        # Group and get last values of specified columns
        return gdf.agg({
            'activity': 'last',
            # 'observer': 'last'
        }).reset_index()

    @staticmethod
    def compute_mode(gdf: DataFrameGroupBy) -> pd.DataFrame:
        # Group and get mode values of specified columns
        return gdf.agg({
            'activity': get_mode,
            # 'observer': 'mode'
        }).reset_index()

    @staticmethod
    def compute_percentile(gdf: DataFrameGroupBy) -> pd.DataFrame:
        # Group and get percentile values of specified columns
        return NotImplemented
        return gdf.agg({
            'activity': 'mode',
            # 'observer': 'mode'
        }).reset_index()

    def compute_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        df['posix_time_5min'] = (
            df['posix_time'] // self.config.gps_sample_interval
            ) * self.config.gps_sample_interval

        grouped_df = df.groupby(['device_id', 'posix_time_5min'])

        if self.config.labeled_agg_method == LabelAggTypeType.RAW:
            out = self.compute_raw(grouped_df)

        elif self.config.labeled_agg_method == LabelAggTypeType.MODE:
            out = self.compute_mode(grouped_df)

        elif self.config.labeled_agg_method == LabelAggTypeType.PERCENTILE:
            out = self.compute_percentile(grouped_df)

        out.rename(columns={
            'posix_time_5min': 'posix_time',
            'device_id': 'device_id'
                }, inplace=True)

        return out
