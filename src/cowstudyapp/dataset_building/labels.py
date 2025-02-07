# import numpy as np
import pandas as pd
# from typing import List, Dict, Optional, Set
# from dataclasses import dataclass
# from enum import Enum, auto
from ..config import LabelAggTypeType, LabelConfig  # Import from config instead
from pandas.api.typing import DataFrameGroupBy


class LabelAggregation:
    def __init__(self, config: LabelConfig):
        self.config = config

    @staticmethod
    def compute_raw(gdf:DataFrameGroupBy) -> pd.DataFrame:
        # Group and get last values of specified columns    
        return gdf.agg({
            'activity': 'last',
            'observer': 'last'
        }).reset_index()
    
    @staticmethod
    def compute_mode(gdf:DataFrameGroupBy) -> pd.DataFrame:
        # Group and get mode values of specified columns    
        return gdf.agg({
            'activity': 'mode',
            'observer': 'mode'
        }).reset_index()


    @staticmethod
    def compute_percentile(gdf:DataFrameGroupBy) -> pd.DataFrame:
        # Group and get percentile values of specified columns    
        return NotImplemented
        return gdf.agg({
            'activity': 'mode',
            'observer': 'mode'
        }).reset_index()

    def compute_labels(self, df : pd.DataFrame) -> pd.DataFrame:

        grouped_df = df.groupby(['collar_id','posix_time_5min'])
        
        if self.config.labeled_agg_method == LabelAggTypeType.RAW:
            out = self.compute_raw(grouped_df)

        elif self.config.labeled_agg_method == LabelAggTypeType.MODE:
            out = self.compute_mode(grouped_df)

        elif self.config.labeled_agg_method == LabelAggTypeType.PERCENTILE:
            out = self.compute_percentile(grouped_df)

        out.rename(columns={'posix_time_5min': 'posix_time', 'collar_id': 'device_id'},inplace=True)

        return out
