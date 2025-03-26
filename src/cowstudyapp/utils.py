# src/cowstudyapp/utils.py
from datetime import datetime
import pandas as pd
import numpy as np
import pytz

def to_posix(timestamp: pd.Timestamp | datetime | str) -> int:
    """
    Convert a timestamp to POSIX time (seconds since Unix epoch).
    
    Args:
        timestamp: Input timestamp (can be pandas Timestamp, datetime, or string)
        
    Returns:
        Integer POSIX timestamp (seconds since Unix epoch)
    """
    if isinstance(timestamp, str):
        timestamp = pd.to_datetime(timestamp)
    return int(timestamp.timestamp())

def from_posix(posix_time: int | float, tz=None) -> pd.Timestamp:
    """
    Convert POSIX time to pandas Timestamp.
    
    Args:
        posix_time: POSIX timestamp (seconds since Unix epoch)
        

    Returns:
        pandas Timestamp
    """
    return pd.Timestamp.fromtimestamp(posix_time)



def from_posix_col(posix_col: pd.Series, tz) -> pd.Series:
    """
    Convert POSIX time to pandas Timestamp.
    
    Args:
        posix_time: POSIX timestamp (seconds since Unix epoch)
        

    Returns:
        pandas Timestamp
    """
    return pd.to_datetime(posix_col, unit='s', utc=True).dt.tz_convert(tz)


def add_posix_column(df: pd.DataFrame, timestamp_column: str = 'GMT Time') -> pd.DataFrame:
    """
    Add a posix_time column to a DataFrame based on an existing timestamp column.
    
    Args:
        df: Input DataFrame
        timestamp_column: Name of the column containing timestamps
        
    Returns:
        DataFrame with added posix_time column
    """
    df = df.copy()
    df['posix_time'] = df[timestamp_column].apply(to_posix)
    return df


def round_to_interval(posix_time: int, interval: int = 300, direction='nearest') -> int:
    """
    Round POSIX timestamp to nearest interval.
    
    Args:
        posix_time: POSIX timestamp in seconds
        interval: Interval in seconds (default 300 for 5 minutes)
        direction: ['nearest', 'up', 'down']
        
    Returns:
        Rounded POSIX timestamp
    """
    base:int = posix_time // interval
    remainder:int = (posix_time % interval) 
    ud: bool = remainder >= (interval/2)
    d:int = 1 if ((direction == 'up') | (ud)) else 0

    return (base + d) * interval

    if remainder >= interval/2:
        return (base + 1) * interval
    return base * interval

def round_timestamps(df: pd.DataFrame, col: str = 'posix_time', interval: int = 300, direction='nearest') -> pd.DataFrame:
    """Add column with time rounded to specified interval"""
    df = df.copy()
    # new_col = f"{col}_{interval}s"
    df[col] = df[col].apply(lambda x: round_to_interval(posix_time=x, interval=interval, direction=direction))
    return df

def list_valid_timezones():
    """Print all valid timezone options"""
    print("Valid timezones:")
    for tz in pytz.all_timezones:
        print(f"  - {tz}")

def process_time_column(time_str: str) -> str:
    """Ensure time string has seconds (Label data secific)"""
    return f'{time_str}:00' if len(time_str.split(':')) == 2 else time_str
