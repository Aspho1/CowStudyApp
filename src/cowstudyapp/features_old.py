# src/cowstudyapp/features.py
import numpy as np
import pandas as pd
from typing import List
from pyproj import CRS, Transformer

class AccelerometerFeatures:
    """Generate features from accelerometer data"""
    
    @staticmethod
    def add_magnitude(df: pd.DataFrame) -> pd.DataFrame:
        """Add acceleration magnitude from X,Y,Z components"""
        df = df.copy()
        df['accel_magnitude'] = np.sqrt(
            df['x']**2 + 
            df['y']**2 + 
            df['z']**2
        )
        return df
    
    @staticmethod
    def add_rolling_stats(
        df: pd.DataFrame, 
        window: str = '5min',
        columns: List[str] = None
    ) -> pd.DataFrame:
        """Add rolling statistics for specified columns"""
        df = df.copy()
        if columns is None:
            columns = ['accel_magnitude']
            
        for col in columns:
            rolling = df[col].rolling(window)
            df[f'{col}_mean'] = rolling.mean()
            df[f'{col}_std'] = rolling.std()
            df[f'{col}_max'] = rolling.max()
            
        return df

class GPSFeatures:
    """Generate features from GPS data"""
    
    # Class-level transformer for efficiency
    coordinate_transformer = Transformer.from_crs(
        'EPSG:4326',  # WGS84
        'EPSG:32612',  # UTM Zone 12N
        always_xy=True
    )

    @classmethod
    def add_utm_coordinates(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add UTM coordinates to GPS data.
        
        Args:
            df: DataFrame with 'latitude' and 'longitude' columns
            
        Returns:
            DataFrame with added 'utm_easting' and 'utm_northing' columns
        """
        df = df.copy()
        
        # Vectorized transformation (more efficient than loop)
        utm_coords = np.array(list(map(
            cls.coordinate_transformer.transform,
            df['longitude'],
            df['latitude']
        ))).T
        
        df['utm_easting'] = utm_coords[0]
        df['utm_northing'] = utm_coords[1]
        
        return df


def _compute_time_domain_features(group):
    """
    Compute time domain features for accelerometer data
    group: DataFrame with x, y, z columns
    """
    features = {}
    # For each axis
    for axis in ['x', 'y', 'z']:
        signal = group[f'{axis}'].values
        N = len(signal)
        
        # Mean: μ = (1/N) * Σx[n]
        mean = np.mean(signal)
        features[f'mean_{axis}'] = mean
        
        # Variance: σ² = (1/N) * Σ(x[n] - μ)²
        variance = np.var(signal)
        features[f'variance_{axis}'] = variance
        
        # Zero Crossing Rate: (1/N) * Σ|sign(x[n+1]) - sign(x[n])|/2
        zero_crossings = np.sum(np.diff(np.signbit(signal))) / (2 * N)
        features[f'zero_crossing_rate_{axis}'] = zero_crossings
        
        # Peak-to-peak amplitude: max(x) - min(x)
        peak_to_peak = np.ptp(signal)
        features[f'peak_to_peak_{axis}'] = peak_to_peak
        
        # Root Mean Square: sqrt((1/N) * Σx[n]²)
        rms = np.sqrt(np.mean(signal**2))
        features[f'rms_{axis}'] = rms
        
        # Mean Absolute Deviation: (1/N) * Σ|x[n] - μ|
        mad = np.mean(np.abs(signal - mean))
        features[f'mad_{axis}'] = mad
        
        # Inter-quartile Range: Q3 - Q1
        q75, q25 = np.percentile(signal, [75, 25])
        iqr = q75 - q25
        features[f'iqr_{axis}'] = iqr
        
        # Signal Energy: Σx[n]²
        energy = np.sum(signal**2)
        features[f'energy_{axis}'] = energy
    
    # Signal Magnitude Area: (1/N) * Σ(|x| + |y| + |z|)
    sma = np.mean(np.abs(group['x']) + 
                np.abs(group['y']) + 
                np.abs(group['z']))
    features['sma'] = sma

    features['temperature_acc_mean'] = group['temperature_acc'].mean()

    features['acc_mean'] = group['accel_magnitude'].mean()

    features['acc_var'] = group['accel_magnitude'].var()

    return pd.Series(features)


def aggregate_accelerometer_data(acc_df):
    """
    Aggregate accelerometer data by 5-minute windows
    """
    # Group by the 5-minute timestamp
    grouped = acc_df.groupby(['device_id', 'posix_time_5min'])
    
    # Compute features for each group
    features_df = grouped.apply(_compute_time_domain_features).reset_index()
    
    return features_df
