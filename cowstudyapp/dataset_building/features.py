# src/cowstudyapp/features.py
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple
from pyproj import Transformer
from cowstudyapp.config import FeatureType, FeatureConfig  # Import from config instead


class FeatureValidationError(Exception):
    """Raised when feature computation fails validation"""
    pass


class AccelerometerFeatures:
    """Generate features from accelerometer data."""

    @staticmethod
    def compute_peak_features(signals: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute peak-related features.

        Args:
            signals: Dictionary of signal arrays

        Returns:
            Dictionary with features:
                - peak_to_peak: max - min
                - crest_factor: peak / rms
                - impulse_factor: peak / abs_mean

        Math:
            peak_to_peak = max(x) - min(x)
            crest_factor = max(|x|) / sqrt(mean(x^2))
            impulse_factor = max(|x|) / mean(|x|)
        """
        results = {}
        for axis, signal in signals.items():
            # Peak to peak
            peak_to_peak = float(np.ptp(signal))

            # RMS and absolute mean for factors
            rms = float(np.sqrt(np.mean(signal**2)))
            abs_mean = float(np.mean(np.abs(signal)))

            # Maximum absolute value
            peak = float(np.max(np.abs(signal)))

            results.update(
                {
                    f"{axis}_peak_to_peak": peak_to_peak,
                    f"{axis}_crest_factor": peak / rms if rms != 0 else 0,
                    f"{axis}_impulse_factor": peak / abs_mean if abs_mean != 0 else 0,
                }
            )
        return results

    @staticmethod
    def compute_entropy(
        signals: Dict[str, np.ndarray], bins: int = 20
    ) -> Dict[str, float]:
        """
        Compute signal entropy features.

        Args:
            signals: Dictionary of signal arrays
            bins: Number of bins for histogram (will be adjusted for small windows)

        Returns:
            Dictionary of entropy values per axis

        Math:
            entropy = -sum(p[i] * log(p[i]))
            where p[i] is the probability of signal being in bin i
        """
        results = {}
        for axis, signal in signals.items():
            # Adjust bins for small windows
            n_samples = len(signal)
            actual_bins = min(bins, n_samples)
            if actual_bins < 2:  # Need at least 2 bins for entropy
                results[f"{axis}_entropy"] = 0.0
                continue

            hist, _ = np.histogram(signal, bins=actual_bins, density=True)
            # Add small constant to avoid log(0)
            hist = hist + np.finfo(float).eps
            hist = hist / hist.sum()
            entropy = float(-np.sum(hist * np.log2(hist)))
            results[f"{axis}_entropy"] = entropy
        return results

    @staticmethod
    def compute_correlation_features(
        signals: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """
        Compute correlation between axes.

        Args:
            signals: Dictionary with 'x', 'y', 'z' arrays

        Returns:
            Dictionary with correlation coefficients:
                - xy_corr: correlation between x and y
                - yz_corr: correlation between y and z
                - xz_corr: correlation between x and z

        Math:
            correlation = cov(a,b) / (std(a) * std(b))
        """
        if not all(axis in signals for axis in ["x", "y", "z"]):
            raise FeatureValidationError("Correlation requires x, y, and z axes")

        # Check if we have enough data points (need at least 2 for correlation)
        if any(len(signal) < 2 for signal in signals.values()):
            return {"xy_corr": 0.0, "yz_corr": 0.0, "xz_corr": 0.0}

        results = {}
        pairs = [("x", "y"), ("y", "z"), ("x", "z")]

        for axis1, axis2 in pairs:
            try:
                # Check if either signal has zero variance
                if (np.std(signals[axis1]) == 0) or (np.std(signals[axis2]) == 0):
                    corr = 0.0
                else:
                    corr = float(np.corrcoef(signals[axis1], signals[axis2])[0, 1])
                    # Handle NaN results
                    if np.isnan(corr):
                        corr = 0.0
            except Exception:
                corr = 0.0

            results[f"{axis1}{axis2}_corr"] = corr

        return results

    @staticmethod
    def compute_spectral_features(
        signals: Dict[str, np.ndarray], sample_rate: float
    ) -> Dict[str, float]:
        """
        Compute frequency domain features.
        """
        results = {}
        for axis, signal in signals.items():
            # Check if signal is too short or empty
            if len(signal) < 2:  # Need at least 2 points for FFT
                results.update(
                    {
                        f"{axis}_dominant_freq": 0.0,
                        f"{axis}_dominant_period_minutes": 0.0,
                        f"{axis}_spectral_centroid": 0.0,
                    }
                )
                continue

            # Compute FFT
            fft = np.fft.rfft(signal)
            frequencies = np.fft.rfftfreq(len(signal), d=1 / sample_rate)
            magnitudes = np.abs(fft)

            # Only look at frequencies up to Nyquist frequency
            nyquist_idx = max(1, len(frequencies) // 2)  # Ensure at least 1

            # Check if magnitudes are all zero
            if np.all(magnitudes[:nyquist_idx] == 0):
                results.update(
                    {
                        f"{axis}_dominant_freq": 0.0,
                        f"{axis}_dominant_period_minutes": 0.0,
                        f"{axis}_spectral_centroid": 0.0,
                    }
                )
                continue

            # Dominant frequency (in Hz)
            dom_freq_idx = np.argmax(magnitudes[:nyquist_idx])
            dom_freq = float(frequencies[dom_freq_idx])
            results[f"{axis}_dominant_freq"] = dom_freq

            # Convert to period in minutes for easier interpretation
            # Avoid division by zero
            if dom_freq > 0:
                results[f"{axis}_dominant_period_minutes"] = float(1 / (dom_freq * 60))
            else:
                results[f"{axis}_dominant_period_minutes"] = 0.0

            # Spectral centroid
            total_magnitude = np.sum(magnitudes[:nyquist_idx])
            if total_magnitude > 0:
                centroid = float(
                    np.sum(frequencies[:nyquist_idx] * magnitudes[:nyquist_idx])
                    / total_magnitude
                )
            else:
                centroid = 0.0
            results[f"{axis}_spectral_centroid"] = centroid

        return results

    @staticmethod
    def validate_signals(signals: Dict[str, np.ndarray]) -> None:
        """
        Validate input signals.

        Args:
            signals: Dictionary of signal arrays

        Raises:
            FeatureValidationError: If validation fails
        """
        if not signals:
            raise FeatureValidationError("No signals provided")

        lengths = {len(signal) for signal in signals.values()}
        if len(lengths) > 1:
            raise FeatureValidationError("All signals must have same length")

        if any(np.isnan(signal).any() for signal in signals.values()):
            raise FeatureValidationError("Signals contain NaN values")

    @staticmethod
    def compute_basic_stats(signals: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute basic statistics for multiple signals.

        Args:
            signals: Dictionary of signal arrays
                    {'x': array, 'y': array, 'z': array}

        Returns:
            Dictionary of computed statistics
                {
                    'x_mean': float, 'y_mean': float, 'z_mean': float,
                    'x_var': float, 'y_var': float, 'z_var': float,
                    ...
                }

        Math:
            mean = (1/N) * sum(x[n])
            var  = (1/N) * sum((x[n] - mean)^2)
            where N is signal length
        """
        results = {}
        for axis, signal in signals.items():
            mean = float(np.mean(signal))
            var = float(np.mean((signal - mean) ** 2))

            results[f"{axis}_mean"] = mean
            results[f"{axis}_var"] = var

        return results

    @staticmethod
    def compute_magnitude(signals: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute magnitude from multiple axis signals.

        Args:
            signals: Dictionary with 'x', 'y', 'z' arrays

        Returns:
            Array of magnitude values

        Math:
            magnitude = sqrt(x^2 + y^2 + z^2)
        """
        return np.sqrt(signals["x"] ** 2 + signals["y"] ** 2 + signals["z"] ** 2)

    @staticmethod
    def compute_zero_crossings(signals: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute zero crossing rate for multiple signals.

        Args:
            signals: Dictionary of signal arrays

        Returns:
            Dictionary of zero crossing rates per axis

        Math:
            zcr = (1/N) * sum(|sign(x[n+1]) - sign(x[n])|)/2
            where N is signal length
        """
        results = {}
        for axis, signal in signals.items():
            zcr = float(np.sum(np.diff(np.signbit(signal)))) / (2 * len(signal))
            results[f"{axis}_zcr"] = zcr
        return results


class GPSFeatures:
    """Generate features from GPS data"""

    coordinate_transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:32612", always_xy=True  # WGS84  # UTM Zone 12N
    )

    @classmethod
    def convert_to_utm(cls, row: pd.Series) -> pd.Series:
        """
        Convert GPS coordinates to UTM.

        Args:
            row: Series containing 'longitude' and 'latitude'

        Returns:
            Series with [utm_easting, utm_northing]
        """
        easting, northing = cls.coordinate_transformer.transform(
            row["longitude"], row["latitude"]
        )
        return pd.Series([easting, northing])

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
        df[["utm_easting", "utm_northing"]] = df[["longitude", "latitude"]].apply(
            cls.convert_to_utm, axis=1
        )
        return df


class FeatureComputation:
    """Handles feature computation based on configuration"""

    def __init__(self, config: FeatureConfig):
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate feature configuration"""
        if not (
            self.config.enable_axis_features or self.config.enable_magnitude_features
        ):
            raise FeatureValidationError(
                "Must enable either axis or magnitude features"
            )
        if (
            FeatureType.CORRELATION in self.config.feature_types
            and not self.config.enable_axis_features
        ):
            raise FeatureValidationError("Correlation features require axis features")

    def compute_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Main entry point for feature computation.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: (Computed features DataFrame, Statistics)
        """
        # print(df.columns)

        stats: Dict[str, Any] = {
            "initial_records": len(df),
            "feature_types": [ft.value for ft in self.config.feature_types],
            "enabled_features": {
                "axis_features": self.config.enable_axis_features,
                "magnitude_features": self.config.enable_magnitude_features,
            },
            "windows": {"total": 0, "failed": 0, "success": 0},
            "computed_features": {},
            "computation_errors": [],
        }

        required_cols = ["device_id", "posix_time"]
        if self.config.enable_axis_features:
            required_cols.extend(["x", "y", "z"])

        # print(df.head())
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise FeatureValidationError(f"Missing columns: {missing_cols}")

        # Group by time windows
        window_size = self.config.gps_sample_interval
        df["window"] = (df["posix_time"] // window_size) * window_size

        results = []
        for (device_id, window), group in df.groupby(["device_id", "window"]):
            stats["windows"]["total"] += 1
            try:
                features, window_stats = self._compute_window_features(group)
                features["device_id"] = device_id
                features["posix_time"] = window
                results.append(features)

                # Update feature statistics
                for feature_name, value in features.items():
                    if feature_name not in ["device_id", "posix_time"]:
                        if feature_name not in stats["computed_features"]:
                            stats["computed_features"][feature_name] = {
                                "min": float("inf"),
                                "max": float("-inf"),
                                "sum": 0,
                                "count": 0,
                            }

                        feat_stats = stats["computed_features"][feature_name]
                        feat_stats["min"] = min(feat_stats["min"], value)
                        feat_stats["max"] = max(feat_stats["max"], value)
                        feat_stats["sum"] += value
                        feat_stats["count"] += 1

                stats["windows"]["success"] += 1

            except Exception as e:
                stats["windows"]["failed"] += 1
                stats["computation_errors"].append(
                    {"device_id": device_id, "window": window, "error": str(e)}
                )
                print(f"Warning: Failed window for device {device_id}: {str(e)}")
                continue

        if not results:
            raise FeatureValidationError("No features could be computed")

        # Calculate means for features
        for feature_name, feat_stats in stats["computed_features"].items():
            if feat_stats["count"] > 0:
                feat_stats["mean"] = feat_stats["sum"] / feat_stats["count"]
            del feat_stats["sum"]  # Remove running sum from final stats

        # Calculate success rate
        stats["windows"]["success_rate"] = (
            stats["windows"]["success"] / stats["windows"]["total"] * 100
            if stats["windows"]["total"] > 0
            else 0
        )

        return pd.DataFrame(results), stats

    def _compute_window_features(self, window: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Compute all enabled features for a window of data.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: (Features, Statistics)
        """

        # print("I EXIST!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        features: Dict[str, Any] = {}
        stats: Dict[str, Any] = {"window_size": len(window), "feature_computation": {}}

        try:
            # Prepare signals dictionary
            signals = {}
            if self.config.enable_axis_features:
                signals.update(
                    {
                        "x": window["x"].values,
                        "y": window["y"].values,
                        "z": window["z"].values,
                    }
                )

            if self.config.enable_magnitude_features:
                signals["magnitude"] = AccelerometerFeatures.compute_magnitude(signals)

            # Validate signals before processing
            AccelerometerFeatures.validate_signals(signals)

            # Compute enabled features
            for feature_type in self.config.feature_types:
                computation_stats = {"success": True, "error": None}

                try:
                    if feature_type == FeatureType.BASIC_STATS:
                        stats_features = AccelerometerFeatures.compute_basic_stats(
                            signals
                        )
                        features.update(stats_features)

                    elif feature_type == FeatureType.ZERO_CROSSINGS:
                        zcr = AccelerometerFeatures.compute_zero_crossings(signals)
                        features.update(zcr)

                    elif feature_type == FeatureType.PEAK_FEATURES:
                        peaks = AccelerometerFeatures.compute_peak_features(signals)
                        features.update(peaks)

                    elif feature_type == FeatureType.ENTROPY:
                        entropy = AccelerometerFeatures.compute_entropy(signals)
                        features.update(entropy)

                    elif (
                        feature_type == FeatureType.CORRELATION
                        and self.config.enable_axis_features
                    ):
                        corr = AccelerometerFeatures.compute_correlation_features(
                            signals
                        )
                        features.update(corr)

                    elif feature_type == FeatureType.SPECTRAL:
                        spectral = AccelerometerFeatures.compute_spectral_features(
                            signals, sample_rate=(1 / self.config.acc_sample_interval)
                        )
                        features.update(spectral)

                except Exception as e:
                    computation_stats["success"] = False
                    computation_stats["error"] = str(e)

                stats["feature_computation"][feature_type.value] = computation_stats

            return features, stats

        except Exception as e:
            if isinstance(e, FeatureValidationError):
                raise
            raise FeatureValidationError(f"Feature computation failed: {str(e)}")

