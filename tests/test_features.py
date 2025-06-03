import pytest
import pandas as pd
import numpy as np
from cowstudyapp.dataset_building.features import (
    GPSFeatures,
    AccelerometerFeatures,
    FeatureComputation,
    FeatureValidationError
)
from .test_config_factory import create_test_feature_config


@pytest.fixture
def sample_gps_data():
    return pd.DataFrame({
        'latitude': [45.6669, 45.6670, 45.6671],
        'longitude': [-111.0447, -111.0448, -111.0449],
        'device_id': ['dev1', 'dev1', 'dev1'],
        'posix_time': [1000, 1300, 1600]
    })


@pytest.fixture
def sample_acc_data():
    return pd.DataFrame({
        'x': [0.1, 0.2, 0.3, -0.1, -0.2],
        'y': [0.2, 0.3, 0.4, -0.2, -0.3],
        'z': [0.3, 0.4, 0.5, -0.3, -0.4],
        'device_id': ['dev1'] * 5,
        'posix_time': [1000, 1060, 1120, 1180, 1240]
    })


@pytest.fixture
def sample_signals():
    """Basic accelerometer signals for testing"""
    return {
        'x': np.array([1.0, 2.0, 3.0, 2.0, 1.0]),
        'y': np.array([2.0, 3.0, 4.0, 3.0, 2.0]),
        'z': np.array([2.0, 3.0, 4.0, 3.0, 2.0])
    }


@pytest.fixture
def basic_feature_config():
    """Simple feature configuration for testing"""
    return create_test_feature_config()


# AccelerometerFeatures Tests
class TestAccelerometerFeatures:
    def test_compute_magnitude(self, sample_signals):
        """Test magnitude computation from XYZ signals"""
        magnitude = AccelerometerFeatures.compute_magnitude(sample_signals)
        expected = np.sqrt(
            sample_signals['x']**2 +
            sample_signals['y']**2 +
            sample_signals['z']**2
        )
        np.testing.assert_array_almost_equal(magnitude, expected)

    def test_compute_basic_stats(self, sample_signals):
        """Test basic statistics computation"""
        stats = AccelerometerFeatures.compute_basic_stats(sample_signals)

        # Test X axis stats
        assert stats['x_mean'] == pytest.approx(1.8)
        assert stats['x_var'] == pytest.approx(0.56)

        # Test Y axis stats
        assert stats['y_mean'] == pytest.approx(2.8)
        assert stats['y_var'] == pytest.approx(0.56)

    def test_compute_zero_crossings(self):
        """Test zero crossing computation"""
        signals = {'x': np.array([1.0, -1.0, 1.0, -1.0])}
        zcr = AccelerometerFeatures.compute_zero_crossings(signals)
        assert zcr['x_zcr'] == pytest.approx(0.375)

    def test_compute_peak_features(self, sample_signals):
        """Test peak-related features computation"""
        peaks = AccelerometerFeatures.compute_peak_features(sample_signals)

        # Check peak-to-peak values
        assert peaks['x_peak_to_peak'] == pytest.approx(2.0)
        assert peaks['y_peak_to_peak'] == pytest.approx(2.0)

        # Check if all expected features are present
        for axis in ['x', 'y', 'z']:
            assert f'{axis}_peak_to_peak' in peaks
            assert f'{axis}_crest_factor' in peaks
            assert f'{axis}_impulse_factor' in peaks

    def test_compute_correlation_features(self, sample_signals):
        """Test correlation features computation"""
        corr = AccelerometerFeatures.compute_correlation_features(
                                sample_signals
                                )

        # Check all correlation pairs
        assert 'xy_corr' in corr
        assert 'yz_corr' in corr
        assert 'xz_corr' in corr

        # Y and Z are identical in our sample, so correlation should be 1
        assert corr['yz_corr'] == pytest.approx(1.0)

    def test_validation_error_handling(self):
        """Test signal validation error handling"""
        # Test NaN validation
        signals_with_nan = {
            'x': np.array([1.0, np.nan]),
            'y': np.array([1.0, 2.0])
        }
        with pytest.raises(FeatureValidationError,
                           match="Signals contain NaN values"):
            AccelerometerFeatures.validate_signals(signals_with_nan)

        # Test different lengths validation
        signals_different_lengths = {
            'x': np.array([1.0, 2.0]),
            'y': np.array([1.0])
        }
        with pytest.raises(FeatureValidationError,
                           match="All signals must have same length"):
            AccelerometerFeatures.validate_signals(signals_different_lengths)


# GPSFeatures Tests
class TestGPSFeatures:
    def test_utm_conversion(self, sample_gps_data):
        """Test UTM coordinate conversion"""
        df = GPSFeatures.add_utm_coordinates(sample_gps_data)

        # Check columns exist
        assert 'utm_easting' in df.columns
        assert 'utm_northing' in df.columns

        # Check values are within expected range for Montana
        assert df['utm_easting'].between(400000, 600000).all()
        assert df['utm_northing'].between(5000000, 5200000).all()


# FeatureComputation Tests
class TestFeatureComputation:
    def test_compute_window_features(self, sample_acc_data,
                                     basic_feature_config):
        """Test feature computation for a window of data"""
        computer = FeatureComputation(basic_feature_config)
        features, stats = computer._compute_window_features(
            sample_acc_data)

        # Check basic stats features
        assert 'x_mean' in features
        assert 'y_mean' in features
        assert 'z_mean' in features

        # Check correlation features
        assert 'xy_corr' in features
        assert 'yz_corr' in features
        assert 'xz_corr' in features

        # Check peak features
        assert 'x_peak_to_peak' in features
        assert 'y_peak_to_peak' in features
        assert 'z_peak_to_peak' in features

        # Check magnitude feature if enabled
        if basic_feature_config.enable_magnitude_features:
            assert 'magnitude_mean' in features

    def test_compute_features(self, sample_acc_data, basic_feature_config):
        """Test the main compute_features method"""
        computer = FeatureComputation(basic_feature_config)
        result_df, stats = computer.compute_features(sample_acc_data)

        # Check result DataFrame
        assert 'device_id' in result_df.columns
        assert 'posix_time' in result_df.columns
        assert len(result_df) > 0

        # Check statistics
        assert 'initial_records' in stats
        assert 'windows' in stats
        assert 'computed_features' in stats
