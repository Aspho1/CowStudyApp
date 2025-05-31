from typing import Dict, Any, Set, List
from cowstudyapp.config import (
    FeatureConfig,
    IoConfig,
    FeatureType,
    DataValidationConfig,
    LabelConfig,
    ValidationRule,
    AnalysisConfig,
    VisualsConfig,
    HMMConfig,
    AnalysisFeatureConfig,
    FeatureDistType,
    ConfigManager,
    LabelAggTypeType  # Added this import
)
import tempfile
from pathlib import Path

common = {
    'dataset_name': 'TESTCONFIG',
    'timezone': "America/Denver",
    'gps_sample_interval': 300,
    'accel_sample_interval': 60,
    'excluded_devices': [1,2],
    'random_seed': 123
    }

def create_test_validation_config(**overrides) -> DataValidationConfig:
    """Create a valid DataValidationConfig for testing"""
    defaults = {
        **common,
        'start_datetime': "2022-01-01 00:00:00",
        'end_datetime': "2023-01-01 00:00:00",
        'COVERAGE_THRESHOLD': 70
    }

    config_dict = {**defaults, **overrides}
    return DataValidationConfig(**config_dict)

def create_test_feature_config(**overrides) -> FeatureConfig:
    """Create a valid FeatureConfig for testing with sensible defaults"""
    defaults = {
    **common,
    "enable_axis_features": True,
    "enable_magnitude_features": True,
    "feature_types": {FeatureType.BASIC_STATS, FeatureType.PEAK_FEATURES, FeatureType.ZERO_CROSSINGS, FeatureType.CORRELATION, FeatureType.ENTROPY, FeatureType.SPECTRAL}
    }
    # Override defaults with any provided values
    config_dict = {**defaults, **overrides}
    return FeatureConfig(**config_dict)

def create_test_label_config(**overrides) -> LabelConfig:
    """Create a valid DataValidationConfig for testing"""
    defaults = {
        **common,
        "labeled_agg_method": LabelAggTypeType.RAW
    }
    config_dict = {**defaults, **overrides}
    return LabelConfig(**config_dict)

def create_test_io_config(tmp_path=None, **overrides) -> IoConfig:
    """
    Create a valid IoConfig for testing with temporary directories

    Args:
        tmp_path: Optional pytest tmp_path fixture. If not provided, creates a new temp directory
        overrides: Any config values to override defaults

    Returns:
        IoConfig with proper temporary paths for testing
    """
    # Create a temporary directory if not provided
    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())

    # Create subdirectories
    gps_dir = tmp_path / "gps"
    acc_dir = tmp_path / "accelerometer"
    processed_dir = tmp_path / "processed"
    labeled_dir = tmp_path / "labeled"

    # Create the directories
    gps_dir.mkdir(exist_ok=True, parents=True)
    acc_dir.mkdir(exist_ok=True, parents=True)
    processed_dir.mkdir(exist_ok=True, parents=True)
    labeled_dir.mkdir(exist_ok=True, parents=True)

    # Create an empty labeled data file
    labeled_path = labeled_dir / "gps_observations.csv"
    labeled_path.touch()

    io_defaults = {
        **common,  # Include common config values
        "gps_directory": gps_dir,
        "accelerometer_directory": acc_dir,
        "processed_data_path": processed_dir,
        "labeled_data_path": labeled_path
    }

    # Merge with overrides
    config_dict = {**io_defaults, **overrides}
    return IoConfig(**config_dict)

def create_test_analysis_config(tmp_path=None, **overrides) -> AnalysisConfig:

    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())

    target_dataset = tmp_path / "processed" / 'all_cows_labeled.csv'

    defaults = {
        **common,
        'target_dataset': target_dataset
    }

    config_dict = {**defaults, **overrides}
    return AnalysisConfig(**config_dict)

def create_test_visuals_config(tmp_path=None, **overrides) -> VisualsConfig:

    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())

    # Create subdirectories
    predictions_path = tmp_path / "processed"
    visuals_root_path = tmp_path / "visuals"
    predictions_path.mkdir(exist_ok=True, parents=True)
    visuals_root_path.mkdir(exist_ok=True, parents=True)


    defaults = {
        **common,
        'predictions_path': predictions_path,
        'visuals_root_path': visuals_root_path
    }

    config_dict = {**defaults, **overrides}
    # for k,v in config_dict.items():
    #     print(k, "---->", v)


    return VisualsConfig(**config_dict)

def create_test_config_manager() -> ConfigManager:
    manager = ConfigManager()

    manager.validation = create_test_validation_config()
    manager.features = create_test_feature_config()
    manager.labels = create_test_label_config()
    manager.io = create_test_io_config()
    manager.analysis = create_test_analysis_config()
    # import inspect
    # print("\nClass definition:")
    # print(inspect.getsource(VisualsConfig))
    manager.visuals = create_test_visuals_config()

    return manager
