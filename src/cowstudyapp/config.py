# src/cowstudyapp/config.py
from datetime import datetime
from enum import Enum, auto
import os
from pathlib import Path
import platform
from typing import Optional, Dict, List, Set, Union
from pydantic import BaseModel, DirectoryPath, FilePath, Field, field_validator, ValidationInfo, computed_field
import pytz
import yaml
from .utils import list_valid_timezones
from dataclasses import dataclass, field

##################################### ENUMS #####################################
class DataFormat(Enum):
    '''NOT FULLY IMPLEMENTED -- Options of data input'''
    SINGLE_FILE = "single_file"
    MULTIPLE_FILES = "multiple_files"

class FeatureType(str, Enum):
    """Types of features available for computation"""
    BASIC_STATS = "BASIC_STATS"
    ZERO_CROSSINGS = "ZERO_CROSSINGS"
    PEAK_FEATURES = "PEAK_FEATURES"
    ENTROPY = "ENTROPY"
    CORRELATION = "CORRELATION"
    SPECTRAL = "SPECTRAL"

class LabelAggTypeType(str, Enum):
    """Types of features available for computation"""
    MODE = "MODE"
    RAW = "RAW"
    PERCENTILE = "PERCENTILE"  


##################################### Shared Configuations #####################################

class CommonConfig(BaseModel):
    """
    Configurations shared across multiple sub-configurations
    """
    excluded_devices: List[int] = Field(default_factory=list)
    
    timezone: str = "America/Denver"

    @field_validator('timezone')
    def validate_timezone(cls, v):
        try:
            pytz.timezone(v)
            return v
        except pytz.exceptions.UnknownTimeZoneError:
            list_valid_timezones()
            raise ValueError(f"Unknown timezone: {v}. Please use a timezone from the IANA Time Zone Database.")
        
    # Aggregation settings
    gps_sample_interval: int = Field(default=300)
    acc_sample_interval: int = Field(default=60)

##################################### Configuation file locator #####################################

class ConfigLocator:
    '''
    Returns the location of the highest priority config file
    '''
    @staticmethod
    def get_config_paths() -> list[Path]:
        """Return list of possible config file locations in order of preference"""
        paths = []
        
        # 1. Environment variable if set
        if env_path := os.getenv('COWSTUDYAPP_CONFIG'):
            paths.append(Path(env_path))

        # 2. Project directory (development)
        project_root = Path(__file__).parent.parent.parent
        paths.append(project_root / 'config' / 'default.yaml')
        
        # 3. System-wide location
        if platform.system() == 'Windows':
            paths.append(Path(os.environ['ProgramData']) / 'CowStudyApp' / 'config.yaml')
        else:
            paths.append(Path('/etc/cowstudyapp/config.yaml'))
            
        # 4. User-specific location
        if platform.system() == 'Windows':
            paths.append(Path(os.environ['APPDATA']) / 'CowStudyApp' / 'config.yaml')
        else:
            paths.append(Path.home() / '.config' / 'cowstudyapp' / 'config.yaml')
            
        return paths


##################################### Function Specific Configurations #####################################

class DataValidationConfig(CommonConfig):
    '''
    Configuration options for the validation.py module 
    '''
    # timezone: str = "America/Denver"
    start_datetime: Optional[datetime] = None
    end_datetime: Optional[datetime] = None
    
    # GPS bounds
    lat_min: float = Field(ge=-90.0, le=90.0)
    lat_max: float = Field(ge=-90.0, le=90.0)
    lon_min: float = Field(ge=-180.0, le=180.0)
    lon_max: float = Field(ge=-180.0, le=180.0)
    COVERAGE_THRESHOLD: float = Field(50) 
    

    # Accelerometer bounds
    accel_min: float = Field(ge=-41, le= 41, description="Minimum acceptable acceleration (m/s^2)")
    accel_max: float = Field(ge=-41, le= 41,description="Maximum acceptable acceleration (m/s^2)")
    temp_min: float = Field(description="Minimum acceptable temperature (C)")
    temp_max: float = Field(description="Maximum acceptable temperature (C)")

    # Quality filters
    min_satellites: int = Field(ge=0)
    max_dop: float = Field(gt=0)

    @field_validator('start_datetime', 'end_datetime')
    def localize_datetime(cls, v, values):
        if v is not None:
            if not isinstance(v, datetime):
                raise ValueError(f"Expected datetime, got {type(v)}")
            
            # Get timezone from values if available
            tz = pytz.timezone(values.data.get('timezone', 'America/Denver'))
            
            # Localize the datetime if it's naive
            if v.tzinfo is None:
                v = tz.localize(v)
            return v
        return v

    @field_validator('end_datetime')
    @classmethod
    def end_after_start(cls, v: Optional[datetime], info: ValidationInfo) -> Optional[datetime]:
        if v and info.data.get('start_datetime') and v < info.data['start_datetime']:
            raise ValueError('end_datetime must be after start_datetime')
        return v


class FeatureConfig(CommonConfig):
    """Configuration for features.py"""
    # What to compute features on
    enable_axis_features: bool = Field(False)
    enable_magnitude_features: bool = Field(True)
    
    # Feature types to compute
    feature_types: Set[FeatureType] = field(
        default_factory=lambda: {FeatureType.BASIC_STATS}
    )
    
    def __post_init__(self):
        """Validate configuration"""
        if not (self.enable_axis_features or self.enable_magnitude_features):
            raise ValueError("Must enable either axis or magnitude features")
        if not self.feature_types:
            raise ValueError("Must enable at least one feature type")
        if self.gps_sample_interval <= 0:
            raise ValueError("GPS sample interval must be positive")
        if self.gps_sample_interval <= 0:
            raise ValueError("Accelerometer sample interval must be positive")
        

class LabelConfig(CommonConfig):
    """
    Configuration for labels.py
    """
    
    labeled_agg_method: LabelAggTypeType = Field(
            default=LabelAggTypeType.RAW
        )

# class ValidationConfig(CommonConfig):
#     """Configuration for data validation"""
#     start_datetime: str
#     end_datetime: str
#     lat_min: float = Field(-90.0, ge=-90.0, le=90.0)
#     lat_max: float = Field(90.0, ge=-90.0, le=90.0)
#     lon_min: float = Field(-180.0, ge=-180.0, le=180.0)
#     lon_max: float = Field(180.0, ge=-180.0, le=180.0)
#     accel_min: float = Field(-41.0)
#     accel_max: float = Field(41.0)
#     temp_min: float = Field(-99.0)
#     temp_max: float = Field(99.0)
#     min_satellites: int = Field(0, ge=0)
#     max_dop: float = Field(10.0, gt=0)
    

class IoConfig(CommonConfig):
    format: DataFormat = DataFormat.MULTIPLE_FILES
    gps_directory: Path
    accelerometer_directory: Path
    labeled_data_path: Path
    file_pattern: str = "*.csv"
    # excluded_devices: List[int] = Field(default_factory=list)
    # validation: DataValidationConfig
    # features: FeatureConfig = FeatureConfig
    # labels: LabelConfig = LabelConfig
    # features: FeatureConfig = Field(default_factory=FeatureConfig)
    # labels: LabelConfig = Field(default_factory=LabelConfig)


    @field_validator('labeled_data_path')
    @classmethod
    def validate_file_path(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Labeled data file `{v}` does not exist.")
        return v

    @field_validator('gps_directory', 'accelerometer_directory')
    @classmethod
    def validate_directory(cls, v: Path) -> Path:
        if not v:
            raise ValueError("Directory path must be provided")
        if not v.exists():
            raise ValueError(f"Directory does not exist: {v}")
        if not v.is_dir():
            raise ValueError(f"Path is not a directory: {v}")
        return v

    def validate_config(self) -> None:
        """Validate configuration compatibility"""
        if self.format == DataFormat.MULTIPLE_FILES:
            if not (self.gps_directory and self.accelerometer_directory):
                raise ValueError("Both GPS and accelerometer directories required for multiple_files format")
    

##################################### Configuration Manager #####################################

class ConfigManager:
    """Manages multiple configuration components"""
    def __init__(self):
        self.validation: Optional[DataValidationConfig] = None
        self.features: Optional[FeatureConfig] = None
        self.labels: Optional[LabelConfig] = None
        self.io: Optional[IoConfig] = None

    # @classmethod
    def load_from_file(self, path: Path) -> 'ConfigManager':
        """Load configuration from specific file"""
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        
        # print(config_dict)
        # Load each config component
        if 'validation' in config_dict:
            self.validation = DataValidationConfig(**config_dict.get('validation', {}))
        
        if 'features' in config_dict:
            self.features = FeatureConfig(**config_dict.get('features', {}))
        
        if 'labels' in config_dict:
            self.labels = LabelConfig(**config_dict.get('labels', {}))
        
        if 'io' in config_dict:
            self.io = IoConfig(**config_dict.get('io', {}))

        # Validate cross-component dependencies
        self.validate()
        
        return self

    def validate(self) -> None:
        """Validate cross-component configuration compatibility"""
        # Ensure required components exist
        if not all([self.validation, self.features, self.io]):
            raise ValueError("Missing required configuration components")
        
        # Ensure consistent timezone across components
        timezones = {
            config.timezone 
            for config in [self.validation, self.features, self.labels, self.io] 
            if config is not None
        }
        if len(timezones) > 1:
            raise ValueError(f"Inconsistent timezones across components: {timezones}")
        
        
    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> 'ConfigManager':
        """Load configuration from file or search standard locations"""
        manager = cls()  # Create instance first

        if config_path:
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            return manager.load_from_file(config_path)

        # Try standard locations
        for path in ConfigLocator.get_config_paths():
            if path.exists():
                return manager.load_from_file(path)

        raise FileNotFoundError("No config file found in any standard location")
    

# class AppConfig(BaseModel):
#     """Main application configuration"""
#     format: str = "multiple_files"
#     gps_directory: Path
#     accelerometer_directory: Path
#     file_pattern: str = "*.csv"
#     device_id: Optional[int] = None
#     validation: ValidationConfig
#     features: FeatureConfig

#     @staticmethod
#     def load(config_path: Optional[Path] = None) -> DataSourceConfig:
#         """
#         Load configuration from file.
        
#         Args:
#             config_path: Optional explicit path to config file
            
#         Returns:
#             Loaded configuration
            
#         Raises:
#             FileNotFoundError: If no config file found
#         """
#         if config_path:
#             if not config_path.exists():
#                 raise FileNotFoundError(f"Config file not found: {config_path}")
#             return AppConfig._load_from_file(config_path)
            
#         # Try all possible locations
#         for path in ConfigLocator.get_config_paths():
#             if path.exists():
#                 return AppConfig._load_from_file(path)
                
#         raise FileNotFoundError("No config file found in any standard location")

#     @staticmethod
#     def _load_from_file(path: Path) -> DataSourceConfig:
#         """Load configuration from specific file"""
#         with open(path) as f:
#             config_dict = yaml.safe_load(f)
#         return DataSourceConfig(**config_dict)

#     @staticmethod
#     def save(config: DataSourceConfig, config_path: Path) -> None:
#         """Save configuration to YAML file"""
#         config_dict = config.model_dump()
#         with open(config_path, 'w') as f:
#             yaml.dump(config_dict, f)