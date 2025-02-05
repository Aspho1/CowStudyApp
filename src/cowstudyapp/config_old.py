# src/cowstudyapp/config.py
from datetime import datetime
from enum import Enum, auto
import os
from pathlib import Path
import platform
from typing import Optional, Dict, List, Set
from pydantic import BaseModel, DirectoryPath, FilePath, Field, field_validator, ValidationInfo
import pytz
import yaml
from .utils import list_valid_timezones
from dataclasses import dataclass, field

class DataFormat(Enum):
    SINGLE_FILE = "single_file"
    MULTIPLE_FILES = "multiple_files"

class DataValidationConfig(BaseModel):
    # Time bounds
    timezone: str = "America/Denver"
    start_datetime: Optional[datetime] = None
    end_datetime: Optional[datetime] = None
    
    # GPS bounds
    lat_min: float = Field(ge=-90.0, le=90.0)
    lat_max: float = Field(ge=-90.0, le=90.0)
    lon_min: float = Field(ge=-180.0, le=180.0)
    lon_max: float = Field(ge=-180.0, le=180.0)
    
    # Accelerometer bounds
    accel_min: float = Field(ge=-41, le= 41, description="Minimum acceptable acceleration (m/s^2)")
    accel_max: float = Field(ge=-41, le= 41,description="Maximum acceptable acceleration (m/s^2)")
    temp_min: float = Field(description="Minimum acceptable temperature (C)")
    temp_max: float = Field(description="Maximum acceptable temperature (C)")
    

    # Quality filters
    min_satellites: int = Field(ge=0)
    max_dop: float = Field(gt=0)
    
    excluded_devices: Set[int] = Field(
        default_factory=set,
        description="Device IDs to exclude from processing"
    )


    @field_validator('timezone')
    def validate_timezone(cls, v):
        try:
            pytz.timezone(v)
            return v
        except pytz.exceptions.UnknownTimeZoneError:
            list_valid_timezones()
            raise ValueError(f"Unknown timezone: {v}. Please use a timezone from the IANA Time Zone Database.")


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

class ConfigLocator:
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

class FeatureType(str, Enum):
    """Types of features available for computation"""
    BASIC_STATS = "BASIC_STATS"
    ZERO_CROSSINGS = "ZERO_CROSSINGS"
    # SIGNAL_AREA = "SIGNAL_AREA"
    PEAK_FEATURES = "PEAK_FEATURES"
    ENTROPY = "ENTROPY"
    CORRELATION = "CORRELATION"
    SPECTRAL = "SPECTRAL"

@dataclass
class FeatureConfig:
    """Configuration for feature extraction"""
    # What to compute features on
    enable_axis_features: bool = True
    enable_magnitude_features: bool = True
    
    # Feature types to compute
    feature_types: Set[FeatureType] = field(
        default_factory=lambda: {FeatureType.BASIC_STATS}
    )
    
    # Aggregation settings
    gps_sample_interval: int = 300
    acc_sample_interval: int = 60

    gps_sample_rate: float = 1/gps_sample_interval  # One sample per 5 minutes
    acc_sample_rate: float = 1/acc_sample_interval  # One sample per minute
    
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
        
        # Validate sampling rates
        if FeatureType.SPECTRAL in self.feature_types:
            if self.acc_sample_rate <= 0:
                raise ValueError("Accelerometer sample rate must be positive")
            if self.gps_sample_rate <= 0:
                raise ValueError("GPS sample rate must be positive")
            
            # # Check if sampling rate makes sense
            # if self.acc_sample_rate > 1:  # More than once per second
            #     raise ValueError("Accelerometer sampling rate suspiciously high")
            # if self.gps_sample_rate > 1/60:  # More than once per minute
            #     raise ValueError("GPS sampling rate suspiciously high")

class LabelAggTypeType(str, Enum):
    """Types of features available for computation"""
    MODE = "MODE"
    RAW = "RAW"
    PERCENTILE = "PERCENTILE"  

@dataclass
class LabelConfig(BaseModel):
    """
    Reads the configuration for the labels section of the loaded config file
    """
    
    labeled_agg_method: Set[LabelAggTypeType] = field(
        default_factory=lambda: {LabelAggTypeType.RAW}
    )

    def __post_init__(self):
        """Validate configuration after initialization"""
        if len(self.labeled_agg_method) > 1:
            raise ValueError(
                f"Only one aggregation method for `labeled_agg_method` may be selected. "
                f"Currently selected:\n{self.labeled_agg_method}"
            )



class ValidationConfig(BaseModel):
    """Configuration for data validation"""
    timezone: str = "America/Denver"
    start_datetime: str
    end_datetime: str
    lat_min: float = Field(-90.0, ge=-90.0, le=90.0)
    lat_max: float = Field(90.0, ge=-90.0, le=90.0)
    lon_min: float = Field(-180.0, ge=-180.0, le=180.0)
    lon_max: float = Field(180.0, ge=-180.0, le=180.0)
    accel_min: float = Field(-41.0)
    accel_max: float = Field(41.0)
    temp_min: float = Field(-99.0)
    temp_max: float = Field(99.0)
    min_satellites: int = Field(0, ge=0)
    max_dop: float = Field(10.0, gt=0)
    excluded_devices: List[int] = Field(default_factory=list)


class DataSourceConfig(BaseModel):
    format: DataFormat = DataFormat.MULTIPLE_FILES
    gps_directory: Path
    accelerometer_directory: Path
    labeled_data_path: Path
    file_pattern: str = "*.csv"
    gps_sample_interval: int = 300
    excluded_devices: List[int] = Field(default_factory=list)
    validation: DataValidationConfig
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    labels: LabelConfig = Field(default_factory=LabelConfig)
    # labels


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
    



class AppConfig(BaseModel):
    """Main application configuration"""
    format: str = "multiple_files"
    gps_directory: Path
    accelerometer_directory: Path
    file_pattern: str = "*.csv"
    device_id: Optional[int] = None
    validation: ValidationConfig
    features: FeatureConfig

    @staticmethod
    def load(config_path: Optional[Path] = None) -> DataSourceConfig:
        """
        Load configuration from file.
        
        Args:
            config_path: Optional explicit path to config file
            
        Returns:
            Loaded configuration
            
        Raises:
            FileNotFoundError: If no config file found
        """
        if config_path:
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            return AppConfig._load_from_file(config_path)
            
        # Try all possible locations
        for path in ConfigLocator.get_config_paths():
            if path.exists():
                return AppConfig._load_from_file(path)
                
        raise FileNotFoundError("No config file found in any standard location")

    @staticmethod
    def _load_from_file(path: Path) -> DataSourceConfig:
        """Load configuration from specific file"""
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return DataSourceConfig(**config_dict)

    @staticmethod
    def save(config: DataSourceConfig, config_path: Path) -> None:
        """Save configuration to YAML file"""
        config_dict = config.model_dump()
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)