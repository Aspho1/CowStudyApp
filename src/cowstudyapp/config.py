# src/cowstudyapp/config.py
from datetime import datetime
from enum import Enum, auto
import os
from pathlib import Path
import platform
from typing import Literal, Optional, Dict, List, Set, Union
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

class DistributionTypes(str, Enum):
    """Types of distributions to fit"""
    LOGNORMAL = "LOGNORMAL"
    GAMMA = "GAMMA"
    WEIBULL = "WEIBULL" # DOES NOT WORK
    NORMAL = "NORMAL"
    EXPONENTIAL = "EXPONENTIAL" 
    VONMISES = "VONMISES"
    WRAPPEDCAUCHY = "WRAPPEDCAUCHY"

class AnalysisModes(str, Enum):
    """Types of distributions to fit"""
    PRODUCT = "PRODUCT"
    LOOCV = "LOOCV"

    

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

    
class IoConfig(CommonConfig):
    format: DataFormat = DataFormat.MULTIPLE_FILES
    gps_directory: Path
    accelerometer_directory: Path
    labeled_data_path: Path
    file_pattern: str = "*.csv"
    tag_to_device: Optional[Dict[str, str]] = Field(default=None, description="Mapping of tag IDs to device IDs")
    label_to_value: Optional[Dict[str, str]] = Field(default=None, description="Mapping of shorthand activity labels to words")
    processed_data_path: Path


    @field_validator('labeled_data_path')
    @classmethod
    def validate_file_path(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Labeled data file `{v}` does not exist.")
        return v

    @field_validator('tag_to_device')
    @classmethod
    def validate_tag_to_device(cls, v: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
        if v is not None:
            # Ensure all keys and values are strings
            validated = {}
            for tag, device in v.items():
                # Convert both to strings and strip any whitespace
                tag_str = str(tag).strip()
                device_str = str(device).strip()
                validated[tag_str] = device_str
            return validated
        return None

    @field_validator('label_to_value')
    @classmethod
    def validate_label_to_value(cls, v: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
        if v is not None:
            # Ensure all keys and values are strings
            validated = {}
            for label, value in v.items():
                # Convert both to strings and strip any whitespace
                label_str = str(label).strip()
                value_str = str(value).strip()
                validated[label_str] = value_str
            return validated
        return None

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
    

########################## Analysis ##############################################

class HMMFeatureConfig(BaseModel):
    name: str
    dist: Optional[str] = Field(None)
    dist_type: Optional[str] = Field("regular")


class HMMConfig(BaseModel):
    enabled: bool = True
    states: List[str]
    features: List[HMMFeatureConfig]
    options: dict = Field(default_factory=lambda: {
        "show_dist_plots": True,
        "remove_outliers": True,
        "show_full_range": True,
        "show_correlation": True,
        "number_of_retry_fits": 1,
        "distributions": {
            "regular": [
                DistributionTypes.LOGNORMAL.value,
                DistributionTypes.GAMMA.value,
                DistributionTypes.WEIBULL.value,
                DistributionTypes.NORMAL.value,
                DistributionTypes.EXPONENTIAL.value
            ],
            "circular": [
                DistributionTypes.VONMISES.value,
                DistributionTypes.WRAPPEDCAUCHY.value
            ]
        }
    })

    @field_validator('options')
    def validate_distributions(cls, v):
        if 'distributions' not in v:
            v['distributions'] = {
                'regular': [dist.value for dist in DistributionTypes if dist != DistributionTypes.VONMISES],
                'circular': [DistributionTypes.VONMISES.value]
            }
        else:
            # Convert string values to enum if they aren't already
            if 'regular' in v['distributions']:
                v['distributions']['regular'] = [
                    dist if isinstance(dist, DistributionTypes) else DistributionTypes(dist)
                    for dist in v['distributions']['regular']
                ]
            if 'circular' in v['distributions']:
                v['distributions']['circular'] = [
                    dist if isinstance(dist, DistributionTypes) else DistributionTypes(dist)
                    for dist in v['distributions']['circular']
                ]
        return v

class TrainingInfo(BaseModel):
    training_info_type : str = Field('dataset')
    training_info_path : Optional[str] = None


class AnalysisConfig(CommonConfig):
    mode: AnalysisModes = AnalysisModes.LOOCV
    target_dataset: Path

    training_info:Optional[TrainingInfo] = None

    r_executable: Optional[Path] = None
    hmm: Optional[HMMConfig] = None
    output_dir: Path = Field(default=Path("data/analysis_results"))

    enabled_analyses: List[Literal["hmm"]] = ["hmm"]

    @field_validator('target_dataset')
    @classmethod
    def validate_data_path(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Data file does not exist: {v}")
        return v

    @field_validator('output_dir')
    @classmethod
    def validate_output_dir(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    def validate_config(self) -> None:
        """Validate analysis configuration"""




##################################### Configuration Manager #####################################

class ConfigManager:
    """Manages multiple configuration components"""
    def __init__(self):
        self.validation: Optional[DataValidationConfig] = None
        self.features: Optional[FeatureConfig] = None
        self.labels: Optional[LabelConfig] = None
        self.io: Optional[IoConfig] = None
        self.analysis: Optional[AnalysisConfig] = None  # Add analysis config

    def to_dict(self):
        return {
            'validation': self.validation.__dict__,
            'io': self.io.__dict__, 
            'labels': self.labels.__dict__,
            'features': self.features.__dict__
        }

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

        if 'analysis' in config_dict:
            self.analysis = AnalysisConfig(**config_dict.get('analysis', {}))
        # Validate cross-component dependencies
        # self.validate()
        
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
    











