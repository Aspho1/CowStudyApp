# src/cowstudyapp/config.py
from datetime import datetime
from enum import Enum, auto
import os
from pathlib import Path
import platform
from typing import Literal, Optional, Dict, List, Set
from pydantic import BaseModel, DirectoryPath, FilePath, Field, field_validator, ValidationInfo
import pytz
import yaml
from .utils import list_valid_timezones

##################################### ENUMS #####################################

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


class FeatureDistType(str, Enum):
    """Types of feature distributions"""
    REGULAR = "regular"
    CIRCULAR = "circular"
    COVARIATE = "covariate"

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
    dataset_name: str = Field("unnamed")
    random_seed: int = Field(0)

    # Aggregation settings
    gps_sample_interval: int = Field(default=300)
    acc_sample_interval: int = Field(default=60)

    @field_validator('timezone')
    def validate_timezone(cls, v):
        try:
            pytz.timezone(v)
            return v
        except pytz.exceptions.UnknownTimeZoneError:
            list_valid_timezones()
            raise ValueError(f"Unknown timezone: {v}. Please use a timezone from the IANA Time Zone Database.")
        
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

class ValidationRule(BaseModel):
    """Defines validation rules for a single field"""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    is_required: bool = True
    filter_invalid: bool = True  # If False, just report violations without filtering


class DataValidationConfig(CommonConfig):
    '''
    Configuration options for the validation.py module 
    '''
    # timezone: str = "America/Denver"
    start_datetime: Optional[datetime] = None
    end_datetime: Optional[datetime] = None
    COVERAGE_THRESHOLD: float = Field(50) 
    
    validation_rules: Dict[str, ValidationRule] = Field(
        default_factory=lambda: {
            # Default rules if none provided in config
            'latitude': ValidationRule(min_value=-90.0, max_value=90.0),
            'longitude': ValidationRule(min_value=-180.0, max_value=180.0),
            'dop': ValidationRule(max_value=10.0),
            'satellites': ValidationRule(min_value=0),
            'altitude': ValidationRule(min_value=0, max_value=5000, filter_invalid=False),
            'temperature_gps': ValidationRule(min_value=-99, max_value=99),
            'x': ValidationRule(min_value=-41, max_value=41),
            'y': ValidationRule(min_value=-41, max_value=41),
            'z': ValidationRule(min_value=-41, max_value=41),
            'temperature_acc': ValidationRule(min_value=-99, max_value=99)
        }
    )

    @field_validator('validation_rules')
    def validate_rules(cls, v):
        """Ensure all validation rules are properly formatted"""
        if not isinstance(v, dict):
            raise ValueError("validation_rules must be a dictionary")
        
        # Convert any dict-like rules to ValidationRule objects
        validated_rules = {}
        for field, rule in v.items():
            if isinstance(rule, dict):
                validated_rules[field] = ValidationRule(**rule)
            elif isinstance(rule, ValidationRule):
                validated_rules[field] = rule
            else:
                raise ValueError(f"Invalid validation rule for {field}")
        
        return validated_rules
    
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
    enable_axis_features: bool = Field(default=False)
    enable_magnitude_features: bool = Field(default=True)
    feature_types: Set[FeatureType] = Field(
        default_factory=lambda: {FeatureType.BASIC_STATS}
    )

    def model_post_init(self, __context):
        """Validate configuration"""
        if not (self.enable_axis_features or self.enable_magnitude_features):
            raise ValueError("Must enable either axis or magnitude features")
        if not self.feature_types:
            raise ValueError("Must enable at least one feature type")
        if self.gps_sample_interval <= 0:
            raise ValueError("GPS sample interval must be positive")
        if self.acc_sample_interval <= 0:
            raise ValueError("Accelerometer sample interval must be positive")
     

class LabelConfig(CommonConfig):
    """Configuration for labels.py"""
    valid_activities: List[str] = Field(
        default_factory=list,
        description="List of valid activity labels"
    )
    labeled_agg_method: LabelAggTypeType = Field(
        default=LabelAggTypeType.RAW
    )

    
class IoConfig(CommonConfig):
    gps_directory: Path
    accelerometer_directory: Path
    labeled_data_path: Optional[Path] = Field(None, description="Optional path to the labeled data.")
    cow_info_path: Optional[Path] = Field(default=None, description="Path to the meta information about collars and cows")
    tag_to_device: Optional[Dict[str, str]] = Field(default=None, description="Mapping of tag IDs to device IDs")
    label_to_value: Optional[Dict[str, str]] = Field(default=None, description="Mapping of shorthand activity labels to words")
    processed_data_path: Path


    # @field_validator('labeled_data_path')
    # @classmethod
    # def validate_file_path(cls, v: Path) -> Path:
    #     if not v.exists():
    #         raise ValueError(f"Labeled data file `{v}` does not exist.")
    #     return v

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

    

########################## Analysis ##############################################

class AnalysisFeatureConfig(BaseModel):
    name: str
    dist: Optional[str] = Field(None)
    dist_type: FeatureDistType = Field(default=FeatureDistType.REGULAR)

    @field_validator('dist_type')
    def validate_dist_type(cls, v):
        if v not in FeatureDistType:
            raise ValueError(f"Invalid distribution type: {v}. Must be one of {[e.value for e in FeatureDistType]}")
        return v

    @field_validator('dist')
    def validate_dist(cls, v, info: ValidationInfo):
        # Covariates don't need a distribution
        if info.data.get('dist_type') == FeatureDistType.COVARIATE:
            return None
        return v

class HMMConfig(BaseModel):
    enabled: bool = False
    states: List[str]
    features: List[AnalysisFeatureConfig]
    
    time_covariate:bool = Field(False)

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
    training_info_type: str = Field('dataset')
    training_info_path: Optional[str] = None

    @field_validator('training_info_type')
    @classmethod
    def set_type_from_path(cls, v, info):
        # In field_validator, we use info.data instead of values
        if 'training_info_path' in info.data:
            path = info.data['training_info_path']
            if path and path.endswith('.rds'):
                return 'model'
            elif path and path.endswith('.csv'):
                return 'dataset'
        return v

class LSTMConfig(CommonConfig):

    enabled: bool = False
    states: List[str]
    features: List[str]

    max_length: int|str = 20
    max_time_gap: int = 960
    epochs: int = 100

class AnalysisConfig(CommonConfig):
    mode: AnalysisModes = AnalysisModes.LOOCV
    target_dataset: Path
    training_info: Optional[TrainingInfo] = None
    day_only: bool = False
    r_executable: Optional[Path] = None
    hmm: Optional[HMMConfig] = None
    lstm: Optional[LSTMConfig] = None
    output_dir: Path = Field(default=Path("data/analysis_results"))
    # enabled_analyses: List[Literal["hmm"]] = ["hmm"]


    # Not compatible with build_processed_data function
    # @field_validator('target_dataset')
    # @classmethod
    # def validate_data_path(cls, v: Path) -> Path:
    #     if not v.exists():
    #         raise ValueError(f"Data file does not exist: {v}")
    #     return v

    @field_validator('output_dir')
    @classmethod
    def validate_output_dir(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator('training_info')
    @classmethod
    def create_training_info(cls, v):
        if isinstance(v, str):
            return TrainingInfo(training_info_path=v)
        return v

    def validate_config(self) -> None:
        """Validate analysis configuration"""

##################################### Visualizations Config #####################################

class RadarConfig(BaseModel):
    run: bool
    extension: str
    show_night: bool

class TemperatureGraphConfig(BaseModel):
    run: bool
    minimum_required_values: int
    extension: Optional[str] = Field(default="")
    daynight: str
    show_curve: bool = False
    show_table: bool = False
    export_excel: bool = False


    @field_validator('daynight')
    @classmethod
    def validate_daynight_sel(cls, v: str) -> Path:
        if v.lower() not in ['day', 'night', 'both']:
            raise ValueError(f"config.visuals.temerature_graph.daynight"
                             "selection `{v}` does not exist.")
        return v


class CowInfoGraphConfig(BaseModel):
    run: bool
    # implemented: bool
    extension: Optional[str] = Field(default="")

class MoonPhasesConfig(BaseModel):
    run: bool
    # implemented: bool
    extension: Optional[str] = Field(default="")

class DomainGraphConfig(BaseModel):
    run: bool
    # implemented: bool
    labeled_only: bool
    extension: Optional[str] = Field(default="")

class HeatmapConfig(BaseModel):
    run: bool
    filter_weigh_days: bool
    weigh_days: List[str]
    extension: Optional[str] = Field(default="")

class FeatureDists(BaseModel):
    run: bool

class ConvolutionSurface(BaseModel):
    run: bool



class VisualsConfig(CommonConfig):
    predictions_path: Optional[Path] = None
    visuals_root_path: Path
    radar: RadarConfig
    domain: DomainGraphConfig
    feature_dists: FeatureDists
    convolution_surface: ConvolutionSurface
    temperature_graph: TemperatureGraphConfig
    moon_phases: MoonPhasesConfig
    cow_info_graph: CowInfoGraphConfig
    heatmap: HeatmapConfig

    @field_validator('visuals_root_path')
    @classmethod
    def validate_directory(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Path `{v}` does not exist.")
        return v



    @field_validator('predictions_path')
    @classmethod
    def validate_file_path(cls, v: Optional[Path]) -> Optional[Path]:
        if v is not None and not v.exists():
            print(f"Warning: Path `{v}` does not exist. Visuals will not work "
                  "without a correctly defined predictions_path")
        return v
    


    @field_validator('heatmap')
    def convert_weigh_days(cls, v: HeatmapConfig, values):
        if v.weigh_days:
            tz = pytz.timezone(values.data.get('timezone', 'America/Denver'))
            # Convert string dates to datetime objects
            converted_dates = []
            for date_str in v.weigh_days:
                try:
                    # Parse the date string and create a datetime at midnight
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                    # Localize the datetime
                    dt = tz.localize(dt)
                    converted_dates.append(dt)
                except ValueError as e:
                    raise ValueError(f"Invalid date format in weigh_days: {date_str}. Expected format: YYYY-MM-DD")
            v.weigh_days = converted_dates
        return v


##################################### Configuration Manager #####################################

class ConfigManager:
    """Manages multiple configuration components"""
    def __init__(self):
        self.validation: Optional[DataValidationConfig] = None
        self.features: Optional[FeatureConfig] = None
        self.labels: Optional[LabelConfig] = None
        self.io: Optional[IoConfig] = None
        self.analysis: Optional[AnalysisConfig] = None
        self.visuals: Optional[VisualsConfig] = None

    def to_dict(self):
        return {
            'validation': self.validation.__dict__,
            'io': self.io.__dict__, 
            'labels': self.labels.__dict__,
            'features': self.features.__dict__,
            'analysis': self.analysis.__dict__,
            'visuals': self.visuals.__dict__
        }

    # @classmethod
    def load_from_file(self, path: Path) -> 'ConfigManager':
        """Load configuration from specific file"""
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        
        # Get common settings that will be inherited
        common_config = config_dict.get('common', {})
        
        # For each config section, merge common settings with section-specific settings
        if 'validation' in config_dict:
            merged_validation = {
                **common_config,  # Base settings from common
                **config_dict['validation']  # Section-specific settings (will override common if duplicated)
            }
            self.validation = DataValidationConfig(**merged_validation)
        
        if 'features' in config_dict:
            merged_features = {**common_config, **config_dict['features']}
            self.features = FeatureConfig(**merged_features)
        
        if 'labels' in config_dict:
            merged_labels = {**common_config, **config_dict['labels']}
            self.labels = LabelConfig(**merged_labels)
        
        if 'io' in config_dict:
            merged_io = {**common_config, **config_dict['io']}
            self.io = IoConfig(**merged_io)

        if 'analysis' in config_dict:
            merged_analysis = {**common_config, **config_dict['analysis']}
            self.analysis = AnalysisConfig(**merged_analysis)

        if 'visuals' in config_dict:
            merged_visual = {**common_config, **config_dict['visuals']}
            self.visuals = VisualsConfig(**merged_visual)
        
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
    











