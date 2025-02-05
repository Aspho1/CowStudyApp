from dataclasses import dataclass
from typing import List, Set, Dict, Any
from pathlib import Path
from pydantic import Field
import yaml
from enum import Enum

class ConfigReference:
    """Handle references to other config sections"""
    def __init__(self, path: str):
        self.path = path.split('.')

    def resolve(self, config: Dict) -> Any:
        current = config
        for key in self.path:
            current = current[key]
        return current

class YAMLLoader(yaml.SafeLoader):
    """Custom YAML loader with !ref tag support"""
    pass

def ref_constructor(loader, node):
    value = loader.construct_scalar(node)
    return ConfigReference(value)

YAMLLoader.add_constructor('!ref', ref_constructor)

@dataclass
class CommonConfig:
    """Shared configuration settings"""
    excluded_devices: List[int]
    intervals: Dict[str, int]
    timezone: str

@dataclass
class ValidationConfig:
    """Validation settings"""
    timezone: str = "America/Denver"
    start_datetime: str = "2022-01-15 00:00:00"
    end_datetime: str = "2022-03-22 23:59:59"
    lat_min: float = Field(-90.0, ge=-90.0, le=90.0)
    lat_max: float = Field(90.0, ge=-90.0, le=90.0)
    lon_min: float = Field(-180.0, ge=-180.0, le=180.0)
    lon_max: float = Field(180.0, ge=-180.0, le=180.0)
    accel_min: float = Field(-41.0)
    accel_max: float = Field(41.0)
    temp_min: float = Field(-99.0)
    temp_max: float = Field(99.0)
    min_satellites: int = Field(0, ge=0)
    max_dop: float = Field(99.0, gt=0)
    excluded_devices: List[int] = Field(default_factory=list)


@dataclass
class FeatureConfig:
    """Feature computation settings"""
    enable_axis_features: bool
    enable_magnitude_features: bool
    feature_types: Set[str]
    gps_sample_interval: int
    acc_sample_interval: int

class ConfigManager:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.raw_config = self._load_raw_config()
        self.common = self._init_common_config()
        self._resolve_references()

    def _load_raw_config(self) -> Dict:
        """Load raw YAML configuration"""
        with open(self.config_path) as f:
            return yaml.load(f, Loader=YAMLLoader)

    def _init_common_config(self) -> CommonConfig:
        """Initialize common configuration"""
        common_data = self.raw_config['common']
        return CommonConfig(
            excluded_devices=common_data['excluded_devices'],
            intervals=common_data['intervals'],
            timezone=common_data['timezone']
        )

    def _resolve_references(self):
        """Resolve all !ref tags in the configuration"""
        def resolve_dict(d: Dict) -> Dict:
            for key, value in d.items():
                if isinstance(value, ConfigReference):
                    d[key] = value.resolve(self.raw_config)
                elif isinstance(value, dict):
                    resolve_dict(value)
                elif isinstance(value, list):
                    resolve_list(value)
            return d

        def resolve_list(l: List) -> List:
            for i, value in enumerate(l):
                if isinstance(value, ConfigReference):
                    l[i] = value.resolve(self.raw_config)
                elif isinstance(value, dict):
                    resolve_dict(value)
                elif isinstance(value, list):
                    resolve_list(value)
            return l

        resolve_dict(self.raw_config)

    def get_validation_config(self) -> ValidationConfig:
        """Get validation configuration"""
        return ValidationConfig(**self.raw_config['validation'])

    def get_feature_config(self) -> FeatureConfig:
        """Get feature computation configuration"""
        return FeatureConfig(**self.raw_config['features'])

# Usage example
def load_config(config_path: Path = Path("config/default.yaml")):
    config_manager = ConfigManager(config_path)
    return config_manager
