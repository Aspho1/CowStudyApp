from pathlib import Path
from cowstudyapp.dataset_building.io import DataLoader

# from cowstudyapp.features import aggregate_accelerometer_data
# from cowstudyapp.config import DataSourceConfig
from cowstudyapp.utils import from_posix
from cowstudyapp.config import ConfigManager

# from cowstudyapp.config import load_config
from cowstudyapp.dataset_building.merge import DataMerger
from typing import Optional


def process_sensor_data(config_path: Optional[Path] = None, aggregated: bool = True):
    """
    Main processing pipeline to create feature dataset
    Use aggregated = False to compute the dataset without feature aggregation and calculation (for Hannah)
    """
    config = ConfigManager.load(config_path)

    # Load raw data
    loader = DataLoader(config)
    data = loader.load_data(aggregated)

    for key, df in data.items():
        print(f"\n{key.upper()} Data:")
        print(f"Total records: {len(df)}")
        print(
            f"Time range: {from_posix(df['posix_time'].min())} to {from_posix(df['posix_time'].max())}"
        )
        print(f"Unique devices: {df['device_id'].unique()}")

    # Process accelerometer data
    merger = DataMerger()
    merged_df = merger.merge_sensor_data(data, aggregated)

    # Save results
    output_path = config.io.processed_data_path
    output_path.parent.mkdir(exist_ok=True)
    merged_df.to_csv(output_path, index=False)

    return merged_df


if __name__ == "__main__":
    # config_path = Path("config/RB_22_config.yaml")
    config_path = Path("config/RB_19_config.yaml")
    # config_path = Path("config/default.yaml")
    features_df = process_sensor_data(config_path=config_path)
    print(f"Processed {len(features_df)} records")
