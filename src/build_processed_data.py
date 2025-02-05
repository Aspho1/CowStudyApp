from pathlib import Path
from cowstudyapp.io import DataLoader
# from cowstudyapp.features import aggregate_accelerometer_data
# from cowstudyapp.config import DataSourceConfig
from cowstudyapp.utils import from_posix
from cowstudyapp.config_old import AppConfig
from cowstudyapp.config import load_config
from cowstudyapp.merge import DataMerger

def process_sensor_data(config_path: Path = None):
    """
    Main processing pipeline to create feature dataset
    """
    config_path = Path("config/default_old.yaml")   
    config = AppConfig.load(config_path)
    # config = load_config(config_path)

    # Load raw data
    loader = DataLoader(config)
    data = loader.load_data()
    
    print(data['label'].head())

    for key, df in data.items():
        print(f"\n{key.upper()} Data:")
        print(f"Total records: {len(df)}")
        print(f"Time range: {from_posix(df['posix_time'].min())} to {from_posix(df['posix_time'].max())}")
        print(f"Unique devices: {df['device_id'].unique()}")

    # Process accelerometer data
    merger = DataMerger()
    merged_df = merger.merge_sensor_data(data)
    
    # Save results
    output_path = Path("data/processed/RB_22/unlabeled_all_cows.csv")
    output_path.parent.mkdir(exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    
    return merged_df

if __name__ == "__main__":
    features_df = process_sensor_data()
    print(f"Processed {len(features_df)} feature windows")