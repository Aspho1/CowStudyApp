# scripts/test_merge.py
from pathlib import Path
import pandas as pd
from cowstudyapp.config import AppConfig
from cowstudyapp.io import DataLoader
from cowstudyapp.merge import DataMerger
from cowstudyapp.utils import from_posix

def main():
    # Load config
    config_path = Path("config/default.yaml")
    config = AppConfig.load(config_path)
    
    # Initialize loader
    loader = DataLoader(config)
    
    # Load data
    print("Loading data...")
    data = loader.load_data()
    
    # Print basic info about loaded data
    for key, df in data.items():
        print(f"\n{key.upper()} Data:")
        print(f"Total records: {len(df)}")
        print(f"Time range: {from_posix(df['posix_time'].min())} to {from_posix(df['posix_time'].max())}")
        print(f"Unique devices: {df['device_id'].unique()}")
    
    # Filter for specific cow
    for key in data:
        data[key] = data[key][data[key]['device_id'] == 824]
        print(f"\nFiltered {key} data for cow 824:")
        print(f"Records: {len(data[key])}")
    
    # Merge data
    print("\nMerging data...")
    merger = DataMerger()
    merged_df = merger.merge_sensor_data(data)
    
    print("\nMerged Data Summary:")
    print(f"Total records: {len(merged_df)}")
    print(f"Time range: {from_posix(merged_df['posix_time'].min())} to {from_posix(merged_df['posix_time'].max())}")
    print("\nSample of merged data:")
    print(merged_df.head())
    
    # Save sample of merged data
    output_dir = Path("data/processed/RB_22")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    sample_output = merged_df.head(1000)
    sample_output.to_csv(output_dir / "cow_824_sample.csv", index=False)
    print(f"\nSaved sample of 1000 records to {output_dir/'cow_824_sample.csv'}")

if __name__ == "__main__":
    main()