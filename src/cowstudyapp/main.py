# src/cowstudyapp/main.py
from pathlib import Path
import argparse
from .config import DataSourceConfig, DataFormat
from .io import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description='Process cow study data')
    parser.add_argument('--gps-dir',
                       type=Path,
                       help='Directory containing GPS files (for multiple_files format)')
    parser.add_argument('--accel-dir',
                       type=Path,
                       help='Directory containing accelerometer files (for multiple_files format)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    config = DataSourceConfig(
        gps_directory=args.gps_dir,
        accelerometer_directory=args.accel_dir
    )
    
    loader = DataLoader(config)
    data = loader.load_and_merge()
    
    # Do something with the data
    print(f"Loaded GPS data shape: {data['gps'].shape}")
    print(f"Loaded accelerometer data shape: {data['accelerometer'].shape}")

if __name__ == "__main__":
    main()