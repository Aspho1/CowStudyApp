# from pathlib import Path
# from cowstudyapp.dataset_building.io import DataLoader

# import argparse

# from cowstudyapp.utils import from_posix
# from cowstudyapp.config import ConfigManager

# from cowstudyapp.dataset_building.merge import DataMerger
# from typing import Optional
# import logging
# import sys

# logging.basicConfig(
#     level=logging.INFO,
#     format='[%(asctime)s] %(levelname)s %(message)s',
#     stream=sys.stdout
# )
# logger = logging.getLogger(__name__)

# def process_sensor_data(config_path: Optional[Path] = None, aggregated: bool = True):
#     """
#     Main processing pipeline to create feature dataset
#     Use aggregated = False to compute the dataset without feature aggregation and calculation (for Hannah)
#     """
#     logger.info("Loading config from %s", config_path)
#     config = ConfigManager.load(config_path)


#     print("HERE!!!!!!!!!!!!!!!!!!!!!!!!!")
#     # Load raw data
#     logger.info("Loading raw data…")
#     loader = DataLoader(config)
#     data = loader.load_data(aggregated)
#     logger.info("Got raw datasets: %s", ", ".join(data.keys()))

#     # for key, df in data.items():
#     #     print(f"\n{key.upper()} Data:")
#     #     print(f"Total records: {len(df)}")
#     #     print(
#     #         f"Time range: {from_posix(df['posix_time'].min())} to {from_posix(df['posix_time'].max())}"
#     #     )
#     #     print(f"Unique devices: {df['device_id'].unique()}")

#     # Process accelerometer data
#     merger = DataMerger()
#     merged_df = merger.merge_sensor_data(data, aggregated)

#     # Save results
#     output_path = config.io.processed_data_path
#     output_path.parent.mkdir(exist_ok=True)
#     logger.info("Saving results to %s", output_path)

#     merged_df.to_csv(output_path, index=False)

#     return merged_df


# # if __name__ == "__main__":
# #     # config_path = Path("config/RB_22_config.yaml")
# #     config_path = Path("config/RB_19_config.yaml")
# #     # config_path = Path("config/default.yaml")
# #     features_df = process_sensor_data(config_path=config_path)
# #     print(f"Processed {len(features_df)} records")



# def main():
#     logger.info("Starting build_processed_data with args: %s", sys.argv[1:])

#     parser = argparse.ArgumentParser(
#         prog="build_processed_data",
#         description="Process raw sensor data into feature DataFrame"
#     )
#     parser.add_argument(
#         "task_id",
#         help="ID of this processing task (for logging, etc.)"
#     )
#     parser.add_argument(
#         "config_path",
#         type=Path,
#         help="YAML file with processing configuration"
#     )
#     args = parser.parse_args()

#     try:
#         features_df = process_sensor_data(config_path=args.config_path)
#         logger.info("[%s] Processed %d records", args.task_id, len(features_df))
#     except Exception:
#         logger.exception("Error while processing task %s", args.task_id)
#         sys.exit(1)

# if __name__ == "__main__":
#     main()




from pathlib import Path
from cowstudyapp.dataset_building.io import DataLoader

import argparse

from cowstudyapp.utils import from_posix
from cowstudyapp.config import ConfigManager

from cowstudyapp.dataset_building.merge import DataMerger
from typing import Optional, Callable
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def process_sensor_data(config_path: Optional[Path] = None, aggregated: bool = True, 
                       progress_callback: Optional[Callable[[int, str], None]] = None):
    """
    Main processing pipeline to create feature dataset
    Use aggregated = False to compute the dataset without feature aggregation and calculation (for Hannah)
    
    Args:
        config_path: Path to the YAML configuration file
        aggregated: Whether to aggregate and compute features
        progress_callback: Optional function to report progress (percent, message)
    
    Returns:
        DataFrame containing the processed data
    """
    if progress_callback:
        progress_callback(0, f"Loading config from {config_path}")
    logger.info("Loading config from %s", config_path)
    
    try:
        config = ConfigManager.load(config_path)
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        if progress_callback:
            progress_callback(0, f"Failed to load config: {str(e)}")
        raise

    # Load raw data
    if progress_callback:
        progress_callback(10, "Loading raw data...")
    logger.info("Loading raw data...")
    
    try:
        loader = DataLoader(config)
        data = loader.load_data(aggregated)
        logger.info("Got raw datasets: %s", ", ".join(data.keys()))
        
        if progress_callback:
            progress_callback(30, f"Loaded raw datasets: {', '.join(data.keys())}")
    except Exception as e:
        logger.error(f"Failed to load raw data: {str(e)}")
        if progress_callback:
            progress_callback(10, f"Failed to load raw data: {str(e)}")
        raise

    # Process and merge accelerometer data
    if progress_callback:
        progress_callback(40, "Merging sensor data...")
    logger.info("Merging sensor data...")
    
    try:
        merger = DataMerger()
        merged_df = merger.merge_sensor_data(data, aggregated)
        
        if progress_callback:
            progress_callback(70, f"Merged sensor data: {len(merged_df)} records")
    except Exception as e:
        logger.error(f"Failed to merge sensor data: {str(e)}")
        if progress_callback:
            progress_callback(40, f"Failed to merge sensor data: {str(e)}")
        raise

    # Save results
    if progress_callback:
        progress_callback(80, f"Saving results to {config.io.processed_data_path}")
    
    try:
        output_path = config.io.processed_data_path
        output_path.parent.mkdir(exist_ok=True)
        logger.info("Saving results to %s", output_path)
        
        merged_df.to_csv(output_path, index=False)
        
        if progress_callback:
            progress_callback(100, f"Successfully saved {len(merged_df)} records to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
        if progress_callback:
            progress_callback(80, f"Failed to save results: {str(e)}")
        raise

    return merged_df


def main(config_path=None, progress_callback=None):
    """
    Main entry point that can be called directly from Python or via command line
    
    Args:
        config_path: Path to the YAML configuration file (string or Path object)
        progress_callback: Optional function to report progress (percent, message)
    
    Returns:
        True if processing succeeded, False otherwise
    """
    logger.info("Starting build_processed_data")
    
    # If called from command line and config_path is None, parse arguments
    if config_path is None and len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            prog="build_processed_data",
            description="Process raw sensor data into feature DataFrame"
        )
        parser.add_argument(
            "task_id",
            help="ID of this processing task (for logging, etc.)"
        )
        parser.add_argument(
            "config_path",
            type=Path,
            help="YAML file with processing configuration"
        )
        args = parser.parse_args()
        config_path = args.config_path
        task_id = args.task_id
    else:
        if not config_path:
            config_path: Path = Path("config/RB_22_config.yaml")
        # When called programmatically
        task_id = "direct-call"
        if isinstance(config_path, str):
            config_path = Path(config_path)

    try:
        features_df = process_sensor_data(
            config_path=config_path, 
            progress_callback=progress_callback
        )
        logger.info("[%s] Processed %d records", task_id, len(features_df))
        return True
    except Exception as e:
        logger.exception("Error while processing task %s: %s", task_id, str(e))
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)

