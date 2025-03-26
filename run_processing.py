#!/usr/bin/env python3
"""
Simple wrapper script to run the CowStudyApp processing with progress reporting.
Used by the web interface to process data.
"""
import sys
import os
import time
from pathlib import Path
import json

# Create a progress file for tracking
def setup_progress_tracking(task_id):
    """Setup progress tracking for a task."""
    temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    progress_file = os.path.join(temp_dir, f'progress_{task_id}.json')
    
    with open(progress_file, 'w') as f:
        json.dump({
            'status': 'starting',
            'progress': 10,
            'message': 'Initializing processing...',
            'timestamp': time.time()
        }, f)
    
    return progress_file

def update_progress(progress_file, progress, message, status='running'):
    """Update the progress for a task."""
    with open(progress_file, 'w') as f:
        json.dump({
            'status': status,
            'progress': progress,
            'message': message,
            'timestamp': time.time()
        }, f)

# Get task ID from command line args
task_id = sys.argv[1] if len(sys.argv) > 2 else 'unknown'
progress_file = setup_progress_tracking(task_id)

# Get the absolute path to the CowStudyApp directory
cowstudyapp_dir = os.path.dirname(os.path.abspath(__file__))

# Add the CowStudyApp directory to the Python path
# This makes both 'src' and 'cowstudyapp' available as top-level modules
sys.path.insert(0, cowstudyapp_dir)

update_progress(progress_file, 20, f"Reading configuration from {sys.argv[1]}")

try:
    # Try importing the module
    update_progress(progress_file, 30, "Loading processing modules...")
    from src.build_processed_data import process_sensor_data
    
    if len(sys.argv) < 2:
        update_progress(progress_file, 0, "Error: Missing config path argument", status='failed')
        print("Usage: python run_processing.py <task_id> <config_path>")
        sys.exit(1)
        
    config_path = Path(sys.argv[-1])
    if not config_path.exists():
        update_progress(progress_file, 0, f"Error: Config file {config_path} does not exist", status='failed')
        print(f"Config file {config_path} does not exist")
        sys.exit(1)
        
    update_progress(progress_file, 40, f"Starting data processing with config: {config_path}")
    print(f"Processing data with config: {config_path}")
    
    update_progress(progress_file, 50, "Processing data...")
    features_df = process_sensor_data(config_path=config_path)
    
    update_progress(progress_file, 95, f"Processed {len(features_df)} records successfully", status='completed')
    print(f"Processed {len(features_df)} records")
    
except Exception as e:
    import traceback
    error_msg = f"Error: {str(e)}"
    update_progress(progress_file, 0, error_msg, status='failed')
    print(error_msg)
    print(traceback.format_exc())
    sys.exit(1)







# #!/usr/bin/env python3
# import sys
# from pathlib import Path
# from src.build_processed_data import process_sensor_data

# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python run_processing.py <config_path>")
#         sys.exit(1)
    
#     config_path = Path(sys.argv[1])
#     features_df = process_sensor_data(config_path=config_path)
#     print(f"Processed {len(features_df)} records")

