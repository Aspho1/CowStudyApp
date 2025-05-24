# CowStudyApp

A Python package for processing and analyzing sensor data from cattle-mounted GPS and accelerometer devices, developed for Montana State University's Ranch Sciences and Industrial Engineering Departments.

## Overview
This package processes GPS data, accelerometer data, and labeled data, combining them into a single dataset. It then performs sequential classification of activities using either Hidden Markov Models or Long Short-Term Memory Recurrent Neural Networks. These models produce activity predictions and quality estimations. The package also generates visualizations to provide deeper insights into the data and analysis results.


## Installation

### Development Installation
```bash
# Clone the repository
git clone https://github.com/Aspho1/CowStudyApp.git
cd CowStudyApp

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development version
pip install -e ".[dev]"
```

### Requirements
- Python ≥ 3.8
- R ≥ 4.4.1 (for HMM analysis)
- See `pyproject.toml` for full dependency list

## Project Structure
The package is organized into three main components:

1. Dataset Building - Process and combine raw sensor data (`build_processed_data.py`)
2. Analysis - Apply HMM or LSTM models to classify activities (`run_analysis.py`)
3. Visualization - Generate plots and visual analysis (`make_visuals.py`)
   
All main modules are located in the `src/cowstudyapp` directory.


## Usage

The package provides three main entry points, each configurable via YAML config files.

### Processing data

Place the GPS, Accelerometer, and Label data into their respective directories (defined in the YAML config as `io.gps_directory`, `io.accelerometer_directory`, and `io.labeled_data_path`). When the program is run, two outputs will be placed in `data/processed/DATASET_NAME/`; the `predictions.csv` file and a `.json` summary of preprocessing statistics which communicates information about gaps in data, invalid values, and more data quality issues. 

The config can be set in two ways: 
1. Pass in a `config_path` argument to the call
```bash
python -m src/cowstudyapp/build_processed_data.py config_path config/RB_22_config.yaml
```
2. Make sure the correct config is selected in `src/build_processed_data.py`, then run the script with no arguments.

```python
def main(config_path=None, progress_callback=None):
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

            ################ Set the config here ###############
            config_path: Path = Path("config/RB_22_config.yaml") 
            ####################################################
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
```

#### Comparing datasets

Additionally, there is a script to compare two processed datasets `src/compare_descriptive_stats.py`. 

for instance, to compare RB_19 and RB_22:
```python
if __name__ == "__main__":
    with open('data/processed/RB_19/all_cows_labeled.csv_dqr.json', 'r') as f1:
        RB19_Stats = json.load(f1)
    with open('data/processed/RB_22/all_cows_labeled.csv_dqr.json', 'r') as f2:
        RB22_Stats = json.load(f2)
    
    comp = compare_quality_reports(RB19_Stats, RB22_Stats)
    print_significant_differences(comp)
```


### Running analysis

Make sure the YAML config file (specifically the `analysis` section of it) is set up as desired. At a minimum, make sure the `mode`, `target_dataset`, and optionally `training_info_path` are correct. See the [Config section](##cfg-heading-id) below for more information on the config. The file `src/run_analysis.py` can be called in the same ways as `src/cowstudyapp/build_processed_data.py`. This will create outputs in `data/cv_results`, `data/models`, and or `data/predictions` depending on the config settings. 
```bash
python -m src/cowstudyapp/run_analysis.py config_path config/RB_22_config.yaml
```
### Creating visuals

Visualizations can be turned on and off in the `visuals` section of the config. Point the `visuals.predictions_path` to a `predictions.csv` output from analysis code and run `src/make_visuals.py` to create visualizations.
```bash
python -m src/cowstudyapp/make_visuals.py config_path config/RB_22_config.yaml
```

## Development

### Running Tests

The tests are limited in scope but can debug some basic problems with the modules.

```bash
pytest tests/
```

### Code Quality
```bash
# Format code
black .
isort .

# Type checking
mypy src/cowstudyapp

# Linting
flake8 .
```


## Config and Running the code {##cfg-heading-id}

### Sections
There are three primary sections to this package; processing data, running the analyses, and making visualizations. All three of these sections can be modified through a config file located in `config/`. This config file lets the user make impactful changes to the code without scaling through the entire codebase themselves. 

#### Common
The variables defined in common are available in *all* other config sections. 
Here are the valid fields and some sample values:
```yaml
common:
  dataset_name: "RB_22"             # A prefix to attach to directories and files
  timezone: "America/Denver"        # A timezone to use for timezone aware analysis
  gps_sample_interval: 300          # Expected seconds between GPS
  accel_sample_interval: 60         # Expected seconds between Accelerometer 
  excluded_devices: [841, 1005]     # A list of devices to fully ignore
  random_seed: 123                  # A fixed random seed for reproducibility
```

#### Processing Data

The `io`, `validation`, `features`, and `labels` sections of the config directly relate to `src/cowstudyapp/build_processed_data.py`. These next sections will explain how the user can interact with the related sections of the config. 

##### IO

Variables which are explicitly used for setting input and output parameters of the process data section. 

```yaml
io:
  # The folder holding the .csv GPS files
  gps_directory: "data/raw/RB_22/gps"

  # The folder holding the .csv Accelerometer files
  accelerometer_directory: "data/raw/RB_22/accelerometer"

  # The file (.xlsx or .csv) which holds the labeled activity data
  labeled_data_path: "data/raw/RB_22/labeled/gps_observations_2022.csv"

  # The output location for the processed data file
  processed_data_path: "data/processed/RB_22/processed_data.csv"

  # OPTIONAL: If formatted correctly, a file with animal weight, BCS, and more 
  # metrics can be passed in. This is primarily used for some of the visualizations. 
  cow_info_path: "data/raw/RB_22/cow_info/Aggregate_collar_info.xlsx"
  
  # OPTIONAL: a YAML dictionary defining cow_ids to collar_ids
  tag_to_device:
    '6082': '609'
    ...
    '0024': '607'
    '0011': '615'

  # OPTIONAL: a YAML dictionary defining labels in the labeled_data_path to labels
  # being used in the data processing steps. Keys are the values which appear in 
  # the label data file, values are the transformed values which will appear in the output.
  label_to_value:
    't': 'Traveling'    
    ...
    'd': 'Drinking'
    'f': 'Fighting'
```


##### Validation

Parameters which bind the values in the input output process. Each field in `valdation_rules` has a name which corresponds to a column in the input data. Within each field, the user can set minimum and maximum expected values for the data. Additionally, the user can state whether the column is required (or if an error should be thrown if the column doesn't exist), and if the values out of bounds should be filtered or not. If `filter_invalid` is true, the log will report filtered values and they will be removed. if `filter_invalid` is false, the out of bounds values will be reported but not filtered.

```yaml
validation:
  start_datetime: "2022-01-15 00:00:00" # The start of the study period  
  end_datetime: "2022-03-23 00:00:00"   # The end of the study period
  
  # Minimum ratio of (actual / expected) GPS data needed to include the device
  COVERAGE_THRESHOLD: 70                

  validation_rules:
    #### GPS Feature Validation Rules ####
    latitude:
      min_value: -90.0
      max_value: 90.0
      is_required: true
      filter_invalid: false

    longitude:
      min_value: -180.0
      max_value: 180.0
      is_required: true
      filter_invalid: false
    ...

    #### Accelerometer Feature Validation Rules ####
    x:
      min_value: -41
      max_value: 41
      is_required: true
      filter_invalid: false
    ...

    temperature_acc:
      min_value: -99
      max_value: 99
      is_required: true
      filter_invalid: false
```



##### Features

Declare what types of features should be computed from the raw accelerometer data. `enable_axis_features` and `enable_magnitude_features` say which baseline values the features should be computed on, and the optional values inside of `feature_values` declare which types of features should be calculated on each of the baseline features. In the example config below, mean and variance would be computed for x, y, z, and magnitude.   

```yaml
features:
  enable_axis_features: true        # Compute features for each axes
  enable_magnitude_features: true   # Compute features for the magnitudes
  feature_types:
    - "BASIC_STATS"                 # Mean, variance of the window
    # - "ZERO_CROSSINGS"              # Count of transitions from - to + or + to -
    # - "PEAK_FEATURES"               # Statistics covering the peaks of the window
    # - "ENTROPY"                     # Compute signal entropy features
    # - "CORRELATION"                 # Compute correlation between axes
    # - "SPECTRAL"                    # Compute frequency domain features
```

##### Labels

This section defines how labels should be further transformed. The labels need to be aggregated to the frequency of the GPS data, so some logic needs to be determined to determine how labels will be moved to the lower frequency. if `labeled_agg_method` is "MODE" the most frequent activity in the window will be chosen as the label for the window. if `labeled_agg_method` is "RAW", the last activity in the window will be used as the label for the window. 

```yaml
labels:
  labeled_agg_method: "MODE"    # The most frequent activity in the window
  # labeled_agg_method: "RAW"   # The activity at the end of the window

  # DEPRECIATED The list of all valid activites in the data. Used mostly for visualizations.
  valid_activities:             
    - "Resting"
    - "Grazing"
    - "Traveling"
    - "Fighting"
    - "Grooming"
    - "Drinking"
    - "Mineral"
```


#### Analysis

Here the user can create and evaluate models. The user can set the `mode` to either "LOOCV" or "PRODUCT". "LOOCV" will use leave-one-out-cross-validation to estimate the model performance on the labeled target dataset (note, you *must* have labeled data in order to use this). "PRODUCT" will fit a model to the target dataset and make predictions on the entire dataset. `training_info` is an optional parameter that allows the user to choose either a labeled dataset or a trained model as the basis for predictions on the target_dataset. If `training_info` is not included, `target_dataset` will be used for training. 
```yaml
analysis:
  # mode: "LOOCV"     # Do LOOCV on the target dataset. 
  mode: "PRODUCT"   # predict activity labels for a full dataset
  
  # Required: The dataset to make predictions on
  target_dataset: "data/processed/RB_22/processed_data.csv"  
  
  # Optional. Only for PRODUCT mode. The reference training information. 
  # If training_info ends in .csv, a new model will be created for that 
  # .csv dataset to predict target_dataset. If training_info ends in a 
  # .rds, the existing model from that .rds file will be used to classify 
  # the target dataset.
  training_info: "data/models/HMM/trained_model.rds"
  # training_info: "data/models/LSTM/trained_model.keras"
  # training_info: "data/processed/RB_22_tweaked/processed_data.csv"  

  # If true, ignore nighttime predictions (recast them as "NIGHTTIME") 
  day_only: false

  # Path to your preferred R executable. ONLY for windows systems. UNIX can comment this out. 
  r_executable: "C:/Program Files/R/R-4.4.1/bin/Rscript.exe"

  # Where to store outputs of the analysis script. 
  cv_results: "data/cv_results"
  models: "data/models"
  predictions: "data/predictions"

  # DEPRECIATED, can ignore. Directory location to store the results
  output_dir: "data/analysis_results"
  
  # Hidden Markov Model specific settings
  hmm:

    # Decide wether or not to run a HMM
    enabled: true

    # Use time of day as a covariate in the model (still developing)
    time_covariate: false

    # The states to use in the HMM. All states not in this list will be recast to NA
    states: 
      - "Grazing"
      - "Resting"
      - "Traveling"
    
    # List the names of features, the default distribution for them, and the type of data (either "regular" or "circular")
    features:
      - name: "step"
        dist: null
        dist_type: "regular"
      - name: "angle"
        dist: null
        dist_type: "circular"
      - name: "magnitude_mean"
        dist: null 
        dist_type: "regular"
      - name: "magnitude_var"
        dist: null
        dist_type: "regular"

    options:

      # For "PRODUCT" models, we can save additional plots to the output directory:
      show_dist_plots: true      # Shows feature distributions
      remove_outliers: true      # Hides outliers in the plots 
      show_full_range: true      # Adds a subplot showing the full range
      show_correlation: true     # Saves a correlation matrix of the features
      number_of_retry_fits: 1    # number of retries the model should do

      # This section defines the types of distributions whch should be evaluated. In this example, all implemented distributions are used. 
      distributions:
        regular:
          - "LOGNORMAL"
          - "GAMMA"
          - "NORMAL"
          - "EXPONENTIAL"
        circular:
          - "VONMISES"
          - "WRAPPEDCAUCHY"

  # LSTM is still in development.
  lstm:

    # Choose to run this model or not
    enabled: false

    # If true: run a many-to-one (one per sequence) model. This is performant but slow. 
    # If false: use a many-to-many (one-per-observation) model
    ops: true


    ################ Warning these two will SIGNIFICANTLY increase runtime to find parameters. 
    # Bayesian Optimization
    bayes_opt: false                # Whether to use Bayesian optimization 
    bayes_opt_n_calls: 20           # Number of evaluations to perform
    bayes_opt_resume: true          # Whether to resume from previous optimization
    bayes_opt_fast_eval: false      # Use faster evaluation during optimization

    # Grid Search
    hyperparams_search: false   # Whether to run hyperparameter search
    hyperparams_sample: false   # Whether to sample a subset of parameters for quick testing

    # The states to use in the LSTM
    states: 
      - "Grazing"
      - "Resting"
      - "Traveling"

    # The feature names which will be used in the LSTM
    features:
      - "step"
      - "angle"
      - "magnitude_mean"
      - "magnitude_var"

    # Window length 
    max_length: 20

    # LOOCV only: Sets the number of cows to hold out per fold. Increasing this value will
    # significantly improve run time and increase the amount of testing data.  
    cows_per_cv_fold: 1

    # Max seconds between observations to force a new window (many-to-one only)
    max_time_gap: 960

    # Training epochs
    epochs: 1000


    # Some more training specific settings for LSTMS:
    batch_size: 22
    initial_lr: 0.001
    decay_steps: 10000
    decay_rate: 0.4
    clipnorm: 1.9493856438736408
    patience: 13
    min_delta: 1.6850428524897937e-07
    reg_val: 1e-07
```

#### Visuals

```yaml
visuals:
  predictions_path: 'data/predictions/HMM/predictions.csv'
  visuals_root_path: "data/visuals/HMM/"

  # Runs a smoothing algorithm to reduce noise on a 3d surface.
  convolution_surface:
    run: true
  
  # Not fully implemented. IF cow info is given, this can show how age, weight, BCS and more are distributed in the data.
  cow_info_graph:
    implemented: false
    run: false
  
  # Shows the full time range of the study.
  domain:
    implemented: true
    labeled_only: true
    run: true
  
  # HMM only. Shows the CDFs and eCDFs of the chosen distributions and their respective feature values.  
  feature_dists:
    run: false
  
  # Plots the average time spent grazing by cow by day
  heatmap:
    filter_weigh_days: true
    run: true
    weigh_days:
    - '2022-01-14'
    - '2022-01-25'
    - '2022-01-26'
    - '2022-02-22'
    - '2022-02-23'
    - '2022-03-22'
    - '2022-03-23'
  
  # Shows moon effect (>95% illumination) on individual animals and the whole group of animals. 
  moon_phases:
    extension: moon-phases_output/
    run: true
  
  # Makes polar plots for each animal showing their activity over the days. 
  radar:
    extension: radar_output/
    run: true
    show_night: true
  
  # Fits a GLMM to the prediction data and measures the effect the covariates have on the probability of a cow being in a given state. 
  temperature_graph:
    daynight: day
    export_excel: true
    extension: temperature-graph_output/
    minimum_required_values: 30
    run: true
    show_curve: true
    show_table: true
    terms:
    - hod # Hours of dayling in the day
    - temp # temperature 
    - temp^2
    - time # time of the day
    - time^2
    - time^3
    - time^4

    # Interaction terms
    - hod:temp
    - hod:temp^2
    - hod:time
    - temp:time
    - temp:time^4
    - temp^2:time
```


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- Thomas Lipinski (thomaslipinski@montana.edu)

## Acknowledgments

Developed for Montana State University's Ranch Sciences and Industrial Engineering Departments.

