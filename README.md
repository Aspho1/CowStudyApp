# CowStudyApp

A Python package for processing and analyzing sensor data from cattle-mounted GPS and accelerometer devices, developed for Montana State University's Ranch Sciences and Industrial Engineering Departments.

## Overview

This package will take GPS data, Accelerometer data, and Labeled data and combine them to a single dataset. Then, sequential classification of activities can be done using either Hidden Markov Models or Long Short-Term Memory Reccurrent Neural Networks. These models will produce activity predictions as well as estimations for prediction quality. Additionally, visualizations of the data can be done to allow more transparency into the processes and their outcomes.  

### Todo

* Handle when analysis.PRODUCT training_information dataset doesn't have an activity column in hmm (no labels)
* Generate ALL_ACTIVITIES instead of hardcoding it in the `run_hmm.r` script
* Documentation for changing sources
* Test tests
* 
* Fix hardcoded cutoff in `run_hmm.r` prepare_hmm_data

* Can we plot the distributions fits of the untrained model to see how much the model changes the parameters?

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
- See `pyproject.toml` for full dependency list

## Usage

To allow faster usage of this package, a few models have been stored to `data/analysis_results/hmm/FinalModels`. The RB_19 and RB_22 combined model has four "poor" performing cows filtered out from the training data. 


These sections show how to actually interact with the package using a few entry functions which live in `src/`



### Processing data

Place the GPS, Accelerometer, and Label data into their respective directories (defined in the YAML config as `io.gps_directory`, `io.accelerometer_directory`, and `io.labeled_data_path`). make sure te correct config is selected in `src/build_processed_data.py`, then run this file. The main should look something like this:

```python
if __name__ == "__main__":
    config_path = Path("config/RB_19_config.yaml")
    features_df = process_sensor_data(config_path=config_path)
    print(f"Processed {len(features_df)} records")
```

To compare two processed datasets, the module `src/compare_descriptive_stats.py` can be used. 

for instance, to compare RB_19 and RB_22:
```python
if __name__ == "__main__":
    with open('data\\processed\\RB_19\\data_quality_report.json', 'r') as f1:
        RB19_Stats = json.load(f1)
    with open('data\\processed\\RB_22\\data_quality_report.json', 'r') as f2:
        RB22_Stats = json.load(f2)
    
    comp = compare_quality_reports(RB19_Stats, RB22_Stats)
    print_significant_differences(comp)
```


### Running analysis

Make sure the YAML config file (specifically the `analysis` section of it) is set up as desired. At a minimum, make sure the `mode`, `target_dataset`, and `training_info` are correct. The file `src/run_analysis.py` main section should look something like this:

```python
if __name__ == "__main__":
    ##############################################
    ##########    Set the config here   ##########
    ##############################################
    config_path = Path("config/RB_19_config.yaml")

    config = ConfigManager.load(config_path)

    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    logging.info("Starting analysis...")

    # Create output directory
    output_dir = Path(config.analysis.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created output directory: {output_dir}")

    # Run enabled analyses
    if config.analysis.hmm.enabled:
        logging.info("Running HMM analysis...")
        run_hmm_analysis(
            config=config.analysis,
            target_data_path=Path(config.analysis.target_dataset),
            output_dir=output_dir / "hmm",
        )
    if config.analysis.lstm.enabled:
        logging.info("Running HMM analysis...")
        run_lstm_analysis(
            config=config.analysis,
            target_data_path=Path(config.analysis.target_dataset),
            output_dir=output_dir / "lstm",
        )
```
### Creating visuals

Visualizations can be turned on and off in the `visuals` section of the config. To create visualizations, run `src/make_visuals.py` which should have a main function like this:
```python
    # Load configuration
    config_path = Path("config/RB_19_config.yaml")
    config = ConfigManager.load(config_path)

    if config.analysis is None:
        raise ValueError("ERROR: Missing required analysis config section")  
    if config.visuals is None:
        raise ValueError("ERROR: Missing required Visuals config section")  

    # Load dataset
    target_dataset = load_dataset(Path(config.analysis.target_dataset))
    predictions_dataset = load_dataset(Path(config.visuals.predictions_path))
    
    # Create output directory
    output_dir = create_output_directory(config)
    
    generate_plots(config=config
                 , predictions_dataset=predictions_dataset
                 , target_dataset=target_dataset
    )
```

## Development

### Running Tests
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


## Config and Running the code

### Sections
There are three primary sections to this package; Processing Data, Running Analysis, and making visualizations. All three of these sections can be modified through a config file located in `config/`. Each program lets the user choose a custom config file. 

#### Processing Data

Run the script `src/build_processed_data.py` to create a single file which is input to the analysis section. This corresponds to the `io`, `validation`, `features`, and `labels` sections of the config. These next sections will explain how the user can interact with the related sections of the config. 

Thorough data quality reports are placed in the same directory as the processed data file (with the suffix `_dqr.json`). These reports can be useful for understanding data quality issues and actions taken by the module to remedy them.

##### Common

The variables defined in common are available in all other config subsections. 
Here are the valid fields and some sample values:
```yaml
common:
  dataset_name: "RB_19"             # A prefix to attach to directories and files
  timezone: "America/Denver"        # A timezone to use for timezone aware analysis
  gps_sample_interval: 300          # Expected seconds between GPS
  accel_sample_interval: 60         # Expected seconds between Accelerometer 
  excluded_devices: [609, 612]      # A list of devices to fully ignore
```

##### IO

Variables which are explicitly used for setting input and output parameters of the process data section. 

```yaml
io:
  # The folder holding the .csv GPS files
  gps_directory: "data/raw/RB_19/gps"

  # The folder holding the .csv Accelerometer files
  accelerometer_directory: "data/raw/RB_19/accelerometer"

  # The file (.xlsx or .csv) which holds the labeled activity data
  labeled_data_path: "data/raw/RB_19/labeled/rb_2018_activity_observ.csv"

  # The output location for the processed data file
  processed_data_path: "data/processed/RB_19/all_cows_labeled_19_mode.csv"
  
  # OPTIONAL: a YAML dictionary defining cow_ids to collar_ids
  tag_to_device:
    '6082': '609'
    '6085': '610'
    '6094': '614'
    '5090': '602'
    '5051': '612'
    '5010': '617'
    '3065': '603'
    '3091': '611'
    '0276': '600'
    '0286': '605'
    '0256': '606'
    '0105': '608'
    '0174': '613'
    '0163': '616'
    '0101': '604'
    '0024': '607'
    '0011': '615'

  # OPTIONAL: a YAML dictionary defining labels in the labeled_data_path to labels
  # being used in the data processing steps. Keys are the values which appear in 
  # the label data file, values are the transformed values which will appear in the output.
  label_to_value:
    't': 'Traveling'    
    'g': 'Grazing'
    's': 'Resting' # 'Standing'
    'm': 'Mineral'
    'r': 'Grooming'  
    'l': 'Resting' # 'Laying'
    'd': 'Drinking'
    'f': 'Fighting'
```


##### Validation

Parameters which bind the values in the input output process. Each field in `valdation_rules` has a name which corresponds to a column in the input data. Within each field, the user can set minimum and maximum expected values for the data. Additionally, the user can state whether the column is required (or if an error should be thrown if the column doesnt exist), and if the values out of bounds should be filtered or not. If `filter_invalid` is true, the log will report filtered values and they will be removed. if `filter_invalid` is false, the out of bounds values will be reported but not filtered.

```yaml
validation:
  start_datetime: "2018-11-17 00:00:00" # The start of the study period  
  end_datetime: "2019-02-09 00:00:00"   # The end of the study period
  
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

    dop:
      max_value: 10.0
      is_required: true
      filter_invalid: true

    satellites:
      min_value: 0
      is_required: true
      filter_invalid: false

    altitude:
      min_value: 0
      max_value: 1715
      is_required: false
      filter_invalid: false

    temperature_gps:
      min_value: -99
      max_value: 99
      is_required: true
      filter_invalid: false

    #### Accelerometer Feature Validation Rules ####
    x:
      min_value: -41
      max_value: 41
      is_required: true
      filter_invalid: false

    y:
      min_value: -41
      max_value: 41
      is_required: true
      filter_invalid: false

    z:
      min_value: -41
      max_value: 41
      is_required: true
      filter_invalid: false

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

  # The list of all valid activites in the data. Used mostly for visualizations. 
  valid_activities:             
    - "Resting"
    - "Grazing"
    - "Traveling"
    - "Fighting"
    - "Grooming"
    - "Drinking"
    - "Mineral"
```


The package handles data loading, validation, feature extraction, and merging of these data sources.

#### Analysis

Here the user can create and evaluate models. The user can set the `mode` to either "LOOCV" or "PRODUCT". "LOOCV" will use leave-one-out-cross-validation to estimate the model performance on the labeled target dataset (note, you *must* have labeled data in order to use this). "PRODUCT" will fit a model to the target dataset and make predictions on the entire dataset. `training_info` is an optional parameter that allows the user to choose either a labeled dataset or a trained model as the basis for predctions on the target_dataset. If `training_info` is not included, `target_dataset` will be used for training. 
```yaml
analysis:
  # mode: "LOOCV"     # Do LOOCV on the target dataset. 
  mode: "PRODUCT"   # predict activity labels for a full dataset
  
  # Required: The dataset to make predictions on
  target_dataset: "data/processed/RB_19/all_cows_labeled_final.csv"  
  
  # Optional. Only for PRODUCT mode. The reference training information. 
  # If training_info ends in .csv, a new model will be created for that 
  # .csv dataset to predict target_dataset. If training_info ends in a 
  # .rds, the existing model from that .rds file will be used to classify 
  # the target dataset.
  training_info: "data/analysis_results/hmm/PRODUCT/RB_Combined_20250302_185007/trained_model.rds"
  # training_info: "cowstudyapp/data/processed/dataset2.csv"

  # If true, ignore nighttime predictions (recast them as "NIGHTTIME") 
  day_only: true

  # Path to your preferred R executable
  r_executable: "C:/Program Files/R/R-4.3.0/bin/Rscript.exe"

  # Directory location to store the results
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

      # Can kind of add additional covariates here. They break some other reports like transition matrices. 
      # Need to manually update the fitHMM formula argument from ~1 to ~temperature_gps  
      # - name: "temperature_gps"
      #   dist: null
      #   dist_type: "covariate"
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

    # Max seconds between observations to force a new window
    max_time_gap: 960

    # Training epochs
    epochs: 100
```

#### Visuals

```yaml
visuals:
  predictions_path: 'data/analysis_results/hmm/PRODUCT/RB_19_20250306_013021/predictions.csv'
  visuals_root_path: "data/visuals/RB_19/"

  radar:
    run: true
    extension: 'RadarPlots01/'
    show_night: false

  temperature_graph:
    run: false
    # extension: 'MDT_Plots/'
    minimum_required_values: 30


  domain:
    run: true
    implemented: false
    labeled_only: true
    # extension: 'domain_plots/'

  cow_info_graph:
    run: false
    implemented: false
    # extension: 'cow_info_plots/'

  heatmap:
    run: true
    # extension: 'heatmap/'

    filter_weigh_days: false
    weigh_days:
      - "2019-02-08"
      - "2019-02-09"
```


## Project Structure


```
├── .github/
│   ├── workflows/
│   │   ├── ci.yml
├── .idea/
│   ├── .gitignore
│   ├── CowStudyApp.iml
│   ├── misc.xml
│   ├── modules.xml
│   ├── vcs.xml
│   ├── workspace.xml
├── config/
│   ├── default.yaml
│   ├── RB_19_22_combined_config.yaml
│   ├── RB_19_config.yaml
│   ├── RB_22_config.yaml
├── data/
│   ├── analysis_results/
│   │   ├── hmm/
│   │   │   ├── FinalModels/
│   │   │   │   ├── RB_22_Paper_Model/
│   │   │   │   │   ├── plots/
│   │   │   │   │   │   ├── distributions/
│   │   │   │   │   │   │   ├── distribution_angle.pdf
│   │   │   │   │   │   │   ├── distribution_magnitude_mean.pdf
│   │   │   │   │   │   │   ├── distribution_magnitude_var.pdf
│   │   │   │   │   │   │   ├── distribution_step.pdf
│   │   │   │   │   ├── hmm_config.json
│   │   │   │   │   ├── model_parameters.txt
│   │   │   │   │   ├── performance_metrics.txt
│   │   │   │   │   ├── predictions.csv
│   │   │   │   │   ├── trained_model.rds
│   │   │   │   ├── RB_Combined_Model/
│   │   │   │   │   ├── models/
│   │   │   │   │   ├── plots/
│   │   │   │   │   │   ├── distributions/
│   │   │   │   │   │   │   ├── distribution_angle.pdf
│   │   │   │   │   │   │   ├── distribution_magnitude_mean.pdf
│   │   │   │   │   │   │   ├── distribution_magnitude_var.pdf
│   │   │   │   │   │   │   ├── distribution_step.pdf
│   │   │   │   │   ├── hmm_config.json
│   │   │   │   │   ├── model_parameters.txt
│   │   │   │   │   ├── performance_metrics.txt
│   │   │   │   │   ├── predictions.csv
│   │   │   │   │   ├── trained_model.rds
│   │   │   ├── FinalPredictions/
│   │   │   │   ├── RB_22_Paper_Model_preds/
│   │   │   │   │   ├── plots/
│   │   │   │   │   │   ├── distributions/
│   │   │   │   │   ├── hmm_config.json
│   │   │   │   │   ├── model_parameters.txt
│   │   │   │   │   ├── performance_metrics.txt
│   │   │   │   │   ├── predictions.csv
│   │   │   ├── LOOCV/
│   │   │   ├── PRODUCT/
│   │   ├── lstm/
│   │   │   ├── LOOCV/
│   │   │   │   ├── RB_19_20250310_125651/
│   │   │   │   │   ├── plots/
│   │   │   │   │   │   ├── distributions/
│   │   │   │   ├── RB_19_20250310_125849/
│   │   │   │   │   ├── plots/
│   │   │   │   │   │   ├── distributions/
│   │   │   │   ├── RB_19_20250310_125912/
│   │   │   │   │   ├── plots/
│   │   │   │   │   │   ├── distributions/
│   ├── img/
│   │   ├── AnalysisFlow.drawio
│   │   ├── AnalysisFlow.jpg
│   │   ├── HMM_Draw.jpg
│   ├── processed/
│   │   ├── RB_19/
│   │   │   ├── all_cows_labeled_19_mode.csv
│   │   │   ├── all_cows_labeled_final.csv
│   │   │   ├── all_cows_labeled_mode.csv
│   │   │   ├── all_cows_labeled_mode.xlsx
│   │   │   ├── all_cows_labeled_raw.csv
│   │   │   ├── data_quality_report.json
│   │   ├── RB_19_22_Combined/
│   │   │   ├── all_cows_labeled_combined_minimal.csv
│   │   │   ├── all_cows_labeled_combined.csv
│   ├── raw/
│   │   ├── RB_19/
│   │   │   ├── accelerometer/
│   │   │   │   ├── PinPoint 600 2019-02-11.csv
│   │   │   │   ├── PinPoint 602 2019-02-11.csv
│   │   │   │   ├── PinPoint 603 2019-02-11.csv
│   │   │   │   ├── PinPoint 604 2019-02-11.csv
│   │   │   │   ├── PinPoint 605 2019-02-11.csv
│   │   │   │   ├── PinPoint 606 2019-02-11.csv
│   │   │   │   ├── PinPoint 607 2019-02-11.csv
│   │   │   │   ├── PinPoint 608 2019-02-11.csv
│   │   │   │   ├── PinPoint 609 2019-02-11.csv
│   │   │   │   ├── PinPoint 610 2019-02-11.csv
│   │   │   │   ├── PinPoint 611 2019-02-11.csv
│   │   │   │   ├── PinPoint 612 2019-02-11.csv
│   │   │   │   ├── PinPoint 613 2019-02-11.csv
│   │   │   │   ├── PinPoint 614 2019-02-11.csv
│   │   │   │   ├── PinPoint 615 2019-02-11.csv
│   │   │   │   ├── PinPoint 616 2019-02-11.csv
│   │   │   │   ├── PinPoint 617 2019-02-11.csv
│   │   │   ├── gps/
│   │   │   │   ├── PinPoint 600 2019-02-11 09-53-12.csv
│   │   │   │   ├── PinPoint 602 2019-02-11 12-22-12.csv
│   │   │   │   ├── PinPoint 603 2019-02-11 11-35-51.csv
│   │   │   │   ├── PinPoint 604 2019-02-11 09-48-59.csv
│   │   │   │   ├── PinPoint 605 2019-02-11 12-06-29.csv
│   │   │   │   ├── PinPoint 606 2019-02-11 08-19-58.csv
│   │   │   │   ├── PinPoint 607 2019-02-11 08-32-50.csv
│   │   │   │   ├── PinPoint 608 2019-02-11 09-02-49.csv
│   │   │   │   ├── PinPoint 609 2019-02-11 12-16-24.csv
│   │   │   │   ├── PinPoint 610 2019-02-11 12-03-33.csv
│   │   │   │   ├── PinPoint 611 2019-02-11 08-46-30.csv
│   │   │   │   ├── PinPoint 612 2019-02-11 11-18-09.csv
│   │   │   │   ├── PinPoint 613 2019-02-11 08-37-36.csv
│   │   │   │   ├── PinPoint 614 2019-02-11 11-26-32.csv
│   │   │   │   ├── PinPoint 615 2019-02-11 12-12-18.csv
│   │   │   │   ├── PinPoint 616 2019-02-11 08-42-27.csv
│   │   │   │   ├── PinPoint 617 2019-02-11 11-39-46.csv
│   │   │   ├── labeled/
│   │   │   │   ├── rb_2018_activity_observ.csv
│   ├── visuals/
│   │   ├── ComparingDatasets/
│   │   │   ├── P1.png
│   │   │   ├── P2.png
│   │   ├── RB_19/
│   │   │   ├── RadarPlots01/
│   │   │   │   ├── radar_600.jpg
│   │   │   │   ├── radar_602.jpg
│   │   │   │   ├── radar_603.jpg
│   │   │   │   ├── radar_604.jpg
│   │   │   │   ├── radar_605.jpg
│   │   │   │   ├── radar_606.jpg
│   │   │   │   ├── radar_607.jpg
│   │   │   │   ├── radar_608.jpg
│   │   │   │   ├── radar_610.jpg
│   │   │   │   ├── radar_611.jpg
│   │   │   │   ├── radar_613.jpg
│   │   │   │   ├── radar_614.jpg
│   │   │   │   ├── radar_615.jpg
│   │   │   │   ├── radar_616.jpg
│   │   │   │   ├── radar_617.jpg
│   │   │   ├── heatmap_of_grazing.png
│   │   │   ├── time_domain_regular.png
│   │   │   ├── time_domain_zoomed.png
├── src/
│   ├── cowstudyapp/
│   │   ├── analysis/
│   │   │   ├── HMM/
│   │   │   │   ├── .RDataTmp
│   │   │   │   ├── .Rhistory
│   │   │   │   ├── AddingDayNightCols.R
│   │   │   │   ├── Apply_momentuHMM_dynamic.Rmd
│   │   │   │   ├── Compare_distributions_of_feature_data.R
│   │   │   │   ├── compare_models.r
│   │   │   │   ├── run_hmm.r
│   │   │   │   ├── util.r
│   │   │   ├── RNN/
│   │   │   │   ├── run_lstm.py
│   │   ├── dataset_building/
│   │   │   ├── features.py
│   │   │   ├── io.py
│   │   │   ├── labels.py
│   │   │   ├── merge.py
│   │   │   ├── validation.py
│   │   ├── descriptive_stats/
│   │   ├── visuals/
│   │   │   ├── .Rhistory
│   │   │   ├── make_heatmap_of_predictions.py
│   │   │   ├── show_cow_info_vs_pred.py
│   │   │   ├── show_domain_of_data.py
│   │   │   ├── show_feature_distr.py
│   │   │   ├── show_radar_plots.py
│   │   │   ├── show_temp_vs_activity.py
│   │   │   ├── Show_time_covariates.R
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── main.py
│   │   ├── utils.py
│   ├── build_processed_data.py
│   ├── compare_descriptive_stats.py
│   ├── make_visuals.py
│   ├── run_analysis.py
├── tests/
│   ├── __init__.py
│   ├── test_acc_interpolate.py
│   ├── test_core.py
│   ├── test_features.py
│   ├── test_io.py
│   ├── test_merge.py
│   ├── test_utils.py
├── .gitignore
├── .workspace
├── LICENSE
├── pyproject.toml
├── README.md
├── Rplots.pdf
├── show-tree.ps1```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- Thomas Lipinski (thomaslipinski@montana.edu)

## Acknowledgments

Developed for Montana State University's Ranch Sciences and Industrial Engineering Departments.

