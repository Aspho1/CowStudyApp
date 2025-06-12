# from datetime import datetime
import os

from cowstudyapp.analysis.RNN.utils import (
    random_validation_split,
    sequence_aware_validation_split,
    interleaved_validation_split,
    balanced_class_validation_split,
    stratified_sequence_split,
    get_sequence_ids,
    manual_chunking,
    MaskedAccuracy,
    MaskedSparseCategoricalCrossentropy,
    LabeledDataMetricsCallback,
    silence_tensorflow,
    compute_seed

)

silence_tensorflow()

import itertools
import json
from pathlib import Path
# import logging
# from dataclasses import dataclass
import random
import time
import traceback
from typing import Dict, List
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import logging


# from sklearn.model_selection import GroupKFold
from tensorflow.keras.models import Sequential  #, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Masking, Input
from tensorflow.keras.callbacks import EarlyStopping #, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay
# from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.random import set_seed
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam


from cowstudyapp.analysis.RNN.hyper_parameter_tuning import (
    BayesianOptSearch,
    HyperparamSearch
)

import matplotlib
import platform 
import pathlib 

if platform.system()  == 'Linux': 
    pathlib.WindowsPath = pathlib.PosixPath


if True:
    # Use a non-interactive backend that works well with multiprocessing
    matplotlib.use('Agg')  # This must be done before importing pyplot
else:
    matplotlib.use('QtAgg')  # This must be done before importing pyplot

import matplotlib.pyplot as plt
import seaborn as sns

from cowstudyapp.config import ConfigManager
from cowstudyapp.utils import from_posix_col

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
import multiprocessing as mp



class LSTM_Model:

    def __init__(self, config: ConfigManager):
        self.config = config
        # Store metrics for hyperparameter search
        self.last_accuracy = 0
        self.last_f1_score = 0
        self.last_class_accuracies = {}

        self.activity_map = {}
        for idx, state in enumerate(self.config.analysis.lstm.states):# 
            self.activity_map[state] = idx

        # print(self.activity_map)
        # {"Grazing": 0, "Resting":1, "Traveling": 2, np.nan:-1}
        self.UNLABELED_VALUE = -1

        self.inv_activity_map = {self.activity_map[k]: k for k in self.activity_map.keys()}

        self.features = self.config.analysis.lstm.features
        self.nfeatures = len(self.features)
        self.nclasses = len(self.config.analysis.lstm.states)

        self.dropout_rate = 0.1
        lstm_cfg = self.config.analysis.lstm
        # Create output directory for models
        io_type = 'ops' if lstm_cfg.ops else 'opo'


        self.cv_path = self.config.analysis.cv_results / "LSTM" / io_type
        self.pred_path = self.config.analysis.predictions / "LSTM" / io_type
        self.model_path = self.config.analysis.models / "LSTM" / io_type

        for p in [self.cv_path, self.pred_path, self.model_path]:
            p.mkdir(parents=True, exist_ok=True)

        self.max_length: int = lstm_cfg.max_length
        self.max_time_gap = lstm_cfg.max_time_gap
        self.epochs = lstm_cfg.epochs
        self.cows_per_cv_fold = lstm_cfg.cows_per_cv_fold

        self.batch_size = lstm_cfg.batch_size
        self.initial_lr = lstm_cfg.initial_lr
        self.decay_steps = lstm_cfg.decay_steps
        self.decay_rate = lstm_cfg.decay_rate
        self.clipnorm = lstm_cfg.clipnorm
        self.lstm_size = lstm_cfg.lstm_size
        self.dense_size = lstm_cfg.dense_size

        self.patience = lstm_cfg.patience
        self.min_delta = lstm_cfg.min_delta
        self.reg_val = lstm_cfg.reg_val

        self.seed = None
        # set_seed(self.config.analysis.random_seed)
        # np.random.seed(self.config.analysis.random_seed)
        # random.seed(self.config.analysis.random_seed)
        self.masking_val = -9999


    def run_LSTM(self, progress_callback=None):
        
        if progress_callback is None:
            progress_callback = lambda percent, message: print(f"{percent}%: {message}")
        progress_callback(7, "Preparing dataset for LSTM analysis")

        tf.config.experimental.enable_op_determinism()
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


        untransformed = self._get_target_dataset(add_step_and_angle=True)

        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print(untransformed.head(20))
        # print(untransformed.columns)
        
        required_cols: List[str] = self.config.analysis.lstm.features + ["posix_time", "date", 'device_id', 'activity']
        progress_callback(10, f"Normalizing features: {', '.join(self.config.analysis.lstm.features)}")
        transformed = untransformed[required_cols].copy()
        transformed = self._normalize_features(transformed, self.config.analysis.lstm.features)
        # df = df[required_cols].copy()

        # Check if we're doing Bayesian optimization
        if self.config.analysis.lstm.bayes_opt:
            bayes_opt = BayesianOptSearch(self.config, transformed)
            best_params = bayes_opt.run_search(self)
            
            # Set the best parameters
            for key, value in best_params.items():
                setattr(self, key, value)





        self.build_model() # progress_callback

        sequences = self.build_sequences(transformed) #, progress_callback


        # Check if we're doing hyperparameter search
        # if self.config.analysis.lstm.hyperparams_search:
        #     search = HyperparamSearch(self.config)
        #     results = search.run_search(lstm_model=self, sequences=sequences, df=transformed)
        #     # Use the best parameters
        #     best_params = max(results, key=lambda x: x['f1_score'])['params']
        #     print("\nUsing best parameters for final model:")
        #     print(json.dumps(best_params, indent=2))
        #     # Set the best parameters
        #     for key, value in best_params.items():
        #         setattr(self, key, value)
        #         if key == 'max_length':
        #             self.max_length = value


        # Do either LOOCV or product
        if self.config.analysis.mode == "LOOCV":
            progress_callback(40, "Starting LOOCV")
            ############ Should this use transformed or untransformed????
            self.do_loocv(sequences=sequences, df=transformed, progress_callback=progress_callback)

            
        elif self.config.analysis.mode == "PRODUCT":
            print("Starting PRODUCT")
            progress_callback(40, "Starting PRODUCT")
            self.dont_do_loocv(sequences=sequences, df=untransformed[required_cols], progress_callback=progress_callback)

        else:
            raise ValueError(f"Unknown config mode {self.config.analysis.mode}.")



    def _get_params(self):
        params = {}
        for p in ['max_length', 'batch_size', 'initial_lr',
                  'decay_steps', 'decay_rate', 'clipnorm',
                  'lstm_size', 'dense_size', 'patience',
                  'min_delta', 'reg_val']:

            params[p] = getattr(self,p)
        return params

    def _calculate_seed(self):
        return compute_seed(self.config.analysis.random_seed, self._get_params())

    def _set_seed(self,seed):
        set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        os.environ['PYTHONHASHSEED'] = str(seed)

        self.seed = seed



    def build_model(self,progress_callback=None):
        seed = self._calculate_seed()

        self._set_seed(seed)

        self.layers = [
            Input((self.max_length, self.nfeatures)),
            Masking(mask_value=self.masking_val),
        ]

        if self.config.analysis.lstm.ops:

            # progress_callback(40, "Building OPS model architecture")
            self._build_ops_architecture()
        else:
            # progress_callback(40, "Building OPO model architecture")
            self._build_opo_architecture()

    def build_sequences(self,df,progress_callback=None):
        if self.config.analysis.lstm.ops:
            # progress_callback(20, "Building sequences for one-per-sample (OPS) analysis")
            return self._build_sequences_ops(df, progress_callback)
        else:
            # progress_callback(20, "Building sequences for one-per-observation (OPO) analysis")
            return self._build_sequences_opo(df, progress_callback)



    def _build_ops_architecture(self):
        """Build model architecture for one-per-observation"""
        self.layers.extend([
            LSTM(self.lstm_size,
                return_sequences=False,
                recurrent_dropout=self.dropout_rate,
                dropout=self.dropout_rate,
                kernel_regularizer=L2(self.reg_val),
                recurrent_regularizer=L2(self.reg_val)
                ),
            Dense(self.dense_size, activation='relu'),
            Dropout(self.dropout_rate),  
            Dense(self.nclasses, activation='softmax')
        ])
    

    def _build_opo_architecture(self):
        """Build model architecture for one-per-sequence"""
        self.layers.extend([
            LSTM(self.lstm_size,
                return_sequences=True,
                recurrent_dropout=self.dropout_rate,
                dropout=self.dropout_rate,
                kernel_regularizer=L2(self.reg_val),
                recurrent_regularizer=L2(self.reg_val)
                ),

            Dense(self.dense_size, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(self.nclasses, activation='softmax')
        ])


    def _get_target_dataset(self, add_step_and_angle=True):
        df = pd.read_csv(self.config.analysis.target_dataset)

        # print(df.activity.unique())
        step_size = self.config.analysis.gps_sample_interval//60
        dfs = []
        
        for device_id, cow_data in df.groupby("device_id"):
            # Sort data first
            cow_data = cow_data.sort_values('posix_time')
            
            # Create full index
            time_range = pd.date_range(
                start=pd.to_datetime(cow_data['posix_time'].min(), unit='s'),
                end=pd.to_datetime(cow_data['posix_time'].max(), unit='s'),
                freq=f'{step_size}min'
            )
            full_index_df = pd.DataFrame({
                'posix_time': time_range.astype('int64')//1e9, 
                'device_id': device_id
            })
            
            # Merge to create a complete time series with potential NaN values
            cow_data = pd.merge(full_index_df, cow_data, on=['posix_time', 'device_id'], how='left')
            
            # Now apply the isolated NaN interpolation
            numeric_cols = cow_data.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                # Skip device_id and posix_time
                if col in ['device_id', 'posix_time']:
                    continue
                    
                # Find isolated NaN values (valid values on both sides)
                series = cow_data[col].copy()
                mask = series.isna() & series.shift(1).notna() & series.shift(-1).notna()
                
                # Only interpolate where mask is True (isolated NaNs)
                if mask.any():
                    # Create a temporary series with only the isolated NaNs for interpolation
                    temp = series.copy()
                    temp[~mask] = temp[~mask].ffill()  # Fill non-isolated NaNs temporarily
                    temp = temp.interpolate(method='linear')
                    
                    # Apply only the interpolated isolated NaNs back to original series
                    series[mask] = temp[mask]
                    cow_data[col] = series
            
            if add_step_and_angle:
                # Calculate forward-looking step (distance to next point)
                cow_data['step'] = np.sqrt(
                    (cow_data['utm_easting'].shift(-1) - cow_data['utm_easting'])**2 +
                    (cow_data['utm_northing'].shift(-1) - cow_data['utm_northing'])**2
                )
                
                # Calculate vectors (forward-looking displacement vectors)
                x_diff_current = cow_data['utm_easting'].shift(-1) - cow_data['utm_easting']
                y_diff_current = cow_data['utm_northing'].shift(-1) - cow_data['utm_northing']
                x_diff_prev = cow_data['utm_easting'] - cow_data['utm_easting'].shift(1)
                y_diff_prev = cow_data['utm_northing'] - cow_data['utm_northing'].shift(1)
                
                # Calculate headings
                current_heading = np.arctan2(y_diff_current, x_diff_current)
                prev_heading = np.arctan2(y_diff_prev, x_diff_prev)
                
                # Calculate the difference in heading (φ)
                cow_data['angle'] = current_heading - prev_heading
            
            # Add to the list of processed dataframes
            dfs.append(cow_data)
        
        # Combine all device data
        df_out = pd.concat(dfs).sort_values(['device_id', 'posix_time'])
        
        # Add time columns
        if 'mt' not in df.columns:
            df_out['mt'] = from_posix_col(df_out['posix_time'], self.config.analysis.timezone)
        
        # Extract date component
        df_out['date'] = df_out['mt'].dt.date

        df_out['pct_of_day'] = ((df_out['mt'].dt.hour * 60) + df_out['mt'].dt.minute) / 1440

        # print(df_out['pct_of_day'].describe())
        
        return df_out


    def _normalize_features(self, df, features):
        """Normalize features before sequence building"""
        
        scaler = StandardScaler()
        
        # Normalize features
        normalized_data = scaler.fit_transform(df[features])
        df[features] = normalized_data

        # Store scaler
        self.scaler = scaler
        return df


    def _build_sequences_opo(self, df: pd.DataFrame, progress_callback=None):
        """
        Build sequences for one-per-observation (OPO) analysis.
        Pre-calculates sequence counts and includes every observation exactly once.
        """
        if progress_callback is None:
            progress_callback = lambda percent, message: None
        
        max_length = self.max_length

        num_features = len(self.features)
        
        # Sample interval calculations
        step_size = self.config.analysis.gps_sample_interval // 60  # Sample interval in minutes
        minutes_in_day = 24 * 60
        expected_records = minutes_in_day // step_size
        lost_hour = 60 // step_size  # Records in one hour (for DST handling)
        
        progress_callback(5, "Analyzing data to determine sequence counts")
        
        # Daily sequence mode (for sequence lengths near 288)
        if max_length >= expected_records - lost_hour and max_length <= expected_records + lost_hour:
            is_daily_mode = True
            progress_callback(10, f"Using daily sequence mode (length={max_length}, expected={expected_records})")
            
            # Calculate exact number of sequences (one per day with sufficient data)
            total_sequences = 0
            device_sequences = {}  # To track how many sequences per device
            
            for device_id, device_data in df.groupby("device_id"):
                device_sequences[device_id] = 0
                for date, day_data in device_data.groupby("date"):
                    # We include all days since we need to process all observations
                    device_sequences[device_id] += 1
                    total_sequences += 1
            
            progress_callback(15, f"Will create {total_sequences} daily sequences")
        else:
            is_daily_mode = False
            progress_callback(10, f"Using custom sequence length mode (length={max_length})")
            
            # Calculate exact number of sequences needed
            total_sequences = 0
            device_sequences = {}  # To track how many sequences per device
            
            for device_id, device_data in df.groupby("device_id"):
                # Calculate how many complete sequences we can make
                data_len = len(device_data)
                num_sequences = (data_len + max_length - 1) // max_length  # Ceiling division
                device_sequences[device_id] = num_sequences
                total_sequences += num_sequences
            
            progress_callback(15, f"Will create {total_sequences} sequences of length {max_length}")
        
        # Pre-allocate arrays with exact sizes
        cow_date_keys = np.zeros((total_sequences, 2), dtype=object)
        all_sequences = np.full((total_sequences, max_length, num_features), self.masking_val, dtype=np.float32)
        all_labels = np.full((total_sequences, max_length), self.UNLABELED_VALUE, dtype=np.int32)
        
        # Pre-fill NaN values in the feature columns
        df[self.features] = df[self.features].fillna(self.masking_val)
        
        # Process the data
        sequence_idx = 0
        device_count = 0
        total_devices = len(device_sequences)
        
        for device_id, device_data in df.groupby("device_id"):
            print(f"OPO Building sequences -- Processing device {device_id} ({device_count+1}/{total_devices})")

            device_count += 1
            device_progress_base = 20 + (75 * (device_count - 1) / total_devices)
            device_progress_next = 20 + (75 * device_count / total_devices)
            
            progress_callback(int(device_progress_base), 
                            f"Processing device {device_id} ({device_count}/{total_devices})")
            
            # Ensure data is sorted by time
            device_data = device_data.sort_values('posix_time').reset_index(drop=True)
            
            if is_daily_mode:
                # Process by days
                day_count = 0
                total_days = device_data['date'].nunique()
                
                for date, day_data in device_data.groupby("date"):
                    day_count += 1
                    day_progress = device_progress_base + (device_progress_next - device_progress_base) * day_count / total_days
                    
                    progress_callback(int(day_progress), 
                                    f"Device {device_id}: day {day_count}/{total_days}")
                    
                    # Extract features and labels
                    day_features = day_data[self.features].values
                    day_activities = day_data['activity'].map(lambda x: self.activity_map.get(x, self.UNLABELED_VALUE)).values
                    
                    day_len = len(day_features)
                    
                    # Handle various day lengths properly
                    if day_len <= max_length:
                        # Day fits in sequence - just copy it
                        all_sequences[sequence_idx, :day_len] = day_features
                        all_labels[sequence_idx, :day_len] = day_activities
                    elif day_len == expected_records + lost_hour and max_length < day_len:
                        # Fall back day (extra hour) that exceeds max_length
                        # Take first max_length observations
                        all_sequences[sequence_idx] = day_features[:max_length]
                        all_labels[sequence_idx] = day_activities[:max_length]
                    elif day_len > max_length:
                        # Day longer than max_length - truncate
                        all_sequences[sequence_idx] = day_features[:max_length]
                        all_labels[sequence_idx] = day_activities[:max_length]
                    
                    # Store reference information
                    cow_date_keys[sequence_idx] = [device_id, date]
                    sequence_idx += 1
            else:
                # Process by fixed chunks of max_length
                data_len = len(device_data)
                expected_sequences = device_sequences[device_id]
                
                # Extract all features and activities at once
                device_features = device_data[self.features].values
                device_activities = device_data['activity'].map(lambda x: self.activity_map.get(x, self.UNLABELED_VALUE)).values
                
                # Process in chunks
                for chunk_idx in range(expected_sequences):
                    chunk_progress = device_progress_base + (device_progress_next - device_progress_base) * (chunk_idx + 1) / expected_sequences
                    
                    if (chunk_idx + 1) % max(1, expected_sequences // 10) == 0:  # Update every ~10% of chunks
                        progress_callback(int(chunk_progress), 
                                        f"Device {device_id}: chunk {chunk_idx + 1}/{expected_sequences}")
                    
                    # Calculate chunk boundaries
                    start_idx = chunk_idx * max_length
                    end_idx = min(start_idx + max_length, data_len)
                    chunk_len = end_idx - start_idx
                    
                    # Handle the last chunk which may be partial
                    if chunk_len < max_length:
                        # Place data at the end of the sequence (right-aligned)
                        padding_len = max_length - chunk_len
                        all_sequences[sequence_idx, padding_len:] = device_features[start_idx:end_idx]
                        all_labels[sequence_idx, padding_len:] = device_activities[start_idx:end_idx]
                    else:
                        # Full chunk
                        all_sequences[sequence_idx] = device_features[start_idx:end_idx]
                        all_labels[sequence_idx] = device_activities[start_idx:end_idx]
                    
                    # Store reference information - use middle index as reference
                    ref_idx = (start_idx + end_idx) // 2
                    cow_date_keys[sequence_idx] = [device_id, ref_idx]
                    sequence_idx += 1
        
        # Verify that we created the expected number of sequences
        assert sequence_idx == total_sequences, f"Expected {total_sequences} sequences, created {sequence_idx}"
        
        progress_callback(95, f"Created {sequence_idx} sequences")
        progress_callback(100, "OPO sequence building complete")
        
        return {
            'Cow_Date_Key': cow_date_keys,
            'X': all_sequences,
            'Y': all_labels
        }


    def _build_sequences_ops(self, df: pd.DataFrame, progress_callback=None):
        """
        Build sequences for many-to-one classification with zero padding.
        Each sequence predicts the activity at its last timestamp.
        Uses efficient NumPy operations for better performance.
        """
        if progress_callback is None:
            progress_callback = lambda percent, message: None
        
        # Get total number of rows for progress calculation
        total_rows = len(df)
        
        # Get unique device IDs for reporting
        unique_devices = df['device_id'].unique()
        num_devices = len(unique_devices)
        progress_callback(5, f"Building sequences for {num_devices} devices ({total_rows} observations)")
        
        # Pre-process feature data
        # Fill NaN values in features
        df[self.features] = df[self.features].fillna(self.masking_val)
        
        # Pre-allocate arrays for results - one row per observation
        cow_date_keys = np.zeros((total_rows, 2), dtype=np.int32)
        all_sequences = np.zeros((total_rows, self.max_length, len(self.features)),
                                dtype=np.float32)
        all_labels = np.zeros(total_rows, dtype=np.int32)
        
        # Track the current position in our result arrays
        sequence_count = 0
        
        # Process each device's data
        for idx, (device_id, data) in enumerate(df.groupby("device_id")):
            print(f"OPS Building sequences -- Processing device {device_id} ({idx+1}/{num_devices})")
            device_progress_base = 5 + (90 * idx // num_devices)
            device_progress_target = 5 + (90 * (idx + 1) // num_devices)
            
            progress_callback(device_progress_base, 
                            f"Processing device {device_id} ({idx+1}/{num_devices})")
            
            data_len = len(data)
            device_progress_interval = max(1, data_len // 20)  # Update 20 times per device
            
            # Feature matrix for this device
            feature_matrix = data[self.features].values
            
            # Activity labels
            activities = data['activity'].map(lambda x: self.activity_map.get(x, self.UNLABELED_VALUE)).values
            
            # Posix times for checking time gaps
            posix_times = data['posix_time'].values
            
            # Initialize empty sequence with masking values
            current_sequence = np.full((self.max_length, len(self.features)),
                                    self.masking_val, dtype=np.float32)
            current_sequence_len = 0
            
            # Create sequences for this device
            for i in range(data_len):
                # Report progress periodically
                if i % device_progress_interval == 0:
                    device_progress = device_progress_base + (device_progress_target - device_progress_base) * i // data_len
                    progress_callback(device_progress, 
                                    f"Device {device_id}: {i}/{data_len} observations processed")
                
                # Check for time gap
                if i > 0 and (posix_times[i] - posix_times[i-1] > self.config.analysis.lstm.max_time_gap):
                    # Time gap too large, reset sequence
                    current_sequence.fill(self.masking_val)
                    current_sequence_len = 0
                
                # Add current observation to sequence using a circular buffer approach
                if current_sequence_len < self.max_length:
                    # Sequence not full yet, add at the end
                    current_sequence[self.max_length - current_sequence_len - 1] = feature_matrix[i]
                    current_sequence_len += 1
                else:
                    # Sequence full, shift everything up and add new at the end
                    current_sequence = np.roll(current_sequence, -1, axis=0)
                    current_sequence[-1] = feature_matrix[i]
                
                # Store the sequence for every observation
                all_sequences[sequence_count] = current_sequence
                cow_date_keys[sequence_count] = [device_id, i]
                all_labels[sequence_count] = activities[i]
                sequence_count += 1
        
        # Final progress update
        progress_callback(100, f"Sequence building complete. Created {sequence_count} sequences.")
        
        # Validate that we created the expected number of sequences
        assert sequence_count == total_rows, f"Expected {total_rows} sequences, but created {sequence_count}"
        
        return {
            'Cow_Date_Key': cow_date_keys,
            'X': all_sequences,
            'Y': all_labels,
        }



    def dont_do_loocv(self, sequences, df, progress_callback=None):
        """Process data in product mode with trained or new model"""
        if progress_callback is None:
            progress_callback = lambda percent, message: None
        
        # print(df.head(20))

        import gc

        progress_callback(60, "Preparing data for model training/prediction")
        Cow_Date_Key_full = sequences['Cow_Date_Key']
        X_full = sequences['X']
        Y_full = sequences['Y']
        
        # ONLY TRAIN ON SEQUENCES WITH LABELS
        has_label = (Y_full != -1) if len(Y_full.shape) == 1 else np.any(Y_full[:,:] != -1, 1)
        Cow_Date_Key = Cow_Date_Key_full[has_label]
        X = X_full[has_label]
        Y = Y_full[has_label]
        
        # Split data for training/testing
        progress_callback(65, "Splitting data into training and test sets")
        # train_X, test_X, train_Y, test_Y = train_test_split(
        #     X, Y, random_state=self.config.analysis.random_seed, test_size=.3, shuffle=True
        # )

        # Print label statistics
        # train_label_count = np.sum(train_Y != self.UNLABELED_VALUE)
        # test_label_count = np.sum(test_Y != self.UNLABELED_VALUE)
        # progress_callback(67, f"Training set: {len(train_X)} sequences with {train_label_count} labeled timesteps")
        # progress_callback(68, f"Test set: {len(test_X)} sequences with {test_label_count} labeled timesteps")
        
        model = None
        history = None
        # Training Summary:
        # Final Training Loss: 0.0882
        # Final Training Accuracy: 0.5787
        # Final Validation Loss: 0.1463
        # Final Validation Accuracy: 0.5815

        # Check if we should load a model from a path
        if self.config.analysis.training_info_path and str(self.config.analysis.training_info_path).endswith(".keras"):
            progress_callback(70, "Loading existing model for prediction")
            try:
                model_path = self.config.analysis.training_info_path
                progress_callback(72, f"Loading model from: {model_path}")
                # Load the saved model
                model = tf.keras.models.load_model(model_path, custom_objects={
                    'MaskedConv1D': MaskedConv1D  # Include any custom layers here
                })
                # For consistency with the training case
                progress_callback(75, "Model loaded successfully")
                print("LOADED THE MODEL SUCCESSFULLY FROM ", self.config.analysis.training_info_path)
            except Exception as e:
                print("ERROR LOADING THE MODEL FROM ", self.config.analysis.training_info_path)
                progress_callback(72, f"Error loading model: {str(e)}")
                progress_callback(73, "Falling back to training a new model")
                print(traceback.format_exc())
                model, history = self._make_LSTM(X, Y, test_X, test_Y, progress_callback)
                # Save the global model
                model_path = self.model_path / "global_lstm_model.keras"
                model.save(model_path)
                progress_callback(80, f"New model saved to {model_path}")
                # Plot training history
                if history:
                    self._plot_single_history(history)


        else:
            progress_callback(70, "Training new model")
            model, history = self._make_LSTM(X, Y, progress_callback)
            # Save the global model
            model_path = self.model_path / "global_lstm_model.keras"
            model.save(model_path)
            progress_callback(80, f"Model saved to {model_path}")
            # Plot training history
            if history:
                self._plot_single_history(history)

        del X, Y
        gc.collect()

        # Generate predictions for the full dataset
        progress_callback(85, "Generating predictions for the full dataset")

        # Process predictions in batches to avoid memory issues
        batch_size_pred = 16384  # Adjust based on your system's memory
        n_batches = (len(X_full) + batch_size_pred - 1) // batch_size_pred
        
        if n_batches > 1:
            # Multi-batch prediction
            predictions = []
            for i in range(n_batches):
                start_idx = i * batch_size_pred
                end_idx = min((i + 1) * batch_size_pred, len(X_full))
                progress_callback(85 + (5 * i // n_batches), 
                                f"Predicting batch {i+1}/{n_batches} ({start_idx}-{end_idx})")
                

                # print(X_full.shape)
                # print(X_full[0].shape)
                batch_pred = model.predict(X_full[start_idx:end_idx], verbose=0)
                predictions.append(batch_pred)
                
                # Force garbage collection after each batch
                gc.collect()
            
            # Combine batch predictions
            if len(Y_full.shape) == 1:  # OPS case
                predictions = np.vstack(predictions)
            else:  # OPO case
                predictions = np.vstack([p for p in predictions])
        else:
            # Single batch prediction
            predictions = model.predict(X_full, verbose=0)
        
        # Create a results dataframe
        results_df = pd.DataFrame()
        
        # Process the predictions and map back to the original data
        progress_callback(90, "Creating prediction results dataframe")
        all_cow_preds_aligned = pd.DataFrame(columns=['device_id', 'posix_time', 'predicted_state'])
        unique_cows = np.unique(Cow_Date_Key_full[:,0].astype(int))
        current_cow = 0
        total_cows = len(unique_cows)
        
        # Extract predicted classes from prediction probabilities
        if len(Y_full.shape) == 1:  # OPS (many-to-one) case
            progress_callback(91, "Processing OPS predictions")
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Process one cow at a time to manage memory
            for cow_id in unique_cows:
                current_cow += 1
                progress_callback(91 + int(4 * current_cow / total_cows),
                                f"Processing cow {current_cow}/{total_cows}")
                
                key_for_this_cow = Cow_Date_Key_full[:,0] == cow_id
                this_cows_preds = pd.Series(predicted_classes[key_for_this_cow]).map(
                    lambda x: self.inv_activity_map.get(x, 'HELLO1')
                )
                
                cow_df = df[df['device_id'] == cow_id].copy()
                cow_df['predicted_state'] = this_cows_preds.values[:len(cow_df)]  # Ensure length matches
                results_df = pd.concat([results_df, cow_df], ignore_index=True)
                
                # Force garbage collection periodically
                if current_cow % 5 == 0:
                    gc.collect()
            
        else:  # OPO (one-per-observation) case
            progress_callback(91, "Processing OPO predictions")
            predicted_classes = np.argmax(predictions, axis=2)
            
            # Process one cow at a time
            for cow_id in unique_cows:
                current_cow += 1
                progress_callback(91 + int(4 * current_cow / total_cows),
                                f"Processing cow {current_cow}/{total_cows}")
                
                key_for_this_cow = Cow_Date_Key_full[:,0] == cow_id
                this_cows_preds_ids = predicted_classes[key_for_this_cow]
                this_cows_preds_flattened = this_cows_preds_ids.flatten()
                this_cows_preds = pd.Series(this_cows_preds_flattened).map(
                    lambda x: self.inv_activity_map.get(x, 'HELLO2')
                )
                
                this_cows_first_ts = df[df['device_id'] == cow_id]['posix_time'].iloc[0]
                step_size = self.config.analysis.gps_sample_interval
                n_total = len(this_cows_preds_flattened)
                
                # Create index range safely
                idx_start = int(this_cows_first_ts)
                idx_end = int(this_cows_first_ts + (n_total*step_size))
                idx_step = int(step_size)
                idx = list(range(idx_start, idx_end, idx_step))[:n_total]  # Ensure length matches predictions
                
                # Create temporary dataframe
                this_cows_data = pd.DataFrame({
                    'device_id': cow_id,
                    'posix_time': idx[:len(this_cows_preds)],  # Ensure length matches
                    'predicted_state': this_cows_preds[:len(idx)]  # Ensure length matches
                })
                
                all_cow_preds_aligned = pd.concat([all_cow_preds_aligned, this_cows_data], ignore_index=True)
                
                # Force garbage collection periodically
                if current_cow % 5 == 0:
                    gc.collect()
            
            # Merge with original data
            results_df = pd.merge(
                left=all_cow_preds_aligned,
                right=df,
                on=['device_id', 'posix_time'],
                how='left'
            )
        
        # Clean up large objects before continuing
        del all_cow_preds_aligned, predicted_classes, predictions, X_full, Y_full
        gc.collect()

        results_df[self.features] = self.scaler.inverse_transform(results_df[self.features])
        # Reorder columns for clarity
        results_df = results_df[
                            ['device_id', 'posix_time'] + 
                            [col for col in results_df.columns if col not in ['device_id', 'posix_time', 'activity', 'predicted_state', 'predicted_activity_id', 'index', 'date']] + 
                            ['activity', 'predicted_state']
            ]
        results_df.rename(columns={'device_id': 'ID'}, inplace=True)
        
        # Generate performance metrics for labeled data only
        progress_callback(95, "Generating performance metrics")
        
        # Extract actual and predicted states for labeled data only
        labeled_data = results_df.dropna(subset=['activity']).copy()
        y_true = labeled_data['activity'].values
        y_pred = labeled_data['predicted_state'].values
        
        # Get unique activity states (sorted for consistent display)
        activity_states = sorted(set(self.inv_activity_map.values()))
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred, labels=activity_states)
        
        # Calculate performance metrics by state
        metrics_by_state = {}
        for i, state in enumerate(activity_states):
            TP = conf_matrix[i, i]
            FP = conf_matrix[:, i].sum() - TP
            FN = conf_matrix[i, :].sum() - TP
            TN = conf_matrix.sum() - TP - FP - FN
            
            accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            
            metrics_by_state[state] = {
                'Accuracy': accuracy,
                'Sensitivity': sensitivity,
                'Specificity': specificity,
                'F1 Score': f1
            }
        
        # Calculate overall accuracy
        overall_accuracy = accuracy_score(y_true, y_pred)
        
        # Create the metrics text file
        metrics_path = self.pred_path / "performance_metrics.txt"
        
        with open(metrics_path, 'w') as f:
            # Write report header
            f.write("Performance Metrics Report\n")
            f.write("========================\n\n")
            
            # Write confusion matrix
            f.write("Confusion Matrix:\n")
            f.write("=================\n")
            
            # Create header row for the confusion matrix
            header_row = "Actual \\ Predicted"
            for state in activity_states:
                header_row += f" {state:>10}"
            f.write(header_row + "\n")
            
            # Write the matrix rows
            for i, state in enumerate(activity_states):
                row = f"{state:>17}"
                for j in range(len(activity_states)):
                    row += f" {conf_matrix[i, j]:>10d}"
                f.write(row + "\n")
            
            f.write("\n")
            
            # Write performance metrics by state
            f.write("Performance Metrics by State:\n")
            f.write("============================\n")
            
            # Write header for metrics table
            f.write(f"{'Actual':>12} {'Predicted':>12} {'Accuracy':>10} {'Sensitivity':>12} {'Specificity':>12} {'F1 Score':>10}\n")
            
            # Write metrics for each state
            for state in activity_states:
                metrics = metrics_by_state[state]
                f.write(f"{state:>12} {state:>12} {metrics['Accuracy']:>10.3f} {metrics['Sensitivity']:>12.3f} "
                    f"{metrics['Specificity']:>12.3f} {metrics['F1 Score']:>10.3f}\n")
            
            f.write("\n")
            
            # Write overall accuracy
            f.write(f"Overall Accuracy: {overall_accuracy:.3f}\n")
            
            # Add sample counts
            f.write("\n")
            f.write("Sample Counts:\n")
            f.write("=============\n")
            class_counts = labeled_data['activity'].value_counts().to_dict()
            for state in activity_states:
                count = class_counts.get(state, 0)
                f.write(f"{state:>12}: {count:>6d} samples\n")
            f.write(f"{'Total':>12}: {len(labeled_data):>6d} samples\n")
        
        # Save to CSV
        output_path = self.pred_path / "predictions.csv"

        print("saving predictions to", output_path)
        results_df.to_csv(output_path, index=False)
        time.sleep(1)
        progress_callback(100, f"Predictions saved to {output_path}, metrics saved to {metrics_path}")
        
        return model, history




    def do_loocv(self, sequences: Dict[str, np.ndarray], df, n=None, compute_metrics_only=False, n_jobs=-1,progress_callback=None):
        """
        Run Leave-One-Out Cross Validation with parallelization
        
        Parameters:
        -----------
        sequences: Dictionary with Cow_Date_Key, X, and Y arrays
        df: Original dataframe
        n: Number of folds
        compute_metrics_only: If True, does not plot each fold
        n_jobs: Number of parallel processes to use (-1 for all available)
        """
        if n is None:
            n = self.cows_per_cv_fold

        if progress_callback is None:
            progress_callback = lambda percent, message: None

        Cow_Date_Key:np.ndarray = sequences['Cow_Date_Key']
        X:np.ndarray = sequences['X']
        Y:np.ndarray = sequences['Y']

        print(f"Cow_Date_Key shape: {Cow_Date_Key.shape}")
        print(f"X shape: {X.shape}")
        print(f"Y shape: {Y.shape}")

        # # ONLY TRAIN ON SEQUENCES WITH LABELS
        # has_label = (Y != -1) if len(Y.shape) == 1 else np.any(Y[:,:] != -1, 1)

        if len(Y.shape) == 1:  # OPS mode
            has_label = (Y != self.UNLABELED_VALUE)
        else:  # OPO mode
            # Check if any timestep in the sequence has a valid label
            has_label = np.any(Y != self.UNLABELED_VALUE, axis=1)
        
        print(f"Sequences with at least one label: {np.sum(has_label)} out of {len(has_label)}")


        Cow_Date_Key = Cow_Date_Key[has_label]
        X = X[has_label]
        Y = Y[has_label]

        # For storing results
        all_predictions = []
        all_actual = []
        

        groups:np.ndarray = Cow_Date_Key[:,0]
        # print("ALL GROUPS", groups) 
        unique_cows = np.unique(groups)
        # print("ALL unique groups", unique_cows)
        test_chunks = manual_chunking(unique_cows, n)

        # Determine number of processes
        n_jobs = (mp.cpu_count()-1) if n_jobs == -1 else n_jobs
        n_jobs = min(n_jobs, len(test_chunks))  # Can't use more processes than chunks
        n_jobs=1
        
        # Instead of using a Pool directly, we'll use the starmap approach with pre-built arguments
        print(f"Running LOOCV with {n_jobs} parallel processes")
        progress_callback(45, f"Running LOOCV with {n_jobs} parallel processes")
        # Create argument list for each fold
        fold_args = []
        for i, test_chunk in enumerate(test_chunks):
            test_mask = np.isin(groups, test_chunk)
            train_mask = ~test_mask
            test_X, test_Y = X[test_mask], Y[test_mask]
            train_X, train_Y = X[train_mask], Y[train_mask]
            progress_callback(45, f"Initializing fold with held out device_ids: {test_chunk}")
            fold_args.append((test_chunk, train_X, train_Y, test_X, test_Y, self.seed + i * 10000))
        
        # Process folds in parallel
        if n_jobs > 1:
            mp.set_start_method('spawn', force=True)
            with mp.Pool(processes=n_jobs) as pool:
                # Use starmap to pass multiple arguments
                results = pool.starmap(self._process_fold_mp, fold_args)
        else:
            # Process sequentially if n_jobs=1
            results = [self._process_fold_mp(*args) for args in fold_args]
        
        # Process results
        all_predictions = [r['predictions'] for r in results]
        all_actual = [r['actual'] for r in results]
        all_histories = [r['history'] for r in results if r.get('history')]
        
        # Print fold results
        for i, result in enumerate(results):
            test_chunk = result['test_chunk']
            lbl = '-'.join([str(int(c)) for c in test_chunk])
            print(f"{'-'*80} Results for group {lbl} {'-'*80}")
            print(f"Accuracy on valid data points: {result['accuracy']:.4f}")
            progress_callback(99, f"Testing on held out device_ids {test_chunk} had an accuracy of {result['accuracy']:.4f}")
            print("Class distribution in test data:")
            for class_name, acc in result['class_accuracies'].items():
                count = result['class_counts'].get(class_name, 0)
                f1 = result['class_f1'].get(class_name,np.nan)
                print(f"    Class {class_name} | accuracy: {acc:.4f} | f1: {f1:.4f} | (n={count})")
            for i in range(len(result['confusion_matrix'])):
                print(" | ".join([f"{x:<2}" for x in result['confusion_matrix'][i]]))
            print(f"{'-'*182}")
        
        # Calculate overall metrics
        self._calculate_overall_metrics(all_predictions, all_actual)
        
        # Save and display overall results
        if not compute_metrics_only and all_histories:
            self._summarize_results(all_predictions, all_actual, all_histories)

        return [], []


    def _process_fold_mp(self, test_chunk, train_X, train_Y, test_X, test_Y, seed):
        """
        Process a single fold in LOOCV - this method is designed to be called via multiprocessing
        """

        # ===== OVERALL RESULTS =====
        # Total accuracy: 0.8815
        # Grazing accuracy: 0.8719 (n=203)
        # Resting accuracy: 0.8922 (n=232)
        # Traveling accuracy: 0.8696 (n=46)

        self._set_seed(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['PYTHONHASHSEED'] = str(seed)
        tf.config.experimental.enable_op_determinism()


        # Full training
        model, history, pred_y_classes = self._make_LSTM(
            test_X=test_X, test_Y=test_Y, train_X=train_X, train_Y=train_Y
        )
        history_dict = history
        
        if len(test_Y.shape) > 1:
            flat_pred = pred_y_classes.flatten()
            flat_actual = test_Y.flatten()
        else:
            flat_actual = test_Y
            flat_pred = pred_y_classes
        
        # Only evaluate on valid data points
        valid_indices = flat_actual != self.UNLABELED_VALUE
        valid_pred = flat_pred[valid_indices]
        valid_actual = flat_actual[valid_indices]
        
        # Calculate overall accuracy
        accuracy = np.mean(valid_pred == valid_actual)
        
        # Per-class metrics
        class_accuracies = {}
        class_f1 = {}
        class_counts = {}
        for class_name, class_id in self.activity_map.items():
            if class_id == self.UNLABELED_VALUE:
                continue  # Skip NaN class
            
            class_mask = flat_actual == class_id
            if np.sum(class_mask) > 0:
                class_acc = np.mean(flat_pred[class_mask] == class_id)
                class_accuracies[class_name] = float(class_acc)
                class_f1[class_name] = f1_score(valid_actual == class_id, valid_pred == class_id)
                class_counts[class_name] = int(np.sum(class_mask))
        
        # Confusion matrix - create it from flattened arrays
        cm = confusion_matrix(
            valid_actual, 
            valid_pred,
            labels=sorted([v for k,v in self.activity_map.items() if k is not np.nan])
        ).tolist()  # Convert to list for serialization

        
        # print(class_f1)
        # Return a dictionary of results
        return {
            'test_chunk': test_chunk,
            'predictions': pred_y_classes,
            'actual': test_Y,
            'accuracy': float(accuracy),
            'class_f1': class_f1,
            'class_accuracies': class_accuracies,
            'class_counts': class_counts,
            'confusion_matrix': cm,
            'history': history_dict
        }



    def _make_LSTM(self, train_X, train_Y, test_X=None, test_Y=None, progress_callback=None):
        """Build and train LSTM model with improved handling of imbalanced data
        
        Parameters:
        -----------
        train_X, train_Y: Training data
        test_X, test_Y: Test data
        Returns:
        --------
        model, history, pred_final
        """
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Optional: control GPU exposure
        if progress_callback is None:
            progress_callback = lambda percent, message: None
        
        progress_callback(73, "Building and compiling LSTM model")
        

        model = Sequential(self.layers)
        # print(model.summary())
        # learning rate schedule

        lr_schedule = ExponentialDecay(
            self.initial_lr,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate
        )

        optimizer = Adam(learning_rate=lr_schedule, clipnorm=self.clipnorm)
        # print("DECAY RATE!:", self.decay_rate)
        
        model.compile(
            optimizer=optimizer,
            loss=MaskedSparseCategoricalCrossentropy(unlabeled_value=self.UNLABELED_VALUE),
            metrics=[MaskedAccuracy(unlabeled_value=self.UNLABELED_VALUE)]
        )

        sw = np.where(train_Y != self.UNLABELED_VALUE, 1.0, 0.0)
        

        validation_method = self.config.analysis.lstm.validation_method
        test_size = self.config.analysis.lstm.validation_size
        
        progress_callback(75, f"Performing {validation_method} validation split")
        
        # Create sequence IDs if needed for sequence-aware methods
        if validation_method != "random":
            sequence_ids = get_sequence_ids(train_X, train_Y)
        
        # Perform the validation split with original labels (including UNLABELED_VALUE)
        if validation_method == "sequence_aware":
            x_train, x_valid, y_train, y_valid, sw_train, sw_valid = sequence_aware_validation_split(
                train_X, train_Y, sw, sequence_ids, self.UNLABELED_VALUE, test_size
            )
        elif validation_method == "interleaved":
            x_train, x_valid, y_train, y_valid, sw_train, sw_valid = interleaved_validation_split(
                train_X, train_Y, sw, sequence_ids, self.UNLABELED_VALUE
            )
        elif validation_method == "balanced_class":
            x_train, x_valid, y_train, y_valid, sw_train, sw_valid = balanced_class_validation_split(
                train_X, train_Y, sw, sequence_ids, self.UNLABELED_VALUE, test_size
            )
        elif validation_method == "stratified":
            x_train, x_valid, y_train, y_valid, sw_train, sw_valid = stratified_sequence_split(
                train_X, train_Y, sw, sequence_ids, self.UNLABELED_VALUE, test_size
            )
        else:  # Default to random
            x_train, x_valid, y_train, y_valid, sw_train, sw_valid = random_validation_split(
                train_X, train_Y, sw, test_size, self.seed
            )
        
        # # Now convert the unlabeled values to 0 for model training
        # y_train_clipped = np.where(y_train == self.UNLABELED_VALUE, 0, y_train)
        # y_valid_clipped = np.where(y_valid == self.UNLABELED_VALUE, 0, y_valid)
        

        # labeled_metrics_callback = LabeledDataMetricsCallback(
        #     validation_data=(x_valid, y_valid),
        #     unlabeled_value=self.UNLABELED_VALUE
        # )

        # Train the model
        progress_callback(80, "Training LSTM model")
        history = model.fit(
            x_train,
            y_train,  # Use clipped version for training
            validation_data=(x_valid, y_valid),  # Use clipped version for validation
            # sample_weight=sw_train,
            epochs=self.config.analysis.lstm.epochs,
            batch_size=self.batch_size,
            callbacks=[
                EarlyStopping(
                    monitor='val_masked_accuracy',
                    patience=self.patience,
                    restore_best_weights=True,
                    min_delta=self.min_delta,
                    mode='max'
                ),
            ],
            verbose=0,
        )

        if (test_X is not None) and (test_Y is not None):

            # Generate predictions
            progress_callback(90, "Generating predictions")
            pred_y = model.predict(test_X, verbose=0)
            pred_y_classes = np.argmax(pred_y, axis=-1)  # Use axis=1 since we flattened
            progress_callback(100, "LSTM model training complete")
            # return model, history, pred_y_classes
            return model, history, pred_y_classes

        else:
            return model, history



    def _plot_single_history(self, history):
        """Plot training history for a single model"""
        plt.figure(figsize=(12, 4))
        
        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], 'b-', label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], 'r-', label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot Accuracy - prefer labeled_accuracy if available
        plt.subplot(1, 2, 2)
        plt.plot(history.history['masked_accuracy'], 'b-', alpha=0.5, label='Training Accuracy')
        plt.plot(history.history['val_masked_accuracy'], 'r-', label='Validation Accuracy')
            
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.savefig(self.model_path / "lstm_global_history.png", dpi=300)
        plt.close()
        
        # Print final metrics
        print("\nTraining Summary:")
        print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
        

        print(f"Final Loss: {history.history['loss'][-1]:.4f}")
        print(f"Final Training Accuracy (labeled data only): {history.history['masked_accuracy'][-1]:.4f}")

        print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
        print(f"Final Validation Accuracy: {history.history['val_masked_accuracy'][-1]:.4f}")


    # Add a method to calculate overall metrics
    def _calculate_overall_metrics(self, all_predictions, all_actual):
        """Calculate overall metrics across all CV folds"""
        # Combine all predictions and actual values
        flat_pred = np.concatenate([p.flatten() for p in all_predictions])
        flat_actual = np.concatenate([a.flatten() for a in all_actual])
        
        # Only evaluate on valid data points
        valid_mask = flat_actual != self.UNLABELED_VALUE
        valid_pred = flat_pred[valid_mask]
        valid_actual = flat_actual[valid_mask]
        
        # Calculate overall accuracy
        self.last_accuracy = np.mean(valid_pred == valid_actual)
        
        # Calculate F1 score
        try:
            self.last_f1_score = f1_score(
                valid_actual, 
                valid_pred,
                average='micro',
                labels=sorted([v for k,v in self.activity_map.items() if k is not np.nan])
            )
        except Exception as e:
            print(f"Error calculating F1 score: {e}")
            self.last_f1_score = 0
        
        # Calculate class-wise accuracy
        self.last_class_accuracies = {}
        for class_name, class_id in self.activity_map.items():
            if class_id == self.UNLABELED_VALUE:
                continue  # Skip NaN class
                
            class_mask = valid_actual == class_id
            if np.sum(class_mask) > 0:
                class_acc = np.mean(valid_pred[class_mask] == class_id)
                self.last_class_accuracies[class_name] = float(class_acc)


    def _summarize_results(self, all_predictions, all_actual, all_histories):
        """Summarize and save all model results"""
        # Combine all predictions and actual values
        flat_pred = np.concatenate([p.flatten() for p in all_predictions])
        flat_actual = np.concatenate([a.flatten() for a in all_actual])

        # Only evaluate on valid data points
        valid_mask = flat_actual != self.UNLABELED_VALUE
        valid_pred = flat_pred[valid_mask]
        valid_actual = flat_actual[valid_mask]
        
        # Calculate overall accuracy
        accuracy = np.mean(valid_pred == valid_actual)
        print(f"\n===== OVERALL RESULTS =====")
        print(f"Total accuracy: {accuracy:.4f}")
        
        # Display class-wise accuracy
        for class_name, class_id in self.activity_map.items():
            if class_id == self.UNLABELED_VALUE:
                continue  # Skip NaN class
                
            class_mask = valid_actual == class_id
            if np.sum(class_mask) > 0:
                class_acc = np.mean(valid_pred[class_mask] == class_id)
                print(f"{class_name} accuracy: {class_acc:.4f} (n={np.sum(class_mask)})")
        
        # Save confusion matrix
        # try:

        cm = confusion_matrix(
            valid_actual,
            valid_pred,
            labels=sorted([v for k,v in self.activity_map.items() if k is not np.nan])
        )

        print(classification_report(valid_actual, valid_pred,labels=self.config.analysis.lstm.states, digits=3,zero_division=1))

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            xticklabels=[self.inv_activity_map[i] for i in sorted([v for k,v in self.activity_map.items() if k is not np.nan])],
            yticklabels=[self.inv_activity_map[i] for i in sorted([v for k,v in self.activity_map.items() if k is not np.nan])]
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # Save the figure
        plt.savefig(self.cv_path / "lstm_confusion_matrix.png", dpi=300)
        plt.close()

        # Plot training history
        self._plot_training_histories(all_histories)
            
        # except Exception as e:
        #     print(f"Error generating visualization: {e}")
        
        # Save detailed metrics to file
        with open(self.cv_path / "lstm_results.txt", "w") as f:

            f.write(f"Parameter values\n")
            for param, param_val in self._get_params().items():
                f.write(f"{param}: {param_val}\n")

            f.write(f"\nOverall accuracy: {accuracy:.4f}\n\n")
            f.write("Class-wise accuracy:\n")
            for class_name, class_id in self.activity_map.items():
                if class_id == self.UNLABELED_VALUE:
                    continue

                class_mask = valid_actual == class_id
                if np.sum(class_mask) > 0:
                    class_acc = np.mean(valid_pred[class_mask] == class_id)
                    f.write(f"{class_name}: {class_acc:.4f} (n={np.sum(class_mask)})\n")


        # Save detailed metrics to file
        with open(self.cv_path / "lstm_cv_preds.csv", "w") as f:
            f.write(f"Actual,Predicted\n")
            for i in range(len(valid_pred)):
                f.write(f"{valid_actual[i]}, {valid_pred[i]}\n")




    def _plot_training_histories(self, histories):
        """Plot training histories across all folds with mean and standard deviation"""
        # Find max length of histories
        max_epochs = max(len(h.history['loss']) for h in histories)
        epochs = range(1, max_epochs + 1)
        
        # Create figure with two subplots
        fig, (loss_plt, acc_plt) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), layout='constrained')
        
        # Metrics to plot in each subplot
        loss_metrics = [
            ('loss', 'b', 'Train Loss'),
            ('val_loss', 'r', 'Val Loss')
        ]

        acc_metrics = [
            ('masked_accuracy', 'b', 'Train Accuracy'),
            ('val_masked_accuracy', 'r', 'Val Accuracy')
            # ('val_labeled_accuracy', 'r', 'Val Accuracy')
        ]
        
        # Plot individual histories and calculate means/std for loss subplot
        for loss_key, color, label in loss_metrics:
            # Initialize array for this metric
            metric_values = np.zeros((len(histories), max_epochs))
            # print(loss_key)

            # Plot individual histories with low alpha
            for i, h in enumerate(histories):

                curr_len = len(h.history[loss_key])
                loss_plt.plot(epochs[:curr_len], h.history[loss_key][:curr_len],
                            f'{color}-', alpha=0.1)
                #
                # print("Index", i)
                # print("Values", h.history[loss_key])
                # print("Length", len(h.history[loss_key]))
                # print("Last Value", h.history[loss_key][-1])
                # Fill array for mean/std calculation
                metric_values[i, :curr_len] = h.history[loss_key][:curr_len]
                metric_values[i, curr_len:] = h.history[loss_key][-1]  # Pad with last value
            
            # Plot mean line
            mean_values = np.mean(metric_values, axis=0)
            std_values = np.std(metric_values, axis=0)
            loss_plt.plot(epochs, mean_values, f'{color}-', label=f'{label}')
            
            # Plot standard deviation band
            loss_plt.fill_between(epochs, mean_values - std_values, mean_values + std_values,
                                color=color, alpha=0.2)
        

        # print("STARTING ACC")
        # Plot individual histories and calculate means/std for accuracy subplot
        for acc_key, color, label in acc_metrics:
            # Initialize array for this metric
            metric_values = np.zeros((len(histories), max_epochs))
            
            print(acc_key)
            # Plot individual histories with low alpha
            for i, h in enumerate(histories):


                if acc_key not in h.history:
                    print(f"Warning: '{acc_key}' missing from history {i}")
                    continue



                curr_len = len(h.history[acc_key])
                acc_plt.plot(epochs[:curr_len], h.history[acc_key][:curr_len],
                            f'{color}-', alpha=0.1)
                
                # Fill array for mean/std calculation
                metric_values[i, :curr_len] = h.history[acc_key][:curr_len]
                # print("!!!!!!!!!!!",curr_len, metric_values[i])
                # print(h.history.keys())

                metric_values[i, curr_len:] = h.history[acc_key][-1]  # Pad with last value
            
            # Plot mean line
            mean_values = np.mean(metric_values, axis=0)
            std_values = np.std(metric_values, axis=0)
            acc_plt.plot(epochs, mean_values, f'{color}-', label=f'{label}')
            
            # Plot standard deviation band
            acc_plt.fill_between(epochs, mean_values - std_values, mean_values + std_values,
                            color=color, alpha=0.2)
        
        # Set titles and labels
        loss_plt.set_title('Training Loss')
        loss_plt.set_xlabel('Epoch')
        loss_plt.set_ylabel('Loss')
        loss_plt.legend()
        
        acc_plt.set_title('Training Accuracy')
        acc_plt.set_xlabel('Epoch')
        acc_plt.set_ylabel('Accuracy')
        acc_plt.legend()
        
        # plt.tight_layout()
        plt.savefig(self.cv_path / "lstm_training_history.png", dpi=300)
        plt.close()