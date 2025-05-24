from datetime import datetime
import itertools
import json
from pathlib import Path
import logging
# from dataclasses import dataclass
import random
import time
import traceback
from typing import Dict, List
import pandas as pd
import numpy as np

import keras
# from keras.models import Sequential
# from keras.layers import Dense, LSTM, Dropout, Masking


from sklearn.model_selection import train_test_split 
import tensorflow as tf
# from sklearn.model_selection import GroupKFold
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Masking, Input, BatchNormalization, Conv1D, TimeDistributed, Flatten, GlobalAveragePooling1D, Attention, Add, LayerNormalization, MultiHeadAttention
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.random import set_seed
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam



from keras import metrics
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import classification_report
import matplotlib
# Use a non-interactive backend that works well with multiprocessing
matplotlib.use('Agg')  # This must be done before importing pyplot

import matplotlib.pyplot as plt
import seaborn as sns

from cowstudyapp.config import ConfigManager, LSTMConfig, AnalysisConfig
from cowstudyapp.utils import from_posix, from_posix_col

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import multiprocessing as mp
from functools import partial

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective
from skopt import dump, load
import os

class BayesianOptSearch:
    """Bayesian optimization for hyperparameter tuning using Gaussian Processes"""
    
    def __init__(self, config):
        self.config = config
        self.results = []
        self.lstm_model = None
        self.sequences = None
        self.df = None

        # self.output_dir = "data/analysis_results"
        self.output_dir = self.config.analysis.output_dir
        
        # Define the search space
        self.space = [
            Integer(10, 288, name='max_length'),
            Integer(4, 32, name='batch_size'),
            Real(1e-5, 1e-3, "log-uniform", name='initial_lr'),
            Integer(500, 10000, name='decay_steps'),
            Real(0.4, 0.95, name='decay_rate'),
            Real(0.1, 2.0, name='clipnorm'),
            Integer(10, 50, name='patience'),
            Real(1e-7, 1e-5, "log-uniform", name='min_delta'),
            Real(1e-7, 1e-4, "log-uniform", name='reg_val'),
            # Optional: Categorical parameters
            # Categorical(['adam', 'rmsprop'], name='optimizer'),
            # Categorical(['relu', 'tanh'], name='activation'),
        ]
        
        # Number of calls to make to the objective function
        self.n_calls = config.analysis.lstm.bayes_opt_n_calls
        
        # Path to save/load optimization results
        io_type = 'ops' if self.config.analysis.lstm.ops else 'opo'
        self.results_path = os.path.join(self.output_dir, io_type, "bayes_opt_results.pkl")
        
    def run_search(self, lstm_model, sequences, df):
        """Run Bayesian optimization search"""
        self.lstm_model = lstm_model
        self.sequences = sequences
        self.df = df


        # Instead of trying to use a decorator, use a simple function
        def objective(x):
            # Convert the parameter vector to a dictionary
            params = {dim.name: x[i] for i, dim in enumerate(self.space)}
            return self._objective(params)

        # Check if we should resume from previous run
        if os.path.exists(self.results_path) and self.config.analysis.lstm.bayes_opt_resume:
            print(f"Resuming optimization from {self.results_path}")
            previous_result = load(self.results_path)
            
            # Call optimization with previous result as starting point
            result = gp_minimize(
                objective,
                self.space,
                n_calls=self.n_calls,
                x0=previous_result.x_iters,  # Use previous iterations
                y0=previous_result.func_vals,  # Use previous function values
                random_state=self.config.analysis.random_seed,
                n_random_starts=0,  # No random starts since we're using previous results
                verbose=True,
                callback=self._on_step
            )
        else:
            # Start fresh optimization
            result = gp_minimize(
                objective,
                self.space,
                n_calls=self.n_calls,
                random_state=self.config.analysis.random_seed,
                verbose=True,
                callback=self._on_step
            )
        
        # Save the final results
        dump(result, self.results_path, store_objective=False)
        
        # Get best parameters
        best_params = {}
        for key, val in zip([dim.name for dim in self.space], result.x):

            if isinstance(val, (np.integer, np.int32, np.int64)):
                best_params[key] = int(val)
            elif isinstance(val, (np.floating, np.float32, np.float64)):
                best_params[key] = float(val)
            else:
                best_params[key] = val

        # best_params = dict(zip([dim.name for dim in self.space], result.x))
        print("\nBest parameters found:")
        print(json.dumps(best_params, indent=2))
        print(f"Best F1 score: {-result.fun:.4f}")  # Negative because we're minimizing
        
        # Generate and save plots
        self._save_optimization_plots(result)
        
        return best_params


    def _objective(self, params):
        """Objective function to minimize (negative F1 score)"""
        # Convert any NumPy types to native Python types for JSON serialization
        printable_params = {}
        for key, value in params.items():
            if isinstance(value, (np.integer, np.int32, np.int64)):
                printable_params[key] = int(value)
            elif isinstance(value, (np.floating, np.float32, np.float64)):
                printable_params[key] = float(value)
            else:
                printable_params[key] = value
        
        print(f"\n{'='*20} Testing parameters: {'='*20}")
        print(json.dumps(printable_params, indent=2))
        
        # Set the parameters on the model (using original values)
        self._set_params(params)
        
        # Run a small LOOCV with current parameters
        start_time = time.time()
        
        # Use fewer CV splits for speed
        n_splits = 3 if self.config.analysis.lstm.bayes_opt_fast_eval else 7
        
        if self.config.analysis.mode == "LOOCV":
            _, _ = self.lstm_model.do_loocv(
                sequences=self.sequences, 
                df=self.df, 
                n=n_splits, 
                compute_metrics_only=self.config.analysis.lstm.bayes_opt_fast_eval
            )
        else:  # PRODUCT mode
            _, _ = self.lstm_model.dont_do_looc(
                sequences=self.sequences, 
                df=self.df
            )
            
        elapsed_time = time.time() - start_time
        
        # Get the negative F1 score (we want to maximize F1, but gp_minimize minimizes)
        score = -self.lstm_model.last_f1_score
        
        # Store result with Python native types
        result = {
            'params': printable_params,
            'accuracy': float(self.lstm_model.last_accuracy),
            'f1_score': float(self.lstm_model.last_f1_score),
            'class_accuracies': {k: float(v) for k, v in self.lstm_model.last_class_accuracies.items()},
            'elapsed_time': float(elapsed_time)
        }
        self.results.append(result)
        
        print(f"F1 Score: {self.lstm_model.last_f1_score:.4f}, Accuracy: {self.lstm_model.last_accuracy:.4f}")
        print(f"Class accuracies: {self.lstm_model.last_class_accuracies}")
        print(f"Evaluation time: {elapsed_time:.1f} seconds")
        
        return score

    def _on_step(self, res):
        """Callback function called after each iteration"""
        # Save intermediate results
        try:
            dump(res, self.results_path, store_objective=False)
        except Exception as e:
            print(f"Warning: Could not save intermediate results: {e}")
        
        # Print current best with proper type conversion
        print(f"Current best F1 score: {-res.fun:.4f} with parameters: ")
        best_params = {}
        for i, dim in enumerate(self.space):
            value = res.x[i]
            if isinstance(value, (np.integer, np.int32, np.int64)):
                best_params[dim.name] = int(value)
            elif isinstance(value, (np.floating, np.float32, np.float64)):
                best_params[dim.name] = float(value) 
            else:
                best_params[dim.name] = value
        
        try:
            print(json.dumps(best_params, indent=2))
        except TypeError as e:
            print(f"Warning: Could not print parameters as JSON: {e}")
            print(f"Parameters: {best_params}")



    def _set_params(self, params):
        """Set hyperparameters on the LSTM model"""
        # Assign parameters to the model
        for param_name, param_value in params.items():
            setattr(self.lstm_model, param_name, param_value)
            
            # Special case for max_length which also needs to update sequence_length
            if param_name == 'max_length':
                self.lstm_model.sequence_length = param_value
                self.lstm_model.config.analysis.lstm.max_length = param_value
    


    def _save_optimization_plots(self, result):
        """Save optimization visualization plots"""
        output_dir = Path(self.output_dir)
        
        # Plot convergence
        plt.figure(figsize=(10, 6))
        plot_convergence(result)
        io_type = 'ops' if self.config.analysis.lstm.ops else 'opo'
        plt.savefig(output_dir / io_type / "bayes_opt_convergence.png", dpi=300)
        plt.close()
        
        # Plot individual parameter effects (partial dependence)
        fig, ax = plt.subplots(3, 3, figsize=(15, 12))
        plot_objective(result, dimensions=range(len(self.space)), n_points=10, ax=ax.ravel())
        plt.tight_layout()
        plt.savefig(output_dir / "bayes_opt_parameters.png", dpi=300)
        plt.close()

class HyperparamSearch:
    """Hyperparameter search for LSTM models"""
    
    def __init__(self, config):
        self.config = config
        self.results = []
        
        # Define hyperparameter grid
        self.param_grid = {
            'max_length': [10, 20, 30, 40] if not config.analysis.lstm.ops else [20],
            'epochs': [1000],  # Keep fixed to save time, rely on early stopping
            'batch_size': [8, 16],
            'initial_lr': [1e-4, 5e-4],
            'decay_steps': [1000, 5000],
            'decay_rate': [0.75, 0.85],
            'clipnorm': [0.5, 1.0],
            'patience': [10, 20, 40],
            'min_delta': [1e-6],
            'reg_val': [1e-5, 1e-6],
            # Add more parameters as needed
        }
        
        # For quick testing, use a small subset
        if config.analysis.lstm.hyperparams_sample:
            # Randomly sample a smaller grid for testing
            self.param_combinations = self._sample_params(400)  # Test 5 random combinations
        else:
            # Generate all combinations (warning: could be a lot!)
            keys = list(self.param_grid.keys())
            values = list(itertools.product(*[self.param_grid[key] for key in keys]))
            self.param_combinations = [dict(zip(keys, v)) for v in values]
            print(f"Generated {len(self.param_combinations)} parameter combinations")

    def _sample_params(self, n_samples):
        """Randomly sample n_samples parameter combinations"""
        sampled_params = []
        for _ in range(n_samples):
            params = {}
            for key, values in self.param_grid.items():
                params[key] = random.choice(values)
            sampled_params.append(params)
        return sampled_params
        
    def run_search(self, lstm_model, sequences, df):
        """Run hyperparameter search"""
        for i, params in enumerate(self.param_combinations):
            print(f"\n{'='*30} Testing hyperparameter set {i+1}/{len(self.param_combinations)} {'='*30}")
            print(json.dumps(params, indent=2))
            
            # Set the parameters
            self._set_params(lstm_model, params)
            
            # Run LOOCV with current parameters
            start_time = time.time()
            if self.config.analysis.mode == "LOOCV":
                models, histories = lstm_model.do_loocv(sequences=sequences, df=df, n=6)  # Use fewer splits for speed
            else:  # PRODUCT mode
                models, histories = lstm_model.dont_do_loocv(sequences=sequences, df=df)
                
            elapsed_time = time.time() - start_time
            
            # Store results
            result = {
                'params': params,
                'accuracy': lstm_model.last_accuracy,
                'f1_score': lstm_model.last_f1_score,
                'class_accuracies': lstm_model.last_class_accuracies,
                'elapsed_time': elapsed_time
            }
            self.results.append(result)
            
            # Save intermediate results
            self._save_results()
            
        # Sort and print final results
        self._print_best_results()
        return self.results
        
    def _set_params(self, lstm_model, params):
        """Set hyperparameters on the LSTM model"""
        # Assign parameters to the model
        lstm_model.config.analysis.lstm.max_length = params['max_length']
        lstm_model.sequence_length = params['max_length']
        lstm_model.config.analysis.lstm.epochs = params['epochs']
        lstm_model.batch_size = params['batch_size']
        lstm_model.initial_lr = params['initial_lr']
        lstm_model.decay_steps = params['decay_steps']
        lstm_model.decay_rate = params['decay_rate']
        lstm_model.clipnorm = params['clipnorm']
        lstm_model.patience = params['patience']
        lstm_model.min_delta = params['min_delta']
        lstm_model.reg_val = params['reg_val']
        
    def _save_results(self):
        """Save current results to file"""
        output_dir = Path(self.output_dir)

        io_type = 'ops' if self.config.analysis.lstm.ops else 'opo'
        results_file = output_dir / io_type / "hyperparam_search_results.json"
        
        # Sort results by validation accuracy
        sorted_results = sorted(self.results, key=lambda x: x['f1_score'], reverse=True)
        
        with open(results_file, 'w') as f:
            json.dump({
                'results': sorted_results,
                'best_params': sorted_results[0]['params'] if sorted_results else None
            }, f, indent=2)
            
    def _print_best_results(self):
        """Print the best hyperparameter combinations"""
        # Sort by validation accuracy
        sorted_results = sorted(self.results, key=lambda x: x['f1_score'], reverse=True)
        
        print("\n===== HYPERPARAMETER SEARCH RESULTS =====")
        print(f"Total combinations tested: {len(self.results)}")
        
        if sorted_results:
            print("\nTop 3 configurations:")
            for i, result in enumerate(sorted_results[:3]):
                print(f"\n{i+1}. F1 Score: {result['f1_score']:.4f}, Accuracy: {result['accuracy']:.4f}")
                print(f"   Class accuracies: {result['class_accuracies']}")
                print(f"   Time: {result['elapsed_time']:.1f} seconds")
                print("   Parameters:")
                for k, v in result['params'].items():
                    print(f"     {k}: {v}")

@keras.saving.register_keras_serializable()
class MaskedConv1D(tf.keras.layers.Conv1D):
    def compute_mask(self, inputs, mask=None):
        # preserve the time-step mask (True = valid) 
        return mask
    def call(self, inputs, mask=None):
        out = super().call(inputs)
        if mask is not None:
            # zero out conv outputs at masked positions
            out *= tf.cast(tf.expand_dims(mask, -1), out.dtype)
        return out

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
        # {"Grazing": 0, "Resting":1, "Traveling": 2, np.nan:-1}
        self.UNLABELED_VALUE = -1

        self.inv_activity_map = {self.activity_map[k]: k for k in self.activity_map.keys()}

        self.nfeatures = len(self.config.analysis.lstm.features)
        self.nclasses = len(self.config.analysis.lstm.states)

        # if self.config.analysis.lstm.max_length == 'daily':
        #     max_length = 288
        # else:
        #     max_length = self.config.analysis.lstm.max_length 
        # self.sequence_length = max_length


        # # Store hyperparameters that can be tuned
        # self.batch_size = 8
        # self.initial_lr = 1e-4
        # self.decay_steps = 5000
        # self.decay_rate = 0.75
        # self.clipnorm = 1.0
        # self.patience = 25
        # self.min_delta = 1e-6
        # self.reg_val = 1e-5

        self.dropout_rate = 0.1
        lstm_cfg = self.config.analysis.lstm
        # Create output directory for models
        io_type = 'ops' if lstm_cfg.ops else 'opo'
        # res_type = self.config.analysis.cv_results if self.config.analysis.mode == "LOOCV" else self.config.analysis.models
        # self.output_dir = Path(res_type) / "LSTM" / io_type


        self.cv_path = self.config.analysis.cv_results / "LSTM" / io_type
        self.pred_path = self.config.analysis.predictions / "LSTM" / io_type
        self.model_path = self.config.analysis.models / "LSTM" / io_type

        for p in [self.cv_path, self.pred_path, self.model_path]:
            p.mkdir(parents=True, exist_ok=True)



        # output_dir = Path(self.output_dir) / io_type / "lstm_models" 
        # self.output_dir.mkdir(exist_ok=True, parents=True)


        ############ Pull these from lstm_cfg
        ############ Return stuff! (LOOCV / prediction results)


        
        # OPO Best Parameters (n=30)
        # self.sequence_length = 70
        # self.batch_size = 13
        # self.initial_lr = 0.0008013770340749126
        # self.decay_steps = 500
        # self.decay_rate = 0.510164871944301
        # self.clipnorm = 0.7306814672664718
        # self.patience = 50
        # self.min_delta = 8.991637285631857e-07
        # self.reg_val = 3.415197283259112e-05



        # OPS
        # self.max_length = 160
        # self.batch_size = 22
        # self.initial_lr = 0.001
        # self.decay_steps = 10000
        # self.decay_rate = 0.4
        # self.clipnorm = 1.9493856438736408
        # self.patience = 13
        # self.min_delta = 1.6850428524897937e-07
        # self.reg_val = 1e-07


        self.sequence_length = lstm_cfg.max_length
        self.max_time_gap = lstm_cfg.max_time_gap
        self.epochs = lstm_cfg.epochs
        self.cows_per_cv_fold = lstm_cfg.cows_per_cv_fold

        self.batch_size = lstm_cfg.batch_size
        self.initial_lr = lstm_cfg.initial_lr
        self.decay_steps = lstm_cfg.decay_steps
        self.decay_rate = lstm_cfg.decay_rate
        self.clipnorm = lstm_cfg.clipnorm
        self.patience = lstm_cfg.patience
        self.min_delta = lstm_cfg.min_delta
        self.reg_val = lstm_cfg.reg_val


        set_seed(self.config.analysis.random_seed)
        np.random.seed(self.config.analysis.random_seed)
        random.seed(self.config.analysis.random_seed)
        self.masking_val = -9999

        self.layers = [
            Input((self.sequence_length, self.nfeatures)),
            Masking(mask_value=self.masking_val),      
            ]


    def run_LSTM(self, progress_callback=None):
        
        if progress_callback is None:
            progress_callback = lambda percent, message: print(f"{percent}%: {message}")
        progress_callback(7, "Preparing dataset for LSTM analysis")
        untransformed = self._get_target_dataset(add_step_and_angle=True)

        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(untransformed[['step']].head(20))
        
        required_cols: List[str] = self.config.analysis.lstm.features + ["posix_time", "date", 'device_id', 'activity']
        progress_callback(10, f"Normalizing features: {', '.join(self.config.analysis.lstm.features)}")
        transformed = untransformed[required_cols].copy()
        transformed = self._normalize_features(transformed, self.config.analysis.lstm.features)
        # df = df[required_cols].copy()

        # Build sequences based on configuration
        if self.config.analysis.lstm.ops:
            progress_callback(20, "Building sequences for one-per-sample (OPS) analysis")
            sequences = self._build_sequences_ops(transformed, progress_callback)
            progress_callback(40, "Building OPS model architecture")
            self._build_ops_architecture()
        else:
            progress_callback(20, "Building sequences for one-per-observation (OPO) analysis")
            sequences = self._build_sequences_opo(transformed, progress_callback)
            progress_callback(40, "Building OPO model architecture")
            self._build_opo_architecture()
        # Check if we're doing Bayesian optimization
        if self.config.analysis.lstm.bayes_opt:
            bayes_opt = BayesianOptSearch(self.config)
            best_params = bayes_opt.run_search(self, sequences, transformed)
            
            # Set the best parameters
            for key, value in best_params.items():
                setattr(self, key, value)
                if key == 'max_length':
                    self.sequence_length = value
                    self.config.analysis.lstm.max_length = value

            # Rebuild model architecture with optimized parameters
            self.layers = [
                Input((self.sequence_length, self.nfeatures)),
                Masking(mask_value=self.masking_val),
            ]
            if self.config.analysis.lstm.ops:
                self._build_ops_architecture()
            else:
                self._build_opo_architecture()

        # print("run3")
        # Check if we're doing hyperparameter search
        if self.config.analysis.lstm.hyperparams_search:
            search = HyperparamSearch(self.config)
            results = search.run_search(self, sequences, transformed)
            # Use the best parameters
            best_params = max(results, key=lambda x: x['f1_score'])['params']
            print("\nUsing best parameters for final model:")
            print(json.dumps(best_params, indent=2))
            # Set the best parameters
            for key, value in best_params.items():
                setattr(self, key, value)
                if key == 'max_length':
                    self.sequence_length = value
                    self.config.analysis.lstm.max_length = value


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

    def _build_ops_architecture(self):
        """Build model architecture for one-per-observation"""
        self.layers.extend([
            LSTM(32, 
                return_sequences=False,
                recurrent_dropout=self.dropout_rate,
                dropout=self.dropout_rate,
                kernel_regularizer=L2(self.reg_val),
                recurrent_regularizer=L2(self.reg_val)
                ),
            # LSTM(128, 
            #     return_sequences=False,
            #     recurrent_dropout=self.dropout_rate,
            #     dropout=self.dropout_rate,
            #     kernel_regularizer=L2(self.reg_val),
            #     recurrent_regularizer=L2(self.reg_val)
            #     ),
            Dense(32, activation='relu'),
            Dropout(self.dropout_rate),  
            Dense(self.nclasses, activation='softmax')
        ])
    

    def _build_opo_architecture(self):
        """Build model architecture for one-per-sequence"""
        self.layers.extend([
            # LSTM(256, 
            #     return_sequences=True,
            #     recurrent_dropout=self.dropout_rate,
            #     dropout=self.dropout_rate,
            #     kernel_regularizer=L2(self.reg_val),
            #     recurrent_regularizer=L2(self.reg_val)
            #     ),
            # MaskedConv1D(filters=32, kernel_size=35, padding='same', activation='relu'),
            # MaskedConv1D(filters=32, kernel_size=3, padding='same', activation='relu')(m),

            LSTM(32, 
                return_sequences=True,
                recurrent_dropout=self.dropout_rate,
                dropout=self.dropout_rate,
                kernel_regularizer=L2(self.reg_val),
                recurrent_regularizer=L2(self.reg_val)
                ),


            # LSTM(64, 
            #     return_sequences=True,
            #     recurrent_dropout=self.dropout_rate,
            #     dropout=self.dropout_rate,
            #     kernel_regularizer=L2(self.reg_val),
            #     recurrent_regularizer=L2(self.reg_val)
            #     ),
            # Time distributed layers
            TimeDistributed(Dense(32, activation='relu')),
            TimeDistributed(Dropout(self.dropout_rate)),
            TimeDistributed(Dense(self.nclasses, activation='softmax'))
        ])


    def _get_target_dataset(self, add_step_and_angle=True):
        df = pd.read_csv(self.config.analysis.target_dataset)

        print(df.activity.unique())
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









    # def _build_sequences_opo(self, df:pd.DataFrame):
    #     max_length = self.sequence_length
    #     features = self.config.analysis.lstm.features
    #     step_size = self.config.analysis.gps_sample_interval//60  # Sample interval in minutes

    #     minutes_in_day = 24 * 60
    #     expected_records = minutes_in_day // step_size
    #     # self.sequence_length = expected_records

    #     # df['activity'].fillna(self.UNLABELED_VALUE,inplace=True)
    #     lost_hour = (60//step_size)

    #     Cow_Date_Key = []
    #     X = []
    #     Y = []

    #     if (max_length>=276) & (max_length < 288):
    #         print("WARNING: Daylight savings had not been accounted for.")

    #     if max_length == 288:
    #         print("WARNING: Daylight is only account for when an hour is lost, not gained. (If the data are from spring you are fine).")
    #         for device_id, device_data in df.groupby("device_id"):
    #             # self.data_map[device_id] = {}
    #             for date, day_data in device_data.groupby("date"):
    #                 if len(day_data) < expected_records * 0.5:  # Skip days with too few records
    #                     # print(f"Skipping day with insufficient data: {date}, {len(day_data)}/{expected_records}")
    #                     continue
                    
    #                 # Extract features, replace NaN with masking value
    #                 seq :pd.DataFrame = day_data[features].copy().reset_index(drop=True) #.fillna(self.masking_val)
    #                 labels :pd.DataFrame = day_data[['activity']].copy().reset_index(drop=True)

    #                 labels = labels.map(lambda x: self.activity_map.get(x, -1))

    #                 if len(seq) == expected_records - lost_hour:
    #                     # print(f"Missing data due to timezone shift. Inserting {lost_hour} values to the end of the sequence.")
    #                     masking_df = pd.DataFrame({
    #                         col: [np.nan] * lost_hour for col in seq.columns
    #                     })
    #                     seq = pd.concat([seq, masking_df], ignore_index=True)

    #                     nan_labels = pd.DataFrame({'activity': [np.nan] * lost_hour})
    #                     nan_labels = nan_labels.map(lambda x: self.activity_map.get(x, -1))
    #                     labels = pd.concat([labels, nan_labels], ignore_index=True)

    #                 else:
    #                     pass
    #                 seq.fillna(self.masking_val, inplace=True)
    #                 labels.fillna(self.UNLABELED_VALUE,inplace=True)
                    
    #                 # Add sequence to our dataset
    #                 Cow_Date_Key.append([device_id,date])
    #                 X.append(seq)
    #                 Y.append(labels.to_numpy().ravel())


    #     else: 
            

    #         # print(f"Input DataFrame shape: {df.shape}")
    #         # print(f"Input DataFrame columns: {df.columns.tolist()}")
    #         # print(f"Activity unique values: {df['activity'].unique().tolist()}")
    #         # print(f"NaN values in activity column: {df['activity'].isna().sum()}")
            
    #         # # Debug our activity_map
    #         # print(f"Activity map: {self.activity_map}")
    #         # print(f"UNLABELED_VALUE: {self.UNLABELED_VALUE}")

    #         for device_id, data in df.groupby("device_id"):
    #             # sequences = []  # List of observation sequences
    #             # labels = []    # List of activity labels
                
    #             for i in range(max_length, len(data), max_length): # Head of the list
    #                 current_sequence = []
    #                 current_labels = []

    #                 for j in range(i-max_length, i):
    #                     row = data.iloc[j]
    #                     current_features = []
    #                     # Get feature values for current observation
    #                     for f in self.config.analysis.lstm.features:
    #                         if pd.isna(row[f]):
    #                             current_features.append(self.masking_val)
    #                         else:
    #                             current_features.append(float(row[f]))


    #                     current_label=self.activity_map.get(row['activity'], self.UNLABELED_VALUE)
    #                     current_labels.append(current_label)
    #                     # Add current observation to sequence
    #                     current_sequence.append(current_features)
    #                     # Cow_Date_Key.append([device_id,i+j])  
    #                 # if len(X) < 3:  # Just for the first few sequences
    #                 #     print(f"Sample sequence {len(X)+1}:")
    #                 #     print(f"  - Labels: {current_labels[:5]}...{current_labels[-5:]}")
    #                 #     print(f"  - Any valid labels: {any(l != self.UNLABELED_VALUE for l in current_labels)}")
    #                 #     print(f"  - Device ID: {device_id}, Sequence index: {i}")


    #                 X.append(current_sequence)
    #                 Y.append(current_labels)
    #                 Cow_Date_Key.append([device_id,i])

    #     # print(Cow_Date_Key)
    #     Cow_Date_Key = np.array(Cow_Date_Key)
    #     X = np.array(X, dtype=np.float32)  # Ensure float32 for X
    #     Y = np.array(Y, dtype=np.int32)    # Ensure int32 for Y


    #     # print(Cow_Date_Key.shape)
    #     # print(Cow_Date_Key[:2,:])
    #     # print(X.shape)
    #     # print(Y.shape)


    #     # print(Y[:5])
    #     # print(Cow_Date_Key)
    #     return {
    #         'Cow_Date_Key': Cow_Date_Key,
    #         'X': X,
    #         'Y': Y #.squeeze(axis=2),
    #     }

    #     # else:



    # def _build_sequences_opo(self, df:pd.DataFrame, progress_callback=None):
    #     """
    #     Build sequences for one-per-observation analysis.
    #     Includes memory optimization and progress reporting.
    #     """
    #     if progress_callback is None:
    #         progress_callback = lambda percent, message: None
            
    #     max_length = self.sequence_length
    #     features = self.config.analysis.lstm.features
    #     step_size = self.config.analysis.gps_sample_interval//60  # Sample interval in minutes

    #     minutes_in_day = 24 * 60
    #     expected_records = minutes_in_day // step_size
    #     lost_hour = (60//step_size)

    #     Cow_Date_Key = []
    #     X = []
    #     Y = []
        
    #     # Count total days for progress reporting
    #     total_days = 0
    #     for _, device_data in df.groupby("device_id"):
    #         total_days += device_data['date'].nunique()
        
    #     progress_callback(20, f"Building OPO sequences from {total_days} cow-days")
        
    #     days_processed = 0
        
    #     if max_length == 288:
    #         progress_callback(25, "Using daily sequence mode (288 samples per day)")
            
    #         for device_id, device_data in df.groupby("device_id"):
    #             device_data = device_data.sort_values('posix_time')
                
    #             for date, day_data in device_data.groupby("date"):
    #                 days_processed += 1
                    
    #                 # Progress update
    #                 progress_pct = 25 + int(30 * days_processed / total_days)
    #                 progress_callback(progress_pct, f"Processing day {days_processed}/{total_days}: Device {device_id}, Date {date}")
                    
    #                 if len(day_data) < expected_records * 0.5:  # Skip days with too few records
    #                     continue
                    
    #                 # Extract features, replace NaN with masking value
    #                 seq = day_data[features].copy().reset_index(drop=True)
    #                 labels = day_data[['activity']].copy().reset_index(drop=True)
                    
    #                 labels = labels.map(lambda x: self.activity_map.get(x, -1))

    #                 if len(seq) == expected_records - lost_hour:
    #                     # Handle daylight savings time issues
    #                     masking_df = pd.DataFrame({
    #                         col: [np.nan] * lost_hour for col in seq.columns
    #                     })
    #                     seq = pd.concat([seq, masking_df], ignore_index=True)

    #                     nan_labels = pd.DataFrame({'activity': [np.nan] * lost_hour})
    #                     nan_labels = nan_labels.map(lambda x: self.activity_map.get(x, -1))
    #                     labels = pd.concat([labels, nan_labels], ignore_index=True)

    #                 seq.fillna(self.masking_val, inplace=True)
    #                 labels.fillna(self.UNLABELED_VALUE, inplace=True)
                    
    #                 # Add sequence to our dataset
    #                 Cow_Date_Key.append([device_id, date])
    #                 X.append(seq.values)
    #                 Y.append(labels.to_numpy().ravel())
                    
    #                 # Force garbage collection periodically
    #                 if days_processed % 100 == 0:
    #                     import gc
    #                     gc.collect()
    #     else:
    #         progress_callback(25, f"Using custom sequence length: {max_length}")
            
    #         days_processed = 0
    #         sequence_count = 0
            
    #         for device_id, data in df.groupby("device_id"):
    #             data = data.sort_values('posix_time').reset_index(drop=True)
    #             days_processed += data['date'].nunique()
                
    #             # Progress update
    #             progress_pct = 25 + int(30 * days_processed / total_days)
    #             progress_callback(progress_pct, f"Processing device {device_id}: {data['date'].nunique()} days")
                
    #             # Process in sliding windows of size max_length
    #             for i in range(max_length, len(data), max_length // 2):  # 50% overlap for better coverage
    #                 current_sequence = []
    #                 current_labels = []
                    
    #                 sequence_start = max(0, i - max_length)
    #                 sequence_end = min(len(data), i)
                    
    #                 for j in range(sequence_start, sequence_end):
    #                     row = data.iloc[j]
    #                     current_features = []
                        
    #                     # Get feature values for current observation
    #                     for f in features:
    #                         if pd.isna(row[f]):
    #                             current_features.append(self.masking_val)
    #                         else:
    #                             current_features.append(float(row[f]))
                        
    #                     current_label = self.activity_map.get(row['activity'], self.UNLABELED_VALUE)
    #                     current_labels.append(current_label)
    #                     current_sequence.append(current_features)
                    
    #                 # Only add if we have at least 50% of max_length
    #                 if len(current_sequence) >= max_length / 2:
    #                     # Pad if needed
    #                     if len(current_sequence) < max_length:
    #                         padding_needed = max_length - len(current_sequence)
    #                         padding = [[self.masking_val] * len(features)] * padding_needed
    #                         current_sequence = padding + current_sequence
    #                         current_labels = [self.UNLABELED_VALUE] * padding_needed + current_labels
                        
    #                     sequence_count += 1
    #                     X.append(current_sequence)
    #                     Y.append(current_labels)
    #                     Cow_Date_Key.append([device_id, i])
                    
    #                 # Force garbage collection periodically
    #                 if sequence_count % 1000 == 0:
    #                     import gc
    #                     gc.collect()
        
    #     progress_callback(60, "Converting sequences to numpy arrays")
        
    #     # Convert to numpy arrays efficiently
    #     X_array = np.array(X, dtype=np.float32)
    #     Y_array = np.array(Y, dtype=np.int32)
    #     Key_array = np.array(Cow_Date_Key)
        
    #     progress_callback(80, f"Created {len(X_array)} sequences covering {days_processed} cow-days")
        
    #     return {
    #         'Cow_Date_Key': Key_array,
    #         'X': X_array,
    #         'Y': Y_array
    #     }


    def _build_sequences_opo(self, df: pd.DataFrame, progress_callback=None):
        """
        Build sequences for one-per-observation (OPO) analysis.
        Pre-calculates sequence counts and includes every observation exactly once.
        """
        if progress_callback is None:
            progress_callback = lambda percent, message: None
        
        max_length = self.sequence_length
        features = self.config.analysis.lstm.features
        num_features = len(features)
        
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
        df[features] = df[features].fillna(self.masking_val)
        
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
                    day_features = day_data[features].values
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
                device_features = device_data[features].values
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
        df[self.config.analysis.lstm.features] = df[self.config.analysis.lstm.features].fillna(self.masking_val)
        
        # Pre-allocate arrays for results - one row per observation
        cow_date_keys = np.zeros((total_rows, 2), dtype=np.int32)
        all_sequences = np.zeros((total_rows, self.sequence_length, len(self.config.analysis.lstm.features)), 
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
            feature_matrix = data[self.config.analysis.lstm.features].values
            
            # Activity labels
            activities = data['activity'].map(lambda x: self.activity_map.get(x, self.UNLABELED_VALUE)).values
            
            # Posix times for checking time gaps
            posix_times = data['posix_time'].values
            
            # Initialize empty sequence with masking values
            current_sequence = np.full((self.sequence_length, len(self.config.analysis.lstm.features)), 
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
                if current_sequence_len < self.sequence_length:
                    # Sequence not full yet, add at the end
                    current_sequence[self.sequence_length - current_sequence_len - 1] = feature_matrix[i]
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
        
        print(df.head(20))

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
        train_X, test_X, train_Y, test_Y = train_test_split(
            X, Y, random_state=self.config.analysis.random_seed, test_size=.3, shuffle=True
        )
        
        # Print label statistics
        train_label_count = np.sum(train_Y != self.UNLABELED_VALUE)
        test_label_count = np.sum(test_Y != self.UNLABELED_VALUE)
        progress_callback(67, f"Training set: {len(train_X)} sequences with {train_label_count} labeled timesteps")
        progress_callback(68, f"Test set: {len(test_X)} sequences with {test_label_count} labeled timesteps")
        
        model = None
        history = None

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
                model, history, _ = self._make_LSTM(train_X, train_Y, test_X, test_Y, progress_callback)
                # Save the global model
                model_path = self.model_path / "global_lstm_model.keras"
                model.save(model_path)
                progress_callback(80, f"New model saved to {model_path}")
                # Plot training history
                if history:
                    self._plot_single_history(history)


        else:
            progress_callback(70, "Training new model")
            model, history, _ = self._make_LSTM(train_X, train_Y, test_X, test_Y, progress_callback)
            # Save the global model
            model_path = self.model_path / "global_lstm_model.keras"
            model.save(model_path)
            progress_callback(80, f"Model saved to {model_path}")
            # Plot training history
            if history:
                self._plot_single_history(history)
        
        del train_X, train_Y, test_X, test_Y
        gc.collect()

        # Generate predictions for the full dataset
        progress_callback(85, "Generating predictions for the full dataset")

        # Process predictions in batches to avoid memory issues
        batch_size_pred = 1024  # Adjust based on your system's memory
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









    def manual_chunking(self, cow_ids, chunk_size):
        """
        Divide cow_ids into chunks of approximately chunk_size.
        
        Parameters:
        - cow_ids: List of cow IDs to divide
        - chunk_size: Desired size for each chunk
        
        Returns:
        - List of lists, where each inner list contains cow IDs for one chunk
        """
        # 1) Shuffle a copy (so we don't clobber the original)
        cows = cow_ids[:]
        random.shuffle(cows)
        
        # 2) Compute how many chunks we need
        total_cows = len(cows)
        n_chunks = (total_cows + chunk_size - 1) // chunk_size  # Ceiling division
        

        print("TOTAL COWS", total_cows)
        print("N_CHUNKS", n_chunks)
        # 3) Compute the actual chunk sizes
        k, r = divmod(total_cows, n_chunks)
        # First 'r' chunks get size k+1, the rest get k
        
        # 4) Create the chunks
        chunks = []
        idx = 0
        for i in range(n_chunks):
            size = k + 1 if i < r else k
            chunks.append(cows[idx:idx + size])
            idx += size
        
        # 5) Shuffle the chunks
        random.shuffle(chunks)
        print(chunks)
        
        return chunks


    def do_loocv(self, sequences: Dict[str, np.ndarray], df, n=11, compute_metrics_only=False, n_jobs=-1,progress_callback=None):
        """
        Run Leave-One-Out Cross Validation with parallelization
        
        Parameters:
        -----------
        sequences: Dictionary with Cow_Date_Key, X, and Y arrays
        df: Original dataframe
        n: Number of folds
        compute_metrics_only: If True, uses a faster approach for hyperparameter tuning
        n_jobs: Number of parallel processes to use (-1 for all available)
        """
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
        print("ALL GROUPS", groups[:5]) 
        unique_cows = np.unique(groups)
        print("ALL unique groups", unique_cows)
        test_chunks = self.manual_chunking(unique_cows, self.config.analysis.lstm.cows_per_cv_fold)

        # Determine number of processes
        n_jobs = (mp.cpu_count()-2) if n_jobs == -1 else n_jobs
        n_jobs = min(n_jobs, len(test_chunks))  # Can't use more processes than chunks


        ################################# PLEASE STOP CRASHING ##############################
        n_jobs = 1 ##########################################################################
        
        # Instead of using a Pool directly, we'll use the starmap approach with pre-built arguments
        print(f"Running LOOCV with {n_jobs} parallel processes")
        progress_callback(45, f"Running LOOCV with {n_jobs} parallel processes")
        # Create argument list for each fold
        fold_args = []
        for test_chunk in test_chunks:
            test_mask = np.isin(groups, test_chunk)
            train_mask = ~test_mask
            test_X, test_Y = X[test_mask], Y[test_mask]
            train_X, train_Y = X[train_mask], Y[train_mask]
            progress_callback(45, f"Initializing fold with held out device_ids: {test_chunk}")
            fold_args.append((test_chunk, train_X, train_Y, test_X, test_Y, compute_metrics_only))
        
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
                print(f"    Class {class_name} accuracy: {acc:.4f} (n={count})")
            print(result['confusion_matrix'])
            print(f"{'-'*182}")
        
        # Calculate overall metrics
        self._calculate_overall_metrics(all_predictions, all_actual)
        
        # Save and display overall results
        if not compute_metrics_only and all_histories:
            self._summarize_results(all_predictions, all_actual, all_histories)
        
        # Return empty lists where models and histories would be (can't pickle models easily)
        return [], []

    # Add this method to your class to be used by multiprocessing
    def _process_fold_mp(self, test_chunk, train_X, train_Y, test_X, test_Y, compute_metrics_only):
        """
        Process a single fold in LOOCV - this method is designed to be called via multiprocessing
        """

        import matplotlib
        matplotlib.use('Agg')
        # Create a fresh model for this process
        if compute_metrics_only:
            # Fast training for hyperparameter search
            model = Sequential(self.layers)
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'],
            )
            
            # Train for just a few epochs
            history = model.fit(
                train_X, 
                np.where(train_Y == -1, 0, train_Y), 
                epochs=10,  # Reduced epochs for speed
                batch_size=self.batch_size,
                verbose=0
            )
            
            # Predict
            pred_y = model.predict(test_X, verbose=0)
            if len(train_Y.shape) == 1:
                pred_y_classes = np.argmax(pred_y, axis=1)
            else:
                pred_y_classes = np.argmax(pred_y, axis=2)
                
            history_dict = history
        else:
            # Full training
            model, history, pred_y_classes = self._make_LSTM(
                test_X=test_X, test_Y=test_Y, train_X=train_X, train_Y=train_Y
            )
            history_dict = history
        



        # Calculate metrics for this fold - handle both OPS and OPO cases
        if len(test_Y.shape) == 1:  # One-per-observation case
            valid_indices = test_Y > -1
            accuracy = np.mean(pred_y_classes[valid_indices] == test_Y[valid_indices])
            
            # Per-class metrics
            class_accuracies = {}
            class_counts = {}
            for class_idx in np.unique(test_Y):
                if class_idx == -1:
                    continue
                class_mask = (test_Y == class_idx)
                if np.sum(class_mask) > 0:
                    class_acc = np.mean(pred_y_classes[class_mask] == class_idx)
                    class_name = self.inv_activity_map.get(class_idx, f"Unknown ({class_idx})")
                    class_accuracies[class_name] = float(class_acc)
                    class_counts[class_name] = int(np.sum(class_mask))
            
            # Confusion matrix
            try:
                cm = confusion_matrix(
                    test_Y[valid_indices], 
                    pred_y_classes[valid_indices],
                    labels=sorted([v for k,v in self.activity_map.items() if k is not np.nan])
                ).tolist()  # Convert to list for serialization
            except Exception as e:
                cm = f"Error generating confusion matrix: {e}"
        
        else:  # One-per-sequence case (OPO)
            # Flatten arrays for evaluation
            flat_pred = pred_y_classes.flatten()
            flat_actual = test_Y.flatten()
            
            # Only evaluate on valid data points
            valid_indices = flat_actual != self.UNLABELED_VALUE
            valid_pred = flat_pred[valid_indices]
            valid_actual = flat_actual[valid_indices]
            
            # Calculate overall accuracy
            accuracy = np.mean(valid_pred == valid_actual)
            
            # Per-class metrics
            class_accuracies = {}
            class_counts = {}
            for class_name, class_id in self.activity_map.items():
                if class_id == self.UNLABELED_VALUE:
                    continue  # Skip NaN class
                
                class_mask = flat_actual == class_id
                if np.sum(class_mask) > 0:
                    class_acc = np.mean(flat_pred[class_mask] == class_id)
                    class_accuracies[class_name] = float(class_acc)
                    class_counts[class_name] = int(np.sum(class_mask))
            
            # Confusion matrix - create it from flattened arrays
            try:
                cm = confusion_matrix(
                    valid_actual, 
                    valid_pred,
                    labels=sorted([v for k,v in self.activity_map.items() if k is not np.nan])
                ).tolist()  # Convert to list for serialization
            except Exception as e:
                cm = f"Error generating confusion matrix: {e}"
        

        # # Calculate metrics for this fold
        # valid_indices = test_Y > -1 if len(test_Y.shape) == 1 else np.any(test_Y != -1, axis=1)

        # #     valid_indices = test_Y > -1

        # #     accuracy = np.mean(pred_y_classes[valid_indices] == test_Y[valid_indices])
        # accuracy = np.mean(pred_y_classes[valid_indices] == test_Y[valid_indices])
        
        # # Per-class metrics
        # class_accuracies = {}
        # class_counts = {}
        # for class_idx in np.unique(test_Y):
        #     if class_idx == -1:
        #         continue
        #     class_mask = (test_Y == class_idx)
        #     if np.sum(class_mask) > 0:
        #         class_acc = np.mean(pred_y_classes[class_mask] == class_idx)
        #         class_name = self.inv_activity_map.get(class_idx, f"Unknown ({class_idx})")
        #         class_accuracies[class_name] = float(class_acc)
        #         class_counts[class_name] = int(np.sum(class_mask))
        
        # # Confusion matrix
        # try:
        #     cm = confusion_matrix(
        #         test_Y[valid_indices], 
        #         pred_y_classes[valid_indices],
        #         labels=sorted([v for k,v in self.activity_map.items() if k is not np.nan])
        #     ).tolist()  # Convert to list for serialization
        # except Exception as e:
        #     cm = f"Error generating confusion matrix: {e}"
        
        # Return a dictionary of results
        return {
            'test_chunk': test_chunk,
            'predictions': pred_y_classes,
            'actual': test_Y,
            'accuracy': float(accuracy),
            'class_accuracies': class_accuracies,
            'class_counts': class_counts,
            'confusion_matrix': cm,
            'history': history_dict
        }




    def _make_LSTM(self, train_X, train_Y, test_X, test_Y, progress_callback=None):
        """Build and train LSTM model with improved handling of imbalanced data
        
        Parameters:
        -----------
        train_X, train_Y: Training data
        test_X, test_Y: Test data
        Returns:
        --------
        model, history, pred_final
        """
        if progress_callback is None:
            progress_callback = lambda percent, message: None
        
        progress_callback(73, "Building and compiling LSTM model")

        def weighted_categorical_focal_loss(gamma=2.0):
            def loss(y_true, y_pred):
                y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
                # Convert sparse to one-hot
                y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=self.nclasses)
                # Calculate focal weight
                focal_weight = tf.pow(1.0 - y_pred, gamma)
                # Calculate weighted focal loss
                loss = -tf.reduce_sum(y_true_one_hot * focal_weight * tf.math.log(y_pred), axis=-1)
                return loss
            return loss
        
        ops = len(train_Y.shape) == 1
            
        n_epochs = self.config.analysis.lstm.epochs


        model = Sequential(self.layers)
        
        # lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        #     initial_learning_rate=0.001,
        #     decay_steps=n_epochs * len(train_X) // batchsize,
        #     alpha=0.00001  # Minimum learning rate
        # )
        
    #     # learning rate schedule

        lr_schedule = ExponentialDecay(
            self.initial_lr,
            decay_steps=self.decay_steps, #2000
            decay_rate=self.decay_steps # 1.5
        )

        optimizer = Adam(
            learning_rate=lr_schedule,
            clipnorm=self.clipnorm,  # Clip gradients to prevent explosions
            # clipvalue=0.5
        )


        model.compile(
            optimizer=optimizer,
            # loss=SparseCategoricalCrossentropy(),
            loss='sparse_categorical_crossentropy',
            # loss=weighted_categorical_focal_loss(gamma=2.0),
            metrics=['accuracy'],
            # metrics=[metrics.F1Score(name="f1", average="macro"), "accuracy"],
            
            # metrics=[masked_sparse_categorical_accuracy],
        )
        
        # model.summary()

        flat_labels = train_Y[train_Y != -1].ravel()

        present = np.unique(flat_labels)

        ### OLD
        # class_weights = compute_class_weight(
        #     class_weight='balanced',
        #     classes=present,
        #     y=flat_labels
        # )

        ### NEW
        class_counts = np.bincount(flat_labels.astype(int))[present]
        total = np.sum(class_counts)
        # More aggressive weighting for minority classes 
        class_weights = total / (class_counts * len(present))
        class_weights = class_weights / np.min(class_weights)  # Normalize
        ###

        # Training class weights
        cw_dict_present = dict(zip(present, class_weights))
        cw_arr = np.ones(self.nclasses, dtype='float32')

        for cls, w in cw_dict_present.items():
            cw_arr[cls] = w
        mask = (train_Y != -1).astype('float32')  # shape (N, T)
        train_Y_clipped = np.where(train_Y == -1, 0, train_Y)  # replace unlabeled with class 0

        sw = (mask * cw_arr[train_Y_clipped])# / (mask.sum(axis=1, keepdims=True) +1e-8)

        # Make validation and class weights

        val_split = 0.3
        # # # if ops:
        # # Can randomly select. no serial dependence between sequences. 
        # val_idx = np.zeros(len(train_X),dtype=np.int16)
        # val_idx[:int(len(train_X)/val_split)] = 1
        # np.random.shuffle(val_idx)


        # val_X = train_X[val_idx]
        # val_Y = train_Y_clipped[val_idx]
        # val_sw = sw[val_idx]


        # train_X = train_X[~val_idx]
        # train_Y_clipped = train_Y_clipped[~val_idx]
        # sw = sw[~val_idx]
        


        history = model.fit(
            train_X,
            train_Y_clipped,
            # validation_data=(val_X,val_Y,val_sw),
            validation_split=val_split,
            sample_weight=sw,

            # class_weight=class_weight_dict if ops else None,
            # # sample_weight=valid_idx,
            # sample_weight = None if ops else sw,
            epochs=n_epochs,
            batch_size=self.batch_size,
            callbacks=[
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.patience,
                    restore_best_weights=True,
                    min_delta=self.min_delta     # More sensitive to improvements
                ),
                # ReduceLROnPlateau(
                #     monitor='val_loss',
                #     factor=0.05,
                #     patience=10,
                #     min_lr=0.00001
                # ),
            ],
            verbose=0
        )

        pred_y = model.predict(test_X, verbose=0)
        
        if ops: 
            pred_y_classes = np.argmax(pred_y, axis=1)  # Use axis=1 since we flattened
        else:
            pred_y_classes = np.argmax(pred_y, axis=2)
        

        # Gridsearch Code
        # Calculate metrics for later use in hyperparameter search
        valid_indices = test_Y > -1
        self.last_accuracy = np.mean(pred_y_classes[valid_indices] == test_Y[valid_indices])
        
        # Calculate F1 score (better metric for imbalanced classes)
        if np.any(valid_indices):
            try:
                self.last_f1_score = f1_score(
                    test_Y[valid_indices], 
                    pred_y_classes[valid_indices],
                    average='macro'
                )
            except:
                self.last_f1_score = 0
                
        # Store class accuracies
        self.last_class_accuracies = {}
        for class_idx in np.unique(test_Y):
            if class_idx == -1:
                continue
            class_mask = (test_Y == class_idx)
            if np.sum(class_mask) > 0:
                class_acc = np.mean(pred_y_classes[class_mask] == class_idx)
                class_name = self.inv_activity_map.get(class_idx, f"Unknown ({class_idx})")
                self.last_class_accuracies[class_name] = float(class_acc)
        
        return model, history, pred_y_classes




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
        
        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], 'b-', label='Training Accuracy')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        

        plt.savefig(self.model_path / "lstm_global_history.png", dpi=300)
        plt.close()
        
        # Print final metrics
        print("\nTraining Summary:")
        print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
        print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
        if 'val_loss' in history.history:
            print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
            print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")



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
                average='macro',
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
        try:
            cm = confusion_matrix(
                valid_actual, 
                valid_pred,
                labels=sorted([v for k,v in self.activity_map.items() if k is not np.nan])
            )
            
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
            
        except Exception as e:
            print(f"Error generating visualization: {e}")
        
        # Save detailed metrics to file
        with open(self.cv_path / "lstm_results.txt", "w") as f:
            f.write(f"Overall accuracy: {accuracy:.4f}\n\n")
            f.write("Class-wise accuracy:\n")
            for class_name, class_id in self.activity_map.items():
                if class_id == self.UNLABELED_VALUE:
                    continue
                    
                class_mask = valid_actual == class_id
                if np.sum(class_mask) > 0:
                    class_acc = np.mean(valid_pred[class_mask] == class_id)
                    f.write(f"{class_name}: {class_acc:.4f} (n={np.sum(class_mask)})\n")


    def _plot_training_histories(self, histories):
        """Plot training histories across all folds"""
        plt.figure(figsize=(12, 4))
                
        metric_name = 'accuracy'
        val_metric_name = 'val_' + metric_name

        # Find max length of histories
        max_epochs = max(len(h.history['loss']) for h in histories)
        
        # Initialize arrays for mean and std calculations
        losses = np.zeros((len(histories), max_epochs))
        accuracies = np.zeros((len(histories), max_epochs))
        val_accuracies = np.zeros((len(histories), max_epochs))
        
        # Fill arrays with padding
        for i, h in enumerate(histories):
            # Pad or truncate loss
            curr_len = len(h.history['loss'])
            losses[i, :curr_len] = h.history['loss']
            losses[i, curr_len:] = h.history['loss'][-1]  # Pad with last value
            
            curr = h.history[metric_name]
            accuracies[i, :len(curr)] = curr
            accuracies[i, len(curr):] = curr[-1]
            val_accuracies[i, :len(h.history[val_metric_name])] = h.history[val_metric_name]
            val_accuracies[i, len(h.history[val_metric_name]):] = h.history[val_metric_name][-1]
        
        # Plot Loss
        plt.subplot(1, 2, 1)
        epochs = range(1, max_epochs + 1)
        
        # Plot individual histories
        for i in range(len(histories)):
            plt.plot(epochs[:len(histories[i].history['loss'])], 
                    histories[i].history['loss'], 
                    'b-', alpha=0.1)
        
        # Plot mean and std
        mean_loss = np.mean(losses, axis=0)
        std_loss = np.std(losses, axis=0)
        plt.plot(epochs, mean_loss, 'r-', label='Mean Loss')
        plt.fill_between(epochs, mean_loss-std_loss, mean_loss+std_loss, 
                        color='r', alpha=0.2, label='±1 std')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot Accuracy
        plt.subplot(1, 2, 2)
        
        # Plot individual histories
        for i in range(len(histories)):
            plt.plot(epochs[:len(histories[i].history[metric_name])], 
                    histories[i].history[metric_name], 
                    'b-', alpha=0.1)
        
        # Plot mean and std
        mean_acc = np.mean(accuracies, axis=0)
        std_acc = np.std(accuracies, axis=0)
        mean_val_acc = np.mean(val_accuracies, axis=0)

        plt.plot(epochs, mean_acc, 'r-', label='Mean Accuracy')
        plt.plot(epochs, mean_val_acc, 'g-', label='Mean Val_Accuracy')
        plt.fill_between(epochs, mean_acc-std_acc, mean_acc+std_acc, 
                        color='r', alpha=0.2, label='±1 std')
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        

        plt.savefig(self.cv_path / "lstm_training_history.png", dpi=300)
        plt.close()


    def test_isolated_nan_interpolation(self):
        # import pandas as pd
        # import numpy as np
        
        # Create test cases
        test_cases = [
            # Case 1: Isolated NaN in the middle
            [1.0, 2.0, np.nan, 4.0],
            
            # Case 2: Two consecutive NaNs
            [1.0, np.nan, np.nan, 4.0],
            
            # Case 3: NaN at the beginning
            [np.nan, 2.0, 3.0, 4.0],
            
            # Case 4: NaN at the end
            [1.0, 2.0, 3.0, np.nan],
            
            # Case 5: Mixed pattern
            [1.0, np.nan, 3.0, np.nan, np.nan, 6.0],
            
            # Case 6: No NaNs
            [1.0, 2.0, 3.0, 4.0]
        ]
        
        print("Test cases for isolated NaN interpolation:")
        for i, test_data in enumerate(test_cases):
            # Create DataFrame with a single column
            df = pd.DataFrame({'col': test_data})
            print(f"\nTest case {i+1}:")
            print(f"Original: {df['col'].tolist()}")
            
            # Apply standard interpolation
            df_standard = df.copy()
            df_standard['col'] = df_standard['col'].interpolate(method='linear')
            print(f"Standard interpolation: {df_standard['col'].tolist()}")
            
            # Apply our custom approach
            df_custom = df.copy()
            series = df_custom['col']
            mask = series.isna() & series.shift(1).notna() & series.shift(-1).notna()
            if mask.any():
                temp = series.copy()
                temp[~mask] = temp[~mask].fillna(method='ffill')
                temp = temp.interpolate(method='linear')
                series[mask] = temp[mask]
                df_custom['col'] = series
            print(f"Custom interpolation: {df_custom['col'].tolist()}")
        
        return "Tests completed"
