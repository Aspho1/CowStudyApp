import os

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
# from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective, plot_gaussian_process
from skopt import dump, load

import itertools
import json
from pathlib import Path
# from typing import Dict, List
import pandas as pd
import numpy as np
import keras
from tensorflow.random import set_seed
import random
import time
from matplotlib import pyplot as plt
# random.s

# from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
# import multiprocessing as mp


from cowstudyapp.config import ConfigManager
from cowstudyapp.analysis.RNN.utils import compute_seed

class BayesianOptSearch:
    """Bayesian optimization for hyperparameter tuning using Gaussian Processes"""
    
    def __init__(self, config: ConfigManager, df: pd.DataFrame):
        self.config = config
        self.results = []
        self.lstm_model=None
        self.sequences = None
        self.df: pd.DataFrame = df
        self.random_seed = self.config.analysis.random_seed


        ops = 'ops' if self.config.analysis.lstm.ops else 'opo'

        self.output_dir = self.config.analysis.cv_results / 'lstm' / ops / 'v6'
        # self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True,exist_ok=True)
        
        # self.output_dir = self.config.analysis.output_dir
        
        # Define the search space
        self.space = [
            Integer(10, 288, name='max_length'),
            Integer(4, 32, name='batch_size'),
            Real(1e-5, 1, "log-uniform", name='initial_lr'),
            Integer(100, 3000, name='decay_steps'),
            Real(0.01, 0.95, name='decay_rate'),
            Real(0.1, 1.5, name='clipnorm'),

            Categorical([16, 32, 64], name='lstm_size'),
            Categorical([16, 32, 64], name='dense_size'),


            # Integer(15, 15, name='patience'),
            # Real(1e-3, 1e-3, "log-uniform", name='min_delta'),
            # Real(1e-3, 1e-3, "log-uniform", name='reg_val'),
            # Optional: Categorical parameters
            # Categorical(['adam', 'rmsprop'], name='optimizer'),
            # Categorical(['relu', 'tanh'], name='activation'),
        ]

        self.param_labels = [
            "max_length",
            "batch_size",
            "initial_lr",
            "decay_steps",
            "decay_rate",
            "clipnorm",
            'lstm_size',
            'dense_size'
            # "patience",
            # "min_delta",
            # "reg_val",
        ]
        
        # Number of calls to make to the objective function
        self.n_calls = config.analysis.lstm.bayes_opt_n_calls
        
        # Path to save/load optimization results
        # io_type = 'ops' if self.config.analysis.lstm.ops else 'opo'
        self.results_path = str(self.output_dir / "bayes_opt_results.pkl")

        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"Setting results path to `{self.results_path}`")
        
    def run_search(self, lstm_model):
        """Run Bayesian optimization search"""
        self.lstm_model = lstm_model

        if self.lstm_model.config.analysis.mode != "LOOCV":
            raise ValueError("MUST use LOOCV when performing BO.")

        # acq_func="EI"
        # acq_func="LCB"
        acq_func="PI"

        # noise =

        # Instead of trying to use a decorator, use a simple function
        def objective(x):
            # Convert the parameter vector to a dictionary
            params = {dim.name: x[i] for i, dim in enumerate(self.space)}
            return self._objective(params)

        # Check if we should resume from previous run
        if os.path.exists(self.results_path) and self.config.analysis.lstm.bayes_opt_resume:
            print(f"Resuming optimization from `{self.results_path}`", type(self.results_path))

            previous_result = load(str(self.results_path))
            
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
                acq_func=acq_func,
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
                acq_func=acq_func,
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


        # param_hash = hash(frozenset(params.items())) & 0xFFFFFFFF
        # derived_seed = (self.random_seed + param_hash) & 0xFFFFFFFF
        # derived_seed = compute_seed(self.random_seed, params)
        #
        # # Set seeds for this evaluation
        # set_seed(derived_seed)
        # np.random.seed(derived_seed)
        # random.seed(derived_seed)


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

        # set_seed(self.config.analysis.random_seed)
        # np.random.seed(self.config.analysis.random_seed)
        # random.seed(self.config.analysis.random_seed)
        
        # Run a small LOOCV with current parameters
        start_time = time.time()

        self.lstm_model.build_model() # progress_callback

        self.sequences = self.lstm_model.build_sequences(self.df) #, progress_callback

        # Use fewer CV splits for speed
        cows_per_fold = self.config.analysis.lstm.cows_per_cv_fold
        
        if self.config.analysis.mode == "LOOCV":
            _, _ = self.lstm_model.do_loocv(
                sequences=self.sequences, 
                df=self.df, 
                n=cows_per_fold, 
                compute_metrics_only=self.config.analysis.lstm.bayes_opt_fast_eval
            )
        else:  # PRODUCT mode
            _, _ = self.lstm_model.dont_do_loocv(
                sequences=self.sequences, 
                df=self.df
            )
            
        elapsed_time = time.time() - start_time
        
        # Get the accuracy
        # score = self.lstm_model.last_accuracy
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
#
#             # Special case for max_length which also needs to update sequence_length
#             if param_name == 'max_length':
#                 self.lstm_model.sequence_length = param_value
#


    def _save_optimization_plots(self, result):
        """Save optimization visualization plots"""
        print(f"Saving plots to `{self.output_dir}`")
        # Plot convergence
        plt.figure(figsize=(10, 6))
        plot_convergence(result)
        plt.savefig(self.output_dir / "bayes_opt_convergence.png", dpi=300)
        plt.close()
        
        # Plot individual parameter effects (partial dependence)
        # fig, ax = plt.subplots(3, 3, figsize=(15, 12))
        plot_objective(result,dimensions=self.param_labels) #, dimensions=range(len(self.space))
        plt.tight_layout()
        plt.savefig(self.output_dir / "bayes_opt_parameters.png", dpi=300)
        plt.close()

        #
        # plot_gaussian_process(result) #, dimensions=range(len(self.space))
        # plt.tight_layout()
        # plt.savefig(self.output_dir / "bayes_opt_gp_res.png", dpi=300)
        # plt.close()

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
                models, histories = lstm_model.do_loocv(sequences=sequences, df=df, n=6)
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
