# src/cowstudyapp/visuals/show_temp_vs_activity_Bayes.py

from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
import os
import ephem
import multiprocessing
import sys
import gc
import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import subprocess
from pathlib import Path
import platform

# os.environ['R_HOME'] = "C:\\Program Files\\R\\R-4.4.1"


from cowstudyapp.utils import from_posix_col
from cowstudyapp.config import ConfigManager


def find_r_executable(config_path) -> Path:
    """Find the R executable on the system"""
    if platform.system() == "Windows":
        # Common R installation paths on Windows
        possible_paths = [
            config_path, 
            r"C:/Program Files/R/R-4.4.3/bin/Rscript.exe",
            r"C:/Program Files/R/R-4.4.0/bin/Rscript.exe",
            r"C:/Program Files/R/R-4.3.2/bin/Rscript.exe",
            r"C:/Program Files/R/R-4.3.0/bin/Rscript.exe",
        ]

        # Check R_HOME environment variable
        r_home = os.environ.get("R_HOME")
        print("RHOME:", r_home)
        if r_home:
            possible_paths.append(Path(r_home) / "bin" / "Rscript.exe")

        # # Try to find Rscript in PATH
        # try:
        #     result = subprocess.run(
        #         ["where", "Rscript.exe"], capture_output=True, text=True
        #     )
        #     if result.returncode == 0:
        #         possible_paths.extend(result.stdout.splitlines())
        # except subprocess.SubprocessError:
        #     pass

        # Return first existing path
        for path in possible_paths:
            print("CHECKING ", path)
            if Path(path).exists():
                return Path(path)

    else:  # Linux/Mac
        try:
            # First try to find Rscript on the path
            rscript_path = subprocess.check_output(["which", "Rscript"]).decode().strip()
            return Path(rscript_path)
        except subprocess.SubprocessError:
            # If which doesn't work, check common locations
            possible_paths = [
                "/usr/bin/Rscript",
                "/usr/local/bin/Rscript"
            ]
            for path in possible_paths:
                if Path(path).exists():
                    return Path(path)

    raise FileNotFoundError(
        "Could not find R executable. Please install R and ensure it's in your PATH or "
        "designated in the `r_executable` field in your configuration file."
    )



class TermSelector:
    """Class to handle model term selection using a correlation matrix format"""
    
    def __init__(self, base_vars:List[str] = ['temp', 'age', 'weight', 'time'], max_degree:int=4):
        self.base_vars = base_vars
        self.max_degree = max_degree
    
    def get_model_terms(self, term_list: List[str]) -> Tuple[List[str], List[str]]:
        """
        Convert term dictionary to lists of main effects and interactions
        
        Parameters:
        -----------
        term_dict : Dict[str, bool]
            Dictionary with terms as keys and boolean values
            
        Returns:
        --------
        Tuple[List[str], List[str]]
            Lists of main effects and interactions to include in the model
        """
        main_effects = []
        interactions = []
        
        for term in term_list:
            if ':' in term:
                interactions.append(term)
            else:
                main_effects.append(term)
                
        interactions = [self._standardize_interaction_order(term) for term in interactions]
                
        return main_effects, interactions
    
    def _standardize_interaction_order(self, interaction_term: str) -> str:
        """
        Standardize the order of terms in an interaction (e.g., 'time:temp' -> 'temp:time')
        to ensure consistent ordering alphabetically and by polynomial degree
        
        Parameters:
        -----------
        interaction_term : str
            Original interaction term (e.g., 'time:temp' or 'time^2:temp')
            
        Returns:
        --------
        str
            Standardized interaction term with consistent ordering
        """
        if ':' not in interaction_term:
            return interaction_term
            
        # Split the interaction term
        parts = interaction_term.split(':')
        
        # Function to extract base variable and degree
        def extract_components(term):
            if '^' in term:
                base, degree = term.split('^')
                degree = int(degree)
            else:
                base = term
                degree = 1
            return base, degree
        
        # Extract components for each part
        components = [extract_components(part) for part in parts]
        
        # Sort components by base variable first, then by degree (higher degree first)
        # This ensures alphabetical ordering first, then polynomial degree
        sorted_components = sorted(components, key=lambda x: (x[0], -x[1]))
        
        # Rebuild the interaction term
        sorted_terms = [f"{base}^{degree}" if degree > 1 else base for base, degree in sorted_components]
        return ':'.join(sorted_terms)
    
    def standardize_term_names(self, terms: List[str]) -> List[str]:
        """Convert human-readable terms to model variable names"""
        term_mapping = {
            'temp': 'temp_z',
            'temp^2': 'temp_z_sq',
            'temp^3': 'temp_z_cub',
            'temp^4': 'temp_z_qrt',
            'age': 'age_z',
            'age^2': 'age_z_sq',
            'age^3': 'age_z_cub',
            'age^4': 'age_z_qrt',
            'weight': 'weight_z',
            'weight^2': 'weight_z_sq',
            'weight^3': 'weight_z_cub',
            'weight^4': 'weight_z_qrt',
            'time': 'time_z',
            'time^2': 'time_z_sq',
            'time^3': 'time_z_cub',
            'time^4': 'time_z_qrt',
            'hod': 'hod_z',
            'hod^2': 'hod_z_sq',
            'hod^3': 'hod_z_cub',
            'hod^4': 'hod_z_qrt',
        }
        
        # Map simple terms
        result = []
        for term in terms:
            if ':' in term:
                # Handle interaction terms
                parts = term.split(':')
                mapped_parts = [term_mapping.get(part, part) for part in parts]
                result.append(':'.join(mapped_parts))
            else:
                # Handle main effects
                result.append(term_mapping.get(term, term))
                
        return result


class BehaviorModelBuilder:
    """Class to prepare data and build models for analyzing behavior"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        # Set up Norris, MT observer for sun calculations
        self.observer = ephem.Observer()
        self.observer.lat = '45.575' # Get median lat
        self.observer.lon = '-111.625' # Get median lon
        self.observer.elevation = 1715 # Get median alt
        self.observer.pressure = 0
        self.observer.horizon = '-0:34'
        self.sunrise_buffer = -1.5  # After Sunrise
        self.sunset_buffer = 1.5  # After Sunset

        self.base_effects = ['temp', 'time', 'hod']
        if self.config.io.cow_info_path and self.config.io.cow_info_path.exists():
            self.base_effects.extend(['age', 'weight'])

        self.degree_map = {
            '':1,
            '_sq':2,
            '_cub':3,
            '_qrt':4
        }

    def get_z_names(self):
        return [f"{be}_z" for be in self.base_effects]

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare the dataset for modeling by adding all necessary columns
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw dataframe with cow behavior data
            
        Returns:
        --------
        pd.DataFrame
            Processed dataframe with all required columns
        """
        # Add suntime columns (day/night classification)
        df = self._add_suntime_cols(df)
        
        # Add temperature data if needed
        df = self._add_temp_col(df)
        
        # Add cow metadata (age, weight)
        if self.config.io.cow_info_path and self.config.io.cow_info_path.exists():
            df = self._add_meta_info(df, filter_weight=False)
        
        return df
    
    def create_model_dataframe(self, df: pd.DataFrame, selected_terms: List[str]) -> pd.DataFrame:
        """
        Create a clean dataframe with all terms needed for modeling
        
        Parameters:
        -----------
        df : pd.DataFrame
            Processed dataframe with all basic columns
        selected_terms : List[str]
            List of terms to include in the model
            
        Returns:
        --------
        pd.DataFrame
            Clean dataframe ready for modeling with all required terms
        """
        # Create baseline dataframe with standard z-scored variables
        clean_df = pd.DataFrame()
        clean_df['ID'] = df['ID']
        
        
        means={}
        stds={}


        for effect in self.base_effects:
            means[effect]=df[effect].mean()
            stds[effect]=df[effect].std()
            z_name = f'{effect}_z'
            clean_df[z_name] = (df[effect] - means[effect]) / stds[effect]





        # Testing orthogonal polynomials

        # Generate orthogonal polynomial terms for time
        if any(term.startswith('time_z') for term in selected_terms):
            # Find highest polynomial degree needed
            max_degree = 1
            for suffix, degree in self.degree_map.items():
                if f'time_z{suffix}' in selected_terms:
                    max_degree = max(max_degree, degree)
            
            # Generate orthogonal polynomial basis
            # from numpy.polynomial import polynomial as P
            time_values = clean_df['time_z'].values
            
            # Create orthogonal polynomial terms
            ortho_polys = np.zeros((len(time_values), max_degree + 1))
            
            # First term is just the constant
            ortho_polys[:, 0] = 1.0
            
            if max_degree >= 1:
                # Scale to [-1, 1] for numerical stability
                scaled_time = 2 * (time_values - time_values.min()) / (time_values.max() - time_values.min()) - 1
                
                # Get orthogonal polynomial values
                legendre_values = np.polynomial.legendre.legvander(scaled_time, max_degree)
                
                # Add to the array, skip first column (constant term)
                ortho_polys[:, 1:] = legendre_values[:, 1:]
            
            # Store original time variable
            original_time_z = clean_df['time_z'].copy()
            
            # Replace polynomial terms with orthogonal versions
            for degree in range(1, max_degree + 1):
                suffix = '' if degree == 1 else ('_sq' if degree == 2 else ('_cub' if degree == 3 else '_qrt'))
                col_name = f'time_z{suffix}'
                if col_name in selected_terms:
                    clean_df[col_name] = ortho_polys[:, degree]
            
            # Store mapping for interpretation
            clean_df.attrs['ortho_time_mapping'] = {
                'original_time': original_time_z,
                'ortho_basis': ortho_polys,
                'max_degree': max_degree
            }
        
        # Handle all other polynomial terms normally
        for effect in self.base_effects:
            if effect != 'time':  # Skip time as we handled it specially
                z_name = f'{effect}_z'
                for suffix, degree in self.degree_map.items():
                    if f'{z_name}{suffix}' in selected_terms:
                        clean_df[f'{z_name}{suffix}'] = clean_df[z_name] ** degree

        # Add indicator variables for each state
        for state in self.config.analysis.hmm.states:
            clean_df[f'is_{state.lower()}'] = (df['predicted_state'] == state).astype(int)
            
        # Store standardization info for later
        clean_df.attrs['means'] = means
        clean_df.attrs['stds'] = stds
        
        # print(clean_df.head())
        return clean_df
    
    def build_model_formula(self, main_effects: List[str], interactions: List[str], state: str) -> str:
        """
        Build the formula for a mixed-effects model
        
        Parameters:
        -----------
        main_effects : List[str]
            List of main effect terms
        interactions : List[str]
            List of interaction terms
        state : str
            The behavior state to model
            
        Returns:
        --------
        str
            Formula for the mixed-effects model
        """
        # Combine main effects and interactions
        fixed_effects = ' + '.join(main_effects)
        if interactions:
            fixed_effects += ' + ' + ' + '.join(interactions)
            
        # Add random effects
        random_effects = "(1|ID)"  # Random intercepts for cow ID
        
        # Create formula
        formula = f'is_{state.lower()} ~ {fixed_effects} + {random_effects}'
        
        return formula
    
    # Helper methods for data preparation
    def _add_suntime_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate sunrise/sunset times for all rows"""
        # print("Calculating sunrise and sunset times...")
        
        # Make sure we have dates
        if 'mt' not in df.columns:
            df['mt'] = from_posix_col(df['posix_time'], self.config.analysis.timezone)
        
        # Extract date component
        df['date'] = df['mt'].dt.date
        df['time'] = (df['mt'].dt.minute + df['mt'].dt.hour * 60)
        # Filter weigh days
        if self.config.visuals.heatmap.filter_weigh_days:
            weigh_days = pd.to_datetime(self.config.visuals.heatmap.weigh_days)
            if weigh_days is not None:
                df = df[~df["date"].isin([wd.date() for wd in weigh_days])]

        # Get unique dates
        unique_dates = df['date'].unique()
        print(f"Calculating sun times for {len(unique_dates)} unique dates...")
        
        # Calculate sunrise/sunset for each unique date
        sun_times = {}
        for date in unique_dates:
            # Calculate sunrise time
            self.observer.date = date.strftime('%Y/%m/%d 00:00:00')
            sunrise = pd.to_datetime(str(self.observer.next_rising(ephem.Sun())))
            sunrise = sunrise.tz_localize('UTC').tz_convert(self.config.analysis.timezone)
            
            # Calculate sunset time
            self.observer.date = date.strftime('%Y/%m/%d 12:00:00')
            sunset = pd.to_datetime(str(self.observer.next_setting(ephem.Sun())))
            sunset = sunset.tz_localize('UTC').tz_convert(self.config.analysis.timezone)
            
            sun_times[date] = {
                'sunrise': sunrise + pd.Timedelta(hours=self.sunrise_buffer),
                'sunset': sunset + pd.Timedelta(hours=self.sunset_buffer)
            }
        
        # Create dataframes for merging
        sunrise_df = pd.DataFrame([
            {'date': date, 'sunrise': times['sunrise']} 
            for date, times in sun_times.items()
        ])
        
        sunset_df = pd.DataFrame([
            {'date': date, 'sunset': times['sunset']} 
            for date, times in sun_times.items()
        ])

        # Merge sunrise times
        df = pd.merge(df, sunrise_df, on='date', how='left')
        
        # Merge sunset times
        df = pd.merge(df, sunset_df, on='date', how='left')

        # Add daylight indicator
        df['is_daylight'] = ((df['mt'] >= df['sunrise']) & (df['mt'] <= df['sunset']))
        df['hod'] = (df['sunset'] - df['sunrise']).dt.total_seconds()/3600

        # grazing_percentages = (df[df['predicted_state'] == 'Grazing']
        #                     .groupby(['ID', 'date'])
        #                     .size()
        #                     .div(df.groupby(['ID', 'date']).size())
        #                     .mul(100))

        # # Now get the overall statistics
        # overall_stats = grazing_percentages.describe()
        # print("\nOverall Grazing Statistics (% of time):")
        # print(overall_stats)
        # # print(df.head())
        # return
        # Add relative time variables
        # df['minutes_after_sunrise'] = (df['mt'] - df['sunrise']).dt.total_seconds() / 60
        # df['minutes_of_daylight'] = (df['sunset'] - df['sunrise']).dt.total_seconds() / 60
        # df['rel_time'] = df['minutes_after_sunrise'] / df['minutes_of_daylight']

        
        return df
    
    def _add_temp_col(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temperature data by joining with temperature dataset"""
        if 'temp' not in df.columns:
            if 'temperature_gps' in df.columns:
                df.rename(columns={'temperature_gps': 'temp'}, inplace=True)
            else:
                # Read temperature data
                temp_df = pd.read_csv(self.config.analysis.target_dataset)
                
                # Ensure we have the right columns for joining
                required_cols = ['device_id', 'posix_time', 'temperature_gps']
                if not all(col in temp_df.columns for col in required_cols):
                    raise ValueError(f"Temperature dataset must contain columns: {required_cols}")
                
                # Perform inner join
                df = pd.merge(
                    df,
                    temp_df[['device_id', 'posix_time', 'temperature_gps']],
                    left_on=['ID', 'posix_time'],
                    right_on=['device_id', 'posix_time'],
                    how='inner'
                )
                
                # Rename the temp column
                df.rename(columns={'temperature_gps': 'temp'}, inplace=True)
                
                # Drop the extra device_id column from the join
                df.drop('device_id', axis=1, inplace=True)
        
        return df
    
    def _add_meta_info(self, df: pd.DataFrame, filter_weight:bool) -> pd.DataFrame:
        """Add cow metadata (age, weight)"""
        meta_info_df = pd.read_excel(self.config.io.cow_info_path)
        meta_info_df["age"] = 2022 - meta_info_df['year_b']
        meta_info_df.set_index('collar_id', inplace=True)
        meta_info_df['BW_preg'] = meta_info_df['BW_preg'] * 0.4535924  # lbs to kg
        
        age_dict = meta_info_df['age'].to_dict()
        weight_dict = meta_info_df['BW_preg'].to_dict()

        df['age'] = df["ID"].map(age_dict)
        df['weight'] = df["ID"].map(weight_dict)

        # Filter out rows with missing weights
        if filter_weight:
            df = df[df['weight'] > 0]

        
        return df


class BehaviorAnalyzer:
    """Class to analyze cow behavior in relation to temperature and other factors"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.model_builder = BehaviorModelBuilder(config)
        self.term_selector = TermSelector()

        # Only set R_HOME on Windows, not on Linux
        if platform.system() == "Windows":
            r_executable = find_r_executable(self.config.analysis.r_executable)
            os.environ['R_HOME'] = str(r_executable.parent.parent)
        
        # Display mappings for variable names
        self.display_map = {
            'temp_z': 'Temperature',
            'time_z': 'Time',
            'age_z': 'Age',
            'weight_z': 'Weight',
            'hod_z': 'Hours of Daylight'
        }
        
        # Suffix mappings for order terms
        self.order_suffix = {
            2: " (Quadratic)",
            3: " (Cubic)",
            4: " (Quartic)"
        }

        # Generate parameter labels
        self.param_labels = self._generate_param_labels()

        # Scale factors for interpretable units
        self.scale_factors = {
            'temp': 5.0,     # per 5°C
            'age': 1.0,      # per year
            'weight': 30.0,  # per 30kg
            # Could add this but not sure I want to. 
            'time': 60       # per 15 minutes
        }



    def optimize_r_performance(self):
        """Set up R environment and ensure required packages are installed"""
        try:
            # First try to import and initialize rpy2
            import rpy2.robjects as robjects
            from rpy2.robjects.packages import importr, isinstalled, conversion, default_converter
            import rpy2

            print(f"rpy2 version: {rpy2.__version__}")
            print(f"Current R_HOME: {os.environ.get('R_HOME', 'Not set')}")
            print(f"Python executable: {sys.executable}")

            print("-"*80)

            # Print detailed R environment information
            with conversion.localconverter(default_converter):
                r_version = robjects.r('R.version.string')
                print(f"R version: {r_version[0]}")
                
                # Get library paths actually being used by R
                print("R library paths:")
                lib_paths = robjects.r('.libPaths()')
                for i, path in enumerate(lib_paths):
                    print(f"  [{i}] {path}")
                
                # Check for required packages
                required_packages = ['lme4', 'Matrix', 'lmerTest']
                missing_packages = []
                
                for pkg in required_packages:
                    if not isinstalled(pkg):
                        print(f"Package {pkg} is not installed")
                        missing_packages.append(pkg)
                    else:
                        pkg_version = robjects.r(f'packageVersion("{pkg}")')
                        print(f"Package {pkg} version: {pkg_version[0]}")
                
                # Install missing packages if needed
                if missing_packages:
                    print(f"Installing missing packages: {', '.join(missing_packages)}")
                    utils = importr('utils')
                    for pkg in missing_packages:
                        print(f"Installing {pkg}...")
                        utils.install_packages(pkg, repos="https://cloud.r-project.org")
                        print(f"{pkg} installed")
                        
                # Load required packages to ensure they work
                print("Loading required packages...")
                matrix = importr('Matrix')
                lme4 = importr('lme4')
                print("Required packages loaded successfully")
                        
                # Configure R performance settings
                cores = max(1, multiprocessing.cpu_count() - 4)
                robjects.r(f'options(mc.cores = {cores})')
                
                if 'OMP_NUM_THREADS' not in os.environ:
                    os.environ['OMP_NUM_THREADS'] = str(min(12, cores))
                    
                print("R performance optimizations applied")
                return True
            
        except Exception as e:
            print(f"Warning: R setup failed: {str(e)}")
            print("Stack trace:")
            import traceback
            traceback.print_exc()
            print("Continuing without R optimizations...")
            return False

    def _generate_param_labels(self) -> Dict[str, str]:
        """Generate human-readable labels for model parameters"""
        labels = {
            '(Intercept)': 'Baseline at average predictors'
        }
        
        # Generate main effect labels
        for var_name, display_name in self.display_map.items():
            # Linear effect
            labels[var_name] = self._get_effect_description(var_name)
            
            # Higher order effects
            for order in range(2, 5):
                suffix = '_sq' if order == 2 else ('_cub' if order == 3 else '_qrt')
                labels[f"{var_name}{suffix}"] = f"{display_name} effect{self.order_suffix[order]}"
        
        # Generate interaction labels
        for var1, name1 in self.display_map.items():
            for var2, name2 in self.display_map.items():
                if var1 != var2:
                    labels[f"{var1}:{var2}"] = f"{name1} × {name2} interaction"
            
            # Interactions with higher order terms
            for var2, name2 in self.display_map.items():
                if var1 != var2:
                    for order in range(2, 5):
                        suffix = '_sq' if order == 2 else ('_cub' if order == 3 else '_qrt')
                        labels[f"{var1}:{var2}{suffix}"] = f"{name1} × {name2}{self.order_suffix[order]} interaction"
                        labels[f"{var2}{suffix}:{var1}"] = f"{name1} × {name2}{self.order_suffix[order]} interaction"
        
        return labels
    
    def _get_effect_description(self, var_name: str) -> str:
        """Get descriptive label for a parameter's effect"""
        display_name = self.display_map.get(var_name, var_name)
        
        if var_name == 'temp_z':
            return 'Temperature effect (5°C increase)'
        elif var_name == 'age_z':
            return 'Age effect (1-year increase)'
        elif var_name == 'weight_z':
            return 'Weight effect (30kg increase)'
        elif var_name == 'time_z':
            return 'Time of day (linear)'
        elif var_name == 'hod_z':
            return 'Hours of Daylight'
        else:
            return f"{display_name} effect"




    def analyze(self, df: pd.DataFrame, term_matrix: str = None) -> Dict:
        """
        Analyze cow behavior based on the provided data and term selection
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw dataframe with cow behavior data
        term_matrix : str, optional
            Term selection matrix to use for model building
            
        Returns:
        --------
        Dict
            Analysis results
        """
        try:
            self.optimize_r_performance()
        except Exception as e:
            print(f"Warning: R optimization failed: {e}")
            print("Continuing without R optimizations...")

        # Prepare the dataset
        prepared_df = self.model_builder.prepare_data(df)
        
        # Sample data if necessary
        if len(prepared_df) > 50000:
            sample_frac = self.config.visuals.temperature_graph.sample_size
            print(f"Using a {sample_frac*100:.0f}% random sample ({len(prepared_df)} -> {int(len(prepared_df)*sample_frac)} rows)")
            prepared_df = prepared_df.sample(frac=sample_frac, random_state=self.config.visuals.random_seed)
        
        # Split data by day/night
        df_day = prepared_df[prepared_df['is_daylight']].copy()
        df_night = prepared_df[~prepared_df['is_daylight']].copy()
        

        # Default term selection: linear terms + interactions
        # selected_term_list = [
        #     'temp', 
        #     'temp^2', 
        #     # 'age', 
        #     'hod',
        #     # 'weight',
        #     'time',
        #     # 'time^2',
        #     # 'time^3',                
        #     'time^4',
        #     'hod:time',
        #     # 'hod:time^2',
        #     # 'hod:time^3',
        #     # 'hod:time^4',
        #     # 'temp:age', 
        #     # 'temp:weight', 
        #     # 'age:weight',
        #     'hod:temp',
        #     'time:temp',
        #     'time^2:temp',
        #     'time^3:temp',
        #     'time^4:temp'
        # ]

        selected_term_list = self.config.visuals.temperature_graph.terms
        
        # Get main effects and interactions
        main_effects, interactions = self.term_selector.get_model_terms(selected_term_list)
        
        # Standardize term names for modeling
        std_main_effects = self.term_selector.standardize_term_names(main_effects)
        std_interactions = self.term_selector.standardize_term_names(interactions)
        
        # Combine all terms needed for the model
        all_terms = std_main_effects + [term for term in std_interactions if ':' in term]
        
        # Create model for day and/or night
        results = {}
        
        if self.config.visuals.temperature_graph.daynight in ["day", 'both']:
            print(f"\nAnalyzing DAY behavior... ({len(df_day)} rows)")
            # results['both'] = self._analyze_period(prepared_df, 'both', all_terms, std_main_effects, std_interactions)
            results['day'] = self._analyze_period(df_day, 'day', all_terms, std_main_effects, std_interactions)
            
            # print(f"\nAnalyzing DAY behavior... ({len(df)} rows)")
            # results['day'] = self._analyze_period(prepared_df, 'both', all_terms, std_main_effects, std_interactions)
        
        
        if self.config.visuals.temperature_graph.daynight in ["night", 'both']:
            print("\nAnalyzing NIGHT behavior...")
            results['night'] = self._analyze_period(df_night, 'night', all_terms, std_main_effects, std_interactions)
        


        # AUC STUFF
        # for period in ['day', 'night']:
        for state in self.config.analysis.hmm.states:
            self.analyze_temperature_effects(results, 'day', state)

        # Generate result table
        table = self._create_results_table(results)
        print("\nINTERPRETABLE MODEL RESULTS:")
        print(table)



        # Just before the return statement
        if self.config.visuals.temperature_graph.show_curve:
            plot_path = os.path.join(self.config.visuals.visuals_root_path, 'behavior_probabilities.png')
            self.plot_state_probabilities(results, save_path=plot_path, by='temp')       
            # self.plot_state_probabilities(results, save_path=plot_path, by='hod')       

        # Export to CSV if needed
        if self.config.visuals.temperature_graph.export_excel:
            csv_path = os.path.join(self.config.visuals.visuals_root_path, 'behavior_model_results.xlsx')
            self._export_results_to_datafile(results, csv_path)
            print(f"Results exported to CSV: {csv_path}")
        
        return results
    
    def _analyze_period(self, df: pd.DataFrame, period: str, all_terms: List[str], 
                        main_effects: List[str], interactions: List[str]) -> Dict:
        """
        Analyze data for a specific period (day or night)
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data for the specific period
        period : str
            'day' or 'night'
        all_terms : List[str]
            All terms needed for the model
        main_effects : List[str]
            Main effect terms
        interactions : List[str]
            Interaction terms
            
        Returns:
        --------
        Dict
            Results for this period
        """
        try:
            from pymer4.models import Lmer
        except Exception as e:
            print(f"Error importing pymer4: {e}")
            print("This may be due to R configuration issues. Try installing/updating:")
            print("1. libffi-dev package on your system")
            print("2. rpy2 with: pip install rpy2==3.5.13")  # Specify a version known to work
            print("3. Check that R and the required R packages are installed")

        print(f"\n{'-'*80}")
        print(f"ANALYZING PARAMETER EFFECTS ON BEHAVIOR DURING {period.upper()}")
        print(f"{'-'*80}")
        
        # Print data summary
        print(f"\nData Summary for {period}:")
        print(f"Temperature range: {df['temp'].min():.1f} to {df['temp'].max():.1f}°C")
        print(f"HOD range: {df['hod'].min():.1f} to {df['hod'].max():.1f} hours")
        print(f"Number of cows: {df['ID'].nunique()}")
        print(f"Total observations: {len(df)}")
        time_min, time_max = df.time.min(), df.time.max()
        print(f"time min {time_min} - {time_max}")

        try:
            print(f"Age range: {df['age'].min()} to {df['age'].max()} years")
            print(f"Weight range: {df['weight'].min():.1f} to {df['weight'].max():.1f} kg")
        except Exception as e:
            pass

        # Create model dataframe
        clean_df = self.model_builder.create_model_dataframe(df, all_terms)
        
        # Calculate standardization info for interpretation
        means = clean_df.attrs['means']
        stds = clean_df.attrs['stds']
        
        # Scale factors for interpretable units

        
        # Store results for each behavior state
        period_results = {
            'standardization_info': {'means': means, 'stds': stds},
            'scale_factors': self.scale_factors,
            'states': {},
            'time_domain': [time_min, time_max]
        }
        

        #################### TEMPORARY VIF CALCULATIONS
        # print(type(clean_df))
        # print(clean_df)
        # print(clean_df.columns)
        # X = clean_df[['temp_z', 'time_z', 'time_z_sq', 'time_z_cub', 'time_z_qrt', 'hod_z']]
        # X['temp_time'] = X['time_z'] * X['temp_z']
        # X['temp_time_4'] = X['time_z_qrt'] * X['temp_z']
        # import statsmodels.api as sm
        # X = sm.add_constant(X)

        # # Calculate VIF
        # from statsmodels.stats.outliers_influence import variance_inflation_factor
        # vif_df = pd.DataFrame()
        # vif_df['Variable'] = X.columns
        # vif_df['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        # print(vif_df)

        # return

        #################### END VIF






        # Analyze each behavior state
        for state in self.config.analysis.hmm.states:
            print(f"\n{'-'*40}")
            print(f"ANALYZING {state.upper()} STATE - {period.upper()}")
            print(f"{'-'*40}")
            
            # Build model formula
            formula = self.model_builder.build_model_formula(main_effects, interactions, state)
            print(f"Model formula: {formula}")
            
            try:
                import rpy2
                import rpy2.robjects as robjects

                # Run garbage collection before fitting
                gc.collect()
                # if 'rpy2.robjects' in sys.modules:
                #     from rpy2.robjects.packages import conversion, default_converter
                #     with conversion.localconverter(default_converter):
                #         robjects.r('gc()')
                
                # Fit the model
                model = Lmer(formula, data=clean_df, family='binomial')
                fit_result = model.fit()

                # Renaming the indexes to be alphabetical

                for idx in fit_result.index:
                    if ':' in idx:
                        # This is an interaction term
                        parts = idx.split(':')
                        # Sort by alphabetical order first, then by term complexity if needed
                        sorted_parts = sorted(parts)
                        new_idx = ':'.join(sorted_parts)
                        
                        if new_idx != idx:
                            # Rename the index
                            fit_result.rename(index={idx: new_idx}, inplace=True)
                


                # Print model results
                print(f"\nModel Results for {state} ({period}):")
                print(fit_result)

                # Store results
                period_results['states'][state] = {
                    'formula': formula,
                    'model': model,
                    'fit_result': fit_result
                }
                
                # Print interpretable results
                self._print_interpretable_results(state, period, fit_result, means, stds)
                
                # Print model fit statistics
                if hasattr(model, 'AIC'):
                    print(f"\nModel Fit Statistics:")
                    print(f"  AIC: {model.AIC:.1f}")
                    print(f"  BIC: {model.BIC:.1f}")
                    
            except Exception as e:
                print(f"Error fitting model for {state} during {period}: {str(e)}")
                import traceback
                traceback.print_exc()
                period_results['states'][state] = {'error': str(e)}
                
            # Clean up memory
            gc.collect()         

        return period_results    


    def _get_display_name_of_term(self, term: str):
        """Convert model term to human-readable display name."""
        # Extract order (1, 2, 3, 4) and clean term
        order = 1
        clean_term = term
        
        if term.endswith("_sq"):
            order = 2
            clean_term = term.replace("_sq", "")
        elif term.endswith("_cub"):
            order = 3
            clean_term = term.replace("_cub", "")
        elif term.endswith("_qrt"):
            order = 4
            clean_term = term.replace("_qrt", "")
            
        
        if clean_term in self.display_map:
            clean_term = self.display_map[clean_term]
        else:
            print(f"UNKNOWN TERM `{clean_term}`")
        
        
        return clean_term + (self.order_suffix.get(order, ''))

    def _print_interpretable_results(self, state: str, period: str, fit_result: pd.DataFrame, 
                                    means: Dict[str, float], stds: Dict[str, float]):
        """Print human-readable interpretation of model results."""
        print(f"\nInterpretable Results for {state} ({period}):")
        
        # Get baseline probability
        intercept = fit_result.loc['(Intercept)', 'Estimate']
        baseline_prob = 1 / (1 + np.exp(-intercept))
        print(f"• Baseline probability of {state} during {period}: {baseline_prob:.3f} ({baseline_prob*100:.1f}%)")
        
        # Process each parameter in the results
        for term, row in fit_result.iterrows():
            if term == '(Intercept)':
                continue
                
            effect_dir = "increases" if row['Estimate'] > 0 else "decreases"
            
            # Handle interaction terms
            if ":" in term:
                terms = term.split(":")
                term1 = self._get_display_name_of_term(terms[0])
                term2 = self._get_display_name_of_term(terms[1])
                print(f"• {term1} effect {effect_dir} with higher {term2} {row['Sig']}")
            
            # Skip higher order terms that will be handled as part of their base term
            elif term.endswith("_sq") or term.endswith("_cub") or term.endswith("_qrt"):
                # This is a higher-order term, we'll handle it with other terms of same base
                continue
                
            # Handle main effects
            elif term in self.model_builder.get_z_names():
                display_name = self._get_display_name_of_term(term)
                
                # Calculate marginal effect
                var_name = term.replace('_z', '')
                std = stds.get(var_name, 1.0)
                scale = self.scale_factors.get(var_name, 1.0)
                scaled_effect = row['Estimate'] * (scale / std)
                margin_effect = baseline_prob * (1 - baseline_prob) * scaled_effect
                
                # Check for non-linear patterns (higher order terms)
                has_sq = f"{term}_sq" in fit_result.index
                has_cub = f"{term}_cub" in fit_result.index
                has_qrt = f"{term}_qrt" in fit_result.index
                
                if has_sq or has_cub or has_qrt:
                    pattern = "non-linear"
                    if has_sq and fit_result.loc[f"{term}_sq", 'Estimate'] > 0:
                        pattern = "U-shaped"
                    elif has_sq:
                        pattern = "∩-shaped"
                    
                    print(f"• {display_name} has a {pattern} effect {row['Sig']}")
                    
                    # Add information about higher-order components
                    orders = []
                    if has_sq: orders.append("quadratic")
                    if has_cub: orders.append("cubic") 
                    if has_qrt: orders.append("quartic")
                    if orders:
                        print(f"  (Includes {', '.join(orders)} components)")
                else:
                    # Format message based on the variable
                    if term == 'temp_z':
                        print(f"• Temperature effect: 5°C increase causes {abs(margin_effect*100):.1f}% {effect_dir} in probability {row['Sig']}")
                    elif term == 'age_z':
                        print(f"• Age effect: 1 year increase causes {abs(margin_effect*100):.1f}% {effect_dir} in probability {row['Sig']}")
                    elif term == 'weight_z':
                        print(f"• Weight effect: 30kg increase causes {abs(margin_effect*100):.1f}% {effect_dir} in probability {row['Sig']}")
                    elif term == 'hod_z':
                        print(f"• Hours of daylight effect: 1 hour increase causes {abs(margin_effect*100):.1f}% {effect_dir} in probability {row['Sig']}")
                    elif term == 'time_z':
                        print(f"• Time of day effect: Behavior tends to {effect_dir} as day progresses {row['Sig']}")
                    else:
                        print(f"• {display_name} effect: Causes {abs(margin_effect*100):.1f}% {effect_dir} in probability {row['Sig']}")
            
            # Handle any other unrecognized terms
            else:
                print(f"• Unknown term {term}: Estimate = {row['Estimate']:.4f} {row['Sig']}")


    def _create_results_table(self, results: Dict) -> str:
        """
        Create a formatted table of results
        
        Parameters:
        -----------
        results : Dict
            Analysis results
            
        Returns:
        --------
        str
            Formatted table as string
        """
        from tabulate import tabulate
        
        # Prepare table data
        table_data = []
        
        # Determine which periods to include
        periods = []
        if 'day' in results and self.config.visuals.temperature_graph.daynight in ['day', 'both']:
            periods.append('day')
        if 'night' in results and self.config.visuals.temperature_graph.daynight in ['night', 'both']:
            periods.append('night')
        
        # Create headers
        header = ["Parameter", "Behavior"]
        subheader = ["", ""]
        
        for period in periods:
            period_title = f"{period.capitalize()} Effects"
            header.extend([period_title, "", "", "", ""])
            subheader.extend(["Prob. Change", "Coef.", "OR", "p-value", "Sig."])
        
        table_data.append(header)
        table_data.append(subheader)
        
        # Find all parameters used in any model
        all_params = set(['(Intercept)'])  # Intercept always exists
        
        for period in periods:
            for state in self.config.analysis.hmm.states:
                if state in results[period]['states'] and 'fit_result' in results[period]['states'][state]:
                    fit_result = results[period]['states'][state]['fit_result']
                    all_params.update(fit_result.index)
        
        # Create table rows for each parameter
        for param in sorted(all_params, key=lambda x: 
                            (0 if x == '(Intercept)' else
                             1 if x in self.model_builder.get_z_names() else
                             2 if any(sq in x for sq in ['_sq', '_cub', '_qrt']) else
                             3 if ':' in x else 4)):
                            
            if param in self.param_labels:
                # Add parameter header row
                table_data.append([self.param_labels[param], ""])
                
                # Add rows for each behavior state
                for state in self.config.analysis.hmm.states:
                    row_data = ["", state.capitalize()]
                    
                    # Add data for each period
                    for period in periods:
                        # Get results for this state and parameter for the current period
                        fit_result = None
                        if (state in results[period]['states'] and 
                            'fit_result' in results[period]['states'][state] and
                            param in results[period]['states'][state]['fit_result'].index):
                            fit_result = results[period]['states'][state]['fit_result'].loc[param]
                        
                        # Add formatted values for this period
                        row_data = self._format_table_row(param, row_data, state, fit_result, results[period])
                    
                    # Add completed row to table
                    table_data.append(row_data)
            else:
                print("UNHANDLED PARAMETER ", param)

        # Add Model Fit Statistics section
        table_data.append(["Model Fit Statistics", ""])
        
        # Add rows for AIC/BIC for each state and period
        for state in self.config.analysis.hmm.states:
            row_data = ["", state.capitalize()]
            
            for period in periods:
                aic = "N/A"
                bic = "N/A"
                
                if (state in results[period]['states'] and 
                    'model' in results[period]['states'][state] and
                    hasattr(results[period]['states'][state]['model'], 'AIC')):
                    model = results[period]['states'][state]['model']
                    aic = f"AIC: {model.AIC:.1f}"
                    bic = f"BIC: {model.BIC:.1f}"
                
                row_data.extend([aic, bic, "", "", ""])
            
            table_data.append(row_data)
        
        # Format table using tabulate
        table_str = tabulate(table_data, headers="firstrow", tablefmt="fancy_grid")
        
        # Add explanatory notes
        note = (
            "\nNotes:\n"
            "• Prob. Change: Probability change for the given parameter change\n"
            "• Coef.: Original coefficient (back-transformed for interpretability)\n"
            "• OR: Odds Ratio\n"
            "• Sig.: Statistical significance (* p<0.05, ** p<0.01, *** p<0.001, . p<0.1)\n"
            "• Temperature effects are per 5°C increase; Age effects are per year; Weight effects are per 30kg"
        )
        
        return table_str + note

    def _format_table_row(self, param: str, row_data: List, state: str, period_result, period_info) -> List:
        """
        Format a table row with results for a specific parameter, state, and period
        
        Parameters:
        -----------
        param : str
            The parameter name
        row_data : List
            Current row data to append to
        state : str
            The behavior state
        period_result : pd.Series or None
            Model results for this parameter, or None if not available
        period_info : Dict
            Period information including standardization and scale factors
            
        Returns:
        --------
        List
            Updated row data with formatted values
        """
        if period_result is None:
            # No results available for this parameter/state/period
            return row_data + ["N/A", "N/A", "N/A", "N/A", "N/A"]
        
        # Get baseline probability
        baseline_prob = None
        for s, state_results in period_info['states'].items():
            if s == state and 'fit_result' in state_results and '(Intercept)' in state_results['fit_result'].index:
                intercept = state_results['fit_result'].loc['(Intercept)', 'Estimate']
                baseline_prob = 1 / (1 + np.exp(-intercept))
                break
        
        # Standardization and scaling info
        means = period_info['standardization_info']['means']
        stds = period_info['standardization_info']['stds']
        scale_factors = period_info['scale_factors']

        
        
        # Format probability change based on parameter type
        if param == '(Intercept)':
            # For baseline, show actual probability
            prob_text = f"{baseline_prob*100:.1f}%"
        else:
            if baseline_prob is not None:
                if param in ['temp_z', 'age_z', 'weight_z', 'time_z']:
                    # Linear main effects - calculate marginal effect
                    var_name = param.replace('_z', '')
                    std = stds.get(var_name, 1.0)
                    scale_factor = scale_factors.get(var_name, 1.0)
                    scaled_effect = period_result['Estimate'] * (scale_factor / std)
                    margin_effect = baseline_prob * (1 - baseline_prob) * scaled_effect
                    prob_change = margin_effect * 100
                    
                    direction = "+" if prob_change > 0 else ""
                    prob_text = f"{direction}{prob_change:.1f}%"
                    
                elif param.endswith(('_sq', '_cub', '_qrt')):
                    # Polynomial terms - show curve type
                    if param.endswith('_sq'):
                        curve_type = "U-shaped" if period_result['Estimate'] > 0 else "∩-shaped"
                        prob_text = curve_type
                    elif param.endswith('_cub'):
                        prob_text = "Cubic effect"
                    else:
                        prob_text = "Quartic effect"
                        
                # elif param == 'rel_time':
                #     # Time of day linear effect
                #     margin_effect = baseline_prob * (1 - baseline_prob) * period_result['Estimate']
                #     prob_change = margin_effect * 100
                #     direction = "+" if prob_change > 0 else ""
                #     prob_text = f"{direction}{prob_change:.1f}%"
                    
                elif ':' in param:
                    # Interaction terms
                    margin_effect = baseline_prob * (1 - baseline_prob) * period_result['Estimate']
                    prob_change = margin_effect * 100
                    direction = "+" if prob_change > 0 else ""
                    prob_text = f"{direction}{prob_change:.1f}%"
                    
                else:
                    # Generic case
                    margin_effect = baseline_prob * (1 - baseline_prob) * period_result['Estimate']
                    prob_change = margin_effect * 100
                    direction = "+" if prob_change > 0 else ""
                    prob_text = f"{direction}{prob_change:.1f}%"
            else:
                # Fallback if baseline probability not available
                prob_effect = period_result['Prob'] - 0.5
                direction = "+" if prob_effect > 0 else ""
                prob_text = f"{direction}{prob_effect*100:.1f}%"
        
        # Format coefficient
        if param in ['temp_z', 'age_z', 'weight_z', 'time_z']:
            # Back-transform standardized coefficients
            var_name = param.replace('_z', '')
            std = stds.get(var_name, 1.0)
            scale_factor = scale_factors.get(var_name, 1.0)
            orig_coef = period_result['Estimate'] * (scale_factor / std)
        else:
            # Use coefficient as is
            orig_coef = period_result['Estimate']
        
        # Format numeric values
        coef_text = f"{orig_coef:.4f}" if isinstance(orig_coef, (int, float)) else str(orig_coef)
        odds_ratio = period_result['OR']
        odds_ratio_text = f"{odds_ratio:.4f}" if isinstance(odds_ratio, (int, float)) else str(odds_ratio)
        
        # Format p-value
        p_val = period_result['P-val']
        p_val_text = "<0.001" if p_val < 0.001 else f"{p_val:.4f}"
        
        # Get significance
        sig = period_result['Sig']
        
        # Return formatted row
        return row_data + [prob_text, coef_text, odds_ratio_text, p_val_text, sig]

    def _export_results_to_datafile(self, results: Dict, filename: str):
        """
        Export results to a CSV file for easy import into Excel or other tools
        
        Parameters:
        -----------
        results : Dict
            Analysis results
        filename : str
            Path to save CSV file
        """
        # Determine which periods to include
        periods = []
        if 'day' in results and self.config.visuals.temperature_graph.daynight in ['day', 'both']:
            periods.append('day')
        if 'night' in results and self.config.visuals.temperature_graph.daynight in ['night', 'both']:
            periods.append('night')
        
        # Create column headers
        columns = ["Parameter", "Behavior"]
        for period in periods:
            period_cap = period.capitalize()
            columns.extend([
                f"{period_cap} Prob Change", 
                f"{period_cap} Coef", 
                f"{period_cap} OR", 
                f"{period_cap} p-value", 
                f"{period_cap} Sig"
            ])
        
        # Create a table for CSV export
        table_data = []
        
        
        # Find all parameters used in any model
        all_params = set(['(Intercept)'])  # Intercept always exists
        
        for period in periods:
            for state in self.config.analysis.hmm.states:
                if state in results[period]['states'] and 'fit_result' in results[period]['states'][state]:
                    fit_result = results[period]['states'][state]['fit_result']
                    all_params.update(fit_result.index)
        
        # Create table rows for each parameter
        for param in sorted(all_params, key=lambda x: 
                        (0 if x == '(Intercept)' else
                        1 if x in self.model_builder.get_z_names() else
                        2 if any(sq in x for sq in ['_sq', '_cub', '_qrt']) else
                        3 if ':' in x else 4)):
                        
            if param in self.param_labels:
                # Add parameter header row
                parameter_label = self.param_labels[param]
                
                # Add rows for each behavior state
                for state in self.config.analysis.hmm.states:
                    row = [parameter_label, state.capitalize()]
                    
                    # Add data for each period
                    for period in periods:
                        # Get results for this state and parameter for the current period
                        fit_result = None
                        if (state in results[period]['states'] and 
                            'fit_result' in results[period]['states'][state] and
                            param in results[period]['states'][state]['fit_result'].index):
                            fit_result = results[period]['states'][state]['fit_result'].loc[param]
                        
                        # Add formatted values for this period
                        row = self._format_table_row(param, row, state, fit_result, results[period])
                    
                    table_data.append(row)
        
        # Create a DataFrame and save to CSV
        df = pd.DataFrame(table_data, columns=columns)
        df.to_excel(filename, index=False)


    def calculate_auc(self, time_values: np.ndarray, probabilities: np.ndarray) -> float:
        """
        Calculate area under the curve using trapezoidal integration
        """
        # Normalize x-axis to 0-1 range for comparable areas
        time_normalized = (time_values - time_values.min()) / (time_values.max() - time_values.min())
        return np.trapz(probabilities, time_normalized)

    def analyze_temperature_effects(self, results: Dict, period: str, state: str):
        """
        Analyze how temperature affects behavior probability across the day
        """
        period_results = results[period]
        means = period_results['standardization_info']['means']
        stds = period_results['standardization_info']['stds']
        
        # Get model results for the state
        model_result = period_results['states'][state]['fit_result']
        
        # Calculate probabilities across the day for each temperature
        time_values = np.linspace(0, 1440, 100)  # Full day in minutes
        temp_values = np.linspace(
            means['temp'] - 4 * stds['temp'],
            means['temp'] + 4 * stds['temp'],
            9
        )
        
        aucs = []
        for temp in temp_values:
            probabilities = []
            temp_z = (temp - means['temp']) / stds['temp']
            
            for time in time_values:
                time_z = (time - means['time']) / stds['time']

                logit = self._calculate_logit(time_z, 'temp', temp_z, model_result)
                prob = 1 / (1 + np.exp(-logit))
                probabilities.append(prob)
                
            auc = self.calculate_auc(time_values, np.array(probabilities))
            aucs.append(auc)
        
        # Print analysis
        print(f"\nTemperature Effect Analysis for {state} ({period}):")
        print(f"{'Temperature':>12} {'AUC':>10} {'% Change':>10}")
        print("-" * 32)
        
        baseline_auc = aucs[4]  # Middle temperature
        for temp, auc in zip(temp_values, aucs):
            pct_change = ((auc - baseline_auc) / baseline_auc) * 100
            print(f"{temp:>12.1f}°C {auc:>10.3f} {pct_change:>10.1f}%")



    def _calculate_logit(self, time_z, by, b_z, model_result, orthogonal_data=None):
        """Calculate logit value for a given time and predictor value using model coefficients"""
        logit = model_result.loc['(Intercept)', 'Estimate']
        
        # Handle time polynomials with orthogonal polynomials if available
        if orthogonal_data is not None:
            max_degree = orthogonal_data['max_degree']
            
            # Scale time_z to [-1, 1] for Legendre polynomial calculation
            # Need to use the original time range for proper scaling
            orig_min = orthogonal_data['original_time'].min()
            orig_max = orthogonal_data['original_time'].max()
            scaled_time = 2 * (time_z - orig_min) / (orig_max - orig_min) - 1
            
            # Calculate orthogonal polynomial values for this time point
            time_polys = np.polynomial.legendre.legval(scaled_time, np.eye(max_degree + 1))
            
            # Add main time polynomial terms
            for degree in range(1, max_degree + 1):
                suffix = '' if degree == 1 else ('_sq' if degree == 2 else ('_cub' if degree == 3 else '_qrt'))
                term = f'time_z{suffix}'
                if term in model_result.index:
                    logit += model_result.loc[term, 'Estimate'] * time_polys[degree]
            
            # Add the by variable (temperature, etc.) main effect
            if f'{by}_z' in model_result.index:
                logit += model_result.loc[f'{by}_z', 'Estimate'] * b_z
                
            # Add quadratic term for by variable if it exists
            if f'{by}_z_sq' in model_result.index:
                logit += model_result.loc[f'{by}_z_sq', 'Estimate'] * (b_z ** 2)
            
            # Add interaction terms between by variable and orthogonal time polynomials
            for degree in range(1, max_degree + 1):
                suffix = '' if degree == 1 else ('_sq' if degree == 2 else ('_cub' if degree == 3 else '_qrt'))
                interaction_term = f'{by}_z:time_z{suffix}'
                rev_interaction_term = f'time_z{suffix}:{by}_z'
                
                # Check both possible orderings of the interaction term
                if interaction_term in model_result.index:
                    logit += model_result.loc[interaction_term, 'Estimate'] * b_z * time_polys[degree]
                elif rev_interaction_term in model_result.index:
                    logit += model_result.loc[rev_interaction_term, 'Estimate'] * b_z * time_polys[degree]
                
                # Also check for interactions with squared by variable
                sq_interaction = f'{by}_z_sq:time_z{suffix}'
                rev_sq_interaction = f'time_z{suffix}:{by}_z_sq'
                
                if sq_interaction in model_result.index:
                    logit += model_result.loc[sq_interaction, 'Estimate'] * (b_z ** 2) * time_polys[degree]
                elif rev_sq_interaction in model_result.index:
                    logit += model_result.loc[rev_sq_interaction, 'Estimate'] * (b_z ** 2) * time_polys[degree]
        else:
            # Use regular polynomial terms
            coef_patterns = {
                f'{by}_z': b_z,
                f'{by}_z_sq': b_z ** 2,
                'time_z': time_z,
                'time_z_sq': time_z ** 2,
                'time_z_cub': time_z ** 3,
                'time_z_qrt': time_z ** 4,
                f'{by}_z:time_z': b_z * time_z,
                f'{by}_z_sq:time_z': (b_z ** 2) * time_z,
                f'{by}_z:time_z_sq': b_z * (time_z ** 2),
                f'{by}_z:time_z_cub': b_z * (time_z ** 3),
                f'{by}_z:time_z_qrt': b_z * (time_z ** 4)
            }
            
            # Apply each coefficient if it exists in the model
            for coef_name, coef_value in coef_patterns.items():
                if coef_name in model_result.index:
                    logit += model_result.loc[coef_name, 'Estimate'] * coef_value

        return logit

    def plot_state_probabilities(self, results: Dict, save_path: Optional[str] = None, by:str='temp'):
        """
        Plot state probabilities over time of day and temperature.
        
        Parameters:
        -----------
        results : Dict
            Analysis results from the analyze method
        save_path : str, optional
            Path to save the generated figure
        """

        
        # Check if we have results to plot
        if not results:
            print("No results to plot")
            return
            
        # Determine which periods to plot
        periods = []
        if 'day' in results and self.config.visuals.temperature_graph.daynight in ['day', 'both']:
            periods.append('day')
        if 'night' in results and self.config.visuals.temperature_graph.daynight in ['night', 'both']:
            periods.append('night')
            
        if 'both' in results:
            periods.append('both')
            
        if not periods:
            print("No day/night periods to plot")
            return
        
        # Set up plot style
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 13,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 11,
            'figure.titlesize': 16
        })
        
        # Iterate through each period
        for period in periods:
            period_results = results[period]
            
            # Get standardization info for interpretation
            means = period_results['standardization_info']['means']
            stds = period_results['standardization_info']['stds']
    
            width_in_inches = 190/25.4
            height_in_inches = width_in_inches * 0.5  # Slightly shorter to reduce white space
            
            # Create a figure with subplots for each state
            fig, axes = plt.subplots(nrows=1, ncols=len(self.config.analysis.hmm.states), 
                                    figsize=(width_in_inches, height_in_inches), sharey=True)
            # fig.suptitle(f"Predicted Behavior Probabilities - {period.capitalize()}", fontsize=16)
            
            # Create a color palette for temperature lines
            by_colors = sns.color_palette("coolwarm", n_colors=5)
            
            # Temperature range to plot (unstandardized)
            by_values = np.linspace(
                means[by] - 2 * stds[by],
                means[by] + 2 * stds[by],
                5
            )
            # print(by_values)
            
            # Convert time domain from minutes to a range covering the period
            # For day period, use daytime hours (e.g., 6am-8pm)
            # For night period, use nighttime hours (e.g., 8pm-6am)
            # if period == 'day':
            # time_min, time_max = period_results['time_domain'] 
            time_min, time_max = 0, 1440
            # Convert minutes to hours for display
            hour_min = int(time_min / 60) 
            hour_max = int(time_max / 60)
            
            # Create time values in minutes (for calculations)
            time_values = np.linspace(time_min, time_max, 100)
            
            # Create corresponding hour values (for display)
            hour_values = time_values / 60
            # else:  # night period
            #     time_min, time_max = period_results['time_domain']
            #     # Convert minutes to hours for display
            #     hour_min = int(time_min / 60)
            #     hour_max = int(time_max / 60)
                
            #     # Create time values in minutes (for calculations)
            #     time_values = np.linspace(time_min, time_max, 100)
                
            #     # Create corresponding hour values (for display)
            #     hour_values = time_values / 60
            
            # Plot each state in a separate subplot
            for i, state in enumerate(self.config.analysis.hmm.states):
                ax = axes[i] if len(self.config.analysis.hmm.states) > 1 else axes
                ax.set_title(f"{state.capitalize()} State")
                ax.set_xlabel("Time of Day (hours)")
                
                # Format x-axis with hour labels
                ax.set_xticks(np.linspace(hour_min, hour_max, 7))
                ax.set_xticklabels([f"{int(h)}h" for h in np.linspace(hour_min, hour_max, 7)])
                ax.tick_params(axis='x', rotation = 90)
                if i == 0:
                    ax.set_ylabel("Probability")
                    
                # Set y-axis limits for probability
                ax.set_ylim(0, 1)
                orthogonal_data = None
                # Only plot if we have model results for this state
                if (state in period_results['states'] and 
                    'model' in period_results['states'][state] and 
                    'fit_result' in period_results['states'][state]):
                    
                    model = period_results['states'][state]['model']
                    model_result = period_results['states'][state]['fit_result']
                    
                    
                    if hasattr(model, 'data') and hasattr(model.data, 'attrs') and 'ortho_time_mapping' in model.data.attrs:
                        orthogonal_data = model.data.attrs['ortho_time_mapping']
                        # print("HAS ORTHOGONAL")
                        # print(orthogonal_data)

                    # Update this for non temperature 'by'
                    # Create legend labels with actual temperature values
                    if by=='temp':
                        by_labels = [f"{b:.1f}°C" for b in by_values]
                    if by=='hod':
                        by_labels = [f"{b:.1f} hours" for b in by_values]


                        
                    # Plot a line for each temperature value
                    for j, b in enumerate(by_values):
                        # Standardize tby variable
                        b_z = (b - means[by]) / stds[by]

                        
                        # Calculate probabilities for this temperature across time range
                        probabilities = []
                        for time in time_values:
                            # Standardize time
                            time_z = (time - means['time']) / stds['time']
                            logit = self._calculate_logit(time_z, by, b_z, model_result, orthogonal_data)
                            prob = 1 / (1 + np.exp(-logit))
                            probabilities.append(prob)
                        # Plot this temperature line (using hour_values for x-axis)
                        ax.plot(hour_values, probabilities, color=by_colors[j], 
                                label=by_labels[j], linewidth=2.5)
                    
                    # Add legend to the first subplot only
                    if i == 2:

                        t = "Temperature" if by == 'temp' else ('Hours of Daylight' if by == 'hod' else by.capitalize())
                        ax.legend(title=t, loc='upper right')
                else:
                    ax.text(0.5, 0.5, "No model available", ha='center', va='center', 
                            transform=ax.transAxes, fontsize=12)
            
            # Adjust layout
            plt.tight_layout()
            fig.subplots_adjust(top=0.84)
            
            # Save the figure if requested
            if save_path:
                period_save_path = save_path.replace('.png', f'_{period}.png')
                plt.savefig(period_save_path, dpi=300, bbox_inches='tight')
                print(f"Figure saved to {period_save_path}")
            
            plt.show()
