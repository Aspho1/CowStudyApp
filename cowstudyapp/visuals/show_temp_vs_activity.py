import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.regression.mixed_linear_model as mlm
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

import ephem
# from datetime import datetime




from cowstudyapp.utils import from_posix_col
from cowstudyapp.config import ConfigManager

class GrazingVersusTemperature:
    def __init__(self, config: ConfigManager):
        self.config = config
        # Set up Norris, MT observer
        self.observer = ephem.Observer()
        self.observer.lat = '45.575'
        self.observer.lon = '-111.625'
        self.observer.elevation = 1715
        self.observer.pressure = 0
        self.observer.horizon = '-0:34'
        self._model_counter = []  # Track model numbers

    def _get_sun_times(self, date):
        """Calculate sunrise/sunset times for a given date"""
        self.observer.date = date.strftime('%Y/%m/%d 00:00:00')
        sunrise = pd.to_datetime(str(self.observer.next_rising(ephem.Sun()))).tz_localize('UTC')
        
        self.observer.date = date.strftime('%Y/%m/%d 12:00:00')
        sunset = pd.to_datetime(str(self.observer.next_setting(ephem.Sun()))).tz_localize('UTC')
        
        return sunrise, sunset


    def _add_temp_col(self, df):
        """Add temperature data by joining with temperature dataset"""
        if 'temperature' not in df.columns:
            if 'temperature_gps' in df.columns:
                df.rename(columns={'temperature_gps': 'temperature'}, inplace=True)
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
                
                # Rename the temperature column
                df.rename(columns={'temperature_gps': 'temperature'}, inplace=True)
                
                # Drop the extra device_id column from the join
                df.drop('device_id', axis=1, inplace=True)
        
        meta_info_df = pd.read_excel(self.config.io.cow_info_path)
        meta_info_df["Age"] = 2022 - meta_info_df['year_b']
        meta_info_df.set_index('collar_id',inplace=True)
        self.age_dict = meta_info_df['Age'].to_dict()
        meta_info_df['BW_preg'] = meta_info_df['BW_preg']*0.4535924 # lbs to kg
        self.weight_dict = meta_info_df['BW_preg'].to_dict()
                                

        # df["Age"] = df['ID'].map(age_dict)
        # df["Weight"] = df['ID'].map(weight_dict)
        # df = df[df.Weight > 0]
        return df
        

    def compare_temp_behavior(self, df):
        # Prepare datetime columns
        df['mt'] = from_posix_col(df['posix_time'], self.config.analysis.timezone)
        df['date'] = df['mt'].dt.date
        df['hour'] = df['mt'].dt.hour
        
        # Add temperature data
        df = self._add_temp_col(df)
        
        # Create 3-hour time windows
        df['time_window'] = (df['hour'] // 3) * 3
        
        # Ensure consistent dtypes
        df['ID'] = df['ID'].astype('int64')
        df['time_window'] = df['time_window'].astype('int64')
        
        # Initialize period column for individual records
        df['period_individual'] = pd.NA
        
        # Dictionary to store sunrise/sunset times for debugging
        sun_times = {}
        
        # Dictionary to store day vs night temperatures for verification
        day_temps = []
        night_temps = []
        
        # Classify time periods based on actual sunrise/sunset
        for date, day_data in df.groupby('date'):
            # Get sunrise time
            self.observer.date = date.strftime('%Y/%m/%d 00:00:00')
            sunrise = pd.to_datetime(str(self.observer.next_rising(ephem.Sun()))).tz_localize('UTC').tz_convert(self.config.analysis.timezone)
            
            # Get sunset time
            self.observer.date = date.strftime('%Y/%m/%d 12:00:00')
            sunset = pd.to_datetime(str(self.observer.next_setting(ephem.Sun()))).tz_localize('UTC').tz_convert(self.config.analysis.timezone)
            
            # Store times for debugging
            sun_times[date] = {
                'sunrise': sunrise.strftime('%H:%M'),
                'sunset': sunset.strftime('%H:%M')
            }
            
            # Print some detailed info for debugging
            if date == list(df['date'].unique())[0]:  # First date
                print(f"\nDetailed time analysis for {date}:")
                print(f"  Sunrise (UTC): {pd.to_datetime(str(self.observer.next_rising(ephem.Sun()))).tz_localize('UTC')}")
                print(f"  Sunrise (local): {sunrise}")
                print(f"  Sunset (UTC): {pd.to_datetime(str(self.observer.next_setting(ephem.Sun()))).tz_localize('UTC')}")
                print(f"  Sunset (local): {sunset}")
                print(f"  Sunrise hour: {sunrise.hour}, Sunset hour: {sunset.hour}")
                print(f"  Day period hours: {[h for h in range(sunrise.hour-1, sunset.hour+2)]}")
            
            # Update period for all records on this date
            mask = (df['date'] == date)
            day_hours = list(range(sunrise.hour-1, sunset.hour+2))
            
            # Apply classification - NOTE: Check this logic carefully
            df.loc[mask, 'period_individual'] = df.loc[mask, 'hour'].apply(
                lambda h: 'Day' if h in day_hours else 'Night'
            )
            
            # Gather temperature data for day vs night for this date
            date_records = df[mask]
            day_records = date_records[date_records['period_individual'] == 'Day']
            night_records = date_records[date_records['period_individual'] == 'Night']
            
            if not day_records.empty:
                day_temps.append({
                    'date': date,
                    'mean_temp': day_records['temperature'].mean(),
                    'min_temp': day_records['temperature'].min(),
                    'max_temp': day_records['temperature'].max()
                })
            
            if not night_records.empty:
                night_temps.append({
                    'date': date,
                    'mean_temp': night_records['temperature'].mean(),
                    'min_temp': night_records['temperature'].min(),
                    'max_temp': night_records['temperature'].max()
                })
        
        # Print temperature comparison
        day_temp_df = pd.DataFrame(day_temps)
        night_temp_df = pd.DataFrame(night_temps)
        
        print("\nDay vs Night Temperature Comparison:")
        print(f"Day mean temperature: {day_temp_df['mean_temp'].mean():.2f}°C")
        print(f"Night mean temperature: {night_temp_df['mean_temp'].mean():.2f}°C")
        print(f"Overall temperature difference: {day_temp_df['mean_temp'].mean() - night_temp_df['mean_temp'].mean():.2f}°C")
        
        # Calculate some stats on the temperature ranges
        print("\nTemperature ranges:")
        print(f"Day temperatures: {day_temp_df['min_temp'].min():.1f}°C to {day_temp_df['max_temp'].max():.1f}°C")
        print(f"Night temperatures: {night_temp_df['min_temp'].min():.1f}°C to {night_temp_df['max_temp'].max():.1f}°C")
        
        # Plot day vs night temperature distributions to verify
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df, x='temperature', hue='period_individual', 
                    kde=True, stat='density', common_norm=False, ax=ax)
        ax.set_title('Temperature Distribution: Day vs Night')
        plt.savefig(
            os.path.join(self.config.visuals.visuals_root_path, 'day_night_temp_distribution.png'),
            bbox_inches='tight',
            dpi=300
        )
        plt.close()

        # Log some sunrise/sunset times for verification
        print("\nSample of sunrise/sunset times:")
        for i, (date, times) in enumerate(list(sun_times.items())[:5]):  # First 5 days
            print(f"Date: {date}, Sunrise: {times['sunrise']}, Sunset: {times['sunset']}")
        




        
        # Check if we have any windows with mixed classifications
        mixed_windows = 0
        for (date, id, window), group in df.groupby(['date', 'ID', 'time_window']):
            if group['period_individual'].nunique() > 1:
                mixed_windows += 1
        
        print(f"\nFound {mixed_windows} time windows with mixed day/night classifications")
        
        # Group by date, ID, and time window to determine the majority period for each window
        window_periods = df.groupby(['date', 'ID', 'time_window'])['period_individual'].agg(
            lambda x: x.value_counts().index[0] if not x.value_counts().empty else "Unknown"
        ).reset_index()
        
        # Ensure consistent dtypes in window_periods
        window_periods['ID'] = window_periods['ID'].astype('int64')
        window_periods['time_window'] = window_periods['time_window'].astype('int64')
        
        # Merge window period back to ensure entire windows have consistent classification
        df = pd.merge(
            df,
            window_periods,
            on=['date', 'ID', 'time_window'],
            how='left'
        )
        
        # Rename the merged period column
        df.rename(columns={'period_individual_y': 'period'}, inplace=True)
        
        # Initialize the behavior DataFrame with 3-hour windows
        window_data = []
        
        # Group by cow, date, time window and day/night period
        for (cow_id, date, window, period), group in df.groupby(['ID', 'date', 'time_window', 'period']):
            if self.weight_dict[cow_id] <= 0:
                continue
            row_data = {
                'cow_id': cow_id,
                'Age': self.age_dict[cow_id],
                'Weight': self.weight_dict[cow_id],
                'date': date,
                'time_window': window,
                'period': period,
                'temperature': group['temperature'].mean(),
                'window_label': f"{window:02d}:00-{(window+3):02d}:00"
            }
            
            # Calculate proportion for each state
            for state in self.config.analysis.hmm.states:
                row_data[state] = (group['predicted_state'] == state).mean()
                
            window_data.append(row_data)
        
        behavioral_data = pd.DataFrame(window_data)
        
        
        # Categorize temperature into meaningful ranges as per Adams et al.
        behavioral_data['temp_category'] = pd.cut(
            behavioral_data['temperature'], 
            bins=[-float('inf'), -7, 4, float('inf')],
            labels=['Cold (< -7°C)', 'Cool (-7 to 4°C)', 'Mild (> 4°C)']
        )
        self._plotting_manager(behavioral_data=behavioral_data)
        return behavioral_data

    def _plotting_manager(self, behavioral_data):

        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 18,
            'axes.labelsize': 16,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14
        })
        
        self.colors = {
            'Grazing': {'points': '#2ecc71', 'line': '#1a8c4a'},  # Green (lighter for points, darker for line)
            'Resting': {'points': '#3498db', 'line': '#1a5c8c'},  # Blue (lighter for points, darker for line)
            'Traveling': {'points': '#e74c3c', 'line': '#992d22'}  # Red (lighter for points, darker for line)
        }

        if self.config.visuals.temperature_graph.daynight == 'both':
            fig, axs = plt.subplots(1, 2, figsize=(20, 10))
            self._plot_mixed_effects_analysis_oneperiod(behavioral_data=behavioral_data,ax=axs[0], period='day')
            self._plot_mixed_effects_analysis_oneperiod(behavioral_data=behavioral_data,ax=axs[1], period='night')
        
        elif self.config.visuals.temperature_graph.daynight == 'day':
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            self._plot_mixed_effects_analysis_oneperiod(behavioral_data=behavioral_data,ax=ax, period='day')
        
        elif self.config.visuals.temperature_graph.daynight == 'night':
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            self._plot_mixed_effects_analysis_oneperiod(behavioral_data=behavioral_data,ax=ax, period='night')
        else:
            raise ValueError("Invalid selection for `config.visuals.temperature_graph.daynight`")


        fig.suptitle("Temperature Effects on Cow Behavior (Hours per 3-Hour Window)\n* p<0.05, ** p<0.01, *** p<0.001", fontsize=20, y=0.98)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.config.visuals.visuals_root_path, f'temp_behavior_{self.config.visuals.temperature_graph.daynight}.png'),
            bbox_inches='tight',
            dpi=300
        )
        plt.close()

        # self._plot_temperature_category_analysis(behavioral_data)
        
    def _plot_mixed_effects_analysis_oneperiod(self, behavioral_data, ax, period):
        period = period.capitalize()

        # Create a summary of statistics for the figure
        stat_summary = []
        
        # Store models by state for later table export
        models_by_state = {}
        
        # Convert proportion to hours (3-hour windows)
        for state in self.config.analysis.hmm.states:
            behavioral_data[f"{state}_hours"] = behavioral_data[state] * 3
        
        period_data = behavioral_data[behavioral_data['period'] == period]
        
        # Print period summary statistics
        self._print_period_summary(period_data, period)
        
        # Define which models to run - easy to comment/uncomment as needed
        models_to_run = {
            'linear': False,                 # Basic temperature-only model 
            'main_effects': False,           # Temperature + age + weight
            'age_weight': False,             # Age × weight interaction
            'temp_interactions': False,      # Temperature × age/weight interactions
            'all_interactions': True,       # All possible interactions
            'mixed_effects': False,          # Random effects for cow_id
            'mixed_with_interactions': False, # Mixed effects with interactions
            'quadratic': False,             # Quadratic temperature effect
            'cubic': False                  # Cubic temperature effect
        }
        
        # Initialize model counter
        self._model_counter = []
        
        # Plot each behavior
        for i, state in enumerate(self.config.analysis.hmm.states):
            print(f"\n{'-'*40}")
            print(f"Analysis for {state} during {period} period:")
            print(f"{'-'*40}")
            
            # Prepare centered variables for interactions
            period_data = self._prepare_centered_variables(period_data)
            
            # Run all requested statistical models
            models_results = self._run_statistical_models(
                period_data, 
                state,
                models_to_run
            )
            
            # Store models for this state
            models_by_state[state] = models_results
            
            # Select best model based on scientific judgment, not just statistics
            best_model, best_model_name, plot_curve = self._select_best_model(
                models_results, 
                state, 
                period,
                scientific_judgment=True  # Prioritize biological plausibility
            )
            
            # Store stats from linear model for legend
            linear_model = models_results.get('linear', None)
            if linear_model:
                significance = self._get_significance(linear_model.pvalues['temperature'])
                
                stat_summary.append({
                    'period': period,
                    'state': state,
                    'slope': linear_model.params['temperature'],
                    'intercept': linear_model.params['Intercept'],
                    'p_value': linear_model.pvalues['temperature'],
                    'significance': significance,
                    'model_description': best_model_name if best_model != linear_model else "temperature only"
                })
            
            # Create scatter plot with linear regression
            self._plot_behavior_line(ax, period_data, state, i)
            
            # If selected model is polynomial and we want to show curves, add the curve
            if plot_curve and 'show_curve' in self.config.visuals.temperature_graph and self.config.visuals.temperature_graph.show_curve:
                poly_model = models_results.get('quadratic' if 'quadratic' in best_model_name else 'cubic', None)
                if poly_model:
                    self._add_polynomial_curve(ax, poly_model, period_data, state, i)
        
        # Set axis labels and titles
        self._format_plot_axes(ax, period)
        
        # Create and add the custom legend
        self._add_custom_legend(ax, stat_summary, period)
        
        # Generate the table of results for the specified model
        if self.config.visuals.temperature_graph.show_table:
            table_str = self._format_model_as_table(models_by_state, "all_interactions", "unicode")
            print("\nModel Comparison Table (All Interactions Model):")
            print(table_str)
            
            # Export to Excel if requested
            if self.config.visuals.temperature_graph.export_excel:
                self.export_model_results_to_excel(models_by_state, "all_interactions")
        
        return models_by_state  # Return models for potential further analysis


    def _prepare_centered_variables(self, data):
        """Prepare centered variables for analysis"""
        # Create explicit copy to avoid chained assignment warnings
        data_copy = data.copy()
        
        data_copy['age_centered'] = data_copy['Age'] - data_copy['Age'].mean()
        data_copy['weight_centered'] = data_copy['Weight'] - data_copy['Weight'].mean()
        data_copy['temp_centered'] = data_copy['temperature'] - data_copy['temperature'].mean()
        
        # Create interaction terms
        data_copy['age_temp_interaction'] = data_copy['age_centered'] * data_copy['temp_centered']
        data_copy['weight_temp_interaction'] = data_copy['weight_centered'] * data_copy['temp_centered']
        data_copy['age_weight_interaction'] = data_copy['age_centered'] * data_copy['weight_centered']
        
        return data_copy

    def _print_period_summary(self, data, period):
        """Print summary statistics for the period"""
        print(f"\n{'='*80}")
        print(f"STATISTICAL COMPARISON FOR {period.upper()} PERIOD")
        print(f"{'='*80}")
        print(f"Age range in data: {data['Age'].min()}-{data['Age'].max()} years")
        print(f"Mean age: {data['Age'].mean():.1f} years")
        print(f"Weight range in data: {data['Weight'].min()}-{data['Weight'].max()} kg")
        print(f"Mean weight: {data['Weight'].mean():.1f} kg")
        print(f"Temperature range: {data['temperature'].min():.1f}-{data['temperature'].max():.1f}°C")
        print(f"Mean temperature: {data['temperature'].mean():.1f}°C")
        print(f"Number of unique cows: {data['cow_id'].nunique()}")
        print(f"Total observations: {len(data)}")

    def _run_statistical_models(self, data, state, models_to_run):
        """Run all requested statistical models and return results"""
        import statsmodels.formula.api as smf
        
        results = {}
        hours_col = f"{state}_hours"
        
        # 1. Linear model (temperature only)
        if models_to_run['linear']:
            model = smf.ols(f"{hours_col} ~ temperature", data=data).fit()
            self._print_model_results(model, "STANDARD OLS REGRESSION (temperature only)", 
                                    ['Intercept', 'temperature'])
            results['linear'] = model
        
        # 2. Main effects model (temperature + age + weight)
        if models_to_run['main_effects']:
            model = smf.ols(f"{hours_col} ~ temp_centered + age_centered + weight_centered", 
                            data=data).fit()
            self._print_model_results(model, "OLS REGRESSION (temperature + age + weight)",
                                    ['Intercept', 'temp_centered', 'age_centered', 'weight_centered'])
            results['main_effects'] = model
        
        # 3. Age × weight interaction model
        if models_to_run['age_weight']:
            model = smf.ols(
                f"{hours_col} ~ temp_centered + age_centered + weight_centered + age_weight_interaction", 
                data=data
            ).fit()
            self._print_model_results(model, "OLS REGRESSION (temperature + age + weight + age×weight)",
                                    ['Intercept', 'temp_centered', 'age_centered', 'weight_centered', 
                                    'age_weight_interaction'])
            results['age_weight'] = model
        
        # 4. Temperature interactions model
        if models_to_run['temp_interactions']:
            model = smf.ols(
                f"{hours_col} ~ temp_centered + age_centered + weight_centered + " + 
                "age_temp_interaction + weight_temp_interaction", 
                data=data
            ).fit()
            self._print_model_results(model, "OLS REGRESSION (temperature, age, weight & temp interactions)",
                                    ['Intercept', 'temp_centered', 'age_centered', 'weight_centered',
                                    'age_temp_interaction', 'weight_temp_interaction'])
            results['temp_interactions'] = model
        
        # 5. All interactions model
        if models_to_run['all_interactions']:
            model = smf.ols(
                f"{hours_col} ~ temp_centered + age_centered + weight_centered + " + 
                "age_temp_interaction + weight_temp_interaction + age_weight_interaction", 
                data=data
            ).fit()
            self._print_model_results(model, "OLS REGRESSION (all main effects & all interactions)",
                                    ['Intercept', 'temp_centered', 'age_centered', 'weight_centered',
                                    'age_temp_interaction', 'weight_temp_interaction', 'age_weight_interaction'])
            results['all_interactions'] = model
        
        # 6. Mixed effects model (temperature only)
        if models_to_run['mixed_effects']:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", UserWarning)
                
                try:
                    # Mixed effects model with random intercept for cow_id
                    mixed_model = mlm.MixedLM(
                        endog=data[hours_col],
                        exog=sm.add_constant(data['temperature']),
                        groups=data['cow_id']
                    ).fit(reml=True)
                    
                    print("\n6. MIXED EFFECTS MODEL (temperature only, random intercept for cow_id):")
                    print(f"   Intercept: {mixed_model.fe_params.iloc[0]:.3f} hours")
                    print(f"   Slope (temperature): {mixed_model.fe_params.iloc[1]:.3f} hours/°C")
                    print(f"   P-value (temperature): {mixed_model.pvalues.iloc[1]:.5f}")
                    print(f"   Converged: {mixed_model.converged}")
                    
                    results['mixed_effects'] = mixed_model
                    
                except Exception as e:
                    print(f"\n6. MIXED EFFECTS MODEL: Failed - {str(e)}")
        
        # 7. Mixed effects with interactions
        if models_to_run['mixed_with_interactions']:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", UserWarning)
                
                try:
                    # Create design matrix with main effects
                    exog = sm.add_constant(data[['temp_centered', 'age_centered', 'weight_centered']])
                    
                    # Try mixed effects model with main effects
                    mixed_main_model = mlm.MixedLM(
                        endog=data[hours_col],
                        exog=exog,
                        groups=data['cow_id']
                    ).fit(reml=True)
                    
                    print("\n7. MIXED EFFECTS MODEL (temperature + age + weight):")
                    print(f"   Intercept: {mixed_main_model.fe_params.iloc[0]:.3f} hours")
                    print(f"   Slope (temperature): {mixed_main_model.fe_params.iloc[1]:.3f} hours/°C")
                    print(f"   Slope (age): {mixed_main_model.fe_params.iloc[2]:.3f} hours/year")
                    print(f"   Slope (weight): {mixed_main_model.fe_params.iloc[3]:.5f} hours/kg")
                    print(f"   P-value (temperature): {mixed_main_model.pvalues.iloc[1]:.5f}")
                    print(f"   P-value (age): {mixed_main_model.pvalues.iloc[2]:.5f}")
                    print(f"   P-value (weight): {mixed_main_model.pvalues.iloc[3]:.5f}")
                    print(f"   Converged: {mixed_main_model.converged}")
                    
                    results['mixed_with_interactions'] = mixed_main_model
                    
                except Exception as e:
                    print(f"\n7. MIXED EFFECTS MODEL (with interactions): Failed - {str(e)}")
        
        # 8. Quadratic model (if enabled)
        if models_to_run['quadratic']:
            model = smf.ols(f"{hours_col} ~ temperature + I(temperature**2)", data=data).fit()
            self._print_model_results(model, "POLYNOMIAL REGRESSION (quadratic)",
                                    ['Intercept', 'temperature', 'I(temperature ** 2)'])
            
            # Note the shape of the parabola
            quad_coef = model.params['I(temperature ** 2)']
            shape = "U-shaped" if quad_coef > 0 else "inverted U-shaped"
            print(f"   Quadratic model shows {shape} relationship")
            
            results['quadratic'] = model
        
        # 9. Cubic model (if enabled)
        if models_to_run['cubic']:
            model = smf.ols(f"{hours_col} ~ temperature + I(temperature**2) + I(temperature**3)", 
                            data=data).fit()
            self._print_model_results(model, "POLYNOMIAL REGRESSION (cubic)",
                                    ['Intercept', 'temperature', 'I(temperature ** 2)', 'I(temperature ** 3)'])
            
            results['cubic'] = model
        
        return results



    def _format_model_as_table(self, models_by_state, model_type="all_interactions", format_type="unicode"):
        """
        Format model results as a pretty table
        
        Parameters:
        -----------
        models_by_state : dict
            Dictionary mapping state names to model results
        model_type : str
            Key of the model to format (e.g., 'all_interactions')
        format_type : str
            'unicode' for console output or 'dataframe' for Excel export
        
        Returns:
        --------
        str or DataFrame
            Formatted table as string (unicode) or DataFrame (for Excel)
        """
        # Define parameter names and their labels
        param_labels = {
            'Intercept': 'Intercept',
            'temp_centered': 'Temperature',
            'age_centered': 'Age',
            'weight_centered': 'Weight',
            'age_weight_interaction': 'Age × Weight',
            'age_temp_interaction': 'Age × Temp',
            'weight_temp_interaction': 'Weight × Temp'
        }
        
        # Define units for each parameter
        units = {
            'Intercept': 'hours',
            'temp_centered': 'hours/°C',
            'age_centered': 'hours/year',
            'weight_centered': 'hours/kg',
            'age_weight_interaction': 'hours/year/kg',
            'age_temp_interaction': 'hours/°C/year',
            'weight_temp_interaction': 'hours/°C/kg'
        }
        
        # Parameter order
        param_order = [
            'Intercept', 'temp_centered', 'age_centered', 'weight_centered',
            'age_weight_interaction', 'age_temp_interaction', 'weight_temp_interaction'
        ]
        
        # Create a table structure
        table_data = []
        
        # Add header row
        header = ["Parameter", "Units"]
        for state in models_by_state.keys():
            header.extend([f"{state} Estimate", f"{state} p-value"])
        
        table_data.append(header)
        
        # Add parameter rows
        for param in param_order:
            row = [param_labels[param], units[param]]
            
            for state, models in models_by_state.items():
                model = models.get(model_type)
                
                if model and param in model.params:
                    # Format the estimate with appropriate precision
                    if param in ['weight_centered', 'age_weight_interaction', 'weight_temp_interaction']:
                        estimate = f"{model.params[param]:.5f}"
                    else:
                        estimate = f"{model.params[param]:.3f}"
                    
                    # Format p-value with stars for significance
                    p_value = model.pvalues.get(param, float('nan'))
                    p_value_str = f"{p_value:.5f}"
                    
                    # Add significance stars
                    if param != 'Intercept':  # Don't test significance of intercept
                        if p_value < 0.001:
                            p_value_str += " ***"
                        elif p_value < 0.01:
                            p_value_str += " **"
                        elif p_value < 0.05:
                            p_value_str += " *"
                    
                    row.extend([estimate, p_value_str])
                else:
                    row.extend(["N/A", "N/A"])
            
            table_data.append(row)
        
        # Add model fit statistics row
        fit_row = ["Model Fit", ""]
        for state, models in models_by_state.items():
            model = models.get(model_type)
            if model:
                r_squared = f"R² = {model.rsquared:.3f}"
                aic = f"AIC = {model.aic:.1f}"
                fit_row.extend([r_squared, aic])
            else:
                fit_row.extend(["N/A", "N/A"])
        
        table_data.append(fit_row)
        
        # Format as Unicode table or DataFrame
        if format_type == "unicode":
            from tabulate import tabulate
            return tabulate(table_data, headers="firstrow", tablefmt="fancy_grid")
        else:
            # Create DataFrame for Excel export
            df = pd.DataFrame(table_data[1:], columns=table_data[0])
            return df

    def export_model_results_to_excel(self, models_by_state, model_type="all_interactions", filename=None):
        """
        Export model results to Excel
        
        Parameters:
        -----------
        models_by_state : dict
            Dictionary mapping state names to model results
        model_type : str
            Key of the model to format (e.g., 'all_interactions')
        filename : str, optional
            Path to save Excel file, defaults to temperature_model_results.xlsx in visuals folder
        """
        # Get results as DataFrame
        results_df = self._format_model_as_table(models_by_state, model_type, format_type="dataframe")
        
        # Set default filename if not provided
        if filename is None:
            filename = os.path.join(self.config.visuals.visuals_root_path, 'temperature_model_results.xlsx')
        
        # Export to Excel
        results_df.to_excel(filename, index=False)
        print(f"Model results exported to {filename}")

    def _print_model_results(self, model, title, param_names):
        """Print formatted model results"""
        model_num = len(self._model_counter) + 1
        self._model_counter.append(model_num)
        
        print(f"\n{model_num}. {title}:")
        
        # Print parameters
        param_labels = {
            'Intercept': 'Intercept',
            'temperature': 'Slope (temperature)',
            'temp_centered': 'Slope (temperature)',
            'age_centered': 'Slope (age)',
            'weight_centered': 'Slope (weight)',
            'age_weight_interaction': 'Interaction (age × weight)',
            'age_temp_interaction': 'Interaction (age × temp)',
            'weight_temp_interaction': 'Interaction (weight × temp)',
            'I(temperature ** 2)': 'Quadratic term',
            'I(temperature ** 3)': 'Cubic term'
        }
        
        units = {
            'Intercept': 'hours',
            'temperature': 'hours/°C',
            'temp_centered': 'hours/°C',
            'age_centered': 'hours/year',
            'weight_centered': 'hours/kg',
            'age_weight_interaction': 'hours/year/kg',
            'age_temp_interaction': 'hours/°C/year',
            'weight_temp_interaction': 'hours/°C/kg',
            'I(temperature ** 2)': 'hours/°C²',
            'I(temperature ** 3)': 'hours/°C³'
        }
        
        format_precision = {
            'Intercept': '.3f',
            'temperature': '.3f',
            'temp_centered': '.3f',
            'age_centered': '.3f',
            'weight_centered': '.5f',
            'age_weight_interaction': '.5f',
            'age_temp_interaction': '.3f',
            'weight_temp_interaction': '.5f',
            'I(temperature ** 2)': '.5f',
            'I(temperature ** 3)': '.6f'
        }
        
        for param in param_names:
            if param in model.params:
                format_str = format_precision.get(param, '.3f')
                value_str = f"{model.params[param]:{format_str}}"
                print(f"   {param_labels.get(param, param)}: {value_str} {units.get(param, '')}")
        
        # Print p-values for each parameter (except intercept)
        for param in param_names:
            if param in model.pvalues and param != 'Intercept':
                print(f"   P-value ({param_labels.get(param, param).lower()}): {model.pvalues[param]:.5f}")
        
        # Print model fit statistics
        print(f"   R-squared: {model.rsquared:.3f}")
        print(f"   AIC: {model.aic:.1f}")

    def _select_best_model(self, models_results, state, period, scientific_judgment=True):
        """Select the best model based on statistical criteria and scientific judgment"""
        # Initialize with linear model if available
        if 'linear' not in models_results:
            return None, "No models available", False
        
        best_model = models_results['linear']
        best_model_name = "temperature only"
        best_model_aic = best_model.aic
        plot_curve = False
        
        # Function to check if a model is better using AIC
        def is_better_model(new_model, current_best_aic, min_aic_diff=2):
            # Check if AIC improvement is substantial
            return new_model.aic < (current_best_aic - min_aic_diff)
        
        # Evaluate main effects model
        if 'main_effects' in models_results:
            model = models_results['main_effects']
            if is_better_model(model, best_model_aic):
                best_model = model
                best_model_name = "temperature + age + weight"
                best_model_aic = model.aic
        
        # Evaluate age × weight interaction model
        if 'age_weight' in models_results:
            model = models_results['age_weight']
            if is_better_model(model, best_model_aic):
                best_model = model
                best_model_name = "temperature + age + weight + age×weight"
                best_model_aic = model.aic
        
        # Evaluate temperature interactions model
        if 'temp_interactions' in models_results:
            model = models_results['temp_interactions']
            if is_better_model(model, best_model_aic):
                best_model = model
                best_model_name = "temperature + age + weight + temperature interactions"
                best_model_aic = model.aic
        
        # Evaluate all interactions model
        if 'all_interactions' in models_results:
            model = models_results['all_interactions']
            if is_better_model(model, best_model_aic):
                best_model = model
                best_model_name = "all main effects + all interactions"
                best_model_aic = model.aic
        
        # Evaluate mixed effects model
        if 'mixed_effects' in models_results:
            # Mixed models don't have AIC in same scale, so use scientific judgment
            if scientific_judgment:
                model = models_results['mixed_effects']
                # Check if the temperature effect is significant and convergence was achieved
                if model.pvalues.iloc[1] < 0.05 and model.converged:
                    best_model = model
                    best_model_name = "mixed effects (random cow intercept)"
                    # No AIC for comparison
        
        # Only check polynomial models if scientific judgment is FALSE
        # This prevents biologically implausible curves from being selected
        if not scientific_judgment:
            # Evaluate quadratic model
            if 'quadratic' in models_results:
                model = models_results['quadratic']
                # Only consider if quadratic term is significant
                if model.pvalues['I(temperature ** 2)'] < 0.05 and is_better_model(model, best_model_aic, min_aic_diff=4):
                    best_model = model
                    best_model_name = "quadratic temperature"
                    best_model_aic = model.aic
                    plot_curve = True
                    
            # Evaluate cubic model
            if 'cubic' in models_results:
                model = models_results['cubic']
                # Only consider if cubic term is significant
                if model.pvalues['I(temperature ** 3)'] < 0.05 and is_better_model(model, best_model_aic, min_aic_diff=6):
                    best_model = model
                    best_model_name = "cubic temperature"
                    best_model_aic = model.aic
                    plot_curve = True
        
        print(f"\nSelected model for {state}: {best_model_name}" + 
            (f" (AIC: {best_model_aic:.1f})" if hasattr(best_model, 'aic') else ""))
        
        # If we're using a non-linear model, note this
        if best_model != models_results['linear']:
            print(f"Note: Basic plot shows simple temperature relationship, but {best_model_name} model is statistically better")
        
        return best_model, best_model_name, plot_curve

    def _get_significance(self, p_value):
        """Get significance stars based on p-value"""
        if p_value < 0.001:
            return "***"
        elif p_value < 0.01:
            return "**"
        elif p_value < 0.05:
            return "*"
        else:
            return ""

    def _plot_behavior_line(self, ax, data, state, index):
        """Plot the basic regression line for a behavior"""
        sns.regplot(
            x='temperature',
            y=f"{state}_hours",
            data=data,
            label=state,
            color=self.colors[state]['points'],
            line_kws={
                'color': self.colors[state]['line'], 
                'linewidth': 3,
                'linestyle': ['-', '--', '-.'][index]
            },
            scatter_kws={
                'alpha': 0.0,
                's': 15,
                'edgecolor': 'none'
            },
            ax=ax
        )

    def _add_polynomial_curve(self, ax, poly_model, data, state, index):
        """Add a polynomial curve to the plot"""
        
        # Get temperature range for the curve
        temp_min = data['temperature'].min()
        temp_max = data['temperature'].max()
        
        # Use an extended temperature range for better visualization
        padding = (temp_max - temp_min) * 0.05  # 5% padding
        temp_range = np.linspace(temp_min - padding, temp_max + padding, 100)
        
        # Create the predicted values based on model type
        if 'I(temperature ** 3)' in poly_model.params:
            # Cubic model
            predicted = poly_model.params['Intercept'] + \
                        poly_model.params['temperature'] * temp_range + \
                        poly_model.params['I(temperature ** 2)'] * temp_range**2 + \
                        poly_model.params['I(temperature ** 3)'] * temp_range**3
        else:
            # Quadratic model
            predicted = poly_model.params['Intercept'] + \
                        poly_model.params['temperature'] * temp_range + \
                        poly_model.params['I(temperature ** 2)'] * temp_range**2
        
        # Add constraint to keep predictions between 0 and 3 hours
        predicted = np.clip(predicted, 0, 3)
        
        # Plot the curve
        ax.plot(temp_range, predicted, color=self.colors[state]['line'], 
                linestyle=['-', '--', '-.'][index], linewidth=3,
                alpha=0.7)
                
        print(f"Note: Added polynomial curve to plot (constrained to 0-3 hours)")

    def _format_plot_axes(self, ax, period):
        """Format the plot axes with titles and labels"""
        ax.set_title(f"{period}time Behavior Patterns", pad=20)
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Hours per 3-Hour Window")
        ax.set_ylim(0, 3)  # Max is 3 hours
        ax.grid(True, alpha=0.2)  # Lighter grid
        ax.tick_params(axis='both', which='major', labelsize=14)

    def _add_custom_legend(self, ax, stat_summary, period):
        """Add a custom legend to the plot"""
        period_stats = [s for s in stat_summary if s['period'] == period]
        legend_elements = []
        
        for s in period_stats:
            slope_dir = "↑" if s['slope'] > 0 else "↓"
            # Format: Behavior: intercept + slope × T °C (model info)
            legend_text = f"{s['state']}: {s['intercept']:.2f} + ({s['slope']:.3f} × T) hours {s['significance']}"
            if s['model_description'] != "temperature only":
                legend_text += f" [Better: {s['model_description']}]"
                
            legend_elements.append(plt.Line2D([0], [0], color=self.colors[s['state']]['line'], 
                                            linestyle=['-', '--', '-.'][period_stats.index(s)],
                                            lw=3, label=legend_text))
        
        # Add custom legend
        ax.legend(handles=legend_elements, title="Behavior = Intercept + (Slope × Temp)", 
                loc='upper right', framealpha=0.9)






    def _plot_ols_analysis_oneperiod(self, behavioral_data, ax, period):
        period = period.capitalize()

        # Create a summary of statistics for the figure
        stat_summary = []
        
        # Convert proportion to hours (3-hour windows)
        for state in self.config.analysis.hmm.states:
            behavioral_data[f"{state}_hours"] = behavioral_data[state] * 3
        

        period_data = behavioral_data[behavioral_data['period'] == period]

        # Plot each behavior
        for i, state in enumerate(self.config.analysis.hmm.states):
            # Create scatter plot with regression using hours instead of proportions
            sns.regplot(
                x='temperature',
                y=f"{state}_hours",  # Use hours column
                data=period_data,
                label=state,
                color=self.colors[state]['points'],
                line_kws={
                    'color': self.colors[state]['line'], 
                    'linewidth': 3,
                    'linestyle': ['-', '--', '-.'][i]  # Different line styles for better distinction
                },
                scatter_kws={
                    'alpha': 0.0,
                    's': 15,
                    'edgecolor': 'none'
                },
                ax=ax
            )
            
            # Use simple OLS regression instead of mixed effects model to avoid convergence issues
            import statsmodels.formula.api as smf
            
            # Simple OLS regression
            model = smf.ols(f"{state}_hours ~ temperature", data=period_data).fit()
            
            # Store stats for legend
            significance = "*" if model.pvalues['temperature'] < 0.05 else ""
            if model.pvalues['temperature'] < 0.01:
                significance = "**"
            if model.pvalues['temperature'] < 0.001:
                significance = "***"
            
            # Store both slope and intercept
            stat_summary.append({
                'period': period,
                'state': state,
                'slope': model.params['temperature'],  # This is now in hours/°C
                'intercept': model.params['Intercept'],  # Y-intercept in hours
                'p_value': model.pvalues['temperature'],
                'significance': significance
            })
            
            print(f"\n=== {period} - {state} Analysis ===")
            print(f"Intercept: {model.params['Intercept']:.3f} hours")
            print(f"Slope (temperature): {model.params['temperature']:.3f} hours/°C")
            print(f"P-value: {model.pvalues['temperature']:.3f}")
        
        # Set axis labels and titles
        ax.set_title(f"{period}time Behavior Patterns", pad=20)
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Hours per 3-Hour Window")
        ax.set_ylim(0, 3)  # Max is 3 hours
        ax.grid(True, alpha=0.2)  # Lighter grid
        
        # Create a custom legend with stats info including intercept
        period_stats = [s for s in stat_summary if s['period'] == period]
        # print(period_stats)
        legend_elements = []
        
        for s in period_stats:
            slope_dir = "↑" if s['slope'] > 0 else "↓"
            # Format: Behavior: intercept + slope × T °C
            legend_text = f"{s['state']}: {s['intercept']:.2f} + ({s['slope']:.3f} × T) hours {s['significance']}"
            legend_elements.append(plt.Line2D([0], [0], color=self.colors[s['state']]['line'], 
                                            linestyle=['-', '--', '-.'][period_stats.index(s)],
                                            lw=3, label=legend_text))
        
        # Add custom legend
        ax.legend(handles=legend_elements, title="Behavior = Intercept + (Slope × Temp)", 
                loc='upper right', framealpha=0.9)
        
        # Increase tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=14)



    def _plot_temperature_category_analysis(self, behavioral_data):
        """Plot boxplot analysis by temperature category (following Adams et al.)"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Convert proportions to hours for each behavior
        for state in self.config.analysis.hmm.states:
            behavioral_data[f"{state}_hours"] = behavioral_data[state] * 3
        
        # Reshape data for categorical analysis (using hours)
        plot_data = pd.melt(
            behavioral_data,
            id_vars=['temp_category', 'period'],
            value_vars=[f"{state}_hours" for state in self.config.analysis.hmm.states],
            var_name='behavior',
            value_name='hours'
        )
        
        # Clean up behavior names by removing "_hours" suffix
        plot_data['behavior'] = plot_data['behavior'].str.replace('_hours', '')
        
        # Create box plot
        sns.boxplot(
            x='temp_category',
            y='hours',
            hue='behavior',
            data=plot_data,
            palette={'Grazing': '#2ecc71', 'Resting': '#3498db', 'Traveling': '#e74c3c'},
            ax=ax
        )
        
        # Add statistical annotations
        from scipy import stats
        
        # Print stats for each behavior across temperature categories
        for behavior in self.config.analysis.hmm.states:
            behavior_data = plot_data[plot_data['behavior'] == behavior]
            f_stat, p_val = stats.f_oneway(
                *[group['hours'].values for name, group in behavior_data.groupby('temp_category')]
            )
            print(f"\n{behavior} ANOVA across temperature categories:")
            print(f"F-statistic: {f_stat:.4f}")
            print(f"p-value: {p_val:.4f}")
        
        ax.set_title("Behavior Hours by Temperature Category", fontsize=18)
        ax.set_xlabel("Temperature Category", fontsize=16)
        ax.set_ylabel("Hours per 3-Hour Window", fontsize=16)
        ax.set_ylim(0, 3)  # Max is 3 hours
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend(title="Behaviors", fontsize=14)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.config.visuals.visuals_root_path, 'temp_behavior_categories.png'),
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
