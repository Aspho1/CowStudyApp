import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import ephem
from datetime import datetime

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
            row_data = {
                'cow_id': cow_id,
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
        
        # Create plots
        self._plot_mixed_effects_analysis(behavioral_data)
        # self._plot_temperature_category_analysis(behavioral_data)
        
        return behavioral_data



    def _plot_mixed_effects_analysis(self, behavioral_data):
        """Plot mixed effects model analysis of temperature vs behavior"""
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Define colors with better contrast between points and lines
        colors = {
            'Grazing': {'points': '#2ecc71', 'line': '#1a8c4a'},  # Green (lighter for points, darker for line)
            'Resting': {'points': '#3498db', 'line': '#1a5c8c'},  # Blue (lighter for points, darker for line)
            'Traveling': {'points': '#e74c3c', 'line': '#992d22'}  # Red (lighter for points, darker for line)
        }
        
        periods = ['Day', 'Night']
        
        # Increase font sizes
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 18,
            'axes.labelsize': 16,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14
        })
        
        # Create a summary of statistics for the figure
        stat_summary = []
        
        # Plot for each day/night period
        for idx, period in enumerate(periods):
            period_data = behavioral_data[behavioral_data['period'] == period]
            ax = axes[idx]
            
            # Plot each behavior
            for i, state in enumerate(self.config.analysis.hmm.states):
                # Create scatter plot with regression - use different colors for points and line
                sns.regplot(
                    x='temperature',
                    y=state,
                    data=period_data,
                    label=state,
                    color=colors[state]['points'],
                    line_kws={
                        'color': colors[state]['line'], 
                        'linewidth': 3,
                        'linestyle': ['-', '--', '-.'][i]  # Different line styles for better distinction
                    },
                    scatter_kws={
                        'alpha': 0.2,  # More transparency
                        's': 15,      # Smaller points
                        'edgecolor': 'none'  # No point outlines
                    },
                    ax=ax
                )
                
                # Run statistical analysis with mixed effects model
                from statsmodels.regression.mixed_linear_model import MixedLM
                
                model = MixedLM(period_data[state], 
                            sm.add_constant(period_data['temperature']),
                            groups=period_data['cow_id']).fit()
                
                # Store stats for legend
                significance = "*" if model.pvalues[1] < 0.05 else ""
                if model.pvalues[1] < 0.01:
                    significance = "**"
                if model.pvalues[1] < 0.001:
                    significance = "***"
                
                stat_summary.append({
                    'period': period,
                    'state': state,
                    'slope': model.fe_params[1],
                    'p_value': model.pvalues[1],
                    'significance': significance
                })
                
                print(f"\n=== {period} - {state} Analysis ===")
                print(f"Fixed effect (temperature): {model.fe_params[1]:.3e}")
                print(f"P-value: {model.pvalues[1]:.3f}")
            
            # Set axis labels and titles
            ax.set_title(f"{period}time Behavior Patterns", pad=20)
            ax.set_xlabel("Temperature (°C)")
            ax.set_ylabel("Proportion of 3-Hour Window")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.2)  # Lighter grid
            
            # Create a custom legend with stats info
            period_stats = [s for s in stat_summary if s['period'] == period]
            legend_elements = []
            
            for s in period_stats:
                slope_dir = "↑" if s['slope'] > 0 else "↓"
                legend_text = f"{s['state']}: {slope_dir} {abs(s['slope']):.2e} {s['significance']}"
                legend_elements.append(plt.Line2D([0], [0], color=colors[s['state']]['line'], 
                                                linestyle=['-', '--', '-.'][period_stats.index(s)],
                                                lw=3, label=legend_text))
            
            # Add custom legend
            ax.legend(handles=legend_elements, title="Behaviors (slope & significance)", 
                    loc='upper right', framealpha=0.9)
            
            # # Add explanatory note about significance
            # ax.text(0.98, 0.02, , 
            #     transform=ax.transAxes, ha='right', fontsize=10)
            
            # Increase tick label sizes
            ax.tick_params(axis='both', which='major', labelsize=14)
        
        fig.suptitle("Temperature Effects on Cow Behavior (3-Hour Windows)\n* p<0.05, ** p<0.01, *** p<0.001", fontsize=20, y=0.98)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.config.visuals.visuals_root_path, 'temp_behavior_daynight.png'),
            bbox_inches='tight',
            dpi=300
        )
        plt.close()


    def _plot_temperature_category_analysis(self, behavioral_data):
        """Plot boxplot analysis by temperature category (following Adams et al.)"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Reshape data for categorical analysis
        plot_data = pd.melt(
            behavioral_data,
            id_vars=['temp_category', 'period'],
            value_vars=self.config.analysis.hmm.states,
            var_name='behavior',
            value_name='proportion'
        )
        
        # Create box plot
        sns.boxplot(
            x='temp_category',
            y='proportion',
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
                *[group['proportion'].values for name, group in behavior_data.groupby('temp_category')]
            )
            print(f"\n{behavior} ANOVA across temperature categories:")
            print(f"F-statistic: {f_stat:.4f}")
            print(f"p-value: {p_val:.4f}")
        
        ax.set_title("Behavior Proportions by Temperature Category", fontsize=18)
        ax.set_xlabel("Temperature Category", fontsize=16)
        ax.set_ylabel("Proportion of 3-Hour Window", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend(title="Behaviors", fontsize=14)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.config.visuals.visuals_root_path, 'temp_behavior_categories.png'),
            bbox_inches='tight',
            dpi=300
        )
        plt.close()

