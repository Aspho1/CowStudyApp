import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ephem
import pytz
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM
import statsmodels.api as sm

from cowstudyapp.utils import from_posix_col
from cowstudyapp.config import ConfigManager



# Add helper function
def format_p_value(p):
    if p < 0.001:
        return "0.001"
    elif p < 0.01:
        return "0.01"
    elif p < 0.05:
        return "0.05"
    else:
        return f"{p:.3f}"
    

class MoonPhasesGrazing:
    def __init__(self, config: ConfigManager):
        self.config = config
        # Set up Norris, MT observer
        self.observer = ephem.Observer()
        self.observer.lat = '45.575'
        self.observer.lon = '-111.625'
        self.observer.elevation = 1715
        self.observer.pressure = 0  # Disable atmospheric refraction correction
        self.observer.horizon = '-0:34'  # Standard sunrise/sunset definition
        
        self.buffer = 1.5  # Hours buffer around sunrise/sunset
        self.full_moon_threshold = 95  # Moon illumination percentage to classify as full moon
        self.moon = ephem.Moon()

    def get_moon_phase(self, date):
        """Calculate moon phase for given date"""
        self.moon.compute(date)
        return self.moon.phase  # Returns percentage illuminated (0-100)

    def _add_moon_data(self, df):
        """Add moon phase information to the dataframe"""
        # Convert timestamps to local time
        df['mt'] = from_posix_col(df['posix_time'], self.config.analysis.timezone)
        df['date'] = df['mt'].dt.date
        df['hour'] = df['mt'].dt.hour
        
        # Initialize columns
        df['is_night'] = False
        df['moon_phase'] = np.nan
        df['is_full_moon'] = False
        
        # First, calculate all moon phases
        unique_dates = df['date'].unique()
        moon_phases = []
        for date in unique_dates:
            moon_phase = self.get_moon_phase(date)
            moon_phases.append({
                'date': date,
                'phase': moon_phase,
                'is_full': moon_phase > self.full_moon_threshold
            })
        
        moon_phase_df = pd.DataFrame(moon_phases)
        print("\nMoon phase distribution:")
        print(moon_phase_df['phase'].describe())
        print("\nPhase counts by threshold:")
        for threshold in [80, 85, 90, 95]:
            count = (moon_phase_df['phase'] > threshold).sum()
            pct = count / len(moon_phase_df) * 100
            print(f"Above {threshold}%: {count} days ({pct:.1f}%)")
        
        # Create a mapping dictionary for moon phases
        moon_phase_dict = moon_phase_df.set_index('date')[['phase', 'is_full']].to_dict('index')
        
        # Calculate sunrise/sunset for each date and classify day/night
        sun_times = {}
        for date in unique_dates:
            # Get sunrise time
            self.observer.date = date.strftime('%Y/%m/%d 00:00:00')
            sunrise = pd.to_datetime(str(self.observer.next_rising(ephem.Sun()))).tz_localize('UTC').tz_convert(self.config.analysis.timezone)
            
            # Get sunset time
            self.observer.date = date.strftime('%Y/%m/%d 12:00:00')
            sunset = pd.to_datetime(str(self.observer.next_setting(ephem.Sun()))).tz_localize('UTC').tz_convert(self.config.analysis.timezone)
            
            # Store times with buffers
            sunrise_buffer = sunrise - pd.Timedelta(hours=self.buffer)
            sunset_buffer = sunset + pd.Timedelta(hours=self.buffer)
            sun_times[date] = {
                'sunrise': sunrise,
                'sunrise_buffer': sunrise_buffer,
                'sunset': sunset,
                'sunset_buffer': sunset_buffer
            }
            
            # Update records for this date
            mask = (df['date'] == date)
            
            # Update night/day classification
            df.loc[mask, 'is_night'] = df.loc[mask, 'mt'].apply(
                lambda t: t <= sunrise_buffer or t >= sunset_buffer
            )
            
            # Update moon phase information
            df.loc[mask, 'moon_phase'] = moon_phase_dict[date]['phase']
            df.loc[mask, 'is_full_moon'] = moon_phase_dict[date]['is_full']
        
        return df, sun_times
        
    def compare_moon_behavior(self, df):
        """Analyze and visualize the relationship between moon phase and grazing behavior"""
        # Filter out weigh days if defined in config
        if hasattr(self.config.visuals, 'weigh_days'):
            weigh_days = [pd.to_datetime(d).date() for d in self.config.visuals.weigh_days]
            df = df[~df['date'].isin(weigh_days)]
        
        # Add moon phase and day/night classification
        df, sun_times = self._add_moon_data(df)
        
        # Print some diagnostics to help debug
        print(f"Total records after moon data added: {len(df)}")
        print(f"Moon phase stats: min={df['moon_phase'].min()}, max={df['moon_phase'].max()}")
        print(f"Full moon records: {df['is_full_moon'].sum()} ({df['is_full_moon'].mean()*100:.1f}%)")
        print(f"Night records: {df['is_night'].sum()} ({df['is_night'].mean()*100:.1f}%)")
        
        # Group by cow, date, and night/day
        grazing_stats = df.groupby(
            ['ID', 'date', 'is_night', 'is_full_moon']
        ).apply(
            lambda x: (x['predicted_state'] == 'Grazing').mean()
        ).reset_index(name='grazing_percentage')
        
        print(f"\nGrazing stats records: {len(grazing_stats)}")
        
        # Convert boolean is_night to string for easier plotting
        grazing_stats['is_night'] = grazing_stats['is_night'].map({True: 'Night', False: 'Day'})
        
        # Print summary statistics
        # self._print_moon_statistics(grazing_stats)
        
        # Create visualizations
        self._plot_moon_day_night_comparison(grazing_stats)
        self._plot_individual_cow_responses(grazing_stats)
        
        return grazing_stats

    def _print_moon_statistics(self, df):
        """Print statistical analysis of moon effects on grazing behavior"""
        print("\nMoon Phase Analysis Statistics:")
        
        # Overall statistics
        print(f"Total data points: {len(df)}")
        print(f"Full moon periods: {df['is_full_moon'].sum()} ({df['is_full_moon'].mean()*100:.1f}%)")
        
        # Day/Night breakdown
        print("\nData points by period:")
        print(df.groupby('is_night')['is_full_moon'].value_counts())
        
        # Statistical tests for day
        day_data = df[df['is_night'] == 'Day']
        night_data = df[df['is_night'] == 'Night']
        
        print("\nDay Analysis - Full Moon vs. Regular Moon:")
        day_full = day_data[day_data['is_full_moon']]['grazing_percentage']
        day_regular = day_data[~day_data['is_full_moon']]['grazing_percentage']
        
        if len(day_full) > 0 and len(day_regular) > 0:
            stat, p = stats.ttest_ind(day_full, day_regular)
            diff = day_full.mean() - day_regular.mean()
            print(f"  Mean grazing % (full moon): {day_full.mean():.4f}")
            print(f"  Mean grazing % (regular): {day_regular.mean():.4f}")
            print(f"  Difference: {diff:.4f} ({diff/day_regular.mean()*100:.1f}% change)")
            print(f"  t-statistic: {stat:.4f}")
            print(f"  p-value: {p:.4f}")
            print(f"  Statistically significant: {'Yes' if p < 0.05 else 'No'}")
        
        print("\nNight Analysis - Full Moon vs. Regular Moon:")
        night_full = night_data[night_data['is_full_moon']]['grazing_percentage']
        night_regular = night_data[~night_data['is_full_moon']]['grazing_percentage']
        
        if len(night_full) > 0 and len(night_regular) > 0:
            stat, p = stats.ttest_ind(night_full, night_regular)
            diff = night_full.mean() - night_regular.mean()
            print(f"  Mean grazing % (full moon): {night_full.mean():.4f}")
            print(f"  Mean grazing % (regular): {night_regular.mean():.4f}")
            print(f"  Difference: {diff:.4f} ({diff/night_regular.mean()*100:.1f}% change)")
            print(f"  t-statistic: {stat:.4f}")
            print(f"  p-value: {p:.4f}")
            print(f"  Statistically significant: {'Yes' if p < 0.05 else 'No'}")


            
    
    def _plot_moon_day_night_comparison(self, df):
        """Create boxplots comparing grazing during full moon vs regular moon periods"""
        # Define colors
        colors = {
            'Full Moon': '#f39c12',  # Golden color for full moon
            'Regular Moon': '#34495e'  # Dark blue for regular moon
        }
        
        # Increase font sizes
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 18,
            'axes.labelsize': 16,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14
        })
        
        # Create figure with two subplots (day and night)
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Day data
        day_data = df[df['is_night'] == 'Day']
        sns.boxplot(
            x='is_full_moon',
            y='grazing_percentage',
            data=day_data,
            ax=axes[0],
            palette=[colors['Regular Moon'], colors['Full Moon']],
            fliersize=0
        )
        
        # Add statistical annotation for day
        day_full = day_data[day_data['is_full_moon']]['grazing_percentage']
        day_reg = day_data[~day_data['is_full_moon']]['grazing_percentage']
        if len(day_full) > 0 and len(day_reg) > 0:
            _, p_day = stats.ttest_ind(day_full, day_reg)
            sig_stars = ''
            if p_day < 0.05: sig_stars = '*'
            if p_day < 0.01: sig_stars = '**'
            if p_day < 0.001: sig_stars = '***'
            
            day_diff = day_full.mean() - day_reg.mean()
            day_pct = day_diff / day_reg.mean() * 100
            
            axes[0].annotate(
                f"{sig_stars}\n{day_diff:.3f} ({day_pct:+.1f}%)\np<{format_p_value(p_day)}", 
                xy=(0.5, 0.95),
                xycoords='axes fraction',
                ha='center',
                va='top',
                fontsize=14,
                bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.8)
            )


        # Night data
        night_data = df[df['is_night'] == 'Night']
        sns.boxplot(
            x='is_full_moon',
            y='grazing_percentage',
            data=night_data,
            ax=axes[1],
            palette=[colors['Regular Moon'], colors['Full Moon']],
            fliersize=0
        )
        
        # Add statistical annotation for night
        night_full = night_data[night_data['is_full_moon']]['grazing_percentage']
        night_reg = night_data[~night_data['is_full_moon']]['grazing_percentage']
        if len(night_full) > 0 and len(night_reg) > 0:
            _, p_night = stats.ttest_ind(night_full, night_reg)
            sig_stars = ''
            if p_night < 0.05: sig_stars = '*'
            if p_night < 0.01: sig_stars = '**'
            if p_night < 0.001: sig_stars = '***'
            
            night_diff = night_full.mean() - night_reg.mean()
            night_pct = night_diff / night_reg.mean() * 100
            
            # axes[1].annotate(
            #     f"{sig_stars}\n{night_diff:.3f} ({night_pct:+.1f}%)", 
            #     xy=(0.5, 0.95),
            #     xycoords='axes fraction',
            #     ha='center',
            #     va='top', 
            #     fontsize=14,
            #     bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.8)
            # )
        
            axes[1].annotate(
                f"{sig_stars}\n{night_diff:.3f} ({night_pct:+.1f}%)\np<{format_p_value(p_night)}", 
                xy=(0.5, 0.95),
                xycoords='axes fraction',
                ha='center',
                va='top',
                fontsize=14,
                bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.8)
            )

        # Customize plots
        for i, (ax, title) in enumerate(zip(axes, ['Daytime Grazing Patterns', 'Nighttime Grazing Patterns'])):
            ax.set_title(title, pad=20)
            ax.set_xlabel("Moon Phase")
            ax.set_ylabel("Proportion of Time Spent Grazing")
            ax.set_ylim(0, 1)
            ax.set_xticklabels(['Regular Moon', 'Full Moon'])
            ax.grid(True, alpha=0.2)
        
        fig.suptitle("Effect of Moon Phase on Grazing Behavior\n* p<0.05, ** p<0.01, *** p<0.001", 
                     fontsize=20, y=0.98)
        
        # # Add text explaining findings
        # plt.figtext(0.5, 0.01, 
        #            "Analysis shows how full moon periods affect grazing behavior during day and night. " +
        #            "Values indicate absolute difference and percent change in grazing time.", 
        #            ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(
            os.path.join(self.config.visuals.visuals_root_path, 'moon_grazing_effect.png'),
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
        


    def _plot_individual_cow_responses(self, df):
        # Focus on nighttime data where moon effects are strongest
        night_data = df[df['is_night'] == 'Night']
        
        # Create figure
        plt.figure(figsize=(15, 8))
        
        # Define colors explicitly
        colors = ['#34495e', '#f39c12']  # Dark blue for regular, gold for full moon
        
        # Create box plot for each cow
        ax = sns.boxplot(
            x='ID',
            y='grazing_percentage',
            hue='is_full_moon',
            data=night_data,
            palette=colors,  # Use our explicit colors
            fliersize=0
        )
        
        # Create custom legend handles with explicit colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#34495e', edgecolor='black', label='Regular Moon'),
            Patch(facecolor='#f39c12', edgecolor='black', label='Full Moon')
        ]
        
        # Create custom legend with our explicit patches
        plt.legend(
            handles=legend_elements,
            title='Moon Phase',
            loc='upper right',
            bbox_to_anchor=(1.0, 0.98),
            framealpha=0.9
        )
        
        # Customize plot
        plt.title('Individual Cow Nighttime Grazing Responses to Moon Phase', fontsize=18)
        plt.xlabel('Cow ID', fontsize=16)
        plt.ylabel('Proportion of Time Spent Grazing', fontsize=16)
        plt.ylim(0, 0.6)
        # plt.grid(axis='y', alpha=0.4)
        plt.grid(axis='x', alpha=0.4)
        
        # Perform statistical tests for each cow and annotate significant differences
        for i, cow in enumerate(sorted(night_data['ID'].unique())):
            cow_data = night_data[night_data['ID'] == cow]
            full = cow_data[cow_data['is_full_moon']]['grazing_percentage']
            reg = cow_data[~cow_data['is_full_moon']]['grazing_percentage']
            
            if len(full) > 5 and len(reg) > 5:  # Only test if we have enough data
                _, p = stats.ttest_ind(full, reg)
                if p < 0.05:
                    sig = '*'
                    if p < 0.01: 
                        sig = '**'
                        if p < 0.001: 
                            sig = '***'
                    # Position stars slightly lower to avoid legend overlap
                    ax.annotate(sig, xy=(i, 0.50), ha='center', fontsize=14)
                    # ax.annotate(sig, xy=(i, full.quantile(.80) + 0.05), ha='center', fontsize=14)
                    # ax.annotate(sig, xy=(i, 0.95), ha='center', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.config.visuals.visuals_root_path, 'moon_grazing_individual_cows.png'),
            bbox_inches='tight',
            dpi=300
        )
        plt.close()