import os
from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import statsmodels.api as sm
from matplotlib.patches import Patch
from cowstudyapp.utils import from_posix_col
from cowstudyapp.config import ConfigManager
from statsmodels.stats.multicomp import pairwise_tukeyhsd

class GrazingVersusTemperature:
    def __init__(self, config: ConfigManager):
        self.config = config

    def _prepare_data(self, df):
        df['mt'] = from_posix_col(df['posix_time'], self.config.analysis.timezone)

        df["date"] = df["mt"].dt.date
        
        if ('temperature' not in df.columns):
            if ('temperature_gps' in df.columns):
                df.rename(columns={'temperature_gps', 'temperature'},inplace=True)
            else:
                df['temperature'] = pd.read_csv(self.config.analysis.target_dataset)['temperature_gps']
            

        # Filter out weigh days if needed
        # df = df[~df['date'].isin(self.config.visuals.weigh_days)]
        return df

    def _bin_temperature(self, temp):
        """Create temperature bins"""
        if temp < -10:
            return 0
        elif temp < -5:
            return 1
        elif temp < 0:
            return 2
        elif temp < 5:
            return 3
        elif temp < 10:
            return 4
        else:
            return 5

    # def compare_temp_behavior(self, df, min_records=180):
    #     df = self._prepare_data(df)

    #     # Calculate daily ratios per behavior
    #     daily_behavior = df.groupby(["ID", "date"]).agg({
    #         'predicted_state': lambda x: {
    #             state: (x == state).mean() 
    #             for state in self.config.analysis.hmm.states
    #         },
    #         'temperature': 'min'  # or whatever temperature column you have
    #     }).reset_index()

    #     # Flatten the behavior dictionary
    #     for state in self.config.analysis.hmm.states:
    #         daily_behavior[f'ratio_{state.lower()}'] = daily_behavior['predicted_state'].apply(
    #             lambda x: x[state]
    #         )
    #     daily_behavior.drop('predicted_state', axis=1, inplace=True)

    #     # Create temperature bins
    #     daily_behavior["binned_temp"] = daily_behavior.temperature.apply(self._bin_temperature)


    #     # Prepare data for plotting
    #     plot_df = pd.melt(
    #         daily_behavior,
    #         id_vars=['binned_temp'],
    #         value_vars=[f'ratio_{state.lower()}' for state in self.config.analysis.hmm.states],
    #         var_name='behavior',
    #         value_name='ratio'
    #     )

    #     # Clean behavior names
    #     plot_df['behavior'] = plot_df['behavior'].apply(
    #         lambda x: x.replace('ratio_', '').capitalize()
    #     )

    #     print(plot_df['binned_temp'].value_counts())

    #     # Count the number of samples in each temperature bin
    #     bin_counts = plot_df.groupby('binned_temp').size()
    #     # Get only the bins that have data
    #     active_bins = bin_counts[bin_counts > self.config.visuals.temperature_graph.minimum_required_values].index

    #     # Define full color palette
    #     all_colors = ['#FF9999', '#FF6666', '#FF3333', '#CC0000', '#990000', '#660000']
    #     # Get only the colors for bins that have data
    #     colors = [all_colors[i] for i in active_bins]

    #     # Filter plot_df to only include bins with data
    #     plot_df = plot_df[plot_df['binned_temp'].isin(active_bins)]



    #     p_values = {}
    #     for behavior in self.config.analysis.hmm.states:
    #         # Group data by temperature bins
    #         behavior_by_temp = [group['ratio'].values 
    #                         for name, group in plot_df[plot_df['behavior'] == behavior.capitalize()].groupby('binned_temp')]
            
    #         # Perform one-way ANOVA
    #         f_stat, p_value = stats.f_oneway(*behavior_by_temp)
    #         p_values[behavior] = {'f_statistic': f_stat, 'p_value': p_value}



    #     # Print ANOVA results
    #     print("\nANOVA Results:")
    #     for behavior, results in p_values.items():
    #         print(f"\n{behavior}:")
    #         print(f"F-statistic: {results['f_statistic']:.3f}")
    #         print(f"p-value: {results['p_value']:.3f}")


    #     for behavior in self.config.analysis.hmm.states:
    #         behavior_data = plot_df[plot_df['behavior'] == behavior.capitalize()]
            
    #         tukey = pairwise_tukeyhsd(behavior_data['ratio'], 
    #                                 behavior_data['binned_temp'],
    #                                 alpha=0.05)
            
    #         print(f"\n{behavior} - Tukey's HSD test results:")
    #         print(tukey)


    #     # Create plot
    #     fig, ax = plt.subplots(layout="constrained", figsize=(12, 6))

    #     # Create boxplot with filtered data
    #     sns.boxplot(
    #         data=plot_df,
    #         x="behavior",
    #         y="ratio",
    #         hue="binned_temp",
    #         ax=ax,
    #         fliersize=0,
    #         palette=colors
    #     )

    #     # Format axes
    #     ax.set_yticklabels([f"{100*t:.0f}%" for t in ax.get_yticks()])
    #     # ax.set_xticklabels([
    #     #     f"{behavior}\np={p_values[behavior]:.3f}"
    #     #     for behavior in self.config.analysis.hmm.states
    #     # ])

    #     # Update the plot labels with ANOVA p-values
    #     ax.set_xticklabels([
    #         f"{behavior}\np={p_values[behavior]['p_value']:.3f}"
    #         for behavior in self.config.analysis.hmm.states
    #     ])

    #     # Customize labels
    #     ax.set_xlabel("Cow Behavior")
    #     ax.set_ylabel("Percent of day")

    #     # Create custom legend only for temperature bins that have data
    #     all_legend_labels = ["<-10", "<-5", "<0", "<5", "<10", ">=10"]
    #     legend_labels = [all_legend_labels[i] for i in active_bins]
    #     legend_patches = [
    #         Patch(facecolor=color, edgecolor='black', label=label)
    #         for color, label in zip(colors, legend_labels)
    #     ]

    #     # Add legend
    #     legend = ax.legend(
    #         handles=legend_patches,
    #         title="Min daily Temp (C)",
    #         loc='upper left'
    #     )
    #     legend.get_frame().set_facecolor('white')
    #     legend.get_frame().set_edgecolor('black')
    #     legend.get_frame().set_alpha(0.9)
    #     # Set title and save
    #     fig.suptitle("Relation between minimum temperature and daily behavior patterns")
    #     plt.savefig(os.path.join(self.config.visuals.visuals_root_path, 'temp_behavior_analysis.png'))
    #     plt.close()

    #     # Print statistics
    #     self._print_statistics(daily_behavior)

    # def _print_statistics(self, df):
    #     """Print detailed statistics about temperature effects on behavior"""
    #     X = sm.add_constant(df.temperature)
        
    #     for behavior in self.config.analysis.hmm.states:
    #         print(f"\n=== {behavior} Analysis ===")
    #         model = sm.OLS(df[f'ratio_{behavior.lower()}'], X).fit()
    #         print(model.summary())



    def compare_temp_behavior(self, df, min_records=180):
        df = self._prepare_data(df)

        # Calculate daily ratios per behavior
        daily_behavior = df.groupby(["ID", "date"]).agg({
            'predicted_state': lambda x: {
                state: (x == state).mean() 
                for state in self.config.analysis.hmm.states
            },
            'temperature': ['mean', 'min', 'max']  # Capture more temperature metrics
        }).reset_index()

        print(daily_behavior.head())

        # Flatten the behavior dictionary
        for state in self.config.analysis.hmm.states:
            daily_behavior[f'ratio_{state.lower()}'] = daily_behavior['predicted_state'].apply(
                lambda x: x[state]
            )
        daily_behavior.drop('predicted_state', axis=1, inplace=True)

        # Create figure
        fig, axes = plt.subplots(1, len(self.config.analysis.hmm.states), 
                                figsize=(15, 5), layout="constrained")
        
        # For each behavior state
        for idx, state in enumerate(self.config.analysis.hmm.states):
            ax = axes[idx]
            
            # Create scatter plot
            sns.regplot(data=daily_behavior,
                    x='temperature',  # or temperature[('temperature', 'mean')] if using multiindex
                    y=f'ratio_{state.lower()}',
                    ax=ax,
                    scatter_kws={'alpha':0.5},
                    line_kws={'color': 'red'})
            
            # Run statistical analysis
            X = sm.add_constant(daily_behavior['temperature'])
            y = daily_behavior[f'ratio_{state.lower()}']
            model = sm.OLS(y, X).fit()
            
            # Add regression details to plot
            r_squared = model.rsquared
            p_value = model.f_pvalue
            slope = model.params[1]
            
            ax.set_title(f"{state}\nR²={r_squared:.3f}, p={p_value:.3f}\nslope={slope:.3e}")
            ax.set_xlabel("Temperature (°C)")
            ax.set_ylabel("Proportion of day")
            ax.set_ylim(0, 1)
            
            # Print detailed statistics
            print(f"\n=== {state} Analysis ===")
            print(model.summary())

        fig.suptitle("Relationship between temperature and daily behavior patterns")
        plt.savefig(os.path.join(self.config.visuals.visuals_root_path, 
                                'temp_behavior_analysis.png'))
        plt.close()

        # Additional analysis of daily patterns
        self._analyze_daily_patterns(daily_behavior)

    def _analyze_daily_patterns(self, daily_behavior):
        """Analyze how behaviors vary throughout the day"""
        print("\nCorrelation Analysis:")
        
        # Calculate correlation matrix between temperature and behaviors
        behavior_cols = [f'ratio_{state.lower()}' for state in self.config.analysis.hmm.states]
        corr_matrix = daily_behavior[['temperature'] + behavior_cols].corr()
        
        print("\nTemperature correlations with behaviors:")
        print(corr_matrix['temperature'][behavior_cols])