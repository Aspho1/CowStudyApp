import os
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# from matplotlib.dates import WeekdayLocator
# import matplotlib.patches as mpatches
import seaborn as sns

# from pathlib import Path
# import logging
# from datetime import datetime

from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
# import statsmodels.api as sm

from cowstudyapp.utils import from_posix_col
from cowstudyapp.config import ConfigManager


def plot_tukey_results(tukey_results, title, y_values):
    """
    Create a bar plot with horizontal lines connecting statistically similar groups,
    removing redundant lines that are completely encompassed by longer lines.
    """
    # Extract results table into a DataFrame
    results_df = pd.DataFrame(
        [tuple(str(x) for x in row) for row in tukey_results._results_table[1:]],
        columns=['group1', 'group2', 'meandiff', 'p_adj', 'lower', 'upper', 'reject']
    )
    results_df['p_adj'] = results_df['p_adj'].astype(float)
    
    # Get mean values for each group
    means = y_values.groupby(level=0).mean()
    sorted_groups = means.sort_values().index
    sorted_means = means[sorted_groups]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot bars
    x = np.arange(len(sorted_groups))
    bars = ax.bar(x, sorted_means.values)
    
    # Find all non-significant pairs
    significant_ranges = []
    for group1 in sorted_groups:
        for group2 in sorted_groups:
            if group1 >= group2:
                continue
                
            mask = ((results_df['group1'] == str(group1)) & (results_df['group2'] == str(group2)) |
                   (results_df['group1'] == str(group2)) & (results_df['group2'] == str(group1)))
            
            if not mask.any():
                continue
                
            p_adj = results_df[mask]['p_adj'].iloc[0]
            
            if p_adj > 0.05:  # Not significantly different
                pos1 = np.where(sorted_groups == group1)[0][0]
                pos2 = np.where(sorted_groups == group2)[0][0]
                significant_ranges.append((min(pos1, pos2), max(pos1, pos2)))

    # Remove redundant ranges
    final_ranges = []
    significant_ranges.sort(key=lambda x: (x[1] - x[0]), reverse=True)  # Sort by range length
    
    for current_range in significant_ranges:
        is_redundant = False
        for existing_range in final_ranges:
            if (current_range[0] >= existing_range[0] and 
                current_range[1] <= existing_range[1]):
                is_redundant = True
                break
        if not is_redundant:
            final_ranges.append(current_range)

    # Plot the non-redundant ranges
    max_val = sorted_means.max()
    y_start = max_val * 1.05
    y_spacing = max_val * 0.02
    
    final_ranges.sort(key=lambda x: (x[0]))  # Sort by range length
    for i, (start, end) in enumerate(final_ranges):
        ax.hlines(y=y_start + i*y_spacing, xmin=start, xmax=end, color='gray', linewidth=1)
    
    # Customize plot
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_groups, rotation=45, ha='right')
    ax.set_ylabel('Grazing Percentage')
    ax.set_title(title)
    
    # Add value labels on bars
    for i, v in enumerate(sorted_means):
        ax.text(i, v, f'{v:.1f}', ha='center', va='bottom')
    
    # Adjust y-axis to show all lines
    ax.set_ylim(0, y_start + len(final_ranges)*y_spacing)
    
    plt.tight_layout()
    plt.show()


def analyze_cows(df):
    print("=== Analysis by Cow ===")
    print("H1: At least one cow's mean grazing time is different")
    
    cow_groups = [group for _, group in df.groupby('ID')['grazing_percentage']]
    
    # print(cow_groups)
    f_stat, p_value = stats.f_oneway(*cow_groups)
    
    print(f"\nOne-way ANOVA results:")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    tukey = pairwise_tukeyhsd(df['grazing_percentage'], df['ID'])
    print("\nTukey's HSD test results:")
    print(tukey)

    
    plot_tukey_results(tukey, "Comparisons Between Cows", df.set_index('ID')['grazing_percentage'])


def analyze_days(df):
    print("\n=== Analysis by Day ===")
    print("H1: At least one day's mean grazing time is different")
    
    day_groups = [group for _, group in df.groupby('day')['grazing_percentage']]
    f_stat, p_value = stats.f_oneway(*day_groups)
    
    print(f"\nOne-way ANOVA results:")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    tukey = pairwise_tukeyhsd(df['grazing_percentage'], df['day'])
    print("\nTukey's HSD test results:")
    print(tukey)
    
    plot_tukey_results(tukey, "Comparisons Between Days", df.set_index('day')['grazing_percentage'])


class HeatMapMaker:
    """Makes heatmap of activity over the study."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the visualizer with configuration.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        width_in_inches = 190/25.4
        height_in_inches = width_in_inches * (.5)
        self.figure_size = (width_in_inches, height_in_inches)

        # self.activity_colors = {
        #     activity: color for activity, color in zip(
        #         self.config.labels.valid_activities,
        #         ['blue', 'green', 'red', 'orange', 'cyan', 'purple', 'brown']  # Add more colors if needed
        #     )
        # }
        
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare the dataset for visualization.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        df = df.copy()
        # df['mt'] = (df['posix_time']
        #             .apply(from_posix)
        #             .dt.tz_localize('UTC')  # First localize to UTC
        #             .dt.tz_convert(self.config.analysis.timezone))  # Then convert to desired timezone


        df['mt'] = from_posix_col(df['posix_time'], self.config.analysis.timezone)
        
        df["day"] = df["mt"].dt.date


        return df
    

    def _aggregate_by_day(self, df:pd.DataFrame, filter_out_weigh_days:bool=True):

        if self.config.visuals.heatmap.weigh_days is not None:
            weigh_days = self.config.visuals.heatmap.weigh_days
            if weigh_days is None:
                raise ValueError("Weigh Days must be defined in the config.")
            
            df = df[~df["day"].isin([wd.date() for wd in weigh_days])]

        grazing_pct = df.groupby(['ID', 'day']).apply(
            lambda x: (x['predicted_state'] == 'Grazing').mean() * 100
        ).reset_index(name='grazing_percentage')

        return grazing_pct
        
    
    def make_graph(self, grazing_pct:pd.DataFrame):
        # pivot data for heatmap
        heatmap_data = grazing_pct.pivot(
            index='ID',
            columns='day',
            values='grazing_percentage'
        )
        
        # Calculate average percentage per day (across all cows)
        day_averages = heatmap_data.mean(axis=0)
        
        # Calculate average percentage per cow (across all days)
        cow_averages = heatmap_data.mean(axis=1)
        
        # Calculate the overall mean grazing percentage
        overall_mean = np.mean(heatmap_data)
        print(f"Overall average grazing percentage: {overall_mean:.2f}%")
        
        # Calculate deviations from the overall mean
        day_deviations = day_averages - overall_mean
        cow_deviations = (cow_averages - overall_mean)[::-1]
        
        heatmap_data = heatmap_data.sort_index(ascending=True)
        
        # Create figure with 2x2 grid
        fig, axs = plt.subplots(2, 2, figsize=(self.figure_size[0]*1.2, self.figure_size[1]*1.5),
                            gridspec_kw={'width_ratios': [4, 1], 'height_ratios': [1, 4]})
        
        # 1,1: Day deviations from mean (top-left)
        ax_day_dev = axs[0, 0]
        
        # Create zero line (overall mean)
        ax_day_dev.axhline(y=0, color='gray', linestyle='-', alpha=0.7, linewidth=1)
        
        # Create x range for days
        x_positions = np.arange(len(day_averages)) + 0.5
        
        # Use bars for deviations
        pos_mask = day_deviations >= 0
        neg_mask = day_deviations < 0
        
        # Handle positive deviations
        if any(pos_mask):
            ax_day_dev.bar(x_positions[pos_mask], day_deviations[pos_mask],
                        color='#d95f02', alpha=0.6, edgecolor=None, width=1.0)
        
        # Handle negative deviations
        if any(neg_mask):
            ax_day_dev.bar(x_positions[neg_mask], day_deviations[neg_mask],
                        color='#1f78b4', alpha=0.6, edgecolor=None, width=1.0)
        
        # Set axis limits
        ax_day_dev.set_xlim(0, len(day_averages))
        
        # Round to nearest 5% for y-axis limits
        dev_max = max(5, 5 * math.ceil(max(abs(day_deviations)) / 5))
        ax_day_dev.set_ylim(-dev_max, dev_max)
        
        # Set ticks every 5%
        y_ticks = np.arange(-dev_max, dev_max + 5, 5)
        ax_day_dev.set_yticks(y_ticks)
        ax_day_dev.set_yticklabels([f'{x:+.1f}%' for x in y_ticks])  # + sign for positive values
        
        # Add a label showing the overall mean
        ax_day_dev.text(0.5, 0.97, f'Mean: {overall_mean:.1f}%', transform=ax_day_dev.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # No title for publication figure
        # ax_day_dev.set_title('Daily Deviation from Mean Grazing %', fontsize=12)
        ax_day_dev.set_ylabel('Deviation')
        ax_day_dev.set_xticks([])  # Hide x-ticks as they're shown in the heatmap
        ax_day_dev.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax_day_dev.spines['top'].set_visible(False)
        ax_day_dev.spines['right'].set_visible(False)
        
        # 2,1: Main heatmap (bottom-left)
        ax_heatmap = axs[1, 0]
        c = 100/3
        cmap = sns.diverging_palette(250, 30, l=60, s=80, center="light", as_cmap=True)
        heatmap = sns.heatmap(
            heatmap_data,
            annot=False,
            fmt=".0f",
            cmap=cmap,
            center=c,
            cbar=False,  # We'll create a separate colorbar
            ax=ax_heatmap
        )
        
        # Minimal axis labels for the heatmap
        # For publication, we can simplify by removing or reducing these
        # x_tick_positions = np.arange(len(heatmap_data.columns))[::4] + 0.5  # Show fewer labels (every 4th)
        # x_tick_labels = [d.strftime('%m-%d') for d in heatmap_data.columns[::4]]
        # ax_heatmap.set_xticks(x_tick_positions)
        # ax_heatmap.set_xticklabels(x_tick_labels, rotation=45, ha='right', fontsize=9)
        
        # Simplified y-axis (cow IDs) - consider removing or using smaller font
        # ax_heatmap.set_yticks(np.arange(len(heatmap_data.index)) + 0.5)
        # ax_heatmap.set_yticklabels(heatmap_data.index, fontsize=7)
                
        ax_heatmap.set_xticks([])  # For x-axis
        ax_heatmap.set_yticks([])  # For y-axis

        ax_heatmap.set_xlabel("Date", fontsize=11)
        ax_heatmap.set_ylabel("Cow ID", fontsize=11)
        
        # 1,2: Colorbar (top-right)
        ax_colorbar = axs[0, 1]
        # Create a ScalarMappable with the same colormap as the heatmap
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        
        # Define colorbar range similar to the heatmap
        vmin = min(heatmap_data.min().min(), c - (c - heatmap_data.min().min()))
        vmax = max(heatmap_data.max().max(), c + (heatmap_data.max().max() - c))
        sm.set_clim(vmin, vmax)
        
        # Add colorbar to the specified axes
        cbar = plt.colorbar(sm, cax=ax_colorbar)
        cbar.set_label('Grazing %')
        
        # 2,2: Cow deviations from mean (bottom-right)
        ax_cow_dev = axs[1, 1]
        
        # Create zero line (overall mean)
        ax_cow_dev.axvline(x=0, color='gray', linestyle='-', alpha=0.7, linewidth=1)
        
        # Create horizontal positions for each cow (align with heatmap y-positions)
        y_positions = np.arange(len(cow_averages)) + 0.5
        
        # Create horizontal bars for cow deviations
        pos_mask = cow_deviations >= 0
        neg_mask = cow_deviations < 0
        
        # Handle positive deviations
        if any(pos_mask):
            ax_cow_dev.barh(y_positions[pos_mask], cow_deviations[pos_mask],
                        height=0.8, color='#d95f02', alpha=0.6, edgecolor=None)
        
        # Handle negative deviations
        if any(neg_mask):
            ax_cow_dev.barh(y_positions[neg_mask], cow_deviations[neg_mask],
                        height=0.8, color='#1f78b4', alpha=0.6, edgecolor=None)
        
        # Set y-axis limits to match heatmap
        ax_cow_dev.set_ylim(0, len(cow_averages))
        ax_cow_dev.set_yticks([])  # Hide y-ticks as they're shown in the heatmap
        
        # Round to nearest 5% for x-axis limits
        dev_max = max(5, 5 * math.ceil(max(abs(cow_deviations)) / 5))
        ax_cow_dev.set_xlim(-dev_max, dev_max)
        
        # Set ticks every 5%
        x_ticks = np.arange(-dev_max, dev_max + 5, 5)
        ax_cow_dev.set_xticks(x_ticks)
        ax_cow_dev.set_xticklabels([f'{x:+.0f}%' for x in x_ticks], fontsize=8)  # + sign for positive values
        
        ax_cow_dev.set_xlabel('Deviation', fontsize=9)
        # No title for bottom right in publication
        # ax_cow_dev.set_title('Cow Deviation', fontsize=10)
        ax_cow_dev.grid(True, axis='x', linestyle='--', alpha=0.3)
        ax_cow_dev.spines['top'].set_visible(False)
        ax_cow_dev.spines['right'].set_visible(False)
        
        # Adjust layout
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.config.visuals.visuals_root_path, 'heatmap_of_grazing.png'), dpi=300, bbox_inches='tight')
        plt.close()


    # Day wise aggregations
    # def make_graph(self, grazing_pct:pd.DataFrame):
    #     # pivot data for heatmap
    #     heatmap_data = grazing_pct.pivot(
    #         index='ID',
    #         columns='day',
    #         values='grazing_percentage'
    #     )
        
    #     # Calculate average percentage per day (across all cows)
    #     day_averages = heatmap_data.mean(axis=0)
        
    #     # Print overall average 
    #     print(f"Overall average grazing percentage: {np.mean(heatmap_data):.2f}%")
        
    #     heatmap_data = heatmap_data.sort_index(ascending=True)
        
    #     # Create figure with additional space for the average plot
    #     fig, (ax_avg, ax) = plt.subplots(2, 1, 
    #                                     figsize=(self.figure_size[0], self.figure_size[1]*1.2),
    #                                     gridspec_kw={'height_ratios': [1, 4]},
    #                                     layout='constrained')
        
    #     # Plot the daily averages on top subplot
    #     ax_avg.plot(np.arange(len(day_averages)), day_averages, 'o-', color='#d95f02', linewidth=1.5)
    #     ax_avg.set_xlim(0, len(day_averages)-1)
    #     ax_avg.set_ylim(0, max(day_averages) * 1.1)
    #     ax_avg.set_title('Average % of Day Spent Grazing', fontsize=12)
    #     ax_avg.set_ylabel('Percentage')
    #     ax_avg.set_xticks([])  # Hide x-ticks as they're shown in the heatmap
    #     ax_avg.grid(True, linestyle='--', alpha=0.7)
        
    #     # Create the heatmap on the bottom subplot
    #     c = 100/3
    #     cmap = sns.diverging_palette(250, 30, l=60, s=80, center="light", as_cmap=True)
    #     heatmap = sns.heatmap(
    #         heatmap_data,
    #         annot=False,
    #         fmt=".0f",
    #         cmap=cmap,
    #         center=c,
    #         cbar=True,
    #         ax=ax
    #     )
        
    #     cbar = heatmap.collections[0].colorbar
    #     cbar.set_ticklabels([f'{x:.0f}%' for x in cbar.get_ticks()])
    #     ax.tick_params(axis='y', labelsize=12)
        
    #     x_tick_positions = np.arange(len(heatmap_data.columns))[::2] + 0.5
    #     x_tick_labels = [d.strftime('%m-%d') for d in heatmap_data.columns[::2]]
        
    #     # Set the positions and labels
    #     ax.set_xticks(x_tick_positions)
    #     ax.set_xticklabels(x_tick_labels, rotation=45, ha='right', fontsize=10)
    #     ax.set_yticks([])
    #     ax.set_yticklabels([])
    #     ax.set_xlabel("Date", fontsize=14)
    #     ax.set_ylabel("Cow ID", fontsize=14)
        
    #     plt.savefig(os.path.join(self.config.visuals.visuals_root_path, 'heatmap_of_grazing.png'), dpi=300)
    #     plt.close()



    # Used for paper. No annotations or title. 
    # def make_graph(self, grazing_pct:pd.DataFrame):


    #     # print(grazing_pct.describe())
    #     heatmap_data = grazing_pct.pivot(
    #         index='ID',
    #         columns='day',
    #         values='grazing_percentage'
    #     )
    #     print(f"Overall average grazing percentage: {np.mean(heatmap_data):.2f}%")

    #     heatmap_data = heatmap_data.sort_index(ascending=True)
    #     fig, ax = plt.subplots(layout='constrained', figsize=self.figure_size)
        
    #     # sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="coolwarm", cbar=False, ax=ax, center=100/3)
    #     c = 100/3
    #     # vmin, vmax = (1/3)*c, (5/3)*c  # Narrower range around 33
    #     # vmin, vmax = 0, 100  # Narrower range around 33
    #     cmap = sns.diverging_palette(250, 30, l=60, s=80, center="light", as_cmap=True)
    #     heatmap = sns.heatmap(
    #         heatmap_data, 
    #         annot=False, 
    #         fmt=".0f", 
    #         cmap=cmap,
    #         # cmap="RdYlBu_r",
    #         # cmap="coolwarm",
    #         center=c,
    #         # vmin=vmin,
    #         # vmax=vmax,
    #         cbar=True,  # Add colorbar to show the scale
    #         ax=ax
    #     )
        
    #     cbar = heatmap.collections[0].colorbar
    #     cbar.set_ticklabels([f'{x:.0f}%' for x in cbar.get_ticks()])
        
    #     ax.tick_params(axis='y', labelsize=12)
    #     # ax.set_xticks([x + 0.5 for x in range(len(heatmap_data.columns))])
    #     # date_labels = [d.strftime('%m-%d') for d in heatmap_data.columns]
    #     # # ax.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=10)
    #     # ax.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=10)



    #     x_tick_positions = np.arange(len(heatmap_data.columns))[::2] + 0.5
    #     x_tick_labels = [d.strftime('%m-%d') for d in heatmap_data.columns[::2]]
    #     # Set the positions and labels
    #     ax.set_xticks(x_tick_positions)
    #     ax.set_xticklabels(x_tick_labels, rotation=45, ha='right', fontsize=10)

    #     ax.set_yticks([])

    #     ax.set_yticklabels([])
    #     ax.set_xlabel("Date", fontsize=14)
    #     ax.set_ylabel("Cow ID", fontsize=14)
    #     plt.savefig(os.path.join(self.config.visuals.visuals_root_path, 'heatmap_of_grazing.png'), dpi=300)
    #     # plt.show()
    #     plt.close()




    # def make_graph(self, grazing_pct: pd.DataFrame):
    #     """Generate a more compact and readable heatmap."""
        
    #     # Pivot data for heatmap
    #     heatmap_data = grazing_pct.pivot(
    #         index='ID',
    #         columns='day',
    #         values='grazing_percentage'
    #     )
        
    #     mean_grazing = np.mean(heatmap_data)
    #     print(f"Mean grazing percentage: {mean_grazing:.1f}%")
        
    #     # Sort IDs in ascending order
    #     heatmap_data = heatmap_data.sort_index(ascending=True)
        
    #     # Create figure with publication dimensions (190mm wide)
    #     fig, ax = plt.subplots(figsize=self.figure_size, dpi=300)
        
    #     # Calculate center value for diverging colormap
    #     c = 100/3  # ~33.33% as center point
        
    #     # Use a simpler colormap with better contrast
    #     cmap = sns.diverging_palette(240, 10, s=80, l=55, as_cmap=True)
        
    #     # Determine font sizes based on grid size
    #     n_rows, n_cols = heatmap_data.shape
    #     annot_fontsize = max(4, min(7, 150 / (n_rows * n_cols/2)))  # Smaller annotations
        
    #     # Create heatmap with simplified appearance
    #     hm = sns.heatmap(
    #         heatmap_data,
    #         annot=True,
    #         fmt=".0f",  # Show integer percentages
    #         cmap=cmap,
    #         center=c,
    #         cbar=False,  # Remove colorbar for cleaner look
    #         linewidths=0.1,  # Thinner cell borders
    #         annot_kws={'fontsize': annot_fontsize, 'weight': 'normal'},
    #         ax=ax
    #     )
        
    #     # Set x-axis tick properties with less information
    #     # Reduce number of date labels by showing every n-th date
    #     date_stride = max(1, n_cols // 10)  # Show at most 10 date labels
    #     ax.set_xticks([x + 0.5 for x in range(0, len(heatmap_data.columns), date_stride)])
    #     ax.set_xticklabels([heatmap_data.columns[i] for i in range(0, len(heatmap_data.columns), date_stride)], 
    #                     rotation=45, ha='right', fontsize=6)
        
    #     # Set y-axis tick properties
    #     ax.set_yticks([y + 0.5 for y in range(len(heatmap_data.index))])
    #     ax.set_yticklabels(heatmap_data.index, fontsize=6)
        
    #     # Simplified axis labels with smaller font
    #     ax.set_xlabel("Date", fontsize=8, labelpad=5)
    #     ax.set_ylabel("Collar ID", fontsize=8, labelpad=5)
        
    #     # Remove the title to reduce clutter
        
    #     # Tighten layout to make the best use of space
    #     plt.tight_layout()
        
    #     # Save figure with appropriate resolution
    #     output_path = os.path.join(self.config.visuals.visuals_root_path, 'heatmap_of_grazing.png')
    #     plt.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
        
    #     # Also save PDF version for publication
    #     pdf_path = os.path.join(self.config.visuals.visuals_root_path, 'heatmap_of_grazing.pdf')
    #     plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
        
    #     plt.show()
    #     plt.close()
        
    #     print(f"Heatmap saved to {output_path}")


    def super_manual(self):

        d = {
            "ID" : [996, 998, 1021, 1015, 988, 1022, 999, 828, 837, 1030, 1006, 1028, 824, 1017, 832, 993, 830, 827, 1008, 838, 826, 831],
            "Grazing_pct" : [19.93631495, 20.55396404, 20.98718189, 22.70918938, 23.10080956, 30.12403216, 30.55111842, 33.08767417, 33.87003538, 34.15033312, 35.1696882, 35.31140005, 35.59029895, 36.27281081, 36.55143398, 36.75395755, 36.95381571, 37.04254371, 38.01600455, 38.4118246, 39.04647907, 40.96997032],
            "Length_of_bar": [5, 5, 5, 5, 5, 5, 6, 13, 15, 15, 15, 14, 14, 14, 15, 15, 15, 15, 14, 14, 12, 8],
            "Start_of_the_bar": [1, 1, 1, 1, 1, 6, 6, 6, 6, 6, 7, 8, 8, 8, 8, 8, 8, 8, 9, 9, 11, 15]
        }

        df = pd.DataFrame(d)

        fig, ax = plt.subplots()
        xticks = [i for i in range(len(d["ID"]))]
        ax.bar(xticks, df["Grazing_pct"])
        ax.set_xticklabels(d['ID'])

        y=45

        plb, psb = 0,0
        for id, gp, lb, sb in zip(d['ID'], d['Grazing_pct'], d['Length_of_bar'], d['Start_of_the_bar']):
            if not ((lb == plb) & (sb == psb)):
                ax.hlines(y = y, xmin=sb-1, xmax=sb+lb-2)
                plb = lb
                psb = sb
                y+=2

        plt.show()

    def run(self, df:pd.DataFrame):

        df = self._prepare_data(df)

        # if self.config.visuals.weigh_days is not None:

        grazing_pct = self._aggregate_by_day(df)

        # self.super_manual()

        # analyze_cows(grazing_pct)
        # analyze_days(grazing_pct)



        self.make_graph(grazing_pct=grazing_pct)
