import statsmodels.api as sm
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from pathlib import Path
from scipy import stats
from cowstudyapp.utils import from_posix_col
from cowstudyapp.config import ConfigManager

# Add helper function
def format_p_value(p):
    if p < 0.001:
        return "<0.001"
    elif p < 0.01:
        return "<0.01"
    elif p < 0.05:
        return "<0.05"
    else:
        return f"={p:.3f}"


fontsizes = {
    1: 16,
    2: 18,
    3: 20,
    4: 24,
    5: 28
}

class GrazingVersusCowInfo:
    def __init__(self, config: ConfigManager):
        self.config = config

    def _prepare_data(self, df):
        df['mt'] = from_posix_col(df['posix_time'], self.config.analysis.timezone)
        df["date"] = df["mt"].dt.date
        return df

    def _get_cow_info(self):
        if not Path.exists(self.config.io.cow_info_path):
            raise ValueError("Must define a config.io.cow_info_path to use this function.")
            
        return pd.read_excel(self.config.io.cow_info_path)




    def pairplot_cow_info(self, df):
        cow_info = self._get_cow_info()
        cow_info['age'] = 2022 - cow_info['year_b']
        cow_info.rename(columns={'BW_preg':'weight_lbs'}, inplace=True)
        cow_info = cow_info[cow_info.weight_lbs > 0]
        cow_info['weight_kg'] = cow_info['weight_lbs'] * 0.4535924
        cow_info = cow_info[['age', 'weight_kg', 'TRT']]

        # Create figure with better spacing
        fig, axs = plt.subplots(2, 2, figsize=(7, 7))
        plt.subplots_adjust(hspace=0.3, wspace=0.3)

        for rowi, row in enumerate(axs):
            R = ['weight_kg', 'age'][rowi]
            for coli, ax in enumerate(row):
                C = ['weight_kg', 'age'][coli]
                
                if rowi == 0 and coli == 1:
                    ax.remove()  # Remove the unused axis
                    continue
                elif R == C:
                    # Improve histogram appearance
                    # Use different bins depending on the variable
                    if R == 'age':
                        bins = range(int(cow_info[R].min()), int(cow_info[R].max()) + 2)  # +2 to include the last value
                    else:
                        bins = 15
                        
                    ax.hist(cow_info[R], bins=bins, edgecolor='black', alpha=0.7)
                    ax.grid(True, alpha=0.3)
                    
                    # Add mean line
                    mean_val = cow_info[R].mean()
                    ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8)
                    ax.text(mean_val, ax.get_ylim()[1],
                        f'Mean: {mean_val:.1f}',
                        rotation=0, color='red',
                        ha='center', va='bottom')
                    
                    # Add "Count" label for histograms
                    ax.set_ylabel('Count', fontsize=10)
                    
                else:
                    # Improve scatter plot appearance
                    ax.scatter(cow_info[R], cow_info[C], alpha=0.6)
                    ax.grid(True, alpha=0.3)
                    
                    # Add trend line
                    z = np.polyfit(cow_info[R], cow_info[C], 1)
                    p = np.poly1d(z)
                    ax.plot(cow_info[R], p(cow_info[R]), "r--", alpha=0.8)
                    
                    # Set ylabel for scatter plots
                    ax.set_ylabel(f'{C} {"(kg)" if "weight" in C else "(years)"}', fontsize=10)
                
                ax.set_xlabel(f'{R} {"(kg)" if "weight" in R else "(years)"}', fontsize=10)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(labelsize=9)

        fig.suptitle("Comparison of Cow Attributes",
                    fontweight='bold',
                    fontsize=16,
                    y=0.95)
        
        plt.show()


    def compare_cow_info(self, df, min_records=180):

        def age_rules(x):
            if x <= 5:
                return "3-5"
            elif x <= 8:
                return "6-8"
            return "9+"

        df = self._prepare_data(df)
        cow_info = self._get_cow_info()

        # Calculate activity counts per cow per day
        cow_day_cts = df.groupby(["ID", "date"])\
            .agg(
                **{
                    f"{state}_count": ('predicted_state', lambda x, s=state: (x == s).sum())
                    for state in self.config.analysis.hmm.states
                },
                total_count=('predicted_state', 'count')
            ).reset_index()

        # Calculate age
        cow_info['age'] = 2022 - cow_info['year_b']

        cow_info['age_group'] = cow_info['age'].apply(lambda x: age_rules(x))
        print(cow_info.head())



        # Merge cow info with activity data
        analysis_df = cow_day_cts.merge(
            cow_info, 
            left_on="ID", 
            right_on="collar_id", 
            how="inner"
        ).drop(columns=["collar_id", "cow_id", "TRT"])  # Remove TRT column

        # Filter for minimum records and valid body weight
        analysis_df = analysis_df[analysis_df["total_count"] >= min_records]
        analysis_df = analysis_df[analysis_df["BW_preg"] > 0]

        # Calculate percentages for each activity state
        for state in self.config.analysis.hmm.states:
            analysis_df[f'{state}_percentage'] = (analysis_df[f'{state}_count'] / analysis_df['total_count']) * 100

        # Melt the DataFrame for easier analysis
        melted_df = pd.melt(
            analysis_df,
            id_vars=['ID', 'date', 'age', 'age_group', 'BW_preg', 'sex'],  # Removed BCS_preg
            value_vars=[f'{state}_percentage' for state in self.config.analysis.hmm.states],
            var_name='activity',
            value_name='percentage'
        )

        # Clean up activity names
        melted_df['activity'] = melted_df['activity'].str.replace('_percentage', '')

        # Calculate mean percentages per cow
        collar_means = melted_df.groupby(['ID', 'activity', 'age', 'age_group', 'BW_preg', 'sex'])['percentage'].mean().reset_index()
        
        # Create separate plots for each activity type
        # This completely avoids the subplot spacing issue
        activities = self.config.analysis.hmm.states
        
        # Set a clean style with high contrast

        # Use more distinguishable colors
        box_color = '#3498db'  # Bright blue
        line_color = '#c0392b'  # Dark red
        scatter_color = '#2ecc71'  # Green
        
        # Create a directory for individual plots if it doesn't exist
        plots_dir = os.path.join(self.config.visuals.visuals_root_path, 'cow_activity_plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Create a master figure for all plots
        master_fig, master_axes = plt.subplots(len(activities), 2, figsize=(20, 9*len(activities)))
        
        plt.subplots_adjust(hspace=0.4, wspace=0.3)

        sns.set_style("white")
        
        # For better readability
        # plt.rcParams.update({
        #     'font.size': 18,

        #     'axes.labelsize': 20,
        #     'axes.titlesize': 20,
        #     'xtick.labelsize': 28,
        #     'ytick.labelsize': 18,
        #     'legend.fontsize': 18,
        # })


        master_fig.suptitle('Cow Characteristics vs Daily Activity Patterns'
                            , fontsize=fontsizes[4]
                            , fontweight='bold'
                            , y=0.95)
        
        for idx, activity in enumerate(activities):
            activity_data = collar_means[collar_means['activity'] == activity]
            
            # Create subplot for Age
            # Sort by age for better visualization
            age_order = sorted(activity_data['age_group'].unique())


            sns.boxplot(data=activity_data, x='age_group', y='percentage', ax=master_axes[idx, 0],
                       color=box_color, width=0.6, order=age_order)

            # Add mean points with more visible markers
            sns.pointplot(data=activity_data, x='age_group', y='percentage',
                        color=line_color, ci=95, markers='D',
                        scale=0.8, join=True, order=age_order, ax=master_axes[idx, 0])

            master_axes[idx, 0].set_title(f'Age Group vs {activity}'
                                          , fontsize=fontsizes[4]
                                          , fontweight='bold')
            master_axes[idx, 0].set_xlabel('Age Group (years)'
                                           , fontsize=fontsizes[3]
                                           , fontweight='bold')
            master_axes[idx, 0].set_ylabel(f'% Time {activity}'
                                           , fontsize=fontsizes[3]
                                           , fontweight='bold')
            
            # Make tick labels darker and larger
            master_axes[idx, 0].tick_params(axis='both', colors='black', labelsize=fontsizes[2])
            for label in master_axes[idx, 0].get_xticklabels() + master_axes[idx, 0].get_yticklabels():
                label.set_fontweight('bold')
                
            # Add grid for better readability
            master_axes[idx, 0].grid(True, linestyle='--', alpha=0.7)

            # Calculate p-value for age and add interpretation
            age_groups = activity_data['age_group'].unique()
            if len(age_groups) > 1:
                f_stat, p_val = stats.f_oneway(*[
                    activity_data[activity_data['age_group'] == age_group]['percentage']
                    for age_group in age_groups
                ])
                
                # Interpret the statistical significance
                if p_val < 0.05:
                    sig_text = f"1-way ANOVA\n(p{format_p_value(p_val)})"
                else:
                    sig_text = f"No significant effect\n(p{format_p_value(p_val)})"
                
                # Place the annotation at the bottom of the plot, not using transform
                y_min, y_max = master_axes[idx, 0].get_ylim()
                text_y = y_min - (y_max - y_min) * 0.15  # Position below the x-axis
                master_axes[idx, 0].text(
                    (len(age_order)-1)*.5,  # Center of x-axis
                    text_y, 
                    sig_text, 
                    ha='center',
                    fontsize=fontsizes[3],
                    bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9, boxstyle='round,pad=0.5')
                )
                # Extend y-axis limits to include the annotation
                master_axes[idx, 0].set_ylim(text_y - (y_max - y_min) * 0.1, y_max)

            # Create subplot for Weight
            activity_data_weight = activity_data.dropna(subset=['BW_preg'])
            
            # Higher contrast plotting
            scatter = master_axes[idx, 1].scatter(
                activity_data_weight['BW_preg'],
                activity_data_weight['percentage'],
                c=scatter_color,
                s=80,
                alpha=0.7,
                edgecolor='black'
            )
            
            # Add regression line
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                activity_data_weight['BW_preg'],
                activity_data_weight['percentage']
            )
            
            x_range = np.linspace(
                activity_data_weight['BW_preg'].min(),
                activity_data_weight['BW_preg'].max(),
                100
            )
            master_axes[idx, 1].plot(
                x_range,
                intercept + slope * x_range,
                color=line_color,
                linewidth=3
            )
            
            # Add confidence band
            from statsmodels.sandbox.regression.predstd import wls_prediction_std
            X = sm.add_constant(activity_data_weight['BW_preg'])
            model = sm.OLS(activity_data_weight['percentage'], X).fit()
            
            x_pred = sm.add_constant(x_range)
            y_pred = model.predict(x_pred)
            _, lower, upper = wls_prediction_std(model, x_pred, alpha=0.05)
            
            master_axes[idx, 1].fill_between(
                x_range,
                lower,
                upper,
                color=line_color,
                alpha=0.2
            )
                        
            master_axes[idx, 1].set_title(f'Body Weight vs {activity}'
                                          , fontsize=fontsizes[4]
                                          , fontweight='bold')
            master_axes[idx, 1].set_xlabel('Body Weight (lbs)'
                                           , fontsize=fontsizes[3]
                                           , fontweight='bold')
            master_axes[idx, 1].set_ylabel(f'% Time {activity}'
                                           , fontsize=fontsizes[3]
                                           , fontweight='bold')
            
            # Make tick labels darker and larger
            master_axes[idx, 1].tick_params(axis='both', colors='black', labelsize=fontsizes[2])
            for label in master_axes[idx, 1].get_xticklabels() + master_axes[idx, 1].get_yticklabels():
                label.set_fontweight('bold')
                
            # Add grid for better readability
            master_axes[idx, 1].grid(True, linestyle='--', alpha=0.7)

            weight_range = activity_data_weight['BW_preg'].max() - activity_data_weight['BW_preg'].min()
            effect_magnitude = abs(slope * weight_range)
            
            if p_value < 0.05:
                weight_text = f"Significant relationship\n(p{format_p_value(p_value)}, R²={r_value**2:.3f})"
            else:
                weight_text = f"No significant effect\n(p{format_p_value(p_value)})"
                
            # Place the annotation at the bottom of the plot, not using transform
            y_min, y_max = master_axes[idx, 1].get_ylim()
            text_y = y_min - (y_max - y_min) * 0.15  # Position below the x-axis
            master_axes[idx, 1].text(
                (activity_data_weight['BW_preg'].max() + activity_data_weight['BW_preg'].min()) / 2,  # Center of x-axis
                text_y, 
                weight_text, 
                ha='center',
                fontsize=fontsizes[3],
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9, boxstyle='round,pad=0.5')
            )
            # Extend y-axis limits to include the annotation
            master_axes[idx, 1].set_ylim(text_y - (y_max - y_min) * 0.1, y_max)

        # Save the master figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])


        
        plt.savefig(os.path.join(self.config.visuals.visuals_root_path, 'cow_characteristics_vs_activity.png'),
                   dpi=300, bbox_inches='tight')
        plt.close(master_fig)