from matplotlib import pyplot as plt, cm

from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from matplotlib.dates import MO, DateFormatter, WeekdayLocator, DayLocator
import matplotlib.patches as mpatches
import seaborn as sns

from pathlib import Path
import logging
from datetime import datetime

from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm

from cowstudyapp.utils import from_posix_col
from cowstudyapp.config import ConfigManager

class GrazingVersusCowInfo:
    def __init__(self, config: ConfigManager):
        self.config=config


    def _prepare_data(self, df):

        df['mt'] = from_posix_col(df['posix_time'], self.config.analysis.timezone)

        df["date"] = df["mt"].dt.date

        # df = df[~df['date'].isin(self.config.visuals.weigh_days)]
        return df

    def _get_cow_info(self):
        if not Path.exists(self.config.io.cow_info_path):
            raise ValueError("Must define a config.io.cow_info_path to use this function.")
            
        return pd.read_excel(self.config.io.cow_info_path)




    # def make_graph(self, min_records=180) -> None:
    def compare_cow_info(self, df, min_records = 180):
        df = self._prepare_data(df)
        cow_info = self._get_cow_info()

        cow_day_cts = df.groupby(["ID", "date"])\
            .agg(
                **{
                    f"{state}_count": ('predicted_state', lambda x, s=state: (x == s).sum())
                    for state in self.config.analysis.hmm.states
                },
                total_count=('predicted_state', 'count')  # or just 'size' would work too
            ).reset_index()

        cow_info['age'] = 2022 - cow_info['year_b']

        analysis_df = cow_day_cts.merge(
            cow_info, 
            left_on="ID", 
            right_on="collar_id", 
            how="inner"
        ).drop(columns=["collar_id", "cow_id"])

        analysis_df = analysis_df[analysis_df["total_count"] >= min_records]
        analysis_df = analysis_df[analysis_df["BW_preg"] > 0]


        for state in self.config.analysis.hmm.states:
            analysis_df[f'{state}_percentage'] = (analysis_df[f'{state}_count'] / analysis_df['total_count']) * 100

        # Melt the DataFrame to get it into the same format as the old analysis
        melted_df = pd.melt(
            analysis_df,
            id_vars=['ID', 'date', 'TRT', 'age', 'BW_preg', 'BCS_preg', 'sex'],
            value_vars=[f'{state}_percentage' for state in self.config.analysis.hmm.states],
            var_name='activity',
            value_name='percentage'
        )

        # Clean up activity names by removing '_percentage'
        melted_df['activity'] = melted_df['activity'].str.replace('_percentage', '')

        # Calculate mean percentages per collar (same as collar_means in old code)
        collar_means = melted_df.groupby(['ID', 'activity', 'TRT', 'age', 'BW_preg', 'BCS_preg', 'sex'])['percentage'].mean().reset_index()

        # Create separate figures for each activity
        activities = self.config.analysis.hmm.states
        # First, adjust the figure layout to leave room for annotations
        fig, axes = plt.subplots(len(activities), 4, figsize=(20, 15))
        # Add more space at the bottom of the figure
        plt.subplots_adjust(bottom=0.2)

        fig.suptitle('Cow Characteristics vs Daily Activity Patterns')
        for idx, activity in enumerate(activities):
            activity_data = collar_means[collar_means['activity'] == activity]
            

            # melted_df['age'] = 2022 - melted_df['year_b']  # or whatever your study year is

            # In the plotting section, replace year_b with age
            # Birth Year Analysis becomes Age Analysis
            sns.boxplot(data=activity_data, x='age', y='percentage', ax=axes[idx,0])

            # Add mean and CI
            sns.pointplot(data=activity_data, x='age', y='percentage', 
                        color='red', ci=95, markers='_', 
                        scale=0.5, ax=axes[idx,0])

            axes[idx,0].set_title(f'Age vs {activity}')
            axes[idx,0].set_xlabel('Age (years)')
            axes[idx,0].set_ylabel(f'Average % Day {activity}')

            # You might also want to sort the x-axis in ascending order of age
            activity_data = activity_data.sort_values('age')


            # 1. Birth Year Analysis with Effect Size
            # sns.boxplot(data=activity_data, x='year_b', y='percentage', ax=axes[idx,0])
            
            # # Add mean and CI
            # sns.pointplot(data=activity_data, x='year_b', y='percentage', 
            #             color='red', ci=95, markers='_', 
            #             scale=0.5, ax=axes[idx,0])
            
            # axes[idx,0].set_title(f'Birth Year vs {activity}')
            # axes[idx,0].set_xlabel('Birth Year')
            # axes[idx,0].set_ylabel(f'Average % Day {activity}')
            
            # Calculate effect size (eta-squared) for birth year
            ages = activity_data['age'].unique()
            if len(ages) > 1:  # Only calculate if we have multiple years
                f_stat, p_val = stats.f_oneway(*[
                    activity_data[activity_data['age'] == age]['percentage'] 
                    for age in ages
                ])
                df_between = len(ages) - 1
                df_total = len(activity_data) - 1
                eta_sq = (df_between * f_stat) / (df_between * f_stat + df_total)
                # axes[idx,0].text(0.5, -0.15, f'p = {p_val:.4f}\nη² = {eta_sq:.3f}', 
                #             ha='center', transform=axes[idx,0].transAxes)

                axes[idx,0].text(0.5, -0.35, 
                                f'One-way ANOVA:\np = {p_val:.4f}\nη² = {eta_sq:.3f}', 
                                ha='center', 
                                transform=axes[idx,0].transAxes)


            # 2. Body Weight Analysis
            activity_data_weight = activity_data.dropna(subset=['BW_preg'])
            sns.regplot(data=activity_data_weight, 
                        x='BW_preg', 
                        y='percentage', 
                        ax=axes[idx,1],
                        scatter_kws={'alpha':0.5},
                        line_kws={'color': 'red'})
                        
            axes[idx,1].set_title(f'Body Weight vs {activity}')
            axes[idx,1].set_xlabel('Body Weight (lbs)')
            axes[idx,1].set_ylabel(f'Average % Day {activity}')

            # Weight correlation and effect size
            # corr, p_val = stats.pearsonr(activity_data_weight['BW_preg'], 
            #                             activity_data_weight['percentage'])
            # # Calculate R-squared
            # r_squared = corr ** 2

            # axes[idx,1].text(0.5, -0.35, 
            #     f'p = {p_val:.4f}\nR² = {r_squared:.3f}', 
            #     ha='center', 
            #     transform=axes[idx,1].transAxes)
            

            pearson_corr, pearson_p = stats.pearsonr(activity_data_weight['BW_preg'], 
                                                    activity_data_weight['percentage'])
            spearman_corr, spearman_p = stats.spearmanr(activity_data_weight['BW_preg'], 
                                                    activity_data_weight['percentage'])

            axes[idx,1].text(0.5, -0.35, 
                f'Pearson:\nR²={pearson_corr**2:.3f}\np={pearson_p:.4f}\n', 
                ha='center', 
                transform=axes[idx,1].transAxes)

            # axes[idx,1].text(0.5, -0.35, 
            #     f'Pearson: r={pearson_corr:.3f}, p={pearson_p:.4f}\n' + 
            #     f'Spearman: ρ={spearman_corr:.3f}, p={spearman_p:.4f}', 
            #     ha='center', 
            #     transform=axes[idx,1].transAxes)




            # 3. BCS Analysis
            activity_data_bcs = activity_data.dropna(subset=['BCS_preg'])
            sns.regplot(data=activity_data_bcs, 
                        x='BCS_preg', 
                        y='percentage', 
                        ax=axes[idx,2],
                        scatter_kws={'alpha':0.5},
                        line_kws={'color': 'red'})

            axes[idx,2].set_title(f'Body Condition Score vs {activity}')
            axes[idx,2].set_xlabel('BCS')
            axes[idx,2].set_ylabel(f'Average % Day {activity}')

            # # BCS correlation and effect size
            # bcs_corr, bcs_p = stats.pearsonr(activity_data_bcs['BCS_preg'], 
            #                                 activity_data_bcs['percentage'])
            # # Calculate R-squared
            # bcs_r_squared = bcs_corr ** 2

            # # axes[idx,2].text(0.5, -0.15, 
            # #                 f'r = {bcs_corr:.3f}\np = {bcs_p:.4f}\nR² = {bcs_r_squared:.3f}', 
            # #                 ha='center', 
            # #                 transform=axes[idx,2].transAxes)

            # axes[idx,2].text(0.5, -0.35, 
            #     f'p = {bcs_p:.4f}\nR² = {bcs_r_squared:.3f}', 
            #     ha='center', 
            #     transform=axes[idx,2].transAxes)
            


            bcs_pearson_corr, bcs_pearson_p = stats.pearsonr(activity_data_bcs['BCS_preg'], 
                                                            activity_data_bcs['percentage'])
            bcs_spearman_corr, bcs_spearman_p = stats.spearmanr(activity_data_bcs['BCS_preg'], 
                                                                activity_data_bcs['percentage'])

            axes[idx,2].text(0.5, -0.35, 
                f'Pearson:\nR²={bcs_pearson_corr**2:.3f}\np={bcs_pearson_p:.4f}\n', 
                ha='center', 
                transform=axes[idx,2].transAxes)






            # 4. Treatment Analysis with Effect Size
            axes[idx,3].set_title(f'Treatment vs {activity}')  # Add missing title
            sns.boxplot(data=activity_data, x='TRT', y='percentage', ax=axes[idx,3])
            
            # Add mean and CI
            sns.pointplot(data=activity_data, x='TRT', y='percentage', 
                        color='red', ci=95, markers='_', 
                        scale=0.5, ax=axes[idx,3])
            
            # Calculate effect size (Cohen's d) between treatments
            treatments = sorted(activity_data['TRT'].unique())
            if len(treatments) > 1:
                # Calculate pairwise Cohen's d
                cohens_d_dict = {}
                for i, trt1 in enumerate(treatments):
                    for trt2 in treatments[i+1:]:
                        group1 = activity_data[activity_data['TRT'] == trt1]['percentage']
                        group2 = activity_data[activity_data['TRT'] == trt2]['percentage']
                        
                        # Pooled standard deviation
                        n1, n2 = len(group1), len(group2)
                        s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
                        s_pooled = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2))
                        
                        # Cohen's d
                        d = (np.mean(group1) - np.mean(group2)) / s_pooled
                        cohens_d_dict[f'{trt1}-{trt2}'] = d

                # Add Cohen's d to plot
                treatments = activity_data['TRT'].unique()
                
                
                h_stat, p_val = stats.kruskal(*[
                    activity_data[activity_data['TRT'] == trt]['percentage'] 
                    for trt in treatments
                ])

                # axes[idx,3].text(0.5, -0.15, f'p = {p_val:.4f}', ha='center', transform=axes[idx,3].transAxes)
                d_text = '\n'.join([f'{k}: d={v:.2f}' for k,v in cohens_d_dict.items()]) + f"\nKruskal p = {p_val:.4f}"

                # d_text = '\n'.join([f'{k}: d={v:.2f}' for k,v in cohens_d_dict.items()])
                axes[idx,3].text(0.5, -0.35, 
                                d_text, 
                                ha='center', 
                                transform=axes[idx,3].transAxes)




        plt.tight_layout(pad=3.0, h_pad=2.0, w_pad=1.0)
        plt.savefig(os.path.join(self.config.visuals.visuals_root_path, 'activity_analysis.png'))
        plt.close()

        # Enhanced printing of statistics
        for activity in activities:
            activity_data = collar_means[collar_means['activity'] == activity]
            print(f"\n=== {activity} Analysis ===")
            
            # Print effect sizes along with other statistics
            print(f"\n{activity} Treatment Effects:")
            for trt1 in treatments:
                for trt2 in treatments:
                    if trt1 < trt2:
                        group1 = activity_data[activity_data['TRT'] == trt1]['percentage']
                        group2 = activity_data[activity_data['TRT'] == trt2]['percentage']
                        
                        # Calculate Cohen's d
                        n1, n2 = len(group1), len(group2)
                        s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
                        s_pooled = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2))
                        d = (np.mean(group1) - np.mean(group2)) / s_pooled
                        
                        print(f"{trt1} vs {trt2}:")
                        print(f"  Cohen's d: {d:.3f}")
                        print(f"  Mean difference: {np.mean(group1) - np.mean(group2):.3f}%")
                        print(f"  95% CI: [{np.percentile(group1-group2, 2.5):.3f}, {np.percentile(group1-group2, 97.5):.3f}]")



