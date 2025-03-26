from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from matplotlib.dates import MO, DateFormatter, WeekdayLocator, DayLocator
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D  # Add this import at the top


from pathlib import Path
import logging
from datetime import datetime

from scipy import stats

from cowstudyapp.utils import from_posix
from cowstudyapp.config import ConfigManager

class Feature_Plotter:
    """Class to handle showing the distributions of features by hidden state."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the visualizer with configuration.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.figure_size = (10, 6)

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

        print("Calculating step and angle for the data...")
        df = df.copy()
        if self.config.analysis is not None:
            df['mt'] = (df['posix_time']
                        .apply(from_posix)
                        .dt.tz_localize('UTC')  # First localize to UTC
                        .dt.tz_convert(self.config.analysis.timezone))  # Then convert to desired timezone
        else:
            raise ValueError("Analysis Config is None. Unable to recast to the common timezone.")
        
        # Add step length and turning angle
        df["step"] = None
        df["angle"] = None
        
        # Process each cow separately
        for cow_id, cow_data in df.groupby("device_id"):

            print(f"Starting cow {cow_id}...")
            # Get indices for this cow's data
            cow_indices = cow_data.index
            
            # Calculate step sizes and turning angles for all points except the last one
            for i in range(len(cow_indices)-1):
                curr_idx = cow_indices[i]
                next_idx = cow_indices[i+1]

                # Current and next points
                x_c, y_c = df.loc[curr_idx, "utm_easting"], df.loc[curr_idx, "utm_northing"]
                x_n, y_n = df.loc[next_idx, "utm_easting"], df.loc[next_idx, "utm_northing"]
                
                # Calculate step size
                step = np.sqrt((x_n - x_c)**2 + (y_n - y_c)**2)
                df.loc[curr_idx, "step"] = step
                
                # Calculate turning angle (needs three points)
                if i > 0:
                    prev_idx = cow_indices[i-1]
                    # Previous point
                    x_p, y_p = df.loc[prev_idx, "utm_easting"], df.loc[prev_idx, "utm_northing"]
                    
                    # Calculate vectors
                    vector1 = np.array([x_c - x_p, y_c - y_p])
                    vector2 = np.array([x_n - x_c, y_n - y_c])
                    
                    # Calculate angle between vectors
                    dot_product = np.dot(vector1, vector2)
                    norms = np.linalg.norm(vector1) * np.linalg.norm(vector2)
                    
                    # Avoid division by zero and floating point errors
                    cos_angle = np.clip(dot_product / norms if norms != 0 else 0, -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    
                    # Determine sign of angle using cross product
                    cross_product = np.cross([x_c - x_p, y_c - y_p], [x_n - x_c, y_n - y_c])
                    angle = angle if cross_product >= 0 else -angle
                    
                    df.loc[curr_idx, "angle"] = angle
            
            # Set the last row's step and angle to None for this cow
            df.loc[cow_indices[-1], ["step", "angle"]] = None

        df.dropna(axis=0, subset=['activity'], inplace=True)

        self.step_upper = np.percentile(df['step'].dropna(), 99.5)
        df.loc[df['step'] > self.step_upper, 'step'] = np.nan

        self.mag_mean_upper = np.percentile(df['magnitude_mean'].dropna(), 99.5)
        df.loc[df['magnitude_mean'] > self.mag_mean_upper, 'magnitude_mean'] = np.nan
        # df['magnitude_mean'] = df['magnitude_mean'].clip(upper=self.mag_mean_upper)

        self.mag_mean_lower = np.percentile(df['magnitude_mean'].dropna(), 0.5)
        df.loc[df['magnitude_mean'] < self.mag_mean_lower, 'magnitude_mean'] = np.nan
        # df['magnitude_mean'] = df['magnitude_mean'].clip(lower=self.mag_mean_lower)

        self.mag_var_upper = np.percentile(df['magnitude_var'].dropna(), 98)
        df.loc[df['magnitude_mean'] < self.mag_var_upper, 'magnitude_var'] = np.nan
        # df['magnitude_var'] = df['magnitude_var'].clip(upper=self.mag_var_upper)


        return df
        
    
    def generate_plot(self, df: pd.DataFrame, output_dir: Path) -> None:
        """
        Generate and save all visualization plots.
        
        Args:
            df: Input DataFrame
            output_dir: Directory to save the plots
        """

        df = self._prepare_data(df)

        print(df.columns)
        print(df.head())
        
        # Generate different focus views
        # for focus in [None, 'zoomed', 'study']:
        fig, axs = plt.subplots(2,2, figsize=self.figure_size)

        states = ['Grazing', 'Resting', 'Traveling']
        features = {
            "step" : {
                "unit": "Meters",
                "distribution": "lognormal",
                "title": "Step Length",
                "location" : [2.9658136, 2.065888, 4.7846135],
                "scale" : [0.7423612, 1.020416, 0.8717496],
            },
            "magnitude_mean" : {
                "unit": "m/s²",
                "distribution": "lognormal",
                "title": "MeanSVM",
                "location" : [2.1666345, 2.09016918, 2.1162559],
                "scale" :    [0.1100728, 0.07333718, 0.1308942],
            },
            "magnitude_var" : {
                "unit": "(m/s²)²",
                "distribution": "gamma",
                "title": "VarSVM",
                "mean" : [1.0658029, 0.1164454, 0.8657320],
                "sd" :   [0.8112104, 0.1704795, 0.7310639],
            },
            "angle" : {
                "unit": "Radians",
                "distribution": "wrapped_cauchy",
                "title": "Turning Angle",
                "mean" :          [-0.2490977, 2.9681273, -0.6582004],
                "concentration" : [ 0.3884124, 0.3602101,  0.6856821],
            },
        }


        colors = {'Grazing': 'green', 'Resting': 'blue', 'Traveling': 'red'}
        handles, labels = [], []
        # One feature per ax
            # Modify the x-range calculation for better visualization
        for ax, (fname, fdata) in zip(axs.flatten(), features.items()):
            feature_title = f"{fdata['title']}\n({fdata['distribution']})"
            ax.set_title(feature_title)
            
            # Add x-axis label with units
            ax.set_xlabel(f"{fdata['unit']}")
            
            # Create x range for plotting theoretical distributions
            if fdata['distribution'] == 'wrapped_cauchy':
                x = np.linspace(-np.pi, np.pi, 200)
            else:
                # Use percentile-based range instead of max
                # data_max = np.percentile(df[fname].dropna(), 99)
                x = np.linspace(0, df[fname].max(), 200)


            for i, state in enumerate(states):
                state_data = df[df['activity'] == state][fname]
                
                # Plot histogram of actual data
                ax.hist(state_data, bins=30, density=True, alpha=0.3, color=colors[state])
                
                # Plot theoretical distribution
                if fdata['distribution'] == 'lognormal':
                    y = stats.lognorm.pdf(x, s=fdata['scale'][i], 
                                        scale=np.exp(fdata['location'][i]))
                elif fdata['distribution'] == 'gamma':
                    alpha = (fdata['mean'][i] / fdata['sd'][i])**2
                    beta = fdata['mean'][i] / (fdata['sd'][i]**2)
                    y = stats.gamma.pdf(x, a=alpha, scale=1/beta)
                elif fdata['distribution'] == 'wrapped_cauchy':
                    y = stats.vonmises.pdf(x, kappa=fdata['concentration'][i], 
                                         loc=fdata['mean'][i])
                
                ax.plot(x, y, color=colors[state], linestyle='-')
            
            # ax.legend()
            ax.grid(True, alpha=0.3)

            
            
            if fname == 'step':
                ax.set_xlim(0, self.step_upper)
            elif fname == 'magnitude_var':
                ax.set_xlim(0, self.mag_var_upper)
            elif fname == 'magnitude_mean':
                ax.set_xlim(self.mag_mean_lower, self.mag_mean_upper)
            elif fname == 'angle':
                ax.set_xlim(-np.pi, np.pi)

            if len(handles) == 0:
                handles = [
                    mpatches.Patch(color=colors[state], alpha=0.3, label=f'{state} (data)') for state in states
                ] + [
                    Line2D([0], [0], color=colors[state], label=f'{state} (fitted)') for state in states
                ]

        # Option 2: Legend below all subplots
        fig.legend(handles=handles,
                #   bbox_to_anchor=(0.5, -0.05),  # (x, y) where y<0 moves it below plots
                  loc='lower center',
                  title="Behavioral States",
                  ncol=3)  # make it horizontal

        # plt.tight_layout(rect=[0, 0.1, 1, 1])  # leave space at bottom


        plt.show()


        # Save the plot
        # output_path = output_dir / f"features_plot.png"
        # fig.savefig(output_path, bbox_inches='tight', dpi=300)
        # plt.close(fig)
            
        # logging.info(f"Saved plot to: {output_path}")