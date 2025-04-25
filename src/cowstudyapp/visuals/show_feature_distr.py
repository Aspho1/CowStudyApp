from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from matplotlib.dates import MO, DateFormatter, WeekdayLocator, DayLocator
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D  # Add this import at the top


from pathlib import Path
# import logging
# from datetime import datetime

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


        self.features = {
            "step" : {
                "unit": "Meters",
                "distribution": "lognormal",
                "title": "Step Length",
                "location" : [],
                "scale" : [],
                # "location" : [2.9658136, 2.065888, 4.7846135],
                # "scale" : [0.7423612, 1.020416, 0.8717496],
            },
            "magnitude_mean" : {
                "unit": "m/s²",
                "distribution": "lognormal",
                "title": "MeanSVM",
                "location" : [],
                "scale" :    [],
                # "location" : [2.1666345, 2.09016918, 2.1162559],
                # "scale" :    [0.1100728, 0.07333718, 0.1308942],
            },
            "magnitude_var" : {
                "unit": "ln(1 + (m/s²)²)",
                "distribution": "gamma",
                "title": "VarSVM",
                "mean" : [],
                "sd" :   [],
                # "mean" : [1.0658029, 0.1164454, 0.8657320],
                # "sd" :   [0.8112104, 0.1704795, 0.7310639],
            },
            "angle" : {
                "unit": "Radians",
                "distribution": "wrapped_cauchy",
                "title": "Turning Angle",
                "mean" :          [],
                "concentration" : [],
                # "mean" :          [-0.2490977, 2.9681273, -0.6582004],
                # "concentration" : [ 0.3884124, 0.3602101,  0.6856821],
            },
        }


        # self.activity_colors = {
        #     activity: color for activity, color in zip(
        #         self.config.labels.valid_activities,
        #         ['blue', 'green', 'red', 'orange', 'cyan', 'purple', 'brown']  # Add more colors if needed
        #     )
        # }
        
    def _get_model_params(self):
        params_file = Path(self.config.visuals.predictions_path).parent / 'model_parameters.txt'

        with open(params_file, 'r') as f:
            lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].replace(":","").strip() in self.features:
                feature = lines[i].replace(":","").strip()
                for j in range(i+2,i+5):
                    param_data = lines[j].split()
                    if (len(param_data) > 0):
                        if param_data[0] in self.features[feature]:
                            values = [float(p) for p in param_data[1:4]]
                            self.features[feature][param_data[0]] = values
        # print(self.features) 
        print(f"Updated features from {params_file}.")           
            

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare the dataset for visualization.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """

        df['magnitude_var'] = np.log(df['magnitude_var']+1)
        df.dropna(axis=0, subset=['activity'], inplace=True)


        self.step_upper = np.percentile(df['step'].dropna(), 99.5)
        df.loc[df['step'] > self.step_upper, 'step'] = np.nan

        self.mag_mean_upper = np.percentile(df['magnitude_mean'].dropna(), 99.5)
        df.loc[df['magnitude_mean'] > self.mag_mean_upper, 'magnitude_mean'] = np.nan
        # df['magnitude_mean'] = df['magnitude_mean'].clip(upper=self.mag_mean_upper)

        self.mag_mean_lower = np.percentile(df['magnitude_mean'].dropna(), 0.5)
        df.loc[df['magnitude_mean'] < self.mag_mean_lower, 'magnitude_mean'] = np.nan
        # df['magnitude_mean'] = df['magnitude_mean'].clip(lower=self.mag_mean_lower)

        # self.mag_var_upper = np.percentile(df['magnitude_var'].dropna(), 80)
        self.mag_var_upper = np.percentile(df['magnitude_var'].dropna(), 99.5)
        df.loc[df['magnitude_var'] > self.mag_var_upper, 'magnitude_var'] = np.nan
        # df.loc[df['magnitude_var'] < self.mag_var_upper, 'magnitude_var'] = np.nan
        # df['magnitude_var'] = df['magnitude_var'].clip(upper=self.mag_var_upper)

        return df


    def generate_plot(self, df: pd.DataFrame, output_dir: Path) -> None:

        self._get_model_params()
        # return
        df = self._prepare_data(df)
        
        # Create one figure per state, with subplots for each feature
        states = ['Grazing', 'Resting', 'Traveling']
        colors = {'Grazing': 'green', 'Resting': 'blue', 'Traveling': 'red'}



        for state in states:
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"{state} State: Feature Distributions", fontsize=16)
            
            # state_data = df[df['activity'] == state]
            # state_data = df[df['activity'] == state][fname].dropna()
            
            for ax, (fname, fdata) in zip(axs.flatten(), self.features.items()):

                state_data = df[df['activity'] == state][fname].dropna()


                i = states.index(state)
                feature_title = f"{fdata['title']}"
                ax.set_title(feature_title)
                ax.set_xlabel(f"{fdata['unit']}")
                
                # Create appropriate x range
                if fdata['distribution'] == 'wrapped_cauchy':
                    x = np.linspace(-np.pi, np.pi, 200)
                else:
                    x = np.linspace(0, np.percentile(df[fname].dropna(), 99.5), 200)
                
                # Plot histogram with KDE
                # sns.histplot(state_data[fname], bins=30, kde=True, 
                #             color=colors[state], alpha=0.5, ax=ax)


                if len(state_data) > 0:
                    q75, q25 = np.percentile(state_data, [75, 25])
                    iqr = q75 - q25
                    bin_width = 2 * iqr / (len(state_data) ** (1/3))
                    if bin_width > 0:
                        num_bins = max(10, min(50, int(np.ceil((state_data.max() - state_data.min()) / bin_width))))
                    else:
                        num_bins = 20
                else:
                    num_bins = 20
                
                hist, bins, _ = ax.hist(
                    state_data, 
                    bins=num_bins,
                    density=True,  # This is crucial - normalizes to probability density
                    alpha=0.3, 
                    color=colors[state],
                    label=f"{state} data"
                )
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
                
                # Plot fitted line with higher prominence
                ax.plot(x, y, color='black', linestyle='-', linewidth=2.5,
                    label='Fitted distribution')
                
                # Set appropriate limits
                if fname == 'step':
                    ax.set_xlim(0, self.step_upper)
                elif fname == 'magnitude_var':
                    ax.set_xlim(0, self.mag_var_upper)
                elif fname == 'magnitude_mean':
                    ax.set_xlim(self.mag_mean_lower, self.mag_mean_upper)
                elif fname == 'angle':
                    ax.set_xlim(-np.pi, np.pi)
                
                # Add distribution parameters text box
                if fdata['distribution'] == 'lognormal':
                    params_text = f"μ = {fdata['location'][i]:.2f}\nσ = {fdata['scale'][i]:.2f}"
                elif fdata['distribution'] == 'gamma':
                    params_text = f"Mean = {fdata['mean'][i]:.2f}\nSD = {fdata['sd'][i]:.2f}"
                elif fdata['distribution'] == 'wrapped_cauchy':
                    params_text = f"Mean = {fdata['mean'][i]:.2f}\nConc = {fdata['concentration'][i]:.2f}"
                
                ax.text(0.95, 0.95, params_text, transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
                
                ax.legend(['Data', 'Fitted distribution'])
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            
            # Save the plot
            output_path = output_dir / f"features_{state.lower()}.png"
            fig.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close(fig)  
    

    def plot_ecdf_comparison(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Plot ECDF vs theoretical CDF for better comparison"""
        df = self._prepare_data(df)
        states = ['Grazing', 'Resting', 'Traveling']
        colors = {'Grazing': 'green', 'Resting': 'blue', 'Traveling': 'red'}
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        
        for ax_idx, (fname, fdata) in enumerate(self.features.items()):
            ax = axs.flatten()[ax_idx]
            ax.set_title(f"{fdata['title']} - CDF Comparison", fontsize=14)
            ax.set_xlabel(f"{fdata['unit']}", fontsize=12)
            ax.set_ylabel("Cumulative Probability", fontsize=12)
            
            if fdata['distribution'] == 'wrapped_cauchy':
                # Skip circular data for this plot
                ax.text(0.5, 0.5, "CDF not applicable\nfor circular data", 
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
                continue
                
            x = np.linspace(0, np.percentile(df[fname].dropna(), 99), 200)
            
            for i, state in enumerate(states):
                state_data = df[df['activity'] == state][fname].dropna()
                
                # Plot empirical CDF
                sorted_data = np.sort(state_data)
                y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                ax.step(sorted_data, y, where='post', color=colors[state], alpha=0.7,
                    label=f"{state} (empirical)")
                
                # Plot theoretical CDF
                if fdata['distribution'] == 'lognormal':
                    y_theor = stats.lognorm.cdf(x, s=fdata['scale'][i], 
                                            scale=np.exp(fdata['location'][i]))
                elif fdata['distribution'] == 'gamma':
                    alpha = (fdata['mean'][i] / fdata['sd'][i])**2
                    beta = fdata['mean'][i] / (fdata['sd']**2)
                    y_theor = stats.gamma.cdf(x, a=alpha, scale=1/beta)
                    
                ax.plot(x, y_theor, color=colors[state], linestyle='-', linewidth=2.5,
                    label=f"{state} (fitted)")
            
            # Set appropriate axis limits
            if fname == 'step':
                ax.set_xlim(0, self.step_upper)
            elif fname == 'magnitude_var':
                ax.set_xlim(0, self.mag_var_upper)
            elif fname == 'magnitude_mean':
                ax.set_xlim(self.mag_mean_lower, self.mag_mean_upper)
                
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            
            # Add legend to each subplot
            ax.legend(loc='lower right')
        
        plt.tight_layout()
        fig.suptitle("CDF Comparison by Behavioral State", fontsize=16, y=0.98)
        # plt.savefig(output_dir / "features_cdf_comparison.png", bbox_inches='tight', dpi=300)
        plt.show()


    def plot_publication_cdf_comparison(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Create a publication-quality CDF comparison plot for all features"""
        df = self._prepare_data(df)
        states = ['Grazing', 'Resting', 'Traveling']
        colors = {'Grazing': 'forestgreen', 'Resting': 'navy', 'Traveling': 'firebrick'}
        
        # Use publication-quality figure settings
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 11,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 300
        })
        
        # Calculate width for a full-page figure (190mm is standard journal width)
        width_in_inches = 190/25.4  # Convert mm to inches
        height_in_inches = width_in_inches /2.5  # Make it shorter for a row layout
        
        # Create figure with 3 subplots in a row for non-circular distributions
        fig, axs = plt.subplots(3, 1, figsize=(width_in_inches, height_in_inches), layout='constrained')
        
        # Custom linestyles for different states (for grayscale compatibility)
        linestyles = {'Grazing': '-', 'Resting': '--', 'Traveling': '-.'}
        
        # Add panel labels (a, b, c) for publication
        panel_labels = ['a', 'b', 'c']
        
        # Dictionary to store KS statistics and p-values for printing
        ks_results = {}
        
        # Filter to include only non-circular distributions
        non_circular_features = {k: v for k, v in self.features.items() if k != 'angle'}
        
        # Plot the non-circular distributions
        for ax_idx, (fname, fdata) in enumerate(non_circular_features.items()):
            ax = axs[ax_idx]
            
            # Add panel label in upper left corner
            ax.text(0.03, 0.97, panel_labels[ax_idx], transform=ax.transAxes,
                    fontsize=12, fontweight='bold', va='top')
            
            # Set descriptive titles with distribution type
            ax.set_title(f"{fdata['title']} ({fdata['distribution']})")
            ax.set_xlabel(f"{fdata['unit']}")
            if ax_idx == 0:  # Only add y-label to the leftmost plot
                ax.set_ylabel("Cumulative Probability")
            
            # Create x-range appropriate for the data
            x_min = 0
            x_max = np.percentile(df[fname].dropna(), 99)
            x = np.linspace(x_min, x_max, 300)  # More points for smoother curves
            
            # Store KS results for this feature
            ks_results[fname] = {}
            
            # Plot each state
            for i, state in enumerate(states):
                state_data = df[df['activity'] == state][fname].dropna()
                
                # ECDF: Sort data and calculate cumulative probabilities
                sorted_data = np.sort(state_data)
                ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                
                # Plot empirical CDF with thinner lines
                ax.step(sorted_data, ecdf, where='post', 
                    color=colors[state], linestyle=linestyles[state],
                    linewidth=1.2, alpha=0.7, 
                    label=f"{state} observed")
                
                # Calculate theoretical CDF
                if fdata['distribution'] == 'lognormal':
                    theoretical_cdf = stats.lognorm.cdf(x, s=fdata['scale'][i], 
                                                    scale=np.exp(fdata['location'][i]))
                elif fdata['distribution'] == 'gamma':
                    # Fixed: Use the correct index for both mean and sd
                    mean_val = fdata['mean'][i]
                    sd_val = fdata['sd'][i]
                    alpha = (mean_val / sd_val)**2
                    beta = mean_val / (sd_val**2)
                    theoretical_cdf = stats.gamma.cdf(x, a=alpha, scale=1/beta)
                
                # Plot theoretical CDF
                ax.plot(x, theoretical_cdf, color=colors[state], linestyle='-',
                    linewidth=2.0, alpha=0.9, label=f"{state} fitted")
                
                # Calculate and store Kolmogorov-Smirnov statistic for goodness-of-fit
                if len(state_data) > 5:  # Only calculate if we have enough data
                    if fdata['distribution'] == 'lognormal':
                        ks_stat, p_val = stats.kstest(
                            state_data, 
                            lambda x: stats.lognorm.cdf(x, s=fdata['scale'][i], 
                                                    scale=np.exp(fdata['location'][i]))
                        )
                    elif fdata['distribution'] == 'gamma':
                        mean_val = fdata['mean'][i]
                        sd_val = fdata['sd'][i]
                        alpha = (mean_val / sd_val)**2
                        beta = mean_val / (sd_val**2)
                        ks_stat, p_val = stats.kstest(
                            state_data,
                            lambda x: stats.gamma.cdf(x, a=alpha, scale=1/beta)
                        )
                    
                    # Store results
                    ks_results[fname][state] = {'KS': ks_stat, 'p': p_val}
            
            # Set appropriate limits for each feature
            if fname == 'step':
                ax.set_xlim(0, min(self.step_upper, x_max*1.05))
            elif fname == 'magnitude_var':
                ax.set_xlim(0, min(self.mag_var_upper, x_max*1.05))
            elif fname == 'magnitude_mean':
                ax.set_xlim(max(self.mag_mean_lower, 0), 
                        min(self.mag_mean_upper, x_max*1.05))
            
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3, linestyle=':')
        
        # Add a single legend for all subplots
        handles = []
        for state in states:
            handles.append(Line2D([0], [0], color=colors[state], linestyle='-',
                                linewidth=2, label=f"{state} fitted"))
            handles.append(Line2D([0], [0], color=colors[state], linestyle=linestyles[state],
                                linewidth=1.2, alpha=0.7, label=f"{state} observed"))
        
        # Place legend below the subplots
        fig.legend(handles=handles, loc='lower center', 
                bbox_to_anchor=(0.5, 0), ncol=3)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.35)  # Make room for the legend
        
        # Save the non-circular plot
        # output_path = output_dir / "cdf_comparison_non_circular.pdf"
        # fig.savefig(output_path, dpi=300, bbox_inches='tight')
        # output_path_png = output_dir / "cdf_comparison_non_circular.png"
        # fig.savefig(output_path_png, dpi=300, bbox_inches='tight')
        

        plt.show()


        # Now create a separate figure for the circular data
        circular_fig, circular_ax = plt.subplots(figsize=(width_in_inches/3, width_in_inches/3), 
                                                subplot_kw={'projection': 'polar'})
        
        # Find the angle data
        angle_data = self.features['angle']
        
        # Plot circular histogram for each state
        for i, state in enumerate(states):
            state_data = df[df['activity'] == state]['angle'].dropna()
            
            if len(state_data) > 0:
                # Plot histogram
                bins = np.linspace(-np.pi, np.pi, 36)
                hist, bin_edges = np.histogram(state_data, bins=bins, density=True)
                width = 2*np.pi / len(bins)
                bin_centers = bin_edges[:-1] + width/2
                
                # Plot as bars in polar coordinates
                bars = circular_ax.bar(bin_centers, hist, width=width, bottom=0.0,
                                alpha=0.3, color=colors[state], 
                                label=f"{state} observed")
                
                # Generate fitted distribution curve
                theta = np.linspace(-np.pi, np.pi, 200)
                radii = stats.vonmises.pdf(theta, 
                                        kappa=angle_data['concentration'][i],
                                        loc=angle_data['mean'][i])
                
                # Scale to match histogram
                scale_factor = np.max(hist) / np.max(radii) if np.max(hist) > 0 else 1
                radii = radii * scale_factor
                
                # Plot fitted curve
                circular_ax.plot(theta, radii, color=colors[state], linewidth=2,
                        label=f"{state} fitted")
                
                # Store parameter values for printing
                ks_results['angle'] = {
                    state: {
                        'mean': angle_data['mean'][i],
                        'concentration': angle_data['concentration'][i]
                    } for i, state in enumerate(states)
                }
        
        # Configure polar plot
        circular_ax.set_title(f"Turning Angle (wrapped_cauchy)")
        circular_ax.set_theta_zero_location('N')  # 0 at the top
        circular_ax.set_theta_direction(-1)  # Clockwise
        circular_ax.set_rlabel_position(0)  # Move radial labels away from chart
        circular_ax.grid(True, alpha=0.3, linestyle=':')
        
        # Add a legend for the circular plot
        circular_handles = []
        for state in states:
            circular_handles.append(Line2D([0], [0], color=colors[state], linestyle='-',
                                        linewidth=2, label=f"{state}"))
        circular_fig.legend(handles=circular_handles, loc='lower center', 
                        bbox_to_anchor=(0.5, 0.05), ncol=3)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        
        # Save the circular plot
        # circular_output_path = output_dir / "turning_angle_distribution.pdf"
        # circular_fig.savefig(circular_output_path, dpi=300, bbox_inches='tight')
        # circular_output_path_png = output_dir / "turning_angle_distribution.png"
        # circular_fig.savefig(circular_output_path_png, dpi=300, bbox_inches='tight')
        
        # Show both plots
        plt.show()
        
        # Print the KS statistics for easy copying to the paper
        print("\n=== Kolmogorov-Smirnov Test Results ===")
        for feature, state_results in ks_results.items():
            if feature != 'angle':
                print(f"\n{self.features[feature]['title']} ({self.features[feature]['distribution']})")
                for state, results in state_results.items():
                    print(f"{state}: KS={results['KS']:.3f}, p={results['p']:.3f}")
            else:
                print(f"\nTurning Angle Parameters (wrapped_cauchy)")
                for state, results in state_results.items():
                    print(f"{state}: mean={results['mean']:.3f}, concentration={results['concentration']:.3f}")
        
        # print(f"\nNon-circular distributions saved to {output_path_png}")
        # print(f"Circular distribution saved to {circular_output_path_png}")


    def plot_publication_cdf_comparison_all_4(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Create a publication-quality CDF comparison plot for all features including angle in Cartesian coordinates"""
        self._get_model_params()
        # return
        
        df = self._prepare_data(df)
        states = ['Grazing', 'Resting', 'Traveling']
        colors = {'Grazing': 'forestgreen', 'Resting': 'navy', 'Traveling': 'firebrick'}
        
        # Use publication-quality figure settings
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 11,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 300
        })
        
        # Calculate width for a full-page figure (190mm is standard journal width)
        width_in_inches = 190/25.4  # Convert mm to inches
        height_in_inches = width_in_inches * (4/5)  # Make it shorter for a row layout
        
        # Create figure with 2x2 subplots layout
        fig, axs = plt.subplots(2, 2, figsize=(width_in_inches, height_in_inches), layout='constrained')
        
        # Custom linestyles for different states (for grayscale compatibility)
        linestyles = {'Grazing': '-', 'Resting': '--', 'Traveling': '-.'}
        
        # Add panel labels (a, b, c, d) for publication
        panel_labels = ['a', 'b', 'c', 'd']
        
        # Dictionary to store KS statistics and p-values for printing
        ks_results = {}
        
        # Process each feature
        for ax_idx, (fname, fdata) in enumerate(self.features.items()):
            ax = axs.flatten()[ax_idx]
            
            # Add panel label in upper left corner
            ax.text(0.03, 0.97, panel_labels[ax_idx], transform=ax.transAxes,
                    fontsize=12, fontweight='bold', va='top')
            
            # Set descriptive titles with distribution type
            ax.set_title(f"{fdata['title']} ({fdata['distribution']})")
            ax.set_xlabel(f"{fdata['unit']}")
            

            if fname == 'angle':
                ax.set_ylabel("Probability Density")
            else:
                ax.set_ylabel("Cumulative Probability")
            
            # For the angle feature 
            if fname == 'angle':
                # Create x-range for angle with more points to ensure smoothness
                x = np.linspace(-np.pi, np.pi, 500)
                
                # Store results for angle
                ks_results[fname] = {}
                
                # Plot each state
                for i, state in enumerate(states):
                    state_data = df[df['activity'] == state][fname].dropna()
                    
                    # Create histogram for the data with proper normalization
                    hist, bins = np.histogram(state_data, bins=36, range=(-np.pi, np.pi), density=True)
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    bin_width = bins[1] - bins[0]
                    
                    # Plot histogram as bars
                    ax.bar(bin_centers, hist, width=bin_width, alpha=0.3, 
                        color=colors[state], label=f"{state} observed")
                    
                    # Implement a proper wrapped Cauchy density function manually
                    # Formula: f(x) = (1-c^2) / (2*pi*(1+c^2-2*c*cos(x-loc)))
                    c: float = fdata['concentration'][i]
                    loc: float = fdata['mean'][i]
                    
                    # Normalize x to be relative to the location parameter
                    x_shifted = x - loc
                    
                    # Ensure x_shifted is in [-np.pi, np.pi]
                    x_shifted = np.mod(x_shifted + np.pi, 2*np.pi) - np.pi
                    
                    # Calculate wrapped Cauchy density
                    numerator = 1 - c**2
                    denominator = 2 * np.pi * (1 + c**2 - 2*c*np.cos(x_shifted))
                    y_fitted = numerator / denominator
                    

                    # ################### Can we do a CDF for wrapped cauchy? if it is unreasonable to do and, uncommon to do in circular statistics, let me know.
                    # def WC_CDF(p, x, c):
                    #     numerator = ((p**2 - 1) * np.atanh(((np.pi * p**2 + p + np.pi) * np.tan(x/2)) / np.sqrt(-np.pi**2 * p**4 + (1 - 2 * np.pi**2) * p**2 - np.pi**2)))
                    #     denominator = np.sqrt(-np.pi**2 * p**4 + (1 - 2 * np.pi**2) * p**2 - np.pi**2)
                    #     return (numerator/denominator) + c
                    
                    # Plot the PDF without scaling
                    ax.plot(x, y_fitted, color=colors[state], linestyle='-',
                            linewidth=2.0, label=f"{state} fitted")
                    
                    # Store parameters for reporting
                    ks_results[fname][state] = {
                        'mean': fdata['mean'][i],
                        'concentration': fdata['concentration'][i]
                    }
                
                # Set angle-specific limits and labels
                ax.set_xlim(-np.pi, np.pi)
                ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
                ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
            else:
                # Handle non-circular data (CDFs)
                # Create x-range appropriate for the data
                x_min = 0
                x_max = np.percentile(df[fname].dropna(), 99)
                x = np.linspace(x_min, x_max, 300)
                
                # Store KS results for this feature
                ks_results[fname] = {}
                
                # Plot each state
                for i, state in enumerate(states):
                    state_data = df[df['activity'] == state][fname].dropna()
                    
                    # ECDF: Sort data and calculate cumulative probabilities
                    sorted_data = np.sort(state_data)
                    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                    
                    # Plot empirical CDF with thinner lines
                    ax.step(sorted_data, ecdf, where='post', 
                        color=colors[state], linestyle=linestyles[state],
                        linewidth=1.2, alpha=0.7, 
                        label=f"{state} observed")
                    
                    # Calculate theoretical CDF
                    if fdata['distribution'] == 'lognormal':
                        theoretical_cdf = stats.lognorm.cdf(x, s=fdata['scale'][i], 
                                                        scale=np.exp(fdata['location'][i]))
                    elif fdata['distribution'] == 'gamma':
                        mean_val = fdata['mean'][i]
                        sd_val = fdata['sd'][i]
                        alpha = (mean_val / sd_val)**2
                        beta = mean_val / (sd_val**2)
                        theoretical_cdf = stats.gamma.cdf(x, a=alpha, scale=1/beta)
                    
                    # Plot theoretical CDF
                    ax.plot(x, theoretical_cdf, color=colors[state], linestyle='-',
                        linewidth=2.0, alpha=0.9, label=f"{state} fitted")
                    
                    # Calculate and store Kolmogorov-Smirnov statistic for goodness-of-fit
                    if len(state_data) > 5:  # Only calculate if we have enough data
                        if fdata['distribution'] == 'lognormal':
                            ks_stat, p_val = stats.kstest(
                                state_data, 
                                lambda x: stats.lognorm.cdf(x, s=fdata['scale'][i], 
                                                        scale=np.exp(fdata['location'][i]))
                            )
                        elif fdata['distribution'] == 'gamma':
                            mean_val = fdata['mean'][i]
                            sd_val = fdata['sd'][i]
                            alpha = (mean_val / sd_val)**2
                            beta = mean_val / (sd_val**2)
                            ks_stat, p_val = stats.kstest(
                                state_data,
                                lambda x: stats.gamma.cdf(x, a=alpha, scale=1/beta)
                            )
                        
                        # Store results
                        ks_results[fname][state] = {'KS': ks_stat, 'p': p_val}
                
                # Set appropriate limits for each feature
                if fname == 'step':
                    ax.set_xlim(0, min(self.step_upper, x_max*1.05))
                elif fname == 'magnitude_var':
                    ax.set_xlim(0, min(self.mag_var_upper, x_max*1.05))
                elif fname == 'magnitude_mean':
                    ax.set_xlim(max(self.mag_mean_lower, 0), 
                            min(self.mag_mean_upper, x_max*1.05))
                
                ax.set_ylim(0, 1.05)
            
            # Add grid to all plots
            ax.grid(True, alpha=0.3, linestyle=':')
        
        # Add a single legend for all subplots
        handles = []
        for state in states:
            handles.append(Line2D([0], [0], color=colors[state], linestyle='-',
                                linewidth=2, label=f"{state} fitted"))
            handles.append(Line2D([0], [0], color=colors[state], linestyle=linestyles[state] if fname != 'angle' else '-',
                                linewidth=1.2, alpha=0.7, label=f"{state} observed"))
        
        # Place legend below the subplots
        fig.legend(handles=handles, loc='lower center', 
                bbox_to_anchor=(0.5, 0), ncol=3)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.18, hspace=.4)  # Make room for the legend
        
        # # Save the plot if needed
        # output_path = output_dir / "feature_distributions.pdf"
        # fig.savefig(output_path, dpi=300, bbox_inches='tight')
        output_path_png = output_dir / "feature_distributions.png"
        fig.savefig(output_path_png, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print the statistics for reporting
        print("\n=== Distribution Parameters and Test Results ===")
        for feature, state_results in ks_results.items():
            if feature != 'angle':
                print(f"\n{self.features[feature]['title']} ({self.features[feature]['distribution']})")
                for state, results in state_results.items():
                    print(f"{state}: KS={results['KS']:.3f}, p={results['p']:.3f}")
            else:
                print(f"\nTurning Angle Parameters (wrapped_cauchy)")
                for state, results in state_results.items():
                    print(f"{state}: mean={results['mean']:.3f}, concentration={results['concentration']:.3f}")
        
        # print(f"\nPlots saved to {output_path_png}")





















    ############## Old dont delete but ignore

    # def generate_plot(self, df: pd.DataFrame, output_dir: Path) -> None:
    #     """
    #     Generate and save all visualization plots.
        
    #     Args:
    #         df: Input DataFrame
    #         output_dir: Directory to save the plots
    #     """

    #     df = self._prepare_data(df)

    #     print(df.columns)
    #     print(df.head())
        
    #     # Generate different focus views
    #     # for focus in [None, 'zoomed', 'study']:
    #     fig, axs = plt.subplots(2,2, figsize=self.figure_size)

    #     states = ['Grazing', 'Resting', 'Traveling']
    #     features = {
    #         "step" : {
    #             "unit": "Meters",
    #             "distribution": "lognormal",
    #             "title": "Step Length",
    #             "location" : [2.9658136, 2.065888, 4.7846135],
    #             "scale" : [0.7423612, 1.020416, 0.8717496],
    #         },
    #         "magnitude_mean" : {
    #             "unit": "m/s²",
    #             "distribution": "lognormal",
    #             "title": "MeanSVM",
    #             "location" : [2.1666345, 2.09016918, 2.1162559],
    #             "scale" :    [0.1100728, 0.07333718, 0.1308942],
    #         },
    #         "magnitude_var" : {
    #             "unit": "(m/s²)²",
    #             "distribution": "gamma",
    #             "title": "VarSVM",
    #             "mean" : [1.0658029, 0.1164454, 0.8657320],
    #             "sd" :   [0.8112104, 0.1704795, 0.7310639],
    #         },
    #         "angle" : {
    #             "unit": "Radians",
    #             "distribution": "wrapped_cauchy",
    #             "title": "Turning Angle",
    #             "mean" :          [-0.2490977, 2.9681273, -0.6582004],
    #             "concentration" : [ 0.3884124, 0.3602101,  0.6856821],
    #         },
    #     }


    #     colors = {'Grazing': 'green', 'Resting': 'blue', 'Traveling': 'red'}
    #     handles, labels = [], []
    #     # One feature per ax
    #         # Modify the x-range calculation for better visualization
    #     for ax, (fname, fdata) in zip(axs.flatten(), features.items()):
    #         feature_title = f"{fdata['title']}\n({fdata['distribution']})"
    #         ax.set_title(feature_title)
            
    #         # Add x-axis label with units
    #         ax.set_xlabel(f"{fdata['unit']}")
            
    #         # Create x range for plotting theoretical distributions
    #         if fdata['distribution'] == 'wrapped_cauchy':
    #             x = np.linspace(-np.pi, np.pi, 200)
    #         else:
    #             # Use percentile-based range instead of max
    #             # data_max = np.percentile(df[fname].dropna(), 99)
    #             x = np.linspace(0, df[fname].max(), 200)


    #         for i, state in enumerate(states):
    #             state_data = df[df['activity'] == state][fname]
                
    #             # Plot histogram of actual data
    #             ax.hist(state_data, bins=30, density=True, alpha=0.3, color=colors[state])
                
    #             # Plot theoretical distribution
    #             if fdata['distribution'] == 'lognormal':
    #                 y = stats.lognorm.pdf(x, s=fdata['scale'][i], 
    #                                     scale=np.exp(fdata['location'][i]))
    #             elif fdata['distribution'] == 'gamma':
    #                 alpha = (fdata['mean'][i] / fdata['sd'][i])**2
    #                 beta = fdata['mean'][i] / (fdata['sd'][i]**2)
    #                 y = stats.gamma.pdf(x, a=alpha, scale=1/beta)
    #             elif fdata['distribution'] == 'wrapped_cauchy':
    #                 y = stats.vonmises.pdf(x, kappa=fdata['concentration'][i], 
    #                                      loc=fdata['mean'][i])
                
    #             ax.plot(x, y, color=colors[state], linestyle='-')
            
    #         # ax.legend()
    #         ax.grid(True, alpha=0.3)

            
            
    #         if fname == 'step':
    #             ax.set_xlim(0, self.step_upper)
    #         elif fname == 'magnitude_var':
    #             ax.set_xlim(0, self.mag_var_upper)
    #             ax.set
    #         elif fname == 'magnitude_mean':
    #             ax.set_xlim(self.mag_mean_lower, self.mag_mean_upper)
    #         elif fname == 'angle':
    #             ax.set_xlim(-np.pi, np.pi)

    #         if len(handles) == 0:
    #             handles = [
    #                 mpatches.Patch(color=colors[state], alpha=0.3, label=f'{state} (data)') for state in states
    #             ] + [
    #                 Line2D([0], [0], color=colors[state], label=f'{state} (fitted)') for state in states
    #             ]

    #     # Option 2: Legend below all subplots
    #     fig.legend(handles=handles,
    #             #   bbox_to_anchor=(0.5, -0.05),  # (x, y) where y<0 moves it below plots
    #               loc='lower center',
    #               title="Behavioral States",
    #               ncol=3)  # make it horizontal

    #     # plt.tight_layout(rect=[0, 0.1, 1, 1])  # leave space at bottom


    #     plt.show()


    #     # Save the plot
    #     # output_path = output_dir / f"features_plot.png"
    #     # fig.savefig(output_path, bbox_inches='tight', dpi=300)
    #     # plt.close(fig)
            
    #     # logging.info(f"Saved plot to: {output_path}")