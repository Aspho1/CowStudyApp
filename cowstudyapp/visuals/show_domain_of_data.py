from typing import List, Optional
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.dates import MO, DateFormatter, WeekdayLocator, DayLocator, HourLocator
import matplotlib.patches as mpatches

from pathlib import Path
import logging
from datetime import datetime

from cowstudyapp.utils import from_posix_col
from cowstudyapp.config import ConfigManager

class DataDomainVisualizer:
    """Class to handle the visualization of data domains."""
    
    def __init__(self, config: ConfigManager, focus: Optional[str] = None):
        """
        Initialize the visualizer with configuration.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.figure_size = (10, 6)
        # self.focus = self.config.visuals.domain.focus

        self.activity_colors = {
            activity: color for activity, color in zip(
                self.config.labels.valid_activities,
                ['blue', 'green', 'red', 'orange', 'cyan', 'purple', 'brown']  # Add more colors if needed
            )
        }

        if self.config.visuals.domain.labeled_only: 
            self.target_col = "activity" 
            self.focus="zoomed"
        else:
            self.target_col = "predicted_state"
            self.focus=None
            
        
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare the dataset for visualization.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        df['mt'] = from_posix_col(df['posix_time'], self.config.analysis.timezone)
        return df
    
    
    def _create_time_domain_plot(self, df: pd.DataFrame) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(layout="constrained", figsize=self.figure_size)
        ylabels: List[str] = []

        # After creating the figure but before plotting:
        plt.rcParams['timezone'] = self.config.analysis.timezone
        df, start, end = self._get_activity_time_range(df)
        
        # Create legend handles and labels lists
        legend_handles = []
        legend_labels = []
        
        if self.focus != 'zoomed':
            # Add 'No Activity' to legend once
            no_activity_handle = plt.scatter([], [], 
                                        color='lightgray',
                                        alpha=0.3,
                                        s=20)
            legend_handles.append(no_activity_handle)
            legend_labels.append('No Activity')
        
        print(df.columns)
        
        # Plot data points
        for IDx, (ID, device_data) in enumerate(df.groupby("ID")):
            ylabels.append(ID)
            
            # Plot points without activity (gray)
            no_activity_mask = device_data[self.target_col].isna()
            
            ax.scatter(x=device_data[no_activity_mask].mt, 
                    y=[IDx] * no_activity_mask.sum(),
                    color='lightgray',
                    alpha=0.3,
                    s=20)
            
            # Plot points with activity (colored by activity)
            for activity in device_data[self.target_col].dropna().unique():
                if activity in self.activity_colors:
                    activity_mask = device_data[self.target_col] == activity
                    ax.scatter(x=device_data[activity_mask].mt, 
                            y=[IDx] * activity_mask.sum(),
                            color=self.activity_colors[activity],
                            alpha=0.6,
                            s=30)
                    
                    # Add to legend if not already there
                    if activity not in legend_labels:
                        handle = plt.scatter([], [], 
                                        color=self.activity_colors[activity],
                                        alpha=0.6,
                                        s=30)
                        legend_handles.append(handle)
                        legend_labels.append(activity)

        # Add legend with all activities
        ax.legend(legend_handles, legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Rest of the function remains the same
        self._configure_time_axis(ax, start, end)
        self._configure_plot_styling(ax, ylabels, fig)
        
        return fig, ax



    def _get_activity_time_range(self, df: pd.DataFrame) -> tuple[pd.DataFrame, datetime, datetime]:
        # if self.config.visuals.domain.labeled_only:
            # df = df[df[self.target_col].notna()]

        if self.focus == 'zoomed':
            df = df[df[self.target_col].isin(self.config.analysis.hmm.states)]
            # activity_data = df[df[self.target_col].notna()]
            if df.empty:
                raise ValueError(f"Unable to use `focus={self.focus}` as there is no activity data.")
            
            # start_time = df['mt'].min() - pd.Timedelta(hours=2)
            # end_time = df['mt'].max() + pd.Timedelta(hours=2)
            # return df, start_time, end_time
        
        start_time = df['mt'].min() - pd.Timedelta(hours=2)
        end_time = df['mt'].max() + pd.Timedelta(hours=2)

        return df, start_time, end_time
    

    def _configure_time_axis(self, ax: plt.Axes, start:datetime, end:datetime) -> None:
        """Configure the time axis based on focus mode."""
        if self.focus == 'zoomed':
            ax.xaxis.set_major_locator(HourLocator(interval=12))
        else:
            ax.xaxis.set_major_locator(WeekdayLocator(byweekday=MO, interval=1))

        ax.set_xlim(start, end)

        if self.focus == 'zoomed':
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M:%S'))

        else:    
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))

        # Rotate labels
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')
    

    def _configure_plot_styling(
        self, 
        ax: plt.Axes, 
        ylabels: List[str], 
        fig: plt.Figure
    ) -> None:
        """Configure the plot styling and labels."""
        plt.subplots_adjust(right=0.8)
        ax.set_yticks(range(len(ylabels)), ylabels)
        ax.set_ylabel("Collar ID")
        ax.set_xlabel("Datetime")
        fig.suptitle("Observations by Collar ID", fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def generate_plot(self, df: pd.DataFrame) -> None:
        """
        Generate and save all visualization plots.
        
        Args:
            df: Input DataFrame
            output_dir: Directory to save the plots
        """

        focus=self.focus

        df = self._prepare_data(df)
        
        # Generate different focus views
        # for focus in [None, 'zoomed', 'study']:
        fig, ax = self._create_time_domain_plot(df)
        

        if focus == "zoomed":
            focus_str = "zoomed"
        elif focus is None:
            focus_str = "regular"
        else:
            raise ValueError(f"Unknown `focus={self.focus}`.")
            
        output_path = self.config.visuals.visuals_root_path 

        if self.config.visuals.domain.extension is not None:
            output_path = output_path / self.config.visuals.domain.extension
            
        output_path = output_path / f"time_domain_{focus_str}.png"
            
        # Save the plot
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
            
        logging.info(f"Saved plot to: {output_path}")