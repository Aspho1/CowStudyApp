from typing import List, Optional
import numpy as np
import pandas as pd

from matplotlib.dates import MO, DateFormatter, WeekdayLocator, DayLocator
import matplotlib.patches as mpatches
import seaborn as sns

from pathlib import Path
from matplotlib import colors as mcolors, pyplot as plt, cm

from matplotlib.dates import MO, DateFormatter, DayLocator, WeekdayLocator
import matplotlib.patches as mpatches

from datetime import datetime, timedelta

from cowstudyapp.utils import from_posix
from cowstudyapp.config import ConfigManager


class RadarPlotOfCow:
    '''
    For the original HMM Predictions    
    '''

    def __init__(self, config: ConfigManager):
        super().__init__()

        self.config = config
        self.figure_size = (10, 6)
        self.dest = self.config.visuals.visuals_root_path / self.config.visuals.radar.extension


        self.act_map = {
            "Grazing" : "green",
            "Resting" : "lightblue",
            "Traveling" : "red",
        }
        if not self.config.visuals.radar.show_night:
            self.act_map["NIGHTTIME"] = 'darkgray'

        
        self.maxdays = (self.config.validation.end_datetime - self.config.validation.start_datetime).days

        # Generate 24 tick positions (one for each hour)
        self.theta_ticks = np.linspace(num=24, start=0, stop=2 * np.pi, endpoint=False)
        self.time_labels = [(f"{i:02d}:00") for i in range(len(self.theta_ticks))]


        # Generate labels for each hour, starting from 21:00 at 0 radians and going counterclockwise
        # self.time_labels = [(f"{(21 + i) % 24:02d}:00") for i in range(24)]

        # self.time_labels = [(f"{int(i*(24/(2*np.pi))):02d}:00") for i in self.theta_ticks]
        self.time_labels = [(f"{i:02d}:00") for i in range(len(self.theta_ticks))]
        
        # print("theta ticks", self.theta_ticks)
        # print("theta labels", self.time_labels)


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
        
        df['mt'] = (df['posix_time']
                    .apply(from_posix)
                    .dt.tz_localize(self.config.analysis.timezone))  # Directly interpret as Denver time

        df["r_date"] = df["mt"].dt.date

        df["r_time"] = df.mt.dt.hour + (df.mt.dt.minute / 60)

        # return
        # Calculate angles in radians
        df["angles"] = df["r_time"] * (2 * np.pi / 24)

        df["days_after_start"] = df["r_date"].apply(lambda x: (x - self.config.validation.start_datetime.date()).days + 1)
        df['activity_color'] = df['predicted_state'].apply(lambda activity: self.act_map[activity])

        return df
    

    def make_cow_gallery_images(self, df:pd.DataFrame):

        df = self._prepare_data(df)

        for ID, df in df.groupby("ID"):
            self.make_radar_single_cow(ID=ID, df=df, end_format='jpeg')
            print(f"ID {ID} has finished.")
            # return
            # return



    def make_radar_single_cow(self, ID = 824, df=None, end_format='jpeg',show=False) -> None:
        fig,ax = plt.subplots(  figsize=(12,12)
                               , subplot_kw={'projection': 'polar'})
        
        self.plot_single_cow_radar(ID=ID,ax=ax,df=df)
        self.end_plot(fig=fig, end_format=end_format, ID=ID)

        
    def end_plot(self, fig, end_format = None, ID = None, show=False):
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markersize=8, markerfacecolor=color) 
                for label, color in self.act_map.items()]
        if show:
            plt.show()
        
        fig.legend(handles=handles, loc="lower right", title="Activity")
        plt.subplots_adjust(
            left=.05,
            right=.95,
            top=0.93,
            bottom=0.05,
            wspace=0,
            hspace=.9
        )
        if ID is None:
            # print(ID)
            raise ValueError(f"The ID MUST be passed to end_plot.")

        if end_format is None:
            plt.show()

        elif end_format == "svg":
            plt.savefig(f"{self.dest}/radar_{ID}.svg", format="svg")
        
        elif end_format == "jpeg":
            plt.savefig(f"{self.dest}/radar_{ID}.jpg", format="jpeg", dpi=200)
        
        else:
            raise NotImplementedError(f"The output format of `{end_format}` is not yet supported")


        plt.close()



    def plot_single_cow_radar(self, ID:int, ax:plt.Axes, df:pd.DataFrame):
        ax.set_title(f"ID: {ID}", fontweight="bold", pad=25, fontsize=12)
        ax.set_yticklabels([])
        ax.set_xticks(self.theta_ticks, self.time_labels)

        ax.set_ylim(0, self.maxdays)
        
        # Calculate point sizes that increase with radius
        # Map days_after_start to a size range of 0.5 to 3
        min_s = 0.5
        max_s = 8

        sizes = df["days_after_start"].apply(lambda x: min_s + (x/self.maxdays) * (max_s-min_s))
        
        # Plot using pre-calculated angles and days with variable sizes
        ax.scatter(df["angles"], 
                df["days_after_start"],
                c=df["activity_color"], 
                s=sizes, 
                alpha=0.7)

        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2)
        ax.grid(True)
        ax.xaxis.grid(True, linestyle=':', color='gray', alpha=0.7)
        ax.yaxis.grid(False)