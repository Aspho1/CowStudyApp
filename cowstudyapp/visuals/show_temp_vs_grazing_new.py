import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from matplotlib.colors import LinearSegmentedColormap
# from datetime import datetime
# from scipy import ndimage


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV

# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.signal import find_peaks

from cowstudyapp.utils import from_posix_col
from cowstudyapp.config import ConfigManager



class ActivityVersusTemperatureNew:
    def __init__(self, config: ConfigManager, fit_method:str='polynomial', degree:int=4, state:str='Grazing'):
        """init the new State versus temperature class.
        This class creates a 3D representation of state, day, and time. 
        A model can be fitted to the class. 

        
        config: ConfigManager
            The ConfigManager object which holds some parameters for 
            this analysis
        fit_method : str
            'polynomial' or 'gpr' (Gaussian Process Regression)
        degree : int
            Degree of polynomial (only used if method='polynomial')
        Returns:
        --------
        None
        """
        self.config = config
        self.fit_method=fit_method
        self.degree=degree
        self.state=state
        self.scale_factors = {
            'minute': 1440,  # minutes in a day
            'day': None,     # will be set based on data
            'temp': 20       # approximate temperature range
        }


    def _add_cols(self, df):
        """Add necessary columns to the dataframe"""
        df = df.copy()
        
        # Add datetime columns
        df['mt'] = from_posix_col(df['posix_time'], self.config.analysis.timezone)
        df['date'] = df['mt'].dt.date

        if self.config.visuals.heatmap.filter_weigh_days:
            weigh_days = pd.to_datetime(self.config.visuals.heatmap.weigh_days)
            if weigh_days is None:
                raise ValueError("Weigh Days must be defined in the config.")
            
            df = df[~df["date"].isin([wd.date() for wd in weigh_days])]


        df['minute'] = df['mt'].dt.hour * 60 + df['mt'].dt.minute
        
        # Add temperature column
        if 'temperature' not in df.columns:
            if 'temperature_gps' in df.columns:
                df.rename(columns={'temperature_gps': 'temperature'}, inplace=True)
            else:
                temp_df = pd.read_csv(self.config.analysis.target_dataset)
                required_cols = ['device_id', 'posix_time', 'temperature_gps']
                
                for rc in required_cols:
                    print(rc, ": ", rc in temp_df.columns)

                if not all(col in temp_df.columns for col in required_cols):


                    print(col in temp_df.columns for col in required_cols)
                    raise ValueError(f"Temperature dataset must contain columns: {required_cols}")
                
                df = pd.merge(
                    df,
                    temp_df[['device_id', 'posix_time', 'temperature_gps']],
                    left_on=['ID', 'posix_time'],
                    right_on=['device_id', 'posix_time'],
                    how='inner'
                ).drop('device_id', axis=1).rename(columns={'temperature_gps': 'temperature'})
        
        return df

    def _create_pivot_tables(self, df):
        """Create pivot tables for state proportions and temperature"""
        # Convert dates to numeric values (days from start)
        dates = sorted(df['date'].unique())
        date_mapping = {date: i for i, date in enumerate(dates)}
        df['day_num'] = df['date'].map(date_mapping)
        
        # print(df.describe())

        # Create pivot tables
        pivot_state = df.pivot_table(
            index='day_num',
            columns='minute',
            values='predicted_state',
            aggfunc=lambda x: (x == self.state).mean()
        ).ffill().bfill()

        pivot_temp = df.pivot_table(
            index='day_num',
            columns='minute',
            values='temperature',
            aggfunc='mean'
        ).bfill().ffill()
        #.ffill().bfill()
        
        return pivot_state, pivot_temp, dates

    def _smooth_surfaces(self, Z_state, Z_temp, sigma=2):
        """Apply Gaussian smoothing to surfaces"""
        import scipy.ndimage as ndimage
        Z_smooth = ndimage.gaussian_filter(Z_state, sigma=2)
        Z_temp_smooth = ndimage.gaussian_filter(Z_temp, sigma=1)
        return Z_smooth, Z_temp_smooth

    def _create_meshgrid(self, pivot_state, n_points=288):
        """Create meshgrid for surface plot"""
        X, Y = np.meshgrid(
            np.linspace(min(pivot_state.columns), max(pivot_state.columns), n_points),
            np.linspace(min(pivot_state.index), max(pivot_state.index), pivot_state.shape[0])
        )
        return X, Y

    def _setup_3d_plot(self, figsize=(12, 10)):
        """Set up 3D plot figure and axis"""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        return fig, ax

    def _format_axis_labels(self, ax, dates, fontsize=12):
        """Format axis labels and ticks"""
        # Set minute ticks (2-hour intervals)
        hour_minutes = list(range(0, 1441, 120))
        hour_labels = [f"{h//60:02d}:{h%60:02d}" for h in hour_minutes]
        ax.set_xticks(hour_minutes)
        ax.set_xticklabels(hour_labels, rotation=45, ha='right')
        
        # Set day ticks
        day_step = max(1, len(dates) // 8)
        day_ticks = list(range(0, len(dates), day_step))
        day_labels = [dates[i].strftime('%m/%d') for i in day_ticks]
        ax.set_yticks(day_ticks)
        ax.set_yticklabels(day_labels)
        
        # Set labels
        ax.set_xlabel('Time of Day', fontsize=fontsize, labelpad=20)
        ax.set_ylabel('Date', fontsize=fontsize, labelpad=20)
        ax.set_zlabel('Proportion', fontsize=fontsize, labelpad=20)

    def plot_3d_surface(self, df, ID=None):
        """Create 3D surface plot with temperature coloring"""
        # Process data
        df = self._add_cols(df)
        
        # Filter by ID if specified
        if ID is not None:
            df = df[df.ID == ID].copy()
            print(f"Analyzing cow {ID}: {len(df)} observations")
        else:
            print(f"Analyzing {len(df.ID.unique())} cows: {len(df)} observations")
        
        # Create pivot tables
        pivot_state, pivot_temp, dates = self._create_pivot_tables(df)
        
        # Print some diagnostics
        print(f"\nState proportions summary:")
        print(f"Mean: {pivot_state.values.mean():.3f}")
        print(f"Max: {pivot_state.values.max():.3f}")
        print(f"Non-zero values: {(pivot_state.values > 0).sum()}")
        
        # Smooth surfaces
        Z_smooth, Z_temp_smooth = self._smooth_surfaces(pivot_state.values, pivot_temp.values)
        
        # Create meshgrid
        X, Y = self._create_meshgrid(pivot_state)
        
        # Set up plot
        fig, ax = self._setup_3d_plot()
        
        # Create surface plot
        cmap = plt.cm.RdYlBu_r
        norm = plt.Normalize(Z_temp_smooth.min(), Z_temp_smooth.max())
        
        surf = ax.plot_surface(X, Y, Z_smooth,
                             facecolors=cmap(norm(Z_temp_smooth)),
                             edgecolor='none', alpha=0.9)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('Temperature (°C)', fontsize=12)
        
        # Format axis
        self._format_axis_labels(ax, dates)
        
        # Add title
        title = f'Daily {self.state} Pattern'
        if ID is not None:
            title += f' for Cow #{ID}'
        else:
            title += f' Across All Cows (n={len(df.ID.unique())})'
        plt.title(title + '\nHeight: proportion, Color: temperature', fontsize=14, pad=20)
        
        # Rotate view
        # ax.view_init(elev=0, azim=0)
        ax.view_init(30,-90,30)
        
        plt.show()
        
        return pivot_state, pivot_temp



    def plot_3d_surface_publication(self, df, ID=None):
        """Create publication-ready 3D surface plot with two views of the same data"""
        # Process data
        df = self._add_cols(df)
        
        # Filter by ID if specified
        if ID is not None:
            df = df[df.ID == ID].copy()
            print(f"Analyzing cow {ID}: {len(df)} observations")
        else:
            print(f"Analyzing {len(df.ID.unique())} cows: {len(df)} observations")
        
        # Create pivot tables
        pivot_state, pivot_temp, dates = self._create_pivot_tables(df)
        
        # Print some diagnostics
        print(f"\nState proportions summary:")
        print(f"Mean: {pivot_state.values.mean():.3f}")
        print(f"Max: {pivot_state.values.max():.3f}")
        print(f"Non-zero values: {(pivot_state.values > 0).sum()}")
        
        # Smooth surfaces
        Z_smooth, Z_temp_smooth = self._smooth_surfaces(pivot_state.values, pivot_temp.values)
        
        # Create meshgrid
        X, Y = self._create_meshgrid(pivot_state)
        
        # Set up plot - use 190mm width for publication (converted to inches)
        width_in_inches = 190/25.4  # Convert mm to inches
        height_in_inches = width_in_inches * 0.5  # 2:1 aspect ratio
        
        fig = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=300)
        
        # Create colormap
        cmap = plt.cm.RdYlBu_r
        norm = plt.Normalize(Z_temp_smooth.min(), Z_temp_smooth.max())
        
        # Create left panel (side view)
        ax1 = fig.add_axes([0.05, 0.15, 0.55, 0.80], projection='3d')
        ax2 = fig.add_axes([0.50, 0.15, 0.40, 0.80], projection='3d')

        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(X.shape, Y.shape, Z_smooth.shape)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(X)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(Y)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(Z_smooth)


        surf1 = ax1.plot_surface(X, Y, Z_smooth,
                            facecolors=cmap(norm(Z_temp_smooth)),
                            edgecolor='none', alpha=0.9,
                            rcount=100, ccount=100)  # Higher resolution
        
        # Format first axis - only show time labels, hide date labels
        ax1.set_xlabel('Time of Day', fontsize=9, labelpad=10)
        ax1.set_ylabel('', fontsize=0)  # Empty label
        ax1.set_zlabel('Proportion', fontsize=9, labelpad=10)
        # Set minute ticks (2-hour intervals)
        hour_minutes = list(range(0, 1441, 180))
        hour_labels = [f"{h//60:02d}:00" for h in hour_minutes]
        ax1.set_xticks(hour_minutes)
        ax1.set_xticklabels(hour_labels, rotation=45, ha='right', fontsize=8)
        ax1.tick_params(axis='x', which='major', pad=-5)
        
        # Hide y-axis ticks and labels
        ax1.set_yticks([])
        ax1.set_yticklabels([])
        
        # Set z-ticks with appropriate fontsize
        ax1.tick_params(axis='z', labelsize=8)
        
        # Set side view for left panel
        ax1.view_init(elev=0, azim=-90)
        
        # Adjust axis limits for better visibility
        ax1.set_zlim(0, min(1.0, Z_smooth.max() * 1.2))
        
        # Create right panel (perspective view)

        surf2 = ax2.plot_surface(X, Y, Z_smooth,
                            facecolors=cmap(norm(Z_temp_smooth)),
                            edgecolor='none', alpha=0.9,
                            rcount=100, ccount=100)
        
        # Format second axis with full labels
        ax2.set_xlabel('Time of Day', fontsize=9, labelpad=4)
        ax2.set_ylabel('Date', fontsize=9, labelpad=6)
        ax2.set_zlabel('Proportion', fontsize=9, labelpad=-6)
        
        # Set minute ticks
        ax2.set_xticks(hour_minutes)
        ax2.set_xticklabels(hour_labels, rotation=45, ha='right', fontsize=8)
        
        # Set day ticks
        day_step = max(1, len(dates) // 5)  # Show ~5 dates on axis
        day_ticks = list(range(0, len(dates), day_step))
        day_labels = [dates[i].strftime('%m/%d') for i in day_ticks]
        ax2.set_yticks(day_ticks)
        ax2.set_yticklabels(day_labels, fontsize=8, rotation = -45)
        
        # Set z-ticks with appropriate fontsize
        ax2.tick_params(axis='x', which='major', pad=-8)
        ax2.tick_params(axis='y', which='major', pad=-6)
        ax2.tick_params(axis='z', labelsize=8, pad=-2)
        
        # Adjust viewing angle to better show dates (less extreme tilt)
        ax2.view_init(elev=20, azim=140, roll=0)
        
        # Adjust axis limits
        ax2.set_zlim(0, min(1.0, Z_smooth.max() * 1.2))
        
        # Add colorbar to the figure
        cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.70])  # [left, bottom, width, height]
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Temperature (°C)', fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        
        # Add title
        title = f'Daily {self.state} Pattern Across All Cows (n={len(df.ID.unique())})'
        if ID is not None:
            title = f'Daily {self.state} Pattern for Cow #{ID}'
        subtitle = 'Height: proportion, Color: temperature'
        
        # # Add titles with proper positioning
        # fig.suptitle(title, fontsize=11, y=0.95, x=0.60)
        # fig.text(0.60, 0.86, subtitle, ha='center', fontsize=9)
        
        # Save the figure
        output_path = os.path.join(self.config.visuals.visuals_root_path, f'3d_surface_{self.state.lower()}.png')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        
        # plt.show()
        
        return pivot_state, pivot_temp



    def fit_surface(self, pivot_state, pivot_temp):
        """Fit a function to the state surface with improved R² through better feature engineering
        
        Parameters:
        -----------
        pivot_state : pd.DataFrame
            Pivot table with state proportions
        pivot_temp : pd.DataFrame
            Pivot table with temperatures
        model_type : str
            'polynomial', 'fourier', 'combined', or 'best' (tries multiple approaches and selects best)
        
        Returns:
        --------
        fitted_surface : np.array
            The fitted values
        model : object
            The fitted model
        """

        # Prepare coordinates and values
        x = pivot_state.columns.values  # minutes
        y = pivot_state.index.values    # days
        X, Y = np.meshgrid(x, y)
        Z = pivot_state.values
        T = pivot_temp.values
        
        # Update day scale factor
        self.scale_factors['day'] = len(y)
        self.scale_factors['temp'] = max(10, T.max() - T.min())
        
        # Scale inputs
        X_scaled = X / self.scale_factors['minute']
        Y_scaled = Y / self.scale_factors['day']
        T_scaled = T / self.scale_factors['temp']
        
        # Prepare input features
        X_flat = np.column_stack((
            X_scaled.flatten(),      # scaled minutes
            Y_scaled.flatten(),      # scaled days
            T_scaled.flatten()       # scaled temperature
        ))
        Z_flat = Z.flatten()
        # Define number of Fourier components
        n_components = 5  # Reduced from 5 to improve performance
        
        # Get Fourier features
        minutes = X_flat[:, 0] * self.scale_factors['minute']
        fourier_features = []
        feature_names = []
        
        for i in range(1, n_components + 1):
            period = 2 * np.pi * i / 1440
            fourier_features.append(np.sin(minutes * period))
            fourier_features.append(np.cos(minutes * period))
            feature_names.append(f'sin_{i}')
            feature_names.append(f'cos_{i}')
        
        # Add polynomial features (degree 2 is usually sufficient with Fourier)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X_flat)
        poly_names = poly.get_feature_names_out(['minute', 'day', 'temp'])
        
        # Combine features
        X_combined = np.column_stack([
            *fourier_features,
            X_poly
        ])
        all_feature_names = feature_names + list(poly_names)
        
        # Fit Ridge regression with combined features
        combined_model = Ridge(alpha=0.1)
        combined_model.fit(X_combined, Z_flat)
        Z_fitted_combined = combined_model.predict(X_combined).reshape(X.shape)
        r2_combined = combined_model.score(X_combined, Z_flat)
        print(f"Combined fit R-squared: {r2_combined:.3f}")
        
        # Store model information
        combined_model.feature_names = all_feature_names
        combined_model.n_components = n_components
        combined_model.poly_features = poly
        
        model = combined_model
        Z_fitted, r2 = (Z_fitted_combined, r2_combined)

        
        # Store additional information for interpretation
        
        model.scale_factors = self.scale_factors
        model.X_flat = X_flat  # Store original features for reference
        
        # Create a consistent prediction function that works for all model types
        def create_prediction_features(minutes, day, temp):
            """Create the appropriate feature matrix for each model type"""
            # Scale the inputs
            minute_scaled = minutes / model.scale_factors['minute']
            day_scaled = day / model.scale_factors['day'] 
            temp_scaled = temp / model.scale_factors['temp']
            
            # Base features
            X_base = np.column_stack([
                minute_scaled,
                np.full_like(minutes, day_scaled),
                np.full_like(minutes, temp_scaled)
            ])
            
            # Create Fourier features
            fourier_features = []
            for i in range(1, model.n_components + 1):
                period = 2 * np.pi * i / 1440
                fourier_features.append(np.sin(minutes * period))
                fourier_features.append(np.cos(minutes * period))
            
            # Create polynomial features
            X_poly = model.poly_features.transform(X_base)
            
            # Combine features
            return np.column_stack([*fourier_features, X_poly])
            
            # else:
            #     raise ValueError(f"Unknown model type: {model_type}")


        # Prediction function
        def predict_for_day_temp(day, temp, minutes=np.linspace(0, 1440, 288)):
            """Predict state probability for a given day and temperature across minutes"""
            X_pred = create_prediction_features(minutes, day, temp)
            return model.predict(X_pred)
        
        # Attach the function to the model
        model.predict_for_day_temp = predict_for_day_temp
        
        return Z_fitted, model



    def plot_fitted_surface(self, df, ID=None):
        """Plot original and fitted surfaces side by side"""

        df = self._add_cols(df)
        # Get the data
        pivot_state, pivot_temp, _ = self._create_pivot_tables(df)
        
        # Smooth the original surface
        Z_smooth, Z_temp_smooth = self._smooth_surfaces(pivot_state.values, pivot_temp.values)
        
        # Fit the surface
        Z_fitted, model = self.fit_surface(pivot_state, pivot_temp)
        
        # Create meshgrid
        X, Y = self._create_meshgrid(pivot_state)
        
        # Create figure with two subplots
        fig = plt.figure(figsize=(20, 10))
        
        # Plot original surface
        ax1 = fig.add_subplot(121, projection='3d')
        cmap = plt.cm.RdYlBu_r
        norm = plt.Normalize(Z_temp_smooth.min(), Z_temp_smooth.max())
        
        surf1 = ax1.plot_surface(X, Y, Z_smooth,
                            facecolors=cmap(norm(Z_temp_smooth)),
                            edgecolor='none', alpha=0.9)
        
        self._format_axis_labels(ax1, sorted(df['date'].unique()))
        ax1.set_title('Original Surface', fontsize=14)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax1, shrink=0.5, aspect=10, label='Temperature (°C)')
        
        # Plot fitted surface
        ax2 = fig.add_subplot(122, projection='3d')
        # surf2 = ax2.plot_surface(X, Y, Z_fitted,
        #                     cmap='viridis',
        #                     edgecolor='none', alpha=0.9)
        surf2 = ax2.plot_surface(X, Y, Z_fitted,
                            facecolors=cmap(norm(Z_temp_smooth)),
                            edgecolor='none', alpha=0.9)

        # surf1 = ax1.plot_surface(X, Y, Z_smooth,
        #                     facecolors=cmap(norm(Z_temp_smooth)),
        #                     edgecolor='none', alpha=0.9)
        
        self._format_axis_labels(ax2, sorted(df['date'].unique()))


        model_type = getattr(model, 'model_type', self.fit_method)

        # fig_txt = f"(method=poly({self.degree}))"

        ax2.set_title(f'Fitted Surface `{model_type}`', fontsize=14)
        # fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, label='Fitted Proportion')
        
        # Add overall title
        title = f'Daily {self.state} Pattern'
        if ID is not None:
            title += f' for Cow #{ID}'
        else:
            title += f' Across All Cows (n={len(df.ID.unique())})'
        plt.suptitle(title, fontsize=16, y=0.95)
        
        plt.tight_layout()
        plt.show()
        
        return model, pivot_state




    def analyze_fitted_surface(self, model, pivot_state):
        """Analyze the fitted surface and display the model coefficients"""
        model_type = model.model_type
        
        print(f"\nAnalysis of {model_type.capitalize()} Model:")
        print("====================================")
        
        # Get the temperature range from the original data for predictions
        temp_range = np.linspace(-10, 15, 6)  # -10°C to 15°C in 6 steps
        
        # Get a middle day for analysis
        mid_day = len(pivot_state.index) // 2
        
        # Time points for analysis
        minutes = np.linspace(0, 1440, 288)  # 5-minute intervals
        
        # Analyze model predictions at different temperatures
        print("\nDaily patterns at different temperatures:")
        for temp in temp_range:
            print(f"\nTemperature: {temp:.1f}°C")
            
            # Predict for this temperature
            Z_pred = model.predict_for_day_temp(mid_day, temp, minutes)
            
            peaks, _ = find_peaks(Z_pred, height=0.2, distance=30)
            
            if len(peaks) > 0:
                print(f"{self.state} peaks:")
                for peak in peaks:
                    time = minutes[peak]
                    print(f"  {int(time//60):02d}:{int(time%60):02d} ({Z_pred[peak]:.2f})")
            else:
                print(f"No significant {self.state} peaks found")
        
        # Create temperature vs grazing heatmap
        try:
            self._plot_temp_grazing_heatmap(model, pivot_state)
        except Exception as e:
            print(f"Error creating heatmap: {e}")
        
        return model



    def _plot_temp_grazing_heatmap(self, model, pivot_state):
        """Create a heatmap showing how selected state varies with time and temperature"""
        # Create a grid of time and temperature (reduced grid for performance)
        times = np.linspace(0, 1440, 72)   # 20-minute intervals
        temps = np.linspace(-10, 15, 11)   # 2.5°C intervals
        
        mid_day = len(pivot_state.index) // 2
        
        # Create meshgrid
        TIME, TEMP = np.meshgrid(times, temps)
        Z = np.zeros_like(TIME)
        
        # Make predictions for each time-temperature combination
        for i, temp in enumerate(temps):
            Z[i, :] = model.predict_for_day_temp(mid_day, temp, times)
        
        # Create the plot with adjusted figure size
        plt.figure(figsize=(12, 6))
        
        # Use contourf for filled contour plot
        contour = plt.contourf(TIME, TEMP, Z, 20, cmap='viridis')
        
        # Add contour lines
        contour_lines = plt.contour(TIME, TEMP, Z, 5, colors='white', alpha=0.5, linewidths=0.5)
        plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
        
        # Format axes
        plt.xlabel('Time of Day', fontsize=12)
        plt.ylabel('Temperature (°C)', fontsize=12)
        plt.title(f'Effect of Temperature on {self.state} Throughout the Day', fontsize=14)
        
        # Format time axis
        hour_ticks = np.arange(0, 1441, 180)  # Every 3 hours
        hour_labels = [f"{h//60:02d}:{h%60:02d}" for h in hour_ticks]
        plt.xticks(hour_ticks, hour_labels)
        
        # Add colorbar
        cbar = plt.colorbar(contour)
        cbar.set_label(f'Proportion {self.state}', fontsize=12)
        
        # Add grid
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Handle the tight layout warning by using different parameters
        plt.tight_layout(pad=1.1)
        plt.savefig(os.path.join(self.config.visuals.visuals_root_path,f'{self.state}_temperature_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.show()

    ########### Single cow, only smooth within days
    # def _plot_3d_surface(self, df, cow_id, state):
    #     """Create a 3D surface plot showing how grazing patterns change across days"""
    #     import scipy.ndimage as ndimage
    #     from scipy.interpolate import griddata

    #     # Convert dates to numeric values (days from start)
    #     dates = sorted(df['date'].unique())
    #     date_mapping = {date: i for i, date in enumerate(dates)}
    #     df['day_num'] = df['date'].map(date_mapping)
        
    #     # Create pivot tables for both state proportions and temperature
    #     pivot_state = df.pivot_table(
    #         index='day_num', 
    #         columns='minute', 
    #         values='prop_state',
    #         aggfunc='mean'
    #     ).fillna(0)
        
    #     pivot_temp = df.pivot_table(
    #         index='day_num', 
    #         columns='minute', 
    #         values='temperature',
    #         aggfunc='mean'
    #     ).fillna(method='ffill').fillna(method='bfill')
        
    #     # Smooth only in the X direction (within each day)
    #     Z_state = pivot_state.values
    #     Z_temp = pivot_temp.values
        
    #     # Apply 1D Gaussian filter to each day separately
    #     Z_smooth = np.zeros_like(Z_state)
    #     Z_temp_smooth = np.zeros_like(Z_temp)
        
    #     for i in range(Z_state.shape[0]):
    #         Z_smooth[i, :] = ndimage.gaussian_filter1d(Z_state[i, :], sigma=2.0)
    #         Z_temp_smooth[i, :] = ndimage.gaussian_filter1d(Z_temp[i, :], sigma=2.0)

    #     # Create the meshgrid
    #     X, Y = np.meshgrid(
    #         np.linspace(min(pivot_state.columns), max(pivot_state.columns), 288),  # 5-minute intervals
    #         np.linspace(min(pivot_state.index), max(pivot_state.index), len(dates))
    #     )

    #     # Create the plot
    #     fig = plt.figure(figsize=(12, 10))
    #     ax = fig.add_subplot(111, projection='3d')
        
    #     # Set up custom colormap
    #     cmap = plt.cm.RdYlBu_r
        
    #     # Create the surface plot with color based on temperature
    #     surf = ax.plot_surface(X, Y, Z_smooth, 
    #                         facecolors=cmap(plt.Normalize(Z_temp_smooth.min(), 
    #                                                     Z_temp_smooth.max())(Z_temp_smooth)),
    #                         edgecolor='none', alpha=0.9)
        
    #     # Create a proxy surface for the colorbar
    #     proxy_surface = plt.cm.ScalarMappable(cmap=cmap, 
    #                                         norm=plt.Normalize(Z_temp_smooth.min(), 
    #                                                         Z_temp_smooth.max()))
    #     proxy_surface.set_array([])

    #     # Customize the plot
    #     ax.set_xlabel('Minute of Day', fontsize=12, labelpad=10)
    #     ax.set_ylabel('Day', fontsize=12, labelpad=10)
    #     ax.set_zlabel(f'Proportion {state}', fontsize=12, labelpad=10)
        
    #     # Set minute ticks at 2-hour intervals
    #     hour_minutes = list(range(0, 1441, 120))
    #     hour_labels = [f"{h//60:02d}:{h%60:02d}" for h in hour_minutes]
    #     ax.set_xticks(hour_minutes)
    #     ax.set_xticklabels(hour_labels, rotation=45, ha='right')
        
    #     # Set day ticks to show actual dates (some of them)
    #     day_step = max(1, len(dates) // 8)  # Show ~8 dates on axis
    #     day_ticks = list(range(0, len(dates), day_step))
    #     day_labels = [dates[i].strftime('%m/%d') for i in day_ticks]
    #     ax.set_yticks(day_ticks)
    #     ax.set_yticklabels(day_labels)
        
    #     # Add a color bar showing temperature scale
    #     cbar = fig.colorbar(proxy_surface, ax=ax, shrink=0.5, aspect=10)
    #     cbar.set_label('Temperature (°C)', fontsize=12)
        
    #     # Add title
    #     plt.title(f'Daily {state} Pattern for Cow #{cow_id}\nHeight: {state} proportion (smoothed within-day), Color: Temperature', 
    #             fontsize=14, pad=20)
        
    #     plt.show()
