# from datetime import datetime
# import os
import sys
import argparse
# import json
from pathlib import Path
import logging
import pandas as pd
# from typing import Optional

from cowstudyapp.config import ConfigManager


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load and validate the dataset.
    
    Args:
        path: Path to the dataset file
        
    Returns:
        pd.DataFrame: Loaded dataset
        
    Raises:
        FileNotFoundError: If dataset file doesn't exist
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    
    # logging.info(f"Loading dataset from: {path}")
    return pd.read_csv(path)

def create_output_directory(config: ConfigManager) -> Path: 
    output_dir: Path = config.visuals.visuals_root_path
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir



def generate_plots(config, predictions_dataset, target_dataset, progress_callback):
    if 'progress_callback' not in locals():  # Fixed missing variable definition
        progress_callback = lambda percent, message: print(f"{percent}%: {message}")

    # progress_callback(7, "Preparing dataset for LSTM analysis")

    total_visuals = sum([config.visuals.domain.run,
                         config.visuals.feature_dists.run,
                         config.visuals.heatmap.run,
                         config.visuals.radar.run,
                         config.visuals.cow_info_graph.run,
                         config.visuals.temperature_graph.run,
                         config.visuals.convolution_surface.run,
                         config.visuals.moon_phases.run,
                         ])
    
    weights = [1, 2, 2, 5, 1, 20, 4, 3]  # Define weights directly
    total_visuals_weights = []
    
    # Calculate actual weights based on enabled visualizations
    for i, enabled in enumerate([
        config.visuals.domain.run,
        config.visuals.feature_dists.run,
        config.visuals.heatmap.run,
        config.visuals.radar.run,
        config.visuals.cow_info_graph.run,
        config.visuals.temperature_graph.run,
        config.visuals.convolution_surface.run,
        config.visuals.moon_phases.run
    ]):
        if enabled:
            total_visuals_weights.append(weights[i])
    
    progress_callback(11, f"Making {total_visuals} types of visualizations.")
    
    total_weight = sum(total_visuals_weights)
    
    # Calculate proportional weights (remaining 89%)
    if total_weight > 0:  # Avoid division by zero
        proportional_weights = [11 + (w / total_weight * 89) for w in total_visuals_weights]
    else:
        proportional_weights = []

    # Calculate cumulative weights for progress tracking
    cumil_weights = [11]
    for w in proportional_weights:
        cumil_weights.append(round(cumil_weights[-1] + w,2))
    
    # Index to track which visualization we're on
    vis_index = 0
    
    #############################################################################
    if config.visuals.domain.run:
        progress_callback(cumil_weights[vis_index], f"Starting the domain visualization.")
        logger.info(f"Starting domain info visualization ")
        from cowstudyapp.visuals.show_domain_of_data import DataDomainVisualizer
        visualizer = DataDomainVisualizer(config)
        visualizer.generate_plot(predictions_dataset.copy())
        vis_index += 1


    #############################################################################
    if config.visuals.feature_dists.run:
        logger.info(f"Starting feature distributions visualization ")
        progress_callback(cumil_weights[vis_index], f"Starting the HMM feature Distributions visualization.")

        from cowstudyapp.visuals.show_feature_distr import Feature_Plotter
        fp = Feature_Plotter(config=config)
        # fp.generate_plot(predictions_dataset, output_dir=output_dir)
        # fp.plot_ecdf_comparison(predictions_dataset, output_dir=output_dir)
        # fp.plot_publication_cdf_comparison(predictions_dataset, output_dir=output_dir)
        fp.plot_publication_cdf_comparison_all_4(predictions_dataset.copy())
        vis_index += 1


    #############################################################################
    if config.visuals.heatmap.run:
        logger.info(f"Starting heatmap of grazing visualization ")
        progress_callback(cumil_weights[vis_index], f"Starting the Grazing Heatmap Visualization.")

        from cowstudyapp.visuals.make_heatmap_of_predictions import HeatMapMaker
        hmm = HeatMapMaker(config=config)
        hmm.run(predictions_dataset.copy())
        vis_index += 1


    ############################################################################
    if config.visuals.radar.run:
        logger.info(f"Starting radarplots of activity visualizations")
        progress_callback(cumil_weights[vis_index], f"Starting the Radar Plots Visualizations.")

        from cowstudyapp.visuals.show_radar_plots import RadarPlotOfCow
        radar = RadarPlotOfCow(config=config)
        radar.make_cow_gallery_images(df=predictions_dataset.copy())
        radar.make_radar_TWO_cows(df=predictions_dataset.copy(), show=False)
        vis_index += 1

    #############################################################################
    if config.visuals.cow_info_graph.run:
        logger.info(f"Starting cow meta_info visualization")
        progress_callback(cumil_weights[vis_index], f"Starting the meta-cow info visualization.")


        
        from cowstudyapp.visuals.show_cow_info_vs_pred import GrazingVersusCowInfo
        gvc = GrazingVersusCowInfo(config=config)
        gvc.pairplot_cow_info(df=predictions_dataset.copy())
        gvc.compare_cow_info(df=predictions_dataset.copy())
        vis_index += 1
    
    
    ##############################################################################
    if config.visuals.temperature_graph.run:

        print(predictions_dataset.head(20))
        logger.info(f"Starting GLMM temperature visualization. This may take a while")
        progress_callback(cumil_weights[vis_index], f"Starting the GLMM temperature visualization. This make take a while")

        
        ######################## 3-h binned ########################
        # from cowstudyapp.visuals.show_temp_vs_activity import GrazingVersusTemperature
        # gvt = GrazingVersusTemperature(config=config)
        # gvt.compare_temp_behavior(df=predictions_dataset)


        # from cowstudyapp.visuals.show_temp_vs_activity_Bayes import GrazingVersusTemperatureBayes
        # gvt = GrazingVersusTemperatureBayes(config=config)
        # gvt.analyze_binomial_glmm(df=predictions_dataset)

        from cowstudyapp.visuals.show_effects_glmm import BehaviorAnalyzer
        BA = BehaviorAnalyzer(config = config)
        BA.analyze(df=predictions_dataset.copy())

        vis_index += 1

    ######################## 3D Plot ########################
    if config.visuals.convolution_surface.run:
        logger.info(f"Starting Convolutional surface of grazing visualization")
        progress_callback(cumil_weights[vis_index], f"Starting the convolutional surface temperature visualization.")

        from cowstudyapp.visuals.show_temp_vs_grazing_new import ActivityVersusTemperatureNew
        gvt = ActivityVersusTemperatureNew(config=config, degree=5, state='Grazing')
        # gvt = GrazingVersusTemperatureNew(config=config, fit_method='gpr')
        # gvt.plot_3d_surface(predictions_dataset)
        gvt.plot_3d_surface_publication(predictions_dataset.copy())
        vis_index += 1
        
        # # Analyze the fitted surface
        # model, pivot_state = gvt.plot_fitted_surface(predictions_dataset)
        # gvt.analyze_fitted_surface(model, pivot_state)


    
    ##############################################################################
    if config.visuals.moon_phases.run:
        logger.info(f"Starting moon phases interaction with grazing visualization")
        progress_callback(cumil_weights[vis_index], f"Starting Moon phases visualizations.")

        from cowstudyapp.visuals.show_moon_versus_grazing import MoonPhasesGrazing
        mpg = MoonPhasesGrazing(config=config)
        mpg.compare_moon_behavior(df=predictions_dataset)
        vis_index += 1

    progress_callback(99, f"Done with visualizations.")





def main(config_path=None, progress_callback=None):
    if progress_callback is None:
        progress_callback = lambda percent, message: print(f"{percent}%: {message}")
    logger.info("Starting make_visuals")
    if config_path is None and len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            prog="make_visuals",
            description="Make visuals showing prediction patterns"
        )
        parser.add_argument(
            "task_id",
            help="ID of this visuals task (for logging, etc.)"
        )
        parser.add_argument(
            "config_path",
            type=Path,
            help="YAML file with visuals configuration"
        )
        args = parser.parse_args()
        config_path = args.config_path
        task_id = args.task_id

    else:
        if not config_path:
            config_path = "config/RB_22_config.yaml"
        config_path = Path(config_path)
        # print("!! Config Path", config_path)
        
        task_id = "direct-call"
    config = ConfigManager.load(config_path)

    try:
        progress_callback(10, "Loading datasets")
        target_dataset = load_dataset(Path(config.analysis.target_dataset))
        predictions_dataset = load_dataset(Path(config.visuals.predictions_path))

        # Initialize visualizer and generate plots
        generate_plots(config=config
                    , predictions_dataset=predictions_dataset
                    , target_dataset=target_dataset
                    , progress_callback = progress_callback
        )

        return True

    except Exception as e:
        logger.exception("Error while performing task %s: %s", task_id, str(e))
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
