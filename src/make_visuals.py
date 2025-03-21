from datetime import datetime
import os
from pathlib import Path
# import logging
import pandas as pd
from typing import Optional

from cowstudyapp.config import ConfigManager

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



def generate_plots(config, predictions_dataset, target_dataset):

    #############################################################################
    if config.visuals.domain.run:
        from cowstudyapp.visuals.show_domain_of_data import DataDomainVisualizer
        visualizer = DataDomainVisualizer(config)
        visualizer.generate_plot(predictions_dataset)


    #############################################################################
    # if config.visuals.!!!!!!!!!!!!!!.run:
    #     from cowstudyapp.visuals.show_feature_distr import Feature_Plotter
    #     fp = Feature_Plotter(config=config)
    #     fp.generate_plot(predictions_dataset, output_dir=output_dir)


    #############################################################################
    if config.visuals.heatmap.run:
        from cowstudyapp.visuals.make_heatmap_of_predictions import HeatMapMaker
        hmm = HeatMapMaker(config=config)
        hmm.run(predictions_dataset)


    ############################################################################
    if config.visuals.radar.run:
        from cowstudyapp.visuals.show_radar_plots import RadarPlotOfCow
        radar = RadarPlotOfCow(config=config)
        radar.make_cow_gallery_images(df=predictions_dataset)


    #############################################################################
    if config.visuals.cow_info_graph.run:
        from cowstudyapp.visuals.show_cow_info_vs_pred import GrazingVersusCowInfo
        gvc = GrazingVersusCowInfo(config=config)
        gvc.compare_cow_info(df=predictions_dataset)
    
    
    # #############################################################################
    if config.visuals.temperature_graph.run:
        from cowstudyapp.visuals.show_temp_vs_activity import GrazingVersusTemperature
        gvt = GrazingVersusTemperature(config=config)
        gvt.compare_temp_behavior(df=predictions_dataset)
    
    


if __name__ == "__main__":
    # Load configuration
    # config_path = Path("config/RB_22_config.yaml")
    config_path = Path("config/RB_19_config.yaml")
    config = ConfigManager.load(config_path)


    if config.analysis is None:
        raise ValueError("ERROR: Missing required analysis config section")  
    if config.visuals is None:
        raise ValueError("ERROR: Missing required Visuals config section")  

    
    # Load dataset
    target_dataset = load_dataset(Path(config.analysis.target_dataset))
    predictions_dataset = load_dataset(Path(config.visuals.predictions_path))
    
    # Create output directory
    output_dir = create_output_directory(config)
    
    # Initialize visualizer and generate plots
    generate_plots(config=config
                 , predictions_dataset=predictions_dataset
                 , target_dataset=target_dataset
    )